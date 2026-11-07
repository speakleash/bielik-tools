"""Unit tests for the Bielik vLLM reasoning parsers."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("vllm")

from vllm.reasoning.deepseek_r1_reasoning_parser import (  # noqa: E402
    DeepSeekR1ReasoningParser,
)
from vllm.reasoning.identity_reasoning_parser import (  # noqa: E402
    IdentityReasoningParser,
)

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
PARSER_FILES = [
    "bielik_vllm_reasoning_parser.py",
    "bielik_vllm_reasoning_parser_v0.13.0.py",
]
THINK_VOCAB = {"<think>": 10, "</think>": 11, "a": 1}
NO_THINK_VOCAB = {"a": 1}


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def _load(filename):
    path = TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=PARSER_FILES)
def parser_module(request):
    try:
        module = _load(request.param)
    except ImportError as exc:
        pytest.skip(f"{request.param} does not support the installed vLLM: {exc}")
    return module


@pytest.fixture
def parser_cls(parser_module):
    return parser_module.BielikReasoningParser


def test_thinking_disabled_uses_identity(parser_cls):
    parser = parser_cls(FakeTokenizer(THINK_VOCAB), chat_template_kwargs={})
    assert isinstance(parser._parser, IdentityReasoningParser)


def test_thinking_enabled_uses_deepseek_r1(parser_cls):
    parser = parser_cls(FakeTokenizer(THINK_VOCAB),
                        chat_template_kwargs={"enable_thinking": True})
    assert isinstance(parser._parser, DeepSeekR1ReasoningParser)
    reasoning, content = parser.extract_reasoning(
        "<think>17*3 = 51</think>Wynik to 51.", request=None)
    assert reasoning == "17*3 = 51"
    assert content == "Wynik to 51."


@pytest.mark.parametrize("vocab", [NO_THINK_VOCAB, {"<think>": 10}, {"</think>": 11}])
def test_thinking_enabled_without_think_tokens_falls_back(parser_cls, vocab):
    # Tokenizers without <think>/</think> special tokens (e.g.
    # Bielik-4.5B-v3.0) must not fail the request.
    parser = parser_cls(FakeTokenizer(vocab),
                        chat_template_kwargs={"enable_thinking": True})
    assert isinstance(parser._parser, IdentityReasoningParser)
    reasoning, content = parser.extract_reasoning("Wynik to 51.", request=None)
    assert reasoning is None
    assert content == "Wynik to 51."


def test_unrelated_initialization_error_propagates(parser_module, monkeypatch):
    def broken_parser(*args, **kwargs):
        raise RuntimeError("unrelated initialization failure")

    monkeypatch.setattr(parser_module, "DeepSeekR1ReasoningParser", broken_parser)
    with pytest.raises(RuntimeError, match="unrelated initialization failure"):
        parser_module.BielikReasoningParser(
            FakeTokenizer(THINK_VOCAB),
            chat_template_kwargs={"enable_thinking": True},
        )


def test_adjust_request_keeps_special_tokens_when_thinking():
    # Bielik v3.0 marks <think>/</think> as special tokens; stripping them
    # makes reasoning and answer indistinguishable.
    module = _load("bielik_vllm_reasoning_parser.py")
    request = type("Request", (), {"skip_special_tokens": True})()
    on = module.BielikReasoningParser(
        FakeTokenizer(THINK_VOCAB), chat_template_kwargs={"enable_thinking": True})
    assert on.adjust_request(request).skip_special_tokens is False
    request.skip_special_tokens = True
    off = module.BielikReasoningParser(
        FakeTokenizer(THINK_VOCAB), chat_template_kwargs={})
    assert off.adjust_request(request).skip_special_tokens is True


def test_prompt_with_think_tags_in_system_text_is_not_reasoning_end():
    # The advanced template quotes <think>...</think> in the system prompt.
    # vLLM checks the prompt with is_reasoning_end(); that must stay False so
    # streaming starts in the reasoning phase.
    module = _load("bielik_vllm_reasoning_parser.py")
    vocab = {**THINK_VOCAB, "<|im_start|>": 20}
    parser = module.BielikReasoningParser(
        FakeTokenizer(vocab), chat_template_kwargs={"enable_thinking": True})
    prompt = [20, 1, 10, 1, 11, 1, 20, 1]  # system mentions think tags
    assert parser.is_reasoning_end(prompt) is False
    assert parser.is_reasoning_end(prompt + [10, 1, 11, 1]) is True
    assert parser.is_reasoning_end([10, 1, 11]) is True  # no turn marker


def test_adjust_request_untouched_on_missing_think_tokens_fallback():
    module = _load("bielik_vllm_reasoning_parser.py")
    request = type("Request", (), {"skip_special_tokens": True})()
    parser = module.BielikReasoningParser(
        FakeTokenizer(NO_THINK_VOCAB), chat_template_kwargs={"enable_thinking": True})
    assert parser.adjust_request(request).skip_special_tokens is True


# --------------------------------------------------------------------------
# One parser for the advanced (enable_thinking) and v3.1/v3.2 (reasoning_effort)
# templates.
# --------------------------------------------------------------------------
TURN_VOCAB = {**THINK_VOCAB, "<|im_start|>": 20, "<|im_end|>": 21}
START, END, IM_START, IM_END, A = 10, 11, 20, 21, 1


def _default_parser(**chat_kwargs):
    module = _load("bielik_vllm_reasoning_parser.py")
    return module.BielikReasoningParser(FakeTokenizer(TURN_VOCAB), chat_template_kwargs=chat_kwargs)


@pytest.mark.parametrize("kwargs, thinking", [
    ({}, False),
    ({"enable_thinking": True}, True),
    ({"enable_thinking": False}, False),
    ({"reasoning_effort": "high"}, True),
    ({"reasoning_effort": "low"}, True),
    ({"reasoning_effort": "none"}, False),
    ({"reasoning_effort": "NONE"}, False),
    # reasoning_effort wins over enable_thinking (v3.1 template semantics)
    ({"reasoning_effort": "none", "enable_thinking": True}, False),
    ({"reasoning_effort": "medium", "enable_thinking": False}, True),
    # what vLLM 0.32 builds for a top-level reasoning_effort
    ({"reasoning_effort": "high", "enable_thinking": True}, True),
])
def test_thinking_switch(kwargs, thinking):
    parser = _default_parser(**kwargs)
    assert isinstance(parser._parser, DeepSeekR1ReasoningParser) is thinking


def test_open_generation_prompt_keeps_thinking():
    # v3.1 template, reasoning on: the turn ends with an open <think>.
    parser = _default_parser(reasoning_effort="high")
    prompt = [IM_START, A, A, IM_END, IM_START, A, START]
    assert parser.is_reasoning_end(prompt) is False
    assert parser.extract_reasoning("myślę</think>Wynik.", request=None) == ("myślę", "Wynik.")


def test_closed_generation_prompt_returns_content():
    # v3.1 template ignores enable_thinking and closes <think></think>; the
    # whole output is the answer, also in non-streaming mode.
    parser = _default_parser(enable_thinking=True)
    prompt = [IM_START, A, IM_END, IM_START, A, START, END]
    assert parser.is_reasoning_end(prompt) is True
    assert parser.extract_reasoning("391", request=None) == (None, "391")
    assert parser.is_reasoning_end([IM_START, A, START]) is True  # stays content


@pytest.mark.parametrize("earlier", [
    # system prompt quotes an example "<think>...</think>"
    [IM_START, A, START, A, END, A, IM_END],
    # an earlier assistant turn with closed reasoning
    [IM_START, A, IM_END, IM_START, A, START, A, END, A, IM_END, IM_START, A, IM_END],
])
def test_think_tags_outside_the_current_turn_do_not_close_reasoning(earlier):
    parser = _default_parser(enable_thinking=True)
    assert parser.is_reasoning_end(earlier + [IM_START, A]) is False
    assert parser.is_reasoning_end(earlier + [IM_START, A, START]) is False
    quoted = 'Przykład: <tool_call>{"name": "x", "arguments": {}}</tool_call>'
    reasoning, content = parser.extract_reasoning(quoted + "</think>Odpowiedź.", request=None)
    assert reasoning == quoted  # a tool call quoted in reasoning stays reasoning
    assert content == "Odpowiedź."


def test_output_ids_without_turn_marker_do_not_switch_mode():
    parser = _default_parser(reasoning_effort="high")
    assert parser.is_reasoning_end([A, END, A]) is True
    assert isinstance(parser._parser, DeepSeekR1ReasoningParser)


def test_thinking_off_parser_is_unchanged_by_prompt():
    parser = _default_parser()
    assert parser.is_reasoning_end([IM_START, A, START, END]) is True
    assert parser.extract_reasoning("<think>x</think>y", request=None) == (None, "<think>x</think>y")


def test_without_the_prompt_thinking_follows_the_request_flags():
    # vLLM 0.32 does not show the prompt to the parsers of extra non-streaming
    # choices (n > 1) or with include_reasoning=false. The parser then keeps
    # DeepSeek-R1 semantics: output without </think> counts as reasoning
    # (documented limitation; reasoning_effort avoids it on v3.1/v3.2).
    parser = _default_parser(enable_thinking=True)
    assert parser.extract_reasoning("391", request=None) == ("391", None)
