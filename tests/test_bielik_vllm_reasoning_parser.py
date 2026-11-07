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
    "bielik_vllm_reasoning_parser_v3.1.py",
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


V31 = "bielik_vllm_reasoning_parser_v3.1.py"
CLOSED_PROMPT = [10, 11]   # v3.1 template: <think></think> already in the prompt
OPEN_PROMPT = [1, 1]       # advanced template: prompt ends before any think tag


def _v31(**kwargs):
    return _load(V31).BielikReasoningParser(
        FakeTokenizer(THINK_VOCAB), chat_template_kwargs=kwargs)


def test_legacy_flag_with_closed_prompt_is_content():
    # v3.1 template ignores enable_thinking and forces <think></think>.
    parser = _v31(enable_thinking=True)
    assert parser.is_reasoning_end(CLOSED_PROMPT) is True
    assert parser.extract_reasoning("391.", request=None) == (None, "391.")


def test_legacy_flag_with_open_prompt_keeps_reasoning_behaviour():
    # Advanced template: the model opens <think> itself; nothing changes.
    parser = _v31(enable_thinking=True)
    assert parser.is_reasoning_end(OPEN_PROMPT) is False
    assert parser.extract_reasoning("<think>a</think>b", request=None) == ("a", "b")
    assert parser.extract_reasoning("a</think>b", request=None) == ("a", "b")
    assert parser.extract_reasoning("<think>a", request=None)[0] == "a"
    assert parser.extract_reasoning("tagless", request=None) == ("tagless", None)


def test_prompt_state_is_recorded_from_the_first_call_only():
    parser = _v31(enable_thinking=True)
    parser.is_reasoning_end(OPEN_PROMPT)
    parser.is_reasoning_end(CLOSED_PROMPT)  # later calls see generated ids
    assert parser.extract_reasoning("a</think>b", request=None) == ("a", "b")


def test_reasoning_effort_is_unaffected_by_prompt_state():
    parser = _v31(reasoning_effort="high")
    parser.is_reasoning_end(CLOSED_PROMPT)
    assert parser.extract_reasoning("a</think>b", request=None) == ("a", "b")


def test_legacy_flag_false_never_short_circuits():
    parser = _v31(enable_thinking=False)
    parser.is_reasoning_end(CLOSED_PROMPT)
    assert parser.extract_reasoning("391.", request=None) == (None, "391.")
