"""Exercise vLLM's real native registry and streaming API without runtime patches."""

import importlib.metadata

import pytest

if importlib.metadata.version("vllm").split("+")[0] != "0.20.0":
    pytest.skip("Native vLLM 0.20 plugin", allow_module_level=True)

from test_streaming_tool_parser_default import (
    flatten,
    load_unified_020,
    request_data,
    tokenizer,
)


def parser_for(tokenizer, request_data, **chat_kwargs):
    from vllm.parser.parser_manager import ParserManager

    plugin = load_unified_020()
    cls = ParserManager.get_parser(
        tool_parser_name="bielik_020",
        reasoning_parser_name="bielik_020",
        enable_auto_tools=True,
    )
    assert cls is plugin.Bielik020Parser
    return cls(tokenizer, request_data.tools, chat_template_kwargs=chat_kwargs)


def step(parser, tokenizer, request, text):
    return parser.parse_delta(
        text,
        tokenizer.encode(text, add_special_tokens=False),
        request,
        prompt_token_ids=tokenizer.encode("<think>", add_special_tokens=False),
    )


@pytest.mark.parametrize(
    "chat_kwargs",
    [
        {"reasoning_effort": "low"},
        {"reasoning_effort": "low", "enable_thinking": False},
        {"enable_thinking": True},
    ],
)
def test_reasoning_survives_when_incomplete_tool_produces_no_delta(
    tokenizer, request_data, chat_kwargs
):
    parser = parser_for(tokenizer, request_data, **chat_kwargs)
    first = step(
        parser,
        tokenizer,
        request_data,
        'analysis</think><tool_call>{"name":"calc","arguments":',
    )
    assert first.reasoning == "analysis"
    assert not first.tool_calls
    second = step(parser, tokenizer, request_data, '{"x":2}}</tool_call>')
    assert not second.reasoning
    assert flatten([first, second]) == [("calc", {"x": 2})]
    assert not parser._collect_handoff and parser._handoff_reasoning is None


def test_plain_text_handoff_keeps_both_fields(tokenizer, request_data):
    parser = parser_for(tokenizer, request_data, reasoning_effort="low")
    delta = step(parser, tokenizer, request_data, "analysis</think>answer")
    assert delta.reasoning == "analysis"
    assert delta.content == "answer"


def test_interleaved_requests_do_not_share_pending_reasoning_or_tools(
    tokenizer, request_data
):
    a = parser_for(tokenizer, request_data, reasoning_effort="low")
    b = parser_for(tokenizer, request_data, reasoning_effort="low")
    a1 = step(
        a, tokenizer, request_data, 'A</think><tool_call>{"name":"calc","arguments":'
    )
    b1 = step(
        b,
        tokenizer,
        request_data,
        'B</think><tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>',
    )
    a2 = step(a, tokenizer, request_data, '{"x":1}}</tool_call>')
    assert a1.reasoning == "A" and b1.reasoning == "B" and not a2.reasoning
    assert flatten([a1, a2]) == [("calc", {"x": 1})]
    assert flatten([b1]) == [("calc", {"x": 2})]


def test_none_disables_reasoning_even_with_legacy_enable_flag(tokenizer, request_data):
    parser = parser_for(
        tokenizer, request_data, reasoning_effort="none", enable_thinking=True
    )
    delta = step(
        parser,
        tokenizer,
        request_data,
        '<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>',
    )
    assert not delta.reasoning
    assert flatten([delta]) == [("calc", {"x": 2})]


def test_superseded_runtime_patch_is_rejected(tokenizer, request_data, monkeypatch):
    from vllm.parser.abstract_parser import DelegatingParser

    def old_backport(self, *args, **kwargs):
        reasoning_message = None
        return reasoning_message

    monkeypatch.setattr(DelegatingParser, "parse_delta", old_backport)
    with pytest.raises(RuntimeError, match="Restore the original"):
        parser_for(tokenizer, request_data, reasoning_effort="low")
