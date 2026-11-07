"""A named tool_choice must force exactly one call of that function."""

import json
import re

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)

from test_streaming_tool_parser_v31 import TOOL, collect, flatten, request_data, tokenizer


def named(name):
    return ChatCompletionNamedToolChoiceParam(
        type="function", function={"name": name})


NAMED = named("calc")


@pytest.fixture
def adjusted(tokenizer, request_data):
    request_data.tool_choice = NAMED
    parser = TOOL(tokenizer, request_data.tools)
    return parser, parser.adjust_request(request_data)


def test_named_choice_becomes_constrained_auto(adjusted):
    _, request = adjusted
    assert request.tool_choice == "auto"
    assert request.skip_special_tokens is False
    assert request.structured_outputs.regex.startswith(r"\s{0,2}<tool_call>")


@pytest.mark.parametrize("arguments", [
    {}, {"x": 2}, {"query": 'a <b> "q" </tool_call> z'}, {"n": {"k": [1, None]}},
])
def test_named_regex_accepts_one_call_of_that_function(adjusted, arguments):
    regex = adjusted[1].structured_outputs.regex
    for sep in ((", ", ": "), (",", ":")):
        call = ('<tool_call>{"name"%s"calc"%s"arguments"%s%s}</tool_call>'
                % (sep[1], sep[0], sep[1], json.dumps(arguments)))
        assert re.fullmatch(regex, call), call
        assert re.fullmatch(regex, " \n" + call + "\n")


@pytest.mark.parametrize("output", [
    '<tool_call>{"name": "other", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>',
    'Sprawdzę. <tool_call>{"name": "calc", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "calc", "arguments": {}}',
    '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
    '<tool_call>{"name": "calc", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
    '<tool_call>{"name": "other", "arguments": {}}</tool_call>',
    'Cześć!',
])
def test_named_regex_rejects_other_output(adjusted, output):
    assert re.fullmatch(adjusted[1].structured_outputs.regex, output) is None


def test_named_regex_escapes_function_names(tokenizer, request_data):
    request_data.tool_choice = named("a.b(c)")
    parser = TOOL(tokenizer, request_data.tools)
    regex = parser.adjust_request(request_data).structured_outputs.regex
    assert re.fullmatch(regex, '<tool_call>{"name": "a.b(c)", "arguments": {}}</tool_call>')
    assert re.fullmatch(regex, '<tool_call>{"name": "aXb(c)", "arguments": {}}</tool_call>') is None


def test_named_streaming_with_ordinary_token_tags(adjusted, tokenizer):
    parser, request = adjusted
    text = '<tool_call>{"name": "calc", "arguments": {"x": 2}}</tool_call>'
    deltas = collect(parser, tokenizer, request, text, 3, split_special_tokens=True)
    assert flatten(deltas) == [('calc', {'x': 2})]
    assert not ''.join(d.content or '' for d in deltas)


def test_other_tool_choices_unchanged(tokenizer, request_data):
    for choice in ("auto", "none"):
        request_data.tool_choice = choice
        parser = TOOL(tokenizer, request_data.tools)
        assert parser.adjust_request(request_data).structured_outputs is None


def test_named_nesting_limit_is_four_levels(adjusted):
    regex = adjusted[1].structured_outputs.regex
    wrap = '<tool_call>{"name": "calc", "arguments": %s}</tool_call>'
    assert re.fullmatch(regex, wrap % '{"a":{"b":{"c":{"d":1}}}}')
    assert re.fullmatch(regex, wrap % '{"a":{"b":{"c":{"d":{"e":1}}}}}') is None


def test_named_nonstreaming_extracts_forced_output(adjusted):
    parser, request = adjusted
    text = '<tool_call>{"name": "calc", "arguments": {"x": 2}}</tool_call>'
    result = parser.extract_tool_calls(text, request)
    assert result.tools_called
    assert result.tool_calls[0].function.name == "calc"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 2}


def test_required_still_uses_required_regex(tokenizer, request_data):
    request_data.tool_choice = "required"
    parser = TOOL(tokenizer, request_data.tools)
    regex = parser.adjust_request(request_data).structured_outputs.regex
    two = '<tool_call>{"name":"a","arguments":{}}</tool_call>' * 2
    assert re.fullmatch(regex, two)
