"""Required-mode constraints must not leave room for an unbounded prose lead-in."""

import json
import re

import pytest

from test_streaming_tool_parser_v31 import TOOL, collect, flatten, request_data, tokenizer


@pytest.fixture
def required_regex(tokenizer, request_data):
    request_data.tool_choice = "required"
    parser = TOOL(tokenizer, request_data.tools)
    return parser.adjust_request(request_data).structured_outputs.regex


@pytest.mark.parametrize("arguments", [
    {}, {"nested": {"items": [1, True, None]}},
    {"query": 'Zażółć <baz> "quoted" literal <tool_call> text'},
])
def test_required_accepts_complete_json_arguments(
    required_regex, arguments, tokenizer, request_data,
):
    call = '<tool_call>' + json.dumps({"name": "calc", "arguments": arguments}) + '</tool_call>'
    assert re.fullmatch(required_regex, call)
    assert re.fullmatch(required_regex, ' \n' + call + '\n' + call + ' ')
    parsed = TOOL(tokenizer, request_data.tools).extract_tool_calls(call, request_data)
    assert parsed.tools_called
    assert json.loads(parsed.tool_calls[0].function.arguments) == arguments


@pytest.mark.parametrize("output", [
    'Cześć! W czym mogę pomóc?',
    'Sprawdzę. <tool_call>{"name":"calc","arguments":{}}</tool_call>',
    '<tool_call>{"name":"calc","arguments":{}}</tool_call> Gotowe.',
    '<tool_call>not JSON</tool_call>',
    '<tool_call>{"name":"calc","arguments":{}}',
    '\n\n\n<tool_call>{"name":"calc","arguments":{}}</tool_call>',
    '\n' * 64,
])
def test_required_rejects_prose_and_unfinished_wrappers(required_regex, output):
    assert re.fullmatch(required_regex, output) is None


def test_auto_still_allows_text_without_adding_constraints(tokenizer, request_data):
    parser = TOOL(tokenizer, request_data.tools)
    assert parser.adjust_request(request_data).structured_outputs is None
    text = 'Sprawdzę. <tool_call>{"name":"calc","arguments":{}}</tool_call>'
    result = parser.extract_tool_calls(text, request_data)
    assert result.content == 'Sprawdzę. '
    assert result.tools_called


@pytest.mark.parametrize("batch", [1, 3, 100000])
def test_required_streaming_accepts_tags_spelled_with_ordinary_tokens(
    tokenizer, request_data, batch,
):
    request_data.tool_choice = "required"
    parser = TOOL(tokenizer, request_data.tools)
    request_data = parser.adjust_request(request_data)
    text = ' \n<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call> '
    ids = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    assert parser.tool_call_start_token_id not in ids
    assert parser.tool_call_end_token_id not in ids
    deltas = collect(parser, tokenizer, request_data, text, batch, split_special_tokens=True)
    assert flatten(deltas) == [('calc', {'x': 2})]
    assert not ''.join(d.content or '' for d in deltas)


def test_required_streaming_does_not_execute_a_truncated_text_tag(tokenizer, request_data):
    request_data.tool_choice = "required"
    parser = TOOL(tokenizer, request_data.tools)
    request_data = parser.adjust_request(request_data)
    deltas = collect(parser, tokenizer, request_data, '<tool_call>{"name":"calc"', 1,
                     split_special_tokens=True)
    assert flatten(deltas) == []
    assert not ''.join(d.content or '' for d in deltas)
