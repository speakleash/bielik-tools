"""A partial or malformed wrapper must never become an executable call."""
import json

import pytest

pytest.importorskip("vllm")
from test_parser_replay import make_capture, replay_module
from test_streaming_tool_parser_v31 import TOOL, collect, flatten, request_data, tokenizer

CALL = '<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>'


def test_every_truncation_of_a_call_is_non_executable(tokenizer, request_data):
    for cut in range(len(CALL)):
        partial = CALL[:cut]
        parsed = TOOL(tokenizer, request_data.tools).extract_tool_calls(partial, request_data)
        assert not parsed.tools_called, (cut, partial)
        assert parsed.tool_calls == []
        deltas = collect(TOOL(tokenizer, request_data.tools), tokenizer, request_data, partial, 1)
        assert flatten(deltas) == [], (cut, partial)


@pytest.mark.parametrize("payload", [
    '[]', 'null', '2', '{}', '{"name":"calc"}', '{"name":"","arguments":{}}',
    '{"name":2,"arguments":{}}', '{"name":"calc","arguments":[]}',
    '{"name":"calc","arguments":null}', '{"name":"calc","arguments":"{}"}',
    '{"name":"calc","arguments":{"x":NaN}}', '{"name":"calc","arguments":{"x":Infinity}}',
    '{"name":"calc","arguments":{"x":1e999}}', '{"name":"calc","arguments":{"x":01}}',
    '{"name":"calc","arguments":{"x":"\\q"}}',
])
def test_malformed_completed_calls_remain_data(tokenizer, request_data, payload):
    text = '<tool_call>' + payload + '</tool_call>'
    full = TOOL(tokenizer, request_data.tools).extract_tool_calls(text, request_data)
    assert not full.tools_called and full.content == text
    deltas = collect(TOOL(tokenizer, request_data.tools), tokenizer, request_data, text, 1)
    assert flatten(deltas) == []
    assert ''.join(d.content or '' for d in deltas) == text


@pytest.mark.parametrize("text", [
    'Before ' + CALL + ' After',
    CALL + '\nBetween ' + CALL + ' After',
    '<tool_call>{"name":"bad","arguments":[]}</tool_call>' + CALL,
    '<tool_call>' + json.dumps({"name": "calc", "arguments": {
        "text": 'quote: " slash: \\ literal </tool_call> and <tool_call>'}}) + '</tool_call>',
])
def test_valid_calls_and_surrounding_data_have_parity(tokenizer, text):
    record = make_capture(tokenizer, text)
    expected = replay_module.replay(record, tokenizer)
    assert expected["tool_calls"]
    for batch in (1, 2, 7, 100000):
        length = len(record["output_token_ids"])
        boundaries = list(range(batch, length, batch)) + [length]
        assert replay_module.replay(record, tokenizer, boundaries) == expected


def test_large_string_and_split_escaped_closing_tag(tokenizer, request_data):
    value = "x" * 2048 + '\\" </tool_call> " ' + "y" * 2048
    text = '<tool_call>' + json.dumps({"name": "calc", "arguments": {"text": value}}) + '</tool_call>'
    deltas = collect(TOOL(tokenizer, request_data.tools), tokenizer, request_data, text, 3)
    assert flatten(deltas) == [("calc", {"text": value})]


def test_stream_boundary_scanning_work_is_linear():
    reads = [0]
    class CountedText(str):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            reads[0] += len(value)
            return value
    boundary = TOOL.__init__.__globals__["_StreamingBoundary"]()
    text = json.dumps({"name": "calc", "arguments": {"text": "x" * 32000}}) + '</tool_call>'
    found = -1
    for end in range(4, len(text) + 4, 4):
        found = boundary.find(CountedText(text[:end]), 0)
    assert found == len(text) - len('</tool_call>')
    assert reads[0] < len(text) + 100  # only a short split wrapper is revisited
