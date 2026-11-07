"""Offline protocol checks for the shared live client, including fragmented SSE."""
import importlib.util
import json
from io import BytesIO
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "testing_api.py"
spec = importlib.util.spec_from_file_location("testing_api_test", PATH)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def event(delta, *, index=0, finish=None):
    return {"choices": [{"index": index, "delta": delta, "finish_reason": finish}]}


def stream(*events, done=True):
    data = [b": keepalive\n", b"\n"]
    data += [("data: " + json.dumps(e) + "\n\n").encode() for e in events]
    if done:
        data.append(b"data: [DONE]\n")
    return data


def test_stream_combines_fragments_and_keeps_choices_separate():
    first = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "ad", "arguments": '{"a":'}}]}
    second = {"tool_calls": [{"index": 0, "function": {"name": "d", "arguments": '17,"b":23}'}}]}
    result = api.read_stream(stream(event({"role": "assistant"}), event({"reasoning": "Myślę."}),
        event(first), event({"content": "osobna odpowiedź"}, index=1, finish="stop"), event(second, finish="tool_calls")))
    assert len(result["choices"]) == 2
    a, b = result["choices"]
    assert a["tool_calls"] == [{"id": "call_1", "name": "add", "arguments": {"a": 17, "b": 23}}]
    assert a["reasoning"] == "Myślę." and not a["content"]
    assert b["content"] == "osobna odpowiedź" and not b["reasoning"]


@pytest.mark.parametrize("events,done", [
    ([], True), ([event({"content": "partial"})], True),
    ([event({"content": "answer"}, finish="stop")], False),
    ([{"error": {"message": "generation failed"}}], True),
])
def test_incomplete_or_error_stream_fails(events, done):
    with pytest.raises(api.ProtocolError):
        api.read_stream(stream(*events, done=done))


@pytest.mark.parametrize("arguments", ["[]", "null", "2", "not JSON", '{"a":NaN}',
                                      '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}'])
def test_non_object_or_invalid_tool_arguments_fail(arguments):
    with pytest.raises(ValueError):
        api.tool_value("add", arguments)


def test_schema_check_rejects_bool_and_extra_fields():
    choice = {"index": 0, "content": '{"answer":true}', "reasoning": "", "tool_calls": [], "finish_reason": "stop"}
    assert api.check_choice(choice, {"schema": True})
    assert not api.check_choice({**choice, "content": '{"answer":391}'}, {"schema": True})
    assert api.check_choice({**choice, "content": '{"answer":391,"extra":1}'}, {"schema": True})


def test_none_and_reasoning_expectations_are_enforced():
    choice = {"index": 0, "content": "", "reasoning": "hidden", "tool_calls": [{"id": "x", "name": "add", "arguments": {}}], "finish_reason": "tool_calls"}
    assert len(api.check_choice(choice, {"tools": [], "reasoning": False, "content": True, "finish": ["stop"]})) == 4


def test_truncated_markup_is_allowed_only_when_explicit():
    choice = {"index": 0, "content": "<tool_call>", "reasoning": "", "tool_calls": [], "finish_reason": "length"}
    assert api.check_choice(choice, {"tools": [], "finish": ["length"]})
    assert not api.check_choice(choice, {"tools": [], "finish": ["length"], "allow_markup": True})


@pytest.mark.parametrize("streaming,indices", [(False, [0, 0]), (True, [7, 8]), (False, [True, 0])])
def test_chat_rejects_wrong_choice_indices(monkeypatch, streaming, indices):
    if streaming:
        wire = b"".join(stream(*(event({"content": "x"}, index=i, finish="stop") for i in indices)))
    else:
        wire = json.dumps({"choices": [{"index": i, "message": {"content": "x"}, "finish_reason": "stop"} for i in indices]}).encode()
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: BytesIO(wire))
    with pytest.raises(api.ProtocolError, match="choice index"):
        api.chat("http://localhost:1", {"n": 2, "stream": streaming})


def test_fixed_tool_argument_expectation_is_enforced():
    choice = {"index": 0, "content": "", "reasoning": "", "tool_calls": [
        {"id": "x", "name": "add", "arguments": {"not_an_add_parameter": "wrong"}}], "finish_reason": "tool_calls"}
    assert api.check_choice(choice, {"tools": ["add"], "arguments": [{"a": 17, "b": 23}]})


def test_stream_rejects_non_json_constants():
    delta = {"tool_calls": [{"index": 0, "id": "x", "function": {"name": "add", "arguments": '{"a":NaN}'}}]}
    with pytest.raises(api.ProtocolError, match="Non-JSON"):
        api.read_stream(stream(event(delta, finish="tool_calls")))


def test_repeated_tool_id_is_identity_not_concatenated_text():
    first = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "add", "arguments": '{"a":'}}]}
    second = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"arguments": '17,"b":23}'}}]}
    result = api.read_stream(stream(event(first), event(second, finish="tool_calls")))
    assert result["choices"][0]["tool_calls"][0]["id"] == "call_1"
    second["tool_calls"][0]["id"] = "changed"
    with pytest.raises(api.ProtocolError, match="ID changed"):
        api.read_stream(stream(event(first), event(second, finish="tool_calls")))


def test_only_named_choice_accepts_legacy_stop_finish():
    import sys
    sys.path.insert(0, str(PATH.parent))
    spec = importlib.util.spec_from_file_location("matrix_finish_test", PATH.parents[1] / "tests" / "live_api_matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    for name, _, expected in matrix.cases("v31"):
        if name.startswith("named-"):
            assert set(expected["finish"]) == {"stop", "tool_calls"}
        elif name.startswith(("required-", "auto-")):
            assert expected["finish"] == ["tool_calls"]
