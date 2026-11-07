import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("pydantic_ai")
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, RetryInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("pydantic_ai_probe_test", ROOT / "scripts" / "probe_pydantic_ai.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_raw_argument_validation_precedes_framework_coercion():
    valid = ModelResponse(parts=[ToolCallPart("add", {"a": 17, "b": 23}, "a")], finish_reason="tool_call")
    assert not probe.validate_response(valid)
    invalid = ModelResponse(parts=[ToolCallPart("add", {"a": "17", "b": "23"}, "a")], finish_reason="tool_call")
    assert probe.validate_response(invalid)


@pytest.mark.parametrize("finish", [None, "length", "stop"])
def test_tool_response_must_have_correct_finish(finish):
    response = ModelResponse(parts=[ToolCallPart("add", {"a": 17, "b": 23}, "a")], finish_reason=finish)
    assert probe.validate_response(response)


def test_duplicate_call_ids_and_unknown_names_fail():
    response = ModelResponse(parts=[ToolCallPart("add", {"a": 17, "b": 23}, "a"),
        ToolCallPart("weather", {"city": "Kraków"}, "a")], finish_reason="tool_call")
    assert probe.validate_response(response)
    assert probe.validate_response(ModelResponse(parts=[ToolCallPart("shell", {}, "x")], finish_reason="tool_call"))


def test_structured_output_type_is_strict():
    assert probe.Answer(answer=391).answer == 391
    for bad in (True, "391"):
        with pytest.raises(ValueError):
            probe.Answer(answer=bad)


def test_native_model_retry_is_identified_by_its_cause():
    retry = ToolReturnPart("add", "synthetic error", "x", outcome="retried",
                          retry=RetryInfo(cause="model_retry", attempt=1, max_retries=2))
    success = ToolReturnPart("add", {"value": 40}, "y")
    assert probe.model_retry_returns([ModelRequest(parts=[retry, success])]) == [retry]
    assert not probe.model_retry_returns([ModelRequest(parts=[success])])


def test_parallel_attempts_do_not_prove_validation_repair(monkeypatch):
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import TextPart
    from pydantic_ai.usage import RequestUsage

    calls = [0]
    async def respond(messages, info):
        calls[0] += 1
        if calls[0] == 1:
            parts = [ToolCallPart("add", {"a": 17, "b": 23}, "a1"),
                     ToolCallPart("add", {"a": 17, "b": 23}, "a2")]
            finish = "tool_call"
        else:
            parts, finish = [TextPart("40")], "stop"
        return ModelResponse(parts=parts, finish_reason=finish,
                             usage=RequestUsage(input_tokens=1, output_tokens=1))
    monkeypatch.setattr(probe, "OpenAIChatModel", lambda *a, **k: FunctionModel(respond))
    report = probe.scenario("http://localhost:1/v1", "offline", "v31", "validation_retry", False, False)
    assert not report["passed"]
    assert "Native ModelRetry path was not exercised" in report["errors"]
    assert len(report["attempts"]) == 2
    assert report["attempts"][0]["turn"] == report["attempts"][1]["turn"]


@pytest.mark.parametrize("finish", ["tool_calls", "length"])
def test_native_stream_completes_before_parallel_dispatch(monkeypatch, finish):
    import httpx
    import json
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider

    requests = []
    def transport(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            delta = {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "a", "type": "function", "function": {
                    "name": "add", "arguments": '{"a":17,"b":23}'}},
                {"index": 1, "id": "w", "type": "function", "function": {
                    "name": "weather", "arguments": '{"city":"Kraków"}'}}]}
            stop = finish
        else:
            delta, stop = {"role": "assistant", "content": "add=40; weather=21"}, "stop"
        events = [{"id": "test", "object": "chat.completion.chunk", "created": 1,
                   "model": "offline", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                  {"id": "test", "object": "chat.completion.chunk", "created": 1,
                   "model": "offline", "choices": [{"index": 0, "delta": {}, "finish_reason": stop}]}]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    client = AsyncOpenAI(api_key="offline", base_url="http://offline/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    provider = OpenAIProvider(openai_client=client)
    monkeypatch.setattr(probe, "OpenAIProvider", lambda **kwargs: provider)
    report = probe.scenario("http://offline/v1", "offline", "v31", "parallel", False, True)
    if finish == "tool_calls":
        assert report["passed"], report["errors"]
        assert len(requests) == 2
        assert [c["turn"] for c in report["executions"]] == [0, 0]
    else:
        assert not report["passed"]
        assert report["executions"] == []
