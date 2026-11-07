"""Regression checks for smoke-test clients; no live server required."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def smoke(monkeypatch):
    monkeypatch.setitem(sys.modules, "_vllm_env", SimpleNamespace(make_client=None))
    path = Path(__file__).resolve().parents[1] / "examples/parser_smoke_1_7.py"
    spec = importlib.util.spec_from_file_location("parser_smoke_client", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def stream_with_usage(reasoning_field):
    def chunk(*, calls=None, finish=None, **fields):
        delta = SimpleNamespace(content=None, tool_calls=calls, **fields)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])

    def call(*, name=None, arguments=None, cid=None):
        return SimpleNamespace(
            index=0, id=cid,
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    usage = SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=42))
    return [
        usage,
        chunk(**{reasoning_field: "Check weather"}),
        chunk(calls=[call(cid="call_1", name="get_weather", arguments='{"city":')]),
        usage,
        chunk(calls=[call(arguments='"Tokio"}')], finish="tool_calls"),
        usage,
    ]


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_aggregator_preserves_split_tool_and_reasoning_around_usage(smoke, reasoning_field):
    result = smoke._aggregate_stream(stream_with_usage(reasoning_field))
    assert result["reasoning"] == "Check weather"
    assert result["finish"] == "tool_calls"
    assert result["n_tools"] == 1
    call = result["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": "Tokio"}


def test_streaming_probe_ignores_usage_without_losing_tool_arguments(smoke, monkeypatch):
    monkeypatch.setattr(smoke, "chat", lambda *a, **kw: stream_with_usage("reasoning"))
    result = smoke.test_3_streaming(None, "test-model")
    assert result.ok, result.detail
    assert json.loads(result.extras["calls"][0]["arguments"]) == {"city": "Tokio"}
