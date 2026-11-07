import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("langchain_openai")
pytest.importorskip("langgraph")
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("langgraph_probe_test", ROOT / "scripts" / "probe_langgraph.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def model():
    return probe.ChatBielik(model="test", base_url="http://localhost:1/v1", api_key="dummy")


def test_stock_drop_is_explicit_and_adapter_preserves_full_reasoning():
    raw = {"choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "391", "reasoning": "17*23"}}]}
    stock = ChatOpenAI(model="test", api_key="dummy")._create_chat_result(raw)
    assert "reasoning" not in stock.generations[0].message.additional_kwargs
    result = model()._create_chat_result(raw)
    assert result.generations[0].message.additional_kwargs["reasoning"] == "17*23"


def test_adapter_preserves_reasoning_chunks_and_history():
    llm = model()
    chunk = llm._convert_chunk_to_generation_chunk({"choices": [{"index": 0, "delta": {
        "reasoning_content": "plan"}}]}, AIMessageChunk, None)
    assert chunk.message.additional_kwargs["reasoning"] == "plan"
    payload = llm._get_request_payload([AIMessage("40", additional_kwargs={"reasoning": "17+23=40"})])
    assert payload["messages"][0]["reasoning"] == "17+23=40"


@pytest.mark.parametrize("kind", ["coercion", "missing_finish", "length", "duplicate_ids"])
def test_invalid_intermediate_turn_cannot_reach_tool_node(monkeypatch, kind):
    calls = [{"id": "a", "name": "add", "args": {"a": 17, "b": 23}, "type": "tool_call"}]
    finish = "tool_calls"
    name = "single"
    if kind == "coercion":
        calls[0]["args"] = {"a": "17", "b": "23"}
    elif kind == "missing_finish":
        finish = None
    elif kind == "length":
        finish = "length"
    else:
        name = "parallel"
        calls.append({"id": "a", "name": "weather", "args": {"city": "Kraków"}, "type": "tool_call"})
    response = AIMessage("", tool_calls=calls, response_metadata={"finish_reason": finish})
    monkeypatch.setattr(probe.ChatBielik, "invoke", lambda *a, **k: response)
    report = probe.scenario("http://localhost:1/v1", "test", "v31", name, False, False)
    assert not report["passed"]
    assert report["executions"] == []
