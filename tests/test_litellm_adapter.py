import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("litellm_probe_test", ROOT / "scripts" / "probe_litellm.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Dump:
    def __init__(self, data):
        self.data = data
    def model_dump(self):
        return self.data


def test_sdk_forwards_custom_fields_and_retains_reasoning(monkeypatch):
    seen = []
    def completion(**kwargs):
        seen.append(kwargs)
        return Dump({"choices": [{"index": 0, "finish_reason": "stop", "message": {
            "content": "391", "reasoning_content": "17*23"}}]})
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    result = probe.sdk_chat("http://local/v1", {"model": "b", "messages": [],
        "chat_template_kwargs": {"reasoning_effort": "medium"}, "thinking_token_budget": 64,
        "include_reasoning": True})
    assert result["choices"][0]["reasoning"] == "17*23"
    assert seen[0]["model"] == "hosted_vllm/b"
    assert seen[0]["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "medium"},
        "thinking_token_budget": 64, "include_reasoning": True}


def test_sdk_stream_preserves_reasoning_and_tool_fragments(monkeypatch):
    chunks = [Dump({"choices": [{"index": 0, "delta": {"reasoning": "plan"}}]}),
              Dump({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "x",
                  "function": {"name": "add", "arguments": '{"a":17,"b":23}'}}]}, "finish_reason": "tool_calls"}]})]
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=lambda **k: iter(chunks)))
    result = probe.sdk_chat("http://local/v1", {"model": "b", "messages": [], "stream": True})
    assert result["choices"][0]["reasoning"] == "plan"
    assert result["choices"][0]["tool_calls"][0]["arguments"] == {"a": 17, "b": 23}


def test_sdk_stream_error_is_not_a_successful_termination(monkeypatch):
    def broken():
        yield Dump({"choices": [{"index": 0, "delta": {"content": "partial"}}]})
        raise OSError("synthetic connection loss")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=lambda **k: broken()))
    with pytest.raises(OSError):
        probe.sdk_chat("http://local/v1", {"model": "b", "messages": [], "stream": True})


def test_missing_finish_for_one_choice_fails_and_retains_sdk_chunks(monkeypatch):
    chunks = [Dump({"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": "stop"},
                                {"index": 1, "delta": {"content": "b"}}]})]
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=lambda **k: iter(chunks)))
    with pytest.raises(ValueError, match="Incomplete SSE") as caught:
        probe.sdk_chat("http://local/v1", {"model": "b", "messages": [], "stream": True, "n": 2})
    assert caught.value.sdk_chunks == [chunks[0].data]
