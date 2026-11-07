"""Check the public request shape used by the structured-output example."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def example(monkeypatch):
    client = Mock()
    monkeypatch.setitem(sys.modules, "_vllm_env", SimpleNamespace(
        make_client=lambda: (client, "test-model"),
    ))
    monkeypatch.setitem(sys.modules, "termcolor", None)
    path = Path(__file__).resolve().parents[1] / "examples/structured_output.py"
    spec = importlib.util.spec_from_file_location("structured_example", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("effort", ["none", "medium"])
def test_json_schema_and_separate_thinking_budget_reach_server(example, effort):
    schema = example.CarDescription.model_json_schema()
    example.chat_completion_request(
        [{"role": "user", "content": "Choose a car"}], schema,
        reasoning_effort=effort,
    )
    sent = example.client.chat.completions.create.call_args.kwargs
    assert sent["max_tokens"] == 800
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "CarDescription", "schema": schema, "strict": True},
    }
    assert schema["additionalProperties"] is False
    body = sent["extra_body"]
    assert "guided_json" not in body
    assert body["skip_special_tokens"] is False
    assert body["chat_template_kwargs"] == {
        "reasoning_effort": effort, "enable_thinking": effort != "none",
    }
    if effort == "none":
        assert "thinking_token_budget" not in body
    else:
        assert body["thinking_token_budget"] == 256 < sent["max_tokens"]


def test_truncated_answer_warns_without_promoting_reasoning(example, caplog):
    message = {"role": "assistant", "content": None, "reasoning": "unfinished"}
    example.client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(finish_reason="length", message=SimpleNamespace(
            model_dump=lambda: message,
        )),
    ])
    history = []
    example.add_turn("Choose a car", history, reasoning_effort="medium")
    assert history[-1]["content"] is None
    assert history[-1]["reasoning"] == "unfinished"
    assert "reached max_tokens" in caplog.text
