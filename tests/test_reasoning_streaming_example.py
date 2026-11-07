"""Client regressions without a server or model weights."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(params=[False, True], ids=["default-environment", "no-termcolor"])
def example(monkeypatch, request):
    if request.param:
        # Reproduce a minimal CI environment even when termcolor is installed.
        monkeypatch.setitem(sys.modules, "termcolor", None)
    client = Mock()
    monkeypatch.setitem(
        sys.modules,
        "_vllm_env",
        SimpleNamespace(make_client=lambda: (client, "test-model")),
    )
    path = Path(__file__).resolve().parents[1] / "examples/reasoning_streaming.py"
    spec = importlib.util.spec_from_file_location("reasoning_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(*, finish=None, **fields):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(**fields), finish_reason=finish)]
    )


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_mixed_final_delta_preserves_reasoning_and_answer(example, field):
    result = example.process_streamed_response([
        SimpleNamespace(choices=[]),
        chunk(**{field: "Check: "}),
        chunk(**{field: "2+2=4.", "content": "4", "finish": "stop"}),
    ])
    assert result == {
        "role": "assistant", "reasoning": "Check: 2+2=4.", "content": "4"
    }


def test_reasoning_only_is_not_promoted_to_final_answer(example, caplog):
    result = example.process_streamed_response([
        chunk(reasoning="Unfinished analysis", finish="length")
    ])
    assert result["content"] == ""
    assert result["reasoning"] == "Unfinished analysis"
    assert "without a final answer" in caplog.text


def test_none_stream_has_only_answer(example):
    result = example.process_streamed_response([
        chunk(content="War"), chunk(content="szawa", finish="stop")
    ])
    assert result == {"role": "assistant", "reasoning": None, "content": "Warszawa"}


@pytest.mark.parametrize("answer", [None, "4"])
def test_printed_stream_closes_thinking_label(example, capsys, answer):
    example.process_streamed_response([
        chunk(reasoning="Check", content=answer, finish="stop")
    ], print_stream=True)
    output = capsys.readouterr().out
    assert output.count("[ Thinking... ]") == 1
    assert output.count("[ Thinking finished ]") == 1
    assert output.index("Check") < output.index("[ Thinking finished ]")
    if answer:
        assert output.index("[ Thinking finished ]") < output.index(answer)


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "max"])
def test_request_keeps_template_switches_consistent(example, effort):
    messages = [{"role": "user", "content": "2+2?"}]
    example.chat_completion_request(messages, effort)
    kwargs = example.client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == messages
    assert kwargs["stream"] is True
    assert kwargs["extra_body"] == {
        "skip_special_tokens": False,
        "chat_template_kwargs": {
            "reasoning_effort": effort, "enable_thinking": effort != "none"
        },
    }
