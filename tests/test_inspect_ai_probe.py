import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("inspect_ai")
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("inspect_probe_test", ROOT / "scripts" / "probe_inspect_ai.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def call(id="a", args=None):
    return ToolCall(id=id, function="add", arguments=args or {"a": 17, "b": 23})


def test_raw_call_validation_before_native_tool_execution():
    msg = ChatMessageAssistant(content="", tool_calls=[call()])
    assert not probe.validate_calls(msg, "tool_calls")
    assert probe.validate_calls(msg, "unknown")
    assert probe.validate_calls(msg, "tool_calls", disabled=True)
    assert probe.validate_calls(ChatMessageAssistant(content="", tool_calls=[call(), call()]), "tool_calls")
    assert probe.validate_calls(ChatMessageAssistant(content="", tool_calls=[call(args={"a": "17", "b": "23"})]), "tool_calls")


def test_tool_results_must_match_each_call_once():
    assistant = ChatMessageAssistant(content="", tool_calls=[call()])
    result = ChatMessageTool(content="40", tool_call_id="a")
    final = ChatMessageAssistant(content="40")
    assert not probe.correlation_errors([assistant, result, final])
    assert probe.correlation_errors([assistant, final])
    assert probe.correlation_errors([assistant, result, result, final])
    assert probe.correlation_errors([assistant])


def test_synthetic_history_uses_native_reasoning_metadata():
    from inspect_ai.model import ContentReasoning
    from inspect_ai.model._openai import openai_assistant_content

    message = ChatMessageAssistant(content=[ContentReasoning(internal="reasoning", reasoning="17+23=40"),
                                             {"type": "text", "text": "40"}])
    text, extra = openai_assistant_content(message)
    assert text.strip() == "40" and "<think>" not in text
    assert extra == {"reasoning": "17+23=40"}


def test_final_answer_must_use_the_returned_values():
    assert probe.returned_values_present("Suma40, temperatura21°C", [40, 21])
    for text in ("Nie wiem i nie podam żadnego wyniku.", "140 i -21", "40.5 i 21"):
        assert not probe.returned_values_present(text, [40, 21])


@pytest.mark.parametrize("name,batches", [
    ("parallel", [[("add", {"a": 17, "b": 23}), ("weather", {"city": "Kraków"})]]),
    ("named", [[("add", {"a": 17, "b": 23})]]),
    ("retry", [[("weather", {"city": "Kraków"})], [("weather", {"city": "Kraków"})]]),
])
def test_native_scorer_rejects_ignoring_successful_tool_results(name, batches):
    import asyncio
    import json
    from types import SimpleNamespace
    from inspect_ai.model import ModelOutput, ChatCompletionChoice
    from inspect_ai.scorer import INCORRECT

    task, ledger = probe.build_task(name, False, "v31")
    messages = []
    for turn, batch in enumerate(batches):
        calls = [ToolCall(id=f"{turn}-{index}", function=n, arguments=a)
                 for index, (n, a) in enumerate(batch)]
        messages.append(ChatMessageAssistant(content="", tool_calls=calls))
        for call in calls:
            value = ledger.execute(call.function, call.arguments)
            ledger.calls[-1]["turn"] = turn
            messages.append(ChatMessageTool(content=json.dumps(value), tool_call_id=call.id))
    final = ChatMessageAssistant(content="Nie wiem i nie podam żadnego wyniku.")
    messages.append(final)
    output = ModelOutput(model="mock", choices=[ChatCompletionChoice(message=final, stop_reason="stop")])
    score = asyncio.run(task.scorer[0](SimpleNamespace(messages=messages, output=output), None))
    assert score.value == INCORRECT
    assert "returned tool values" in score.explanation
