import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multiturn_probe_test", ROOT / "tests" / "live_multiturn_isolation.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def choice(calls=(), content="", reasoning="", finish="tool_calls"):
    return {"choices": [{"index": 0, "content": content, "reasoning": reasoning,
                         "tool_calls": list(calls), "finish_reason": finish}]}


def test_parallel_results_are_returned_with_matching_ids_in_reverse_order(monkeypatch):
    bodies = []
    def chat(url, body):
        bodies.append(body)
        if len(bodies) == 1:
            return choice([{"id": "a", "name": "add", "arguments": {"a": 17, "b": 23}},
                           {"id": "w", "name": "weather", "arguments": {"city": "Kraków"}}], reasoning="plan")
        messages = body["messages"]
        assert messages[1]["reasoning"] == "plan"
        assert [m["tool_call_id"] for m in messages[-2:]] == ["w", "a"]
        return choice(content="40 i 21", finish="stop")
    monkeypatch.setattr(probe, "chat", chat)
    report = probe.require_parallel(probe.conversation("local", "m", "synthetic", enabled=True,
        expected_counts={"add": 1, "weather": 1}, answers=(40, 21)))
    assert report["passed"], report
    assert len(report["executions"]) == 2


def test_unknown_tool_is_not_executed_and_failure_has_artifact(monkeypatch):
    monkeypatch.setattr(probe, "chat", lambda *a, **k: choice([
        {"id": "x", "name": "shell", "arguments": {"command": "touch file"}}]))
    report = probe.conversation("local", "m", "synthetic")
    assert not report["passed"] and report["executions"] == []
    assert len(report["turns"]) == 1


def test_missing_final_answer_and_no_parallel_calls_fail(monkeypatch):
    monkeypatch.setattr(probe, "chat", lambda *a, **k: choice(content="", finish="length"))
    report = probe.require_parallel(probe.conversation("local", "m", "synthetic"))
    assert not report["passed"]
    assert len(report["errors"]) == 2


def test_unbounded_tool_loop_is_stopped(monkeypatch):
    monkeypatch.setattr(probe, "chat", lambda *a, **k: choice([
        {"id": "a", "name": "add", "arguments": {"a": 1, "b": 2}}]))
    report = probe.conversation("local", "m", "synthetic", max_turns=2)
    assert not report["passed"]
    assert len(report["turns"]) == 2


def test_wrong_parallel_arguments_fail_even_when_final_numbers_match(monkeypatch):
    replies = iter([choice([
        {"id": "a", "name": "add", "arguments": {"a": 0, "b": 40}},
        {"id": "w", "name": "weather", "arguments": {"city": "Warszawa"}}]),
        choice(content="40 i 21", finish="stop")])
    monkeypatch.setattr(probe, "chat", lambda *a, **k: next(replies))
    report = probe.require_parallel(probe.conversation("local", "m", "synthetic",
        expected_counts={"add": 1, "weather": 1}, answers=(40, 21),
        expected_arguments={"add": {"a": 17, "b": 23}, "weather": {"city": "Kraków"}}))
    assert not report["passed"]
    assert "Tool arguments differ" in " ".join(report["errors"])


def test_parallel_duplicate_calls_are_not_a_retry(monkeypatch):
    replies = iter([choice([
        {"id": "w1", "name": "weather", "arguments": {"city": "Kraków"}},
        {"id": "w2", "name": "weather", "arguments": {"city": "Kraków"}}]),
        choice(content="21", finish="stop")])
    monkeypatch.setattr(probe, "chat", lambda *a, **k: next(replies))
    report = probe.conversation("local", "m", "synthetic", fail_first_weather=True,
        expected_counts={"weather": 2}, answers=(21,))
    assert not report["passed"]
    assert "later assistant turn" in " ".join(report["errors"])


def test_wrong_numeric_substring_is_not_a_correct_answer(monkeypatch):
    monkeypatch.setattr(probe, "chat", lambda *a, **k: choice(content="140 i -21", finish="stop"))
    assert not probe.conversation("local", "m", "synthetic", answers=(40, 21))["passed"]
