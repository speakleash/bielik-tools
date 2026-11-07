import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("bfcl_eval")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("bfcl_probe_test", ROOT / "scripts" / "evaluate_bfcl_ast.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_real_bfcl_checker_accepts_correct_and_rejects_wrong_args():
    examples, truth, _ = probe.data("simple_python", 0, 1)
    example = examples[0]
    correct = [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]
    assert probe.score(example, correct, truth[example["id"]], "simple_python")["valid"]
    wrong = [{"name": "calculate_triangle_area", "arguments": {"base": 11, "height": 5}}]
    assert not probe.score(example, wrong, truth[example["id"]], "simple_python")["valid"]


def test_generated_python_stays_data_and_is_not_executed(tmp_path):
    target = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path({str(target)!r}).touch()"
    example = {"function": [{"name": "danger", "parameters": {"type": "dict", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}}]}
    verdict = probe.score(example, [{"name": "danger", "arguments": {"command": payload}}],
                          [{"danger": {"command": [payload]}}], "simple_python")
    assert verdict["valid"]
    assert not target.exists()


@pytest.mark.parametrize("category", ["multi_turn_base", "java", "executable", "unknown"])
def test_executable_or_unknown_categories_are_rejected(category):
    with pytest.raises(ValueError):
        probe.data(category, 0, 1)


def test_schema_conversion_is_recursive_and_unknown_types_fail():
    assert probe.schema({"type": "dict", "properties": {"xs": {"type": "array", "items": {"type": "float"}}}}) == {
        "type": "object", "properties": {"xs": {"type": "array", "items": {"type": "number"}}}}
    with pytest.raises(ValueError):
        probe.schema({"type": "JavaClass"})


@pytest.mark.parametrize("mode,expected", [("correct", 0), ("wrong", 1), ("error", 2)])
def test_cli_gates_and_retains_response_on_checker_error(monkeypatch, tmp_path, mode, expected):
    import json
    example = {"id": "example", "question": [[{"role": "user", "content": "test"}]], "function": []}
    monkeypatch.setattr(probe, "data", lambda *a: ([example], {"example": []}, {}))
    response = {"choices": [{"tool_calls": []}], "raw": "retained"}
    monkeypatch.setattr(probe, "chat", lambda *a: response)
    def checker(*args):
        if mode == "error":
            raise RuntimeError("checker failure")
        return {"valid": mode == "correct"}
    monkeypatch.setattr(probe, "score", checker)
    output = tmp_path / "report.json"
    assert probe.main(["--model", "test", "--categories", "simple_python", "--limit", "1",
                       "--min-score", "1", "--output", str(output)]) == expected
    report = json.loads(output.read_text())
    assert len(report["rows"]) == 4
    assert all(row["response"] == response for row in report["rows"])
