import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("quality_eval_test", ROOT / "scripts" / "evaluate_bielik_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def answer(text, calls=None, finish="stop"):
    return {"content": text, "reasoning": "", "tool_calls": calls or [], "finish_reason": finish}


def test_quality_requires_the_answer_not_a_numeric_substring():
    case = quality.CASES[0]
    assert not quality.score(answer("391."), case)
    for text in ("1391", "3910", "Nie 391, tylko 392", "<think>391</think>"):
        assert quality.score(answer(text), case)


def test_tool_argument_quality_is_separate_from_valid_json():
    case = next(c for c in quality.CASES if c["id"] == "add_tool")
    assert not quality.score(answer("", case["calls"], "tool_calls"), case)
    assert quality.score(answer("", [{"name": "add", "arguments": {"a": 0, "b": 40}}], "tool_calls"), case)


def test_schema_rejects_wrong_values_types_and_extra_fields():
    case = next(c for c in quality.CASES if c["id"] == "schema")
    assert not quality.score(answer('{"answer":391}'), case)
    for text in ('{"answer":392}', '{"answer":true}', '{"answer":391,"extra":1}', 'not JSON'):
        assert quality.score(answer(text), case)


def test_wilson_interval_and_empty_run():
    low, high = quality.wilson(5, 10)
    assert 0 < low < .5 < high < 1
    with pytest.raises(ValueError):
        quality.wilson(0, 0)


def test_summary_does_not_merge_reasoning_or_sampling_groups():
    rows = [{"effort": "none", "sampling": "greedy", "correct": True},
            {"effort": "medium", "sampling": "sampled", "correct": False}]
    summary = quality.summarize(rows)
    assert summary["none/greedy"]["accuracy"] == 1
    assert summary["medium/sampled"]["accuracy"] == 0


def test_cli_distinguishes_report_quality_failure_and_run_error(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(quality, "chat", lambda *a, **k: {"choices": [answer("wrong")]})
    def run(name, extra=()):
        path = tmp_path / name
        code = quality.main(["--model", "offline", "--samples", "0", "--output", str(path), *extra])
        return code, json.loads(path.read_text())
    code, report = run("report.json")
    assert code == 0 and report["quality_passed"] is None and not report["run_errors"]
    code, report = run("gate.json", ["--min-score", "1"])
    assert code == 1 and report["quality_passed"] is False
    def unavailable(*a, **k):
        raise OSError("synthetic missing server")
    monkeypatch.setattr(quality, "chat", unavailable)
    code, report = run("unavailable.json", ["--min-score", "0"])
    assert code == 2 and report["run_errors"] == len(quality.CASES) * 2
