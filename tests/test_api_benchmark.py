import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("api_benchmark_test", ROOT / "scripts" / "benchmark_bielik_api.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_percentile_interpolation_and_validation():
    assert bench.percentile([4, 1, 3, 2], .5) == 2.5
    assert bench.percentile([1], .95) == 1
    with pytest.raises(ValueError):
        bench.percentile([], .95)


@pytest.mark.parametrize("raw", [{}, {"usage": {"completion_tokens": True}}, [{"usage": None}]])
def test_missing_or_invalid_token_usage_is_not_estimated(raw):
    with pytest.raises(ValueError):
        bench.usage_of({"raw": raw})


def test_stream_usage_can_arrive_after_choices():
    assert bench.usage_of({"raw": [{"choices": []}, {"usage": {"completion_tokens": 12}}]})["completion_tokens"] == 12


@pytest.mark.parametrize("levels,repeats,warmup", [([], 1, 0), ([1, 1], 1, 0), ([0], 1, 0), ([1], 0, 0), ([1], 1, -1)])
def test_invalid_runs_are_rejected_before_network(levels, repeats, warmup):
    with pytest.raises(ValueError):
        bench.run("local", "model", "v31", levels, repeats, warmup)


def test_failed_requests_remain_visible_in_statistics(monkeypatch):
    def fail(*a, **k):
        raise OSError("synthetic unavailable server")
    monkeypatch.setattr(bench, "chat", fail)
    reports = bench.run("local", "m", "v31", [1, 2], 2, warmup=0)
    assert all(len(r["rows"]) == 6 for r in reports)
    assert all(g["completed"] == 0 and g["errors"] == 2 and g["latency_p50_s"] is None
               for r in reports for g in r["summary"].values())


def test_manifest_requires_comparable_provenance(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{}')
    with pytest.raises(ValueError):
        bench.manifest(path)


def test_cli_runs_and_reports_warmup_errors(monkeypatch, tmp_path):
    import json
    metadata = tmp_path / "manifest.json"
    metadata.write_text(json.dumps({"model_revision": "m", "vllm_version": "v", "parser_commit": "p", "template_sha256": "a"*64}))
    monkeypatch.setattr(bench, "run", lambda *a, **k: [{"concurrency": 1, "summary": {}, "rows": [{"passed": True}]}])
    args = ["--model", "offline", "--manifest", str(metadata)]
    assert bench.main([*args, "--output", str(tmp_path / "ok.json")]) == 0
    def failure(*a, **k):
        raise OSError("synthetic warmup failure")
    monkeypatch.setattr(bench, "run", failure)
    output = tmp_path / "failed.json"
    assert bench.main([*args, "--output", str(output)]) == 2
    assert "warmup failure" in json.loads(output.read_text())["manifest"]["run_error"]
