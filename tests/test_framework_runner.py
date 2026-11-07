"""Runner contracts: failures stay failures; setup is explicit; children are owned."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_frameworks.py"
spec = importlib.util.spec_from_file_location("framework_runner_tests", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def api_report():
    return {"cases": [{"case": f"case-{i}", "passed": True} for i in range(40)]}


def test_exit_zero_is_insufficient_for_missing_or_failed_results():
    assert runner.summarize("api", api_report(), 0, {})["status"] == "pass"
    assert runner.summarize("api", {"cases": []}, 0, {})["status"] == "fail"
    value = api_report()
    value["cases"][0]["passed"] = False
    assert runner.summarize("api", value, 0, {})["status"] == "fail"
    assert runner.summarize("api", api_report(), 1, {})["status"] == "fail"
    value = api_report()
    value["cases"][1]["case"] = value["cases"][0]["case"]
    assert runner.summarize("api", value, 0, {})["status"] == "fail"


def test_known_issue_requires_same_case_passing_in_direct_api_and_still_fails():
    label = "choices2-thinkingFalse-reasoningFalse-streamTrue"
    rows = [{"case": label, "passed": False, "errors": ["ProtocolError: Incomplete SSE response"]}]
    value = {"matrix": rows, "loops": {}}
    known = runner.summarize("litellm-sdk", value, 1, {label: True})
    assert known["status"] == "fail"
    assert known["failures"][-1]["classification"] == "known-litellm-n2-stream"
    unknown = runner.summarize("litellm-sdk", value, 1, {label: False})
    assert unknown["failures"][-1]["classification"] == "needs-investigation"


def test_framework_failure_is_not_automatically_blame_assigned():
    value = {"rows": [{"case": "single", "passed": False, "errors": ["wrong answer"]}]}
    report = runner.summarize("langgraph", value, 1, {})
    assert report["failures"][-1]["classification"] == "needs-investigation"


def test_missing_environment_never_installs_without_setup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess/installation"))
    args = SimpleNamespace(env_root=tmp_path, setup=False)
    with pytest.raises(RuntimeError, match="--setup"):
        runner.environment("langgraph", args)


def test_wrong_pin_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, "pins", lambda name: {"langgraph": "1.2.20"})
    binary = tmp_path / "langgraph/bin/python"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout='{"langgraph":"wrong"}'))
    with pytest.raises(RuntimeError, match="Wrong dependency versions"):
        runner.environment("langgraph", SimpleNamespace(env_root=tmp_path, setup=False))


def test_failed_environment_does_not_silently_skip_and_no_server_start(tmp_path, monkeypatch):
    def missing(*a):
        raise RuntimeError("missing dependency")
    monkeypatch.setattr(runner, "environment", missing)
    monkeypatch.setattr(runner, "request_models", lambda *a: {"data": [{"id": "bielik-test"}]})
    monkeypatch.setattr(runner, "run_probe", lambda name, *a: (
        runner.summarize("api", api_report(), 0, {}), api_report()))
    output = tmp_path / "report"
    assert runner.main(["--all", "--env-root", str(tmp_path), "--output-dir", str(output)]) == 2
    data = json.loads((output / "summary.json").read_text())
    assert not data["passed"]
    assert sum(p["status"] == "error" for p in data["probes"]) == 4
    with pytest.raises(FileExistsError):
        runner.main(["--all", "--output-dir", str(output)])


def test_setup_only_does_not_contact_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", lambda *a: (Path(sys.executable), {}))
    monkeypatch.setattr(runner, "request_models", lambda *a: pytest.fail("Unexpected model request"))
    assert runner.main(["--all", "--setup", "--setup-only", "--output-dir", str(tmp_path / "report")]) == 0


def test_execution_error_is_retained_and_later_framework_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", lambda *a: (Path(sys.executable), {}))
    monkeypatch.setattr(runner, "request_models", lambda *a: {"data": [{"id": "bielik-test"}]})
    called = []
    def probe(name, *args):
        called.append(name)
        if name == "api":
            raise subprocess.TimeoutExpired("probe", 1)
        return {"probe": name, "status": "pass", "passed": 32, "total": 32}, {}
    monkeypatch.setattr(runner, "run_probe", probe)
    output = tmp_path / "report"
    assert runner.main(["--framework", "langgraph", "--framework", "inspect-ai",
                        "--output-dir", str(output)]) == 2
    assert called == ["api", "langgraph", "inspect-ai"]
    assert json.loads((output / "summary.json").read_text())["probes"][0]["status"] == "error"


def test_benchmarks_are_separate_and_guide_requires_local_tokenizer(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["--all", "--benchmark", "bfcl"])
    with pytest.raises(SystemExit):
        runner.main(["--benchmark", "guidellm", "--output-dir", str(tmp_path / "report")])


def test_timeout_terminates_actual_owned_child(tmp_path):
    if sys.platform != "linux":
        pytest.skip("POSIX process groups")
    pidfile = tmp_path / "child.pid"
    program = "import os,time,pathlib; pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        runner.execute([sys.executable, "-c", program], tmp_path / "log", dict(os.environ), .5)
    pid = int(pidfile.read_text())
    assert not Path(f"/proc/{pid}").exists()


def test_interrupted_run_cannot_leave_a_green_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "environment", lambda *a: (Path(sys.executable), {}))
    monkeypatch.setattr(runner, "request_models", lambda *a: {"data": [{"id": "bielik-test"}]})
    def probe(name, *args):
        if name != "api":
            raise KeyboardInterrupt()
        return runner.summarize("api", api_report(), 0, {}), api_report()
    monkeypatch.setattr(runner, "run_probe", probe)
    output = tmp_path / "report"
    with pytest.raises(KeyboardInterrupt):
        runner.main(["--framework", "langgraph", "--output-dir", str(output)])
    value = json.loads((output / "summary.json").read_text())
    assert not value["passed"] and not value["completed"]


@pytest.mark.parametrize("code", [2, 3, -15])
def test_native_operational_exit_remains_error_even_with_complete_report(code):
    result = runner.summarize("api", api_report(), code, {})
    assert result["status"] == "error"
    assert result["exit_code"] == code
    assert result["failures"]


def test_bfcl_run_error_is_operational_error():
    data = {"rows": [{"id": "case", "thinking": False, "stream": False,
                      "correct": False, "run_error": "Timeout"}]}
    result = runner.summarize("bfcl", data, 2, {})
    assert result["status"] == "error"
    assert result["failures"][0]["errors"] == ["Timeout"]


def test_crashed_probe_preserves_native_exit_and_log_path(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runner, "execute", lambda *args: 3)
    args = SimpleNamespace(model="offline", family="v31", base_url="http://localhost:1/v1", timeout=1)
    summary, data = runner.run_probe("api", Path(sys.executable), args, tmp_path, {}, {})
    assert summary["status"] == "error" and summary["exit_code"] == 3
    assert summary["log"] == str(tmp_path / "api.log")
    assert data == {}
