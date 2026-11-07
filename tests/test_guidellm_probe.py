import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("guidellm_probe_test", ROOT / "scripts" / "probe_guidellm.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def report():
    return {"benchmarks": [{"config": {"strategy": {"type_": "concurrent", "streams": 1}}, "metrics": {"request_totals": {
        "successful": 2, "errored": 0, "incomplete": 0, "total": 2}}, "requests": {
        "successful": [{"tool_calls": [{"function": {"name": "add", "arguments": '{"a":17,"b":23}'}}]},
                       {"output": "40"}], "errored": [], "incomplete": []}}]}


def test_native_errors_cannot_be_hidden_by_cli_exit_zero():
    value = report()
    assert not probe.validate_report(value, [1], 2, True)
    value["benchmarks"][0]["metrics"]["request_totals"]["errored"] = 1
    assert probe.validate_report(value, [1], 2, True)


def test_missing_profiles_requests_tool_calls_and_followup_fail():
    assert probe.validate_report({"benchmarks": []}, [1], 2, True)
    value = report()
    value["benchmarks"][0]["requests"]["successful"].pop()
    assert probe.validate_report(value, [1], 2, True)
    assert probe.validate_report(report(), [1], 2, False)
    value = report()
    value["benchmarks"][0]["requests"]["successful"][0]["tool_calls"][0]["function"]["name"] = "shell"
    assert probe.validate_report(value, [1], 2, True)


def test_scenario_uses_local_tokenizer_fixed_result_and_finite_limits(tmp_path):
    import json
    value = probe.scenario("http://localhost:8000", "bielik", tmp_path, "v31", True, True, [1,4], 8,
                           tmp_path / "output.json")["spec"]
    assert value["tokenizer"]["load_kwargs"] == {"local_files_only": True, "trust_remote_code": False}
    assert value["data_finalizer"]["tool_call_mode"] == "client"
    assert value["constraints"] == [{"kind": "max_requests", "count": 8}, {"kind": "max_duration", "seconds": 180}]
    assert len(value["data"][0]["data"]) == 16
    for row in value["data"][0]["data"]:
        assert row["tool_response"] == '{"value":40}'
        assert json.loads(row["tools"])[0]["function"]["name"] == "add"


def test_incomplete_requests_and_wrong_arguments_fail():
    value = report()
    value["benchmarks"][0]["metrics"]["request_totals"]["incomplete"] = 1
    assert probe.validate_report(value, [1], 2, True)
    value = report()
    value["benchmarks"][0]["requests"]["successful"][0]["tool_calls"][0]["function"]["arguments"] = '{"a":18,"b":23}'
    assert probe.validate_report(value, [1], 2, True)


def test_duplicate_or_wrong_concurrency_profile_fails():
    value = report()
    value["benchmarks"] *= 2
    assert probe.validate_report(value, [1,4], 2, True)
    value = report()
    value["benchmarks"][0]["config"]["strategy"]["type_"] = "synchronous"
    assert probe.validate_report(value, [1], 2, True)
