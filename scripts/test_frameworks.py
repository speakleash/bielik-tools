#!/usr/bin/env python3
"""Run optional framework probes with isolated environments and one summary."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = ("litellm", "langgraph", "pydantic-ai", "inspect-ai")
BENCHMARKS = ("bfcl", "guidellm")
SCRIPTS = {"langgraph": "probe_langgraph.py", "pydantic-ai": "probe_pydantic_ai.py",
           "inspect-ai": "probe_inspect_ai.py", "bfcl": "evaluate_bfcl_ast.py", "guidellm": "probe_guidellm.py"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def pins(name):
    lines = (ROOT / "tests" / "integrations" / name / "requirements.txt").read_text().splitlines()
    return {pkg.split("[")[0]: version for line in map(str.strip, lines) if line and not line.startswith("#")
            for pkg, version in [line.split("==")]}


def environment(name, args):
    """Installation happens only with --setup; a missing/wrong pin never skips a test."""
    folder = args.env_root / name
    python = folder / "bin" / "python"
    if not python.is_file() and not args.setup:
        raise RuntimeError(f"Missing environment {name}; run again with --setup")
    expected = pins(name)
    if args.setup:
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("--setup requires uv; install it explicitly or provide prepared environments")
        if not python.exists():
            subprocess.run([uv, "venv", "--python", args.python, "--no-python-downloads", str(folder)], check=True)
        command = [uv, "pip", "install", "--python", str(python), "--only-binary=:all:"]
        if name == "bfcl":
            command += ["--no-deps"]  # audited AST path only; never load vendor runners
        if name == "guidellm":
            command += ["--torch-backend", "cpu"]
        subprocess.run(command + ["-r", str(ROOT / "tests" / "integrations" / name / "requirements.txt")], check=True)
    if not python.is_file():
        raise RuntimeError(f"Missing environment {name}; run again with --setup")
    code = ("import importlib.metadata as m,json; "
            "print(json.dumps({p:m.version(p) for p in " + repr(list(expected)) + "}))")
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=30, check=True)
    versions = json.loads(result.stdout)
    if versions != expected:
        raise RuntimeError(f"Wrong dependency versions for {name}: {versions}; run --setup")
    return python, versions


def stop_process(process):
    """Each child has its own process group, including GuideLLM's workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A leader can exit while its workers remain alive in the same group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def execute(command, log, env, timeout):
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            stop_process(process)


def request_models(base_url, key, timeout=5):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with urllib.request.urlopen(urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers),
                                timeout=timeout) as response:
        return json.load(response)


@contextmanager
def proxy(python, args, directory, env):
    """Own a temporary loopback proxy; credentials are environment-only."""
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    key = "sk-" + secrets.token_hex(24)
    config = {"model_list": [{"model_name": args.model, "litellm_params": {
        "model": "hosted_vllm/" + args.model, "api_base": args.base_url,
        "api_key": "os.environ/BIELIK_UPSTREAM_API_KEY"}}],
        "litellm_settings": {"drop_params": False, "telemetry": False},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "disable_spend_logs": True}}
    path = directory / "proxy-config.json"
    path.write_text(json.dumps(config, indent=2))
    child_env = dict(env, LITELLM_MASTER_KEY=key,
                     BIELIK_UPSTREAM_API_KEY=env.get("BIELIK_TEST_API_KEY", "dummy"))
    url = f"http://127.0.0.1:{port}/v1"
    with (directory / "proxy.log").open("w") as log:
        process = subprocess.Popen([str(python.parent / "litellm"), "--config", str(path),
            "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=child_env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Temporary LiteLLM proxy exited; see proxy.log")
                try:
                    models = request_models(url, key, timeout=1)
                    if args.model in {m["id"] for m in models["data"]}:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(.2)
            else:
                raise RuntimeError("Temporary LiteLLM proxy startup timed out; see proxy.log")
            yield url, dict(env, BIELIK_TEST_API_KEY=key)
        finally:
            stop_process(process)


def rows_from(name, data):
    if name == "api":
        return data["cases"]
    if name.startswith("litellm-"):
        return data["matrix"] + [{"case": label, **row} for label, row in data["loops"].items()]
    if name == "bfcl":
        return [{"case": f"{r['id']}/{r['thinking']}/{r['stream']}", "passed": r["correct"] and not r.get("run_error"),
                 "errors": [r["run_error"]] if r.get("run_error") else ([] if r["correct"] else ["AST score incorrect"])}
                for r in data["rows"]]
    return data["rows"]


def summarize(name, data, code, baseline):
    rows = rows_from(name, data)
    expected = {"api": 40, "litellm-sdk": 44, "litellm-proxy": 44,
                "langgraph": 32, "pydantic-ai": 36, "inspect-ai": 32, "guidellm": 4}
    failures = []
    identities = [(r.get("case"), r.get("thinking"), r.get("stream")) for r in rows]
    if ((name in expected and len(rows) != expected[name])
            or len(set(identities)) != len(identities)):
        failures.append({"case": "report-contract", "errors": ["Missing or duplicate scenario results"],
                         "classification": "invalid-report"})
    if data.get("passed") is False and not any(r.get("passed") is not True for r in rows):
        failures.append({"case": "probe-contract", "errors": ["Native probe reported failure"],
                         "classification": "native-probe-failure"})
    for row in rows:
        if row.get("passed") is not True:
            label = row.get("case", "unknown")
            errors = row.get("errors") or ["Probe assertion failed; see native report"]
            known = (name.startswith("litellm-") and label.startswith("choices2-") and label.endswith("streamTrue")
                     and baseline.get(label) is True and any("Incomplete SSE response" in e for e in errors))
            failures.append({"case": label, "errors": errors,
                             "classification": "known-litellm-n2-stream" if known else "needs-investigation"})
    if code != 0 and not failures:
        failures.append({"case": "process-exit", "errors": [f"Native process exited {code}"],
                         "classification": "native-process-error"})
    status = "error" if code not in (0, 1) else "pass" if rows and not failures and code == 0 else "fail"
    return {"probe": name, "status": status,
            "passed": sum(r.get("passed") is True for r in rows), "total": len(rows), "failures": failures,
            "exit_code": code}


def run_probe(name, python, args, directory, env, baseline, url=None):
    output = directory / (name + ".json")
    log = directory / (name + ".log")
    common = ["--model", args.model, "--family", args.family]
    if name == "api":
        command = [str(python), str(ROOT / "tests" / "live_api_matrix.py")]
    elif name.startswith("litellm-"):
        command = [str(python), str(ROOT / "scripts" / "probe_litellm.py"), "--backend", name.split("-")[1]]
    else:
        command = [str(python), str(ROOT / "scripts" / SCRIPTS[name])]
    command += common
    if name == "guidellm":
        command += ["--target", args.base_url.removesuffix("/v1"), "--tokenizer", str(args.tokenizer),
                    "--output-dir", str(directory / "guidellm-native")]
        output = directory / "guidellm-native" / "summary.json"
    else:
        command += ["--base-url", url or args.base_url, "--output", str(output)]
    if name == "bfcl":
        command += ["--limit", str(args.bfcl_limit), "--min-score", "1"]
    code = execute(command, log, env, args.timeout)
    try:
        data = read_json(output)
    except (OSError, ValueError) as error:
        return {"probe": name, "status": "error", "exit_code": code, "report": str(output),
                "log": str(log), "error": f"Native report unavailable: {error}; see {log}"}, {}
    summary = summarize(name, data, code, baseline)
    summary.update(report=str(output), log=str(log))
    return summary, data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="all four frameworks, including LiteLLM SDK and proxy")
    selection.add_argument("--framework", choices=FRAMEWORKS, action="append")
    selection.add_argument("--benchmark", choices=BENCHMARKS, action="append", help="separate optional longer probes")
    ap.add_argument("--model", default="bielik-test")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--env-root", type=Path, default=ROOT / ".framework-envs")
    ap.add_argument("--setup", action="store_true", help="explicitly create/update the selected pinned environments")
    ap.add_argument("--setup-only", action="store_true", help="prepare dependencies without contacting a model")
    ap.add_argument("--python", default="3.12", help="installed Python for uv venv; never downloaded automatically")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--tokenizer", type=Path, help="existing local tokenizer, required only for GuideLLM")
    ap.add_argument("--bfcl-limit", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1800, help="maximum seconds per probe")
    args = ap.parse_args(argv)
    args.env_root = args.env_root.resolve()
    if args.tokenizer:
        args.tokenizer = args.tokenizer.resolve()
    selected = list(FRAMEWORKS if args.all else args.framework or args.benchmark)
    if len(set(selected)) != len(selected) or args.timeout < 1 or args.bfcl_limit < 1:
        ap.error("Selections must be distinct; timeout and BFCL limit must be positive")
    if args.setup_only and not args.setup:
        ap.error("--setup-only requires --setup")
    if "guidellm" in selected and (not args.tokenizer or not args.tokenizer.is_dir()) and not args.setup_only:
        ap.error("GuideLLM needs --tokenizer pointing to an existing local directory")
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1") + "/v1"
    directory = (args.output_dir or ROOT / ".framework-results" /
                 (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3))).resolve()
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    env = dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True", LITELLM_TELEMETRY="False", DO_NOT_TRACK="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYDANTIC_AI_NO_BANNER="1")
    report = {"model": args.model, "base_url": args.base_url, "family": args.family,
              "started_at": datetime.now(timezone.utc).isoformat(), "versions": {}, "probes": [], "completed": False,
              "attribution": "A failed framework case alone cannot distinguish adapter defects from model variability. "
                             "Only the matched LiteLLM n=2/direct-API evidence receives a known-issue label."}
    def save():
        report["passed"] = report["completed"] and bool(report["probes"]) and all(p["status"] == "pass" for p in report["probes"])
        (directory / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save()
    environments = {}
    for name in selected:
        try:
            python, versions = environment(name, args)
            environments[name] = python
            report["versions"][name] = versions
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            report["probes"].append({"probe": name, "status": "error", "error": str(error)})
        save()
    if args.setup_only:
        report["setup_only"] = True
        report["completed"] = True
        report["passed"] = len(environments) == len(selected)
        (directory / "summary.json").write_text(json.dumps(report, indent=2))
        print(f"Environments ready: {len(environments)}/{len(selected)}; {directory / 'summary.json'}")
        return 0 if report["passed"] else 2
    try:
        models = request_models(args.base_url, env.get("BIELIK_TEST_API_KEY"))
        if args.model not in {m["id"] for m in models["data"]}:
            raise ValueError("Requested model is absent from /v1/models")
    except (OSError, ValueError, KeyError) as error:
        report["probes"].append({"probe": "server", "status": "error", "error": str(error)})
        save()
        print(f"Server unavailable; see {directory / 'summary.json'}")
        return 2
    baseline = {}
    routes = [("api", Path(sys.executable))]
    for name, python in environments.items():
        routes += [("litellm-sdk", python), ("litellm-proxy", python)] if name == "litellm" else [(name, python)]
    try:
        for name, python in routes:
            try:
                if name == "litellm-proxy":
                    with proxy(python, args, directory, env) as (url, proxy_env):
                        result, data = run_probe(name, python, args, directory, proxy_env, baseline, url)
                else:
                    result, data = run_probe(name, python, args, directory, env, baseline)
                if name == "api" and "cases" in data:
                    baseline = {r["case"]: r["passed"] for r in data["cases"]}
                report["probes"].append(result)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
                report["probes"].append({"probe": name, "status": "error", "error": str(error),
                    "log": str(directory / ("proxy.log" if name == "litellm-proxy" else name + ".log"))})
            save()
            result = report["probes"][-1]
            print(f"{name}: {result['status'].upper()} {result.get('passed', 0)}/{result.get('total', 0)}", flush=True)
            for failure in result.get("failures", []):
                print(f"  {failure['case']} [{failure['classification']}]: " + "; ".join(failure["errors"]), flush=True)
            if result.get("error"):
                print("  " + result["error"], flush=True)
        report["completed"] = True
    finally:
        save()
    print(f"Report: {directory / 'summary.json'}")
    return 2 if any(p["status"] == "error" for p in report["probes"]) else 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
