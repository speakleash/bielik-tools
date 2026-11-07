#!/usr/bin/env python3
"""Bounded native GuideLLM load probes; fixed client-side tool results only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from testing_api import ADD, thinking

VERSION = "0.8.1"


def scenario(target, model, tokenizer, family, enabled, tools, streams, requests, output):
    row = {"text": "Ile to17 razy23? Podaj wyłącznie liczbę."}
    if tools:
        row = {"text": "Wywołaj add z a=17 i b=23. Podaj wynik zwrócony przez narzędzie.",
               "tools": json.dumps([ADD], ensure_ascii=False), "tool_choice": "required",
               "tool_response": '{"value":40}'}
    # Native max_requests counts individual model turns, including tool followup.
    # Supply enough static rows for every profile; no generated text becomes code.
    return {"spec": {
        "backend": {"kind": "openai_http", "target": target, "model": model,
            "stream": True, "verify": True, "timeout": 90, "max_tokens": 512,
            "tool_call_missing_behavior": "error_stop", "multiturn_reasoning": False,
            "extras": {"body": {"temperature": 0, "seed": 1701,
                "chat_template_kwargs": thinking(family, enabled),
                **({"thinking_token_budget": 64} if enabled else {})}}},
        "profile": {"kind": "concurrent", "streams": streams, "warmup": 0, "cooldown": 0},
        "constraints": [{"kind": "max_requests", "count": requests},
                        {"kind": "max_duration", "seconds": 180}],
        "tokenizer": {"kind": "hf_auto", "model": str(tokenizer),
            "load_kwargs": {"local_files_only": True, "trust_remote_code": False}},
        "data": [{"kind": "in_memory_dict_list", "data": [dict(row) for _ in range(requests * 2)]}],
        "data_loader": {"kind": "pytorch", "samples": requests * 2, "num_workers": 1},
        "data_finalizer": {"kind": "generative", "tool_call_mode": "client"},
        "outputs": [{"kind": "json", "path": str(output)}]}}


def validate_report(report, streams, requests, tools):
    errors = []
    benchmarks = report.get("benchmarks", [])
    if len(benchmarks) != len(streams):
        errors.append("Missing native benchmark profiles")
    for index, benchmark in enumerate(benchmarks):
        strategy = benchmark.get("config", {}).get("strategy", {})
        if (index >= len(streams) or strategy.get("type_") != "concurrent"
                or strategy.get("streams") != streams[index]):
            errors.append(f"Profile {index}: wrong concurrency strategy")
        totals = benchmark.get("metrics", {}).get("request_totals", {})
        if (totals.get("successful") != requests or totals.get("errored") != 0
                or totals.get("incomplete") != 0 or totals.get("total") != requests):
            errors.append(f"Profile {index}: unsuccessful/incomplete/request-count mismatch: {totals}")
        recorded = benchmark.get("requests", {})
        if recorded.get("errored") or recorded.get("incomplete"):
            errors.append(f"Profile {index}: native request errors")
        successful = recorded.get("successful", [])
        if len(successful) != requests:
            errors.append(f"Profile {index}: missing recorded requests")
        if tools:
            calls = [call for r in successful for call in r.get("tool_calls") or []]
            if not calls or not any(r.get("output") for r in successful):
                errors.append(f"Profile {index}: missing tool call or followup output")
            for call in calls:
                try:
                    function = call["function"]
                    if function["name"] != "add" or json.loads(function["arguments"]) != {"a": 17, "b": 23}:
                        errors.append(f"Profile {index}: unexpected generated tool arguments")
                except (KeyError, TypeError, ValueError):
                    errors.append(f"Profile {index}: malformed tool call")
        elif any(r.get("tool_calls") for r in successful):
            errors.append(f"Profile {index}: unexpected tool calls")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, type=Path, help="existing local tokenizer directory")
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--streams", default="1,4")
    ap.add_argument("--requests", default=8, type=int, help="model turns per profile, not conversations")
    ap.add_argument("--output-dir", required=True, type=Path, help="new directory, never overwritten")
    args = ap.parse_args(argv)
    streams = [int(s) for s in args.streams.split(",")]
    if (not streams or any(s < 1 for s in streams) or len(set(streams)) != len(streams)
            or args.requests < 2 * max(streams) or not args.tokenizer.is_dir()):
        ap.error("Use a local tokenizer and requests >= 2*max(streams), positive distinct streams")
    if importlib.metadata.version("guidellm") != VERSION:
        ap.error(f"Install pinned guidellm=={VERSION} in an isolated environment")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for tools in (False, True):
        for enabled in (False, True):
            label = f"{'tool' if tools else 'plain'}-{'medium' if enabled else 'none'}"
            output = args.output_dir / (label + ".json")
            config = args.output_dir / (label + "-scenario.json")
            config.write_text(json.dumps(scenario(args.target, args.model, args.tokenizer, args.family,
                enabled, tools, streams, args.requests, output), ensure_ascii=False, indent=2))
            env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", DO_NOT_TRACK="1")
            errors = []
            with (args.output_dir / (label + ".log")).open("w") as log:
                try:
                    result = subprocess.run([sys.executable, "-m", "guidellm", "run", "--config", str(config),
                        "--disable-console"], env=env, stdout=log, stderr=subprocess.STDOUT,
                        timeout=240 * len(streams), check=False)
                    if result.returncode:
                        errors.append(f"Native CLI exit {result.returncode}")
                    # GuideLLM can exit0 even with failed requests. Inspect its report.
                    errors += validate_report(json.loads(output.read_text()), streams, args.requests, tools)
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    errors.append(f"{type(error).__name__}: {error}")
            rows.append({"case": label, "passed": not errors, "errors": errors, "report": str(output)})
            print(f"{label}: {errors or 'PASS'}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"guidellm_version": VERSION,
        "requests_are_model_turns": True, "rows": rows}, indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
