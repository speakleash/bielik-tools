#!/usr/bin/env python3
"""Measure user-visible latency and throughput for fixed local API workloads."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

from testing_api import ADD, chat, check_choice, thinking


def percentile(values, p):
    if not values or not 0 <= p <= 1:
        raise ValueError("Need observations and percentile in [0,1]")
    values = sorted(values)
    position = (len(values)-1)*p
    low = int(position)
    high = min(low+1, len(values)-1)
    return values[low] + (values[high]-values[low])*(position-low)


def usage_of(result):
    raw = result["raw"]
    usage = (raw.get("usage") if isinstance(raw, dict) else
             next((chunk["usage"] for chunk in reversed(raw) if chunk.get("usage")), None))
    if not usage or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] < 0:
        raise ValueError("Server must return completion token usage")
    return usage


def manifest(path):
    data = json.loads(path.read_text())
    required = ("model_revision", "vllm_version", "parser_commit", "template_sha256")
    if any(not isinstance(data.get(key), str) or not data[key].strip() for key in required):
        raise ValueError(f"Manifest must pin {required}")
    digest = data["template_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise ValueError("template_sha256 must be a SHA256 hex digest")
    return data


def workload(model, family, kind):
    enabled = kind == "reasoning"
    body = {"model": model, "messages": [{"role": "user", "content": "Ile to 17 razy 23? Podaj krótko wynik."}],
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": 384, "chat_template_kwargs": thinking(family, enabled)}
    expected = {"tools": [], "reasoning": enabled, "content": True, "finish": ["stop"]}
    if enabled:
        body["thinking_token_budget"] = 64
    if kind == "tool":
        body.update(messages=[{"role": "user", "content": "Wywołaj add z a=17 i b=23."}],
                    tools=[ADD], tool_choice="required")
        expected = {"tools": ["add"], "arguments": [{"a": 17, "b": 23}], "finish": ["tool_calls"]}
    return body, expected


def run(base_url, model, family, levels, repeats, warmup=2):
    if not levels or len(set(levels)) != len(levels) or any(type(c) is not int or c < 1 for c in levels):
        raise ValueError("Concurrency levels must be distinct positive integers")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    for _ in range(warmup):
        for kind in ("plain", "reasoning", "tool"):
            body, expected = workload(model, family, kind)
            result = chat(base_url, body)
            if check_choice(result["choices"][0], expected):
                raise ValueError(f"Warmup {kind} returned an invalid response")
    reports = []
    for concurrency in levels:
        def request(kind):
            body, expected = workload(model, family, kind)
            row = {"workload": kind, "request": body, "passed": False}
            try:
                result = chat(base_url, body)
                row.update(response=result, usage=usage_of(result),
                           errors=check_choice(result["choices"][0], expected))
                row["passed"] = not row["errors"]
            except Exception as error:
                row["errors"] = [f"{type(error).__name__}: {error}"]
            return row
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(request, [kind for _ in range(repeats) for kind in ("plain", "reasoning", "tool")]))
        elapsed = time.perf_counter() - start
        groups = defaultdict(list)
        for row in rows:
            groups[row["workload"]].append(row)
        summary = {}
        for kind, group in groups.items():
            passed = [r for r in group if r["passed"]]
            latency = [r["response"]["elapsed_s"] for r in passed]
            first = [r["response"]["first_output_s"] for r in passed
                     if r["response"]["first_output_s"] is not None]
            summary[kind] = {"completed": len(passed), "errors": len(group)-len(passed),
                "latency_p50_s": percentile(latency, .5) if latency else None,
                "latency_p95_s": percentile(latency, .95) if latency else None,
                "first_output_p50_s": percentile(first, .5) if first else None,
                "mean_output_tokens": statistics.mean(r["usage"]["completion_tokens"] for r in passed) if passed else None}
        reports.append({"concurrency": concurrency, "wall_s": elapsed, "rows": rows, "summary": summary,
                        "successful_requests_per_s": sum(r["passed"] for r in rows)/elapsed,
                        "successful_output_tokens_per_s": sum(r.get("usage", {}).get("completion_tokens", 0)
                            for r in rows if r["passed"])/elapsed})
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--repeats", type=int, default=5, help="requests per workload per concurrency")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    metadata = manifest(args.manifest)
    levels = [int(c) for c in args.concurrency.split(",")]
    if any(c > 64 for c in levels):
        ap.error("This smoke benchmark caps concurrency at64")
    try:
        runs = run(args.base_url, args.model, args.family, levels, args.repeats, args.warmup)
    except Exception as error:
        runs = []
        metadata["run_error"] = f"{type(error).__name__}: {error}"
    report = {"manifest": metadata, "model": args.model, "family": args.family,
              "python": sys.version, "warmup_requests": args.warmup * 3, "runs": runs}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    for entry in runs:
        print(entry["concurrency"], json.dumps(entry["summary"]))
    return 0 if runs and all(r["passed"] for run in runs for r in run["rows"]) else 2


if __name__ == "__main__":
    sys.exit(main())
