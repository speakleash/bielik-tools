#!/usr/bin/env python3
"""Exercise LiteLLM SDK or a locally configured LiteLLM proxy.

Uses the same40 API cases and safe tool-loop ledger as the direct client.
Install the pinned optional requirements; default pytest does not need them.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from testing_api import chat, check_choice, normalize_message, read_stream

ROOT = Path(__file__).resolve().parents[1]


def load_probe(name):
    spec = importlib.util.spec_from_file_location("litellm_" + name, ROOT / "tests" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sdk_chat(base_url, body):
    # LiteLLM otherwise fetches its cost map on import. Cost/pricing are not
    # needed to verify a local model; use the bundled map and disable telemetry.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    import litellm

    start = time.perf_counter()
    standard = dict(body)
    model = standard.pop("model")
    extra = {key: standard.pop(key) for key in (
        "chat_template_kwargs", "thinking_token_budget", "skip_special_tokens", "include_reasoning")
             if key in standard}
    response = litellm.completion(model="hosted_vllm/" + model, api_base=base_url,
        api_key=os.environ.get("BIELIK_TEST_API_KEY", "dummy"), extra_body=extra, **standard)
    if body.get("stream"):
        setup_time = time.perf_counter() - start
        raw_chunks = []
        def events():
            for chunk in response:
                data = chunk.model_dump()
                raw_chunks.append(data)
                yield ("data: " + json.dumps(data) + "\n").encode()
            # The SDK consumes the original wire terminator; reaching the end
            # of its iterator cleanly is the corresponding library boundary.
            yield b"data: [DONE]\n"
        try:
            result = read_stream(events())
        except Exception as error:
            error.sdk_chunks = raw_chunks
            raise
        if result["first_output_s"] is not None:
            result["first_output_s"] += setup_time
    else:
        raw = response.model_dump()
        result = {"choices": [normalize_message(c) for c in raw["choices"]], "raw": raw}
    indices = [c["index"] for c in result["choices"]]
    if (len(indices) != body.get("n", 1) or any(type(i) is not int for i in indices)
            or set(indices) != set(range(body.get("n", 1)))):
        raise ValueError("LiteLLM returned invalid choice identities")
    result["elapsed_s"] = time.perf_counter() - start
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["sdk", "proxy"], required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", required=True, help="vLLM URL for SDK; LiteLLM URL for proxy")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--only", help="comma-separated exact matrix case names; loops still run")
    args = ap.parse_args(argv)
    if args.backend == "proxy" and not os.environ.get("BIELIK_TEST_API_KEY"):
        ap.error("Set BIELIK_TEST_API_KEY to the proxy's ephemeral LITELLM_MASTER_KEY")
    matrix, loops = load_probe("live_api_matrix"), load_probe("live_multiturn_isolation")
    client = sdk_chat if args.backend == "sdk" else chat
    cases = list(matrix.cases(args.family))
    if args.only:
        names = set(args.only.split(","))
        unknown = names - {c[0] for c in cases}
        if unknown:
            ap.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c[0] in names]
    rows = []
    for name, payload, expected in cases:
        body = {"model": args.model, "max_tokens": 512, "temperature": 0, "seed": 1701, **payload}
        row = {"case": name, "request": body, "passed": False}
        try:
            result = client(args.base_url, body)
            errors = [f"choice {c['index']}: {e}" for c in result["choices"] for e in check_choice(c, expected)]
            row.update(passed=not errors, errors=errors, response=result)
        except Exception as error:
            row["errors"] = [f"{type(error).__name__}: {error}"]
            if hasattr(error, "sdk_chunks"):
                row["sdk_chunks"] = error.sdk_chunks
        rows.append(row)
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}: {row['errors']}", flush=True)
    loop_rows = {}
    for stream in (False, True):
        for enabled in (False, True):
            label = f"retry-thinking{enabled}-stream{stream}"
            loop_rows[label] = loops.conversation(args.base_url, args.model,
                "Sprawdź weather dla Krakowa. Jeśli retryable=true, ponów dokładnie raz. Podaj temperaturę.",
                family=args.family, enabled=enabled, stream=stream, fail_first_weather=True,
                expected_counts={"weather": 2}, expected_arguments={"weather": {"city": "Kraków"}},
                answers=(21,), client=client)
            print(f"[{'PASS' if loop_rows[label]['passed'] else 'FAIL'}] {label}: {loop_rows[label]['errors']}", flush=True)
    passed = bool(rows) and all(r["passed"] for r in rows) and all(r["passed"] for r in loop_rows.values())
    report = {"backend": args.backend, "client_litellm_version": importlib.metadata.version("litellm"),
              "passed": passed, "matrix": rows, "loops": loop_rows}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
