#!/usr/bin/env python3
"""Small Polish regression set: quality scores separate from transport errors.

This is a reproducible smoke benchmark, not a general model leaderboard.
No model-generated tool is executed. Default mode reports scores; only an
explicit --min-score turns semantic accuracy into a gate. Run errors always
fail, independently of the quality threshold.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from testing_api import ADD, WEATHER, ANSWER_SCHEMA, chat, thinking

CASES = [
    {"id": "multiplication", "prompt": "Ile to 17 razy 23? Podaj wyłącznie liczbę.", "text": "391"},
    {"id": "division", "prompt": "Ile to 144 podzielone przez 12? Podaj wyłącznie liczbę.", "text": "12"},
    {"id": "polish_extraction", "prompt": "Ala mieszka w Gdańsku. Bartek mieszka w Łodzi. W którym mieście mieszka Bartek? Podaj wyłącznie nazwę miasta w mianowniku.", "text": "Łódź"},
    {"id": "instruction", "prompt": "Odpowiedz dokładnie jednym słowem: bursztyn", "text": "bursztyn"},
    {"id": "schema", "prompt": "Zwróć wynik 17 razy 23 w polu answer.", "schema": True, "json": {"answer": 391}},
    {"id": "add_tool", "prompt": "Wywołaj add z a=17 i b=23.", "tools": [ADD, WEATHER], "calls": [{"name": "add", "arguments": {"a": 17, "b": 23}}]},
    {"id": "weather_tool", "prompt": "Sprawdź pogodę w Krakowie narzędziem weather.", "tools": [ADD, WEATHER], "calls": [{"name": "weather", "arguments": {"city": "Kraków"}}]},
    {"id": "no_tool", "prompt": "Powiedz krótko cześć. Nie potrzebujesz żadnych narzędzi.", "tools": [ADD, WEATHER], "calls": [], "nonempty": True},
]


def score(choice, case):
    reasons = []
    calls = [{"name": c["name"], "arguments": c["arguments"]} for c in choice["tool_calls"]]
    if calls != case.get("calls", []):
        reasons.append("tool selection or arguments")
    if "text" in case:
        # Permit surrounding whitespace and a single terminal sentence mark,
        # but do not accept a correct number buried in an incorrect answer.
        if not re.fullmatch(re.escape(case["text"]) + r"[.!]?", choice["content"].strip(), flags=re.IGNORECASE):
            reasons.append("exact answer")
    if "json" in case:
        try:
            actual = json.loads(choice["content"])
        except ValueError:
            actual = None
        # JSON equality must not conflate booleans with integers.
        if (actual != case["json"] or not isinstance(actual, dict)
                or type(actual.get("answer")) is not int):
            reasons.append("structured answer")
    if case.get("nonempty") and not choice["content"].strip():
        reasons.append("empty answer")
    expected_finish = "tool_calls" if case.get("calls") else "stop"
    if choice["finish_reason"] != expected_finish:
        reasons.append("termination")
    if any(tag in choice["content"] for tag in ("<think>", "</think>", "<tool_call>", "</tool_call>")):
        reasons.append("markup leak")
    return reasons


def wilson(successes, total):
    if not total:
        raise ValueError("Cannot summarize an empty evaluation")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z*z/total
    center = (proportion + z*z/(2*total)) / denominator
    radius = z * math.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["effort"] + "/" + row["sampling"]].append(row)
    return {key: {"correct": sum(r["correct"] for r in values), "total": len(values),
                  "accuracy": sum(r["correct"] for r in values)/len(values),
                  "wilson95": wilson(sum(r["correct"] for r in values), len(values))}
            for key, values in groups.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--samples", type=int, default=3, help="sampled completions per case, plus one greedy")
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--min-score", type=float, help="optional accuracy gate, applied separately to each group")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.samples < 0 or (args.min_score is not None and not 0 <= args.min_score <= 1):
        ap.error("samples must be non-negative and min-score in [0,1]")
    rows = []
    for enabled in (False, True):
        for case in CASES:
            for sample in range(args.samples + 1):
                payload = {"model": args.model, "messages": [{"role": "user", "content": case["prompt"]}],
                           "max_tokens": 512, "temperature": 0 if sample == 0 else 0.7,
                           "seed": args.seed + sample, "chat_template_kwargs": thinking(args.family, enabled),
                           **({"thinking_token_budget": 128} if enabled else {})}
                if case.get("tools"):
                    payload["tools"] = case["tools"]
                if case.get("schema"):
                    payload["response_format"] = ANSWER_SCHEMA
                row = {"case": case["id"], "effort": "medium" if enabled else "none",
                       "sampling": "greedy" if sample == 0 else "sampled", "sample": sample,
                       "request": payload, "correct": False}
                try:
                    response = chat(args.base_url, payload)
                    row["response"] = response
                    row["semantic_errors"] = score(response["choices"][0], case)
                    row["correct"] = not row["semantic_errors"]
                except Exception as error:
                    row["run_error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                print(f"{row['effort']}/{row['sampling']} {case['id']}: {row['correct']}", flush=True)
    summary = summarize(rows)
    run_errors = sum("run_error" in r for r in rows)
    quality_passed = (all(g["accuracy"] >= args.min_score for g in summary.values())
                      if args.min_score is not None else None)
    report = {"model": args.model, "family": args.family, "seed": args.seed,
              "case_sha256": hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
              "quality_gate": args.min_score, "quality_passed": quality_passed,
              "run_errors": run_errors, "summary": summary, "rows": rows}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    print("QUALITY REPORT ONLY" if args.min_score is None else f"QUALITY GATE: {quality_passed}")
    return 2 if run_errors else 1 if quality_passed is False else 0


if __name__ == "__main__":
    sys.exit(main())
