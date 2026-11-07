#!/usr/bin/env python3
"""Boundary API matrix. Requires a local live server; never collected by pytest."""
import argparse
import json
from pathlib import Path
import sys
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from testing_api import ADD, WEATHER, ANSWER_SCHEMA, chat, check_choice, named, thinking


def cases(family):
    plain = [{"role": "user", "content": "Ile to 17 razy 23? Odpowiedz krótko."}]
    tool_prompt = [{"role": "user", "content": "Wywołaj add z a=17 i b=23."}]
    for enabled in (False, True):
        flags = {"chat_template_kwargs": thinking(family, enabled)}
        budget = {"thinking_token_budget": 64} if enabled else {}
        for stream in (False, True):
            suffix = f"{'on' if enabled else 'off'}-{'stream' if stream else 'full'}"
            base = {"stream": stream, **flags, **budget}
            yield "plain-" + suffix, {**base, "messages": plain}, {"tools": [], "reasoning": enabled, "content": True, "finish": ["stop"]}
            for mode, choice in (("auto", "auto"), ("none", "none"), ("required", "required"), ("named", named("add"))):
                yield mode + "-" + suffix, {**base, "messages": tool_prompt, "tools": [ADD, WEATHER], "tool_choice": choice}, {
                    "tools": [] if mode == "none" else ["add"], "finish": ["stop"] if mode == "none" else ["tool_calls"],
                    "reasoning": enabled, "arguments": [] if mode == "none" else [{"a": 17, "b": 23}]}
            yield "schema-" + suffix, {**base, "messages": plain, "response_format": ANSWER_SCHEMA}, {
                "schema": True, "tools": [], "reasoning": enabled, "finish": ["stop"]}
    for stream in (False, True):
        for enabled in (False, True):
            for include in (False, True):
                label = f"choices2-thinking{enabled}-reasoning{include}-stream{stream}"
                yield label, {"messages": plain, "stream": stream, "n": 2, "temperature": 0.7, "include_reasoning": include,
                              "chat_template_kwargs": thinking(family, enabled),
                              **({"thinking_token_budget": 64} if enabled else {})}, {
                    "tools": [], "content": True, "reasoning": enabled and include, "finish": ["stop"]}
    for stream in (False, True):
        for enabled in (False, True):
            yield f"toplevel-thinking{enabled}-stream{stream}", {
                "messages": plain, "stream": stream,
                "reasoning_effort": "medium" if enabled else "none",
                **({"thinking_token_budget": 64} if enabled else {})}, {
                    "tools": [], "content": True, "reasoning": enabled, "finish": ["stop"]}
        yield f"short-required-stream{stream}", {"messages": tool_prompt, "stream": stream,
            "tools": [ADD], "tool_choice": "required", "max_tokens": 1,
            "chat_template_kwargs": thinking(family, False)}, {
                "tools": [], "finish": ["length"], "allow_markup": True}
            # A one-token truncation may expose an unfinished tag as data.
            # It must never become an executable tool call.
    for skip in (False, True):
        yield f"skip-special-{skip}", {"messages": tool_prompt, "tools": [ADD],
            "tool_choice": "required", "skip_special_tokens": skip,
            "chat_template_kwargs": thinking(family, True), "thinking_token_budget": 64}, {
                "tools": ["add"], "arguments": [{"a": 17, "b": 23}], "reasoning": True, "finish": ["tool_calls"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--only", help="comma-separated exact case names")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    selected = list(cases(args.family))
    if args.only:
        names = args.only.split(",")
        unknown = set(names) - {c[0] for c in selected}
        if unknown:
            ap.error(f"Unknown cases: {sorted(unknown)}")
        selected = [c for c in selected if c[0] in names]
    rows = []
    for name, payload, expected in selected:
        body = {"model": args.model, "max_tokens": 512, "temperature": 0, "seed": 1701, **payload}
        try:
            result = chat(args.base_url, body)
            errors = [f"choice {c['index']}: {e}" for c in result["choices"] for e in check_choice(c, expected)]
            row = {"case": name, "passed": not errors, "errors": errors, "request": body, "response": result}
        except Exception as error:
            detail = error.read().decode("utf-8", errors="replace") if isinstance(error, urllib.error.HTTPError) else ""
            row = {"case": name, "passed": False, "errors": [f"{type(error).__name__}: {error} {detail}"], "request": body}
        rows.append(row)
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}: {row['errors']}", flush=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"cases": rows, "passed": sum(r["passed"] for r in rows), "total": len(rows)}, output, ensure_ascii=False, indent=2)
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
