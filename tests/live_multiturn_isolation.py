#!/usr/bin/env python3
"""Real model tool loops and concurrent per-request state isolation."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from synthetic_tools import SyntheticTools
from testing_api import ADD, WEATHER, chat, check_choice, thinking


def conversation(base_url, model, prompt, *, family="v31", enabled=False, stream=False,
                 expected_counts=None, expected_arguments=None, answers=(),
                 fail_first_weather=False, max_turns=5):
    tools = SyntheticTools(fail_first_weather)
    messages = [{"role": "user", "content": prompt}]
    turns, errors = [], []
    try:
        for _ in range(max_turns):
            body = {"model": model, "messages": list(messages), "tools": [ADD, WEATHER],
                    "temperature": 0, "max_tokens": 512, "stream": stream,
                    "chat_template_kwargs": thinking(family, enabled),
                    **({"thinking_token_budget": 64} if enabled else {})}
            result = chat(base_url, body)
            choice = result["choices"][0]
            turns.append({"request": body, "response": result})
            errors += check_choice(choice, {} if enabled else {"reasoning": False})
            if not choice["tool_calls"]:
                if choice["finish_reason"] != "stop" or not choice["content"].strip():
                    errors.append("Missing completed final answer")
                if any(not re.search(r"(?<![\d-])" + re.escape(str(value)) + r"(?!\d|[.,]\d)",
                                     choice["content"]) for value in answers):
                    errors.append("Final answer does not contain the synthetic results")
                break
            if choice["finish_reason"] != "tool_calls":
                errors.append("Calls must finish with tool_calls")
            ids = [call["id"] for call in choice["tool_calls"]]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate tool_call_id")
            assistant = {"role": "assistant", "content": choice["content"],
                         "tool_calls": [{"id": call["id"], "type": "function", "function": {
                             "name": call["name"], "arguments": json.dumps(call["arguments"], allow_nan=False)}}
                                        for call in choice["tool_calls"]]}
            if choice["reasoning"]:
                assistant["reasoning"] = choice["reasoning"]
            messages.append(assistant)
            # Deliberately return parallel results in reverse order. IDs, not
            # arrival order, must bind the results to their calls.
            for call in reversed(choice["tool_calls"]):
                value = tools.execute(call["name"], call["arguments"])
                tools.calls[-1]["turn"] = len(turns) - 1
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(value, ensure_ascii=False)})
        else:
            errors.append("Tool loop exceeded max_turns")
        if expected_counts is not None and tools.counts() != expected_counts:
            errors.append(f"Tool execution counts {tools.counts()} != {expected_counts}")
        if expected_arguments is not None and any(
                call["arguments"] != expected_arguments.get(call["name"]) for call in tools.calls):
            errors.append("Tool arguments differ from the requested values")
        if fail_first_weather:
            weather_calls = [c for c in tools.calls if c["name"] == "weather"]
            if (len(weather_calls) != 2 or not weather_calls[0]["result"].get("retryable")
                    or weather_calls[1]["turn"] <= weather_calls[0]["turn"]):
                errors.append("Retry must follow the observed error in a later assistant turn")
        if enabled and not any(t["response"]["choices"][0]["reasoning"].strip() for t in turns):
            errors.append("Thinking request produced no reasoning")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {"passed": not errors, "errors": errors, "turns": turns, "executions": tools.calls}


def require_parallel(result):
    if not any(len(t["response"]["choices"][0]["tool_calls"]) >= 2 for t in result["turns"]):
        result["errors"].append("No assistant turn issued parallel tool calls")
        result["passed"] = False
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    rows = {}
    for stream in (False, True):
        rows[f"parallel-{stream}"] = require_parallel(conversation(args.base_url, args.model,
            "Wywołaj add z a=17 i b=23 oraz weather dla Krakowa. Podaj sumę i temperaturę z wyników.",
            family=args.family, stream=stream, expected_counts={"add": 1, "weather": 1},
            expected_arguments={"add": {"a": 17, "b": 23}, "weather": {"city": "Kraków"}}, answers=(40, 21)))
        rows[f"retry-{stream}"] = conversation(args.base_url, args.model,
            "Sprawdź weather dla Krakowa. Jeżeli narzędzie zwróci retryable=true, ponów weather dokładnie raz. Podaj temperaturę.",
            family=args.family, stream=stream, fail_first_weather=True,
            expected_counts={"weather": 2}, expected_arguments={"weather": {"city": "Kraków"}}, answers=(21,))
    # Independent conversations, simultaneously on the same serving process.
    jobs = [(17, 23, False, False), (-1, 5, True, True), (101, 202, False, True), (7, 11, True, False)]
    def run(job):
        a, b, enabled, streaming = job
        result = conversation(args.base_url, args.model,
            f"Wywołaj add z a={a} i b={b}. Podaj wynik narzędzia.", family=args.family,
            enabled=enabled, stream=streaming, expected_counts={"add": 1}, answers=(a+b,))
        if any(c["arguments"] != {"a": a, "b": b} for c in result["executions"]):
            result["errors"].append("Arguments leaked from another request")
            result["passed"] = False
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(run, jobs)):
            rows[f"isolated-{index}"] = result
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    for name, row in rows.items():
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}: {row['errors']}")
    return 0 if all(r["passed"] for r in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
