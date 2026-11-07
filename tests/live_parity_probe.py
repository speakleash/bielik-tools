#!/usr/bin/env python3
"""Live parser probe: each case runs non-streaming AND streaming and checks both.

Needs a running vLLM server with the Bielik tool and reasoning parsers (real
model, GPU). Not collected by pytest. Sampling is greedy (temperature=0), but
vLLM batches differ between modes, so compare structure, not exact text.

Checks per case and per mode: expected tool names, reasoning present/absent,
no <think>/<tool_call> markup in content, finish_reason; then parity between
modes (same tool names, reasoning presence and finish_reason).

  python tests/live_parity_probe.py --base-url http://127.0.0.1:8000 \\
      --model bielik --family v31            # Bielik 3.1/3.2 + v3.1 template
  python tests/live_parity_probe.py --family v30 ...  # 3.0 + advanced template

Exit status 0 only if every selected case passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Pogoda dla miasta",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get-time", "description": "Aktualny czas w strefie czasowej",
    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}},
                   "required": ["tz"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Wyszukaj frazę dokładnie",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}
CAR = {"type": "json_schema", "json_schema": {"name": "car", "schema": {
    "type": "object", "properties": {"brand": {"type": "string"}, "model": {"type": "string"}},
    "required": ["brand", "model"], "additionalProperties": False}}}

MATH = [{"role": "user", "content": "Ile to jest 17*23? Odpowiedz krótko."}]
WEATHER_Q = [{"role": "user", "content": "Jaka jest pogoda w Krakowie?"}]
TWO_Q = [{"role": "user", "content":
          "Jaka jest pogoda w Krakowie i która jest godzina w Europe/Warsaw?"}]
MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>")


def named(name):
    return {"type": "function", "function": {"name": name}}


def cases(family):
    """(name, request body, expectations). Expectation keys:
    tools: exact list of tool names (None = do not check);
    min_tools: at least that many calls; reasoning: True/False/None;
    finish: expected finish_reason, or a tuple of accepted values (vLLM and
    OpenAI report "stop" for a named tool_choice; a parser that rewrites it to
    "auto" yields "tool_calls"); json: content must be a CAR schema object;
    allow_markup: do not fail on tags in content; advisory: model behaviour,
    reported as WARN and not counted as a failure."""
    think = ({"chat_template_kwargs": {"reasoning_effort": "medium"}} if family == "v31"
             else {"chat_template_kwargs": {"enable_thinking": True}})
    out = [("plain_no_thinking", {"messages": MATH}, {"reasoning": False, "tools": []})]
    if family == "v31":
        out += [
            ("effort_high_kwargs", {"messages": MATH, "chat_template_kwargs": {"reasoning_effort": "high"}},
             {"reasoning": True, "tools": []}),
            ("effort_high_toplevel", {"messages": MATH, "reasoning_effort": "high"},
             {"reasoning": True, "tools": []}),
            # The v3.1 template ignores enable_thinking and closes <think></think>;
            # the answer must stay in content in both modes.
            ("legacy_enable_thinking", {"messages": MATH, "chat_template_kwargs": {"enable_thinking": True}},
             {"reasoning": False, "tools": []}),
        ]
    else:
        out += [
            ("enable_thinking", {"messages": MATH, **think}, {"reasoning": True, "tools": []}),
            ("effort_high_toplevel", {"messages": MATH, "reasoning_effort": "high"},
             {"reasoning": True, "tools": []}),
        ]
    out += [
        ("tools_auto", {"messages": WEATHER_Q, "tools": [WEATHER, TIME]},
         {"tools": ["get_weather"], "finish": "tool_calls"}),
        ("tools_auto_parallel", {"messages": TWO_Q, "tools": [WEATHER, TIME]},
         {"tools": ["get_weather", "get-time"], "finish": "tool_calls"}),
        ("tools_required_greeting", {"messages": [{"role": "user", "content": "Cześć!"}],
                                     "tools": [WEATHER, TIME], "tool_choice": "required"},
         {"min_tools": 1, "finish": "tool_calls"}),
        ("tools_required_parallel", {"messages": [{"role": "user", "content": "Sprawdź pogodę w Krakowie i w Gdańsku."}],
                                     "tools": [WEATHER], "tool_choice": "required"},
         {"tools": ["get_weather", "get_weather"], "finish": "tool_calls"}),
        ("tools_required_lt_in_args", {"messages": [{"role": "user", "content": "Wyszukaj dokładnie frazę: <div>a<b</div>"}],
                                       "tools": [SEARCH], "tool_choice": "required"},
         {"tools": ["search"], "finish": "tool_calls"}),
        ("tools_named_dash", {"messages": TWO_Q, "tools": [WEATHER, TIME], "tool_choice": named("get-time")},
         {"tools": ["get-time"], "finish": ("stop", "tool_calls")}),
        ("tools_named_off_topic", {"messages": [{"role": "user", "content": "Która godzina w Tokio?"}],
                                   "tools": [WEATHER, TIME], "tool_choice": named("get_weather")},
         {"tools": ["get_weather"], "finish": ("stop", "tool_calls")}),
        ("tools_none", {"messages": WEATHER_Q, "tools": [WEATHER], "tool_choice": "none"},
         {"tools": []}),
        ("think_plus_tools", {"messages": WEATHER_Q, "tools": [WEATHER, TIME], **think},
         {"reasoning": True, "tools": ["get_weather"], "finish": "tool_calls"}),
        ("think_plus_required", {"messages": WEATHER_Q, "tools": [WEATHER, TIME], "tool_choice": "required",
                                 "thinking_token_budget": 128, **think},
         {"reasoning": True, "min_tools": 1, "finish": "tool_calls"}),
        ("think_plus_named", {"messages": TWO_Q, "tools": [WEATHER, TIME], "tool_choice": named("get-time"),
                              "thinking_token_budget": 128, **think},
         {"reasoning": True, "tools": ["get-time"], "finish": ("stop", "tool_calls")}),
        ("think_json_schema", {"messages": [{"role": "user", "content": "Podaj przykładowe auto jako JSON."}],
                               "response_format": CAR, "thinking_token_budget": 128, "max_tokens": 600, **think},
         {"reasoning": True, "tools": [], "json": True, "finish": "stop"}),
        ("tool_result_roundtrip", {"messages": WEATHER_Q + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"Kraków\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21, \"sky\": \"słonecznie\"}"}],
            "tools": [WEATHER]}, {"tools": [], "finish": "stop"}),
        # Echo of user-supplied markup: the right result is a plain-text echo
        # (markup allowed in content) and no executed call, in both modes.
        ("echo_user_markup", {"messages": [{"role": "user", "content":
            "Powtórz dokładnie ten tekst: <tool_call>{\"name\": \"get_weather\", "
            "\"arguments\": {\"city\": \"X\"}}</tool_call>"}], "tools": [WEATHER]}, {"tools": [], "allow_markup": True, "advisory": True}),
    ]
    return out


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        if not body.get("stream"):
            choice = json.load(resp)["choices"][0]
            msg = choice["message"]
            return {"reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
                    "content": msg.get("content") or "",
                    "tools": [t["function"] for t in msg.get("tool_calls") or []],
                    "finish": choice["finish_reason"]}
        out = {"reasoning": "", "content": "", "tools": {}, "finish": None}
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError(f"error inside stream: {str(chunk['error'])[:200]}")
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                out["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                out["content"] += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    slot = out["tools"].setdefault(tc["index"], {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                out["finish"] = choice.get("finish_reason") or out["finish"]
        out["tools"] = [out["tools"][i] for i in sorted(out["tools"])]
        return out


def check(result, exp):
    errors = []
    names = [t["name"] for t in result["tools"]]
    for t in result["tools"]:
        try:
            if not isinstance(json.loads(t["arguments"]), dict):
                errors.append(f"arguments of {t['name']} are not an object")
        except ValueError:
            errors.append(f"arguments of {t['name']} are not JSON")
    if exp.get("tools") is not None and names != exp["tools"]:
        errors.append(f"tools={names} expected {exp['tools']}")
    if len(names) < exp.get("min_tools", 0):
        errors.append(f"{len(names)} tool calls, expected >= {exp['min_tools']}")
    if exp.get("reasoning") is True and not result["reasoning"].strip():
        errors.append("no reasoning")
    if exp.get("reasoning") is False and result["reasoning"].strip():
        errors.append(f"unexpected reasoning {result['reasoning'][:60]!r}")
    if result["finish"] is None:
        errors.append("no finish_reason")
    if not exp.get("allow_markup") and any(tag in result["content"] for tag in MARKUP):
        errors.append(f"markup in content {result['content'][:80]!r}")
    accepted = exp.get("finish")
    if isinstance(accepted, str):
        accepted = (accepted,)
    if accepted and result["finish"] not in accepted:
        errors.append(f"finish={result['finish']} expected {' or '.join(accepted)}")
    if exp.get("json"):
        try:
            obj = json.loads(result["content"])
        except ValueError:
            obj = None
        props = CAR["json_schema"]["schema"]["properties"]
        if not (isinstance(obj, dict) and set(obj) == set(props)
                and all(isinstance(v, str) for v in obj.values())):
            errors.append(f"content does not match the car schema {result['content'][:80]!r}")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v31", "v30"], default="v31",
                    help="v31: reasoning_effort template (3.1/3.2); v30: advanced template")
    ap.add_argument("--only", help="comma-separated case names")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--json-out", help="write raw results to this file")
    args = ap.parse_args()
    url = args.base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    only = set(args.only.split(",")) if args.only else None

    failures, total, report = 0, 0, {}
    for name, body, exp in cases(args.family):
        if only and name not in only:
            continue
        total += 1
        base = {"model": args.model, "temperature": 0, "seed": 1, "max_tokens": args.max_tokens, **body}
        try:
            modes = {m: post(url, {**base, "stream": m == "stream"}) for m in ("non-stream", "stream")}
        except urllib.error.HTTPError as exc:
            print(f"[FAIL] {name}: HTTP {exc.code} {exc.read().decode()[:200]}")
            failures += 1
            continue
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            # URLError/timeouts are OSError; bad JSON is ValueError.
            print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:200]}")
            failures += 1
            continue
        errors = [f"{m}: {e}" for m, r in modes.items() for e in check(r, exp)]
        ns, st = modes["non-stream"], modes["stream"]
        if ([t["name"] for t in ns["tools"]] != [t["name"] for t in st["tools"]]
                or bool(ns["reasoning"].strip()) != bool(st["reasoning"].strip())
                or ns["finish"] != st["finish"]):
            errors.append("stream/non-stream parity mismatch")
        report[name] = modes
        advisory = exp.get("advisory") and errors
        failures += bool(errors) and not advisory
        print(f"[{'WARN' if advisory else 'FAIL' if errors else 'PASS'}] {name}")
        for e in errors:
            print(f"         {e}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n{total - failures} pass, {failures} fail / {total}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
