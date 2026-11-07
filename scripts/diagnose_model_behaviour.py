#!/usr/bin/env python3
"""Diagnose how a trained Bielik model emits reasoning and tool-call markup.

Works on the raw generated token IDs (vLLM ``return_token_ids``), so the
result does not depend on the tool/reasoning parsers. For each scenario the
script samples N completions (plus one greedy) and reports how often the
model shows each behaviour, for example:

- writes ``<think>``/``</think>``/``<tool_call>`` as ordinary text pieces
  (``" <", "tool", "_", "call", ">"``) instead of the dedicated token ID;
- opens ``<think>`` again although the prompt already opened it, never closes
  it, thinks when thinking is off, or leaves reasoning empty;
- puts a tool call inside reasoning, emits untagged JSON calls, invalid JSON,
  unknown tools, ``parameters`` instead of ``arguments``, string arguments,
  schema violations, calls when none are expected (or none when expected);
- generates ``<|im_start|>`` (role hallucination), copies the template's
  zero-width-space escapes, stops with ``</s>`` instead of ``<|im_end|>``,
  hits the length limit or repeats itself; reasoning language (pl/en).

Literal tags in scenarios with guided decoding (``required``/named) are
reported but not counted as model issues: the grammar may forbid the
dedicated token IDs.

  python scripts/diagnose_model_behaviour.py --base-url http://127.0.0.1:8000 \\
      --model bielik --tokenizer /path/to/bielik-3.2 --family v31
  python scripts/diagnose_model_behaviour.py --model bielik30 --family v30 \\
      --tokenizer speakleash/Bielik-11B-v3.0-Instruct

Needs a running vLLM >= 0.32 server (any Bielik parser pair) and the
tokenizer files locally. Exit 0 unless --strict is given and an issue rate
exceeds --max-issue-rate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter

from transformers import AutoTokenizer

TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>")
ZWSP = "​"
PL_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
PL_WORDS = {"jest", "się", "nie", "że", "to", "na", "w", "i", "czy", "jak", "użytkownik", "muszę", "więc"}
EN_WORDS = {"the", "is", "user", "i", "need", "to", "so", "let", "me", "and", "of", "wants", "check"}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Pogoda dla miasta",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Aktualny czas w strefie czasowej IANA",
    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}},
                   "required": ["tz"]}}}
CONVERT = {"type": "function", "function": {
    "name": "convert_currency", "description": "Przelicz kwotę między walutami",
    "parameters": {"type": "object", "properties": {
        "amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
        "required": ["amount", "from", "to"]}}}
TOOLS = [WEATHER, TIME, CONVERT]

MATH = [{"role": "user", "content": "Pociąg jedzie 120 km w 1,5 godziny. Jaka jest jego średnia prędkość?"}]
WEATHER_Q = [{"role": "user", "content": "Jaka jest teraz pogoda w Gdańsku?"}]
MULTI_Q = [{"role": "user", "content": "Sprawdź pogodę w Krakowie i przelicz 250 EUR na PLN."}]


def scenarios(family):
    """name -> (request body, expectations). Expectations:
    think: True/False/None; call: True/False/None; names: allowed tool names;
    constrained: guided decoding (literal tags expected); echo: the user asks
    to copy markup, so literal tags are a faithful copy, reported as info."""
    on = ({"chat_template_kwargs": {"reasoning_effort": "medium"}} if family == "v31"
          else {"chat_template_kwargs": {"enable_thinking": True}})
    off = ({"chat_template_kwargs": {"reasoning_effort": "none"}} if family == "v31"
           else {"chat_template_kwargs": {"enable_thinking": False}})
    result_turn = WEATHER_Q + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": "{\"city\": \"Gdańsk\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 14, \"sky\": \"pochmurno\"}"}]
    history = [{"role": "user", "content": "Ile to 12*12?"},
               {"role": "assistant", "content": "144", "reasoning": "12*12 = 144"},
               {"role": "user", "content": "A pierwiastek z tego wyniku?"}]
    return {
        "think_on_math": ({"messages": MATH, **on}, {"think": True, "call": False}),
        "think_off_math": ({"messages": MATH, **off}, {"think": False, "call": False}),
        "think_on_history": ({"messages": history, **on}, {"think": True, "call": False}),
        "tools_single": ({"messages": WEATHER_Q, "tools": TOOLS, **off},
                         {"think": False, "call": True, "names": {"get_weather"}}),
        "tools_parallel": ({"messages": MULTI_Q, "tools": TOOLS, **off},
                           {"think": False, "call": True, "names": {"get_weather", "convert_currency"}}),
        "tools_think_single": ({"messages": WEATHER_Q, "tools": TOOLS, **on},
                               {"think": True, "call": True, "names": {"get_weather"}}),
        "tools_chitchat": ({"messages": [{"role": "user", "content": "Cześć! Jak się dziś masz?"}],
                            "tools": TOOLS, **off}, {"think": False, "call": False}),
        "tools_after_result": ({"messages": result_turn, "tools": TOOLS, **off}, {"think": False, "call": False}),
        "tools_echo_markup": ({"messages": [{"role": "user", "content":
            "Przepisz dokładnie, znak w znak: <tool_call>{\"name\": \"get_weather\", "
            "\"arguments\": {\"city\": \"X\"}}</tool_call> oraz <think>test</think>"}],
            "tools": TOOLS, **off}, {"think": False, "call": False, "echo": True}),
        "no_tools_function_question": ({"messages": [{"role": "user", "content":
            "Pokaż przykład, jak model językowy wywołuje funkcję get_weather w formacie JSON."}], **off},
            {"think": False, "call": False}),
        "required_constrained": ({"messages": WEATHER_Q, "tools": TOOLS, "tool_choice": "required", **off},
                                 {"think": False, "call": True, "names": {"get_weather"}, "constrained": True}),
    }


class Diagnoser:
    def __init__(self, tok):
        self.tok = tok
        self.tag_ids = {tok.convert_tokens_to_ids(t): t for t in TAGS}
        self.tag_ids.pop(tok.unk_token_id, None)
        self.eos_s = tok.convert_tokens_to_ids("</s>")

    def segment(self, ids):
        """Split IDs into ('special', tag) and ('text', decoded run) items."""
        items, run = [], []
        for i in ids:
            if i in self.tag_ids:
                if run:
                    items.append(("text", self.tok.decode(run)))
                    run = []
                items.append(("special", self.tag_ids[i]))
            else:
                run.append(i)
        if run:
            items.append(("text", self.tok.decode(run)))
        return items

    def prompt_think_open(self, prompt_ids):
        if not prompt_ids:
            return None
        text = self.tok.decode(prompt_ids)
        turn = text[text.rfind("<|im_start|>"):]
        return turn.rfind("<think>") > turn.rfind("</think>")

    def analyse(self, ids, prompt_ids, finish, exp, tools):
        items = self.segment(ids)
        special, literal = Counter(), Counter()
        for kind, val in items:
            if kind == "special":
                special[val] += 1
            else:
                for tag in TAGS:
                    literal[tag] += val.count(tag)
        # Drop the turn terminator (stop token) from the analysed text.
        while items and items[-1] == ("special", "<|im_end|>"):
            items.pop()
        text = "".join(val for _, val in items).removesuffix("</s>")
        flags = set()
        constrained = exp.get("constrained", False)
        echo = exp.get("echo", False)

        # Structure is read from dedicated token IDs only: literal tags in text
        # runs are neutralised, except under guided decoding, where the grammar
        # may force tags to be spelled with ordinary tokens.
        def neutral(val):
            for tag in TAGS:
                val = val.replace(tag, "\u2039" + tag[1:])
            return val
        struct = "".join(val if kind == "special" or constrained else neutral(val)
                         for kind, val in items).removesuffix("</s>")

        # Markup spelled with ordinary tokens instead of the dedicated ID.
        for tag in TAGS:
            if literal[tag]:
                if echo:
                    flags.add("echo:literal_markup")
                else:
                    flags.add(f"literal:{tag}" + (" (guided)" if constrained else ""))
        if ZWSP in text:
            flags.add("echo:zwsp_escapes" if echo else "zwsp_copied")

        # Reasoning structure.
        open_at_start = self.prompt_think_open(prompt_ids)
        body = ("<think>" if open_at_start else "") + struct
        if open_at_start and struct.lstrip().startswith("<think>"):
            flags.add("think_reopened")
            body = struct  # count the duplicate opening once
        implicit = False
        if "<think>" not in body and "</think>" in body:
            # DeepSeek-R1 style: text before the first </think> is reasoning.
            body, implicit = "<think>" + body, True
        opens = body.count("<think>")
        closes = body.count("</think>")
        if opens > closes:
            flags.add("think_unclosed")
        if closes > opens:
            flags.add("think_close_without_open")
        if opens > 1:
            flags.add("think_multiple_blocks")
        last_close = body.rfind("</think>")
        first_open = body.find("<think>")
        reasoning = body[first_open + 7:last_close] if 0 <= first_open < last_close else (
            body[first_open + 7:] if first_open >= 0 else "")
        answer = body[last_close + 8:] if last_close >= 0 else ("" if first_open >= 0 else body)
        if implicit:
            flags.add("think_open_omitted" if reasoning.strip() else "think_stray_close")
        if exp.get("think") is True and not reasoning.strip():
            flags.add("think_missing_or_empty")
        if exp.get("think") is False and reasoning.strip():
            flags.add("think_when_off")
        if reasoning.strip():
            letters = sum(c.isalpha() for c in reasoning) or 1
            words = re.findall(r"\w+", reasoning.lower())
            pl = sum(c in PL_CHARS for c in reasoning) / letters * 20 + sum(w in PL_WORDS for w in words)
            en = sum(w in EN_WORDS for w in words)
            flags.add("reasoning_lang:" + ("pl" if pl >= en else "en"))
        if "<tool_call>" in reasoning:
            flags.add("tool_call_in_reasoning")

        # Tool calls in the answer part.
        calls = re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.DOTALL)
        if answer.count("<tool_call>") > len(calls):
            flags.add("tool_call_unclosed")
        known = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
        for payload in calls:
            try:
                call = json.loads(payload)
            except ValueError:
                flags.add("call_invalid_json")
                continue
            if not isinstance(call, dict) or "name" not in call:
                flags.add("call_no_name")
                continue
            if "arguments" not in call:
                flags.add("call_parameters_key" if "parameters" in call else "call_no_arguments")
            args = call.get("arguments", call.get("parameters"))
            if isinstance(args, str):
                flags.add("call_arguments_as_string")
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if call["name"] not in known:
                flags.add("call_unknown_tool")
            elif isinstance(args, dict):
                schema = known[call["name"]]
                props = schema.get("properties", {})
                if any(k not in args for k in schema.get("required", [])):
                    flags.add("call_missing_required_arg")
                if props and any(k not in props for k in args):
                    flags.add("call_unknown_arg")
                types = {"string": str, "number": (int, float), "integer": int, "boolean": bool,
                         "object": dict, "array": list}
                for k, v in args.items():
                    want = types.get(props.get(k, {}).get("type"))
                    if want and (not isinstance(v, want) or (want != bool and isinstance(v, bool))):
                        flags.add("call_wrong_arg_type")
        stripped = re.sub(r"<tool_call>.*?</tool_call>", "", answer, flags=re.DOTALL)
        if calls and stripped.strip():
            flags.add("text_around_calls")
        if re.search(r'^\s*\{\s*"name"\s*:', stripped) or "<|function_call|>" in answer:
            flags.add("call_untagged_or_legacy_format")
        if exp.get("call") is True and not calls:
            flags.add("call_missing")
        if exp.get("call") is False and calls:
            flags.add("call_unexpected")
        allowed = exp.get("names")
        if allowed:
            for payload in calls:
                try:
                    if json.loads(payload).get("name") not in allowed:
                        flags.add("call_irrelevant_tool")
                except (ValueError, AttributeError):
                    pass

        # Termination and turn structure.
        if special["<|im_start|>"] or literal["<|im_start|>"]:
            flags.add("generated_im_start")
        if ids and ids[-1] == self.eos_s:
            flags.add("stopped_with_</s>")
        if finish == "length":
            flags.add("length_limit")
        words = re.findall(r"\w+", text.lower())
        grams = [tuple(words[i:i + 4]) for i in range(max(len(words) - 3, 0))]
        if len(grams) > 40 and len(set(grams)) / len(grams) < 0.5:
            flags.add("repetitive")
        return {"flags": sorted(flags), "special": dict(special), "literal": dict(+literal),
                "reasoning": reasoning[:300], "answer": answer[:300], "finish": finish}


INFO_PREFIXES = ("reasoning_lang:", "text_around_calls", "echo:", "think_open_omitted")


def is_issue(flag):
    return not flag.startswith(INFO_PREFIXES) and not flag.endswith("(guided)")


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, help="model directory or Hub ID")
    ap.add_argument("--family", choices=["v31", "v30"], default="v31")
    ap.add_argument("--samples", type=int, default=8, help="sampled completions per scenario")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("--jsonl-out", help="write one JSON line per completion")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-issue-rate", type=float, default=0.0)
    args = ap.parse_args()

    url = args.base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    diag = Diagnoser(AutoTokenizer.from_pretrained(args.tokenizer))
    only = set(args.only.split(",")) if args.only else None
    out = open(args.jsonl_out, "w", encoding="utf-8") if args.jsonl_out else None
    totals, worst = Counter(), 0.0
    tag_totals = {"special": Counter(), "literal": Counter()}

    for name, (body, exp) in scenarios(args.family).items():
        if only and name not in only:
            continue
        base = {"model": args.model, "max_tokens": args.max_tokens, "return_token_ids": True,
                "skip_special_tokens": False, **body}
        runs = [("greedy", {**base, "temperature": 0, "n": 1})]
        if args.samples:
            runs.append(("sampled", {**base, "temperature": args.temperature, "n": args.samples}))
        results = []
        try:
            for label, req in runs:
                resp = post(url, req)
                prompt_ids = resp.get("prompt_token_ids")
                for choice in resp["choices"]:
                    if choice.get("token_ids") is None:
                        raise RuntimeError("server returned no token_ids (needs vLLM >= 0.32)")
                    r = diag.analyse(choice["token_ids"], prompt_ids, choice["finish_reason"],
                                     exp, body.get("tools", []))
                    r["mode"] = label
                    results.append(r)
                    if out:
                        out.write(json.dumps({"scenario": name, **r, "token_ids": choice["token_ids"]},
                                             ensure_ascii=False) + "\n")
        except (urllib.error.URLError, OSError, ValueError, KeyError, RuntimeError) as exc:
            detail = exc.read().decode()[:200] if isinstance(exc, urllib.error.HTTPError) else ""
            print(f"## {name}: ERROR {type(exc).__name__}: {exc} {detail}")
            continue

        n = len(results)
        counts = Counter(f for r in results for f in r["flags"])
        for r in results:
            for kind in ("special", "literal"):
                tag_totals[kind].update(r[kind])
        greedy = results[0]["flags"]
        print(f"## {name}  ({n} completions; greedy: {', '.join(greedy) or 'clean'})")
        for flag, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            marker = "!" if is_issue(flag) else " "
            print(f"   {marker} {flag:40s} {c}/{n}")
            if is_issue(flag):
                totals[flag] += c
                worst = max(worst, c / n)
        if not counts:
            print("     clean")
    if out:
        out.close()

    print("\n## Tag usage over all completions (dedicated ID vs literal text)")
    for tag in TAGS:
        s, l = tag_totals["special"][tag], tag_totals["literal"][tag]
        if s or l:
            print(f"   {tag:14s} id={s:<5d} literal={l}")
    print("\n## Issues over all scenarios (count of completions)")
    for flag, c in totals.most_common():
        print(f"   {flag:40s} {c}")
    if not totals:
        print("   none")
    return 1 if args.strict and worst > args.max_issue_rate else 0


if __name__ == "__main__":
    sys.exit(main())
