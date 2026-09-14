#!/usr/bin/env python3
"""Parser smoke tests 1–7 against a live vLLM chat server (v3.1).

Covers serving-side tool/reasoning parsers via /v1/chat/completions:

1. tool_choice=required (even when a direct answer is tempting)
2. Parallel tool calls
3. Streaming tool calls
4. Thinking + tool in one turn
5. tool_choice=none (no calls despite tools present)
6. Honest reply after tool ERROR
7. reasoning_effort=none vs high

Exit 0 only if all checks pass.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from _vllm_env import make_client

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    extras: dict = field(default_factory=dict)


def _msg_fields(message) -> dict:
    reasoning = getattr(message, "reasoning", None) or getattr(
        message, "reasoning_content", None
    )
    tool_calls = message.tool_calls or []
    return {
        "content": message.content,
        "reasoning": reasoning,
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in tool_calls
        ],
        "n_tools": len(tool_calls),
    }


def chat(
    client,
    model,
    *,
    messages,
    tools=None,
    tool_choice=None,
    reasoning_effort="none",
    stream=False,
    max_tokens=400,
    temperature=0.0,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": reasoning_effort}},
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def test_1_required(client, model) -> CheckResult:
    name = "1_tool_choice_required"
    resp = chat(
        client,
        model,
        messages=[{"role": "user", "content": "Opowiedz coś o Paryżu."}],
        tools=[WEATHER],
        tool_choice="required",
        reasoning_effort="none",
    )
    m = _msg_fields(resp.choices[0].message)
    ok = m["n_tools"] >= 1
    return CheckResult(
        name,
        ok,
        detail="expected >=1 tool_calls" if not ok else f"got {m['n_tools']} tool_calls",
        extras=m,
    )


def test_2_parallel(client, model) -> CheckResult:
    name = "2_parallel_tool_calls"
    resp = chat(
        client,
        model,
        messages=[
            {
                "role": "user",
                "content": "Jaka jest aktualna pogoda w Paryżu i w Berlinie? Sprawdź obie.",
            }
        ],
        tools=[WEATHER],
        tool_choice="auto",
        reasoning_effort="none",
        max_tokens=500,
    )
    m = _msg_fields(resp.choices[0].message)
    cities = []
    for tc in m["tool_calls"]:
        try:
            args = json.loads(tc["arguments"])
            cities.append(str(args.get("city", "")).lower())
        except json.JSONDecodeError:
            cities.append("")
    joined = " ".join(cities)
    has_paris = any("pary" in c or "paris" in c for c in cities)
    has_berlin = any("berlin" in c for c in cities)
    ok = m["n_tools"] >= 2 and has_paris and has_berlin
    return CheckResult(
        name,
        ok,
        detail=(
            f"n_tools={m['n_tools']} cities={cities!r}"
            if not ok
            else f"parallel ok: {cities}"
        ),
        extras=m,
    )


def test_3_streaming(client, model) -> CheckResult:
    name = "3_streaming_tool_calls"
    stream = chat(
        client,
        model,
        messages=[{"role": "user", "content": "Jaka jest aktualna pogoda w Tokio?"}],
        tools=[WEATHER],
        tool_choice="auto",
        reasoning_effort="none",
        stream=True,
    )
    by_idx: dict[int, dict] = {}
    content = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
        if not delta.tool_calls:
            continue
        for tc in delta.tool_calls:
            slot = by_idx.setdefault(
                tc.index, {"id": None, "name": "", "arguments": ""}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
    calls = [by_idx[i] for i in sorted(by_idx)]
    ok = (
        len(calls) >= 1
        and all(c.get("id") and c.get("name") == "get_weather" for c in calls)
        and all(c.get("arguments") for c in calls)
    )
    # arguments should parse as JSON
    if ok:
        try:
            for c in calls:
                json.loads(c["arguments"])
        except json.JSONDecodeError as e:
            ok = False
            detail = f"bad streamed JSON: {e}"
        else:
            detail = f"streamed {len(calls)} tool_calls"
    else:
        detail = f"calls={calls!r} content={content!r}"
    return CheckResult(name, ok, detail=detail, extras={"calls": calls, "content": content})


def test_4_think_and_tool(client, model) -> CheckResult:
    name = "4_thinking_plus_tool"
    resp = chat(
        client,
        model,
        messages=[{"role": "user", "content": "Jaka jest aktualna pogoda w Warszawie?"}],
        tools=[WEATHER],
        tool_choice="auto",
        reasoning_effort="medium",
        temperature=0.7,
        max_tokens=600,
    )
    m = _msg_fields(resp.choices[0].message)
    has_reasoning = bool(m["reasoning"] and str(m["reasoning"]).strip())
    has_tool = m["n_tools"] >= 1
    # Prefer think extracted out of content; soft-fail if think leaked into content only.
    content = m["content"] or ""
    leaked = "</think>" in content or "<think>" in content
    ok = has_tool and (has_reasoning or leaked)
    detail = (
        f"reasoning={'yes' if has_reasoning else 'no'} "
        f"tools={m['n_tools']} leaked_tags={leaked}"
    )
    # Strict: want structured reasoning field when parser works.
    if has_tool and not has_reasoning:
        ok = False
        detail += " (FAIL: tool ok but reasoning not split into message.reasoning)"
    return CheckResult(name, ok, detail=detail, extras=m)


def test_5_tool_none(client, model) -> CheckResult:
    name = "5_tool_choice_none"
    resp = chat(
        client,
        model,
        messages=[{"role": "user", "content": "Jaka jest aktualna pogoda w Paryżu?"}],
        tools=[WEATHER],
        tool_choice="none",
        reasoning_effort="none",
    )
    m = _msg_fields(resp.choices[0].message)
    ok = m["n_tools"] == 0 and bool((m["content"] or "").strip())
    return CheckResult(
        name,
        ok,
        detail=(
            f"n_tools={m['n_tools']} content={((m['content'] or '')[:120])!r}"
            if not ok
            else "no tool_calls, answered in content"
        ),
        extras=m,
    )


def test_6_tool_error_roundtrip(client, model) -> CheckResult:
    name = "6_tool_error_honest"
    # Seed a prior tool call + ERROR result, then ask follow-up.
    messages = [
        {"role": "user", "content": "Jaka jest pogoda w Tokio?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_tokyo_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tokyo_1",
            "name": "get_weather",
            "content": "ERROR: upstream timeout — no data returned",
        },
        {"role": "user", "content": "No to jak jest?"},
    ]
    resp = chat(
        client,
        model,
        messages=messages,
        tools=[WEATHER],
        tool_choice="auto",
        reasoning_effort="none",
    )
    m = _msg_fields(resp.choices[0].message)
    text = (m["content"] or "").lower()
    markers = (
        "błąd",
        "bledu",
        "błędu",
        "error",
        "timeout",
        "nie udało",
        "nie zwróci",
        "nie mogę",
        "nie moge",
        "brak",
        "nie powiod",
        "problem",
        "failed",
    )
    admits = any(x in text for x in markers)
    invents = any(x in text for x in ("22°c", "22 c", "słonecznie w tokio", "sunny in tokyo"))
    ok = m["n_tools"] == 0 and admits and not invents
    return CheckResult(
        name,
        ok,
        detail=(
            f"admits={admits} invents={invents} n_tools={m['n_tools']} "
            f"content={((m['content'] or '')[:200])!r}"
        ),
        extras=m,
    )


def test_7_effort_none_vs_high(client, model) -> CheckResult:
    name = "7_reasoning_effort_none_vs_high"
    prompt = [
        {
            "role": "user",
            "content": (
                "W trzech pudełkach są jabłka. W pierwszym 4, w każdym następnym "
                "dwa razy więcej niż w poprzednim. Ile jabłek łącznie? Pokaż tok."
            ),
        }
    ]
    none_resp = chat(
        client,
        model,
        messages=prompt,
        reasoning_effort="none",
        max_tokens=400,
    )
    high_resp = chat(
        client,
        model,
        messages=prompt,
        reasoning_effort="high",
        temperature=0.7,
        max_tokens=800,
    )
    none_m = _msg_fields(none_resp.choices[0].message)
    high_m = _msg_fields(high_resp.choices[0].message)
    none_r = (none_m["reasoning"] or "").strip()
    high_r = (high_m["reasoning"] or "").strip()
    none_c = none_m["content"] or ""
    high_c = high_m["content"] or ""
    # none: no (or empty) reasoning field; answer in content; preferably mentions 28
    none_ok = (not none_r) and ("28" in none_c)
    # high: non-empty reasoning split out; answer still has 28
    high_ok = bool(high_r) and ("28" in high_c or "28" in high_r)
    # soft length signal
    length_ok = len(high_r) >= len(none_r)
    ok = none_ok and high_ok and length_ok
    detail = (
        f"none: reasoning_len={len(none_r)} content_has_28={'28' in none_c}; "
        f"high: reasoning_len={len(high_r)} answer_has_28="
        f"{'28' in high_c or '28' in high_r}"
    )
    return CheckResult(
        name,
        ok,
        detail=detail,
        extras={"none": none_m, "high": high_m},
    )


def main() -> int:
    client, model = make_client()
    print(f"model={model}")
    tests = [
        test_1_required,
        test_2_parallel,
        test_3_streaming,
        test_4_think_and_tool,
        test_5_tool_none,
        test_6_tool_error_roundtrip,
        test_7_effort_none_vs_high,
    ]
    results: list[CheckResult] = []
    for fn in tests:
        try:
            r = fn(client, model)
        except Exception as e:  # noqa: BLE001
            r = CheckResult(fn.__name__, False, detail=f"EXCEPTION: {type(e).__name__}: {e}")
        results.append(r)
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        if not r.ok and r.extras:
            print(f"         extras={json.dumps(r.extras, ensure_ascii=False)[:500]}")

    n_pass = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_pass
    print(f"\nSUMMARY: {n_pass} pass, {n_fail} fail / {len(results)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
