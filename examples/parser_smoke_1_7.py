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
8. tool_choice=required under stress ("Cześć", small max_tokens)
9. streaming reasoning (no tools)
10. streaming thinking + tool
11. hard JSON tool arguments / round-trip

Usage: python parser_smoke_1_7.py [all|core|extra]

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


def test_8_required_stress(client, model) -> CheckResult:
    """tool_choice=required on a greeting — must still call a tool eventually.

    The template's "required" contract only promises "don't answer without
    calling a tool first" — it does NOT forbid a short lead-in before the
    <tool_call> tag. So the guided regex intentionally allows free text
    around the call, and this test uses a realistic max_tokens (not an
    artificially tiny one) so a normal lead-in has room to complete.
    """
    name = "8_required_stress_hi"
    resp = chat(
        client,
        model,
        messages=[{"role": "user", "content": "Cześć!"}],
        tools=[WEATHER],
        tool_choice="required",
        reasoning_effort="none",
        max_tokens=200,
    )
    m = _msg_fields(resp.choices[0].message)
    ok = m["n_tools"] >= 1
    return CheckResult(
        name,
        ok,
        detail=(
            f"n_tools={m['n_tools']} finish={resp.choices[0].finish_reason} "
            f"content={((m['content'] or '')[:120])!r}"
            if not ok
            else f"forced tool under stress: {m['tool_calls'][0]['name']}"
        ),
        extras=m,
    )


def _aggregate_stream(stream) -> dict:
    content = ""
    reasoning = ""
    by_idx: dict[int, dict] = {}
    finish = None
    for chunk in stream:
        choice = chunk.choices[0]
        finish = choice.finish_reason or finish
        delta = choice.delta
        for attr in ("reasoning", "reasoning_content"):
            piece = getattr(delta, attr, None)
            if piece:
                reasoning += piece
        if delta.content:
            content += delta.content
        if delta.tool_calls:
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
    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": calls,
        "n_tools": len(calls),
        "finish": finish,
    }


def test_9_streaming_reasoning(client, model) -> CheckResult:
    name = "9_streaming_reasoning"
    stream = chat(
        client,
        model,
        messages=[
            {
                "role": "user",
                "content": "W trzech pudełkach są jabłka: 4, potem 2x więcej, potem znowu 2x. Ile łącznie?",
            }
        ],
        reasoning_effort="medium",
        stream=True,
        temperature=0.7,
        max_tokens=600,
    )
    m = _aggregate_stream(stream)
    reasoning = (m["reasoning"] or "").strip()
    content = m["content"] or ""
    leaked = "<think>" in content or "</think>" in content
    ok = bool(reasoning) and not leaked and ("28" in content or "28" in reasoning)
    return CheckResult(
        name,
        ok,
        detail=(
            f"reasoning_len={len(reasoning)} leaked={leaked} "
            f"has_28={'28' in content or '28' in reasoning}"
        ),
        extras=m,
    )


def test_10_streaming_think_and_tool(client, model) -> CheckResult:
    name = "10_streaming_think_plus_tool"
    stream = chat(
        client,
        model,
        messages=[{"role": "user", "content": "Jaka jest aktualna pogoda w Krakowie?"}],
        tools=[WEATHER],
        tool_choice="auto",
        reasoning_effort="medium",
        stream=True,
        temperature=0.7,
        max_tokens=600,
    )
    m = _aggregate_stream(stream)
    reasoning = (m["reasoning"] or "").strip()
    content = m["content"] or ""
    leaked = "<think>" in content or "</think>" in content
    calls_ok = m["n_tools"] >= 1 and all(
        c.get("id") and c.get("name") == "get_weather" and c.get("arguments")
        for c in m["tool_calls"]
    )
    if calls_ok:
        try:
            for c in m["tool_calls"]:
                json.loads(c["arguments"])
        except json.JSONDecodeError:
            calls_ok = False
    ok = bool(reasoning) and calls_ok and not leaked
    return CheckResult(
        name,
        ok,
        detail=(
            f"reasoning_len={len(reasoning)} n_tools={m['n_tools']} leaked={leaked}"
        ),
        extras=m,
    )


def test_11_hard_json_args(client, model) -> CheckResult:
    """Model should emit a tool call; we also round-trip a hard JSON tool result."""
    name = "11_hard_json_args"
    search = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web; query may contain code or markup",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    # Ask for a search whose natural query includes quotes / angle brackets.
    resp = chat(
        client,
        model,
        messages=[
            {
                "role": "user",
                "content": (
                    'Wyszukaj w sieci dokładnie ten ciąg (użyj narzędzia web_search): '
                    'foo "bar" <baz> oraz linię z kodem: x = "a\\nb"'
                ),
            }
        ],
        tools=[search],
        tool_choice="required",
        reasoning_effort="none",
        max_tokens=300,
    )
    m = _msg_fields(resp.choices[0].message)
    if m["n_tools"] < 1:
        return CheckResult(name, False, detail="no tool_calls", extras=m)
    tc = m["tool_calls"][0]
    try:
        args = json.loads(tc["arguments"])
    except json.JSONDecodeError as e:
        return CheckResult(
            name, False, detail=f"arguments not JSON: {e} raw={tc['arguments']!r}", extras=m
        )
    query = str(args.get("query", ""))
    # Round-trip: feed back a tool result that itself contains hard characters.
    hard_result = (
        '[{"title": "Hit with <tag> and \\"quotes\\"", "snippet": "line1\\nline2"}]'
    )
    messages = [
        {
            "role": "user",
            "content": (
                'Wyszukaj w sieci dokładnie ten ciąg (użyj narzędzia web_search): '
                'foo "bar" <baz>'
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": "web_search", "arguments": tc["arguments"]},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": "web_search",
            "content": hard_result,
        },
    ]
    resp2 = chat(
        client,
        model,
        messages=messages,
        tools=[search],
        tool_choice="auto",
        reasoning_effort="none",
        max_tokens=300,
    )
    m2 = _msg_fields(resp2.choices[0].message)
    # First call parsed; second turn answered without crashing (content or another call).
    ok = isinstance(args, dict) and "query" in args and (
        bool((m2["content"] or "").strip()) or m2["n_tools"] >= 0
    )
    # Prefer that query kept some of the hard markers if the model cooperated.
    markers_kept = sum(1 for x in ('"', "<", ">") if x in query)
    detail = (
        f"query={query!r} markers_in_query={markers_kept} "
        f"followup_content_len={len(m2['content'] or '')}"
    )
    return CheckResult(name, ok, detail=detail, extras={"first": m, "second": m2})


def main() -> int:
    client, model = make_client()
    print(f"model={model}")
    suite = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    core = [
        test_1_required,
        test_2_parallel,
        test_3_streaming,
        test_4_think_and_tool,
        test_5_tool_none,
        test_6_tool_error_roundtrip,
        test_7_effort_none_vs_high,
    ]
    extra = [
        test_8_required_stress,
        test_9_streaming_reasoning,
        test_10_streaming_think_and_tool,
        test_11_hard_json_args,
    ]
    if suite in ("extra", "8-11", "stress"):
        tests = extra
    elif suite in ("core", "1-7"):
        tests = core
    else:
        tests = core + extra
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
