"""Small auditable Chat Completions client for live integration tests.

Standard library only. It records responses and validates tool payloads;
it never executes model-generated tools. Importing it does not use network.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request


class ProtocolError(ValueError):
    pass


ADD = {"type": "function", "function": {"name": "add", "description": "Dodaj dwie liczby całkowite.",
       "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"], "additionalProperties": False}}}
WEATHER = {"type": "function", "function": {"name": "weather", "description": "Pogoda w podanym mieście.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                          "required": ["city"], "additionalProperties": False}}}
ANSWER_SCHEMA = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
                 "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"], "additionalProperties": False}}}


def named(name):
    return {"type": "function", "function": {"name": name}}


def thinking(family, enabled):
    return ({"reasoning_effort": "medium" if enabled else "none"} if family == "v31"
            else {"enable_thinking": enabled})


def tool_value(name, arguments):
    if not isinstance(name, str) or not name:
        raise ProtocolError("Tool name must be a non-empty string")
    if not isinstance(arguments, str):
        raise ProtocolError("Tool arguments must be JSON text on the wire")
    def invalid_constant(value):
        raise ProtocolError(f"Non-JSON constant: {value}")
    value = json.loads(arguments, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ProtocolError("Tool arguments must decode to an object")
    # Also reject numeric overflow (e.g. 1e999 -> inf), which cannot be
    # serialized back to interoperable JSON.
    json.dumps(value, allow_nan=False)
    return {"name": name, "arguments": value}


def normalize_message(choice):
    msg = choice["message"]
    if choice.get("finish_reason") is None:
        raise ProtocolError("Missing finish_reason")
    calls = []
    for call in msg.get("tool_calls") or []:
        fn = call["function"]
        calls.append({"id": call.get("id"), **tool_value(fn.get("name"), fn.get("arguments"))})
    return {"index": choice["index"], "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "tool_calls": calls, "finish_reason": choice["finish_reason"]}


def read_stream(response):
    slots, raw, done = {}, [], False
    first_output = None
    start = time.perf_counter()
    for line in response:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        raw.append(chunk)
        if "error" in chunk:
            raise ProtocolError(f"Error inside SSE stream: {chunk['error']}")
        for choice in chunk.get("choices", []):
            index = choice["index"]
            slot = slots.setdefault(index, {"index": index, "content": "", "reasoning": "",
                                           "tool_calls": {}, "finish_reason": None})
            delta = choice.get("delta") or {}
            if first_output is None and any(delta.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls")):
                first_output = time.perf_counter() - start
            slot["content"] += delta.get("content") or ""
            slot["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
            slot["finish_reason"] = choice.get("finish_reason") or slot["finish_reason"]
            for call in delta.get("tool_calls") or []:
                dst = slot["tool_calls"].setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
                dst["id"] += call.get("id") or ""
                fn = call.get("function") or {}
                dst["name"] += fn.get("name") or ""
                dst["arguments"] += fn.get("arguments") or ""
    if not done or not slots or any(c["finish_reason"] is None for c in slots.values()):
        raise ProtocolError("Incomplete SSE response: require choices, finish reasons and [DONE]")
    for slot in slots.values():
        slot["tool_calls"] = [{"id": c["id"], **tool_value(c["name"], c["arguments"])}
                              for _, c in sorted(slot["tool_calls"].items())]
    return {"choices": [c for _, c in sorted(slots.items())], "raw": raw,
            "first_output_s": first_output}


def chat(base_url, body, timeout=120):
    start = time.perf_counter()
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("BIELIK_TEST_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    url = base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if body.get("stream"):
            header_time = time.perf_counter() - start
            result = read_stream(response)
            if result["first_output_s"] is not None:
                result["first_output_s"] += header_time
        else:
            raw = json.load(response)
            if "error" in raw or not raw.get("choices"):
                raise ProtocolError("Empty or error completion response")
            result = {"choices": [normalize_message(c) for c in raw["choices"]], "raw": raw,
                      "first_output_s": None}
    result["elapsed_s"] = time.perf_counter() - start
    indices = [c["index"] for c in result["choices"]]
    count = body.get("n", 1)
    if (len(indices) != count or any(type(i) is not int for i in indices)
            or set(indices) != set(range(count))):
        raise ProtocolError("Server must return each requested choice index exactly once")
    return result


def check_choice(choice, expected):
    errors = []
    names = [c["name"] for c in choice["tool_calls"]]
    if "tools" in expected and names != expected["tools"]:
        errors.append(f"tools {names!r} != {expected['tools']!r}")
    if expected.get("min_tools", 0) > len(names):
        errors.append("not enough tool calls")
    if "arguments" in expected and [c["arguments"] for c in choice["tool_calls"]] != expected["arguments"]:
        errors.append("tool arguments differ from the requested values")
    if "reasoning" in expected and bool(choice["reasoning"].strip()) != expected["reasoning"]:
        errors.append("reasoning presence differs")
    if expected.get("content") and not choice["content"].strip():
        errors.append("empty final content")
    if expected.get("schema"):
        try:
            data = json.loads(choice["content"])
            if not isinstance(data, dict) or set(data) != {"answer"} or type(data["answer"]) is not int:
                errors.append("answer does not follow schema")
        except ValueError:
            errors.append("answer is not JSON")
    if choice["tool_calls"] and any(not c.get("id") for c in choice["tool_calls"]):
        errors.append("tool_call_id is missing")
    if "finish" in expected and choice["finish_reason"] not in expected["finish"]:
        errors.append(f"unexpected finish_reason={choice['finish_reason']}")
    if not expected.get("allow_markup") and any(tag in choice["content"] for tag in ("<think>", "</think>", "<tool_call>", "</tool_call>")):
        errors.append("control markup in final content")
    return errors
