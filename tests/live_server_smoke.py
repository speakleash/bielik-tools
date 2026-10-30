"""Manually test a running Bielik vLLM server (real model and GPU required).

Uses synthetic prompts and a simulated tool result; no external tools execute.
Requires the matching chat template and tool/reasoning plugins on the server.
Example: python tests/live_server_smoke.py --model step72 --base-url http://127.0.0.1:8000
"""

import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="bielik-matrix")
parser.add_argument("--output-dir", type=Path, default=Path("/tmp/bielik-live-smoke"))
parser.add_argument("--skip-reasoning", action="store_true")
parser.add_argument("--timeout", type=float, default=600, help="Seconds per request, including cold kernel preparation")
args = parser.parse_args()
base = args.base_url.rstrip("/")
if not base.endswith("/v1"):
    base += "/v1"
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Dodaj dwie liczby.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]
results = []


def call(label, payload, expect):
    start = time.monotonic()
    record = {"case": label, "request": payload}
    try:
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            if payload.get("stream"):
                chunks = []
                content = ""
                reasoning = ""
                tc = {}
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
                    if not chunk.get("choices"):
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    content += delta.get("content") or ""
                    reasoning += (
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    for item in delta.get("tool_calls") or []:
                        row = tc.setdefault(
                            item["index"],
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if item.get("id"):
                            row["id"] = item["id"]
                        f = item.get("function") or {}
                        row["function"]["name"] += f.get("name") or ""
                        row["function"]["arguments"] += f.get("arguments") or ""
                message = {
                    "content": content,
                    "reasoning": reasoning,
                    "tool_calls": list(tc.values()),
                }
                record["chunks"] = chunks
            else:
                record["response"] = json.load(response)
                message = record["response"]["choices"][0]["message"]
        record["message"] = message
        if expect == "plain":
            assert message.get("content") and not message.get("tool_calls"), message
        elif expect == "reasoning":
            assert message.get("content"), message
            assert message.get("reasoning") or message.get("reasoning_content"), message
        elif expect == "tool":
            calls = message.get("tool_calls") or []
            assert len(calls) == 1, calls
            assert calls[0]["function"]["name"] == "add", calls
            assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}, (
                calls
            )
        record["passed"] = True
    except Exception as exc:
        record["passed"] = False
        record["error"] = repr(exc)
        if hasattr(exc, "read"):
            record["error_body"] = exc.read().decode()
    record["seconds"] = round(time.monotonic() - start, 2)
    (out / f"{label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    results.append(
        {
            key: value
            for key, value in record.items()
            if key in ["case", "passed", "error", "seconds", "error_body"]
        }
    )
    print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return record.get("message")


def payload(text, effort="none", stream=False):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "skip_special_tokens": False,
        "max_tokens": 1024 if effort != "none" else 384,
        "stream": stream,
        "chat_template_kwargs": {
            "reasoning_effort": effort,
            "enable_thinking": effort != "none",
        },
    }


call(
    "plain", payload("Odpowiedz jednym słowem: jak nazywa się stolica Polski?"), "plain"
)
if not args.skip_reasoning:
    call(
        "reasoning",
        payload("Oblicz 17 razy 23. Krótko sprawdź wynik i podaj odpowiedź.", "low"),
        "reasoning",
    )
    call(
        "reasoning-stream",
        payload(
            "Oblicz 13 razy 19. Krótko sprawdź wynik i podaj odpowiedź.", "low", True
        ),
        "reasoning",
    )
for stream in [False, True]:
    p = payload(
        "Użyj narzędzia add, aby dodać 2 i 3. Nie obliczaj wyniku samodzielnie.",
        stream=stream,
    )
    p.update(tools=tools, tool_choice="auto")
    message = call("tools-stream" if stream else "tools-auto", p, "tool")
    if not stream and message and message.get("tool_calls"):
        p["messages"] += [
            dict(
                role="assistant",
                content=message.get("content"),
                tool_calls=message["tool_calls"],
            ),
            {
                "role": "tool",
                "tool_call_id": message["tool_calls"][0]["id"],
                "content": "5",
            },
        ]
        p["tool_choice"] = "none"
        call("tool-followup", p, "plain")
p = payload("Dodaj 2 i 3 przy użyciu narzędzia.")
p.update(tools=tools, tool_choice="required")
p["chat_template_kwargs"]["tool_choice"] = "required"
call("tools-required", p, "tool")
(out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

raise SystemExit(0 if all(row["passed"] for row in results) else 1)
