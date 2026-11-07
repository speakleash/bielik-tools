#!/usr/bin/env python3
"""Native Pydantic AI agents against local Bielik Chat Completions."""
import argparse
import asyncio
import importlib.metadata
import json
import os
import re
from pathlib import Path
import sys
import threading

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.messages import (ModelRequest, ModelResponse, UserPromptPart, TextPart,
    ThinkingPart, ToolCallPart, ToolReturnPart, ModelMessagesTypeAdapter)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from synthetic_tools import SyntheticTools
from testing_api import thinking


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: int


EXPECTED = {"add": {"a": 17, "b": 23}, "weather": {"city": "Kraków"}}
CASES = {
    "plain": {"prompt": "Ile to17 razy23? Podaj wyłącznie liczbę.", "answer": "391", "tools": False},
    "single": {"prompt": "Wywołaj add z a=17 i b=23. Podaj wyłącznie wynik liczbowy.", "counts": {"add": 1}, "choice": "required", "answer": "40"},
    "parallel": {"prompt": "Wywołaj równolegle add z a=17 i b=23 oraz weather dla Krakowa. Podaj wyniki.", "counts": {"add": 1, "weather": 1}, "contains": [40, 21]},
    "named": {"prompt": "Wywołaj add z a=17 i b=23. Nie wywołuj innych narzędzi. Podaj wynik zwrócony przez narzędzie.", "counts": {"add": 1}, "choice": ["add"], "contains": [40]},
    "none": {"prompt": "Wywołaj add z a=17 i b=23.", "choice": "none", "counts": {}},
    "retry": {"prompt": "Sprawdź weather dla Krakowa. Jeśli retryable=true, ponów dokładnie raz. Podaj temperaturę.", "counts": {"weather": 2}, "retry": True, "contains": [21]},
    "schema": {"prompt": "Zwróć wynik17 razy23 w polu answer.", "schema": True, "tools": False},
    "history": {"prompt": "A ile wynosi poprzedni wynik pomnożony przez dwa? Podaj wyłącznie liczbę.", "history": True, "tools": False, "answer": "80"},
    "validation_retry": {"prompt": "Wywołaj add z a=17 i b=23. Przy komunikacie o błędzie walidacji ponów z tymi samymi argumentami. Podaj wynik narzędzia.", "counts": {"add": 1}, "validation_retry": True, "contains": [40]},
}


def validate_response(response):
    calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
    errors = []
    if response.finish_reason != ("tool_call" if calls else "stop"):
        errors.append("Unexpected or missing model finish_reason")
    ids = [p.tool_call_id for p in calls]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        errors.append("Missing or duplicate tool_call_id")
    for part in calls:
        try:
            arguments = part.args_as_dict()
            if part.tool_name not in EXPECTED or arguments != EXPECTED[part.tool_name]:
                errors.append("Raw tool arguments/name differ before framework coercion")
        except Exception as error:
            errors.append(f"Invalid raw tool arguments: {error}")
    return errors


def model_retry_returns(messages):
    return [part for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, ToolReturnPart)
            and part.outcome == "retried" and part.retry is not None
            and part.retry.cause == "model_retry"]


def validation_retry_observed(attempts, messages):
    return (len(attempts) == 2 and attempts[1]["turn"] > attempts[0]["turn"]
            and all(a["name"] == "add" and a["arguments"] == EXPECTED["add"] for a in attempts)
            and any(p.tool_call_id == attempts[0]["tool_call_id"] for p in model_retry_returns(messages)))


def scenario(base_url, model_name, family, name, enabled, streaming):
    case = CASES[name]
    ledger, attempts, errors = SyntheticTools(case.get("retry", False)), [], []
    lock = threading.Lock()
    provider = OpenAIProvider(base_url=base_url, api_key=os.environ.get("BIELIK_TEST_API_KEY", "dummy"))
    profile = OpenAIModelProfile(supports_tools=True, supports_json_schema_output=True,
        supports_forced_tool_choice=True, supports_forced_tool_choice_with_thinking=True,
        openai_chat_thinking_field="reasoning", openai_chat_send_back_thinking_parts=True,
        openai_chat_supports_allowed_tools=False)
    model = OpenAIChatModel(model_name, provider=provider, profile=profile)
    def settings(ctx):
        choice = case.get("choice", "auto") if not ledger.calls else "auto"
        return {"temperature": 0, "max_tokens": 512, "tool_choice": choice,
                "extra_body": {"chat_template_kwargs": thinking(family, enabled),
                    **({"thinking_token_budget": 64} if enabled else {})}}
    agent = Agent(model, output_type=NativeOutput(Answer, strict=True) if case.get("schema") else str,
                  model_settings=settings, retries=2, end_strategy="exhaustive")
    def execute(ctx, name, arguments):
        with lock:
            response = next(m for m in reversed(ctx.messages) if isinstance(m, ModelResponse))
            invalid = validate_response(response)
            if invalid:
                raise ValueError("Refusing dispatch: " + "; ".join(invalid))
            attempts.append({"name": name, "arguments": arguments,
                "turn": sum(isinstance(m, ModelResponse) for m in ctx.messages)-1,
                "tool_call_id": ctx.tool_call_id})
            if case.get("validation_retry") and len(attempts) == 1:
                raise ModelRetry("Synthetic validation error: retry add with exactly a=17,b=23")
            result = ledger.execute(name, arguments)
            ledger.calls[-1]["turn"] = sum(isinstance(m, ModelResponse) for m in ctx.messages)-1
            return result
    if case.get("tools", True):
        @agent.tool
        def add(ctx: RunContext[None], a: int, b: int) -> dict:
            """Dodaj dwie liczby całkowite."""
            return execute(ctx, "add", {"a": a, "b": b})
        @agent.tool
        def weather(ctx: RunContext[None], city: str) -> dict:
            """Pogoda dla miasta."""
            return execute(ctx, "weather", {"city": city})
    history = None
    if case.get("history"):
        history = [ModelRequest(parts=[UserPromptPart("Ile to17+23?")]),
                   ModelResponse(parts=[ThinkingPart("17+23=40", id="reasoning", provider_name="openai"),
                                        TextPart("40")], finish_reason="stop", provider_name="openai")]
    messages, output = [], None
    try:
        limits = UsageLimits(request_limit=6, tool_calls_limit=6)
        if streaming:
            async def streamed_run():
                # Drain each native model stream before dispatch: RunContext then
                # contains the complete raw response, including its finish reason.
                async with agent.iter(case["prompt"], message_history=history, usage_limits=limits) as run:
                    async for node in run:
                        if Agent.is_model_request_node(node):
                            async with node.stream(run.ctx) as events:
                                async for _ in events:
                                    pass
                    return run.result
            result = asyncio.run(streamed_run())
            output, messages = result.output, result.all_messages()
        else:
            result = agent.run_sync(case["prompt"], message_history=history, usage_limits=limits)
            output, messages = result.output, result.all_messages()
        generated = [m for m in messages[(len(history) if history else 0):] if isinstance(m, ModelResponse)]
        for response in generated:
            errors += validate_response(response)
        reasoning = [p.content for m in generated for p in m.parts if isinstance(p, ThinkingPart)]
        if bool(any(text.strip() for text in reasoning)) != enabled:
            errors.append("Reasoning presence differs or was lost by the adapter")
        if ledger.counts() != case.get("counts", {}):
            errors.append(f"Wrong execution counts: {ledger.counts()}")
        if name == "parallel" and not any(sum(isinstance(p, ToolCallPart) for p in m.parts) == 2 for m in generated):
            errors.append("No actual parallel tool calls")
        if case.get("retry"):
            calls = ledger.calls
            if len(calls) != 2 or calls[1]["turn"] <= calls[0]["turn"] or not calls[0]["result"].get("retryable"):
                errors.append("Retry did not follow an observed error")
        if case.get("validation_retry"):
            if not validation_retry_observed(attempts, messages):
                errors.append("Native ModelRetry path was not exercised")
        if case.get("schema"):
            if not isinstance(output, Answer) or output.answer != 391:
                errors.append("Wrong typed structured output")
        elif not isinstance(output, str) or not output.strip():
            errors.append("Empty final output")
        if case.get("answer") and str(output).strip().rstrip(".!") != case["answer"]:
            errors.append("Wrong final answer")
        if any(not re.search(r"(?<![\d-])" + re.escape(str(value)) + r"(?!\d|[.,]\d)",
                             str(output)) for value in case.get("contains", [])):
            errors.append("Final answer omitted or changed returned tool values")
        if isinstance(output, str) and any(t in output for t in ("<think>", "</think>", "<tool_call>", "</tool_call>")):
            errors.append("Markup leaked into final output")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {"case": name, "thinking": enabled, "stream": streaming, "passed": not errors,
            "errors": errors, "output": output.model_dump() if isinstance(output, BaseModel) else output,
            "messages": ModelMessagesTypeAdapter.dump_python(messages, mode="json"),
            "executions": ledger.calls, "attempts": attempts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    selected = args.only.split(",") if args.only else list(CASES)
    if not selected or set(selected)-set(CASES):
        ap.error("Unknown or empty scenario selection")
    rows = []
    for name in selected:
        for enabled in (False, True):
            for streaming in (False, True):
                row = scenario(args.base_url, args.model, args.family, name, enabled, streaming)
                rows.append(row)
                print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}/{enabled}/{streaming}: {row['errors']}", flush=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"pydantic_ai_version": importlib.metadata.version("pydantic-ai-slim"), "rows": rows}, output, ensure_ascii=False, indent=2)
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
