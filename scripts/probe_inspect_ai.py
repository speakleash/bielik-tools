#!/usr/bin/env python3
"""Native Inspect Tasks/solvers/tools/scorers against an existing local server."""
import argparse
import asyncio
import importlib.metadata
import json
import os
import re
from pathlib import Path
import sys

from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.model import (ChatMessageAssistant, ChatMessageUser, ContentReasoning,
                              GenerateConfig, ResponseSchema, get_model)
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.tool import ToolFunction, tool
from inspect_ai.util import JSONSchema

from synthetic_tools import SyntheticTools
from testing_api import ANSWER_SCHEMA, thinking

EXPECTED = {"add": {"a": 17, "b": 23}, "weather": {"city": "Kraków"}}
CASES = {
    "plain": {"prompt": "Ile to 17 razy 23? Podaj wyłącznie liczbę.", "answer": "391", "tools": False},
    "single": {"prompt": "Wywołaj add z a=17 i b=23. Podaj wyłącznie wynik liczbowy.", "counts": {"add": 1}, "choice": "any", "answer": "40"},
    "parallel": {"prompt": "Wywołaj równolegle add z a=17 i b=23 oraz weather dla Krakowa. Podaj wyniki.", "counts": {"add": 1, "weather": 1}, "contains": [40, 21]},
    "named": {"prompt": "Wywołaj add z a=17 i b=23. Nie wywołuj innych narzędzi. Podaj wynik zwrócony przez narzędzie.", "counts": {"add": 1}, "choice": "named", "contains": [40]},
    "none": {"prompt": "Wywołaj add z a=17 i b=23.", "choice": "none", "counts": {}},
    "retry": {"prompt": "Sprawdź weather dla Krakowa. Jeśli retryable=true, ponów dokładnie raz. Podaj temperaturę.", "counts": {"weather": 2}, "retry": True, "contains": [21]},
    "schema": {"prompt": "Zwróć wynik 17 razy 23 w polu answer.", "schema": True, "tools": False},
    "history": {"prompt": "A ile wynosi poprzedni wynik pomnożony przez dwa? Podaj wyłącznie liczbę.", "history": True, "tools": False, "answer": "80"},
}


def validate_calls(message, finish, disabled=False):
    calls = message.tool_calls or []
    errors = []
    if finish != ("tool_calls" if calls else "stop"):
        errors.append("Incorrect or missing finish reason")
    ids = [c.id for c in calls]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        errors.append("Missing or duplicate tool_call_id")
    if disabled and calls:
        errors.append("Tool dispatch is disabled")
    for call in calls:
        if call.parse_error or call.function not in EXPECTED or call.arguments != EXPECTED[call.function]:
            errors.append("Raw tool arguments/name invalid before dispatch")
    return errors


def reasoning(message):
    return "" if isinstance(message.content, str) else "".join(
        p.reasoning for p in message.content if isinstance(p, ContentReasoning))


def returned_values_present(text, values):
    return all(re.search(r"(?<![\d-])" + re.escape(str(value)) + r"(?!\d|[.,]\d)", text) for value in values)


def correlation_errors(messages):
    pending, errors = set(), []
    for message in messages:
        if message.role == "assistant":
            if pending:
                errors.append("Missing tool results before next assistant turn")
            pending = {c.id for c in message.tool_calls or []}
        elif message.role == "tool":
            if message.tool_call_id not in pending:
                errors.append("Unknown or duplicated tool result id")
            else:
                pending.remove(message.tool_call_id)
    if pending:
        errors.append("Unresolved tool results")
    return errors


def build_task(name, enabled, family):
    case = CASES[name]
    ledger, state_ref, dispatch_errors = SyntheticTools(case.get("retry", False)), [], []
    def execute(name, arguments):
        state = state_ref[0]
        message = state.output.message
        invalid = validate_calls(message, state.output.choices[0].stop_reason,
                                 disabled=case.get("choice") == "none")
        if invalid:
            dispatch_errors.extend(invalid)
            raise ValueError("Refusing invalid tool dispatch")
        value = ledger.execute(name, arguments)
        ledger.calls[-1]["turn"] = sum(m.role == "assistant" for m in state.messages)-1
        return json.dumps(value, ensure_ascii=False)

    @tool(name="add")
    def add_tool():
        async def execute_add(a: int, b: int) -> str:
            """Dodaj dwie liczby całkowite.

            Args:
                a: Pierwsza liczba.
                b: Druga liczba.
            """
            return execute("add", {"a": a, "b": b})
        return execute_add

    @tool(name="weather")
    def weather_tool():
        async def execute_weather(city: str) -> str:
            """Pogoda dla miasta.

            Args:
                city: Nazwa miasta.
            """
            return execute("weather", {"city": city})
        return execute_weather

    @solver
    def native_loop():
        async def solve(state, generate):
            state_ref[:] = [state]
            state.tools = [add_tool(), weather_tool()] if case.get("tools", True) else []
            state.tool_choice = ToolFunction("add") if case.get("choice") == "named" else case.get("choice", "auto")
            state = await generate(state, tool_calls="single")
            if state.messages[-1].role == "tool":
                # Explicit initial-call forcing must not force an endless
                # agent loop. Subsequent generations may return the answer.
                state.tool_choice = "auto"
                state_ref[:] = [state]
                state = await generate(state, tool_calls="loop")
            return state
        return solve

    @scorer(metrics=[accuracy()])
    def contract_score():
        async def score(state, target):
            errors = list(dispatch_errors)
            generated = [m for m in state.messages if isinstance(m, ChatMessageAssistant)]
            if case.get("history"):
                generated = generated[1:]
            if not generated:
                errors.append("No model response")
            if bool(any(reasoning(m).strip() for m in generated)) != enabled:
                errors.append("Reasoning presence differs or was lost")
            for message in generated:
                # Stop metadata is available for the final response in state;
                # intermediate statuses were checked before each tool dispatch.
                for call in message.tool_calls or []:
                    if call.parse_error or call.function not in EXPECTED or call.arguments != EXPECTED[call.function]:
                        errors.append("Invalid raw call in transcript")
                if any(t in message.text for t in ("<think>", "</think>", "<tool_call>", "</tool_call>")):
                    errors.append("Markup in assistant text")
            errors += validate_calls(state.output.message, state.output.choices[0].stop_reason,
                                     disabled=case.get("choice") == "none")
            errors += correlation_errors(state.messages)
            if ledger.counts() != case.get("counts", {}):
                errors.append(f"Wrong execution counts: {ledger.counts()}")
            if any(c["arguments"] != EXPECTED[c["name"]] for c in ledger.calls):
                errors.append("Incorrect execution arguments")
            if name == "parallel" and not any(len(m.tool_calls or []) == 2 for m in generated):
                errors.append("No actual parallel tool calls")
            if case.get("retry"):
                calls = ledger.calls
                if len(calls) != 2 or calls[1]["turn"] <= calls[0]["turn"] or not calls[0]["result"].get("retryable"):
                    errors.append("Retry did not follow an observed error")
            final = state.output.message.text.strip()
            if not final or state.output.message.tool_calls:
                errors.append("Missing final answer")
            if case.get("answer") and final.rstrip(".!") != case["answer"]:
                errors.append("Wrong final answer")
            if not returned_values_present(final, case.get("contains", [])):
                errors.append("Final answer omitted or changed returned tool values")
            if case.get("schema"):
                try:
                    value = json.loads(final)
                except ValueError:
                    value = None
                if value != {"answer": 391} or type(value.get("answer") if isinstance(value, dict) else None) is not int:
                    errors.append("Wrong structured answer")
            return Score(value=CORRECT if not errors else INCORRECT, explanation=json.dumps(errors, ensure_ascii=False))
        return score

    messages = []
    if case.get("history"):
        messages = [ChatMessageUser(content="Ile to 17 + 23?"), ChatMessageAssistant(content=[
            ContentReasoning(internal="reasoning", reasoning="17+23=40"),
            {"type": "text", "text": "40"}])]
    messages.append(ChatMessageUser(content=case["prompt"]))
    config = GenerateConfig(temperature=0, max_tokens=512, max_retries=0, timeout=120,
        reasoning_history="all", extra_body={"chat_template_kwargs": thinking(family, enabled),
            **({"thinking_token_budget": 64} if enabled else {})})
    if case.get("schema"):
        config.response_schema = ResponseSchema(name="answer", strict=True,
            json_schema=JSONSchema.model_validate(ANSWER_SCHEMA["json_schema"]["schema"]))
    task = Task(dataset=[Sample(input=messages, id=name)], solver=native_loop(), scorer=contract_score(),
                config=config, message_limit=16, turn_limit=6, fail_on_error=False,
                name="bielik_" + name, metadata={"thinking": enabled})
    return task, ledger


async def evaluate(args):
    rows = []
    for name in args.selected:
        for enabled in (False, True):
            for streaming in (False, True):
                task, ledger = build_task(name, enabled, args.family)
                model = get_model("openai-api/bielik/" + args.model, base_url=args.base_url,
                    api_key=os.environ.get("BIELIK_TEST_API_KEY", "dummy"), memoize=False,
                    responses_api=False, stream=streaming, strict_tools=False)
                logs = await eval_async(task, model=model, log_dir=str(args.log_dir), max_samples=1,
                                        log_model_api=False, log_realtime=False, retry_on_error=0)
                samples = logs[0].samples or [] if logs else []
                passed = bool(logs) and logs[0].status == "success" and len(samples) == 1
                passed = passed and samples[0].error is None and bool(samples[0].scores)
                passed = passed and all(s.value == CORRECT for s in samples[0].scores.values())
                row = {"case": name, "thinking": enabled, "stream": streaming, "passed": bool(passed),
                    "log": logs[0].location if logs else None, "executions": ledger.calls,
                    "samples": [s.model_dump(mode="json") for s in samples]}
                rows.append(row)
                print(f"[{'PASS' if passed else 'FAIL'}] {name}/{enabled}/{streaming}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--log-dir", type=Path)
    args = ap.parse_args(argv)
    args.selected = args.only.split(",") if args.only else list(CASES)
    if not args.selected or set(args.selected)-set(CASES):
        ap.error("Unknown or empty scenario selection")
    args.log_dir = args.log_dir or args.output.parent / "inspect-logs"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(evaluate(args))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"inspect_ai_version": importlib.metadata.version("inspect-ai"), "rows": rows}, output, ensure_ascii=False, indent=2)
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
