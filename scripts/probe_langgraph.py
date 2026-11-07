#!/usr/bin/env python3
"""Native LangGraph tool loops with a pinned ChatOpenAI compatibility adapter."""
import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path
import sys
import threading

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from synthetic_tools import SyntheticTools
from testing_api import ADD, WEATHER, ANSWER_SCHEMA, check_choice, thinking


class ChatBielik(ChatOpenAI):
    """Preserve vLLM reasoning extras that stock ChatOpenAI deliberately drops.

    Test-only adapter for the pinned langchain-openai version. These private
    hooks must be re-tested when upgrading; this is not a universal SDK patch.
    """
    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw["choices"]):
            message = choice["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning"] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if result is not None and chunk.get("choices"):
            delta = chunk["choices"][0].get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning is not None:
                result.message.additional_kwargs["reasoning"] = reasoning
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        if len(messages) != len(payload["messages"]):
            raise ValueError("Cannot align reasoning history to serialized messages")
        for source, target in zip(messages, payload["messages"]):
            if isinstance(source, AIMessage) and source.additional_kwargs.get("reasoning"):
                target["reasoning"] = source.additional_kwargs["reasoning"]
        return payload


def normalized(message):
    if not isinstance(message.content, str):
        raise ValueError("Expected Chat Completions text content")
    if message.invalid_tool_calls:
        raise ValueError("LangChain received malformed tool arguments")
    return {"index": 0, "content": message.content,
            "reasoning": message.additional_kwargs.get("reasoning", ""),
            "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c["args"]}
                           for c in message.tool_calls],
            "finish_reason": message.response_metadata.get("finish_reason")}


CASES = {
    "plain": {"prompt": "Ile to 17 razy 23? Odpowiedz wyłącznie liczbą.", "answer": "391", "tools": False},
    "single": {"prompt": "Wywołaj add z a=17 i b=23. Podaj wyłącznie liczbę zwróconą przez narzędzie.", "counts": {"add": 1}, "answer": "40", "choice": "required"},
    "parallel": {"prompt": "Wywołaj równolegle add z a=17 i b=23 oraz weather dla Krakowa. Podaj wyniki.", "counts": {"add": 1, "weather": 1}, "contains": [40, 21]},
    "named": {"prompt": "Wywołaj add z a=17 i b=23. Nie wywołuj innych narzędzi. Podaj wynik zwrócony przez narzędzie.", "counts": {"add": 1}, "choice": {"type": "function", "function": {"name": "add"}}, "contains": [40]},
    "none": {"prompt": "Wywołaj add z a=17 i b=23.", "choice": "none", "counts": {}},
    "retry": {"prompt": "Sprawdź weather dla Krakowa. Jeśli retryable=true, ponów dokładnie raz. Podaj temperaturę.", "counts": {"weather": 2}, "retry": True, "contains": [21]},
    "schema": {"prompt": "Zwróć wynik 17 razy 23 w polu answer.", "schema": True, "tools": False},
    "history": {"prompt": "A ile wynosi poprzedni wynik pomnożony przez dwa? Podaj wyłącznie liczbę.", "history": True, "tools": False, "answer": "80"},
}


def validate_turn(turn, has_tools=True, choice="auto", enabled=False):
    calls = turn["tool_calls"]
    errors = check_choice(turn, {"finish": ["tool_calls" if calls else "stop"],
        **({"reasoning": False} if not enabled else {})})
    expected = {"add": {"a": 17, "b": 23}, "weather": {"city": "Kraków"}}
    if any(call["name"] not in expected or call["arguments"] != expected[call["name"]] for call in calls):
        errors.append("Raw model arguments do not match the requested tool schema/values")
    ids = [call.get("id") for call in calls]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate tool_call_id")
    if calls and (not has_tools or choice == "none"):
        errors.append("Tools called while disabled")
    if isinstance(choice, dict) and ([c["name"] for c in calls] != [choice["function"]["name"]]):
        errors.append("Named choice was not honored")
    return errors


def scenario(base_url, model, family, name, enabled, streaming):
    case = CASES[name]
    ledger, turns, errors = SyntheticTools(case.get("retry", False)), [], []
    lock = threading.Lock()
    def execute(name, arguments):
        with lock:
            value = ledger.execute(name, arguments)
            ledger.calls[-1]["turn"] = len(turns)-1
            return json.dumps(value, ensure_ascii=False)

    @tool
    def add(a: int, b: int) -> str:
        """Dodaj dwie liczby całkowite."""
        return execute("add", {"a": a, "b": b})

    @tool
    def weather(city: str) -> str:
        """Pogoda dla miasta."""
        return execute("weather", {"city": city})

    llm = ChatBielik(model=model, base_url=base_url,
        api_key=os.environ.get("BIELIK_TEST_API_KEY", "dummy"), temperature=0, max_tokens=512,
        use_responses_api=False, extra_body={"chat_template_kwargs": thinking(family, enabled),
            **({"thinking_token_budget": 64} if enabled else {})})
    has_tools = case.get("tools", True)
    def agent(state):
        active = llm
        selected = "auto"
        if has_tools:
            selected = case.get("choice", "auto") if not turns else "auto"
            active = active.bind_tools([ADD, WEATHER], tool_choice=selected, strict=False)
        if case.get("schema"):
            active = active.bind(response_format=ANSWER_SCHEMA)
        if streaming:
            message = None
            for chunk in active.stream(state["messages"]):
                message = chunk if message is None else message + chunk
            if message is None:
                raise ValueError("Empty LangChain stream")
        else:
            message = active.invoke(state["messages"])
        turn = normalized(message)
        turns.append(turn)
        turn_errors = validate_turn(turn, has_tools, selected, enabled)
        if turn_errors:
            errors.extend(turn_errors)
            raise ValueError("Invalid model response: refusing tool dispatch")
        return {"messages": [message]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([add, weather], handle_tool_errors=False))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    graph = builder.compile()
    initial = []
    if case.get("history"):
        initial = [HumanMessage("Ile to 17 + 23?"), AIMessage("40", additional_kwargs={"reasoning": "17+23=40"})]
    initial.append(HumanMessage(case["prompt"]))
    try:
        graph.invoke({"messages": initial}, {"recursion_limit": 12})
        if not turns:
            raise ValueError("No framework model responses")
        for turn in turns:
            errors += check_choice(turn, {} if enabled else {"reasoning": False})
        if enabled and not any(t["reasoning"].strip() for t in turns):
            errors.append("Reasoning was lost by the framework adapter")
        if ledger.counts() != case.get("counts", {}):
            errors.append(f"Wrong tool execution counts: {ledger.counts()}")
        expected = {"add": {"a": 17, "b": 23}, "weather": {"city": "Kraków"}}
        if any(c["arguments"] != expected[c["name"]] for c in ledger.calls):
            errors.append("Wrong tool arguments")
        final = turns[-1]
        errors += check_choice(final, {"tools": [], "content": True, "finish": ["stop"],
                                      **({"schema": True} if case.get("schema") else {})})
        if case.get("answer") and final["content"].strip().rstrip(".!") != case["answer"]:
            errors.append("Wrong final answer")
        if any(not re.search(r"(?<![\d-])" + re.escape(str(value)) + r"(?!\d|[.,]\d)",
                             final["content"]) for value in case.get("contains", [])):
            errors.append("Final answer omitted or changed returned tool values")
        if name == "parallel" and not any(len(t["tool_calls"]) == 2 for t in turns):
            errors.append("No actual parallel calls")
        if case.get("retry"):
            calls = ledger.calls
            if len(calls) != 2 or calls[1]["turn"] <= calls[0]["turn"] or not calls[0]["result"].get("retryable"):
                errors.append("Retry did not follow an observed error")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {"passed": not errors, "errors": errors, "case": name, "thinking": enabled,
            "stream": streaming, "turns": turns, "executions": ledger.calls}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--only", help="comma-separated scenario names")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    selected = args.only.split(",") if args.only else list(CASES)
    if not selected or set(selected) - set(CASES):
        ap.error("Unknown or empty scenario selection")
    rows = []
    for name in selected:
        for enabled in (False, True):
            for streaming in (False, True):
                row = scenario(args.base_url, args.model, args.family, name, enabled, streaming)
                rows.append(row)
                print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}/{enabled}/{streaming}: {row['errors']}", flush=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"versions": {p: importlib.metadata.version(p) for p in ("langgraph", "langchain-openai")},
                   "rows": rows}, output, ensure_ascii=False, indent=2)
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
