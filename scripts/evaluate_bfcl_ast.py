#!/usr/bin/env python3
"""Run a pinned BFCL Python AST-checker subset through Bielik's live API.

Never execute benchmark functions or model-generated Python. This does not
claim coverage of BFCL's executable/agentic/multiturn categories.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from testing_api import chat, thinking

VERSION = "2026.3.23"
CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple")
MODEL_KEY = "bielik-tools"


@lru_cache(maxsize=1)
def load_checker():
    if importlib.metadata.version("bfcl-eval") != VERSION:
        raise ValueError(f"Use audited bfcl-eval=={VERSION}")
    # Upstream's pure checker imports a vendor-model registry merely to
    # decide whether dots in function names are rewritten. Our local server
    # accepts the original names. Supply only that identity-name setting,
    # without importing all vendor SDKs or executable benchmark handlers.
    name = "bfcl_eval.constants.model_config"
    if name in sys.modules:
        raise ValueError("Run this narrow checker in a dedicated process")
    config = ModuleType(name)
    config.MODEL_CONFIG_MAPPING = {MODEL_KEY: SimpleNamespace(underscore_to_dot=False)}
    sys.modules[name] = config
    try:
        checker = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")
    finally:
        sys.modules.pop(name, None)
    from bfcl_eval.constants.enums import Language
    return checker, Language.PYTHON


def schema(value):
    if not isinstance(value, dict):
        raise ValueError("BFCL schema must be an object")
    kinds = {"dict": "object", "object": "object", "integer": "integer", "int": "integer",
             "float": "number", "number": "number", "boolean": "boolean", "bool": "boolean",
             "string": "string", "str": "string", "array": "array", "list": "array", "tuple": "array"}
    kind = value.get("type", "any")
    if kind != "any" and kind not in kinds:
        raise ValueError(f"Unsupported Python AST subset type: {kind!r}")
    result = {"type": kinds[kind]} if kind != "any" else {}
    for key in ("description", "enum", "required", "minimum", "maximum"):
        if key in value:
            result[key] = value[key]
    if "properties" in value:
        result["properties"] = {key: schema(item) for key, item in value["properties"].items()}
    if "items" in value:
        result["items"] = schema(value["items"])
    if "additionalProperties" in value:
        extra = value["additionalProperties"]
        result["additionalProperties"] = schema(extra) if isinstance(extra, dict) else extra
    return result


def data(category, offset, limit):
    if category not in CATEGORIES or offset < 0 or limit < 1:
        raise ValueError("Select an allowed category, non-negative offset and positive limit")
    import bfcl_eval
    root = Path(bfcl_eval.__file__).resolve().parent / "data"
    source = root / f"BFCL_v4_{category}.json"
    answers = root / "possible_answer" / source.name
    examples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    truth = {row["id"]: row["ground_truth"] for line in answers.read_text().splitlines()
             if line.strip() for row in [json.loads(line)]}
    selected = examples[offset:offset+limit]
    if not selected:
        raise ValueError("Selection contains no benchmark examples")
    if any(row["id"] not in truth for row in selected):
        raise ValueError("Missing ground truth")
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, answers)}
    return selected, truth, checksums


def score(example, calls, truth, category):
    if category not in CATEGORIES:
        raise ValueError("Executable/non-Python categories are not supported")
    checker, language = load_checker()
    outputs = []
    for call in calls:
        if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
            raise ValueError("Native tool calls must have a name and arguments object")
        outputs.append({call["name"]: call["arguments"]})
    return checker.ast_checker(example["function"], outputs, truth, language, category, MODEL_KEY)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--family", choices=["v30", "v31"], default="v31")
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--limit", type=int, default=5, help="examples per category")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--min-score", type=float, help="optional overall semantic accuracy gate")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    categories = args.categories.split(",")
    if (not categories or len(set(categories)) != len(categories) or set(categories)-set(CATEGORIES)
            or args.limit < 1 or args.offset < 0 or (args.min_score is not None and not 0 <= args.min_score <= 1)):
        ap.error("Invalid category/selection/threshold")
    checker, _ = load_checker()
    rows, checksums = [], {}
    for category in categories:
        examples, truth, hashes = data(category, args.offset, args.limit)
        checksums.update(hashes)
        for example in examples:
            if len(example["question"]) != 1:
                raise ValueError("This AST-only runner does not execute multi-turn fixtures")
            tools = [{"type": "function", "function": {"name": f["name"],
                      "description": f.get("description", ""), "parameters": schema(f["parameters"])}}
                     for f in example["function"]]
            for enabled in (False, True):
                for streaming in (False, True):
                    body = {"model": args.model, "messages": example["question"][0], "tools": tools,
                            "stream": streaming, "temperature": 0, "seed": 1701, "max_tokens": 768,
                            "chat_template_kwargs": thinking(args.family, enabled),
                            **({"thinking_token_budget": 128} if enabled else {})}
                    row = {"id": example["id"], "category": category, "thinking": enabled,
                           "stream": streaming, "correct": False, "request": body}
                    try:
                        response = chat(args.base_url, body)
                        row["response"] = response
                        choice = response["choices"][0]
                        verdict = score(example, choice["tool_calls"], truth[example["id"]], category)
                        row.update(response=response, verdict=verdict, correct=bool(verdict["valid"]))
                    except Exception as error:
                        row["run_error"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    print(f"{example['id']}/{enabled}/{streaming}: {row['correct']} {row.get('run_error', '')}", flush=True)
    accuracy = sum(r["correct"] for r in rows)/len(rows)
    run_errors = sum("run_error" in r for r in rows)
    gate = accuracy >= args.min_score if args.min_score is not None else None
    report = {"bfcl_version": VERSION, "checker_sha256": hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest(),
              "dataset_sha256": checksums, "model": args.model, "family": args.family,
              "offset": args.offset, "limit": args.limit, "accuracy": accuracy, "run_errors": run_errors,
              "quality_gate": args.min_score, "quality_passed": gate, "rows": rows}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(f"AST subset accuracy={accuracy:.3f}; run_errors={run_errors}; " +
          ("REPORT ONLY" if gate is None else f"quality_gate={gate}"))
    return 2 if run_errors else 1 if gate is False else 0


if __name__ == "__main__":
    sys.exit(main())
