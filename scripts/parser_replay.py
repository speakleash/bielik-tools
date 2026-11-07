#!/usr/bin/env python3
"""Capture vLLM token IDs, then replay the exact output through local parsers.

No generated tool is executed. Replay is offline and requires vLLM 0.32 and
the same local tokenizer used when capturing. Captures contain full prompts
and outputs: use synthetic data and keep private captures outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request
from functools import lru_cache
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1


def fingerprint(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    data = {"vocab": tokenizer.get_vocab(),
            "backend": backend.to_str() if backend is not None else None,
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {})}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True,
                                    default=str).encode()).hexdigest()


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "tools").glob("bielik_vllm_*parser.py"))}


def decoded_prefixes(tokenizer, prompt_ids, ids, spaces_between_special_tokens=True,
                     suppress_terminal_stop=False):
    """Use vLLM's actual prompt-primed fast detokenizer, including UTF-8 buffering."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

    if not hasattr(tokenizer, "_tokenizer"):
        raise ValueError("Replay requires a fast tokenizer, as used by this vLLM setup")
    params = SamplingParams(skip_special_tokens=False,
                            spaces_between_special_tokens=spaces_between_special_tokens)
    detokenizer = FastIncrementalDetokenizer(tokenizer, SimpleNamespace(
        request_id="bielik-replay", sampling_params=params, prompt_token_ids=prompt_ids))
    prefixes = []
    for index, token_id in enumerate(ids):
        detokenizer.update([token_id], stop_terminated=(
            suppress_terminal_stop and index == len(ids) - 1))
        prefixes.append(detokenizer.output_text)
    return prefixes


def validate_stop_options(request):
    # Custom stops can truncate text *inside* the last token. A capture of
    # IDs alone cannot reproduce that contract without the server's settings.
    if (request.get("stop") or request.get("stop_token_ids")
            or request.get("include_stop_str_in_output") or request.get("ignore_eos")):
        raise ValueError("Custom stop/stop_token_ids/include_stop_str_in_output/ignore_eos are not supported")


def validate_capture(record):
    if record.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported capture format_version")
    if not isinstance(record.get("request"), dict):
        raise ValueError("Capture needs a request object")
    if not isinstance(record.get("tokenizer_sha256"), str):
        raise ValueError("Capture needs a tokenizer fingerprint")
    for name in ("prompt_token_ids", "output_token_ids"):
        ids = record.get(name)
        if (not isinstance(ids, list) or (name == "prompt_token_ids" and not ids)
                or any(type(i) is not int or i < 0 for i in ids)):
            raise ValueError(f"Capture needs integer {name} (prompt must not be empty)")
    if not isinstance(record.get("output_text"), str):
        raise ValueError("Capture needs output_text")
    return record


def capture(base_url, request, tokenizer, timeout=120):
    validate_stop_options(request)
    if request.get("stream") or request.get("n", 1) != 1:
        raise ValueError("Capture currently requires stream=false and n=1")
    body = {**request, "stream": False, "return_token_ids": True,
            "skip_special_tokens": False}
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("BIELIK_TEST_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    endpoint = base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    req = urllib.request.Request(endpoint, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    if len(raw.get("choices", [])) != 1:
        raise ValueError("Expected one completion choice")
    choice = raw["choices"][0]
    if isinstance(choice.get("stop_reason"), str):
        raise ValueError("Server used a stop string; use a server with EOS-only stopping")
    ids = choice.get("token_ids")
    if not isinstance(ids, list) or not isinstance(raw.get("prompt_token_ids"), list):
        raise ValueError("Server must return output and prompt token IDs (vLLM 0.32)")
    # vLLM suppresses the terminal stop token's text, but still passes its
    # ID to the parser. Preserve both aspects rather than deleting the ID.
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    if isinstance(choice.get("stop_reason"), int) and choice["stop_reason"] not in eos_ids:
        raise ValueError("Server used a custom stop token; use EOS-only stopping")
    suppressed = bool(ids and ids[-1] in eos_ids
                      and choice.get("finish_reason") in ("stop", "tool_calls"))
    prefixes = decoded_prefixes(tokenizer, raw["prompt_token_ids"], ids,
                                body.get("spaces_between_special_tokens", True), suppressed)
    record = {"format_version": FORMAT_VERSION, "request": body,
              "tokenizer_sha256": fingerprint(tokenizer),
              "prompt_token_ids": raw["prompt_token_ids"],
              "output_token_ids": ids, "terminal_stop_suppressed": suppressed,
              "output_text": prefixes[-1] if prefixes else "",
              "finish_reason": choice.get("finish_reason"),
              "server_response": raw, "parser_sha256": source_hashes(),
              "runtime": {"python": sys.version}}
    for package in ("vllm", "transformers", "tokenizers"):
        try:
            record["runtime"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record["runtime"][package] = None
    return validate_capture(record)


@lru_cache(maxsize=1)
def load_parser_classes():
    classes = []
    for kind in ("reasoning", "tool"):
        path = ROOT / "tools" / f"bielik_vllm_{kind}_parser.py"
        spec = importlib.util.spec_from_file_location(f"bielik_replay_{kind}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes.append(getattr(module, f"Bielik{kind.title()}Parser"))
    return classes


def parser_for(record, tokenizer):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.abstract_parser import DelegatingParser

    reasoning, tool = load_parser_classes()
    class Combined(DelegatingParser):
        reasoning_parser_cls = reasoning
        tool_parser_cls = tool

    request = ChatCompletionRequest(**record["request"])
    # Use the server's own precedence, including explicit enable_thinking.
    kwargs = request.build_chat_params(None, "auto").chat_template_kwargs
    parser = Combined(tokenizer, request.tools, chat_template_kwargs=kwargs)
    request = parser.adjust_request(request)
    return parser, request


def normalized(reasoning, content, calls):
    return {"reasoning": reasoning or "", "content": content or "",
            "tool_calls": [{"name": c.name, "arguments": json.loads(c.arguments)}
                           for c in calls or []]}


def replay(record, tokenizer, boundaries=None):
    """Return a normalized result; boundaries are cumulative token offsets.

    None selects non-streaming. A supplied partition must cover every token
    exactly once. Token IDs are never reconstructed by re-encoding text.
    """
    validate_capture(record)
    validate_stop_options(record["request"])
    if fingerprint(tokenizer) != record["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint differs from the capture")
    ids = record["output_token_ids"]
    prefixes = decoded_prefixes(tokenizer, record["prompt_token_ids"], ids,
                                record["request"].get("spaces_between_special_tokens", True),
                                record.get("terminal_stop_suppressed", False))
    if (prefixes[-1] if prefixes else "") != record["output_text"]:
        raise ValueError("Captured text does not match decoding its token IDs")
    parser, request = parser_for(record, tokenizer)
    prompt = record["prompt_token_ids"]
    if boundaries is None:
        parser.is_reasoning_end(prompt)
        return normalized(*parser.parse(record["output_text"], request,
                          enable_auto_tools=bool(request.tools), model_output_token_ids=ids))
    if not ids and boundaries == []:
        delta = parser.parse_delta("", [], request, prompt_token_ids=prompt, finished=True)
        if delta and (delta.content or delta.reasoning or delta.tool_calls):
            raise ValueError("Parser produced output for an empty completion")
        return {"reasoning": "", "content": "", "tool_calls": []}
    if (not boundaries or boundaries[-1] != len(ids)
            or any(type(b) is not int for b in boundaries)
            or any(a >= b for a, b in zip([0, *boundaries], boundaries))):
        raise ValueError("Boundaries must strictly increase from 0 through the token count")
    previous, position, reasoning, content, calls = "", 0, "", "", {}
    for end in boundaries:
        current = prefixes[end - 1]
        if not current.startswith(previous):
            raise ValueError("Tokenizer decoding is not prefix-stable")
        delta = parser.parse_delta(current[len(previous):], ids[position:end], request,
                                   prompt_token_ids=prompt, finished=end == len(ids))
        if delta:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
            for call in delta.tool_calls or []:
                slot = calls.setdefault(call.index, {"name": "", "arguments": ""})
                if call.function:
                    slot["name"] += call.function.name or ""
                    slot["arguments"] += call.function.arguments or ""
        previous, position = current, end
    return {"reasoning": reasoning, "content": content,
            "tool_calls": [{"name": c["name"], "arguments": json.loads(c["arguments"])}
                           for _, c in sorted(calls.items())]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["capture", "replay"])
    ap.add_argument("--tokenizer", required=True, help="local tokenizer directory")
    ap.add_argument("--input", required=True, type=Path, help="request JSON or capture JSON")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    data = json.loads(args.input.read_text())
    if args.mode == "capture":
        result = capture(args.base_url, data, tokenizer)
    else:
        result = {}
        for mode, boundaries in (("non_stream", None),
                ("stream", list(range(1, len(data["output_token_ids"]) + 1)))):
            try:
                result[mode] = replay(data, tokenizer, boundaries)
            except Exception as error:
                # Preserve malformed parser output as a failed artifact; never
                # turn invalid arguments into a callable normalized payload.
                result[mode] = {"error": f"{type(error).__name__}: {error}"}
        result["parity"] = (not any("error" in result[mode] for mode in ("non_stream", "stream"))
                            and result["non_stream"] == result["stream"])
    # Never silently replace a previous failure artifact.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    return 1 if args.mode == "replay" and not result["parity"] else 0


if __name__ == "__main__":
    sys.exit(main())
