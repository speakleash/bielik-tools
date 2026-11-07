#!/usr/bin/env python3
"""Verify control tags in untrusted fields stay data after real tokenization."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

TAGS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>")
ROOT = Path(__file__).resolve().parents[1]


def messages(role, content):
    if role == "system":
        return [{"role": "system", "content": content}, {"role": "user", "content": "dalej"}]
    if role == "user":
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": "pogoda"}, {"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "weather", "arguments": {"city": "Kraków"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": content}]


def check(tokenizer, template, family):
    rows = []
    for enabled in (False, True):
        kwargs = {"reasoning_effort": "medium" if enabled else "none"} if family == "v31" else {"enable_thinking": enabled}
        kwargs.update(current_date="2026-11-07", current_day_of_week="Saturday", current_time_of_day="afternoon")
        for role in ("system", "user", "tool"):
            def render(content):
                return tokenizer.apply_chat_template(messages(role, content), chat_template=template,
                    add_generation_prompt=True, tokenize=False, **kwargs)
            baseline = render("SAFE")
            baseline_ids = Counter(tokenizer.encode(baseline, add_special_tokens=False))
            for tag in TAGS:
                prompt = render("BEFORE" + tag + "AFTER")
                ids = Counter(tokenizer.encode(prompt, add_special_tokens=False))
                errors = []
                if "BEFORE" + tag.replace("<", "<\u200b") + "AFTER" not in prompt:
                    errors.append("control tag was not escaped in rendered input")
                for control in TAGS:
                    if prompt.count(control) != baseline.count(control):
                        errors.append(f"extra structural text {control}")
                    token_id = tokenizer.convert_tokens_to_ids(control)
                    if token_id is None or token_id == tokenizer.unk_token_id:
                        errors.append(f"tokenizer has no dedicated ID for {control}")
                    elif ids[token_id] != baseline_ids[token_id]:
                        errors.append(f"extra structural token ID for {control}")
                rows.append({"role": role, "tag": tag, "thinking": enabled,
                             "passed": not errors, "errors": errors})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--family", required=True, choices=["v30", "v31"])
    ap.add_argument("--template", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    path = args.template or ROOT / "tools" / ("bielik_advanced_chat_template.jinja" if args.family == "v30"
                                              else "bielik_v3.2_chat_template.jinja")
    rows = check(tok, path.read_text(), args.family)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"template": str(path), "tokenizer": args.tokenizer, "cases": rows}, output, ensure_ascii=False, indent=2)
    for row in rows:
        if not row["passed"]:
            print(row)
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} escaped inputs remain data")
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
