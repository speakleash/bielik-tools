#!/usr/bin/env python3
"""Check that a model's chat template, tokenizer and reasoning parser agree.

Renders the real template with the real tokenizer (transformers, so the
``{% generation %}`` tag works), then asks the Bielik reasoning parser whether
reasoning has already ended in the prompt, as vLLM does for every request.
A mismatch means the parser will put the answer into ``reasoning`` (or the
reasoning into ``content``) for that request type.

  python scripts/check_parser_template_contract.py ~/repos/models/steps_70_90_110 --family v31
  python scripts/check_parser_template_contract.py ~/repos/models/steps_70_90_110 --family v31 \\
      --template tools/bielik_v3.2_chat_template.jinja
  python scripts/check_parser_template_contract.py speakleash/Bielik-11B-v3.0-Instruct \\
      --family v30 --template tools/bielik_advanced_chat_template.jinja

Needs vLLM (for the parser base classes) and the tokenizer files; no GPU.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MSG = [{"role": "user", "content": "Ile to 2+2?"}]
HISTORY = [{"role": "user", "content": "Ile to 2+2?"},
           {"role": "assistant", "content": "4", "reasoning": "2+2=4"},
           {"role": "user", "content": "A razy 3?"}]

# (name, chat_template_kwargs as vLLM passes them, messages, thinking expected)
CASES = {
    "v31": [
        ("default (effort none)", {}, MSG, False),
        ("reasoning_effort=high", {"reasoning_effort": "high"}, MSG, True),
        ("top-level reasoning_effort=high", {"reasoning_effort": "high", "enable_thinking": True}, MSG, True),
        ("top-level reasoning_effort=none", {"reasoning_effort": "none", "enable_thinking": False}, MSG, False),
        ("legacy enable_thinking=true", {"enable_thinking": True}, MSG, False),
        ("history + reasoning_effort=medium", {"reasoning_effort": "medium"}, HISTORY, True),
    ],
    "v30": [
        ("enable_thinking unset", {}, MSG, False),
        ("enable_thinking=false", {"enable_thinking": False}, MSG, False),
        ("enable_thinking=true", {"enable_thinking": True}, MSG, True),
        ("top-level reasoning_effort=high", {"reasoning_effort": "high", "enable_thinking": True}, MSG, True),
        ("history + enable_thinking=true", {"enable_thinking": True}, HISTORY, True),
    ],
}


def load_parser():
    path = ROOT / "tools" / "bielik_vllm_reasoning_parser.py"
    spec = importlib.util.spec_from_file_location("bielik_reasoning_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BielikReasoningParser


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model directory or Hub ID (tokenizer files only)")
    ap.add_argument("--family", choices=sorted(CASES), required=True)
    ap.add_argument("--template", help="override the tokenizer's chat template")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = Path(args.template).read_text() if args.template else None
    parser_cls = load_parser()
    think_ids = [tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
    specials = set(tok.all_special_ids)
    print(f"{args.model}: <think>/</think> ids={think_ids} in_all_special_ids={[i in specials for i in think_ids]}")

    failures = 0
    for name, kwargs, messages, thinking in CASES[args.family]:
        prompt = tok.apply_chat_template(messages, chat_template=template, add_generation_prompt=True,
                                         tokenize=False, **kwargs)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        parser = parser_cls(tok, chat_template_kwargs=kwargs)
        # vLLM shows the prompt to the parser first; the parser may then
        # switch to content when the template already closed <think>.
        ended = parser.is_reasoning_end(ids)
        parser_thinks = type(parser._parser).__name__ != "IdentityReasoningParser"
        prompt_open = parser_thinks and not ended
        # Template side, independent of the parser: an unclosed <think> in the
        # final assistant turn means the model starts inside reasoning.
        turn = prompt[prompt.rfind("<|im_start|>"):]
        template_open = turn.rfind("<think>") > turn.rfind("</think>")
        # The parser should treat output as reasoning exactly when the prompt
        # leaves the model inside (or about to open) a <think> block, and a
        # thinking-off prompt must never leave <think> open.
        ok = (parser_thinks == thinking and (not parser_thinks or prompt_open)
              and (thinking or not template_open))
        failures += not ok
        tail = prompt[-40:].replace("\n", "\\n")
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: parser_thinking={parser_thinks} "
              f"prompt_reasoning_open={prompt_open} expected_thinking={thinking} tail={tail!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
