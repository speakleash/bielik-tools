#!/usr/bin/env python3
"""Replay a capture with reproducible random token partitions (no server/GPU)."""
import argparse
import json
from pathlib import Path
import random
import sys

from parser_replay import replay


def partition(length, seed, max_chunk=32):
    if length < 0 or max_chunk < 1:
        raise ValueError("length must be non-negative and max_chunk positive")
    rng = random.Random(seed)
    position, boundaries = 0, []
    while position < length:
        position = min(length, position + rng.randint(1, max_chunk))
        boundaries.append(position)
    return boundaries


def check(record, tokenizer, *, seed=0, trials=25, max_chunk=32):
    if trials < 1 or max_chunk < 1:
        raise ValueError("trials and max_chunk must be positive")
    length = len(record["output_token_ids"])
    reference = replay(record, tokenizer)
    rows = []
    # Include both extreme partitions alongside random ones.
    schedules = [("single-token", list(range(1, length + 1))),
                 ("whole-output", [length] if length else [])]
    schedules += [(f"seed-{seed + i}", partition(length, seed + i, max_chunk))
                  for i in range(trials)]
    for label, boundaries in schedules:
        row = {"partition": label, "boundaries": boundaries}
        try:
            result = replay(record, tokenizer, boundaries)
            row.update(passed=result == reference, result=result)
        except Exception as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
        rows.append(row)
    return {"passed": all(row["passed"] for row in rows), "non_stream": reference,
            "seed": seed, "trials": trials, "max_chunk": max_chunk, "partitions": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max-chunk", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    report = check(json.loads(args.capture.read_text()), tok, seed=args.seed,
                   trials=args.trials, max_chunk=args.max_chunk)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(f"{sum(r['passed'] for r in report['partitions'])}/{len(report['partitions'])} partitions match")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
