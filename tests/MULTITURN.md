# Tool loops and per-request state isolation

```bash
python tests/live_multiturn_isolation.py --model bielik --family v31 \
  --output /tmp/multiturn.json
```

This stacks on the shared API matrix client. It exercises eight scenarios:
parallel calls with tool results deliberately returned in reverse ID order,
a deterministic retryable tool failure (full and streaming), and four
conversations concurrently on the same server with different arguments,
streaming modes and thinking switches.

Only `add` and `weather` are executable. `scripts/synthetic_tools.py` checks
exact argument keys/types, returns fixed synthetic values and records each
execution. It never evaluates model text, invokes a shell, accesses files or
contacts external services. There is no generic name-to-function lookup.
Each conversation owns a separate ledger and a bounded five-turn loop.

Assertions require actual parallel calls, exact execution counts, successful
final answers containing the returned values, reasoning isolation and no
cross-request argument leakage. Artifacts preserve every request, response
and execution; an error is a failing result rather than an omitted case.
