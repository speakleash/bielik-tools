# Native Inspect AI evaluation harness

```bash
python scripts/probe_inspect_ai.py --model bielik-test --family v31 \
  --output /tmp/inspect.json --log-dir /tmp/inspect-logs
```

Install the optional pinned requirements. This creates real Inspect Tasks,
Samples, solvers, tools, scorers and `.eval` logs. Eight scenarios cover
plain, required/single, parallel, named, none, retry, native schema and
history with reasoning off/on and explicit full/streaming model transport.
It uses the generic Chat Completions provider with an explicit local URL;
it never starts another vLLM process or downloads weights.

Native `generate` executes tools. A single-call phase applies initial
forcing, then auto mode permits the final answer and subsequent retry loop.
Only the two allowlisted synthetic tools can execute. Raw arguments, IDs
and termination are checked before dispatch; scorer checks tool-result ID
correlation, execution counts, reasoning preservation and final output.
Limits bound messages, turns, retries and API timeouts. There is no external
judge model or service, and API-header logging is disabled.

Success requires a successful evaluation, exactly one scored non-error
sample and all custom scores correct. Empty/missing scores cannot pass.
Artifacts retain native sample transcripts and paths to Inspect logs.
