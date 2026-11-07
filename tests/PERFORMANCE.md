# User-visible API latency and throughput

```bash
python scripts/benchmark_bielik_api.py --model bielik --family v31 \
  --manifest /tmp/server-manifest.json --concurrency 1,4 --repeats 5 \
  --output /tmp/performance.json
```

The required operator-supplied manifest pins `model_revision`,
`vllm_version`, `parser_commit` and `template_sha256` (64hex characters).
Also include quantization, GPU, max-model-len, server command and cache
settings when comparing runs. A local weight-file checksum identifies an
unversioned checkpoint more reliably than a served-model alias.

Each concurrency level measures the same fixed mix of plain, reasoning and
tool requests after warmup rounds covering all three workloads. Reports retain every request/response,
token usage, errors, p50/p95 response latency, median first visible output,
successful request throughput and successful-output token throughput over total
wall time. This includes queueing/prefill and is not engine decode tokens/s.
First-visible-output latency includes parser buffering: a tool call may not
be visible until its arguments are complete. Raw engine TTFT is different.
Token counts come from server usage, never text length estimates.

Prefix caching remains at the server's configured setting; fixed prompts
can hit it. Treat this as a warm synthetic smoke workload, not a production
capacity estimate. Errors are retained and return exit2, not removed from
the denominator. No generated tools execute. These measurements need a
local live server and are not part of default CPU CI.
