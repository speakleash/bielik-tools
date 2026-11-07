# Native GuideLLM load probe

Use an isolated Python3.12 environment. To avoid a second CUDA runtime:

```bash
uv venv /tmp/bielik-guidellm --python 3.12
uv pip install --python /tmp/bielik-guidellm/bin/python --torch-backend cpu \
  --only-binary=:all: -r tests/integrations/guidellm/requirements.txt
/tmp/bielik-guidellm/bin/python scripts/probe_guidellm.py --model bielik-test \
  --tokenizer /path/to/local/tokenizer --requests 8 --streams 1,4 \
  --output-dir /tmp/new-guidellm-results
```

Requires an already running local OpenAI-compatible server; it never launches
vLLM or downloads weights/tokenizers. Both reasoning modes run streamed plain
and client-tool workloads at each concurrency. Four scenario files, native
JSON reports, logs and a summary are retained in a new output directory.
The default executes8profiles×8model turns. Model turns are not conversations:
a client-tool call and its followup count separately. No warmup is excluded;
use longer runs for representative measurements. Fixed prompts/results can
benefit from prefix caching, so this is not an uncached throughput claim.
Record immutable checkpoint, vLLM, parser and template revisions with results.

GuideLLM injects the fixed dataset tool result {"value":40}; it does not execute
model-generated functions/code. Only add is provided. This measures native
client-tool scheduling, HTTP/token latency and throughput rather than tool
execution latency or semantic model accuracy. Full semantic/retry/ID checks
belong to the API/framework probes. This pinned version supports required/auto
tool choice in its data column; named tool forcing is tested elsewhere.
Reasoning history is disabled in GuideLLM because its enabled mode wraps
reasoning in literal think tags rather than our structured reasoning field.

Native CLI exit0 alone is insufficient: the wrapper also requires all expected
profiles and recorded requests, exact successful counts, zero errors/incomplete
requests, and actual expected tool calls plus followup output. Failures return1
and retain logs/reports. Every profile is bounded by180s; each subprocess has a
larger finite timeout. Small default runs are smoke tests, not stable p95 claims.
