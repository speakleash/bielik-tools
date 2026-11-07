# Live Chat Completions boundary matrix

```bash
python tests/live_api_matrix.py --model bielik --family v31 \
  --base-url http://127.0.0.1:8000/v1 --output /tmp/api-matrix.json
```

Use `--family v30` for the advanced template. Start vLLM with the matching
template and unified tool/reasoning parsers. The matrix makes 40 requests:
thinking on/off, full/streaming, four tool-choice modes, JSON Schema,
multiple choices, reasoning hidden/included, top-level reasoning effort,
explicit skip-special flags and a one-token required-call budget.

`n=2` uses seeded sampling (`temperature=0.7`), because vLLM rejects multiple
choices with greedy sampling. Assertions check every returned choice. An
extremely short budget may return unfinished markup as data; the invariant
there is `finish_reason=length` and no executable tool call, not a completed
answer. Normal-budget final answers must not contain control markup.

`scripts/testing_api.py` is a shared standard-library client for later live
integrations. It accumulates fragmented tool names/arguments and separate
choices, requires SSE `[DONE]` and finish reasons, catches stream errors,
and rejects non-object tool arguments. `first_output_s` is user-visible time
to the first content/reasoning/tool delta, not raw engine token latency.

Use `BIELIK_TEST_API_KEY` for authentication. Artifacts contain full synthetic
requests/responses, never the authorization header; existing output files
are not overwritten. `--only` accepts exact case names and rejects typos.
No generated tool is executed and there is no external service dependency.
