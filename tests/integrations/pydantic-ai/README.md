# Native Pydantic AI agents

```bash
python scripts/probe_pydantic_ai.py --model bielik-test --family v31 \
  --output /tmp/pydantic-ai.json
```

Install the optional pinned requirements in an isolated environment. Nine
scenarios run with thinking off/on and full/streamed model output (36 runs):
plain, required/single, parallel, named, none, tool-error retry, native JSON
Schema typed output, message history and native ModelRetry validation repair.
The Agent performs tool orchestration itself; this does not simulate its
loop with a separate HTTP client. Request/tool limits bound every run.

The profile explicitly declares the capabilities of our local Chat
Completions server and preserves `reasoning` in history. It does not infer
OpenAI-specific capabilities from the arbitrary served-model name. Raw
arguments, IDs and termination are checked before executing allowlisted
synthetic tools, so framework type coercion cannot mask malformed model
output. Only add/weather execute; no shell/files/external services are used.
Typed output uses NativeOutput rather than a hidden output-tool workaround.

Streaming uses native Agent.iter and drains each model-request node before
native tool dispatch. This retains the full raw response and finish reason
for validation; run_stream can expose tool calls before that response is in
RunContext.messages. Parallel SSE/length refusal are covered offline.
