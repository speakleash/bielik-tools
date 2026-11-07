# LiteLLM SDK and proxy

Install optional pinned requirements in a separate environment. Start vLLM
with the unified parsers, matching template and served name `bielik-test`.

```bash
python scripts/probe_litellm.py --backend sdk --model bielik-test --family v31 \
  --base-url http://127.0.0.1:8000/v1 --output /tmp/litellm-sdk.json

export LITELLM_MASTER_KEY="sk-$(python -c 'import secrets; print(secrets.token_hex(32))')"
LITELLM_LOCAL_MODEL_COST_MAP=True LITELLM_TELEMETRY=False \
  litellm --config tests/integrations/litellm/proxy.yaml --host 127.0.0.1 --port 4000

BIELIK_TEST_API_KEY="$LITELLM_MASTER_KEY" python scripts/probe_litellm.py --backend proxy --model bielik-test --family v31 \
  --base-url http://127.0.0.1:4000/v1 --output /tmp/litellm-proxy.json
```

Each route runs the same40 direct API matrix cases and four retry tool loops
covering thinking off/on and full/streaming responses. Checks include exact
arguments, multiple choices, hidden reasoning, JSON Schema, named/required/
none tool choices, IDs, finish reasons and retry after an observed error.
Only allowlisted deterministic synthetic tools execute. Custom vLLM fields
go through `extra_body`; they must not be silently dropped by the gateway.

SDK iteration consumes the wire `[DONE]` marker internally. The adapter
therefore requires clean iterator completion plus completed choices rather
than pretending to observe the raw wire terminator. This test uses Chat
Completions; it makes no Responses API compatibility claim. Proxy server
version/config must be recorded by the operator separately from client
version. Local cost-map mode avoids an unnecessary external metadata fetch.

Use a fresh ephemeral proxy key; never commit it or disable the proxy's
master-key requirement. The example binds only to loopback.

Observed compatibility limitation with pinned LiteLLM1.108.2: all four
`n=2` streaming matrix variants can omit a choice's final status in SDK
chunks and proxy SSE, although direct vLLM API cases pass. The default full probe retains
these failures and exits1; it does not invent finish reasons or silently
skip those cases. Failed SDK streams retain their raw chunks for diagnosis.
