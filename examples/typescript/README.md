# Bielik v3.1 from TypeScript

Small, runnable TypeScript examples for **Bielik 11B v3.1 Instruct** behind
any OpenAI-compatible server: vLLM, llama.cpp `llama-server` or Ollama. They
use the official [`openai`](https://www.npmjs.com/package/openai) package plus
a few helpers in [`src/client.ts`](src/client.ts); there is no extra SDK.

| Example | File | Shows |
|---|---|---|
| `npm run chat` | [`src/chat.ts`](src/chat.ts) | one request, one answer |
| `npm run streaming` | [`src/streaming.ts`](src/streaming.ts) | printing tokens as they arrive |
| `npm run tools` | [`src/tool-calling.ts`](src/tool-calling.ts) | a full tool loop (call → run → result → answer) with a Polish prompt |
| `npm run structured` | [`src/structured-output.ts`](src/structured-output.ts) | `response_format: json_schema`, with validation |
| `npm run reasoning -- medium` | [`src/reasoning.ts`](src/reasoning.ts) | v3.1 `reasoning_effort` and the separate reasoning stream |

Requires Node.js 20+ (or Bun).

```bash
cd examples/typescript
npm install
npm test          # offline unit tests, no server needed
npm run typecheck
```

## Configuration

| Variable | Fallback | Default |
|---|---|---|
| `BIELIK_BASE_URL` | `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `BIELIK_MODEL` | `VLLM_MODEL` | `speakleash/Bielik-11B-v3.1-Instruct` |
| `BIELIK_API_KEY` | `VLLM_API_KEY` | `EMPTY` |

The `VLLM_*` names are the ones used by the Python examples in this
repository, so one shell setup works for both.

## Starting a server

### vLLM

Use the command from the model card (vLLM 0.24+, tested upstream on 0.32), run
from the repository root:

```bash
vllm serve speakleash/Bielik-11B-v3.1-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --reasoning-parser-plugin ./tools/bielik_vllm_reasoning_parser.py \
    --reasoning-parser bielik \
    --exclude-tools-when-tool-choice-none \
    --max-model-len 32768

export BIELIK_BASE_URL=http://127.0.0.1:8000/v1
export BIELIK_MODEL=speakleash/Bielik-11B-v3.1-Instruct
```

### llama.cpp

The GGUF ships the v3.1 chat template; `--jinja` (the default in current
builds) makes `llama-server` use it, and `--reasoning-format deepseek` returns
thinking in `reasoning_content`.

```bash
llama-server -m Bielik-11B-v3.1-Instruct.Q8_0.gguf --alias bielik-11b-v3.1 \
    --jinja --reasoning-format deepseek -c 16384 --port 8081

export BIELIK_BASE_URL=http://127.0.0.1:8081/v1
export BIELIK_MODEL=bielik-11b-v3.1
```

### Ollama

```bash
ollama pull SpeakLeash/bielik-11b-v3.1-instruct:Q8_0

export BIELIK_BASE_URL=http://127.0.0.1:11434/v1
export BIELIK_MODEL=SpeakLeash/bielik-11b-v3.1-instruct:Q8_0
```

## Reasoning (`reasoning_effort`)

Bielik v3.1 thinks only when asked: `reasoning_effort` is `none` (default),
`low`, `medium`, `high` or `max`. The v3.0 `enable_thinking` flag is ignored
by the v3.1 template. `withReasoning()` sends the level in two places:

```ts
client.chat.completions.create(withReasoning({ model, messages, max_tokens: 4096 }, "high"));
// → { ..., reasoning_effort: "high", chat_template_kwargs: { reasoning_effort: "high" } }
```

- **vLLM 0.32**: both fields reach the template and the Bielik reasoning parser.
- **llama.cpp**: either field works (the template receives the level).
- **Ollama**: only the top-level `reasoning_effort` works; `chat_template_kwargs`
  is ignored. In Ollama's native `/api/chat`, pass the level as a string,
  `think: "high"`. `think: true` does **not** turn on v3.1 thinking: it renders
  the same prompt as `none`.

Read the thinking with `getReasoning()`; vLLM and Ollama call the field
`reasoning`, llama.cpp `reasoning_content`. `max_tokens` covers thinking plus
the answer, so give it room (the example uses 4096). If a response has
reasoning but empty `content`, the limit was too low; do not treat the
reasoning as the answer.

## What was checked, per backend

Run on 14 Nov 2026 on an Apple M5 MacBook Pro (128 GB) with
`SpeakLeash/bielik-11b-v3.1-instruct:Q8_0` in Ollama 0.40.14 and the same GGUF
in the `llama-server` bundled with it (llama.cpp 0.8.0-dev, `39febd2ad`):

| Feature | Ollama 0.40.14 | llama.cpp | vLLM |
|---|---|---|---|
| chat, streaming | ✅ | ✅ | not run here |
| tool loop (2 parallel calls, then an answer) | ✅ | ✅ | not run here |
| `reasoning_effort` low–max | ✅ top-level only | ✅ | not run here |
| `response_format` (`json_schema`, `json_object`) | ❌ HTTP 400 | ❌ HTTP 400 | not run here |

vLLM was not run (no CUDA GPU on the test machine); its column relies on the
model card and the Python examples in this repository.

**Structured output on llama.cpp and Ollama.** With the v3.1 GGUF, any
`response_format` (even `{"type": "json_object"}`) fails with
`400 Failed to initialize samplers: std::exception`. The server log shows the
grammar it builds for the chat template starts with the literal text
`<|im_start|>assistant\n`, which the sampler cannot initialise. A plain
`/completion` request with `json_schema` works on the same server, and Bielik
v3.0 GGUFs accept `format: "json"` in Ollama, so the problem is in how
llama.cpp derives the grammar from the v3.1 chat template rather than in the
weights. `structured-output.ts` therefore falls back to asking for JSON in a
system prompt and validating the result; that output is **not** constrained.

Other observations: with `reasoning_effort: "none"` the answer usually starts
with a newline (the examples `trim()` it), and the built-in system prompt of
the v3.1 template adds about 1,000 prompt tokens to every request (disable it
with `chat_template_kwargs: { provider_system_prompt: false }` on vLLM or
llama.cpp; Ollama ignores `chat_template_kwargs`).

## Tests

```bash
npm test                     # offline: a fake fetch plays the server
BIELIK_BASE_URL=... BIELIK_MODEL=... npm run test:live   # chat, stream, tool call against a real server
```

The offline tests cover configuration, the reasoning fields, merging streamed
tool-call fragments by index, argument parsing, and the tool loop including
unknown tools, invalid JSON arguments and a round limit.

## Po polsku w skrócie

Przykłady w TypeScripcie dla Bielika 11B v3.1 przez API zgodne z OpenAI
(vLLM, llama.cpp, Ollama): czat, streaming, wywoływanie narzędzi z pętlą,
JSON Schema i `reasoning_effort`. W Ollamie myślenie włącza tylko
`reasoning_effort` na najwyższym poziomie żądania (albo `think: "high"` w
natywnym API, nie `think: true`). `response_format` z plikiem GGUF v3.1 nie
działa obecnie ani w llama.cpp, ani w Ollamie (błąd 400), więc przykład
przechodzi wtedy na JSON z promptu i go waliduje.
