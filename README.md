# bielik-tools

A collection of supplementary tools for working with **Bielik** language models.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/speakleash/bielik-tools.git
cd bielik-tools
```

Install the client dependencies for the basic examples (tool calling, reasoning
and structured outputs) in your Python environment:

```bash
python -m venv .venv-client
source .venv-client/bin/activate
python -m pip install -r examples/requirements.txt
```

Install vLLM separately in the serving environment, selecting a release and
hardware build appropriate for your machine. The client examples can run in a
separate environment without vLLM or model weights.

Configure the basic examples to use the model name exposed by your server:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=speakleash/Bielik-11B-v3.0-Instruct
export VLLM_API_KEY=EMPTY  # replace if the server requires a key
```

`VLLM_MODEL` must match the served model ID (`--served-model-name` when used).
The examples otherwise default to `step72`, which is intended for local testing.
Environment variables take precedence over optional `.env` files read by
`examples/_vllm_env.py`. CrewAI uses its separate configuration below.

All server commands below assume the repository root is the working directory.

## Structured Outputs

Bielik models have been trained to generate structured outputs. Once the model is running in [vLLM](https://github.com/vllm-project/vllm), you can try the [structured_output.py](https://github.com/speakleash/bielik-tools/blob/main/examples/structured_output.py) example to generate structured outputs using OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) APIs.

## Tool Calling

To use function/tool calling, you need to enable the extended chat template. This can be done using the provided [advanced chat template](https://github.com/speakleash/bielik-tools/blob/main/tools/bielik_advanced_chat_template.jinja) and [tool parser](https://github.com/speakleash/bielik-tools/blob/main/tools/bielik_vllm_tool_parser.py). Start vLLM with the following command:

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --chat-template ./tools/bielik_advanced_chat_template.jinja
```

Choose the tool parser file matching your vLLM version:

| vLLM version | Tool parser | Reasoning parser |
|---|---|---|
| ≥ 0.19.0, except 0.20.0 | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` |
| 0.20.0 | [native combined plugin](compatibility/vllm-0.20.md) | same combined plugin |
| 0.15.0 – 0.18.x | `bielik_vllm_tool_parser_v0.15.0.py` | `bielik_vllm_reasoning_parser.py` |
| 0.13.0 – 0.14.x | `bielik_vllm_tool_parser_v0.13.0.py` | `bielik_vllm_reasoning_parser_v0.13.0.py` |
| 0.12.0 | `bielik_vllm_tool_parser_v0.12.0.py` | `bielik_vllm_reasoning_parser_v0.13.0.py` |
| < 0.12.0 | `bielik_vllm_tool_parser_v0.12.0.py` | — |

For vLLM 0.20.0 use the native combined plugin linked above; no vLLM runtime
patch is needed. For template v3.2 the supported target is vLLM **0.24.0+**
(the modern runtime checks cover 0.24.0 and 0.32.0); legacy 0.20 adapters are
not a v3.2 support path.

The 0.19.0 tool-parser cut is the vLLM `ToolParser` API change
(`__init__(tokenizer, tools=…)`).
The default tool and reasoning parsers support both the older
`openai.engine.protocol` and newer `generate.base.protocol` module layouts.
Parser imports, construction and non-streaming extraction are tested on vLLM
0.32.0; historical constructor signatures were checked against the upstream
0.15.0–0.24.0 source. Import, construction and extraction smoke tests also pass
using the v0.12, v0.13, v0.14, v0.15, v0.18, v0.19 and v0.20 source trees
with shared PyTorch/CUDA dependencies (Transformers 4.57.6). The v0.24 source
also passes with Transformers 5.19.0; vLLM 0.24 requires Transformers 5 and
rejects Transformers 4. Reasoning is tested from v0.13.
This is not an end-to-end serving guarantee for every intermediate version.
Legacy parser files remain available; the <=0.12 variant also handles the
relocation of `MistralTokenizer` with a fallback to its original location.
The reasoning parser with a `_v0.13.0` suffix also works on vLLM 0.12.0.
For older vLLM releases, explicitly send `skip_special_tokens: false` when
using reasoning, especially in non-streaming requests.
Versions below 0.12 retain the legacy tool parser but are not covered by the
GPU validation matrix.

Then, run [tool\_calling.py](https://github.com/speakleash/bielik-tools/blob/main/examples/tool_calling.py) or [tool\_calling\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/tool_calling_streaming.py) to see how tool calling works in practice.

## Template and parser variants

The commands in this README use the advanced template and the parser files
without a `_v3.1` suffix. The `_v3.1.py` parsers are a separate pair for the
v3.1 chat template; they do not select or install that template automatically.
Use `--chat-template` to supply the matching template explicitly. Load only one
tool parser and one reasoning parser: the variants register the same name,
`bielik`.

The compatibility table describes vLLM API variants, not Bielik model versions.
For v3.2 the support target is vLLM 0.24.0+; legacy adapters are not a v3.2
backport path.
Unpatched vLLM 0.14.0 and 0.15.0 overwrite the template `reasoning_effort` kwarg
with the request-level value, including `None`. Apply the
[version-specific backports](compatibility/README.md)
before starting those servers to use the original v3.1 template. For 0.20.0
use the [native combined parser](compatibility/vllm-0.20.md)
instead of patching vLLM. The advanced
template remains an available legacy option, not a required workaround.
On 24 GB GPUs, the BF16-to-FP8 load path in vLLM 0.13–0.15 can also exceed
VRAM temporarily. A checkpoint already serialized in FP8 avoids that peak;
reducing the KV-cache allocation alone does not remove the weight-loading peak.

## Reasoning

Reasoning is available in Bielik 11B v2.5 Instruct and Bielik v3.0 Instruct models. Enabling reasoning allows the model to better handle complex questions by expanding its reasoning capabilities. To try it out, start vLLM with the following command:

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --chat-template ./tools/bielik_advanced_chat_template.jinja \
    --reasoning-parser-plugin ./tools/bielik_vllm_reasoning_parser.py \
    --reasoning-parser bielik
```

For the advanced template, request thinking explicitly in the API call:
`chat_template_kwargs={"enable_thinking": true}` (JSON; use `True` in Python).
Send `skip_special_tokens: false` alongside `chat_template_kwargs` in the
request body (`extra_body` with the OpenAI Python client). Older vLLM releases
can strip `</think>` before non-streaming parsing otherwise, merging the final
answer into reasoning. The example sets this flag explicitly.

The reasoning parser separates generated reasoning from the final answer;
selecting a parser alone does not add the advanced template's thinking prompt.
The v3.1 pair instead supports `reasoning_effort` in `chat_template_kwargs`.
`reasoning_streaming.py` sends both settings consistently, so it works with
either pair. Its sample conversation exercises `none`, `medium`, then `none`.
The client accepts both `reasoning` and `reasoning_content` response fields,
including chunks that also contain final-answer content. If generation ends
with reasoning only, it retains that field and warns that no final answer
arrived; increase the token budget or check the template/parser pairing
instead of treating reasoning as the answer.

Then, run [reasoning\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/reasoning_streaming.py) to see how the model performs in reasoning mode.

## llama.cpp (GGUF)

GGUF builds of Bielik can be served with `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) using the same advanced chat template. Official GGUF repositories include [speakleash/Bielik-11B-v3.0-Instruct-GGUF](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF) and [speakleash/Bielik-4.5B-v3.0-Instruct-GGUF](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF).

The commands below are for recent llama.cpp builds. Flag names follow the current [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md); older builds may not support all of them.

```bash
llama-server \
    -hf speakleash/Bielik-11B-v3.0-Instruct-GGUF:Q4_K_M \
    --jinja \
    --chat-template-file ./bielik-tools/tools/bielik_advanced_chat_template.jinja \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --port 8080
```

Use `-m /path/to/model.gguf` instead of `-hf` to load a local file. `--chat-template-file` overrides the template embedded in the GGUF metadata. `--jinja` enables the Jinja template engine, which tool calling requires; recent builds enable it by default.

### Tool calling

With `--jinja`, `llama-server` exposes OpenAI-style function calling on `/v1/chat/completions` ([function calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)). Send `tools` as you would to the OpenAI API, or point the [tool\_calling.py](https://github.com/speakleash/bielik-tools/blob/main/examples/tool_calling.py) example at the server:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "Jaka jest pogoda w Krakowie?"}],
      "tools": [{
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }]
    }'
```

- llama.cpp selects the tool-call parser from the template; if it does not recognise the format, it falls back to a generic handler (the log shows `Chat format: Generic`).
- Multiple tool calls in one turn are disabled by default; send `"parallel_tool_calls": true` to enable them.
- The template has a stricter instruction for `tool_choice == "required"`, but llama.cpp does not pass the request's `tool_choice` field to the template. To render that instruction, also send `"chat_template_kwargs": {"tool_choice": "required"}`.

### Reasoning

The template reads the `enable_thinking` variable. When it is `true`, a Polish instruction to reason inside `<think>…</think>` is appended to the system prompt. When it is `false`, an empty `<think></think>` block is pre-filled. llama.cpp always sets this variable, and with the default `--reasoning auto` its value depends on how llama.cpp analyses the template. Set it explicitly instead:

- for the whole server: `--chat-template-kwargs '{"enable_thinking": true}'`, or `--reasoning off` to disable thinking;
- per request: `"chat_template_kwargs": {"enable_thinking": true}`.

`--reasoning-format` controls how the thoughts are returned. `deepseek` puts them in `message.reasoning_content`, and `none` leaves them unparsed in `message.content`.

## Ollama

Ollama templates use Go `text/template` syntax, not Jinja ([Ollama template docs](https://github.com/ollama/ollama/blob/main/docs/template.mdx)), so `bielik_advanced_chat_template.jinja` cannot be used with Ollama as is. You can run a GGUF directly from Hugging Face:

```bash
ollama run hf.co/speakleash/Bielik-11B-v3.0-Instruct-GGUF:Q4_K_M
```

In this case Ollama picks a chat template from a list of commonly used templates, based on the `tokenizer.chat_template` metadata in the GGUF ([Hugging Face docs](https://huggingface.co/docs/hub/ollama)). The tool calling and `enable_thinking` behaviour described above is not guaranteed there. For tool calling and reasoning, use vLLM or llama.cpp with the template from this repository.

## Multi-Agent with CrewAI

Install the additional CrewAI example dependencies:

```bash
python -m pip install -r examples/requirements-crewai.txt
```

For this example to work you need Tavily API key. Create `.env` file with contents:

```
BASE_URL=http://0.0.0.0:8000/v1
MODEL_NAME=path_or_hf_repo
API_KEY=token-abc123
TAVILY_API_KEY=tvly-apikey123123123123
```

Run vllm

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --chat-template ./tools/bielik_advanced_chat_template.jinja \
    --port 8000 --host 0.0.0.0 \
    --api-key token-abc123
```

Then, run [crewai_to_file.py](https://github.com/speakleash/bielik-tools/blob/main/examples/crewai_to_file.py)
Final contents of the report will be placed in `bielik_output/atrakcje.md`

### v3.1 streaming tool calls

`tools/bielik_vllm_tool_parser_v3.1.py` buffers a tag-wrapped tool call until
`</tool_call>`, then emits its name and complete JSON arguments together. This
makes the result independent of stream chunk boundaries, including a complete
call or several calls in one chunk. Text outside calls remains incremental.
Malformed completed blocks are preserved as text rather than executed; incomplete
calls are not emitted as tools. Tool arguments are delayed until the closing tag
instead of being exposed as potentially incomplete JSON fragments.

Parser-only regression tests (vLLM environment plus pytest; no weights or model
server needed):

```bash
python -m pytest -q tests/test_streaming_tool_parser_v31.py
```
