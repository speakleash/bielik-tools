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

The example uses `response_format={"type": "json_schema", ...}` for chat
requests. The legacy `guided_json` field is ignored by modern vLLM.
Its reasoning example requires vLLM **0.32** with a matching reasoning parser:
it sends `thinking_token_budget=256` in `extra_body` and `max_tokens=800`.
The total output limit includes reasoning, so a separate thinking budget
reserves room for the final JSON. Without it a model can use even a large
output limit inside `<think>`, leaving `content` empty with
`finish_reason="length"`. Increasing the total limit alone need not help.
See vLLM's [thinking budget parameter](https://docs.vllm.ai/en/v0.32.0/api/vllm/sampling_params/).

If generation reaches the total limit, treat its JSON as potentially incomplete.
Keep reasoning separate; do not copy it into `content` as a substitute answer.

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

The advanced template escapes control tags (`<|im_start|>`, `<|im_end|>`,
`<think>`, `</think>`, `<tool_call>`, `</tool_call>`) in system, user and tool
text by inserting a zero-width space after `<`, so these tags in pasted or
tool-returned text cannot start a new turn or role, or become a tool call.
Other special tokens of the tokenizer (for example `</s>`) are not escaped. Assistant
history and the template's own thinking instructions are not changed. Pass
`chat_template_kwargs={"sanitize_input": false}` to restore the raw behaviour.

## Template and parser variants

The commands in this README use the advanced template.
`bielik_vllm_tool_parser.py` and `bielik_vllm_reasoning_parser.py` serve every
Bielik chat template (the advanced template for 3.0 and the v3.1/v3.2
templates). Parsers do not select or install a template; use
`--chat-template` to supply the matching template explicitly. Load only one
tool parser and one reasoning parser: the variants register the same name,
`bielik`.

The per-version legacy tool parsers (`_v0.12.0`, `_v0.13.0`, `_v0.15.0` and the
native 0.20 plugin) are frozen. They keep the earlier `required` constraint,
which allows a single call and forbids `<` inside arguments, and they ignore a
named `tool_choice`. These limitations were not re-tested on those vLLM
versions; use vLLM 0.24+ with `bielik_vllm_tool_parser.py` where possible.

Migration: `tools/bielik_vllm_tool_parser_v3.1.py` and
`tools/bielik_vllm_reasoning_parser_v3.1.py` were removed. Point
`--tool-parser-plugin` / `--reasoning-parser-plugin` at
`tools/bielik_vllm_tool_parser.py` / `tools/bielik_vllm_reasoning_parser.py`
instead; with the old paths vLLM only logs the import error and then fails at
startup because the `bielik` parsers are not registered.

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

## Chat template for Bielik 3.1 and 3.2

`tools/bielik_v3.2_chat_template.jinja` is the chat template of Bielik 3.1 and
3.2. Pass it with `--chat-template` when the checkpoint does not ship it.
Thinking is switched with `reasoning_effort` (`none|low|medium|high|max`,
default `none`); `enable_thinking` is ignored. The template rejects a
`tool_choice` in `chat_template_kwargs` other than `required` or `none`. It
escapes control tags in system, user and tool text, but keeps assistant
content and reasoning raw by default, as the models were trained
(`sanitize_assistant_content` / `sanitize_assistant_reasoning` enable
escaping). Reasoning from earlier turns is dropped unless
`think_render_option="preserved"`.

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
The v3.1/v3.2 templates switch thinking with `reasoning_effort`
(`none|low|medium|high|max`) in `chat_template_kwargs`, or with the top-level
`reasoning_effort` field on vLLM 0.32. The reasoning parser follows
`reasoning_effort` when present and `enable_thinking` otherwise. If the
template has already closed `<think></think>` for the current assistant turn
(for example `enable_thinking` sent to a v3.1/v3.2 template, which ignores it),
the output is returned as content. Only the current assistant turn is
inspected, so `<think>` examples in a system prompt or in earlier turns do not
change the result. With `n > 1` in a non-streaming request, vLLM 0.32 only
tells the first choice's parser about the prompt, and with
`include_reasoning: false` it does not show the prompt to the parser at all; in
both cases the answer of an `enable_thinking` request to a v3.1/v3.2 template
ends up in reasoning (or is dropped). Use `reasoning_effort` with the v3.1/v3.2
templates. Top-level values outside `none|low|medium|high|max`
(for example `minimal`) are rejected by the v3.1/v3.2 templates with HTTP 400.
`reasoning_streaming.py` sends both settings consistently, so it works with
either template. Its sample conversation exercises `none`, `medium`, then `none`.
The client accepts both `reasoning` and `reasoning_content` response fields,
including chunks that also contain final-answer content. If generation ends
with reasoning only, it retains that field and warns that no final answer
arrived; increase the token budget or check the template/parser pairing
instead of treating reasoning as the answer.

Then, run [reasoning\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/reasoning_streaming.py) to see how the model performs in reasoning mode.

## Tests

Unit tests for the tool parser live in `tests/`. They import vLLM, so run them in an environment where vLLM (>= 0.24) is installed; no GPU or model download is needed (a fake tokenizer is used):

```bash
pip install pytest
python -m pytest tests -q
```

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

### Request options that do not reach the chat template

vLLM 0.32 passes only `chat_template_kwargs` (plus `reasoning_effort` and a few
generation flags such as `add_generation_prompt`) to the chat template. The request-level
`tool_choice` and `response_format` are enforced by the parser and vLLM, but
the template does not see them, so its instructions for them are not added.
To add those instructions, repeat the values in `chat_template_kwargs`, for
example `{"tool_choice": "required"}`.

With `tool_choice="none"` the tools are still rendered into the prompt, and
the model may then write a tool call as plain text. On vLLM 0.32 (checked)
start the server with `--exclude-tools-when-tool-choice-none` to drop the
tools from such requests.

Tool calls are recognised from the text `<tool_call>…</tool_call>`. In
non-streaming responses this includes tags that the model spelled with
ordinary tokens, for example when it copies such text from the user or from a
document; in streaming responses outside `required`/named choices only the
dedicated `<tool_call>` token starts a call. Treat tool arguments as untrusted
input either way.

### Streaming tool calls and forced tool choice

For `tool_choice="required"`, the tool parser constrains the final output to
tool-call blocks from the start, after any reasoning. A prose lead-in cannot
consume the token budget before the first call. `auto` mode still permits
ordinary text around calls. The complete call must fit `max_tokens`; an
arbitrarily small limit can still truncate JSON. If reasoning is enabled,
also reserve room for calls with a thinking budget on vLLM 0.32. The grammar
starts after reasoning ends, so `required` with thinking needs the reasoning
parser (`--reasoning-parser bielik`); without it a forced call cannot be
preceded by a `<think>` block.

A named `tool_choice` (`{"type": "function", "function": {"name": ...}}`)
forces exactly one call of that function. The constraint covers the function
name and JSON syntax, not the function's parameter schema, and arguments may
nest at most 4 levels (a warning is logged for deeper schemas). The response
reports `finish_reason="tool_calls"`, whereas vLLM's built-in parsers and the
OpenAI API report `"stop"` for a forced function; check `tool_calls` rather
than `finish_reason`.

`tools/bielik_vllm_tool_parser.py` buffers a tag-wrapped tool call until
`</tool_call>`, then emits its name and complete JSON arguments together. This
makes the result independent of stream chunk boundaries, including a complete
call or several calls in one chunk. Text outside calls remains incremental.
Malformed completed blocks are preserved as text rather than executed; incomplete
calls are not emitted as tools. Tool arguments are delayed until the closing tag
instead of being exposed as potentially incomplete JSON fragments.

Parser-only regression tests (vLLM environment plus pytest; no weights or model
server needed):

```bash
python -m pytest -q tests/test_streaming_tool_parser_v31.py tests/test_streaming_tool_parser_default.py
```

## Test framework compatibility

With a running Bielik server, prepare isolated dependencies and run all framework probes:

```bash
python scripts/test_frameworks.py --all --setup --model bielik-test
```

Later runs need only `python scripts/test_frameworks.py --all --model bielik-test`.
The runner prints a combined result and saves one `summary.json`, retaining
native logs. See [framework testing](tests/integrations/README.md) for server
settings, subsets and separate BFCL/GuideLLM workloads. Known LiteLLM n=2
streaming failures remain failures rather than being silently skipped.
