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
| ≥ 0.24.0 | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` |
| 0.15.0 – 0.23.x | `bielik_vllm_tool_parser_v0.15.0.py` | `bielik_vllm_reasoning_parser.py` |
| 0.13.0 – 0.14.x | `bielik_vllm_tool_parser_v0.13.0.py` | `bielik_vllm_reasoning_parser_v0.13.0.py` |
| ≤ 0.12.0 | `bielik_vllm_tool_parser_v0.12.0.py` | — |

The 0.24.0 tool-parser cut is the vLLM `ToolParser` API change
(`__init__(tokenizer, tools=…)` plus `supports_required_and_named`).
Reasoning parsers are unchanged across 0.15+.

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
[version-specific backports](https://github.com/speakleash/bielik-tools/pull/20)
before starting those servers to use the original v3.1 template. For 0.20.0
use the [native combined parser](https://github.com/speakleash/bielik-tools/pull/21)
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
either pair.

Then, run [reasoning\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/reasoning_streaming.py) to see how the model performs in reasoning mode.

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
