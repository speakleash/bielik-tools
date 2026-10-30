# bielik-tools

A collection of supplementary tools for working with **Bielik** language models.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/speakleash/bielik-tools.git
cd bielik-tools
```

Explore available tools and examples in their respective subdirectories.

## Structured Outputs

Bielik models have been trained to generate structured outputs. Once the model is running in [vLLM](https://github.com/vllm-project/vllm), you can try the [structured_output.py](https://github.com/speakleash/bielik-tools/blob/main/examples/structured_output.py) example to generate structured outputs using OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) APIs.

## Tool Calling

To use function/tool calling, you need to enable the extended chat template. This can be done using the provided [advanced chat template](https://github.com/speakleash/bielik-tools/blob/main/tools/bielik_advanced_chat_template.jinja) and [tool parser](https://github.com/speakleash/bielik-tools/blob/main/tools/bielik_vllm_tool_parser.py). Start vLLM with the following command:

```bash
vllm serve Bielik-11B-v2.5-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./bielik-tools/tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --chat-template ./bielik-tools/tools/bielik_advanced_chat_template.jinja
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

## Reasoning

Reasoning is available in Bielik 11B v2.5 Instruct and Bielik v3.0 Instruct models. Enabling reasoning allows the model to better handle complex questions by expanding its reasoning capabilities. To try it out, start vLLM with the following command:

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --chat-template ./bielik-tools/tools/bielik_advanced_chat_template.jinja \
    --reasoning-parser-plugin ./bielik-tools/tools/bielik_vllm_reasoning_parser.py \
    --reasoning-parser bielik
```

Then, run [reasoning\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/reasoning_streaming.py) to see how the model performs in reasoning mode.

## Multi-Agent with CrewAI

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
    --tool-parser-plugin ./bielik-tools/tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --chat-template ./bielik-tools/tools/bielik_advanced_chat_template.jinja \
    --port 8000 --host 0.0.0.0 \
    --api-key token-abc123
```

Then, run [crewai_to_file.py](https://github.com/speakleash/bielik-tools/blob/main/examples/crewai_to_file.py)  
Final contens of report will be placed in `bielik_output/atrakcje.md` 
