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
| ≥ 0.24.0 | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` |
| 0.15.0 – 0.23.x | `bielik_vllm_tool_parser_v0.15.0.py` | `bielik_vllm_reasoning_parser.py` |
| 0.13.0 – 0.14.x | `bielik_vllm_tool_parser_v0.13.0.py` | `bielik_vllm_reasoning_parser_v0.13.0.py` |
| ≤ 0.12.0 | `bielik_vllm_tool_parser_v0.12.0.py` | — |

The 0.24.0 tool-parser cut is the vLLM `ToolParser` API change
(`__init__(tokenizer, tools=…)` plus `supports_required_and_named`).
Reasoning parsers are unchanged across 0.15+.

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
