# Testing templates, parsers and models

Four levels, cheapest first. Run levels 0–2 on every pull request, and
levels 3–4 before a release, when vLLM is upgraded, or when a chat template or
model checkpoint changes. Copy the result summary into the PR description.

| Level | What | Needs | Time |
|---|---|---|---|
| 0 | Syntax and template render-check | Python + Jinja2 | seconds |
| 1 | Parser unit tests (`pytest tests`) | vLLM 0.32 (CPU is enough) | ~1 min |
| 2 | Template ↔ tokenizer ↔ parser contract | vLLM + local tokenizers | ~1 min |
| 3 | Live smoke and stream/non-stream parity | GPU, model weights | ~5 min per model |
| 4 | Model-quality probes and compatibility matrix | GPU | longer |

## Test matrix

Every configuration that is documented as supported must have a row here.

| Model | Template | Tool parser | Reasoning parser | Thinking switch |
|---|---|---|---|---|
| Bielik-11B-v3.0-Instruct | `tools/bielik_advanced_chat_template.jinja` | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` | `chat_template_kwargs.enable_thinking` or top-level `reasoning_effort` |
| Bielik 3.1 (e.g. `step_72`) | `tools/bielik_v3.2_chat_template.jinja` (or the checkpoint's own) | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` | `chat_template_kwargs.reasoning_effort` or top-level `reasoning_effort` |
| Bielik 3.2 (e.g. `steps_70_90_110`) | `tools/bielik_v3.2_chat_template.jinja` (or the checkpoint's own) | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` | as 3.1 |

Tokenizer differences matter: in 3.0 `<think>`, `</think>`, `<tool_call>` and
`</tool_call>` are *special* tokens (dropped by `skip_special_tokens=True`). In
3.1/3.2 `tokenizer_config.json` marks them `special: false`, but
`special_tokens_map.json` still lists them as additional special tokens, so
whether `skip_special_tokens=True` drops them depends on the decoder and the
transformers version. Test each family separately.

## Level 0 – syntax and templates

```bash
python -m py_compile tools/*.py examples/*.py scripts/*.py tests/*.py
python scripts/check_templates.py
```

`check_templates.py` render-checks every `tools/*.jinja` (and any template
passed as an argument, e.g. a checkpoint's own `chat_template.jinja`) with
plain Jinja2. The rendering that vLLM actually uses is covered by the
transformers-based tests in level 1 and the level 2 script.

Template review checklist (diff against the previous release):

- `add_generation_prompt` ends with `<think>` (thinking on) or `<think></think>`
  (thinking off) exactly as the reasoning parser expects;
- no debug overrides left in (for example a hard-coded `set x = false` after a
  documented default);
- user, system and tool text is sanitized (control tags escaped);
- `raise_exception` messages for every documented invalid input;
- `generation_config.json` `eos_token_id` matches the template's turn end
  (`<|im_end|>`).

## Level 1 – parser unit tests

```bash
python -m pytest tests -q -p no:cacheprovider
```

Run in the same vLLM version as CI (0.32.0). No GPU or weights needed. With
`xgrammar` installed, the named-tool-choice tests also check the grammar that
the decoder uses, not only Python's `re`.

## Level 2 – template ↔ tokenizer ↔ parser contract

```bash
python scripts/check_parser_template_contract.py ~/repos/models/step_72 --family v31
python scripts/check_parser_template_contract.py ~/repos/models/steps_70_90_110 --family v31
python scripts/check_parser_template_contract.py speakleash/Bielik-11B-v3.0-Instruct \
    --family v30 --template tools/bielik_advanced_chat_template.jinja
```

The script renders the real template with the real tokenizer for the cases
below and feeds the prompt token IDs to the reasoning parser:

| Case | Prompt must end with | `is_reasoning_end(prompt)` |
|---|---|---|
| thinking off | `<think></think>` (v3.1) / `<think>\n\n</think>\n\n` (3.0, `enable_thinking=false`) | True |
| thinking on | `<think>` (v3.1) / no prefill (3.0) | False |
| v3.1 template + legacy `enable_thinking=true` | `<think></think>` | True – output must be content |
| history with earlier reasoning | final turn as above | as above |

Also check that `<think>`/`<tool_call>` survive detokenization with the
`skip_special_tokens` value the parsers set (3.0 needs `False`).

## Level 3 – live smoke and parity

Start the server with exactly one tool parser and one reasoning parser (both
register the name `bielik`). Use an FP8 checkpoint or `--quantization` on 24 GB
GPUs. Example for Bielik 3.2:

```bash
vllm serve ~/repos/models/steps_70_90_110 --served-model-name bielik \
  --quantization fp8_per_tensor --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-parser-plugin ./tools/bielik_vllm_tool_parser.py --tool-call-parser bielik \
  --reasoning-parser-plugin ./tools/bielik_vllm_reasoning_parser.py --reasoning-parser bielik \
  --exclude-tools-when-tool-choice-none
```

For 3.0 add `--chat-template ./tools/bielik_advanced_chat_template.jinja`; the
parser files are the same for every model.

Then run both live scripts:

```bash
python tests/live_parity_probe.py --base-url http://127.0.0.1:8000 --model bielik --family v31
VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_MODEL=bielik python examples/parser_smoke_1_7.py all
```

`live_parity_probe.py` sends every request twice (non-streaming and streaming)
and fails on: wrong tool names, invalid JSON arguments, missing or unexpected
reasoning, `<think>`/`<tool_call>` markup in `content`, wrong `finish_reason`,
or a structural stream/non-stream difference (tool names, reasoning present or
not, `finish_reason`; argument values and text are not compared). For a named
`tool_choice` both `stop` (vLLM/OpenAI convention) and `tool_calls` (the Bielik parser
rewrites the choice to `auto`) are accepted. Cases: thinking off/on (kwargs and
top-level `reasoning_effort`), legacy `enable_thinking` on the v3.1 template,
`auto`, parallel, `required` (greeting, parallel, `<` in arguments), named
`tool_choice` (incl. a name with `-`), `none`, thinking + tools/required/named,
JSON schema with `thinking_token_budget`, tool-result round trip, and an echo
of user-supplied `<tool_call>` markup. The echo case is advisory (`WARN`, not a
failure): whether the model copies the text or calls the tool is model
behaviour, but both modes must agree.

Rules for reading results:

- Greedy decoding is not bit-identical between streaming and non-streaming
  (different batches); compare structure, not text. Rerun a parity failure
  once before reporting it, and quote both runs.
- Separate parser failures (markup leak, lost or invented calls, wrong
  reasoning/content split, wrong `finish_reason`) from model behaviour (wrong
  city, retrying after a tool error, verbose answers).
- Do not use `--enforce-eager` or different sampling between runs you compare.

## Level 4 – model quality and compatibility

### Token-level behaviour of a trained checkpoint

Run this for every new checkpoint, before the parser-level tests are trusted:

```bash
python scripts/diagnose_model_behaviour.py --base-url http://127.0.0.1:8000 \
    --model bielik --tokenizer ~/repos/models/steps_70_90_110 --family v31 \
    --samples 8 --jsonl-out diag.jsonl
```

It reads the raw `token_ids` (`return_token_ids`), so parsers cannot hide
anything, and samples each scenario N times (temperature 0.7) plus once
greedily. Per scenario it reports `k/N` for flags such as:

| Flag | Meaning |
|---|---|
| `literal:<tag>` | tag written as ordinary text pieces instead of its token ID; `(guided)` under `required`/named grammar, where that is expected |
| `think_reopened`, `think_unclosed`, `think_multiple_blocks`, `think_stray_close`, `think_close_without_open` | broken reasoning structure relative to the prompt (structure is read from token IDs only) |
| `think_open_omitted` | reasoning starts without `<think>` and ends with `</think>`; the DeepSeek-R1 parser treats it as reasoning (info) |
| `think_when_off`, `think_missing_or_empty` | reasoning switch not obeyed |
| `tool_call_in_reasoning`, `tool_call_unclosed`, `call_untagged_or_legacy_format` | call placement and format |
| `call_invalid_json`, `call_parameters_key`, `call_arguments_as_string`, `call_unknown_tool`, `call_missing_required_arg`, `call_unknown_arg`, `call_wrong_arg_type` | call payload quality |
| `call_missing`, `call_unexpected`, `call_irrelevant_tool` | calls when none are expected, none when expected, or a tool the request does not need |
| `echo:literal_markup`, `echo:zwsp_escapes` | in the echo scenario: copied markup as plain text or with the template's escapes (info; literal copies are what the non-streaming parser may execute) |
| `generated_im_start`, `zwsp_copied`, `stopped_with_</s>`, `length_limit`, `repetitive` | turn structure and termination |
| `reasoning_lang:pl/en`, `text_around_calls` | informational only |

A literal tag is invisible to the streaming parser, which keys on token IDs,
but the non-streaming parser matches text, so any non-zero `literal:` rate
outside guided scenarios produces stream/non-stream differences. Compare
rates between checkpoints, not single completions. `--strict
--max-issue-rate 0.1` turns the report into a gate.

### Other checks

- `examples/parser_smoke_1_7.py extra` and the instruction probes in
  `examples/results/` for regressions between checkpoints;
- older vLLM versions from the README compatibility table: import,
  construction and non-streaming extraction (`tests/vllm_compat_probe.py`);
- record model, template checksum (`md5sum chat_template.jinja`), vLLM version,
  parser commit and the full probe output in `examples/results/`.
