# Native Bielik plugin for vLLM 0.20.0

Use the combined plugin on the original, unmodified vLLM 0.20.0 wheel. It
registers a native `Parser` under `bielik_020` and keeps reasoning when the old
`DelegatingParser` hands a chunk to tool parsing. It reuses the existing tool
parser and reasoning parser, including legacy `enable_thinking` and v3.1
`reasoning_effort` handling. Per-stream state is held only by that parser
instance, and temporary handoff data is cleared after each call.

From the repository root:

```bash
vllm serve /path/to/model \
    --chat-template /path/to/matching/chat_template.jinja \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./tools/bielik_vllm_parser_v0.20.0.py \
    --tool-call-parser bielik_020 \
    --reasoning-parser-plugin ./tools/bielik_vllm_parser_v0.20.0.py \
    --reasoning-parser bielik_020
```

Pass the same file to **both** plugin flags. The reasoning-plugin flag also
loads its registrations inside engine workers. Do not additionally load the
separate Bielik tool/reasoning plugin files. The two parser names must be
`bielik_020` so vLLM selects the native combined parser through `ParserManager`.

This plugin targets exactly vLLM 0.20.0 and is intended for the legacy advanced
and v3.1 template paths. Keep `skip_special_tokens=false` in requests on older
servers. v3.2 targets vLLM **0.24.0 or newer**; no v3.2 backports to 0.20 or
older APIs are provided. Use the template distributed with the model.

No installed vLLM files or global methods are replaced. Do not apply the old
0.20 runtime patch. If it was applied before, restore the original
`vllm/parser/abstract_parser.py` from its adjacent `.bielik-compat.orig` backup
or reinstall the matching wheel before validating the plugin. The plugin
rejects the superseded handoff backport to avoid emitting reasoning twice.

## Validation

On the original vLLM 0.20.0 wheel, the parser suite passes 52 tests. The tests
check native registry selection, a complete response in one chunk, incomplete
calls, simultaneous reasoning/content, interleaved independent streams and
reasoning disabled through `reasoning_effort=none`.

```bash
python -m pytest -q tests
python tests/live_server_smoke.py --model step72 --base-url http://127.0.0.1:8000
```

The live FP8 model check passes all seven probes on the original v3.1 template.
The installed vLLM parser source is byte-identical to the original backup
before and after the tests. Model weights and templates are not edited.
