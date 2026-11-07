# Token replay regression artifacts

Capture synthetic requests from the running vLLM 0.32 server, then replay
their exact token IDs without a GPU. No generated tool is executed.

```bash
python scripts/parser_replay.py capture --tokenizer /local/model \
  --input /tmp/request.json --output /tmp/capture.json
python scripts/parser_replay.py replay --tokenizer /local/model \
  --input /tmp/capture.json --output /tmp/replay.json
```

The request file is a Chat Completions request including `model`, `messages`
and any tools/reasoning settings. Capture supports one non-streaming choice.
For authenticated servers set `BIELIK_TEST_API_KEY`; its value is not saved.
Replay exits 1 on a streaming/non-streaming difference. It checks both the
tokenizer fingerprint and decoded text, retaining the distinction between
dedicated tag tokens and tags spelled with ordinary tokens. Terminal stop
IDs remain in parser input exactly as in vLLM; their text is suppressed when
`terminal_stop_suppressed` is true, including when the API rewrites the finish
reason to `tool_calls`.
Decoding uses vLLM's fast detokenizer primed with the exact prompt, including
leading spaces and incomplete UTF-8. Custom stop strings/tokens, inclusion of
stop text and ignoring EOS are rejected. Use a server with EOS-only stopping;
custom server-side stops reported in `stop_reason` are also rejected.

Each capture contains the request, exact prompt/output IDs, full server
response, tokenizer fingerprint, parser hashes and local runtime versions.
The model/checkpoint revision and server vLLM version must also be recorded
by the operator: a client cannot infer them from a served-model alias.
Keep captures with private conversations outside the repository. Output files
are created exclusively so a previous regression artifact is not overwritten.
Only reviewed synthetic captures should be committed here.
