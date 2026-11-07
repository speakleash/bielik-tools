# Reproducible chunk boundary tests

This extends exact-token capture/replay; it does not generate new model
responses. Run against any synthetic capture from `parser_replay.py`:

```bash
python scripts/check_replay_partitions.py /tmp/capture.json \
  --tokenizer /local/model --seed 1701 --trials 100 --max-chunk 32 \
  --output /tmp/partitions.json
```

Every run tests single-token streaming, the whole output in one chunk and
the requested number of seeded random partitions. The report stores the
exact failing boundaries and normalized outputs. All streamed results are
compared to the same non-streaming parse; IDs, Unicode and JSON are not
re-encoded. Exit 1 means a real parity difference, even for a documented
limitation such as ordinary-token tool markup in auto mode. It is not
silenced as an expected failure.

Offline regression tests also exercise every possible single cut in short
outputs: escapes, Unicode, adjacent calls and calls quoted inside reasoning.
