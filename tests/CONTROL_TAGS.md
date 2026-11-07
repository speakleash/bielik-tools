# Control tags must stay data after tokenization

```bash
python scripts/check_control_tag_tokenization.py --tokenizer /local/bielik32 \
  --family v31 --output /tmp/control-tags32.json
```

Use `--family v30` with the 3.0 tokenizer and advanced template. This is an
offline check on local tokenizer files; no inference, downloads or tool
execution occurs. Each run covers six control tags in system, user and
tool-result text with thinking on/off (36 cases).

Both literal structure and dedicated token-ID counts are compared to the
same conversation with safe content. This catches escaping that is undone
by tokenizer normalization. The template's own control tags remain allowed.
Assistant history's raw-by-default contract is intentional and covered by
existing template tests. This structural check does not claim to prevent
semantic prompt injection. Negative tests disable sanitizing or remove a
control token and require probe failures.
