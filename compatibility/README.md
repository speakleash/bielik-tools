# Backports for older vLLM releases

The effort-forwarding bug in vLLM 0.14.0 / 0.15.0 needs a runtime backport
when retaining the strict v3.1 template:

- **0.14.0 / 0.15.0:** chat serving overwrites the template's `reasoning_effort`
  with the optional request-level field even when it is `None`. Preserve the
  template value when the API field is unset; an explicit API value retains
  its precedence. This permits the unmodified strict v3.1 template.


Apply these backports after installing the corresponding vLLM release and
before starting its server. Use that environment's Python interpreter:

```bash
/path/to/vllm-env/bin/python tools/patch_vllm_compat.py
/path/to/vllm-env/bin/python tools/patch_vllm_compat.py --apply
/path/to/vllm-env/bin/python tools/patch_vllm_compat.py  # reports already applied
```

Only the exact releases listed above are modified. The script checks the source
fragment, validates Python syntax, saves an adjacent `.bielik-compat.orig`
backup, and atomically replaces the file. Repeated application is a no-op.
Unexpected source raises an error instead of guessing. The script does not
import vLLM, start a server, download weights, or install dependencies.

Individual symlinked/hardlinked files (including uv's shared-cache links) are
replaced in the environment, without changing the cache target. A symlinked
package directory is refused; use an environment with real package directories.
Restart an existing server after applying a patch. Reinstalling/upgrading vLLM
can remove the local patch; inspect/apply it again for a supported release.
The original source is retained in the backup for rollback.

These runtime backports complement the version-specific Bielik plugins; they
do not replace the parser import, streaming, and request-configuration fixes.
No global monkey patch or shared per-request state is added to the plugins.

## Tests

The patcher's semantics and file isolation need only pytest:

```bash
python -m pytest -q tests/test_vllm_compat_patches.py
```

For vLLM 0.20.0 use the native combined plugin instead; this installer leaves
0.20 unchanged. Its earlier reasoning-handoff backport has been removed in
favor of the supported plugin API. The v3.2 support target is vLLM 0.24.0+,
without legacy v3.2 adapters.
