# Optional manual integration probes

These probes are opt-in. Default GitHub CPU CI compiles all Python files and
runs the parser/client suite, but does not install the six framework/harness
packages. Dependency-specific tests skip when their packages are absent.
A green default CI check therefore does not certify a live framework run.
Use separate environments and the pinned requirements in each integration's
directory; dependencies are not needed to use the parsers themselves.

All live probes require an existing local OpenAI-compatible server, the
matching Bielik template/unified parsers and the served-model name. Tools
are allowlisted deterministic synthetic add/weather stubs, or fixed dataset
results (GuideLLM). BFCL is an AST-only subset and never executes generated
code. No probe starts vLLM or downloads model weights. Use synthetic prompts:
reports/native logs contain conversations; keep private captures outside Git.

Last complete manual run: **2026-11-07**, local Bielik 3.2 checkpoint with
vLLM **0.32.0**. These are smoke/compatibility results, not guarantees for
new checkpoints, dependency versions or full benchmark scores:

- LiteLLM SDK/proxy **1.108.2**: each API matrix36/40, retry loops4/4.
  All four n=2 streaming variants lack a choice's final status upstream.
  Failures remain visible; the full probe exits1. See [PR49](https://github.com/speakleash/bielik-tools/pull/49).
- LangChain OpenAI **1.7.4**, LangGraph **1.2.20**: native graph32/32.
  ChatBielik uses private version-bound ChatOpenAI hooks; every dependency
  upgrade needs new adapter tests and a full live run. See [PR50](https://github.com/speakleash/bielik-tools/pull/50).
- Pydantic AI slim/openai **2.66.0**: native agents36/36, including ModelRetry
  and NativeOutput. See [PR53](https://github.com/speakleash/bielik-tools/pull/53).
- Inspect AI **0.3.288**: native Tasks/scorers32/32 with .eval logs.
  See [PR52](https://github.com/speakleash/bielik-tools/pull/52).
- bfcl-eval **2026.3.23**: Python AST subset32/32 (two examples per category,
  not full BFCL). Install its pinned requirements with --no-deps as documented
  to avoid unrelated vendor handlers. See [PR51](https://github.com/speakleash/bielik-tools/pull/51).
- GuideLLM **0.8.1**: eight profiles,64 model turns, zero errors/incomplete.
  Use CPU-only PyTorch in its separate environment. Model turns are not
  conversations; default small runs do not establish stable percentiles.
  See [PR54](https://github.com/speakleash/bielik-tools/pull/54).

Each PR supplies scripts/probe_*.py (BFCL: evaluate_bfcl_ast.py), pinned
requirements, targeted offline tests and its own README with commands.
Run those offline tests after changing an adapter, then its full live probe
with reasoning none/medium and streaming/full responses. Record the date,
package versions, immutable checkpoint revision, vLLM version and template/
parser hashes alongside reports; model aliases alone are not provenance.
Update this index whenever pins or the last verified results change. Further
framework installation and live GPU tests remain explicit manual operations.
