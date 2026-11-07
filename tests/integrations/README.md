# One command for framework compatibility

With an existing local Bielik server:

```bash
# First run: create four separate pinned environments, then run everything.
python scripts/test_frameworks.py --all --setup

# Later runs reuse those environments without installation.
python scripts/test_frameworks.py --all
```

The defaults are `--model bielik-test --base-url http://127.0.0.1:8000/v1`
and `--family v31` (Bielik 3.1/3.2). Set `--family v30` for the advanced
Bielik 3.0 template. Setup requires uv and an already installed Python3.12;
it never installs uv, Python, vLLM or model weights. Use `--python /path/to/python`
if necessary. Only `--setup` can install/update dependencies. To prepare
without contacting a model, add `--setup-only`.

Environments live in `.framework-envs/`, one per framework; use `--env-root`
to share them across worktrees. Missing environments or wrong pinned versions
are errors, never skips. The runner checks the served model, runs the direct
API matrix, then LiteLLM SDK/proxy, LangGraph, Pydantic AI and Inspect AI
sequentially against that same server. It starts/stops only its own temporary
loopback LiteLLM proxy, with random environment-only credentials. It never
starts the model server or executes generated shell/code.

The terminal prints one status/count per probe and the path to a single
`summary.json`. A new private `.framework-results/<timestamp>/` directory
holds that summary, native reports and logs (override with `--output-dir`).
Existing directories are never overwritten. The summary includes dependency
versions, every failing case, native report paths and completion status.
Exit0 means the entire selected run passed, exit1 means test failures, exit2
means missing dependencies, server/probe errors or timeouts. An interrupted
run cannot leave a passing summary. Each probe has a finite `--timeout` and
its owned process group is terminated, including workers, on interruption.

Cases cover reasoning none/medium, full/streaming, tool choice, schema, history
and safe synthetic tools as supported by each native probe. Adapter-specific
cases remain extra coverage; the matrix is not identical across libraries.
A failed framework case is not automatically blamed on the model or parser:
inspect its native transcript alongside the direct API result. The known
LiteLLM n=2 streaming issue is labelled only when the matching direct case
passes; it still fails the run. The current complete `--all` run therefore
returns1 for this documented upstream incompatibility, rather than hiding it.
Record immutable checkpoint/vLLM/template/parser provenance alongside reports;
a served-model alias alone does not identify the weights.

For a subset, repeat `--framework`:

```bash
python scripts/test_frameworks.py --framework langgraph --framework pydantic-ai
```

BFCL and GuideLLM are separate optional longer workloads, excluded from --all:

```bash
python scripts/test_frameworks.py --benchmark bfcl --setup --bfcl-limit 5
python scripts/test_frameworks.py --benchmark guidellm --setup --tokenizer /local/model
```

BFCL runs only the audited AST subset (no generated execution), with an explicit
score1 gate for this runner. GuideLLM setup selects CPU-only PyTorch. Its default
is a short load smoke test; longer native probe options are documented in its
README. A model turn is not a conversation, and small runs do not establish
stable latency percentiles. Raw native probes remain directly runnable.

## Individual probes and last manual results

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
