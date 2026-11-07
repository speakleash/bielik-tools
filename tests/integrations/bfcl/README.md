# BFCL Python AST subset through our real API/parsers

```bash
python -m pip install --no-deps -r tests/integrations/bfcl/requirements.txt
python scripts/evaluate_bfcl_ast.py --model bielik-test --family v31 \
  --categories simple_python,multiple,parallel,parallel_multiple --limit 5 \
  --output /tmp/bfcl.json
```

Use a separate environment. The pinned distribution supplies the original
data, ground truth and unmodified Python AST checker. Native API tool calls
are converted directly to its structured input, never to executable Python.
Each selected example runs thinking off/on, full/streaming. The report pins
BFCL/checker/data checksums and records every exact request/response/verdict.
No benchmark function, generated code or external tool is ever executed.

The upstream checker imports all vendor handlers just to choose whether
dots in names become underscores. A process-scoped identity-name shim
supplies only that configuration for our local server, which accepts the
original names. This avoids installing/loading unrelated vendor SDKs, search
clients, vector stores and model runtimes. The audited checker path and its
Python type converters have no eval/exec/subprocess/network/file operations.

Only these four Python AST categories are supported. Executable, agentic,
multiturn, Java and JavaScript categories are deliberately rejected; this is
NOT a full BFCL score or proof of BFCL multiturn compatibility. Native
multiturn behavior is tested separately by the framework probes. The full
package CLI is not invoked. Dataset files remain in the package, not copied
into this repo or training data.

Default exit0 means a completed report, not a quality pass. --min-score
enables an accuracy gate (exit1); run errors always return2. Small --limit
smoke runs are not representative leaderboard results. Record checkpoint,
vLLM/parser/template provenance alongside the report (including checkpoint revision and template/parser hashes).
