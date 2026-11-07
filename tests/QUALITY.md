# Polish checkpoint smoke evaluation

```bash
python scripts/evaluate_bielik_quality.py --model bielik --family v31 \
  --samples 3 --seed 1701 --output /tmp/quality.json
```

Eight fixed Polish tasks cover arithmetic, extraction, a short instruction,
schema-constrained output, tool selection/arguments and avoiding unnecessary
tools. Each runs with thinking off/on, greedily and with N seeded samples.
No tool executes. Scores are per effort/sampling group with Wilson95
intervals; repeated samples are correlated, so intervals are descriptive,
not proof that checkpoints differ statistically. This small regression set
does not replace broad independent benchmarks and must not be used as
training data or the only model selection criterion.

Default exit0 means the evaluation completed, not that quality passed;
the report explicitly says `QUALITY REPORT ONLY`. Add `--min-score 0.9`
to require that accuracy in every group (exit1 on a quality failure).
Transport/protocol/run failures always return2 and count as incorrect.
Artifacts include every request/response, seeds, case-set checksum and
separate semantic/run errors. Record the checkpoint revision, server vLLM
version, parser commit and template checksum alongside each run. Existing
output files are never overwritten. Optional GPU execution is not collected
by pytest; offline scorer tests run without framework dependencies.
