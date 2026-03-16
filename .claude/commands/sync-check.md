# Sync Check

Verify configuration consistency across the canonical experimental sources.

## Sources to compare

1. `run_experiments.py` — grep for PAPER_CONFIG or equivalent constants (M, K, N_REPS, EPSILON, DELTA, SEED, epsilon_mode)
2. `notebooks/demo_benchmark_6.ipynb` — **canonical for ArXiv**; read cells for config definitions
3. `notebooks/demo_benchmark_7.ipynb` — **canonical for TMLR (in development)**; read cells for config definitions
4. `CLAUDE.md` — "Canonical Configuration (PAPER_CONFIG)" section

## Checks to perform

1. **Parameter drift**: Compare values of M, K, N_REPS, EPSILON, DELTA, SEED, REAL_EPSILON, epsilon_mode across all sources. Flag any mismatches.
2. **Experiment list**: Compare experiment names defined in `run_experiments.py` against those referenced in each notebook. Flag experiments present in one but not the other.
3. **Method list**: Compare baseline method names across sources. Flag additions or removals.
4. **Notebook status**: Note which notebook has been run (has outputs) vs is a skeleton.

## Output format

Present a summary table:

| Parameter | run_experiments.py | notebook 6 | notebook 7 | CLAUDE.md | Match? |
|-----------|-------------------|------------|------------|-----------|--------|

Then list any mismatches with specific locations (file + line/cell number).

End with a **PASS** (all consistent) or **FAIL** (mismatches found) verdict.
