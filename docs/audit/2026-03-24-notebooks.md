# Notebook Health Audit — 2026-03-24

Scanned 9 notebooks, found 97 issues:

## File Size

No notebooks exceeded the 2MB threshold. `notebooks/demo_benchmark_6.ipynb` is 1.94MB — approaching the limit.

## Execution Order (HIGH)

- [HIGH] notebooks/demo_benchmark_6.ipynb — Out-of-order execution_count (starts at 2, gaps: 11->16, 16->35, 55->57)
- [HIGH] notebooks/explore_experiment_results.ipynb — Out-of-order execution_count (starts at 26, gaps: 26->28, 29->32)

## Missing Cell IDs (MEDIUM)

- [MEDIUM] notebooks/demo_benchmark_6.ipynb — 30 code cells have missing/empty id fields (indices: 1, 3, 5, 7, 9, 10, 12, 13, 14, 16, 17, 19, 20, 22, 24, 26, 28, 29, 31, 32, 34, 35, 37, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 54, 55, 56, 58, 59)
- [MEDIUM] notebooks/explore_experiment_results.ipynb — 22 code cells have missing/empty id fields (indices: 1, 2, 4, 5, 6, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 43)

## Missing Checkpoint Files (MEDIUM)

All checkpoint references below point to files not present in `checkpoints/`. This is expected if checkpoints have been cleared between sessions (per project policy, `.pkl` files are not committed).

### notebooks/demo_benchmark_6.ipynb (14 missing checkpoints)

- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec1_poc") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec2_baselines") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec3_stability") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec4_sweep") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec5_bc") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec5_bc_stability") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec6_epsilon") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec8_nonlinear") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("variance_decomposition") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec10_table2") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec11_sc_data") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec11_sc_benchmark") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec12_ablation") not found
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — load_checkpoint("sec14_california") not found

### notebooks/demo_benchmark_7_parallel.ipynb (19 missing checkpoints)

- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("v7_sec1_poc") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_mechanism_sweep") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_table2_baselines") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_diagnostics_demo") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_california") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_breast_cancer") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_bc_diagnostics") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_superconductor") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_epsilon") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_ablation") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_nonlinear_v2") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_nonlinear") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_overlapping") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_vardecomp") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_variance_decomposition_crossed") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_first_mover_viz") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_first_mover_bias") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_background_sensitivity") not found
- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — load_checkpoint("nb7p_asymmetric_dgp") not found

## Clean Notebooks (no issues)

- notebooks/nn_validation_phase0.ipynb
- notebooks/quickstart.ipynb
- notebooks/tutorial_01_the_problem.ipynb
- notebooks/tutorial_02_dash_walkthrough.ipynb
- notebooks/tutorial_03_interpreting_outputs.ipynb
- notebooks/tutorial_04_simulation.ipynb

## Summary

97 issues found (0 CRITICAL, 2 HIGH, 95 MEDIUM, 0 LOW)

| Notebook | Size | Exec Order | Missing IDs | Missing Checkpoints |
|---|---|---|---|---|
| demo_benchmark_6.ipynb | 1.94MB | OUT-OF-ORDER | 30 | 14 |
| demo_benchmark_7_parallel.ipynb | 82KB | OK | 0 | 19 |
| explore_experiment_results.ipynb | 780KB | OUT-OF-ORDER | 22 | 0 |
| nn_validation_phase0.ipynb | 9KB | OK | 0 | 0 |
| quickstart.ipynb | 49KB | OK | 0 | 0 |
| tutorial_01_the_problem.ipynb | 242KB | OK | 0 | 0 |
| tutorial_02_dash_walkthrough.ipynb | 148KB | OK | 0 | 0 |
| tutorial_03_interpreting_outputs.ipynb | 184KB | OK | 0 | 0 |
| tutorial_04_simulation.ipynb | 82KB | OK | 0 | 0 |

### Notes

- Missing checkpoint files are expected per project policy (`.pkl` files are gitignored and cleared between sessions). Re-running the notebooks will regenerate them.
- The two HIGH execution-order issues on `demo_benchmark_6` and `explore_experiment_results` indicate cells were run non-sequentially, likely from manual re-executions during development. For canonical notebooks whose outputs are empirical records, this is cosmetic but worth noting.
- The 52 missing cell IDs across two older notebooks are a Jupyter 4.x-era artifact. Running `scripts/check_notebook_ids.py` and re-saving would fix them.
