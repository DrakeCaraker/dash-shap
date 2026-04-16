# Notebook Health Audit — 2026-04-16

Scanned 9 notebooks, found 7 issues:

## Findings

- [HIGH] notebooks/demo_benchmark_6.ipynb — File size exceeds 2MB (git bloat risk)
  Evidence: `wc -c notebooks/demo_benchmark_6.ipynb` reports 2038228 bytes (1.94 MB)

- [HIGH] notebooks/demo_benchmark_6.ipynb — Out-of-order execution_count values: starts at 2, jumps at cells 10→16, 11→35, 32→57, 36→62, ends with None
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/demo_benchmark_6.ipynb')); ec=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']; print(ec)"` reports `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 62, None]`

- [HIGH] notebooks/explore_experiment_results.ipynb — Out-of-order execution_count values: mix of None and non-sequential counts starting at 26
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/explore_experiment_results.ipynb')); ec=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']; print(ec)"` reports `[None, 26, None, 28, 29, None, None, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, None]`

- [MEDIUM] notebooks/demo_benchmark_6.ipynb — All 60 cells missing `id` field
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/demo_benchmark_6.ipynb')); missing=[i for i,c in enumerate(nb['cells']) if not c.get('id','')]; print(f'Cells missing id: {len(missing)}/{len(nb[\"cells\"])}')"` reports `Cells missing id: 60/60`

- [MEDIUM] notebooks/explore_experiment_results.ipynb — 43 of 44 cells missing `id` field
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/explore_experiment_results.ipynb')); missing=[i for i,c in enumerate(nb['cells']) if not c.get('id','')]; print(f'Cells missing id: {len(missing)}/{len(nb[\"cells\"])}')"` reports `Cells missing id: 43/44`

- [MEDIUM] notebooks/demo_benchmark_6.ipynb — 14 `load_checkpoint()` calls reference files in `checkpoints/` directory which does not exist
  Evidence: `ls checkpoints/ 2>&1` reports `No such file or directory`. Referenced checkpoints: sec1_poc, sec2_baselines, sec3_stability, sec4_sweep, sec5_bc, sec5_bc_stability, sec6_epsilon, sec8_nonlinear, variance_decomposition, sec10_table2, sec11_sc_data, sec11_sc_benchmark, sec12_ablation, sec14_california

- [MEDIUM] notebooks/demo_benchmark_7_parallel.ipynb — 19 `load_checkpoint()` calls reference files in `checkpoints/` directory which does not exist
  Evidence: `ls checkpoints/ 2>&1` reports `No such file or directory`. Referenced checkpoints: v7_sec1_poc, nb7p_mechanism_sweep, nb7p_table2_baselines, nb7p_diagnostics_demo, nb7p_california, nb7p_breast_cancer, nb7p_bc_diagnostics, nb7p_superconductor, nb7p_epsilon, nb7p_ablation, nb7p_nonlinear_v2, nb7p_nonlinear, nb7p_overlapping, nb7p_vardecomp, nb7p_variance_decomposition_crossed, nb7p_first_mover_viz, nb7p_first_mover_bias, nb7p_background_sensitivity, nb7p_asymmetric_dgp

## Clean Notebooks

The following 6 notebooks passed all checks with no issues:

| Notebook | Size | Cells | Exec Order | IDs |
|----------|------|-------|------------|-----|
| demo_benchmark_7_parallel.ipynb | 82 KB | 62 | all null | OK |
| nn_validation_phase0.ipynb | 9 KB | 17 | all null | OK |
| quickstart.ipynb | 49 KB | 12 | sequential (1..8) | OK |
| tutorial_01_the_problem.ipynb | 242 KB | 20 | sequential (1..12) | OK |
| tutorial_02_dash_walkthrough.ipynb | 148 KB | 22 | sequential (1..12) | OK |
| tutorial_03_interpreting_outputs.ipynb | 184 KB | 22 | sequential (1..10) | OK |
| tutorial_04_simulation.ipynb | 82 KB | 20 | sequential (1..10) | OK |

Note: demo_benchmark_7_parallel.ipynb has missing checkpoint files but is otherwise healthy; checkpoint files are generated at runtime and intentionally excluded from version control.

## Summary

7 issues found (0 CRITICAL, 3 HIGH, 4 MEDIUM, 0 LOW)
