# Notebook Health Audit — 2026-04-03

Scanned 9 notebooks, found 37 issues:

## HIGH (2)

- [HIGH] notebooks/demo_benchmark_6.ipynb — execution_count values are out-of-order (starts at 2): 11->16, 16->35, 55->57
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/demo_benchmark_6.ipynb')); ec=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']; print([x for x in ec if x])"` reports `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59, 60, 61, 62]`

- [HIGH] notebooks/explore_experiment_results.ipynb — execution_count values are out-of-order (starts at 26, not 1): 26->28, 29->32
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/explore_experiment_results.ipynb')); ec=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code']; print([x for x in ec if x])"` reports `[26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]`

## MEDIUM — Missing Cell IDs (2)

- [MEDIUM] notebooks/demo_benchmark_6.ipynb — All 60 cells missing non-empty id field
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/demo_benchmark_6.ipynb')); print(sum(1 for c in nb['cells'] if not c.get('id')))"` reports `60`

- [MEDIUM] notebooks/explore_experiment_results.ipynb — 43 of 44 cells missing non-empty id field
  Evidence: `python3 -c "import json; nb=json.load(open('notebooks/explore_experiment_results.ipynb')); print(sum(1 for c in nb['cells'] if not c.get('id')))"` reports `43`

## MEDIUM — Missing Checkpoints (33)

The `checkpoints/` directory does not exist. All `load_checkpoint()` calls reference missing files. These are expected to be regenerated at runtime; flagging for awareness.

Evidence: `ls -d checkpoints/ 2>&1` reports `ls: checkpoints/: No such file or directory`

### notebooks/demo_benchmark_6.ipynb (14 checkpoints)

- [MEDIUM] load_checkpoint('sec1_poc')
- [MEDIUM] load_checkpoint('sec2_baselines')
- [MEDIUM] load_checkpoint('sec3_stability')
- [MEDIUM] load_checkpoint('sec4_sweep')
- [MEDIUM] load_checkpoint('sec5_bc')
- [MEDIUM] load_checkpoint('sec5_bc_stability')
- [MEDIUM] load_checkpoint('sec6_epsilon')
- [MEDIUM] load_checkpoint('sec8_nonlinear')
- [MEDIUM] load_checkpoint('variance_decomposition')
- [MEDIUM] load_checkpoint('sec10_table2')
- [MEDIUM] load_checkpoint('sec11_sc_data')
- [MEDIUM] load_checkpoint('sec11_sc_benchmark')
- [MEDIUM] load_checkpoint('sec12_ablation')
- [MEDIUM] load_checkpoint('sec14_california')

### notebooks/demo_benchmark_7_parallel.ipynb (19 checkpoints)

- [MEDIUM] load_checkpoint('v7_sec1_poc')
- [MEDIUM] load_checkpoint('nb7p_mechanism_sweep')
- [MEDIUM] load_checkpoint('nb7p_table2_baselines')
- [MEDIUM] load_checkpoint('nb7p_diagnostics_demo')
- [MEDIUM] load_checkpoint('nb7p_california')
- [MEDIUM] load_checkpoint('nb7p_breast_cancer')
- [MEDIUM] load_checkpoint('nb7p_bc_diagnostics')
- [MEDIUM] load_checkpoint('nb7p_superconductor')
- [MEDIUM] load_checkpoint('nb7p_epsilon')
- [MEDIUM] load_checkpoint('nb7p_ablation')
- [MEDIUM] load_checkpoint('nb7p_nonlinear_v2')
- [MEDIUM] load_checkpoint('nb7p_nonlinear')
- [MEDIUM] load_checkpoint('nb7p_overlapping')
- [MEDIUM] load_checkpoint('nb7p_vardecomp')
- [MEDIUM] load_checkpoint('nb7p_variance_decomposition_crossed')
- [MEDIUM] load_checkpoint('nb7p_first_mover_viz')
- [MEDIUM] load_checkpoint('nb7p_first_mover_bias')
- [MEDIUM] load_checkpoint('nb7p_background_sensitivity')
- [MEDIUM] load_checkpoint('nb7p_asymmetric_dgp')

## Clean Notebooks (5)

The following notebooks passed all checks with no issues:

| Notebook | Size | Cells | Code Cells | IDs | Exec Order | Figures | Checkpoints |
|---|---|---|---|---|---|---|---|
| nn_validation_phase0.ipynb | 9 KB | 17 | 9 | OK | OK | OK | N/A |
| quickstart.ipynb | 49 KB | 12 | 8 | OK | OK | OK | N/A |
| tutorial_01_the_problem.ipynb | 242 KB | 20 | 13 | OK | OK | OK | N/A |
| tutorial_02_dash_walkthrough.ipynb | 148 KB | 22 | 14 | OK | OK | OK | N/A |
| tutorial_03_interpreting_outputs.ipynb | 183 KB | 22 | 11 | OK | OK | OK | N/A |
| tutorial_04_simulation.ipynb | 82 KB | 20 | 11 | OK | OK | OK | N/A |

Note: demo_benchmark_7_parallel.ipynb (82 KB, 62 cells) passed all checks except checkpoint references.

## Size Check

No notebooks exceed the 2MB threshold. demo_benchmark_6.ipynb is the largest at 1.94 MB (just under the limit).

Evidence: `ls -la notebooks/*.ipynb | awk '{print $5, $NF}'` reports `2038228 notebooks/demo_benchmark_6.ipynb` as the largest file.

## Summary

37 issues found (0 CRITICAL, 2 HIGH, 35 MEDIUM, 0 LOW)
