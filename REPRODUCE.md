# Reproducing Paper Results

This document explains how to reproduce the experimental results in:

> Caraker, Arnold, Rhoads (2026). *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* [arXiv:2603.22346](https://arxiv.org/abs/2603.22346) | [Zenodo](https://doi.org/10.5281/zenodo.19060132)

---

## For TMLR Reviewers

This section assumes no prior familiarity with the repository. Follow the steps in order.

### 1. Environment setup

**Requirements:** Python 3.9+, pip, ~50 GB disk for checkpoints and results.

```bash
git clone https://github.com/DrakeCaraker/dash-shap.git
cd dash-shap
pip install -r requirements.lock   # exact paper environment (pinned versions)
pip install -e .                   # install the dash_shap package itself
```

Verify the installation:
```bash
python -c "from dash_shap import DASHPipeline; print('OK')"
pytest -m "not slow"               # 269 unit tests, ~30 seconds — must pass
```

### 2. Smoke test (local, ~2 minutes)

Before committing cloud resources, run one experiment at reduced scale to confirm the pipeline works end-to-end on your machine:

```bash
python run_experiments_parallel.py --experiments linear_sweep
```

This writes `results/tables/synthetic_linear_sweep.json`. Open it and confirm `_meta.code_sha` matches `git rev-parse HEAD` and `_meta.paper_config` matches the canonical config below.

### 3. Canonical configuration

All paper results use this configuration (defined in `dash_shap/config.py`):

| Parameter | Value | Description |
|---|---|---|
| `M` | 200 | Population size (models trained) |
| `K` | 30 | Models selected for consensus |
| `N_REPS` | 50 | Repetitions per experiment |
| `EPSILON` | 0.08 | Absolute filter threshold (synthetic) |
| `DELTA` | 0.05 | Deduplication Spearman threshold |
| `REAL_EPSILON` | 0.05 | Filter threshold for real-world datasets |
| `SEED` | 42 | Global random seed |

The `config_sha` field in each result JSON is a SHA256 fingerprint of this dict. Mismatches indicate a config drift.

### 4. Full reproduction (cloud)

**Hardware:** The paper results were produced on an AWS SageMaker instance with 72 vCPUs and 144 GB RAM (likely `ml.c5.18xlarge`). The runner is designed around this parallelism level.

**Minimum viable hardware:** Any machine with ≥8 vCPUs and ≥16 GB RAM will run all experiments correctly but significantly slower. On an 8-core machine, expect 3–5× longer runtimes.

**Estimated wall-clock time on 72-vCPU instance:**

| Experiment | Est. time | Notes |
|---|---|---|
| `linear_sweep` | ~36 min | Confirmed from run metadata |
| `nonlinear_sweep` | ~30–45 min | Similar structure to linear |
| `table2_baselines` | ~20–30 min | |
| `overlapping` | ~15–25 min | |
| `real_california` | ~30–60 min | Real dataset, SHAP on held-out set |
| `real_breast_cancer` | ~30–60 min | |
| `real_superconductor` | ~60–120 min | Largest dataset (21k × 81 features) |
| `epsilon_sensitivity` | ~20–30 min | |
| `ablation` | ~20–30 min | |
| `variance_decomposition` | ~20–30 min | |
| `asymmetric_dgp` | ~20–30 min | |
| `variance_decomposition_crossed` | ~20–30 min | |
| `background_sensitivity` | ~20–30 min | |
| `first_mover_visualization` | ~15–25 min | |
| `first_mover_bias` | ~20–30 min | |
| `k_sweep_independence` | ~20–30 min | |
| **Total** | **~6–10 hours** | All 16 on 72 vCPUs |

**Run all experiments:**
```bash
python run_experiments_parallel.py
```

**Run a single experiment:**
```bash
python run_experiments_parallel.py --experiments linear_sweep
```

**Resume after interruption** (uses per-rep checkpoints):
```bash
python run_experiments_parallel.py --resume
```

### 5. Verifying results

Each output JSON in `results/tables/` contains a `_meta` block. Check:

```python
import json

data = json.load(open("results/tables/synthetic_linear_sweep.json"))
meta = data["_meta"]

print(meta["code_sha"])        # should match: git rev-parse HEAD
print(meta["code_dirty"])      # should be False for a clean run
print(meta["config_sha"])      # fingerprint of PAPER_CONFIG
print(meta["n_reps"])          # should be 50
print(meta["paper_config"])    # full config snapshot
```

See `results/README.md` for the full `_meta` field reference.

### 6. Interactive exploration

Once results are written, open the canonical TMLR notebook to reproduce all figures and tables:

```bash
jupyter notebook notebooks/demo_benchmark_7_parallel.ipynb
```

Or use the interactive results viewer:

```bash
jupyter notebook notebooks/explore_experiment_results.ipynb
```

### 7. Known limitations of the committed results

Three result files in this repository (`synthetic_linear_sweep.json`, `first_mover_visualization.json`, `k_sweep_independence.json`) were produced on a SageMaker instance before the provenance infrastructure was added to the codebase. Their `_meta` blocks have:

- `config_sha: "backfilled"` — the config fingerprint was reconstructed post-hoc, not captured at runtime
- `hardware: "SageMaker run — hardware details not captured at time of run"` — instance type and RAM were not recorded
- `elapsed_s: null` — timing was not recorded (except `linear_sweep`, whose 2151.7s is embedded in the data)

The data itself is valid (50 reps, correct `PAPER_CONFIG`). These fields are marked `"backfilled"` as an honest notation of the metadata limitation, not a data integrity problem.

All subsequently produced results will have full provenance automatically.

---

## For Collaborators

If you have the existing partial results and want to run only the remaining experiments:

### Continuation run (run only what's missing)

The 18 experiments are listed in `DEFAULT_ORDER` at the bottom of `run_experiments_parallel.py`. Three are already complete:
- `linear_sweep` ✓
- `first_mover_visualization` ✓
- `k_sweep_independence` ✓
- `overlapping` — re-run needed (missing per-rep arrays in existing file)

Run the remainder:
```bash
python run_experiments_parallel.py --experiments \
  overlapping nonlinear_sweep table2_baselines \
  real_california real_breast_cancer real_superconductor \
  epsilon_sensitivity ablation variance_decomposition \
  asymmetric_dgp variance_decomposition_crossed \
  background_sensitivity first_mover_bias
```

### Full clean run (all 18 from scratch)

If you want every result with full provenance from a single commit:
```bash
python run_experiments_parallel.py
```

The overwrite protection in `save_json` will back up the three existing results to `*.bak.json` before overwriting. These backups are gitignored and safe to delete after confirming the new results.

### Checking parallelism on your instance

The runner auto-detects cores and allocates `n_outer × n_inner × nthread ≤ total_cores`. To see the allocation for your machine:

```python
from dash_shap.utils.thread_budget import get_available_cores, compute_thread_budget
print(f"Cores available: {get_available_cores()}")
budget = compute_thread_budget(n_outer=9)
print(f"Budget: {budget}")
```

Override with `DASH_MAX_THREADS=N` environment variable if needed.

### Environment snapshot

At startup, `run_experiments_parallel.py` writes `results/environment.json` with your exact pip package list and hardware. This is committed alongside results and serves as the software provenance record for that run.

---

## Troubleshooting

**Import errors after install:** Run `pip install -e .` from the repo root. The `dash_shap` package must be installed in editable mode; `pip install -r requirements.lock` alone is not sufficient.

**Out-of-memory errors:** Reduce parallelism with `DASH_MAX_PARALLEL_REPS=4` (default is auto-detected from available RAM at 200 MB/worker).

**Checkpoint conflicts when resuming:** If a run was interrupted mid-experiment, use `--resume` to pick up from per-rep checkpoints. Use `--no-cleanup` to keep checkpoints after completion for debugging.

**Mismatched `config_sha`:** If the CI `result-validation` job reports a config mismatch between experiments, check whether `PAPER_CONFIG` in `dash_shap/config.py` was modified between runs. All results in a single submission must share the same `config_sha`.
