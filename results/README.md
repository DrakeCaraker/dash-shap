# Results Directory

This directory contains all experimental outputs from `run_experiments_parallel.py`.

## Directory Structure

```
results/
  tables/          JSON result files — one per experiment
  figures/         Plots and visualizations (PDF/PNG)
  environment.json Hardware + pip-package snapshot at run startup
  PROVENANCE.md    Human-readable append-only run log
  checkpoints/     Per-rep checkpoint files (gitignored)
```

## JSON Artifact Format

Every `tables/*.json` file produced after the reproducibility infrastructure
update contains a `_meta` block as its **first key**:

```json
{
  "_meta": {
    "experiment":   "linear_sweep",
    "timestamp":    "2026-03-23T19:20:37+00:00",
    "code_sha":     "7ff988a3b2c1...",
    "code_dirty":   false,
    "config_sha":   "a3f9b2c1d4e5...",
    "n_reps":       50,
    "paper_config": {"M": 200, "K": 30, "N_REPS": 50, "EPSILON": 0.08, "DELTA": 0.05},
    "elapsed_s":    2151.7,
    "output":       "results/tables/synthetic_linear_sweep.json",
    "hardware": {
      "cpu_count":     72,
      "ram_gb":        144.0,
      "instance_type": "ml.c5.18xlarge",
      "python":        "3.9.6",
      "hostname":      "sagemaker-user"
    }
  },
  ...experiment data...
}
```

### `_meta` Field Reference

| Field | Description |
|---|---|
| `experiment` | Canonical experiment name (matches `--experiments` CLI arg) |
| `timestamp` | ISO-8601 UTC timestamp of the save |
| `code_sha` | Git HEAD SHA at time of run |
| `code_dirty` | `true` if uncommitted changes were present — treat results with caution |
| `config_sha` | SHA256 of the serialized `PAPER_CONFIG` dict |
| `n_reps` | Number of repetitions used |
| `paper_config` | Full `PAPER_CONFIG` snapshot |
| `elapsed_s` | Wall-clock seconds for the experiment |
| `output` | Relative path to this file |
| `hardware` | CPU count, RAM, instance type, Python version, hostname |

## Significance Tests (`_significance`)

Real-world experiments (`california_housing.json`, `breast_cancer.json`,
`superconductor.json`) also contain a `_significance` key with pairwise
Wilcoxon signed-rank tests, Cohen's d, Holm-Bonferroni corrected p-values,
and TOST equivalence tests between DASH and each baseline:

```json
"_significance": {
  "Single Best": {
    "rmse":      {"p": 0.003, "cohens_d": 1.2, "p_holm": 0.009},
    "rmse_tost": {"p1": 0.04, "p2": 0.02, "equivalent": false},
    "stability": {"diff": 0.613, "p": 0.0001, "ci_lo": 0.45, "ci_hi": 0.78}
  },
  ...
}
```

## Dataset Provenance (`_dataset`)

Real-world experiments also include a `_dataset` key recording the exact
sklearn version and data source used:

```json
"_dataset": {
  "sklearn_version": "1.4.0",
  "dataset":         "california_housing",
  "source":          "sklearn.datasets.fetch_california_housing"
}
```

## Known Limitations of Committed Results

Three files in this directory have `config_sha: "backfilled"` in their `_meta` block:

- `synthetic_linear_sweep.json`
- `first_mover_visualization.json`
- `k_sweep_independence.json`

These were produced on a SageMaker instance (commit `88b1bc3`) before the provenance
infrastructure existed. The `_meta` block was added post-hoc. Specifically:

| Field | Status | Meaning |
|---|---|---|
| `config_sha` | `"backfilled"` | Fingerprint reconstructed after the run, not captured at runtime |
| `hardware` | `"SageMaker run — hardware details not captured at time of run"` | Instance type and RAM not recorded |
| `elapsed_s` | `null` | Timing not captured (linear_sweep's 2151.7s is embedded in the data itself) |
| `code_sha` | `88b1bc3` | Real commit SHA — the code is fully traceable |
| `n_reps` | `50` | Confirmed from `len(acc_runs)` in the data |
| `paper_config` | correct | Verified against `dash_shap/config.py` at time of backfill |

**The data is valid.** The `"backfilled"` notation is an honest record of a metadata
limitation, not a data integrity problem. All subsequently produced results will have
full provenance captured automatically at runtime.

## Reproducing Any Result

1. Check `_meta.code_sha` in the JSON file.
2. Checkout that commit: `git checkout <code_sha>`.
3. Verify `_meta.paper_config` matches `dash_shap/config.py`.
4. Run: `python run_experiments_parallel.py --experiments <experiment_name>`.

If `_meta.code_dirty` is `true`, the result was produced from uncommitted
changes and may not be exactly reproducible from the recorded SHA alone.

## `PROVENANCE.md`

`PROVENANCE.md` is an append-only human-readable log created automatically
by `run_experiments_parallel.py`. Each run appends one entry. Entries from
dirty working trees are prefixed with **⚠ DIRTY**.

## `environment.json`

Written once at startup before any experiments run. Contains:
- `timestamp` — when the run started
- `hardware` — CPU, RAM, instance type, Python version
- `packages` — sorted `name==version` list from `importlib.metadata`

Use this to reproduce the exact software environment for a given run.
