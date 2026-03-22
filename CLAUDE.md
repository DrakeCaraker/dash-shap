# DASH-SHAP

DASH (Diversified Aggregation of SHAP) produces stable feature importance explanations under feature collinearity by aggregating SHAP values across a diverse ensemble of independently trained models. Target venue: TMLR. Authors: Caraker, Arnold, Rhoads (2026).

## Directory Map

```
dash_shap/
  core/           Five-stage pipeline modules
    population.py   Stage 1: train M XGBoost models with sampled hyperparameters
    filtering.py    Stage 2: keep models within epsilon of best validation score
    diversity.py    Stage 3: MaxMin greedy selection of K diverse models
    consensus.py    Stage 4: element-wise mean of K SHAP matrices
    diagnostics.py  Stage 5: FSI, IS plots, local disagreement maps
    pipeline.py     DASHPipeline orchestrator class
  baselines/      Eight comparison baselines (single_best, large_single, ensemble_shap, etc.)
  experiments/    Synthetic data generators (linear & nonlinear DGP)
  evaluation/     Metrics: stability, DGP agreement, equity, statistical tests
  utils/          I/O helpers, SHAP utilities
notebooks/        Progressive experiment notebooks (demo_benchmark_{N}.ipynb)
tests/            pytest suite (test_pipeline, test_baselines, test_evaluation, test_synthetic)
docs/             API_REFERENCE.md, BENCHMARK_RESULTS.md, DIAGNOSTICS.md, RESEARCH_DIRECTIONS.md
docs/archive/     Historical documents (IMPLEMENTATION_PLAN.md, PEER_REVIEW.md)
paper/            LaTeX source
```

## Key Entry Points

- `dash_shap.core.pipeline.DASHPipeline` — main class, runs all 5 stages via `.fit()`
- `run_experiments.py` — CLI experiment runner (deprecated — use parallel runner; retained for historical provenance)
- `run_experiments_parallel.py` — **sole actively maintained entry point** (18 experiments, ~3-5x faster via population sharing + parallel SHAP)
- `notebooks/demo_benchmark_6.ipynb` — **authoritative (ArXiv)** interactive benchmark notebook
- `notebooks/demo_benchmark_7.ipynb` — **in development (TMLR)** interactive benchmark notebook
- `notebooks/demo_benchmark_7_parallel.ipynb` — **parallel fork** of notebook 7 (uses `run_experiments_parallel`)
- `notebooks/explore_experiment_results.ipynb` — interactive viewer for experiment output (works with both runners)

## Experiment Synchronization

- `run_experiments.py` is **deprecated** — retained only for historical provenance; do not extend or rely on it
- `run_experiments_parallel.py` is the **sole actively maintained non-interactive runner** — produces identical JSON output via population sharing and parallel SHAP; includes `k_sweep_independence` (not in sequential runner)
- `notebooks/demo_benchmark_7.ipynb` is the canonical **interactive** experimental pipeline — must produce the same results as the parallel runner
- `notebooks/explore_experiment_results.ipynb` visualizes experiment output interactively — works with the parallel runner

## Canonical Configuration (PAPER_CONFIG)

```python
M = 200          # population size
K = 30           # selected models
N_REPS = 50      # repetitions per experiment (ArXiv used 20; TMLR target is 50)
EPSILON = 0.08   # absolute filter threshold (synthetic)
DELTA = 0.05     # deduplication Spearman threshold
SEED = 42
# Real-world datasets use:
REAL_EPSILON = 0.05
epsilon_mode = 'relative'
```

## Pipeline Stages

1. **Population** — train M XGBoost models with random hyperparameters; `colsample_bytree` forced low (0.1–0.5) to break sequential residual dependency
2. **Filtering** — keep models within epsilon of best score (3 modes: absolute, relative, quantile)
3. **Diversity** — MaxMin greedy selection maximizing minimum pairwise cosine distance among importance vectors
4. **Consensus** — interventional TreeSHAP on each selected model, then element-wise mean
5. **Diagnostics** — Feature Stability Index (FSI), IS plots with 4 quadrants, local disagreement maps

## Conventions

- **Lazy imports** via `__getattr__` in all `__init__.py` files
- **4-way data split** in synthetic generators: X_train, X_val, X_explain (SHAP background), X_test (RMSE eval)
- **Checkpoint pattern** in notebooks: `save_checkpoint(name, data)` / `load_checkpoint(name)` writes `.pkl` to `checkpoints/`
- **Notebook naming**: `demo_benchmark_{N}.ipynb` — **6 is authoritative for ArXiv**; **7 is authoritative for TMLR (in development, not yet run)**
- **Current paper draft**: `paper/draft_v7_preprint.tex` (TMLR submission target; v6 was ArXiv-ready)
- **Tests**: `pytest` from repo root. No GPU required.
- **Parallelism** via `joblib` (n_jobs parameter on DASHPipeline)

## Running

```bash
pytest                                         # all tests
pytest tests/test_evaluation.py                # single file
pytest -m "not slow"                           # fast tests only
make test                                      # all tests (via Makefile)
make test-fast                                 # skip slow tests
make lint                                      # ruff check
make fmt                                       # ruff format
make typecheck                                 # mypy
make coverage                                  # pytest with 70% coverage floor
make rebase                                    # rebase on origin/main
python run_experiments.py                      # DEPRECATED — use parallel runner
python run_experiments.py --experiments linear_sweep  # DEPRECATED — use parallel runner
python run_experiments_parallel.py             # all 18 experiments (sole maintained runner)
python run_experiments_parallel.py --experiments linear_sweep
```

## Parallel Optimizations

`run_experiments_parallel.py` produces **identical results** to `run_experiments.py` via three optimizations:

1. **Population sharing** — DASH, RandomSelection, SingleBest(M=200), and NaiveTop-N share the same model population per rep (same seeds, same search space). Eliminates ~3x redundant training in the linear sweep.
2. **Parallel SHAP** — `n_jobs=-1` passed to `compute_consensus()` and all baselines for joblib-parallelized TreeSHAP computation.
3. **Vectorized stability** — `stability_bootstrap_ci()` uses pre-computed rank matrices and `np.corrcoef` instead of per-pair Spearman calls.

Core library changes are backward-compatible: `n_jobs=1` default preserves sequential behavior. `fit_from_population()` methods are additive (new API, existing `fit()` unchanged).

## Git Hooks

Pre-push hook blocks `.pkl` files and files >10MB, and warns if the branch has drifted behind `origin/main`. Activate after cloning:

```bash
git config core.hooksPath .githooks
```

### Drift Prevention

Three layers detect when a branch falls behind `main`:

1. **Pre-push hook** (`.githooks/pre-push`) — warns with commit count on every push (non-blocking)
2. **Session-start hook** (`.claude/hooks/session-start.sh`) — shows drift status when a Claude session begins
3. **CI freshness check** (`.github/workflows/ci.yml`, `freshness` job) — **blocks PR merge** when behind main (exit 1)

To fix a stale branch: `make rebase` then `git push --force-with-lease`.

### GitHub Branch Protection (Recommended)

Enable these settings in GitHub > Settings > Branches > Branch protection rules for `main`:

1. **Require status checks to pass before merging**: `freshness`, `lint`, `test`, `typecheck`
2. **Require branches to be up to date before merging**: checked
3. **Require pull request reviews before merging**: at least 1 approval
4. **Do not allow bypassing the above settings**: checked

## Notebooks

- **Canonical notebooks keep their outputs** — `demo_benchmark_6.ipynb` and `demo_benchmark_7_parallel.ipynb` outputs are empirical records and must not be cleared. Only clear outputs on scratch/dev notebooks.

## Pre-push Checklist

Before any `git push`, run the full local CI pipeline and fix all failures before pushing:

```bash
make lint && make fmt && make typecheck && make test-fast
```

Do not push partial fixes. CI failures on ruff quote style, mypy ignore codes, and sync-check regexes are avoidable locally.

## Do NOT

- Commit `.pkl` files or anything in `checkpoints/`
- Track build artifacts (`dist/`, `build/`, `*.egg-info/`)
- Push notebooks with large embedded outputs (>10MB) — clear outputs first
- Use `dash` as a bare import in tests (shadows the Plotly Dash package — use `from dash_shap.core import ...`)
- Train models with high `colsample_bytree` (>0.5) in DASH population — defeats the diversity mechanism

## Key Results (for quick reference)

All results sourced from `notebooks/demo_benchmark_6.ipynb` (canonical ArXiv source).

At rho=0.9 (20 reps): DASH stability=0.977 vs Single Best=0.958 vs LSM=0.938
At rho=0.95 (20 reps): DASH stability=0.977 vs Single Best=0.951 vs LSM=0.925
Breast Cancer: DASH stability=0.930 vs Single Best (M=200)=0.317 (+0.614)
Superconductor: DASH stability=0.962 vs Single Best=0.830 vs LSM=0.689
California Housing: DASH stability=0.982 vs Single Best=0.967 (+0.015)

> **Note:** `results/tables/asymmetric_dgp.json` must be regenerated by running `experiment_asymmetric_dgp()` via `run_experiments_parallel.py` — the file previously written by the sequential runner used an incompatible schema.

## Claude Code Workflow

### Slash Commands
- `/commit` — safe commit with pkl/large file guards (blocks `.pkl`, warns >500KB)
- `/checkpoint-clear` — list and selectively delete checkpoint/pkl files
- `/notebook-status` — summarize notebook states, flag large outputs, show canonical status
- `/paper-context` — load full research context (EXPERIMENT_GUIDE, BENCHMARK_RESULTS, ROADMAP) for writing tasks
- `/sync-check` — verify PAPER_CONFIG consistency across `run_experiments.py`, notebooks 6 & 7, and CLAUDE.md
- `/experiment-summary` — format results into markdown + LaTeX tables with provenance and regression checks

### Hooks
- **Pre-push** (git): blocks `.pkl` files and files >10MB (activate: `git config core.hooksPath .githooks`)
- **Stop** (user-level): prevents session end with uncommitted/unpushed changes
- **PreToolUse** (project-level): warns when editing notebooks >2MB (catches output bloat before commit)

## Research Program

Key claim: independence between models in the DASH population cancels path-dependent noise in feature attributions. See EXPERIMENT_GUIDE.md for full methodology and method descriptions.
