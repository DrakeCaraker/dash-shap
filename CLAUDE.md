# DASH-SHAP

DASH (Diversified Aggregation of SHAP) produces stable feature importance explanations under feature collinearity by aggregating SHAP values across a diverse ensemble of independently trained models. Target venue: TMLR. Authors: Caraker, Arnold, Rhoads (2026).

## Directory Map

```
dash/
  core/           Five-stage pipeline modules
    population.py   Stage 1: train M XGBoost models with sampled hyperparameters
    filtering.py    Stage 2: keep models within epsilon of best validation score
    diversity.py    Stage 3: MaxMin greedy selection of K diverse models
    consensus.py    Stage 4: element-wise mean of K SHAP matrices
    diagnostics.py  Stage 5: FSI, IS plots, local disagreement maps
    pipeline.py     DASHPipeline orchestrator class
  baselines/      Nine comparison methods (single_best, large_single, ensemble_shap, etc.)
  experiments/    Synthetic data generators (linear & nonlinear DGP)
  evaluation/     Metrics: stability, DGP agreement, equity, statistical tests
  utils/          I/O helpers, SHAP utilities
notebooks/        Progressive experiment notebooks (demo_benchmark_{N}.ipynb)
tests/            pytest suite (test_pipeline, test_baselines, test_evaluation, test_synthetic)
docs/             API_REFERENCE.md, BENCHMARK_RESULTS.md
paper/            LaTeX source
```

## Key Entry Points

- `dash.core.pipeline.DASHPipeline` — main class, runs all 5 stages via `.fit()`
- `run_experiments.py` — CLI experiment runner (10 experiments, plotting, JSON output)
- `notebooks/demo_benchmark_6.ipynb` — **authoritative** benchmark notebook

## Canonical Configuration (PAPER_CONFIG)

```python
M = 200          # population size
K = 30           # selected models
N_REPS = 20      # repetitions per experiment
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
- **Notebook naming**: `demo_benchmark_{N}.ipynb` — higher N supersedes lower. **6 is authoritative**; 0–5 are historical
- **Tests**: `pytest` from repo root. ~47 tests across 4 files. No GPU required.
- **Parallelism** via `joblib` (n_jobs parameter on DASHPipeline)

## Running

```bash
pytest                                         # all tests
pytest tests/test_evaluation.py                # single file
python run_experiments.py                      # all 10 experiments
python run_experiments.py --experiments linear_sweep  # one experiment
```

## Git Hooks

Pre-push hook blocks `.pkl` files and files >1MB. Activate after cloning:

```bash
git config core.hooksPath .githooks
```

## Do NOT

- Commit `.pkl` files or anything in `checkpoints/`
- Track build artifacts (`dist/`, `build/`, `*.egg-info/`)
- Modify notebooks `demo_benchmark` through `demo_benchmark_5` (historical artifacts)
- Push notebooks with large embedded outputs (>1MB) — clear outputs first
- Use `dash` as a bare import in tests (shadows the package — use `from dash.core import ...`)
- Train models with high `colsample_bytree` (>0.5) in DASH population — defeats the diversity mechanism

## Key Results (for quick reference)

At rho=0.95 (20 reps): DASH stability=0.982 vs Single Best=0.953 vs LSM=0.930
Breast Cancer: DASH stability=0.933 vs Single Best=0.534 (+0.399)
Superconductor: DASH stability=0.965 vs Single Best=0.848 vs LSM=0.702

## Research Program

Paper 1 of 5 (see ROADMAP.md). Key claim: independence between models in the DASH population cancels path-dependent noise in feature attributions. See EXPERIMENT_GUIDE.md for full methodology and method descriptions.
