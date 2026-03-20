# Contributing to DASH-SHAP

Thank you for your interest in contributing! This guide covers development setup, testing, and how to extend the project.

## Development Setup

```bash
git clone https://github.com/DrakeCaraker/dash-shap.git
cd dash-shap
pip install -e ".[lightgbm]"      # installs dash_shap in editable mode + optional LightGBM
git config core.hooksPath .githooks  # activate pre-push hook (blocks .pkl and >10MB files)
```

Verify everything works:

```bash
pytest -m "not slow"   # fast test suite (~30 seconds)
make lint              # ruff check
```

## Running Tests

```bash
pytest                    # all tests
pytest -m "not slow"      # fast tests only (skip long experiment runs)
pytest tests/test_pipeline.py  # single file
make test                 # all tests via Makefile
make test-fast            # skip slow tests
make coverage             # tests with 70% coverage floor
```

## Lint, Format, and Type Checking

```bash
make lint       # ruff check (linting)
make fmt        # ruff format (auto-format)
make typecheck  # mypy
```

All three must pass before merging.

## Project Structure

```
dash_shap/
  core/           Five-stage pipeline modules
    population.py   Stage 1: train M XGBoost models
    filtering.py    Stage 2: performance filter (epsilon)
    diversity.py    Stage 3: MaxMin greedy diversity selection
    consensus.py    Stage 4: element-wise mean SHAP matrices
    diagnostics.py  Stage 5: FSI, IS plots, local disagreement maps
    pipeline.py     DASHPipeline orchestrator
  baselines/      Eight comparison baselines
  experiments/    Synthetic data generators (linear & nonlinear DGP)
  evaluation/     Metrics: stability, DGP agreement, equity, statistical tests
  utils/          I/O helpers, SHAP utilities
notebooks/        Progressive benchmark notebooks
tests/            pytest suite
docs/             API_REFERENCE.md, BENCHMARK_RESULTS.md, DIAGNOSTICS.md
paper/            LaTeX source
```

## How to Add a New Baseline

1. Copy an existing baseline from `dash_shap/baselines/` (e.g., `single_best.py`) as a starting point
2. Implement the same interface:
   - `__init__(self, ...)` with any baseline-specific parameters
   - `fit(self, X_train, y_train, X_val, y_val, X_ref=None)` → sets `self.global_importance_`
3. Register it in `dash_shap/baselines/__init__.py` (add to `__all__` and `__getattr__`)
4. Add a test in `tests/test_baselines.py`
5. Add a call in `run_experiments.py` where appropriate

```bash
# Access baselines via:
from dash_shap.baselines import SingleBestBaseline, EnsembleSHAPBaseline
```

## How to Add a New Experiment

1. Write a new function `experiment_my_name()` in `run_experiments.py`, following the existing pattern:
   - Use `PAPER_CONFIG` for canonical parameter values
   - Call `save_json(results, "my_name")` at the end
2. Add a one-line docstring describing the experiment
3. Register it in the `EXPERIMENTS` dict at the bottom of `run_experiments.py`
4. Add it to the CLI `--experiments` help string in the argparse block

```bash
python run_experiments.py --experiments my_name
```

## Commit Conventions

- **No `.pkl` files** — the pre-push hook blocks them. Use `save_json` for results.
- **Clear notebook outputs** before committing notebooks — large embedded outputs are blocked by the pre-push hook (>10MB warning).
- **No secrets** — never commit API keys, credentials, or private data.
- Commit messages should describe *why*, not just *what*.

## Submitting Changes

1. Branch off `main`: `git checkout -b my-feature`
2. Make changes; ensure `pytest -m "not slow"` and `make lint` pass
3. Push and open a pull request against `main`
4. The CI suite (lint, test, typecheck, freshness check) must pass before merge

For questions about the research methodology, see `EXPERIMENT_GUIDE.md` and `docs/DIAGNOSTICS.md`.
