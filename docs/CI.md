# CI Pipeline

The CI workflow (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request targeting `main`. All jobs run in parallel on `ubuntu-latest`.

## Jobs

### freshness (PR only)

Checks whether the PR branch has fallen behind `main`. Emits a GitHub warning annotation with the commit count if stale. Non-blocking — the job always passes.

**Trigger:** Pull requests only.

### lint

Runs [ruff](https://docs.astral.sh/ruff/) for static analysis and formatting.

| Step | Blocking? | What it checks |
|------|-----------|----------------|
| `ruff check .` | Yes | Real bugs: syntax errors, undefined names, unused variables (rules `F`, `E4`, `E7`, `E9`) |
| `ruff format --check .` | No (advisory) | Code formatting consistency; emits a warning if files would be reformatted |

Configuration lives in `pyproject.toml` under `[tool.ruff]`. Notebooks are excluded. Per-file ignores suppress intentional patterns like unused imports in `__init__.py`.

**Fix locally:**
```bash
ruff check . --fix   # auto-fix lint issues
ruff format .        # auto-format
```

### typecheck

Runs [mypy](https://mypy.readthedocs.io/) on `dash_shap/` with `--ignore-missing-imports`. Currently advisory — the job always passes (`|| true`) to avoid blocking on untyped third-party dependencies.

Configuration lives in `pyproject.toml` under `[tool.mypy]`.

### test

Runs the full `pytest` suite on **Python 3.9 and 3.12** with coverage measurement.

| Detail | Value |
|--------|-------|
| Coverage tool | `pytest-cov` |
| Coverage target | 70% minimum (`--cov-fail-under=70`) |
| Coverage report | `term-missing` (printed in logs) |
| Artifact | `.coverage` file uploaded for Python 3.12 |
| Pip caching | Enabled via `actions/setup-python` cache |

**Run locally:**
```bash
pytest -v --cov=dash_shap --cov-report=term-missing
```

### notebook-validation

Two checks on all `.ipynb` files in `notebooks/`:

1. **Parse check** — Opens every notebook with `nbformat` and verifies it parses as valid nbformat v4. Catches corrupted JSON, merge conflicts in notebooks, etc.

2. **Size check** — Measures file size of each notebook:
   - **>10 MB** — Fails the job. Notebooks this large contain embedded outputs that should be cleared before committing.
   - **>2 MB** — Emits a warning. Consider clearing outputs.
   - **<= 2 MB** — Passes silently.

### sync-check

Verifies that `PAPER_CONFIG` parameters are consistent between `run_experiments.py` and `run_experiments_parallel.py`. Checked parameters:

| Parameter | Canonical value |
|-----------|----------------|
| `M` | 200 |
| `K` | 30 |
| `N_REPS` | 50 |
| `EPSILON` | 0.08 |
| `DELTA` | 0.05 |
| `SEED` | 42 |

Any mismatch emits a `::error::` annotation and fails the job.

### file-guards

Prevents accidental commits of forbidden files in PRs:

1. **Pickle guard** — Fails if any `.pkl` file was added or modified in the PR diff. Checkpoint files must stay out of version control.

2. **Large file guard** — Fails if any changed file exceeds 10 MB. Mirrors the pre-push git hook.

Both checks compare against `origin/main...HEAD` to only inspect files changed in the PR.

### security

Runs [pip-audit](https://pypi.org/project/pip-audit/) to scan installed dependencies for known vulnerabilities (CVEs). Currently advisory (`|| true`) — the job logs findings but does not block merges.

**Run locally:**
```bash
pip install pip-audit
pip-audit --strict
```

### latex-lint

Runs [chktex](https://www.nongnu.org/chktex/) on the latest paper draft (`paper/draft_v*_preprint.tex`). Catches common LaTeX issues like missing braces, bad spacing, and punctuation problems. Currently advisory — output is informational only.

### perf-smoke

Runs a lightweight end-to-end pipeline benchmark to catch performance regressions.

| Parameter | Value |
|-----------|-------|
| Data | `generate_synthetic_linear(N=500, P=10)` |
| Pipeline | `M=10, K=5, epsilon=0.08` |
| Ceiling | 120 seconds |
| Timeout | 5 minutes (hard kill) |

Trains 10 XGBoost models, filters, selects 5, computes SHAP, and returns consensus. If the pipeline takes longer than 120s on a CI runner, the job fails — indicating a performance regression in the core stages.

## Job Summary

| Job | Blocking? | Python | Pip cache | Purpose |
|-----|-----------|--------|-----------|---------|
| freshness | No | — | — | Warn on stale PR branches |
| lint | Yes (lint) / No (format) | 3.12 | — | Static analysis |
| typecheck | No | 3.12 | Yes | Type checking |
| test | Yes | 3.9, 3.12 | Yes | Unit tests + coverage |
| notebook-validation | Yes (>10MB) / No (>2MB) | 3.12 | — | Notebook integrity |
| sync-check | Yes | 3.12 | — | Config consistency |
| file-guards | Yes | — | — | Block pkl + large files |
| security | No | 3.12 | Yes | Dependency CVE scan |
| latex-lint | No | — | — | Paper quality |
| perf-smoke | Yes | 3.12 | Yes | Performance regression |

## Configuration Files

- **CI workflow:** `.github/workflows/ci.yml`
- **Ruff config:** `pyproject.toml` → `[tool.ruff]`
- **Mypy config:** `pyproject.toml` → `[tool.mypy]`
- **Pytest config:** `pyproject.toml` → `[tool.pytest.ini_options]`
- **Pre-push hook:** `.githooks/pre-push` (local complement to CI file-guards)
