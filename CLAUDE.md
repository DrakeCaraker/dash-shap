# DASH-SHAP

DASH (Diversified Aggregation of SHAP) produces stable feature importance explanations under feature collinearity by aggregating SHAP values across a diverse ensemble of independently trained models. Target venue: TMLR. Authors: Caraker, Arnold, Rhoads (2026).

## Non-Negotiable Rules

These apply to every session, no exceptions:

1. **Never push directly to main.** Always create a feature branch (`feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `perf/<topic>`) and open a PR. Verify with `git branch --show-current` before the first commit.

2. **Keep commits atomic.** One concern per commit, one concern per branch. Provenance fixes, performance changes, and result uploads are always separate branches. If it would require two sentences to describe the commit, it should be two commits.

3. **Read before planning.** Verify file sizes, API signatures, and module structure by reading the actual code before proposing any plan. Never assume from file names alone.

4. **Canonical notebooks retain their outputs.** `demo_benchmark_6.ipynb` and `demo_benchmark_7_parallel.ipynb` outputs are empirical records — never clear them. Only clear outputs on scratch/dev notebooks.

5. **Bound background processes.** Never spawn more than one test process at a time. If a command seems hung, read its background task output before retrying. Use `pgrep -f pytest` to check for running processes.

6. **Capture corrections immediately.** When the user redirects your approach ("no", "don't", "stop", "instead", "actually"), save a feedback memory *before* continuing with the corrected approach. Check existing memories first to avoid duplicates. Only capture genuine corrections to approach, not routine task requests.

7. **Close superseded PRs.** When a new PR includes the changes from earlier PRs, close the earlier ones with a comment referencing the replacement. Don't leave overlapping PRs open.

8. **Scripts before remote work.** Commit all tooling scripts (setup, run, finalize) to main before cloning to a remote instance. Do not relay commands through chat — copy-paste across terminals causes syntax errors.

9. **Smoke test before long runs.** Run `python run_experiments_parallel.py --smoke --experiments linear_sweep` before any multi-hour experiment run. The smoke test validates the full serialization pipeline in ~1 second and catches bugs that would otherwise crash after hours of compute.

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
  baselines/      Eleven comparison baselines (single_best, large_single, ensemble_shap, stochastic_retrain, random_selection, random_forest, naive_averaging, permutation_importance, nn_baselines, lightgbm_single)
  experiments/    Synthetic data generators (linear & nonlinear DGP)
  evaluation/     Metrics: stability, DGP agreement, equity, statistical tests
  utils/          I/O helpers, SHAP utilities
notebooks/        Progressive experiment notebooks (demo_benchmark_{N}.ipynb)
tests/            pytest suite (test_pipeline, test_baselines, test_evaluation, test_synthetic)
docs/             API_REFERENCE.md, BENCHMARK_RESULTS.md, DIAGNOSTICS.md, GETTING_STARTED.md, CI.md
docs/private/     Encrypted: roadmap.md, research_directions.md, tmlr_update_plan, interim results
docs/archive/     Historical documents (IMPLEMENTATION_PLAN.md, PEER_REVIEW.md)
paper/            LaTeX source
```

## Key Entry Points

- `dash_shap.core.pipeline.DASHPipeline` — main class, runs all 5 stages via `.fit()`
- `run_experiments.py` — CLI experiment runner (deprecated — use parallel runner; retained for historical provenance)
- `run_experiments_parallel.py` — **sole actively maintained entry point** (20 experiments, ~3-5x faster via population sharing + parallel SHAP)
- `notebooks/demo_benchmark_6.ipynb` — **authoritative (ArXiv)** interactive benchmark notebook
- `notebooks/archive/demo_benchmark_7.ipynb` — **archived, superseded by parallel version**
- `notebooks/demo_benchmark_7_parallel.ipynb` — **canonical (TMLR)** interactive benchmark notebook (uses `run_experiments_parallel`)
- `notebooks/explore_experiment_results.ipynb` — interactive viewer for experiment output (works with both runners)

## Experiment Synchronization

- `run_experiments.py` is the canonical **non-interactive** experimental pipeline
- `run_experiments_parallel.py` is the **performance-optimized fork** — produces identical JSON output via population sharing and parallel SHAP
- `notebooks/demo_benchmark_7_parallel.ipynb` is the canonical **interactive** experimental pipeline — both must produce the same results
- `notebooks/explore_experiment_results.ipynb` visualizes experiment output interactively — works with both runners

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
- **Notebook naming**: `demo_benchmark_{N}.ipynb` — **6 is authoritative for ArXiv**; **7_parallel is authoritative for TMLR**; 7 is archived in `notebooks/archive/`
- **Current paper draft**: `paper/draft_v7_preprint.tex` (TMLR submission target; v6 was ArXiv-ready)
- **Tests**: `pytest` from repo root. No GPU required.
- **Parallelism** via `joblib` (n_jobs parameter on DASHPipeline)

## Running

```bash
pytest                                         # all tests
pytest tests/test_evaluation.py                # single file
pytest -m "not slow"                           # fast tests only
make setup                                     # install deps, activate hooks, verify tools
make test                                      # all tests (via Makefile)
make test-fast                                 # skip slow tests
make lint                                      # ruff check
make fmt                                       # ruff format
make typecheck                                 # mypy
make coverage                                  # pytest with 70% coverage floor
make rebase                                    # rebase on origin/main
python run_experiments.py                      # DEPRECATED — use parallel runner
python run_experiments.py --experiments linear_sweep  # DEPRECATED — use parallel runner
python run_experiments_parallel.py             # all 20 experiments (sole maintained runner)
python run_experiments_parallel.py --experiments linear_sweep
python scripts/check_notebook_ids.py          # flag unnamed code cells before editing sessions
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

1. **Require status checks to pass before merging**: `freshness`, `lint`, `test`, `typecheck`, `block-results-to-main`
2. **Require branches to be up to date before merging**: checked
3. **Require pull request reviews before merging**: at least 1 approval
4. **Do not allow bypassing the above settings**: checked

## Notebooks

- **Canonical notebooks retain outputs** — see Non-Negotiable Rules #4. Only clear outputs on scratch/dev notebooks.

## Pre-push Checklist

Before any `git push`, run fast local checks:

```bash
make lint && make test-fast
```

Formatting (`make fmt`) and type checking (`make typecheck`) are handled by CI — formatting is auto-fixed and committed, typecheck runs on every PR. Run them locally only when touching type signatures or debugging a CI failure.

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

## SageMaker Run Protocol

Long-running SageMaker experiments have specific branch/provenance rules to prevent muddying.

### Before starting a run
1. Ensure all runner code and tooling scripts are on `main` first — no code changes on the results branch that aren't also on main
2. Run `python run_experiments_parallel.py --smoke --experiments linear_sweep` to validate the serialization pipeline (~1 second)
3. Create the results branch from main: `git checkout -b results/sagemaker-run-YYYYMMDD`
4. Tag the starting commit: `git tag run-tmlr-YYYYMMDD-start <sha> && git push origin run-tmlr-YYYYMMDD-start`

### During a run
- Results branch gets **data-only commits** (JSON files, figures, `environment.json`)
- Code fixes needed mid-run: feature branch → PR → main → cherry-pick to results branch with note in commit message
- Never commit checkpoint `.pkl` files (pre-push hook blocks this)

### After run completes
1. Run `python scripts/backfill_meta.py` on SageMaker — ensures all JSONs have `_meta` hardware blocks
2. Create a completion commit: `chore: mark run-YYYYMMDD complete — all N experiments finished`
3. Tag the results branch: `git tag results-YYYYMMDD-final`
4. Open a **data-only PR to main** — only JSON/figure additions; code changes already landed via their own PRs
5. Verify provenance, then add the `run-complete` label to the PR — CI blocks merge until this label is present
6. Freeze the branch — no further commits after the PR merges

### Key invariant
> Every line of code on the results branch must also exist on main. The results branch adds only data files.

## Claude Code Workflow

### Slash Commands
- `/commit` — safe commit with pkl/large file guards (blocks `.pkl`, warns >500KB)
- `/checkpoint-clear` — list and selectively delete checkpoint/pkl files
- `/notebook-status` — summarize notebook states, flag large outputs, show canonical status
- `/paper-context` — load full research context (EXPERIMENT_GUIDE, BENCHMARK_RESULTS, ROADMAP) for writing tasks
- `/paper-context-arxiv` — load ArXiv/Zenodo v6 context for responding to ArXiv comments
- `/sync-check` — verify PAPER_CONFIG consistency across `run_experiments.py`, notebooks 6 & 7, and CLAUDE.md
- `/experiment-summary` — format results into markdown + LaTeX tables with provenance and regression checks
- `/ci-fix` — autonomous CI fix loop: runs ruff/mypy/pytest incrementally, fixes failures by type, loops until green (max 5 iterations)
- `/audit` — parallel four-dimension repo audit (notebooks, preprint parity, sensitive data, release readiness) → merged report in `docs/audit/`
- `/vet` — deep self-auditing analysis loop: 3-round factual/reasoning/omissions protocol before presenting results
- `/safe-refactor <target>` — test-gated refactoring: writes characterization tests, applies one change at a time, auto-rollbacks on failure
- `/pr` — standardized branch→commit→push→PR workflow with lint gates and main-branch guard
- `/self-improve` — analyze feedback memories and propose promotions to CLAUDE.md rules or hooks (promotion ladder: memory → rule → hook)
- `/new-work` — branch hygiene and session scoping before code changes
- `/health-check` — assess project AI-dev maturity and recommend improvements
- `/bootstrap` — set up AI-assisted dev infrastructure (hooks, commands, settings)
- `/run-tests` — run full pytest suite with verbose output and summary

### Hooks (fully automated)
- **Pre-push** (git): blocks `.pkl` files and files >10MB (activate: `git config core.hooksPath .githooks`)
- **CI: Branch guards** — blocks PRs from `results/*` → `main` unless `run-complete` label is present; blocks code files in PRs targeting `results/*` branches
- **Stop: CI gate** — runs lint/typecheck/test if source files changed; suggests `/ci-fix` on failure
- **Stop: Feedback capture** — reminds to save uncaptured user corrections as feedback memories
- **Stop: Self-improve check** — classifies feedback memories and surfaces promotion proposals if actionable
- **PostToolUse** — auto-format Python, sync-check config files, lint LaTeX on every edit
- **PreToolUse** — warns when editing notebooks >2MB (catches output bloat before commit)
- **PreCompact** — reminds to save in-progress context before context compression

### Hook Behaviors (what to do when hooks fire)

These instructions describe what Claude should do when Stop/PreCompact hooks run. The hooks themselves only print short status messages — the detailed behavior lives here.

- **Feedback capture** (Stop): Before ending, check if any user corrections from this session still need to be saved as feedback memories (Non-Negotiable Rule #6). Only capture genuine approach corrections, not routine requests. Check existing memories first to avoid duplicates.
- **Self-improve check** (Stop): If the hook reports 3+ feedback memories, read each one, check if it duplicates a CLAUDE.md rule (delete if so), and if any pattern has appeared 2+ times, propose promoting it to a rule. Show the user what was found and ask before making changes.
- **CI gate** (Stop): If CI checks fail, offer to run `/ci-fix` to auto-repair. If the user declines, remind them to run it next session. Do not push code with failing checks.
- **PreCompact**: When context compression starts, save any in-progress task state, critical file paths, current branch, and uncommitted decisions to the plan file or memory so they survive compression.

### Smart Suggestions

Proactively suggest these commands when the conditions are met. Explain briefly why, then ask for confirmation. Never run gated commands without asking.

- **Suggest `/new-work`** when: session starts on `main`, or the user describes a new task without scoping it. Say: *"Want me to create a branch and scope this work first?"*
- **Suggest `/commit`** when: a logical unit of work is complete and there are uncommitted changes. Say: *"That looks complete — want me to checkpoint this?"*
- **Suggest `/ci-fix`** when: a lint, type, or test error occurs mid-session. Say: *"CI checks failed — want me to auto-fix?"*
- **Suggest `/pr`** when: the branch has commits, all checks pass, and the user says the work is done. Say: *"Ready to open a PR?"*
- **Suggest `/safe-refactor`** when: the user asks to restructure, rename, or reorganize code touching multiple files. Say: *"This touches multiple files — want me to use test-gated refactoring so we can rollback safely?"*
- **Suggest `/audit`** when: the user mentions release readiness, submission prep, or asks about project health. Say: *"Want me to run a full audit?"*
- **Suggest `/paper-context`** when: the user asks about paper writing, results, or TMLR submission. Load it automatically.
- **Suggest `/notebook-status`** when: the user is about to edit a notebook. Check canonical status first.
- **Suggest `/checkpoint-clear`** when: session-start reports stale checkpoint files.
- **Suggest `/vet`** when: presenting analysis results, making claims about experimental data, or drafting paper text. Say: *"Want me to audit these findings before we proceed?"*
- **Suggest `/experiment-summary`** when: new experiment results arrive from SageMaker or notebooks. Say: *"New results available — want me to format them?"*
- **Suggest `/sync-check`** when: PAPER_CONFIG values are changed, or before submission prep. Say: *"Config may have drifted — want me to check consistency?"*

## Research Program

Key claim: independence between models in the DASH population cancels path-dependent noise in feature attributions. See EXPERIMENT_GUIDE.md for full methodology and method descriptions.

## Related Repositories

- **[dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean)** — Lean 4 formalization of the attribution impossibility theorem (Paper 3, targeting NeurIPS 2026). Contains 36 Lean files with formal proofs that no feature ranking can simultaneously be faithful, stable, and complete under collinearity. Uses axioms verified by `paper/proofs/verify_lemma6_algebra.py` in this repo.
