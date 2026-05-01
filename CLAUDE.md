# DASH-SHAP

DASH (Diversified Aggregation for Stable Hypotheses) produces stable feature importance explanations under feature collinearity by aggregating SHAP values across a diverse ensemble of independently trained models. Target venue: TMLR. Authors: Caraker, Arnold, Rhoads (2026).

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

10. **Verify agent findings before acting.** When an Explore or research agent reports on code structure, file contents, or repo state, spot-check 2-3 key claims by reading the actual files before building a plan on top of them. Agents can read stale code, miss branches, or misidentify patterns.

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

- `dash_shap.check()` — **simplest API**: 3-line stability check, trains M=25 models, returns `CheckResult` with `.report()`, `.plot()`, `.dash_importance()`
- `dash_shap.core.pipeline.DASHPipeline` — full 5-stage pipeline class, runs all stages via `.fit()`
- `run_experiments.py` — CLI experiment runner (deprecated — use parallel runner; retained for historical provenance)
- `run_experiments_parallel.py` — **sole actively maintained entry point** (18 default experiments, ~3-5x faster via population sharing + parallel SHAP)
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
- **Current paper drafts**: `paper/draft_v7_tmlr.tex` (TMLR submission, anonymous); `paper/draft_v8_reviewer_response.tex` (reviewer response revision with colsample ablation, enrichment theory). v7_preprint is the de-anonymized ArXiv version; v6 was ArXiv-ready.
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
python run_experiments_parallel.py             # all 18 default experiments (sole maintained runner)
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


### Git-Crypt

`docs/private/` is encrypted with git-crypt. Be aware that git-crypt filters can cause phantom unstaged changes during rebase. When encountering unexpected unstaged changes in encrypted files, check if they are git-crypt artifacts (`git-crypt status`) before attempting fixes.

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


## Output Preferences

- When the user asks to see command output, show it raw. Do not summarize or paraphrase terminal output unless explicitly asked to summarize.
- When executing a sequence of commands, show the actual output of each. Do not elide or collapse output into prose.
- If output is very long (>100 lines), show the first and last 20 lines with a count of omitted lines — but never fabricate output.

## Shell Commands

- Prefer single-quoted heredocs (`<<'EOF'`) over double-quoted to avoid escaping issues.
- On macOS, avoid `!` in unquoted strings (history expansion). Use `sed` or Python scripts over complex inline shell escaping.
- When working on a remote instance (SageMaker, EC2), confirm the shell (`bash` vs `sh`) before using bash-specific features like arrays or `[[ ]]`.
- For multi-line file writes, prefer `cat > file << 'EOF'` over chained `echo` or `sed` insertions.

## Communication Style

- When told to stay silent or just execute, do not narrate. Execute without explanation unless errors occur.
- Do not restate what was just done after completing a task. The diff and commit message speak for themselves.
- When the user gives a short directive ("fix it", "merge", "yes"), act immediately. Do not ask for confirmation of something already confirmed.

## Agent Usage

- When spawning sub-agents, ensure they have the necessary tool permissions. If a sub-agent is blocked on permissions, surface this immediately.
- Sub-agents working on files outside the current repo (e.g., `dash-impossibility-lean`) should use Bash for writes — PostToolUse hooks from the current project interfere with Edit/Write on other repos.
- Always verify agent findings before building plans on them (Non-Negotiable Rule #10).

## Key Results (for quick reference)

All results sourced from v7 SageMaker run (50 reps, PAPER_CONFIG). See `docs/BENCHMARK_RESULTS.md` for full tables.

At rho=0.9 (50 reps): DASH stability=0.9767 vs Single Best=0.9577 vs LSM=0.9381
At rho=0.95 (50 reps): DASH stability=0.9774 vs Single Best=0.9515 vs LSM=0.9267
Breast Cancer (50 reps): DASH=0.925 vs SB=0.376 vs SR=0.862 (+0.063 DASH-SR gap)
Superconductor (50 reps): DASH=0.964 vs SB=0.840 vs LSM=0.721
California Housing (50 reps): DASH=0.978 vs SB=0.969 (+0.009, n.s.)

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

## Comprehensive Results Reference

**Full reference:** [docs/RESEARCH_REFERENCE.md](docs/RESEARCH_REFERENCE.md) — exhaustive documentation of every proved theorem, experiment, validation, diagnostic, and their interconnections. Read that file for methodology, provenance, implications, and caveats. The summary below is for quick lookup.

### Related Repositories

| Repo | Purpose | Lean files | Theorems | Axioms | Sorry |
|------|---------|-----------|----------|--------|-------|
| **[dash-shap](https://github.com/DrakeCaraker/dash-shap)** (this repo) | DASH method, experiments, PyPI package | — | — | — | — |
| **[dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean)** | Attribution impossibility (NeurIPS 2026) | 58 | 357 | 6 | 0 |
| **[ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility)** | Universal impossibility + physics (FoP/Nature) | 38 | 482 | 13 | 0 |

### Proved Theorems (Lean 4, zero sorry)

#### Attribution Impossibility (dash-impossibility-lean — NeurIPS scope)

**Core impossibility (zero axioms):**
- `attribution_impossibility` (Trilemma.lean) — no ranking is simultaneously faithful, stable, and complete under interchangeable components. THE core result.
- `attribution_impossibility_weak` — weaker variant with relaxed assumptions.

**Bilemma (zero axioms):**
- `bilemma_of_compatible_eq` (Bilemma.lean) — F+S impossible for maximally incompatible explanation spaces.
- `all_or_nothing` — explanations are either perfectly faithful (unstable) or perfectly stable (unfaithful). No smooth tradeoff.
- `rashomon_unfaithfulness` — any stable method is unfaithful at ≥1 of every 2 Rashomon witnesses.
- `shap_sign_bilemma`, `feature_selection_bilemma`, `counterfactual_bilemma` — three constructive ML instances (Bool/Unit, zero axioms).

**Tightness classification:**
- `tightness_dichotomy` (BeyondBinary.lean) — neutral element exists ↔ F+S achievable. Complete classification.
- `coverageConflict_implies_no_neutral` — coverage conflict diagnoses collapsed tightness.
- `neutral_implies_FS_achievable` — enrichment restores F+S.

**Design space:**
- `design_space_theorem` (DesignSpace.lean) — every attribution method is Family A (faithful+decisive, unstable) or Family B (faithful+stable, indecisive). No Family C.
- `family_a_or_family_b` (DesignSpaceFull.lean) — exhaustive classification.
- `no_complete_faithful_ranking` — completeness forces instability.

**DASH optimality:**
- `dash_unique_pareto_optimal` (ParetoOptimality.lean) — DASH is the unique Pareto-optimal resolution.
- `pareto_frontier_dichotomy` — exactly two non-dominated strategies.
- `dash_pareto_dominance_within_group` — DASH dominates within correlated groups.
- `consensus_variance_from_independence` (VarianceDerivation.lean) — variance decays as σ²/M (Cramér-Rao).
- `double_M_halves_variance_derived` — doubling M halves variance.
- `weighted_variance_ge_consensus_variance` — no weighted scheme beats equal weighting.

**Model-class instantiation:**
- `split_gap_exact` (SplitGap.lean) — GBDT attribution ratio 1/(1-ρ²), diverges as ρ→1.
- Lasso.lean — Lasso ratio ∞ (sign instability for all ρ>0).
- NeuralNet.lean — NN conditional violations under symmetry breaking.
- RandomForest.lean — RF bounded O(1/√T), convergent (independence by construction).

**Quantitative bounds:**
- `alpha_faithful_bound` (AlphaFaithful.lean) — ε-approximate faithfulness bound.
- `stable_ranking_half_unfaithful` — stable rankings are ≥50% unfaithful.
- `attribution_prob_half` (UnfaithfulQuantitative.lean) — probability bound for unfaithfulness.

**Extensions:**
- FairnessAudit.lean — fairness audit impossibility under collinearity.
- MechInterp.lean — mechanistic interpretability ceiling (≥50% circuit disagreement).
- ModelSelection.lean, CausalDiscovery.lean — additional instances.
- RashomonUniversality.lean, RashomonInevitability.lean — Rashomon is generic.
- FlipRate.lean, GaussianFlipRate.lean — flip rate prediction formulas.

#### Universal Impossibility (ostrowski-impossibility — Nature/FoP scope)

**Bilemma extensions:**
- `approximate_bilemma` (ApproximateBilemma.lean) — F+S incompatibility survives ε-approximation at every tolerance.
- `quantitative_bilemma` — unfaith₁ + unfaith₂ ≥ Δ − δ (triangle inequality). Tight.
- `exact_bilemma_from_quantitative` — exact bilemma as special case of quantitative.

**Enrichment theory:**
- `forced_resolution_complete` (EnrichmentForcedResolution.lean) — collapsed tightness can ONLY be resolved by enrichment. DASH is the unique structural resolution class.
- `prime_collapsed_tightness` — prime-indexed collapsed tightness.
- `enrichment_cumulative_unfaithfulness_unbounded` (EnrichmentFunctor.lean) — cost of enrichment grows without bound.

**Social choice:**
- Arrow's impossibility (GeneralTheory.lean) — proved from scratch, IIA decomposition, 2 voters 3 alternatives. First Lean 4 proof.
- May's theorem — majority rule for binary alternatives. Proved from scratch.

**Physics application:**
- Ostrowski bridge to Mathlib (`Rat.AbsoluteValue.equiv_real_or_padic`).
- Freund-Witten three-fold zeta symmetry (FreundWitten.lean).
- Adelic resolution uniqueness (AdelicResolution.lean).
- Physics bilemma: spacetime geometry impossibility via Ostrowski classification.

**Enrichment stack:**
- Physical stack depth ≥ 4 (quantum, adelic, black holes, spacetime emergence).
- `kthBit` construction proving unbounded abstract depth.
- Gödel's incompleteness from `hasGoedelProperty` (weaker than diagonal lemma).

**Cross-domain:**
- Quantum contextuality: Bell/CHSH (16-strategy enumeration), Kochen-Specker (Peres-Mermin).
- Navier-Stokes regularity as conditional tightness + Reynolds parameterization.
- Circuit complexity: depth-2 parity impossibility.
- Diophantine (DPRM) trilemma + Selmer curve.
- Langlands functoriality, GL(2), GL(n) instances.
- Unified N-property meta-theorem (`AbstractImpossibilityN` for any N).

**MI quantitative bridge (universal repo):**
- `mi_implies_positive_gap` — MI > 0 → ∃ models with opposite-sign attributions.
- `total_unfaithfulness_bound` — triangle inequality on unfaithfulness.
- `mi_quantitative_unfaithfulness` — MI > 0 → any stable explanation has error ≥ Δ/2. The exact boundary.

### Completed Experiments (all 50 reps, PAPER_CONFIG, SageMaker v7)

All results in `results/tables/*.json`. Source: `run_experiments_parallel.py`.

| # | Experiment | JSON file | Key finding |
|---|-----------|-----------|-------------|
| 1 | Linear correlation sweep | `synthetic_linear_sweep.json` | DASH stability flat 0.973-0.977 across ρ=0.0-0.95; SB degrades 0.972→0.952 |
| 2 | Overlapping correlation | `overlapping.json` | DASH's largest advantage: +0.079 stability, +0.156 top-k5 over SB |
| 3 | Nonlinear DGP sweep | `nonlinear_sweep.json` | DASH>SR at ρ≥0.9 (0.887 vs 0.857, CIs non-overlapping); all methods degrade |
| 4a | California Housing | `california_housing.json` | DASH 0.978 vs SB 0.969 (+0.009, p=0.063 n.s.) — mild collinearity |
| 4b | Breast Cancer | `breast_cancer.json` | DASH 0.925 vs SB 0.376 (+0.549) — largest improvement, 21 pairs \|r\|>0.9 |
| 4c | Superconductor | `superconductor.json` | DASH 0.964 vs SB 0.840 (+0.124); RS/NTN slightly beat DASH (81 features) |
| 5 | Epsilon sensitivity | `epsilon_sensitivity.json` | Stability varies <0.005 across ε∈{0.03-0.10}. Robust. |
| 6 | Ablation (M, K, ε, δ) | `ablation.json` | M insensitive past 100; K saturates at 20; δ sensitive above 0.05 |
| 7 | Variance decomposition (marginal) | `variance_decomposition.json` | SB: 58% model variance; DASH: 21% model variance |
| 8 | First-mover visualization | `first_mover_visualization.json` | SB/LSM concentrate; DASH distributes within group |
| 9 | First-mover bias isolation | `first_mover_bias.json` | Concentration converges at M≥500 |
| 10 | Table 2 extended baselines | `table2_baselines.json` | 5 additional methods at ρ=0.9 |
| 11 | Background sensitivity | `background_sensitivity.json` | Stability Δ<0.0002 across B∈{50-500}. Not critical. |
| 12 | Asymmetric DGP | `asymmetric_dgp.json` | DASH highest passive leak (0.089 vs SB 0.068) — equity tradeoff |
| 13 | Crossed ANOVA (7×7) | `variance_decomposition_crossed.json` | SB: 40.6% model noise → DASH: 16.2%. 60% reduction. |
| 14 | K sweep independence | `k_sweep_independence.json` | Stability plateaus at K≈20. DASH fails at K=1. |
| 15 | Colsample ablation | `colsample_ablation.json` | Low colsample (0.976) >> High (0.953) at ρ=0.9; no effect at ρ=0.0 |
| 16 | Extensions sanity check | (stdout only) | Assertions pass for Paper 2 claims |
| 17 | High-dimensional scaling | PENDING | Deferred to future work |
| 18 | Success criteria | (meta) | 11/11 criteria pass |

### Theory Bridge Experiments (ostrowski-impossibility validated)

Run via `theory_bridge/*.py`. These validate predictions from the impossibility theorems against DASH experimental data.

| Experiment | Script | Key result | Status |
|-----------|--------|------------|--------|
| Bimodality prediction | `test_bimodality.py` | Dip p<0.002 at ρ≥0.5 (synthetic); p=0.575 on California (NOT confirmed) | VALIDATED (synthetic), NUANCED (real) |
| Coverage conflict | `test_coverage_conflict.py` | Spearman 0.59-0.98 across 4 model classes, 3 datasets | VALIDATED |
| Variance bound | `test_variance_bound.py` | Var[SHAP] = DASH MSE — 0/12 violations | VALIDATED |
| First-mover SHAP correlation | `eta_shap_correlation.py` | ρ_SHAP < 0 for substitutable features (synthetic -0.11 to -0.24) | VALIDATED |
| Model-class structure | `model_class_rigorous.py` | Within-family ρ=0.79-0.94; cross-family not significant | VALIDATED |
| η group-mean test | `eta_group_validation.py` | 12/14 significant after Bonferroni | VALIDATED |
| Info-theoretic predictors | `info_theoretic_validation.py` | 7 data-only predictors tested, max Spearman 0.26 — NO data-only formula | VALIDATED (negative) |
| Spectral predictors | `spectral_validation.py` | Tautological for ridge; weak for non-ridge | NUANCED |
| Cross-domain universality | `cross_domain_validation.py` | CC universal across 7 datasets but ~58% baseline | VALIDATED |
| MI boundary test | `mi_only_dependence_test.py` | MI=1.91 catches X₂=X₁² that ρ=0.08 and VIF=1.008 miss | VALIDATED |

### Retracted Results (do NOT use)

| Result | Why retracted |
|--------|---------------|
| Entropy bimodality | 100% permutation artifact (discretization) |
| Pairwise "audit pairs" | Marginal rates suffice; no pair-specific signal |
| η = 1/g from correlation thresholds | Inverts reality for XGBoost+SHAP |
| Data-only instability prediction | Max Spearman 0.26 — no formula exists |
| Phase transition in stability curve | Gradual, not sharp |

### API Surface

**Core pipeline** (`dash_shap.core`):
- `DASHPipeline` — full 5-stage pipeline, `.fit()`, `.fit_from_population()`, `.fit_from_attributions()`
- `check()` — 3-line stability check API (top-level `dash_shap.check`)
- `compute_consensus(aggregation='mean'|'pca')` — consensus with optional PCA aggregation

**Diagnostics** (`dash_shap.core.diagnostics`):
- `FeatureStabilityIndex` / `ImportanceStabilityPlot` — FSI and IS Plot
- `coverage_conflict()` — nonparametric sign-flip predictor (Spearman 0.59-0.98)
- `compare_flip_predictors()` — CC vs Gaussian formula comparison
- `predict_sign_instability()` / `has_coverage_conflict()` — per-feature sign stability
- `mi_prescreen()` — pairwise MI with permutation threshold, flags hidden pairs
- `shap_residual()` — within-group |SHAP_r| metric
- `local_disagreement_map()` — per-observation SHAP with error bars

**Extensions** (`dash_shap.extensions`):
- `confidence_intervals` — BCa bootstrap CI for importance, FSI, rank
- `partial_order` — π(A>B) pairwise confidence
- `feature_groups` — SHAP-substitutability clustering
- `stable_feature_selection` — importance+stability composite ranking
- `local_uncertainty` — per-observation K×P slice with sign-flip rate
- `robust_certification` — worst-case top-k guarantee
- `theory_bridge` — SNR, predicted flip rates, M recommendation, divergence ratio
- `causal_flags` — robust / collinear / fragile / unimportant labels
- `audit_report` — structured stakeholder report
- `DriftMonitor` — cosine distance between model versions
- `ParetoSelector` — RMSE-stability Pareto frontier
- `federated_consensus` — cross-site consensus without data sharing

**Baselines** (`dash_shap.baselines`): SingleBest, SingleBest(M=200), LargeSingleModel, LSM(Tuned), EnsembleSHAP, StochasticRetrain, RandomSelection, NaiveTopN, RandomForest, PermutationImportance, LightGBM

### Paper Drafts

| File | Version | Status | Scope |
|------|---------|--------|-------|
| `paper/draft_v7_tmlr.tex` | v7 | TMLR submission (anonymous, under review) | Method + empirical validation |
| `paper/draft_v7_preprint.tex` | v7 | ArXiv preprint (de-anonymized) | Same content, author names visible |
| `paper/draft_v8_reviewer_response.tex` | v8 | Reviewer response revision | Adds colsample ablation, MI quantitative bridge, enrichment theory |
| `paper/draft_v1.tex` through `draft_v6_preprint.tex` | v1-v6 | Historical/frozen | Do not modify |

### Three-Level Diagnostic Hierarchy

| Level | Question | Tool | Evidence |
|-------|----------|------|----------|
| **Structure** | What compromises are forced? | Tightness classification (Lean) | 839 theorems across 2 repos, 0 sorry |
| **Existence** | Does this dataset trigger it? | `mi_prescreen()` | MI catches 67-93% hidden dependencies |
| **Magnitude** | How much instability? | `coverage_conflict()` + FSI | Spearman 0.59-0.98 against observed flip rates |
