# Plan: Experimental Pipeline Updates for TMLR Submission

## Context

Analysis of 50-rep SageMaker results revealed open questions the current pipeline cannot answer. Most critically: (1) DASH significantly beats SR on real-world data (Breast Cancer p<0.001, Superconductor p<0.001) but the mechanism isn't isolated — is it colsample_bytree restriction, MaxMin selection, or general hyperparameter diversity? (2) The nonlinear +3% advantage can't be attributed to population design vs diversity selection because RS is missing from the nonlinear sweep. (3) Several experiments have inconsistent method sets or are broken.

This plan adds one new experiment (colsample_bytree ablation) and fixes four issues in existing experiments.

---

## Changes Overview

| # | Change | Priority | Files Modified |
|---|--------|----------|----------------|
| 1 | Colsample_bytree ablation experiment | P0 | `run_experiments_parallel.py`, `dash_shap/core/population.py` |
| 2 | Add RS to nonlinear sweep | P0 | `run_experiments_parallel.py` |
| 3 | Add top-k5 significance tests | P0 | `run_experiments_parallel.py`, `dash_shap/evaluation/__init__.py` |
| 4 | Standardize real-world method sets | P1 | `run_experiments_parallel.py` |
| 5 | Fix k_sweep_independence | P1 | `run_experiments_parallel.py` |

Dropped: TOST for stability (stability is an aggregate metric, not per-rep — TOST requires paired observations; bootstrap CIs already serve this purpose).

---

## Change 1: Colsample_bytree Ablation Experiment [P0]

### Purpose

Directly test whether forced low colsample_bytree is the mechanism behind DASH's advantages. This is the experiment a rigorous reviewer would request.

### Critical Design Issue: RNG Confound

`sample_configurations()` (population.py:29) iterates over search_space keys and calls `rng.choice(values)` for each parameter. Numpy's `RandomState.randint` uses rejection sampling — different list lengths cause different numbers of internal RNG draws, diverging the state for all subsequent parameters.

**Verified empirically:** With `colsample_bytree` lists of length 7 vs 6, downstream parameters (`subsample`, `reg_alpha`, etc.) diverge across conditions. This confounds the ablation — differences in stability could come from different subsample/regularization values, not colsample_bytree.

**Fix:** Generate base configurations once, then replace colsample_bytree using a separate RNG instance. This guarantees all non-colsample hyperparameters are identical across conditions.

### Implementation

#### a) Add `configs` parameter to `generate_model_population` (population.py:125)

```python
def generate_model_population(
    X_train, y_train, X_val, y_val,
    M=200, task="regression", search_space=None, configs=None,  # ← ADD
    sampling_strategy="random", n_estimators=1000,
    early_stopping_rounds=20, n_jobs=-1, seed=42, verbose=True, nthread=None,
):
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    if configs is None:  # ← ADD
        configs = sample_configurations(search_space, M, seed=seed, strategy=sampling_strategy)
    # ... rest unchanged
```

Backward-compatible: existing callers pass neither `configs` nor `search_space`, getting the default behavior.

#### a2) Add `initial_configs` parameter to `DASHPipeline` (pipeline.py)

Pass pre-generated configs through to `generate_model_population`:

```python
# __init__ (line 47): add parameter
def __init__(self, ..., initial_configs=None):
    ...
    self._initial_configs = initial_configs  # line ~155

# fit() (line 291): pass through
self.models_, self.val_scores_, self.configs_ = generate_model_population(
    ..., configs=self._initial_configs, ...
)
```

4 lines total (2 in __init__, 1 storage, 1 in fit). After `fit()`, `dash.models_` and `dash.val_scores_` are available for RS to reuse via `fit_from_population`. No need for a separate `DASHPipeline.fit_from_population` method.

#### b) Add `_run_single_rep_colsample` worker function (~line 4550)

```python
CS_RANGES = {
    "Low (0.1-0.5)": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "High (0.5-1.0)": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Full (0.1-1.0)": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
}

def _run_single_rep_colsample(condition, rep, feature_names, *, nthread=1):
    rep_seed = SEED + rep

    # Generate data based on condition
    if condition == "linear_0.0":
        data = generate_synthetic_linear(N=5000, rho=0.0, seed=rep_seed)
    elif condition == "linear_0.9":
        data = generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)
    else:  # nonlinear_0.9
        data = generate_synthetic_nonlinear(N=5000, rho=0.9, seed=rep_seed)
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = data

    # Generate base configs ONCE (all non-colsample params identical across ranges)
    base_configs = sample_configurations(DEFAULT_SEARCH_SPACE, M, seed=rep_seed)

    per_method = {}

    for label, cs_values in CS_RANGES.items():
        # Replace colsample_bytree using SEPARATE RNG (confound fix)
        rng_cs = np.random.RandomState(rep_seed + 7777)
        configs = []
        for cfg in base_configs:
            c = dict(cfg)
            c["colsample_bytree"] = float(rng_cs.choice(cs_values))
            configs.append(c)

        # DASH trains population with these exact configs, then runs stages 2-5
        dash = DASHPipeline(
            M=M, K=K, epsilon=EPSILON, delta=DELTA,
            selection_method="maxmin", n_jobs=1, nthread=nthread,
            seed=rep_seed, verbose=False, initial_configs=configs,
        )
        dash.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        per_method[f"DASH {label}"] = _extract(dash, ...)

        # RS reuses DASH's population (exact same models)
        rs = RandomSelectionBaseline(M=M, K=K, epsilon=EPSILON, delta=DELTA, ...)
        rs.fit_from_population(dash.models_, dash.val_scores_, Xexp,
                               feature_names=feature_names)
        per_method[f"RS {label}"] = _extract(rs, ...)

        del dash  # free models before next range

    # Controls (default search space, unchanged)
    sr = StochasticRetrainBaseline(N=K, task="regression", n_jobs=1, ...)
    sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    per_method["Stochastic Retrain"] = _extract(sr, ...)

    sb = SingleBestBaseline(n_trials=N_TRIALS_SB, ...)
    sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    per_method["Single Best"] = _extract(sb, ...)

    return condition, rep, per_method
```

#### c) Add `experiment_colsample_ablation()` function

- Flattens (condition × rep) pairs into single Parallel call
- 3 conditions × 50 reps = 150 jobs
- Aggregates: stability, SE, top-k5, equity, accuracy (linear only), RMSE, k_eff
- Runs bootstrap stability tests between all DASH variants and controls
- Runs Wilcoxon equity tests between variants
- Saves to `results/tables/colsample_ablation.json`

#### d) Register in `EXPERIMENTS` dict and `DEFAULT_ORDER`

### What the results will tell us

| Observation | Meaning | Paper implication |
|---|---|---|
| DASH-Low >> DASH-High (stability) | colsample_bytree restriction is the mechanism | Core claim: "forced feature restriction breaks sequential dependency" |
| DASH-Low ≈ DASH-High | Other hyperparameter diversity suffices | Weaker claim: general diversity, not specifically colsample |
| DASH-Low ≈ RS-Low | MaxMin selection doesn't help stability within a range | Confirms equity-only role for MaxMin |
| RS-Low >> SR | Population diversity beats seed diversity even without MaxMin | Population training approach is the contribution |
| DASH-High ≈ SR | Without feature restriction, DASH offers no stability benefit | Strongest evidence for colsample mechanism |
| DASH-Full ≈ DASH-Low | High colsample models don't dilute the benefit | Practical robustness claim |

### Estimated runtime

3 conditions × 50 reps = 150 jobs. Each job: 3 population trainings (200 models × ~0.5s each = ~300s) + 6 SHAP computations (DASH+RS per range, ~75s each = ~450s) + SR (~240s) + SB (~20s) ≈ 1000s per job. With 32 workers: 150 ÷ 32 ≈ 5 rounds × ~17 min = **~85 minutes** on ml.g5.16xlarge.

This is the most expensive new experiment. Could reduce to 2 conditions (drop linear ρ=0.0 — the safety check is less important than the mechanism test) to save ~30 minutes.

---

## Change 2: Add Random Selection to Nonlinear Sweep [P0]

### Purpose

Isolate whether the nonlinear +3% (DASH 0.887 vs SR 0.857 at ρ=0.9) comes from population design or MaxMin selection. If RS matches DASH, the advantage is from stages 1-2.

### Implementation

**File:** `run_experiments_parallel.py`

a) Line 1720 — add `"Random Selection"` to `nl_methods`:
```python
nl_methods = [
    "Single Best", "Large Single Model", "LSM (Tuned)",
    "Stochastic Retrain", "Random Selection",  # ← ADD
    "Random Forest", "DASH (MaxMin)",
]
```

b) `_run_single_rep_nonlinear` (~line 1618) — add RS within the existing for-loop:

Keep the existing per-method for-loop structure. In the DASH branch, save the population before deleting:

```python
elif name == "DASH (MaxMin)":
    m = DASHPipeline(M=M, K=K, ...)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
    imp = m.global_importance_
    # Save population for RS reuse
    _dash_models = m.models_
    _dash_val_scores = m.val_scores_
    preds = m.get_consensus_ensemble_predictions(Xte)

elif name == "Random Selection":
    m = RandomSelectionBaseline(M=M, K=K, epsilon=EPSILON, delta=DELTA, ...)
    m.fit_from_population(_dash_models, _dash_val_scores, Xexp,
                          feature_names=feature_names)
    imp = m.global_importance_
    preds = m.get_consensus_ensemble_predictions(Xte)
```

**Requirement:** "DASH (MaxMin)" must precede "Random Selection" in `nl_methods` list. Ensured by list order at line 1720.

This means DASH and RS operate on the EXACT same population — correct experimental design, not just an optimization.

**Cost:** Minimal — RS's SHAP computation is the only added cost (population shared with DASH). ~15% more runtime.

---

## Change 3: Add Top-K5 Significance Tests [P0]

### Purpose

Formalize SR's top-k5 advantage (0.922 vs 0.863). Per-rep `imp_runs` already exist in both synthetic and real-world results.

### Implementation

**File:** `dash_shap/evaluation/__init__.py`

a) Add `bootstrap_topk5_test` (mirrors `bootstrap_stability_test` at lines 297-342):

```python
def bootstrap_topk5_test(imp_runs_a, imp_runs_b, k=5, n_bootstrap=10000, seed=42):
    """Bootstrap permutation test for top-k overlap stability difference."""
    rng = np.random.RandomState(seed)
    n = len(imp_runs_a)
    assert len(imp_runs_b) == n

    obs_diff = topk_overlap_stability(imp_runs_a, k) - topk_overlap_stability(imp_runs_b, k)

    boot_diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        a_boot = [imp_runs_a[i] for i in idx]
        b_boot = [imp_runs_b[i] for i in idx]
        boot_diffs[b] = topk_overlap_stability(a_boot, k) - topk_overlap_stability(b_boot, k)

    centered = boot_diffs - obs_diff
    p_value = float(np.mean(np.abs(centered) >= np.abs(obs_diff)))
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    return obs_diff, p_value, ci_lo, ci_hi
```

**File:** `run_experiments_parallel.py`

b) After `_stability_tests` computation (line ~1413), add `_topk5_tests` block with same structure.

c) In `_log_pairwise_significance` (line ~613), add top-k5 tests using `imp_runs` already available for real-world experiments.

**Cost:** ~10 seconds per rho level (post-hoc). `topk_overlap_stability` is O(n²) per bootstrap sample but n=50 and k=5, so trivial.

---

## Change 4: Standardize Real-World Method Sets [P1]

### Purpose

Prevent reviewer objections about inconsistent baselines. Add LSM and SB(M=200) to all datasets.

### Current state

| Method | BC | SC | Cal |
|---|---|---|---|
| Single Best | ✅ | ✅ | ✅ |
| SB (M=200) | ❌ | ❌ | ❌ |
| Large Single Model | ❌ | ✅ | ❌ |
| Naive Top-N | ❌ | ❌ | ❌ |
| SR, RS, RF, DASH | ✅ | ✅ | ✅ |

### Changes

Add SB(M=200), LSM, and Naive Top-N to all three experiments. All three use `fit_from_population` — they share DASH's population with no extra training cost. LSM requires its own training (~1 model with K×T_PER_MODEL trees).

**Files:** `run_experiments_parallel.py` — modify `bc_methods`, `sc_methods`, `cal_methods` lists and corresponding worker functions.

---

## Change 5: Fix k_sweep_independence [P1]

### Issues

1. **K=1 DASH failure — ROOT CAUSE IDENTIFIED:** `pipeline.py:328` has `if n_filtered < 2: raise ValueError(...)`. This guard fires even when K=1 and n_filtered=1 (a valid state). The guard was designed to ensure ≥2 models for pairwise diversity, but K=1 skips diversity selection entirely. Confirmed: `compute_diagnostics` produces NaN variance/FSI with K=1 (ddof=1 on single observation) but `global_importance_` is fine — the k_sweep only needs `global_importance_`.
2. **Only DASH and RS tested:** No Stochastic Retrain.
3. **Mislabeled keys:** JSON stores RS as "SR".

### Changes

a) **Fix the K=1 guard** (pipeline.py:328):
```python
# BEFORE:
if n_filtered < 2:
    raise ValueError(f"Only {n_filtered} models passed filter. Increase epsilon.")

# AFTER:
if n_filtered < max(1, min(2, self.K)):
    raise ValueError(f"Only {n_filtered} models passed filter. Increase epsilon.")
```
This allows K=1 with n_filtered=1 while still requiring n_filtered ≥ 2 for K ≥ 2.

b) Add Stochastic Retrain arm to `_run_single_ksweep_pair`:
```python
sr = StochasticRetrainBaseline(N=k_val, task="regression", n_jobs=1, ...)
sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
```

c) Fix output keys: rename "SR" → "RS", add real "SR" for Stochastic Retrain.

d) Add `traceback.format_exc()` to the except block (line 4382-4390) to log failure causes during development.

e) Extend K values to include K=50 (shows diminishing returns past K=30).

---

## Files Modified Summary

| File | Changes |
|---|---|
| `dash_shap/core/population.py` | Add optional `configs` parameter to `generate_model_population` |
| `dash_shap/core/pipeline.py` | Add `initial_configs` parameter to `DASHPipeline.__init__`, pass through in `fit()`; fix K=1 guard (line 328) |
| `dash_shap/evaluation/__init__.py` | Add `bootstrap_topk5_test` function |
| `run_experiments_parallel.py` | Changes 1-5: new experiment, modified nonlinear sweep, added tests, standardized methods, fixed k_sweep |

---

## Verification

1. **Unit tests:**
   - Test `generate_model_population` with `configs` parameter (skips `sample_configurations`)
   - Test `DASHPipeline` with `initial_configs` (trains exact configs, not random)
   - Test `bootstrap_topk5_test` with known-different vectors
   - Test K=1 pipeline fix (n_filtered=1 should not raise ValueError)

2. **Confound verification:**
   - Write a test that generates configs for Low/High/Full ranges and asserts all non-colsample hyperparameters are identical

3. **Smoke tests:**
   ```bash
   python run_experiments_parallel.py --smoke --experiments colsample_ablation
   python run_experiments_parallel.py --smoke --experiments nonlinear_sweep
   python run_experiments_parallel.py --smoke --experiments k_sweep_independence
   ```

4. **Full run on SageMaker:** ~3.5 hours additional runtime
   ```bash
   python run_experiments_parallel.py --experiments \
     colsample_ablation,nonlinear_sweep,real_breast_cancer,\
     real_california,real_superconductor,k_sweep_independence
   ```

---

## Estimated Additional Runtime (ml.g5.16xlarge)

| Experiment | Delta | Reason |
|---|---|---|
| Colsample ablation | +85 min | New (3 conditions × 3 ranges × 50 reps, SHAP-heavy) |
| Nonlinear sweep | +30 min | RS arm (SHAP only, shares population) |
| Real-world datasets (3) | +30 min | Added LSM, SB(M=200), Naive Top-N |
| k_sweep | +15 min | Added SR, extended K range |
| Linear sweep (top-k5) | +2 min | Post-hoc computation |
| **Total** | **~2.5-3 hours** | On top of existing ~6-12h run |

---

## Risks

1. **Colsample ablation could show DASH-Low ≈ DASH-High** (no colsample effect). This weakens the mechanism claim but is still publishable — it means general hyperparameter diversity suffices, which is arguably a simpler and more practical message.
2. **RS in nonlinear sweep could match DASH** (no diversity selection advantage). Expected based on real-world data. The paper narrative shifts from "diversity selection helps" to "population design helps."
3. **K=1 fix might expose other edge cases** in the pipeline. Guard against with: check that consensus SHAP, diagnostics, and global_importance all handle K=1 gracefully (NaN FSI is acceptable).
4. **Standardizing real-world methods adds LSM to Breast Cancer (classification)**. Need to verify LSM supports `task="binary"`. If not, omit LSM from Breast Cancer only.
