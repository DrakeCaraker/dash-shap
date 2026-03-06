# Audit: `notebooks/demo_benchmark_4.ipynb`

**Date**: 2026-03-06
**Scope**: Structure, rigor, reproducibility, and statistical methodology
**Prior audit**: `AUDIT_demo_benchmark_3.md` (20 findings; this document tracks remediation status)

---

## Overall Assessment

Version 4 is a substantial improvement over v3. Twelve of the fifteen prior audit findings have been fully addressed, including the three most critical issues (test-set evaluation, config fragmentation, Wilcoxon power). The notebook now has a clean narrative arc across 56 cells and 15 sections, centralized hyperparameter management via `PAPER_CONFIG`, proper held-out test set usage, and well-documented statistical methodology.

However, this audit identifies one **blocking bug** (import error that prevents execution), several **medium-severity** methodological concerns, and a set of housekeeping items. The ablation studies have been extended to cover ρ ∈ {0.0, 0.9, 0.95} (was ρ=0.9 only) to verify parameter sensitivity generalizes across correlation regimes.

---

## Remediation Status: v3 Audit Findings

| v3 ID | Severity | Issue | Status | Notes |
|-------|----------|-------|--------|-------|
| A1 | Medium | No environment recording | **FIXED** | Cell 1 records Python version, platform, and all package versions (numpy, xgboost, shap, sklearn, scipy, etc.) |
| A2 | Low | Global `warnings.filterwarnings('ignore')` | **FIXED** | Scoped to `FutureWarning` globally + `UserWarning` from shap module only |
| B1 | High | Config fragmentation (M, K, N_REPS inconsistent) | **FIXED** | Centralized `PAPER_CONFIG` dict (Cell 1). Section-level overrides are documented inline with comments (e.g., `NL_M = 200  # Override: lighter compute`) |
| B2 | Medium | `n_trials=50` vs `n_trials=30` inconsistency | **FIXED** | `N_TRIALS_SB = PAPER_CONFIG['N_TRIALS_SB'] = 30` used everywhere. Comment `# B2` at usage sites |
| B3 | Medium | `StochasticRetrainBaseline(N=15)` vs K=30 | **FIXED** | Now uses `N=K` for fair comparison. Comment `# B3` at usage site (Cell 39) |
| B4 | Low | Breast Cancer epsilon undocumented | **FIXED** | Uses global `EPSILON=0.08` consistently |
| C1 | High | Wilcoxon with N=10 has very low power | **FIXED** | Main sweep uses `N_REPS=20`. Section 9 markdown documents power limitation: "minimum corrected p ≈ 0.0004, comfortably below α=0.05" |
| C2 | Medium | Bootstrap CI biased by self-correlation | **FIXED** | Cell 53 implements corrected bootstrap (delete-d jackknife, skips self-pairs) |
| C3 | Medium | Cohen's d direction ambiguity for equity | **FIXED** | Cell 37 reports signed d + "favors" column with direction-aware logic for accuracy (higher=better) vs equity (lower=better) |
| D1 | High | All evaluation on val set (test set unused) | **FIXED** | `X_ref=Xte` (test set) used everywhere. Comments `# D1` at key sites |
| D2 | Medium | Superconductor scaler leaks across re-splits | **FIXED** | `StandardScaler().fit(Xtr_r)` re-fit per rep in Superconductor (Cell 43) and California Housing (Cell 51). Comments `# D2` |
| D3 | Low | Nonlinear DGP `beta_4_to_G` hardcoded seed | **NOT FIXED** | `RandomState(42)` in `synthetic.py:118` still used regardless of main seed. Ground truth importance is identical across reps. Acceptable since accuracy is not reported for nonlinear DGP, but a comment in the code would prevent confusion |
| E1/E2 | Medium | Redundant sweeps; pub figure uses 10-rep data | **FIXED** | Single canonical sweep in Section 4 with `N_REPS=20` + RMSE extraction. Publication figure (Cell 49) draws from this sweep. Stale pass-through cells (26, 31, 32) remain as minor clutter |
| E3 | Medium | Breast Cancer has no repetition analysis | **FIXED** | Cell 24 adds 10-rep stability analysis for Breast Cancer |
| E4 | Medium | DASH (Cluster) dropped without explanation | **FIXED** | Section 4 markdown documents exclusion: cluster is included in proof-of-concept (Section 2) and stability (Section 3) but excluded from the sweep for parsimony |

**Summary**: 12/15 fully fixed, 1 not fixed (low severity), 2 fixed with minor residual clutter.

---

## New Issues

### A. Blocking

| # | Severity | Issue |
|---|----------|-------|
| N1 | **Critical** | **Import error prevents notebook execution.** Cell 1 imports `compute_diagnostics` from `dash.core.consensus`, but the function is defined in `dash.core.diagnostics` (line 14). The notebook cannot run as-is. **Status: FIXED in this audit** — import corrected to `from dash.core.diagnostics import ... compute_diagnostics`. |

### B. Statistical Methodology

| # | Severity | Issue |
|---|----------|-------|
| N2 | High | **Ablation uses ABL_N_REPS=5 — too few for reliable stability estimates.** With 5 importance vectors, stability is computed from C(5,2)=10 pairwise Spearman correlations. This is a very noisy estimator. Conclusions about stability *trends* across parameter values (Section 12) are fragile. Recommend: (a) increase to ABL_N_REPS=10 (C(10,2)=45 pairs, 4.5× more data), or (b) add confidence intervals to acknowledge the uncertainty, or (c) clearly caveat that ablation stability values are trend indicators only. |
| N3 | Medium | **Stability metric not directly comparable across different N_REPS.** The notebook computes stability at N_REPS=20 (main sweep), 10 (epsilon sensitivity, nonlinear, Superconductor, Cal Housing, Breast Cancer), and 5 (ablation). While the mean pairwise Spearman ρ estimator is unbiased regardless of n, its *variance* decreases with n. A stability of 0.92 from 5 reps could easily be 0.88 or 0.96 from 20 reps. Cross-section comparisons (e.g., "ablation stability at M=50 vs sweep stability at ρ=0.9") are not apples-to-apples. Recommend: note this limitation when comparing across sections. |
| N4 | Medium | **Wilcoxon tests only compare 2 of 7 baselines.** Section 9 tests DASH vs Single Best and DASH vs Large Single Model. The Table 2 data (Cell 39) provides per-rep arrays for Ensemble SHAP, Stochastic Retrain, and DASH Dedup at ρ=0.9, but these are not significance-tested. A reviewer may ask why only 2 comparisons were chosen. Recommend: either test all available pairs or explicitly state that only the two primary baselines are tested and why. |

### C. Experimental Design

| # | Severity | Issue |
|---|----------|-------|
| N5 | Medium | **SHAP background data is deterministic first-100 rows of X_ref.** In `compute_consensus()` (`consensus.py:25`), `bg_data = X_ref[:min(background_size, N_prime)]`. The first 100 rows of the test set have a specific ordering from `train_test_split`, and may not be representative. This introduces a subtle systematic bias in SHAP values. Recommend: randomly sample `background_size` rows from X_ref using the pipeline's seed. |
| N6 | Medium | **Nonlinear DGP uses NL_M=200 vs M=500 for linear.** A 2.5× smaller population pool changes the diversity available for selection. The notebook documents this as "lighter compute" but doesn't discuss how the reduced pool might understate DASH's advantage (fewer diverse models to select from) or overstate it (easier for MaxMin to find distinct models in a smaller pool). The comparison between linear and nonlinear DGP sections is weakened. |
| N7 | Medium | **Epsilon has RMSE units — not scale-invariant across datasets.** `EPSILON=0.08` means "within 0.08 RMSE of best" for regression. For synthetic data (RMSE ~2–5), this passes ~114/500 models. For Superconductor (RMSE ~17–20), the same absolute epsilon is extremely tight relative to the score range, potentially passing very few models. The pipeline should arguably use a relative epsilon (e.g., 2% of best score) for real-world datasets, or the per-dataset epsilon should be tuned and documented. |
| N8 | Medium | **Section 3 stability analysis excludes 3 of 8 methods from the 20-rep measurement.** The 20-rep stability loop (Cell 12) runs only 5 methods: Single Best, Large Single Model, Naive Top-N, DASH MaxMin, DASH Cluster. Ensemble SHAP, Stochastic Retrain, and DASH Dedup are measured separately in Section 10 (Cell 39) with `TABLE2_N_REPS = N_REPS` (20 reps, so the count matches). However, these 3 methods are only measured at ρ=0.9, not across the full correlation sweep. This is an asymmetry in experimental coverage. |
| N9 | Low | **`within_group_equity` returns 0 for zero-mean groups.** In `evaluation/__init__.py`, groups with near-zero mean importance get CV=0.0 (the best possible score). If group 10 has β=0 (zero ground-truth importance), it inflates equity scores equally for all methods. The effect is symmetric and doesn't bias comparisons, but the absolute equity values are misleadingly low. |

### D. Presentation and Housekeeping

| # | Severity | Issue |
|---|----------|-------|
| N10 | Low | **Stale pass-through cells (26, 31, 32).** These contain only `print('See Section X')` or `print('deferred')`. They are artifacts of the v3→v4 refactor and add clutter for reviewers. Remove or consolidate. |
| N11 | Low | **Publication figure (Cell 49) mutates global `rcParams`.** After Cell 49, all subsequent cells produce plots with serif fonts, 300 DPI, and changed font sizes. Cells 50+ (bootstrap CIs, California Housing) are unintentionally affected. Recommend: save and restore `rcParams` around the publication figure cell, or use `with plt.style.context(...)`. |
| N12 | Low | **`tau=0.3` in cluster coverage selection is undocumented.** Cell 9 uses `cluster_coverage_selection(imp_vecs, filt_scores, X_train, tau=0.3, K=K)`. The `tau` parameter is not in `PAPER_CONFIG` and lacks justification in the markdown. |
| N13 | Low | **No wall-clock timing summary.** v3 audit recommended timing at notebook end. v4 records package versions (A1 fix) but still has no total execution time summary. For a compute-intensive notebook, knowing that it takes 2 hours vs 20 hours is useful for reproducibility. |
| N14 | Low | **D3 from v3 still open.** Nonlinear DGP `beta_4_to_G` uses `RandomState(42)` regardless of the main `seed` parameter (`synthetic.py:118`). Ground truth importance is identical across all reps. Since accuracy is intentionally not reported for the nonlinear DGP, this is functionally harmless — but a code comment explaining the choice would prevent confusion for future maintainers. |

---

## Changes Made in This Audit

### 1. Import Bug Fix (N1)

**Cell 1**: Changed
```python
from dash.core.consensus import compute_consensus, compute_diagnostics
```
to
```python
from dash.core.consensus import compute_consensus
```
and merged `compute_diagnostics` into the existing `dash.core.diagnostics` import line:
```python
from dash.core.diagnostics import FeatureStabilityIndex, ImportanceStabilityPlot, local_disagreement_map, compute_diagnostics
```

### 2. Multi-ρ Ablation Studies (addresses N2 partially, and prior audit gap)

**Cells 45–47**: Extended ablation studies from ρ=0.9 only to ρ ∈ {0.0, 0.9, 0.95}. This verifies that:
- Parameter sensitivity trends hold at the **safety control** (ρ=0 — no collinearity, DASH should not degrade)
- The **primary evaluation point** (ρ=0.9) results are not idiosyncratic
- The **most extreme** collinearity level (ρ=0.95) doesn't reveal unexpected behavior (e.g., very small M failing catastrophically)

The multi-ρ ablation produces:
- A 4-panel comparison figure overlaying all three ρ levels per parameter
- A summary table with stability and accuracy at each (ρ, parameter, value) combination

---

## Recommended Actions (Priority Order)

1. **Verify N1 fix**: Run Cell 1 to confirm the import correction resolves the `ImportError`.
2. **Address N2**: Increase `ABL_N_REPS` to 10 or add uncertainty estimates to ablation stability values.
3. **Address N5**: Replace deterministic `X_ref[:100]` background with `rng.choice(N_prime, background_size, replace=False)` in `consensus.py`.
4. **Address N7**: Document per-dataset epsilon rationale, or implement relative epsilon option.
5. **Address N4**: Either extend significance tests to all Table 2 baselines or document the restriction.
6. **Clean up N10**: Remove stale pass-through cells (26, 31, 32).
7. **Address N11**: Wrap publication figure in `plt.style.context()` or save/restore `rcParams`.
8. **Address N6**: Add a markdown note discussing the NL_M=200 vs M=500 tradeoff.
9. **Address N12**: Add `tau` to `PAPER_CONFIG` or add justification comment.
10. **Address N13**: Add `time.time()` bookends and print total wall-clock time in the final cell.
11. **Address N14/D3**: Add comment in `synthetic.py:118` explaining the hardcoded seed choice.
