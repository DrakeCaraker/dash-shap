# Audit: `notebooks/demo_benchmark_4.ipynb`

**Date**: 2026-03-06 (re-audit)
**Scope**: Structure, rigor, reproducibility, and statistical methodology
**Prior audits**: `AUDIT_demo_benchmark_3.md` (20 findings, all remediated); first v4 audit (14 findings N1–N14, all fixed)

---

## Overall Assessment

**Re-audit status**: The notebook is in good shape overall. All v3 and first-round v4 findings
remain properly addressed. This final re-audit identified **3 issues that must be fixed before
running** (F1–F3) and **10 advisory items** (A1–A10) that are non-blocking.

Strengths carried forward from prior audits:
- Centralized `PAPER_CONFIG` with all hyperparameters (including `TAU_CLUSTER`)
- Consistent `N_REPS=20` across the main sweep and all real-world/nonlinear sections
- `ABL_N_REPS=10` for ablation studies (C(10,2)=45 pairwise comparisons)
- Randomized SHAP background sampling (seeded for reproducibility)
- Scale-appropriate epsilon for the Superconductor dataset (`SC_EPSILON=0.40`)
- Proper handling of zero-mean groups in within-group equity
- Wall-clock timing and environment recording

---

## Remediation Status: v3 Audit Findings

| v3 ID | Severity | Issue | Status | Notes |
|-------|----------|-------|--------|-------|
| A1 | Medium | No environment recording | **FIXED** | Cell 1 records Python version, platform, and all package versions |
| A2 | Low | Global `warnings.filterwarnings('ignore')` | **FIXED** | Scoped to `FutureWarning` + `UserWarning` from shap |
| B1 | High | Config fragmentation | **FIXED** | Centralized `PAPER_CONFIG` dict with documented overrides |
| B2 | Medium | `n_trials` inconsistency | **FIXED** | `N_TRIALS_SB=30` everywhere |
| B3 | Medium | `StochasticRetrainBaseline(N=15)` vs K=30 | **FIXED** | `N=K` for fair comparison |
| B4 | Low | Breast Cancer epsilon undocumented | **FIXED** | Uses `EPSILON=0.08` consistently |
| C1 | High | Wilcoxon with N=10 has low power | **FIXED** | `N_REPS=20` for main sweep; power documented |
| C2 | Medium | Bootstrap CI biased by self-correlation | **FIXED** | Delete-d jackknife skips self-pairs (Cell 53) |
| C3 | Medium | Cohen's d direction ambiguity | **FIXED** | Signed d + "favors" column |
| D1 | High | Eval on val set instead of test set | **FIXED** | `X_ref=Xte` everywhere |
| D2 | Medium | Superconductor scaler leaks | **FIXED** | Re-fit `StandardScaler` per rep |
| D3 | Low | Nonlinear DGP hardcoded seed | **FIXED** | Explanatory comment added in `synthetic.py:118` |
| E1/E2 | Medium | Redundant sweeps | **FIXED** | Single canonical sweep; stale pass-throughs cleaned |
| E3 | Medium | Breast Cancer missing reps | **FIXED** | 20-rep analysis (upgraded from 10) |
| E4 | Medium | DASH (Cluster) dropped without explanation | **FIXED** | Documented in Section 4 markdown |

**Summary**: 15/15 addressed.

---

## New Issues (N1–N14) — All Implemented

### A. Blocking

| # | Severity | Fix Applied |
|---|----------|-------------|
| N1 | Critical | **Import error fixed.** `compute_diagnostics` moved from `dash.core.consensus` import to `dash.core.diagnostics` import in Cell 1. |

### B. Statistical Methodology

| # | Severity | Fix Applied |
|---|----------|-------------|
| N2 | High | **Ablation reps increased.** `ABL_N_REPS` raised from 5 to 10 (C(10,2)=45 pairwise comparisons). Multi-ρ ablation at {0.0, 0.9, 0.95} also added. |
| N3 | Medium | **Rep counts standardized.** All section-level overrides (`EPS_N_REPS`, `NL_N_REPS`, `SC_N_REPS`, `CAL_N_REPS`, `BC_N_REPS`) now use `N_REPS` (=20). Ablation uses 10. |
| N4 | Medium | **Wilcoxon tests extended.** Cell 37 now tests DASH vs all Table 2 baselines (Ensemble SHAP, Stochastic Retrain, DASH Dedup) at ρ=0.9, in addition to the sweep-based tests. Cell 39 now saves per-rep arrays (`acc_runs`, `eq_runs`). Bonferroni correction updated to 26 tests. |

### C. Experimental Design

| # | Severity | Fix Applied |
|---|----------|-------------|
| N5 | Medium | **SHAP background randomized.** `compute_consensus()` now accepts a `seed` parameter and randomly samples background rows via `rng.choice()`. Threaded through `DASHPipeline` (`self.seed`) and all 4 direct notebook calls. Backward-compatible (defaults to deterministic if `seed=None`). |
| N6 | Medium | **NL_M=500 (matches linear).** Nonlinear DGP now uses the same `M=500` population size as the linear sweep, eliminating the confound of different pool sizes. |
| N7 | Medium | **Scale-appropriate Superconductor epsilon.** New `SC_EPSILON=0.40` used for Superconductor (proportional to its ~18 RMSE, matching the ~2% relative tolerance of `EPSILON=0.08` on synthetic data with ~3 RMSE). Documented in Section 11 markdown. |
| N8 | Medium | **Addressed by N3 fix.** All real-world sections now use `N_REPS=20`, so Table 2 baselines have the same rep count as sweep methods. |
| N9 | Low | **Zero-mean groups excluded.** `within_group_equity()` now skips groups with `|mean| < 1e-10` instead of scoring them as CV=0. Also uses `np.abs(gi.mean())` to handle negative SHAP values. |

### D. Presentation and Housekeeping

| # | Severity | Fix Applied |
|---|----------|-------------|
| N10 | Low | **Stale cell cleaned.** Cell 32 reduced to `pass`. |
| N11 | Low | **rcParams properly managed.** Cell 49 now saves `_saved_rc = dict(plt.rcParams)` before mutation and restores fully afterward. |
| N12 | Low | **`tau` documented.** `TAU_CLUSTER=0.3` added to `PAPER_CONFIG`. Cell 9 references `PAPER_CONFIG['TAU_CLUSTER']`. |
| N13 | Low | **Wall-clock timing added.** `_notebook_start = time.time()` in Cell 1; elapsed time printed in Cell 55. |
| N14 | Low | **Hardcoded seed documented.** Comment in `synthetic.py:118` explains why `RandomState(42)` is intentional for the nonlinear DGP ground-truth coefficients. |

---

## Re-Audit Findings: Must Fix (F1–F3)

| # | Cell | Severity | Description |
|---|------|----------|-------------|
| F1 | 37 | **Critical** | **Cell ordering bug.** Cell 37 "Part 2" references `table2_methods` and `table2_results`, which are not defined until Cell 39. Running top-to-bottom raises `NameError`. **Fix:** Move the "Part 2" Table 2 tests out of Cell 37 into a new cell after Cell 39, or swap Cells 37 and 39. |
| F2 | 51 | **High** | **Missing scale-appropriate epsilon for California Housing.** Cell 51 uses `epsilon=EPSILON` (=0.08), but the California Housing target has a different scale (~2.07 median house value in $100k units, RMSE ~0.5–0.7). The same logic that motivated `SC_EPSILON=0.40` for Superconductor applies here. **Fix:** Add `CAL_EPSILON` proportional to typical RMSE (e.g., 0.05–0.10 may be appropriate, but verify against actual test RMSE). |
| F3 | 37 | **High** | **Bonferroni documentation mismatch.** Cell 36 markdown says the correction factor is 20; the code computes `N_TESTS = 26`. A reviewer will flag this inconsistency. **Fix:** Update Cell 36 markdown to state 26 tests. |

---

## Re-Audit Findings: Advisory (A1–A10)

| # | Cell | Severity | Description |
|---|------|----------|-------------|
| A1 | 37 | Medium | `_run_test` "Favors" column uses `comp_name.split(' vs ')[1]` which yields truncated names for Table 2 entries (e.g., `Stochast` because `short` is capped at 8 chars in Part 2). Cosmetic but confusing in output. |
| A2 | 49 | Medium | `_saved_rc = dict(plt.rcParams)` is a shallow copy. Nested mutable values (e.g., `axes.prop_cycle`) are shared references. Use `plt.rc_context()` context manager for bulletproof restoration. |
| A3 | 53 | Medium | Bootstrap CI uses the percentile method, which has known bias at small sample sizes. BCa (bias-corrected accelerated) would be more appropriate for a publication. |
| A4 | 55 | Medium | Criterion 2 requires `acc >= 0.90` as an absolute Spearman threshold at ρ=0.9. Under high collinearity all methods degrade; a relative-to-baseline threshold would be more robust. |
| A5 | 55 | Medium | Criterion 4 "safety control" allows an RMSE gap of 0.1 at ρ=0.0. This is generous for a "no degradation" check where DASH should nearly match baselines. |
| A6 | 24 | Medium | Breast Cancer reps only collect stability (not accuracy/equity), unlike all other repetition analyses. Inconsistent metrics coverage across sections. |
| A7 | 24 | Medium | Breast Cancer reps re-split already-scaled data. The scaler was fitted on the original Cell 19 train split, so `Xv_r` contains observations that contributed to scaling parameters. Minor data leakage — unlikely to affect stability conclusions. |
| A8 | 34 | Medium | Large Single Model excluded from the nonlinear DGP sweep without documented rationale. The linear sweep includes it as a key comparison. |
| A9 | 16,28 | Low | Variable shadowing: `Xtr`, `Xte`, etc. are overwritten by every sweep/section. Not a bug when run top-to-bottom, but makes partial re-execution fragile. Standard Jupyter caveat. |
| A10 | 44,49 | Low | Figures created via `plt.subplots()` are never closed with `plt.close(fig)`. Minor memory accumulation in a long-running notebook. |

---

## Files Modified (Prior Audit)

| File | Changes |
|------|---------|
| `dash/core/consensus.py` | Added `seed` parameter; randomized background sampling with `rng.choice()` |
| `dash/core/pipeline.py` | Thread `self.seed` to `compute_consensus()` |
| `dash/experiments/synthetic.py` | Added explanatory comment for hardcoded seed at line 118 |
| `dash/evaluation/__init__.py` | Fixed `within_group_equity()` to exclude zero-mean groups and use `np.abs()` |
| `notebooks/demo_benchmark_4.ipynb` | Cells 1, 9, 12, 24, 28, 32, 33, 34, 37, 39, 41, 43, 45, 46, 49, 51, 55 |

---

## Source File Notes (from re-audit)

| File | Finding | Severity |
|------|---------|----------|
| `dash/core/consensus.py` | Uses legacy `np.random.RandomState` (not `default_rng`). Internally consistent but soft-deprecated. | Low |
| `dash/core/diagnostics.py` | `np.var(..., ddof=1)` produces `inf` if K=1 models. Pipeline guards `len(filtered) >= 2`, but direct calls are unguarded. | Medium |
| `dash/experiments/synthetic.py` | Hardcoded `seed=42` for nonlinear ground-truth betas ignores caller-provided seed. Documented but fragile for future use. | Low (known) |
| `dash/evaluation/__init__.py` | Single-element groups report CV=0 (ddof=0). Mathematically correct but may surprise. | Low |

---

## Verification Checklist

1. All code cells parse without `SyntaxError` (verified via `ast.parse`)
2. Notebook JSON is valid (verified via `json.load`)
3. `compute_consensus` signature updated with backward-compatible `seed=None` default
4. `within_group_equity` correctly excludes zero-mean groups
5. All `compute_consensus` call sites pass `seed` parameter
6. Bonferroni correction in Cell 37 accounts for extended test count (26 tests)
7. `SC_EPSILON=0.40` used in Superconductor DASHPipeline call
8. `_notebook_start` and timing summary bookend the notebook
9. **[FAIL]** Cell 37 references `table2_methods`/`table2_results` before definition in Cell 39
10. **[FAIL]** Cell 51 California Housing uses generic `EPSILON` instead of scale-appropriate value
11. **[FAIL]** Cell 36 markdown Bonferroni count (20) does not match code (26)
