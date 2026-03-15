# Peer Review: DASH-SHAP for TMLR Publication

**Reviewer**: Automated analysis (code + paper + experiments)
**Date**: 2026-03-15
**Verdict**: Conditionally Publishable (Major Revision)

---

## I. Overall Assessment

The core idea is sound: averaging SHAP values across a diverse ensemble of independently trained XGBoost models stabilizes feature importance under multicollinearity. The implementation is methodologically correct, the documentation is exceptional, and the research integrity is high. However, the paper has several issues that would be flagged by TMLR reviewers, primarily: (1) an incomplete central results table, (2) the strongest baseline (Stochastic Retrain) is not significantly different from DASH, and (3) missing wall-clock timings.

---

## II. Methodological Correctness

### Core Pipeline: All 5 Stages Verified Correct

| Stage | Module | Verdict | Notes |
|-------|--------|---------|-------|
| 1. Population | `population.py` | CORRECT | Per-model seeds (`seed + i`), `colsample_bytree` forced [0.1-0.5], early stopping |
| 2. Filtering | `filtering.py` | CORRECT | Three modes (absolute/relative/quantile), safety floor of 2 models |
| 3. Diversity | `diversity.py` | CORRECT | MaxMin greedy on L2-normalized gain vectors, cosine distance metric |
| 4. Consensus | `consensus.py` | CORRECT | Interventional TreeSHAP (`feature_perturbation="interventional"`), element-wise mean |
| 5. Diagnostics | `diagnostics.py` | CORRECT | FSI with `ddof=1` unbiased variance, IS Plot with median-based quadrants |

### Minor Code Issues

1. **`diversity.py:38`** — Preliminary SHAP subsample takes `X_ref[:n_subsample]` (first N rows) instead of random sample. Could bias importance estimation if data has non-random ordering.

2. **`pipeline.py:70-72`** — `X_ref` defaults to `X_val` silently. Should emit a warning. The four-way split (A4 fix) mitigates this in experiments but is fragile for external users.

3. **`consensus.py:49`** — Background data for TreeSHAP sampled from the same `X_ref` being explained. Theoretically acceptable but worth noting.

### No Critical Bugs Found

MaxMin greedy selection, filtering logic, interventional SHAP usage, statistical computations, and seed propagation are all correct.

---

## III. Experimental Rigor

### Strengths

- Comprehensive correlation sweep: rho in {0.0, 0.5, 0.7, 0.9, 0.95}, 20 reps each
- Linear + nonlinear DGPs with honest failure reporting
- 3 real-world datasets (Breast Cancer, Superconductor, California Housing)
- Four-way data split prevents leakage (A4 fix)
- Epsilon sensitivity <0.001 across 3x range
- Safety check at rho=0 included
- 6-7 well-chosen baselines covering standard practice, compute-matched, and ablation variants

### Critical Issues

#### 1. Incomplete Table 1 (BLOCKING)
The central results table has `---*` placeholders for LSM (Tuned), Stochastic Retrain, and LSM at most rho levels. Cannot submit with missing entries for primary baselines.

#### 2. Stochastic Retrain Parity
At rho=0.9: DASH=0.9805 vs Stochastic Retrain=0.9795 (gap=0.001, Cohen's d=0.26, not significant). The equity gap (0.1625 vs 0.1719) is not tested for significance. This is the paper's biggest vulnerability.

#### 3. Nonlinear Safety Violation
At rho=0.0 and rho=0.5 under nonlinear DGP, DASH shows worse stability than Single Best (0.9420 vs 0.9437). This violates the paper's own "safety desideratum" from Section 3.4.

#### 4. Missing Wall-Clock Timings
The paper acknowledges this gap (Section 6.3). Training 200 models has non-trivial compute cost.

#### 5. Implemented but Missing from Paper
Naive Averaging and Random Selection baselines are coded but absent from tables. These directly answer "does MaxMin matter?" — critical ablations for the contribution claim.

---

## IV. Paper Quality (draft_v1.tex)

### Strengths
- Clear, precise writing
- Unusually honest limitations section (10 explicit limitations)
- Sound related work coverage
- All equations verified against code
- No overclaiming detected

### Issues
1. Table 1 placeholders (`---*`)
2. Figure 1 placeholder (`[Pipeline diagram to be added]`)
3. Author/affiliation placeholders
4. Paillard citation incomplete
5. Bonferroni vs Holm-Bonferroni inconsistency (code uses Holm, paper says Bonferroni)
6. Dedup > MaxMin at rho=0.9 unexplained
7. California Housing results missing from paper (implemented in code)
8. 26 Bonferroni tests not enumerated

---

## V. Code Architecture

### Strengths
- Clean 5-module pipeline separation with orchestrator class
- Lazy imports via `__getattr__`
- Checkpoint pattern for reproducible notebooks
- joblib parallelism
- Pre-push hook blocks .pkl and >1MB files
- Exceptional documentation (EXPERIMENT_GUIDE: 364 lines with fix tags)

### Test Coverage (Weak)
~47 tests, all smoke-level. Missing:
- Integration test for DASHPipeline.fit()
- Unit tests for MaxMin diversity selection (core contribution)
- Reproducibility test (same seed = same output)
- Validation test (DASH > SB on synthetic data)

---

## VI. Research Integrity: No Red Flags

- Multiple metrics reported
- Safety check at rho=0 included
- Unfavorable results reported transparently
- All methodology fixes documented with impact tags
- No signs of p-hacking, selective reporting, or unfair baseline tuning

---

## VII. Prioritized Action Items

### Must-Fix (Blocking)
1. Populate Table 1 with all baselines at all rho levels
2. Add wall-clock timings
3. Create pipeline diagram (Figure 1)
4. Fill author/affiliation info
5. Fix Bonferroni/Holm-Bonferroni inconsistency

### Should-Fix (Reviewer Will Flag)
6. Add Naive Averaging + Random Selection to paper tables
7. Test equity significance (DASH vs Stochastic Retrain)
8. Add California Housing results
9. Add bootstrap CIs to Table 1
10. Explain or investigate Dedup > MaxMin
11. Tighten nonlinear scope claim to rho >= 0.7

### Should-Fix (Code Quality)
12. Add integration test for DASHPipeline
13. Add unit tests for MaxMin selection
14. Fix diversity.py:38 (random subsample)
15. Add warning when X_ref defaults to X_val

---

## VIII. Strategic Recommendation

The paper's biggest vulnerability is the Stochastic Retrain comparison. Consider reframing: the **primary contribution is the FSI/IS Plot diagnostic framework**, with stability improvement as supporting evidence. The diagnostics are genuinely novel (no existing tool provides per-feature stability auditing without ground truth), and they don't depend on beating Stochastic Retrain. This reframing turns the honest limitation into a non-issue: both methods achieve similar stability, but only DASH provides diagnostic tools.

The codebase quality and documentation transparency are publication-worthy. The methodology is sound. The paper needs its tables completed and its narrative sharpened, but the scientific foundation is solid.
