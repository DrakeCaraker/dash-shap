# Analysis: `notebooks/demo_benchmark_4.ipynb` — Prototype Run

**Date**: 2026-03-06
**Scope**: Rigorous analysis of results from what is acknowledged as a prototype/proof-of-concept run

---

## Execution Status

**14 of 35 code cells produced output.** Cells 1–26 (Sections 1–5 plus intermediate success criteria) ran successfully. Everything from Section 6 onward (cells 28–56) did not execute. This includes: epsilon sensitivity, nonlinear DGP, statistical significance tests, extended baselines (Table 2), Superconductor, ablation studies, publication figures, California Housing, Cohen's d, bootstrap CIs, and the final success criteria.

The notebook validates pipeline mechanics on the core synthetic and one real-world scenario, not the full experimental sweep.

---

## Results Summary

### Section 1: Proof of Concept (ρ=0.9, Linear DGP)

- **Population:** 500 models trained; **11/500 passed** the ε=0.03 filter; diversity selection yielded **K_eff = 3** (hit δ=0.05 minimum-distance threshold).
- **K_eff = 3 is extremely low** for a pipeline configured with K=30. The tight ε=0.03 with M=500 produces a tiny candidate pool. This is the exact concern flagged in the audit's Section 6 ("Priority 0A: Epsilon Sensitivity"), which did not run.
- **Accuracy:** Spearman ρ = 0.9851 vs ground truth — excellent but comparable to all baselines except Large Single Model.
- **Equity:** Within-group CV = 0.2295 — middling; DASH Dedup (0.2124) and DASH Cluster (0.2170) performed better on this single run.

### Section 2: Full Baseline Comparison (single run, ρ=0.9)

| Method | Spearman ρ | Within-Group CV |
|--------|-----------|----------------|
| Single Best | 0.9860 | 0.1950 |
| Large Single Model | 0.9479 | 0.3544 |
| Ensemble SHAP | 0.9831 | 0.2389 |
| Naive Top-N | 0.9860 | 0.2348 |
| Stochastic Retrain | 0.9855 | 0.2324 |
| DASH (Dedup) | 0.9855 | 0.2124 |
| DASH (MaxMin) | 0.9851 | 0.2295 |
| DASH (Cluster) | **0.9875** | **0.2170** |

All methods achieve ρ > 0.98 except Large Single Model (0.9479). Differences are within noise for a single run. DASH Cluster edges ahead in accuracy; DASH Dedup wins on equity. These are single-run point estimates and not statistically meaningful.

### Section 3: Stability Across 20 Repetitions (ρ=0.9) — Most Meaningful Result

| Method | Stability | Accuracy (ρ) | Equity (CV) |
|--------|-----------|-------------|-------------|
| Single Best | 0.9603 | 0.9797 | 0.2182 |
| Large Single Model | 0.9396 | 0.9691 | 0.2539 |
| Naive Top-N | 0.9790 | 0.9890 | 0.1768 |
| **DASH (MaxMin)** | **0.9810** | **0.9901** | **0.1655** |
| DASH (Cluster) | 0.9796 | 0.9893 | 0.1723 |

DASH (MaxMin) wins on all three metrics over Single Best:
- **Stability:** +0.021 (2.2% relative improvement)
- **Accuracy:** +0.010 (1.1% relative improvement)
- **Equity:** −0.053 lower CV (24% relative improvement — the most substantial margin)

The margin over Naive Top-N is slimmer but consistent across all metrics.

### Section 4: Correlation Sweep (ρ ∈ {0.0, 0.5, 0.7, 0.9, 0.95}) — Core Narrative

20 reps per ρ level, 3 methods (Single Best, Large Single Model, DASH MaxMin):

| ρ | DASH Stab | SB Stab | Δ Stab | DASH Eq (CV) | SB Eq (CV) | Δ Equity |
|---|-----------|---------|--------|-------------|-----------|----------|
| 0.0 | 0.9778 | 0.9744 | +0.003 | 0.1511 | 0.1539 | −0.003 |
| 0.5 | 0.9815 | 0.9781 | +0.003 | 0.1583 | 0.1667 | −0.008 |
| 0.7 | 0.9781 | 0.9699 | +0.008 | 0.1719 | 0.1980 | −0.026 |
| 0.9 | 0.9810 | 0.9603 | +0.021 | 0.1655 | 0.2182 | −0.053 |
| 0.95 | 0.9806 | 0.9527 | +0.028 | 0.1618 | 0.2411 | −0.079 |

**Key finding:** DASH's advantage grows monotonically with collinearity. At ρ=0.0, gaps are negligible. At ρ=0.95, DASH is +0.028 better in stability and −0.079 better in equity CV. Large Single Model consistently underperforms both. The monotonic growth pattern is the strongest evidence supporting DASH's core thesis.

### Section 5: Breast Cancer (Real Data)

- Pipeline ran end-to-end on a classification task (30 features, 21 highly correlated pairs).
- **220/500 models passed** the ε=0.02 filter → full K=30 ensemble selected. Healthy K_eff contrasting with the synthetic K_eff=3 problem.
- Top features: worst perimeter, worst concave points, worst area — clinically sensible.
- **Breast Cancer repetition analysis (cell 24) did not produce output**, so no stability/equity numbers for this dataset.

### Intermediate Success Criteria (Cell 26)

All 4 criteria passed:
1. Stability wins: 5/5 ρ levels (**PASS**)
2. Accuracy at ρ=0.9: 0.9901 (**PASS**, target ≥ 0.90)
3. Equity wins: 5/5 (**PASS**)
4. ρ=0 safety control: gap = 0.0019 (**PASS**, need < 0.1)

---

## Critical Observations

### 1. The K_eff = 3 Problem

At ε=0.03 with M=500, only 11 models pass the performance filter, and maxmin diversity selection reduces this to K=3 before hitting the distance threshold. This means the "diversity" mechanism is barely operative — DASH's advantage may derive almost entirely from model averaging over 3 near-identical models, not from genuine diversity selection across 30 complementary models.

The Breast Cancer result (K_eff=30 at ε=0.02) suggests this is specific to the synthetic DGP / epsilon combination, not a fundamental flaw. The unexecuted epsilon sensitivity sweep (Section 6) was designed to investigate exactly this.

### 2. Narrow Comparison in the Sweep

The correlation sweep compares only 3 methods (Single Best, Large Single Model, DASH MaxMin). The full 7-method comparison exists only as a single-run snapshot. This means we lack multi-rep evidence for how DASH compares to Ensemble SHAP, Stochastic Retrain, Naive Top-N, and DASH Dedup across collinearity levels.

### 3. Effect Sizes Are Small at Low Collinearity

At ρ ≤ 0.5, the stability and equity gaps are ≤ 0.008. Without the Wilcoxon tests (Section 9, not run), we cannot determine whether these are statistically significant or within noise. The practical significance at low ρ is questionable.

### 4. Monotonic Growth Pattern Is Robust

The most compelling result: DASH's advantage strictly increases with ρ across all 5 levels, in both stability and equity. This is exactly the behavior predicted by the theoretical motivation (collinearity → SHAP instability → diversity selection helps). The consistency of this pattern across 20 repetitions per level makes it unlikely to be spurious.

### 5. Equity Is the Strongest Differentiator

The largest margins are consistently in within-group equity (CV). At ρ=0.95, DASH achieves 33% lower CV than Single Best (0.1618 vs 0.2411). This suggests DASH's primary practical value is in producing more balanced importance distributions across correlated feature groups, rather than dramatically improving rank accuracy.

---

## What the Prototype Establishes

1. **The DASH pipeline runs end-to-end** (population → filter → diversity → consensus → diagnostics) on both synthetic regression and real-world classification.
2. **The evaluation framework is functional** (stability, accuracy, equity metrics; repetition analysis; correlation sweep).
3. **The core hypothesis is directionally supported**: DASH improves stability and equity under collinearity, with growing advantage as ρ increases.
4. **All 4 success criteria pass** on the subset of results that executed.

## What the Prototype Cannot Support

1. **Paper-level claims** — 21 unexecuted cells represent the bulk of experimental evidence needed for publication.
2. **Epsilon sensitivity conclusions** — the K_eff = 3 issue is unresolved.
3. **Nonlinear generalization** — no evidence DASH works beyond linear targets.
4. **Statistical significance** — no Wilcoxon tests, no effect sizes, no bootstrap CIs.
5. **Ablation robustness** — no M/K/ε/δ sensitivity analysis.
6. **Real-world regression** — Superconductor and California Housing did not run.

---

## Bottom Line

The machinery works and the directional results are internally consistent and encouraging. The monotonic growth of DASH's advantage with collinearity is the strongest finding. However, this is a proof-of-concept validating pipeline correctness, not a complete benchmark. A full run executing all 57 cells is required before any publication-level claims can be made.
