# Benchmark Results

> **Version guide**
> | Version | Notebook | N_REPS | Status | Use for |
> |---------|----------|--------|--------|---------|
> | **v7 (TMLR)** | `demo_benchmark_7_parallel.ipynb` | 50 | ⚠️ IN PROGRESS | TMLR submission |
> | **v6 (ArXiv/Zenodo)** | `demo_benchmark_6.ipynb` | 20 | ✅ FROZEN | ArXiv/Zenodo; responding to ArXiv comments |
>
> When writing the TMLR paper, use v7 numbers only. Use `/paper-context` for TMLR, `/paper-context-arxiv` for ArXiv.

---

## v7 Results (TMLR — in progress)

> ⚠️ **Status: PENDING** — `demo_benchmark_7_parallel.ipynb` has not yet completed its full run.
> Paste results here as the notebook produces them. Do not cite these numbers in the TMLR draft until confirmed.

**Configuration:** M=200, K=30, N_REPS=50, EPSILON=0.08, DELTA=0.05, SEED=42

*Results to be filled in when notebook completes.*

---

## v6 Results (ArXiv/Zenodo — FROZEN)

> ✅ **These numbers are published on ArXiv/Zenodo. Do not modify.**
> For TMLR writing, use v7 numbers above once available.

**Canonical source:** `notebooks/demo_benchmark_6.ipynb` (M=200, K=30, 20 repetitions, PAPER_CONFIG)

> All numbers in this section are sourced from `demo_benchmark_6.ipynb`. The paper (`paper/draft_v6_preprint.tex`) and this section should always agree; any discrepancy should be resolved in favor of `demo_benchmark_6.ipynb`.

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Population size (M) | 200 |
| Selected models (K) | 30 |
| Repetitions (N_REPS) | 20 |
| Performance filter (ε) | 0.08 (synthetic); REAL_EPSILON=0.05 relative (real-world) |
| Diversity threshold (δ) | 0.05 |
| Data split | train / val / explain / test (four-way) |

The benchmark runs 15 sections covering: correlation sweep, overlapping structure, nonlinear DGP, extended baselines (9 methods), real-world datasets (California Housing, Breast Cancer, Superconductor), epsilon sensitivity, ablation studies, variance decomposition, and statistical significance tests.

---

## Baseline Comparison at ρ=0.9

The central comparison: 50 features in 10 correlated groups at ρ=0.9, averaged across 20 repetitions.

```
Method                Stability   DGP Agreement (ρ)  Equity (CV)   RMSE
=====================================================================
Single Best              0.9583         0.9784         0.2235      0.614
Single Best (M=200)      0.9643         0.9813         0.2122      ---
Large Single Model       0.9380         0.9674         0.2616      0.738
LSM (Tuned)              0.9475         0.9721         0.2720      ---
Stochastic Retrain       0.9770         0.9877         0.1823      0.577
Random Selection         0.9760         0.9875         0.1874      ---
DASH (MaxMin)            0.9770         0.9879         0.1760      0.594
Ensemble SHAP            0.9559         0.9766         0.2372      ---
DASH (Dedup)             0.9759         0.9877         0.1646      ---
```

---

## The Correlation Sweep

The central result: how each method performs as correlation increases from 0.0 to 0.95 (20 repetitions per level).

**Stability:**
```
ρ=0.0:    DASH=0.9722   SB=0.9733   LSM=0.9533   SR=0.9754
ρ=0.5:    DASH=0.9770   SB=0.9749   LSM=0.9647   SR=0.9796
ρ=0.7:    DASH=0.9774   SB=0.9692   LSM=0.9626   SR=0.9798
ρ=0.9:    DASH=0.9770   SB=0.9583   LSM=0.9380   SR=0.9770
ρ=0.95:   DASH=0.9772   SB=0.9508   LSM=0.9248   SR=0.9787
```

**Equity (within-group CV, lower is better):**
```
ρ=0.0:    DASH=0.164   SB=0.168   LSM=0.170
ρ=0.5:    DASH=0.163   SB=0.178   LSM=0.194
ρ=0.7:    DASH=0.162   SB=0.193   LSM=0.209
ρ=0.9:    DASH=0.176   SB=0.224   LSM=0.262
ρ=0.95:   DASH=0.172   SB=0.246   LSM=0.284
```

DASH stability is effectively flat across all correlation levels (0.972–0.977), while Single Best degrades from 0.973 to 0.951 and Large Single Model degrades from 0.953 to 0.925. DASH is immune to the correlation-induced instability that plagues single-model explanations because its independent models make their arbitrary choices independently, and averaging cancels the arbitrariness.

---

## Key Conclusions

1. **DASH's advantage is specifically about collinearity.** The stability gap widens from near-zero at ρ=0.0 to +0.026 at ρ=0.95. At zero correlation, all methods perform similarly — DASH is a targeted fix, not a blunt hammer.

2. **Bigger models make explanations worse, not better.** The Large Single Model — matching DASH's total tree count in a single sequential ensemble — achieves the worst stability (0.925), worst DGP agreement (0.961), and worst equity (0.284) of any method at ρ=0.95. Model independence, not model size, is what matters.

3. **DASH distributes credit fairly across correlated features.** At ρ=0.95, DASH's within-group CV of 0.172 vs Single Best 0.246 and LSM 0.284. Where single models arbitrarily concentrate importance on one member of a correlated group, DASH's consensus reflects the group's collective contribution.

4. **DASH is safe when collinearity is absent (linear DGP).** At ρ=0.0, the stability gap between DASH and Single Best is 0.001 — effectively zero under the linear DGP. However, under nonlinear DGPs, DASH shows marginally lower stability at ρ=0.0 and ρ=0.5, so the safety guarantee is conditional on the DGP type.

5. **DASH also achieves competitive predictive RMSE** at every correlation level, disproving the concern that diversified ensembles sacrifice prediction quality.

6. **Statistical rigor.** Wilcoxon signed-rank tests with Holm–Bonferroni correction show statistically significant improvements over Single Best at ρ≥0.7, with large effect sizes (Cohen's d > 1.0). The comparison against Stochastic Retrain is not statistically significant at ρ=0.9 (accuracy d=+0.05, equity d=-0.21), indicating that the operative mechanism is model independence rather than any particular pipeline design.

7. **Robust to hyperparameters.** Epsilon sensitivity analysis shows <0.005 variation in stability across a 3× range of ε values (0.03 to 0.10). Ablation studies show diminishing returns past M=200.

---

## Success Criteria

The benchmark defines 11 formal success criteria. All pass.

| # | Criterion | Result | Threshold |
|---|-----------|--------|-----------|
| 1 | Stability wins (DASH > SB, linear sweep) | **4/5** ρ levels | >= 4/5 |
| 2 | Accuracy at ρ=0.9 (DASH >= SB) | **0.9879 >= 0.9784** | Relative to baseline |
| 3 | Equity wins (DASH CV < SB CV) | **5/5** ρ levels | >= 4/5 |
| 4 | Safety at ρ=0 (stability gap) | **0.0003** | < 0.1 |
| 5 | K_eff increases with ε | **4.0 → 6.5 → 12.2 → 16.2** | Monotonic |
| 6 | Nonlinear DGP: DASH > SB stability (ρ=0.9) | **0.8734 > 0.8336** | DASH wins |
| 7 | Statistical significance | **17/26 Bonferroni, 15/26 Holm–Bonferroni** | >= 50% |
| 8 | Superconductor: DASH > SB | **0.962 > 0.830** | DASH wins |
| 9 | California Housing: DASH > SB | **0.982 > 0.967** | DASH wins |
| 10 | Breast Cancer: DASH > SB | **0.930 > 0.317** | DASH wins |
| 11 | Variance decomposition | **DASH=0.006 vs SB=0.023** | DASH < SB |

---

## Breast Cancer Real-Data Results

The Breast Cancer dataset is a natural showcase for DASH because it contains 30 features with 21 pairs having |r| > 0.9. Features like `mean radius`, `mean perimeter`, and `mean area` are mathematically related and nearly interchangeable.

**Stability across 20 repetitions:**

| Method | Stability (±SE) |
|--------|-----------------|
| Single Best (M=200) | 0.317 ± 0.053 |
| **DASH (MaxMin)** | **0.930 ± 0.005** |

DASH improves stability by +0.614 on this heavily collinear dataset. This is the most dramatic improvement across all experiments. The Single Best (M=200) baseline is tree-count-matched: it trains 200 models and selects the best, yet still produces essentially random importance rankings across runs.

---

## Superconductor UCI Real-Data Results

The Superconductor dataset (21,263 samples, 81 features) provides a larger-scale real-world validation with relative epsilon (REAL_EPSILON=0.05).

| Method | Stability (±SE) | RMSE |
|--------|-----------------|------|
| Single Best | 0.830 ± 0.02 | 9.18 ± 0.11 |
| Large Single Model | 0.689 ± 0.03 | 9.34 ± 0.09 |
| **DASH (MaxMin)** | **0.962 ± 0.01** | **9.15 ± 0.08** |

DASH improves stability by +0.132 over Single Best and +0.273 over Large Single Model, while also achieving marginally better RMSE.

---

## California Housing Real-Data Results

| Method | Stability (±SE) | RMSE |
|--------|-----------------|------|
| Single Best | 0.967 ± 0.01 | 0.460 ± 0.009 |
| **DASH (MaxMin)** | **0.982 ± 0.005** | **0.450 ± 0.005** |

Modest improvement (+0.015), consistent with the mild degree of collinearity in this 8-feature dataset.

---

## Nonlinear DGP Results

DASH's advantage persists under a nonlinear data-generating process with interactions and nonlinear terms at moderate-to-high correlation. At ρ=0.9: DASH stability=0.873 vs Single Best=0.834 (+0.039). At ρ=0.95: DASH stability=0.876 vs Single Best=0.798 (+0.078). All methods degrade more under nonlinearity (stability drops from ~0.93 to ~0.87), but DASH degrades less at high ρ.

**Caveat:** At ρ=0.0, DASH and Single Best perform nearly identically (0.934 vs 0.933). At ρ=0.5, DASH shows a marginal advantage (0.852 vs 0.849). DASH's advantage emerges clearly only at ρ≥0.7 under nonlinearity.

---

## Epsilon Sensitivity

DASH is robust to the performance filter threshold ε. Across ε ∈ {0.03, 0.05, 0.08, 0.10} at ρ=0.9:

| ε | Models Passing | K_eff | Stability | Accuracy | Equity |
|---|----------------|-------|-----------|----------|--------|
| 0.03 | 9.7 | 4.0 ± 1.3 | 0.9734 | 0.9861 | 0.193 |
| 0.05 | 22.2 | 6.5 ± 1.9 | 0.9747 | 0.9868 | 0.191 |
| 0.08 | 48.0 | 12.2 ± 2.6 | 0.9770 | 0.9879 | 0.176 |
| 0.10 | 67.5 | 16.2 ± 3.3 | 0.9777 | 0.9882 | 0.174 |

Stability varies by <0.005 across a 3× range. Performance plateaus early.

---

## Population Size Ablation

| M | Stability | Accuracy |
|---|-----------|----------|
| 50 | 0.9727 | 0.9848 |
| 100 | 0.9719 | 0.9844 |
| 200 | 0.9722 | 0.9850 |
| 500 | 0.9722 | 0.9848 |

Diminishing returns past M=100; M=200 is the default for a margin of safety.
