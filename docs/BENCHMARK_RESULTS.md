# Benchmark Results

**Source notebook:** `demo_benchmark_4_checkpointed.ipynb` (M=200, K=30, 20 repetitions)
**Authoritative notebook:** `demo_benchmark_6.ipynb` (incorporates all methodology fixes)

> Numbers below come from v4. Directional findings hold; exact numbers may shift when re-run with v6. See [Experiment Guide](../EXPERIMENT_GUIDE.md#methodology-fixes-applied) for the full list of fixes.

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

The benchmark runs 15 sections covering: correlation sweep, overlapping structure, nonlinear DGP, extended baselines (Table 2 with 10 methods), real-world datasets (California Housing, Breast Cancer, Superconductor), epsilon sensitivity, ablation studies, variance decomposition, and statistical significance tests.

---

## Baseline Comparison at ρ=0.95

The most demanding test: 50 features in 10 correlated groups at ρ=0.95, averaged across 20 repetitions. Each method is run independently from scratch at each repetition, and stability measures the mean pairwise Spearman correlation of importance rankings across runs.

```
Method                Stability   DGP Agreement (ρ)  Equity (CV)
=============================================================
Single Best              0.9529         0.9755         0.2421
Large Single Model       0.9301         0.9641         0.2708
DASH (MaxMin)            0.9819         0.9907         0.1585
```

DASH (MaxMin) leads on all three metrics at the highest collinearity level:

- **Stability:** 0.9819 vs. 0.9529 for Single Best (+0.0290). DASH produces nearly identical importance rankings across 20 independent runs. Single Best's ranking shifts substantially depending on which arbitrary feature choices a given seed produces.

- **DGP Agreement:** 0.9907 vs. 0.9755 for Single Best. DASH's consensus ranking is closer to the DGP-derived ground truth because averaging cancels the arbitrary feature-selection noise that biases any single model's ranking. (Note: this metric presupposes equitable within-group credit; see [Experiment Guide](../EXPERIMENT_GUIDE.md).)

- **Equity:** CV of 0.1585 vs. 0.2421 for Single Best (34% lower) and 0.2708 for Large Single Model (41% lower). Within each correlated group, DASH distributes importance fairly across all members rather than concentrating it on whichever feature one model happened to grab.

The Large Single Model -- which matches DASH's total compute budget in a single sequential ensemble -- performs worst on every metric. This is the direct evidence for sequential residual dependency: more trees in a single ensemble amplifies the first-mover bias rather than correcting it.

---

## The Correlation Sweep

The central result: how each method performs as correlation increases from 0.0 to 0.95 (full run, 20 repetitions per level).

**Stability:**
```
ρ=0.0:    DASH=0.9765   SB=0.9756   LSM=0.9649   (comparable -- safety check)
ρ=0.5:    DASH=0.9816   SB=0.9784   LSM=0.9702   (small advantage)
ρ=0.7:    DASH=0.9804   SB=0.9697   LSM=0.9635   (advantage emerges)
ρ=0.9:    DASH=0.9805   SB=0.9601   LSM=0.9401   (clear separation)
ρ=0.95:   DASH=0.9819   SB=0.9529   LSM=0.9301   (largest gap)
```

**Equity (within-group CV, lower is better):**
```
ρ=0.0:    DASH=0.1502   SB=0.1522   LSM=0.1547   (comparable)
ρ=0.5:    DASH=0.1510   SB=0.1653   LSM=0.1799   (advantage begins)
ρ=0.7:    DASH=0.1540   SB=0.1973   LSM=0.2021   (growing gap)
ρ=0.9:    DASH=0.1625   SB=0.2171   LSM=0.2528   (25% better than SB)
ρ=0.95:   DASH=0.1585   SB=0.2421   LSM=0.2708   (34% better than SB)
```

DASH stability is effectively flat across all correlation levels (0.9765-0.9819), while Single Best degrades from 0.9756 to 0.9529 and Large Single Model degrades from 0.9649 to 0.9301. DASH is immune to the correlation-induced instability that plagues single-model explanations because its independent models make their arbitrary choices independently, and averaging cancels the arbitrariness.

---

## Key Conclusions

1. **DASH's advantage is specifically about collinearity.** The stability gap widens from +0.0009 at ρ=0.0 to +0.0290 at ρ=0.95. At zero correlation, all methods perform similarly -- DASH is a targeted fix, not a blunt hammer.

2. **Bigger models make explanations worse, not better.** The Large Single Model -- matching DASH's total compute in a single sequential ensemble -- achieves the worst stability (0.9301), worst DGP agreement (0.9641), and worst equity (0.2708) of any method at ρ=0.95. Model independence, not model size, is what matters.

3. **DASH distributes credit fairly across correlated features.** At ρ=0.95, DASH's within-group CV of 0.1585 is 34% lower than Single Best (0.2421) and 41% lower than Large Single Model (0.2708). Where single models arbitrarily concentrate importance on one member of a correlated group, DASH's consensus reflects the group's collective contribution.

4. **DASH is safe when collinearity is absent (linear DGP).** At ρ=0.0, the DGP agreement gap between DASH and Single Best is 0.0005 -- effectively zero under the linear DGP. However, under nonlinear DGPs, DASH shows marginally lower stability at ρ=0.0 and ρ=0.5, so the safety guarantee is conditional on the DGP type.

5. **DASH also has the best predictive RMSE** at every correlation level, disproving the concern that diversified ensembles sacrifice prediction quality. At ρ=0.9: DASH RMSE=0.5821 vs Single Best=0.6043 vs Large Single Model=0.7177.

6. **Statistical rigor.** Wilcoxon signed-rank tests with Bonferroni correction show statistically significant improvements over Single Best at ρ≥0.7, with large effect sizes (Cohen's d > 1.0). Stability confidence intervals use BCa (bias-corrected and accelerated) bootstrap, which corrects for both bias and skewness. However, the comparison against Stochastic Retrain (the strongest baseline) is not statistically significant at ρ=0.9 (Cohen's d = 0.26, stability gap = 0.001), indicating that DASH's marginal improvement over simple seed averaging is modest.

7. **Robust to hyperparameters.** Epsilon sensitivity analysis shows <0.001 variation in stability across a 3× range of ε values (0.03 to 0.10). Ablation studies show diminishing returns past M=200.

---

## Success Criteria

The v6 benchmark defines 11 formal success criteria (expanded from 9 in v4). The v4 results below are directional; v6 adds criteria 10-11.

| # | Criterion | v4 Result | Threshold |
|---|-----------|-----------|-----------|
| 1 | Stability wins (DASH > SB, linear sweep) | **5/5** ρ levels | >= 80% |
| 2 | DGP agreement at ρ=0.9 (DASH >= SB) | **0.9901 >= 0.9796** | Relative to baseline |
| 3 | Equity wins (DASH CV < SB CV) | **5/5** ρ levels | >= 80% |
| 4 | Safety at ρ=0 (DGP agreement gap) | **0.0005** | < 0.1 |
| 5 | K_eff increases with ε | **5.8 → 27.1** | Monotonic |
| 6 | Nonlinear DGP: DASH > SB stability (ρ=0.9) | **0.8955 > 0.8403** | DASH wins |
| 7 | Breast Cancer: DASH stability > 0.80 | **0.9332** | > 0.80 |
| 8 | Superconductor: DASH stability > SB | **0.9654 > 0.8477** | DASH wins |
| 9 | Statistical significance (Bonferroni) | **17/26 = 65%** | >= 50% |
| 10 | Ablation robustness (M, K, ε, δ) | *(v6 only)* | Stability ≥ 0.95 across settings |
| 11 | Real-world ablation scores (DASH ≥ SB) | *(v6 only)* | DASH matches or exceeds |

---

## Breast Cancer Real-Data Results

The Breast Cancer dataset is a natural showcase for DASH because it contains 30 features with 21 pairs having |r| > 0.9. Features like `mean radius`, `mean perimeter`, and `mean area` are mathematically related and nearly interchangeable.

**Stability across 20 repetitions:**

| Method | Stability |
|--------|-----------|
| Single Best | 0.5341 |
| **DASH (MaxMin)** | **0.9332** |

DASH nearly doubles stability on this heavily collinear dataset (+0.3991). This is the most dramatic improvement across all experiments. Top features by consensus importance: `mean concave points` (0.2314), `worst perimeter` (0.2166), `worst concave points` (0.2011).

**The IS Plot reveals the correlation structure unsupervised:**
- Features like `worst concave points` and `worst perimeter` appear as **Robust Drivers** (Quadrant I) -- high importance, low FSI, consistently important across all models.
- Features like `mean radius` and `mean perimeter` appear as **Collinear Cluster Members** (Quadrant II) -- high importance but high FSI, because different models attribute importance to different members of this correlated trio.
- Many of the "SE" (standard error) features appear as **Confirmed Unimportant** (Quadrant III).

**The Local Disagreement Map** for a high-variance patient shows which feature attributions are trustworthy (narrow error bars, e.g., texture and concavity features) and which are model-dependent (wide error bars, e.g., radius vs. perimeter). In a clinical setting, this tells the physician which parts of the explanation are reliable versus uncertain.

---

## Superconductor UCI Real-Data Results

The Superconductor dataset (21,263 samples, 81 features) provides a larger-scale real-world validation with scale-appropriate epsilon (SC_EPSILON=0.40).

| Method | Stability | RMSE |
|--------|-----------|------|
| Single Best | 0.8477 | 9.02±0.09 |
| Large Single Model | 0.7018 | 9.17±0.08 |
| **DASH (MaxMin)** | **0.9654** | **8.97±0.08** |

DASH improves stability by +0.1177 over Single Best and +0.2636 over Large Single Model, while also achieving marginally better RMSE.

---

## Nonlinear DGP Results

DASH's advantage persists under a nonlinear data-generating process with interactions and nonlinear terms at moderate-to-high correlation. At ρ=0.9: DASH stability=0.8955 vs Single Best=0.8403 (+0.0552). At ρ=0.95: DASH stability=0.8955 vs Single Best=0.8191 (+0.0764). All methods degrade more under nonlinearity (stability drops from ~0.98 to ~0.89), but DASH degrades less at high ρ.

**Caveat:** At ρ=0.0 and ρ=0.5, DASH shows marginally *lower* stability than Single Best (0.9420 vs 0.9437 and 0.8678 vs 0.8769, respectively), violating the safety desideratum. DASH's advantage emerges only at ρ≥0.7 under nonlinearity. Practitioners working with nonlinear relationships and low correlation should verify that DASH does not introduce unnecessary noise for their specific use case.

---

## Epsilon Sensitivity

DASH is robust to the performance filter threshold ε. Across ε ∈ {0.03, 0.05, 0.08, 0.10} at ρ=0.9, stability varies by <0.001 (0.9794-0.9805). The effective ensemble size K_eff scales with ε (5.8 → 27.1), but performance plateaus early, meaning practitioners don't need to carefully tune ε.
