# DASH Experimental Guide

**What the experiments test, why each piece matters, and how to interpret the results.**

**Authoritative notebook**: `notebooks/demo_benchmark_7.ipynb` (53 cells, mechanism-first structure, checkpointed)
**Automated script**: `run_experiments.py` (mirrors the notebook, outputs to `results/`)

## The Problem DASH Solves

When you train an XGBoost model on data with correlated features, the model has to pick *one* feature from each correlated group at each split point. Which one it picks is essentially arbitrary -- feature A and feature B carry the same signal, and the model grabs whichever gives a marginal gain advantage at that specific split. Change the hyperparameters slightly (tree depth, learning rate, column subsampling) and the model picks different members of the correlated group. The predictions barely change, but the SHAP values shift dramatically.

This is the Rashomon effect applied to explanations: many models fit the data equally well, but they tell completely different stories about *which features matter*. For anyone using SHAP to make decisions -- feature selection, scientific hypothesis generation, regulatory audit -- this is a serious problem. You're looking at an artifact of model specification, not a property of the data.

## How DASH Works

Instead of trusting one model's arbitrary feature selection, DASH deliberately trains a population of models that are *forced* to use different features. The key mechanism is restricting `colsample_bytree` to low values (0.1-0.5), so each tree only sees a small fraction of features. A model that achieves good predictive accuracy with only 20% of features visible per tree has necessarily found a *different* path through the correlated feature space than a model using a different 20%.

After filtering for performance (only keeping models that actually learned signal) and selecting for diversity (ensuring the ensemble covers different feature utilization patterns), you average their SHAP matrices element-wise. The consensus explanation fairly distributes importance across the correlated group rather than concentrating it on whichever member one model happened to grab.

---

## Canonical Configuration

All experiments share a single `PAPER_CONFIG` defined in the notebook's setup cell and mirrored in `run_experiments.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| M | 200 | Population size (models trained) |
| K | 30 | Maximum models selected for ensemble |
| N_REPS | 20 | Repetitions per experiment (for stability measurement) |
| EPSILON | 0.08 | Performance filter threshold (absolute mode, synthetic data) |
| DELTA | 0.05 | Diversity floor for MaxMin selection |
| SEED | 42 | Base random seed |
| N_TRIALS_SB | 30 | Hyperparameter trials for Single Best baseline |
| T_PER_MODEL | 500 | Trees per model in Large Single Model |
| N_ESTIMATORS_ESHAP | 2000 | Trees for Ensemble SHAP baseline |
| TAU_CLUSTER | 0.3 | Cluster coverage threshold |

**Real-world datasets** use `REAL_EPSILON=0.05` with `epsilon_mode='relative'`. The absolute ε=0.08 is calibrated for synthetic data where validation RMSE is ~2-5. For datasets with different RMSE scales (e.g., Superconductor ~17-20), relative mode filters models within 5% of the best score, ensuring a comparable fraction of the population passes.

---

## The 9 Methods (+ 2 Variants) and What Each Tests

The paper's Table 2 compares 9 primary methods. Two additional DASH variants (Cluster, Dedup) are tested in specific contexts only.

**Single Best**: The standard practice. Tune one XGBoost model across 30 hyperparameter trials, compute SHAP, report importance. This is what most practitioners do today. It is the baseline to beat.

**Single Best (M=200)** (M3 fix): The same single-best approach but searching over 200 trials instead of 30 -- matching DASH's total model training budget. This isolates whether DASH's advantage comes from its diversity-based selection or simply from searching a larger hyperparameter space.

**Large Single Model (LSM)**: The sequential residual dependency test. Trains a single XGBoost with the *same low colsample_bytree* (0.2) that DASH uses and K x T_per_model total trees -- matching DASH's total tree budget. The question it answers: does DASH's advantage come simply from using low colsample_bytree, or does it specifically require *independent* models? Within a single boosting ensemble, sequential residual dependency creates a path-dependent "first mover" bias. If DASH outperforms LSM, it proves that breaking residual dependency matters.

**LSM (Tuned)**: LSM with hyperparameter tuning enabled. Tests whether careful tuning of a single large model can close the gap with DASH.

**Ensemble SHAP** (Paillard et al. baseline): Trains a single large XGBoost with *standard* high colsample_bytree (0.8) and 2000 trees, then computes SHAP. Tests the argument that you should explain one big ensemble rather than aggregate explanations.

**Naive Top-N**: Takes the top K models by validation performance from DASH's population and averages their SHAP matrices *without* diversity selection. Isolates whether diversity selection matters or whether simple averaging is enough. If Naive Top-N matches DASH, the diversity selection is unnecessary overhead.

**Stochastic Retrain**: K models with identical hyperparameters but different random seeds (B3 fix: N=K for fair comparison). SHAP matrices averaged. Tests whether DASH's deliberate hyperparameter diversification is better than natural stochastic variation.

**Random Selection** (M2 fix): Trains the same population as DASH, applies the same performance filter, but selects K models *randomly* instead of using MaxMin diversity. Isolates the value of diversity-aware selection specifically.

**DASH (MaxMin)**: The recommended default. Greedy max-min dissimilarity selection ensures each added model is maximally different from all previously selected models in its feature utilization pattern. Does not require the correlation matrix.

**DASH (Cluster)** *(variant, overlapping experiment only)*: Feature cluster coverage selection. Uses the correlation matrix to identify correlated groups, then selects models covering different clusters. Strong when correlation structure is clean block-diagonal. Not included in the paper's main methods table.

**DASH (Dedup)** *(variant, Table 2 only)*: Rank correlation deduplication. Removes models whose importance vectors are too similar (Spearman rho > 0.95) but does not actively seek diversity. The weakest DASH variant -- a sanity check that even minimal deduplication helps. Evaluated at rho=0.9 only.

### Expected ranking at high rho

DASH (MaxMin) ~ DASH (Cluster) > DASH (Dedup) > Naive Top-N ~ Stochastic Retrain > Random Selection > Ensemble SHAP ~ Large Single Model ~ Single Best.

The key comparison is DASH (MaxMin) vs Large Single Model. Both use low colsample_bytree. DASH trains independent models; LSM trains one sequential ensemble. The gap confirms the first-mover hypothesis.

---

## Experiment 1: Correlation Sweep -- The Central Claim

This is the experiment that makes or breaks the paper. It tests the hypothesis that DASH produces more stable, accurate, and equitable importance rankings than baselines, and that the advantage grows with collinearity.

### Setup

The synthetic data has 50 features in 10 groups of 5, where within-group correlation is rho. The target is a linear combination of group means, with known coefficients descending from 2.0 to 0.0. Because the DGP is linear and symmetric within groups, we know the ground-truth importance: every feature in group g should get importance |beta_g|/5.

We sweep rho in {0.0, 0.5, 0.7, 0.9, 0.95} and run **20 repetitions** at each level. For each repetition, we regenerate the data (same coefficients, new random draws) and run all 7 sweep methods.

### Sweep methods (8)

Single Best, Single Best (M=200), Large Single Model, LSM (Tuned), Stochastic Retrain, Random Selection, Naive Top-N, and DASH (MaxMin). Three additional methods (Ensemble SHAP, Stochastic Retrain, DASH Dedup) are evaluated at rho=0.9 only in the Table 2 section.

### Four metrics

**Stability** (the headline metric): After running the full pipeline 20 times with different random seeds, how consistent are the importance rankings? Measured as the mean pairwise Spearman correlation across all pairs of runs. BCa bootstrap confidence intervals are computed for each method at each rho level.

**DGP Agreement** (formerly "Accuracy"): Spearman correlation between estimated importance and known ground truth. Reported as a sanity check alongside stability and equity, not as the primary evaluation criterion. The ground truth presupposes equitable within-group credit distribution, making this partially circular with equity. **Note**: The paper uses "Accuracy" for brevity; the code uses both `dgp_agreement()` and `importance_accuracy()` (aliases).

**Within-group equity**: Coefficient of variation of importance within each correlated group. Lower is better. Groups with near-zero mean importance are excluded by default.

**RMSE**: Test-set prediction error. Verifies that stability gains do not come at the cost of prediction quality.

### Group-level metrics

**Group-level accuracy (gacc)**: Spearman ρ of group-level importance sums vs. true group betas. Separates "does the method rank groups correctly?" from "does it distribute credit within groups fairly?" **Caveat (C8):** With 10 groups and true betas spanning a 20x range (`[2.0, 1.5, ..., 0.1, 0.0]`), Spearman rank order is trivially preserved by almost any model, causing gacc to saturate at 1.0. This metric only discriminates methods when group betas are close in magnitude.

**Group-level MSE (gmse)**: Normalized MSE of group-level importance proportions vs. true group proportions. Unlike gacc, this captures magnitude accuracy — how well the estimated group-level budget matches the true budget, not just its rank order. This is the discriminative complement to gacc at high ρ where gacc saturates.

### Additional tracking

- **K_eff**: Effective ensemble size (how many models MaxMin actually selects) is tracked per rep for all DASH and ensemble methods.
- **Wall-clock time**: Per-method per-rho timing for compute budget comparison.
- **Per-rep arrays**: Accuracy, equity, and RMSE values are saved per repetition to enable statistical significance tests in Section 9.

### Expected results

**At rho=0** (safety control): All methods should perform comparably. DASH has no advantage -- but it also shouldn't *hurt*.

**At rho=0.9 and rho=0.95**: DASH should dominate. Single Best stability should degrade sharply. The equity gap should be large.

The correlation sweep plot (3 panels: stability, accuracy, equity vs rho) is the paper's central figure.

---

## Experiment 2: Overlapping Correlation Structure

Tests robustness when the correlation structure is not clean block-diagonal. Uses overlapping groups where features at group boundaries are correlated with *both* adjacent groups -- creating chain correlations where A correlates with B and B correlates with C, but A and C are only weakly correlated.

This tests robustness. MaxMin should be particularly robust here because it does not assume any specific correlation structure. Results are saved to `results/tables/overlapping.json` and include stability, accuracy, group-level accuracy, equity, and RMSE.

---

## Experiment 3: Nonlinear DGP

The nonlinear DGP has quadratic terms, interactions (z1 * z2), and a sinusoidal component:

```
y = 1.5 * z1^2 + 0.8 * z1 * z2 + 1.2 * sin(pi * z3) + linear tail + noise
```

where z_g is the mean of group g's features.

**Important caveat (C4)**: The `true_importance` values for the nonlinear DGP are *approximate ordinal rankings*, not exact analytic SHAP values. The coefficients (1.5, 0.8, 1.2) reflect relative magnitude of DGP terms and should be interpreted as group-level rankings rather than cardinal ground truth. Under nonlinearity and collinearity, true SHAP values depend on the joint feature distribution in ways that resist closed-form computation. For this reason, the nonlinear experiment evaluates primarily *stability* and *within-group equity*, not accuracy against ground truth.

5 methods x 5 rho levels x 20 reps. Includes LSM (Tuned) for fair comparison.

---

## Experiment 4: Real Data

Three real-world datasets validate DASH beyond synthetic settings. All use a four-way data split (A4 fix): train / val / explain / test. SHAP is computed on the explain set (`X_ref=X_explain`), separate from the test set used for RMSE evaluation.

### California Housing (8 features, regression)

Natural collinearity -- median income correlates with house value, number of rooms and bedrooms are correlated, latitude/longitude are correlated. Uses `REAL_EPSILON=0.05` with `epsilon_mode='relative'` (F2 fix).

The scaler is re-fit per repetition from raw (unscaled) data to avoid leakage (D2 fix). Reports stability, RMSE, and feature ablation scores (M5 fix: ablation uses the first selected model as a proxy).

### Breast Cancer (30 features, binary classification)

Heavy natural collinearity -- radius, perimeter, and area are mathematically related; mean, SE, and worst-case versions of each measurement create ~21 feature pairs with |r| > 0.9. Uses `task='binary'`.

The IS Plot reveals the correlation structure unsupervised. Features like "mean radius" and "mean perimeter" land in the Collinear Cluster quadrant; "mean concavity" lands in Robust Drivers. The Local Disagreement Map for the highest-variance patient shows which feature attributions are trustworthy.

20-rep stability analysis compares Single Best vs DASH, with full bootstrap CIs. Feature ablation scores are computed per rep.

### Superconductor UCI (81 features, regression)

Large-scale validation with 21,263 samples. Uses `REAL_EPSILON=0.05` with `epsilon_mode='relative'`. 3 methods (Single Best, LSM, DASH MaxMin) x 20 reps.

### Significance tests on real-world data

All three real-world experiments now include Wilcoxon signed-rank tests and Cohen's d effect sizes (C7, F1 fixes) comparing DASH vs each baseline on RMSE and feature ablation scores.

---

## Experiment 5: Epsilon Sensitivity

Tests the performance filter threshold epsilon across {0.03, 0.05, 0.08, 0.10} at rho=0.9.

**Key design**: The model population is trained *once* per repetition, then re-filtered at each epsilon value. This isolates the filter threshold effect from training stochasticity. 20 reps.

Reports:
- **Models passing**: How many of M=200 models survive each filter threshold
- **K_eff**: How many models MaxMin actually selects (may be less than K=30)
- **Stability, accuracy, equity**: Performance at each epsilon

Expected: DASH is robust across a 3x range of epsilon values. K_eff increases with epsilon but performance plateaus early.

**Note**: SHAP is computed on the explain set (`X_ref=X_explain`), not the test set, to avoid data leakage (D2 fix applied in v6).

---

## Experiment 6: Ablation Studies

One-at-a-time parameter variation to characterize DASH's sensitivity to each hyperparameter:

- **M** (population size) in {50, 100, 200, 500}
- **K** (max selected) in {5, 10, 20, 30, 50}
- **epsilon** (performance filter) in {0.01, 0.03, 0.05, 0.08, 0.10}
- **delta** (diversity floor) in {0.01, 0.05, 0.10, 0.20}

Default baseline: M=200, K=30, epsilon=0.08, delta=0.05.

**Multi-rho ablation**: The full ablation runs at rho in {0.0, 0.9, 0.95} to verify that trends hold at the safety control (rho=0), the primary evaluation point (rho=0.9), and extreme collinearity (rho=0.95).

**N_REPS=20** per setting (C1 fix: standardized from the earlier ABL_N_REPS=10 to match the main sweep for comparable stability estimates).

---

## Experiment 7: Variance Decomposition

Decomposes the sources of importance instability into data-sampling vs model-selection components.

### Three conditions (at rho=0.9)

1. **Data-fixed**: Fix data seed, vary model seeds (20 reps). Only model-selection randomness remains -- isolates model-selection instability.
2. **Model-fixed**: Fix model seed, vary data seeds (20 reps). Only data-sampling randomness remains -- isolates data-sampling instability.
3. **Both-varied**: Vary both (20 reps). Total instability (reference).

For each condition, importance vectors are collected and stability is computed.

### Interpretation

Instability is approximated as `1 - stability` for each condition, then the fraction attributable to each source is reported.

**Caveat (C5)**: `1 - stability` is a proxy for instability, not a proper variance. Stability is mean pairwise Spearman rho, which is a rank correlation, not a variance. The "ratios" below are indicative of relative contribution but do not satisfy an exact additive decomposition (model_var + data_var != total_var in general). A proper decomposition would require either element-wise variance with a fully crossed design (R data seeds x R model seeds) or ANOVA on the crossed grid. The current marginal design provides directional evidence but should not be reported as an exact decomposition.

### Expected result

For Single Best, model-selection variance should dominate (the arbitrary hyperparameter choice is the main source of instability). For DASH, both sources should be reduced, with the model-selection component reduced more dramatically because diversity selection directly addresses it.

---

## Experiment 8: First-Mover Visualization

Generates the concentration figure for the paper (Figure 2). Shows per-feature importance within a single correlated group (5 features, each with true importance 0.40) for Single Best, Large Single Model, and DASH (MaxMin). Demonstrates that Single Best and LSM concentrate importance on an arbitrary feature while DASH distributes proportionally.

Run via `run_experiments.py --experiments first_mover_visualization`. Output: `results/figures/first_mover_concentration.{pdf,png}`.

---

## Experiment 9: First-Mover Bias Isolation

Shows how first-mover bias concentration grows with tree count in a single sequential model. Trains models with increasing numbers of trees and measures importance concentration within correlated groups. Confirms the mechanistic prediction: more sequential trees amplify the first-mover advantage.

Run via `run_experiments.py --experiments first_mover_bias`. Output: `results/figures/first_mover_bias_isolation.{pdf,png}`, `results/tables/first_mover_bias.json`.

---

## Experiment 10: Statistical Significance Tests

Applies Wilcoxon signed-rank tests to per-repetition accuracy and equity values from the correlation sweep.

### Test structure

At each rho level, we test:
- DASH vs Single Best (accuracy and equity)
- DASH vs Large Single Model (accuracy and equity)

Plus Table 2 baselines at rho=0.9:
- DASH vs Ensemble SHAP, Stochastic Retrain, DASH Dedup (accuracy and equity)

**Total**: 5 rho x 2 comparisons x 2 metrics + 3 baselines x 2 metrics = **26 tests**.

### Corrections

- **Bonferroni**: Each p-value multiplied by 26. Conservative but transparent.
- **Holm-Bonferroni** (M1 fix): Step-down correction. Less conservative while still controlling family-wise error rate.

### Effect sizes

**Cohen's d** with direction indicators (C3 fix): For accuracy, positive d means DASH is better. For equity (CV), negative d means DASH is better. Magnitude: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

### Power note (C1)

With N_REPS=20, the minimum achievable two-sided Wilcoxon p-value is ~0.00002. After Bonferroni correction (x26), the floor is ~0.0005, comfortably below alpha=0.05.

### Why stability is not tested

Stability is computed as a single number across all repetitions (mean pairwise Spearman rho), not per-repetition. Wilcoxon requires paired per-rep values, so stability cannot be tested this way.

---

## Success Criteria

The notebook checks 11 automated criteria (Section 15):

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | Stability: DASH > Single Best on linear sweep | >= 80% of rho levels |
| 2 | Accuracy: DASH >= Single Best at rho=0.9 | Relative to baseline |
| 3 | Equity: DASH CV < Single Best CV | >= 80% of rho levels |
| 4 | Safety at rho=0: accuracy gap < 0.1 | No degradation |
| 5 | K_eff increases with epsilon | Monotonic |
| 6 | Nonlinear DGP: DASH > SB stability at rho=0.9 | DASH wins |
| 7 | Significance: enough tests significant | >= 50% of tests |
| 8 | Superconductor: DASH stability > SB | DASH wins |
| 9 | California Housing: DASH stability > SB | DASH wins |
| 10 | Breast Cancer: DASH stability > SB | DASH wins |
| 11 | Variance decomposition: DASH model-var < SB model-var | DASH lower |

Criteria 1-4 are the headline claims. Criterion 4 is the safety check. Criteria 5-7 are robustness checks. Criteria 8-10 confirm real-world generalization. Criterion 11 validates the mechanistic explanation.

---

## Methodology Fixes Applied

### v4 audit fixes (A-series)

| ID | Fix | Impact |
|----|-----|--------|
| A1 | Model-selection uncertainty documented | Interpretation |
| A2 | Zero-group equity handling | Minor equity shift |
| A3 | BCa bootstrap for stability CIs | Tighter CIs |
| A4 | Four-way data split (separate explain set) | Removes SHAP/RMSE overlap |
| A5 | DGP agreement rename + circularity caveat | Framing |

### v4-v6 audit fixes (N-series)

| ID | Fix | Impact |
|----|-----|--------|
| B2/N3 | Standardized N_TRIALS_SB=30 everywhere | Consistency |
| B3/N4 | Stochastic Retrain N=K for fair comparison | Fair baseline |
| D2/N7 | Re-fit scaler per rep on real-world data | Removes leakage |
| M1 | Holm-Bonferroni step-down correction | Less conservative tests |
| M2 | Random Selection baseline | Isolates MaxMin value |
| M3 | Single Best (M=200) matched compute | Fair compute comparison |
| M5 | Feature ablation via proxy model | Real-world explanation quality |
| F2 | Relative epsilon mode for real-world datasets | Scale-appropriate filtering |

### v6 review fixes

| ID | Fix | Impact |
|----|-----|--------|
| C1 | ABL_N_REPS -> N_REPS (20) | Comparable stability estimates |
| C4 | Nonlinear true_importance marked approximate | Correct interpretation |
| C5 | Variance decomposition caveat | Honest limitations |
| C7 | Wilcoxon tests for real-world experiments | Statistical rigor |
| D2 | Epsilon sensitivity uses Xexp not Xte | Removes data leakage |
| F1 | Cohen's d effect sizes | Standardized effect magnitude |
| F2 | K_eff tracked for all experiments | Ensemble size reporting |
| F3 | Wall-clock timing table | Compute cost comparison |
| B7 | Breast cancer scaler leakage + ablation | Correct real-world results |
| C8 | Group-level accuracy saturation caveat | gacc (Spearman) saturates at 1.0 with well-separated betas; group_level_mse added as discriminative complement |

---

## Design Decision Tags

The notebook uses short tags to reference specific design decisions:

| Tag | Meaning |
|-----|---------|
| A4 | Separate explain set from train for SHAP |
| B3 | N=K for Stochastic Retrain (fair comparison) |
| C3 | Direction-aware Cohen's d interpretation |
| D2 | Re-split and re-fit scaler per rep (avoid leakage) |
| F2 | Relative epsilon mode for real-world datasets |
| M1 | Holm-Bonferroni step-down correction |
| M2 | Random Selection baseline isolates MaxMin value |
| M3 | Matched compute budget (Single Best M=200) |
| M4 | Seed passed to all baseline .fit() calls |
| M5 | Feature ablation using first selected model as proxy |
| M8 | Bootstrap CI for stability |
| N4 | Per-rep arrays saved for significance tests |
| N11 | Save/restore rcParams for publication figures |
| N13 | Wall-clock timing |

---

## What This Means for the Paper

The correlation sweep fills in the central figure (3 panels: stability, accuracy, equity vs rho). The bar chart at rho=0.9 becomes the summary figure. The IS Plots on real data become showcase figures. The disagreement map demonstrates individual-level uncertainty quantification.

Table 1: Correlation sweep results (8 sweep methods x 5 rho levels).
Table 2: All 10 methods at rho=0.9 (sweep methods + extended baselines).
Table 3: Real-world benchmarks (California, Breast Cancer, Superconductor) with stability, RMSE, ablation, and significance tests.
Table 4: Ablation results (M, K, epsilon, delta sensitivity).

The Large Single Model result is the paper's mechanistic contribution. The variance decomposition provides directional evidence (with stated caveats) that model-selection randomness is the dominant source of instability and that DASH specifically reduces it.

The overlapping structure result goes into Discussion confirming robustness beyond idealized settings. The nonlinear DGP result confirms generalization with a noted caveat about approximate ground truth. The epsilon sensitivity and ablation results demonstrate practical robustness to hyperparameter choices.
