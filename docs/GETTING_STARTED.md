# Getting Started with DASH

## 1. Why Feature Importance Is Hard

Suppose an oncologist asks: which features drive malignancy predictions in a Breast Cancer model? You train an XGBoost model on the 30-feature Breast Cancer dataset (569 patients). Model A (seed=0) reports "worst concave points" as the top driver. You retrain with seed=1. Model B reports "worst perimeter" as the top driver. Both models achieve nearly identical AUC — 0.988 vs 0.989. Which explanation do you report?

This is not a quirk of a bad dataset or a weak model. It happens because many Breast Cancer features measure nearly the same thing: mean radius, mean perimeter, mean area, worst radius, worst perimeter, and worst area all have pairwise correlations above 0.95. They are measuring tumor size from different angles. Any one of them could serve as the "most important" feature, and which one the model selects is partially arbitrary.

**Feature importance** is a summary of how much each input contributes to a model's predictions. **SHAP values** (SHapley Additive exPlanations) are the most principled method for computing these contributions — they are the only method satisfying four natural axioms simultaneously. **Collinearity** (correlated features) breaks the practical usefulness of feature importance: when two features are nearly identical, splitting the credit between them becomes ambiguous.

For a formal treatment of SHAP value definitions and the axioms they satisfy, see `docs/API_REFERENCE.md`.

---

## 2. The First-Mover Mechanism

Any iterative model — gradient boosting, Lasso, neural networks — fits sequentially. Each step reduces residuals left by all previous steps. When two features are highly correlated — say, mean radius and mean perimeter with ρ = 0.97 — whichever one gets used first captures most of the credit. Why? Because once mean radius has been used to reduce residuals, the remaining residuals are *already partially explained* by mean perimeter (since they're nearly the same). Mean perimeter then appears less useful in subsequent iterations, not because it carries less signal, but because the signal was already extracted by its correlated partner.

The feature that gets used first is determined by the model's random seed — a completely arbitrary initialization. Different seeds choose different first-mover features. The first-mover advantage then compounds across iterations: more sequential residual reduction along the first-mover axis → more apparent importance for the first-mover → lower apparent importance for the correlated partner.

This mechanism is formally proved for all iterative optimizers (gradient boosting, Lasso, neural networks) in the companion Attribution Impossibility theorem, with divergence rate 1/(1−ρ²) — at ρ=0.9, the first-mover gets 5.3× more credit.

**This means bigger models make the problem worse.** A single large model with 1,000 trees reinforces the first arbitrary choice through 1,000 rounds of sequential residual reduction. The Large Single Model baseline in the DASH paper is consistently *worse* than the single best small model — not better. See `docs/BENCHMARK_RESULTS.md` for the full comparison.

The consequence is **path-dependent explanation**: the ranking of correlated features depends on which one happened to be chosen first, not on which one carries more causal signal. In Breast Cancer, the true biological drivers are the underlying cellular processes — size, shape, texture — not any particular measurement of them.

---

## 3. What DASH Does

DASH (Diversified Aggregation for Stable Hypotheses) resolves first-mover bias through independence. Instead of training one large model, DASH trains many small models independently, so their arbitrary first-mover choices are *uncorrelated*. When you average their SHAP matrices, the arbitrary noise cancels, and the signal compounds.

The five-stage pipeline:

1. **Train M small models (Stage 1: Population)** — Each model is trained with randomly sampled hyperparameters, with `colsample_bytree` forced to 0.1–0.5. This means each model sees a different subset of features at each split, further breaking the sequential residual dependency. Each model makes its own arbitrary first-mover choice about which correlated feature to "discover" first.

2. **Filter to good-performing models only (Stage 2: Filtering)** — Models whose validation score is more than `epsilon` below the best model are discarded. This ensures you're averaging over high-quality explanations, not noise from underfitting models.

3. **Select K most diverse models (Stage 3: Diversity)** — A MaxMin greedy algorithm selects K models that maximize the minimum pairwise cosine distance between their gain-importance vectors. This ensures the K selected models represent genuinely different "perspectives" on which features matter — not K copies of the same model.

4. **Average SHAP matrices (Stage 4: Consensus)** — Interventional TreeSHAP is computed for each of the K selected models on a held-out reference set (`X_ref`). The K SHAP matrices (each of shape N' × P) are averaged element-wise. The arbitrary noise from first-mover choices cancels; the shared signal — which *group* of features matters — compounds.

5. **Diagnose via FSI and IS plots (Stage 5: Diagnostics)** — The Feature Stability Index (FSI) quantifies how much each feature's attribution varied across the K models. High FSI = the K models disagreed about this feature = likely a collinear cluster member. The Importance-Stability (IS) plot places all features in a two-dimensional space: consensus importance (x-axis) vs. FSI (y-axis). Quadrant II features (high importance, high FSI) are collinear cluster members that should be reported as a group.

Why doesn't **Stochastic Retrain** (same approach, simpler implementation) fully replace DASH? Stochastic Retrain achieves nearly the same stability (~0.977 at ρ = 0.9) with minimal code. DASH's advantages are: (1) the FSI diagnostic identifies *which specific features* are contested, something Stochastic Retrain doesn't provide; (2) DASH distributes credit more equitably across collinear cluster members (within-group CV = 0.175 vs. 0.232 for Single Best); and (3) the IS plot enables principled decision-making about which features to report individually vs. as a group.

---

## 4. How to Know If You Need DASH

Use this checklist before applying DASH:

- **Are your features dependent?** Start with `mi_prescreen(X)` — it detects nonlinear dependencies that `np.corrcoef` misses (e.g., X₂ = X₁² has |ρ|≈0 but MI=1.81). If `n_hidden > 0`, your SHAP rankings are provably unreliable. For a quick linear check, `np.corrcoef(X)` with |r| > 0.7 also works but misses nonlinear cases.
- **Do your explanations need to be reproducible?** If you retrain the model and the feature ranking changes substantially, DASH will stabilize it. If rankings are already stable across seeds, DASH adds overhead without benefit.
- **Do you need to audit which features are collinear proxies vs. robust drivers?** The IS plot and FSI are the only diagnostic tools for this that work without ground truth.

If your features are independent (no correlation above 0.5), DASH adds training overhead with no stability benefit. The single best model will already produce stable explanations.

For the "When to Use DASH" decision guide, see `README.md`.

---

## 5. Theoretical Foundations

DASH is backed by formal impossibility results proved in Lean 4 (zero sorry statements, zero behavioral axioms):

- **The Attribution Trilemma**: No single-model feature ranking can simultaneously be faithful, stable, and complete when features are collinear. This is why single-model SHAP explanations are partially arbitrary — it is mathematically *impossible* for them to be stable.
- **Divergence rate**: The first-mover attribution ratio diverges as 1/(1−ρ²) — at ρ=0.9, the first-mover feature gets 5.3× more credit than its correlated partner.
- **Flip rate = coin flip**: For symmetric correlated features, the probability that a retrained model swaps their ranking is exactly 50%.
- **DASH is optimal**: Equal-weight averaging across M independent models is the minimum-variance unbiased linear estimator (Cauchy-Schwarz / Titu's lemma). The ensemble size formula **M_min = ⌈2.71 · σ²/Δ²⌉** gives the minimum M needed for a 5% flip rate target.
- **The Bilemma** (binary questions): For binary explanation spaces — "is this feature's effect positive or negative?" — the situation is worse: faithful + stable alone is impossible (not just the three-way trilemma). DASH resolves ranking instability via averaging, but sign instability is a harder problem with no single-model fix.

These results are available as practical diagnostics via the **theory bridge extension**:

```python
from dash_shap.extensions.theory_bridge import theory_bridge
tb = theory_bridge(pipe.result_)
print(tb.summary())  # SNR per pair, predicted flip rates, M recommendation
```

For the empirical evidence, see the [DASH paper](https://arxiv.org/abs/2603.22346). For the formal proofs, see the [Lean 4 formalization](https://github.com/DrakeCaraker/dash-impossibility-lean) (315 theorems, 0 sorry statements).

---

## 6. Quick Check (Start Here)

Before diving into the full pipeline, check if your model is affected:

```python
from dash_shap import check
result = check(X, y, feature_names=feature_names)
print(result.report())
```

This trains 25 models, computes SHAP, and tells you which feature rankings are stable vs unstable. If the report shows no unstable pairs, your single-model SHAP is reliable. If it does, read on.

---

## 7. Recommended Learning Path

| Step | Resource | What You'll Learn |
|---|---|---|
| 1 | This document | Concept-first introduction: the problem, the mechanism, how DASH solves it |
| 2 | [Quickstart Notebook](../notebooks/quickstart.ipynb) | 3-minute end-to-end demo on synthetic data: fit, IS plot, FSI summary |
| 3 | [Tutorial 1: The Problem](../notebooks/tutorial_01_the_problem.ipynb) | See SHAP ranking instability on Breast Cancer before DASH is introduced |
| 4 | [Tutorial 2: How DASH Works](../notebooks/tutorial_02_dash_walkthrough.ipynb) | 5-stage walkthrough on Breast Cancer — inspect intermediate outputs at each stage |
| 5 | [Tutorial 3: Reading Results](../notebooks/tutorial_03_interpreting_outputs.ipynb) | IS plot, FSI, local disagreement maps — quadrant action guide with clinical feature names |
| 6 | [Tutorial 4: Parameter Exploration](../notebooks/tutorial_04_simulation.ipynb) | Sweep ρ, M, K, epsilon; understand why the Breast Cancer +0.549 result happens |

---

## 8. Five-Minute Breast Cancer Example

The code below runs DASH end-to-end on the Breast Cancer dataset. It uses tutorial-scale parameters (M=100, K=20) that complete in about 1–2 minutes on a laptop with `n_jobs=-1`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from dash_shap import DASHPipeline

bc = load_breast_cancer()
X, y = bc.data, bc.target
feature_names = list(bc.feature_names)

# Four-way split: train / val / explain / test
# Same split as FAQ.md lines 103-105
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_temp, X_explain, y_temp, _ = train_test_split(X_temp, y_temp, test_size=0.12, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
# Approximate split sizes: X_train ~340, X_val ~86, X_explain ~58, X_test ~85

pipe = DASHPipeline(M=100, K=20, epsilon=0.05, epsilon_mode="relative",
                    task="binary", seed=42, verbose=True)
pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain,
         feature_names=feature_names)

# Summary: top features by consensus importance and their FSI
pipe.get_fsi().summary(top_k=10)

# IS plot: importance vs. stability for all 30 features
pipe.plot_importance_stability()
```

Expected output: the IS plot will show the radius cluster (mean radius, mean perimeter, mean area, worst radius, worst perimeter, worst area) in Quadrant II — high consensus importance, high FSI — because the K models disagree about which size measurement to credit. Features like worst concave points may appear in Quadrant I (high importance, low FSI) if they are individually informative beyond the collinear group.


To check sign stability across the ensemble (do models agree on the *direction* of each feature's effect?):

```python
from dash_shap.core.diagnostics import coverage_conflict

cc = coverage_conflict(pipe.all_shap_matrices_)
for j, name in enumerate(feature_names):
    rate = cc["feature_conflict_rate"][j]
    if rate > 0.1:
        print(f"  {name}: {rate:.0%} of observations have sign disagreement")
```

For detailed interpretation of the IS plot, FSI values, and local disagreement maps, see `notebooks/tutorial_03_interpreting_outputs.ipynb`.
