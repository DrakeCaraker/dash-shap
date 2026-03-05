# DASH: Diversified Aggregation of SHAP

**DASH** is a method for producing stable, fair, and accurate feature importance explanations from XGBoost models, even when features are highly correlated. It works by training a diverse population of independent models and averaging their SHAP (SHapley Additive exPlanations) values into a single consensus explanation.

> Caraker, Arnold, Rhoads (2026)

---

## Table of Contents

- [The Problem](#the-problem)
- [Why Bigger Models Make It Worse](#why-bigger-models-make-it-worse)
- [How DASH Works](#how-dash-works)
- [Beyond XGBoost and SHAP](#beyond-xgboost-and-shap)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Five-Stage Pipeline](#the-five-stage-pipeline)
- [Diagnostics: Importance-Stability Plots and FSI](#diagnostics-importance-stability-plots-and-fsi)
- [Baseline Methods](#baseline-methods)
- [Experiments](#experiments)
- [Benchmark Results](#benchmark-results)
  - [Key Conclusions](#key-conclusions)
  - [Success Criteria](#success-criteria)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## The Problem

Every day, millions of data scientists follow the same workflow: train an XGBoost model, run SHAP, and report that "feature A is most important, feature B is second." That ranking drives real decisions -- it selects features for production systems, generates scientific hypotheses, satisfies regulatory auditors, and explains predictions to stakeholders. The workflow is treated as reliable. It is not.

SHAP values are one of the most popular tools for explaining machine learning predictions. Given a trained model, SHAP tells you how much each feature contributed to a particular prediction. **But SHAP has a hidden fragility when features are correlated** -- and in practice, features are almost always correlated.

Consider a dataset with two highly correlated features, say `height` and `arm_span` (correlation = 0.95). When you train an XGBoost model, the algorithm must choose which feature to split on at each decision node. Since both features carry nearly identical information, the choice is essentially arbitrary -- it depends on small differences in the training data, the random seed, or the hyperparameters. The model's *predictions* are virtually the same regardless of which feature it picks. But the *SHAP values* change dramatically:

- **Model A** happens to split on `height` frequently. SHAP says `height` is the #1 most important feature. `arm_span` gets almost no credit.
- **Model B** (trained with a different random seed) happens to split on `arm_span` frequently. SHAP says `arm_span` is #1. `height` gets almost no credit.

Both models make the same predictions. Both are equally valid. But they tell completely different stories about *why* they make those predictions. This is known as the **Rashomon effect** -- many models fit the data equally well but offer contradictory explanations.

This instability gets worse as correlation increases, and it is not just a theoretical concern. The specific ranking of correlated features depends on which hyperparameters you happened to use, not on a property of the data. Run it again with slightly different settings and features swap positions. The predictions are the same. The explanation is different. Nobody notices because nobody reruns the explanation. In practice:

- A data scientist might conclude that `height` is the key driver and recommend collecting more `height` data, when in reality `arm_span` is equally informative.
- A regulatory audit might flag a model for relying on a sensitive feature, when in fact the model could just as easily have used a non-sensitive proxy.
- A scientific study might report a specific feature as the primary biomarker, when several correlated biomarkers are equally valid.

The ranking is not wrong -- it captures real signal -- but it is partially arbitrary. **The way the applied ML field currently generates feature importance explanations is unreliable, and the intuitive fix (bigger models) makes it worse.**

**DASH solves this problem.** Instead of trusting a single model's arbitrary feature selection, DASH deliberately trains many models that are forced to explore different parts of the feature space, then combines their explanations into a consensus that fairly distributes credit across correlated features.

---

## Why Bigger Models Make It Worse

The natural response to "my explanations are unstable" is "train a more powerful model." We tested that. The result is the most counterintuitive finding of this work: **a single large model with the same total compute as DASH produces the worst explanations of any method tested** -- worse than a simple tuned model, worse than naive averaging, worse on stability, accuracy, and equity simultaneously.

This happens because of a specific mechanism we call **sequential residual dependency**.

In gradient boosting, each tree fits the errors (residuals) left by all previous trees. If tree 1 picks feature A from a correlated pair (A, B), it partially removes A's contribution from the residuals. Tree 2 now sees residuals where A looks less useful -- but B also looks less useful, because B carries the same signal that A already partially captured. The net effect: whichever feature gets picked first gets reinforced across thousands of subsequent trees. More trees means more reinforcement of an arbitrary initial choice.

Low `colsample_bytree` makes this worse, not better. When each tree sees fewer features, the initial selection becomes even more path-dependent -- a tree that happens not to see feature B in its random column sample will default to feature A, further concentrating importance on A.

A single XGBoost model with 10,000-15,000 trees and the same low `colsample_bytree` that DASH uses -- matching DASH's total compute budget in a single sequential model -- amplifies this path dependence to its extreme. In our full benchmark (20 repetitions at ρ=0.95), this Large Single Model configuration achieved:

- **Worst stability** (0.9301 vs. 0.9806 for DASH)
- **Worst accuracy** (0.9641 vs. 0.9901 for DASH)
- **Worst equity** (CV of 0.2708 vs. 0.1618 for DASH)

The ranking that emerges from a large sequential model is not a better version of the truth. It is a more committed version of an arbitrary initial choice.

---

## How DASH Works

The core insight behind DASH is that the instability of SHAP values under collinearity is a property of individual models, not of the data. If you train enough diverse models and average their SHAP values, the arbitrary choices cancel out and the consensus explanation reflects the true underlying importance structure.

DASH achieves this through three key design decisions:

### 1. Forced Feature Restriction

Each model in the DASH population is trained with a low `colsample_bytree` parameter (randomly sampled from 0.1 to 0.5). This means each tree in each model can only see 10-50% of the available features. Different models are forced to rely on different features, even from the same correlated group.

### 2. Model Independence

Unlike a single large XGBoost ensemble (where trees are trained sequentially on residuals), DASH trains each model completely independently from scratch. This is critical because sequential boosting creates a **first-mover bias**: the first tree picks feature `f1` from a correlated group, which modifies the residuals, which makes subsequent trees less likely to pick `f2` from the same group. This path-dependent bias concentrates importance on whichever feature happened to be selected first. Training independent models eliminates this bias entirely.

### 3. Diversity-Aware Selection

Not all models in the population are equally useful. DASH filters for high-performing models (so explanations come from accurate models) and then selects a diverse subset (so the selected models collectively cover different feature usage patterns). This is more effective than simply averaging all models or picking the top performers.

### What DASH Proves

DASH's 0.981 stability, 0.990 accuracy, and 0.162 equity numbers at high collinearity (ρ=0.95, 20 repetitions) are the proof that independence works. But the lasting contribution is the demonstration that explanation instability has a specific mechanistic cause -- sequential path dependence in iterative optimization -- and a specific structural cure -- independence between explained models. Because DASH's independent models make their arbitrary choices independently, averaging cancels the arbitrariness. The consensus reflects what the data supports -- which *group* of correlated features matters -- rather than which individual feature one optimization path happened to favor. This reframes the problem from "SHAP is noisy" to "single-model explanations are fundamentally limited," which is a different and larger claim.

---

## Beyond XGBoost and SHAP

The sequential residual dependency that DASH exposes is not a quirk of XGBoost. It is a property of iterative optimization. Any model where step N+1 depends on the outcome of step N has the potential for the same path-dependent feature utilization:

- **Gradient boosting** (XGBoost, LightGBM, CatBoost): Each tree fits residuals from previous trees. Early feature choices shape all subsequent trees.
- **Neural networks trained by gradient descent**: If early gradient updates latch onto one member of a correlated feature group, the learned representations organize around that choice. Different random initializations produce different first movers.
- **Linear models via coordinate descent**: The order in which coordinates are updated determines which correlated feature absorbs the signal first.

This means the instability problem is potentially not confined to XGBoost and SHAP. Every saliency map, every attention visualization, every feature attribution from any iteratively trained model may inherit the same arbitrary path dependence. Different initializations produce different first movers, which produce different explanations from models with identical performance.

We demonstrated the problem rigorously for one model class (XGBoost) and one explanation method (SHAP). The generalization to other model-explanation pairs is an open question -- but the underlying mechanism (sequential dependence creating path-dependent feature utilization) is shared across iterative optimization methods. The structural cure is the same: explanations derived from independent models, whose arbitrary choices cancel under aggregation.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/DrakeCaraker/dash-shap.git
cd dash-shap

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

**Requirements:** Python >= 3.9

**Dependencies:**
| Package | Version | Purpose |
|---------|---------|---------|
| xgboost | >= 2.0.0 | Gradient boosting models |
| shap | >= 0.44.0 | TreeExplainer for SHAP values |
| scikit-learn | >= 1.4.0 | Data splitting, metrics |
| numpy | >= 1.24.0 | Numerical computation |
| pandas | >= 2.0.0 | Data handling |
| scipy | >= 1.11.0 | Statistical tests, clustering |
| matplotlib | >= 3.8.0 | Plotting |
| seaborn | >= 0.13.0 | Statistical visualization |
| pyyaml | >= 6.0 | Configuration management |
| tqdm | >= 4.65.0 | Progress bars |
| joblib | >= 1.3.0 | Parallel model training |

---

## Quick Start

```python
from dash.core.pipeline import DASHPipeline

# Initialize the pipeline
pipeline = DASHPipeline(
    M=200,                      # Train 200 diverse models
    K=20,                       # Select 20 for consensus
    epsilon=0.02,               # Keep models within 0.02 of best score
    selection_method="maxmin",  # Greedy diversity selection
    task="regression",          # "regression", "binary", or "multiclass"
)

# Fit on your data
pipeline.fit(X_train, y_train, X_val, y_val, X_ref=X_val)

# Get consensus feature importance (mean |SHAP| per feature)
importance = pipeline.global_importance_

# Get feature ranking (most to least important)
ranking = pipeline.get_importance_ranking()

# Plot the Importance-Stability diagram
fig = pipeline.plot_importance_stability()

# Get the Feature Stability Index for each feature
fsi = pipeline.get_fsi()
print(fsi.summary())

# Use the ensemble for predictions too
predictions = pipeline.get_consensus_ensemble_predictions(X_test)
```

---

## The Five-Stage Pipeline

DASH operates as a sequential five-stage pipeline. Each stage has a clear purpose and produces artifacts consumed by the next stage.

### Stage 1: Population Generation

```
Input:  Training data (X_train, y_train), validation data (X_val, y_val)
Output: M trained XGBoost models with validation scores
```

DASH trains `M` XGBoost models (default: 200), each with a randomly sampled hyperparameter configuration. The key hyperparameters and their search ranges are:

| Hyperparameter | Search Range | Purpose |
|---|---|---|
| `max_depth` | 3, 4, 5, 6, 8, 10, 12 | Tree complexity |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1, 0.2, 0.3 | Step size |
| `colsample_bytree` | 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5 | Feature restriction (the critical one) |
| `subsample` | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 | Row sampling |
| `reg_alpha` | 0, 0.01, 0.1, 1.0, 5.0, 10.0 | L1 regularization |
| `reg_lambda` | 0, 0.01, 0.1, 1.0, 5.0, 10.0 | L2 regularization |
| `min_child_weight` | 1, 3, 5, 10, 20 | Minimum leaf weight |

Each model uses early stopping (default: 20 rounds) with up to 1,000 boosting rounds. Training is parallelized across all available CPU cores via `joblib`.

The `colsample_bytree` range of 0.1-0.5 is deliberately low -- it forces each tree to see only a fraction of features, which is the mechanism that creates diversity across models.

### Stage 2: Performance Filtering

```
Input:  M models with validation scores
Output: Filtered subset of models (typically 30-50 from M=200)
```

Not every model in the population performs well -- some random hyperparameter combinations produce poor models. Stage 2 removes these by keeping only models whose validation score is within `epsilon` (default: 0.02) of the best model's score.

For example, if the best model achieves an RMSE of 0.60, all models with RMSE <= 0.63 pass the filter (for RMSE, lower is better; for AUC, higher is better).

### Stage 3: Diversity-Aware Selection

```
Input:  Filtered models with preliminary importance vectors
Output: K selected models (default: 20)
```

From the filtered pool, DASH selects `K` models that are both accurate *and* diverse in how they distribute feature importance. Three selection strategies are available:

**MaxMin Selection (recommended default):**
1. Start with the best-performing model.
2. For each remaining candidate, compute its minimum dissimilarity to all already-selected models (using normalized importance vector dot products).
3. Add the candidate with the highest minimum dissimilarity (the one most different from the current set).
4. Stop when `K` models are selected or the minimum dissimilarity drops below `delta` (default: 0.1).

This is a greedy algorithm that does *not* require knowing the feature correlation structure. It works by ensuring the selected models disagree about which features are important, which naturally covers different members of correlated feature groups.

**Cluster Coverage Selection:**
1. Compute the feature correlation matrix from the training data.
2. Hierarchically cluster features with a distance threshold `tau` (default: 0.3).
3. Select models that collectively cover different representative features from each cluster.

This is more targeted when you know the correlation structure exists, but requires computing the full correlation matrix.

**Deduplication Selection:**
1. Remove models whose importance vectors have Spearman correlation > 0.95 (i.e., near-duplicates).
2. Keep the better-performing model from each duplicate pair.

This is the weakest variant, included as a sanity check. It removes redundancy but does not actively seek diversity.

### Stage 4: Consensus SHAP Aggregation

```
Input:  K selected models, reference data (X_ref)
Output: Consensus SHAP matrix (N' x P), all individual SHAP matrices (K x N' x P)
```

For each of the `K` selected models, DASH computes a full SHAP value matrix using `shap.TreeExplainer` with **interventional** feature perturbation (the recommended approach for tree models with correlated features). Each matrix has shape `(N', P)` where `N'` is the number of reference observations and `P` is the number of features.

The consensus is simply the **element-wise average** across the `K` models:

```
consensus[i, j] = mean(shap_matrices[:, i, j])  over K models
```

This averaging is the core mechanism of DASH. Because different models attribute importance to different members of correlated groups, the average distributes credit more fairly across the group.

### Stage 5: Stability Diagnostics

```
Input:  All K SHAP matrices, consensus matrix
Output: Feature Stability Index (FSI), global importance, variance matrix
```

The final stage computes diagnostic metrics:

- **Global importance**: `I_j = mean(|consensus[:, j]|)` -- the average absolute SHAP value for feature `j` across all reference observations.
- **Variance matrix**: Element-wise variance across the `K` models' SHAP matrices.
- **Feature Stability Index (FSI)**: For each feature `j`:

  ```
  FSI_j = mean_std_j / (mean_abs_consensus_j + epsilon)
  ```

  where `mean_std_j` is the average standard deviation of feature `j`'s SHAP values across models, and `mean_abs_consensus_j` is the average absolute consensus SHAP value. A high FSI means models disagree about this feature's contribution (likely collinear); a low FSI means models agree (robust).

---

## Diagnostics: Importance-Stability Plots and FSI

DASH provides two key diagnostic visualizations that go beyond standard feature importance bar charts.

### The Importance-Stability (IS) Plot

The IS Plot is a 2D scatter plot with **global importance** on the x-axis and **FSI** on the y-axis. Each point is a feature. The plot is divided into four quadrants:

```
                    High FSI
                       |
    IV: Fragile        |    II: Collinear
    Interactions       |    Cluster Members
                       |
    -------------------+-------------------
                       |
    III: Confirmed     |    I: Robust
    Unimportant        |    Drivers
                       |
                    Low FSI
         Low Importance    High Importance
```

- **Quadrant I (Robust Drivers):** High importance, low FSI. These features are consistently important across all models. Their SHAP values are reliable and can be trusted for decision-making.

- **Quadrant II (Collinear Cluster Members):** High importance, high FSI. These features are important but models disagree about *which one* in a correlated group deserves the credit. The high FSI flags them as interchangeable. A practitioner should treat the entire correlated group as collectively important, not single out individual features.

- **Quadrant III (Confirmed Unimportant):** Low importance, low FSI. All models agree these features don't matter. Safe to ignore.

- **Quadrant IV (Fragile Interactions):** Low importance, high FSI. These features have small average importance but high variability. They may participate in weak interaction effects that only appear in some model configurations.

### The Local Disagreement Map

For a single observation, the local disagreement map shows the consensus SHAP value for each feature along with error bars representing +/- 1 standard deviation across the `K` models. This reveals:

- **Narrow error bars:** Models agree on this feature's contribution to this prediction. The explanation is reliable.
- **Wide error bars:** Models disagree. This feature's contribution is model-specification-dependent and should be interpreted with caution.

This is especially useful in high-stakes applications (e.g., medical diagnosis, loan decisions) where you need to know not just *what* the model says, but *how confident the explanation is*.

---

## Baseline Methods

DASH is compared against five baseline methods, each designed to test a specific hypothesis about why DASH works.

### 1. Single Best Model (`SingleBestBaseline`)

**What it does:** Trains many random hyperparameter configurations, picks the single best model by validation score, and computes its SHAP values.

**What it represents:** Standard practice. This is what most practitioners do today.

**Why it fails under collinearity:** A single model makes one arbitrary choice per correlated group. Run it again with a different seed, and it may choose differently, producing a completely different importance ranking.

### 2. Large Single Model (`LargeSingleModelBaseline`)

**What it does:** Trains one XGBoost model with a very large number of trees (default: K x 500 = 10,000 trees) and low `colsample_bytree` (0.2), matching the total compute of DASH.

**What it tests:** *Does model independence matter, or is low `colsample_bytree` alone sufficient?*

This baseline uses the same low feature restriction as DASH but trains everything within a single sequential boosting ensemble. The key difference is the **sequential residual dependency**: each tree is trained on residuals from previous trees, creating a path-dependent first-mover bias where whichever feature is selected first in a correlated group tends to dominate throughout the entire ensemble.

**Expected result:** DASH should outperform this baseline, especially at high correlation. The gap directly measures the value of training independent models (breaking first-mover bias) versus just using low `colsample_bytree`.

### 3. Ensemble SHAP (`EnsembleSHAPBaseline`)

**What it does:** Trains one large XGBoost ensemble (2,000 trees) with *standard* `colsample_bytree` (0.8) and computes SHAP values.

**What it tests:** The standard approach from Paillard et al. (2025) -- explain a single well-tuned ensemble.

**Why it fails:** With `colsample_bytree=0.8`, each tree sees 80% of features, so the model has little incentive to explore alternative correlated features. The first-mover bias is even stronger.

### 4. Naive Top-N (`NaiveAveragingBaseline`)

**What it does:** Takes the top-N models from the DASH population (by validation score) and averages their SHAP values *without* diversity selection.

**What it tests:** *Does the diversity selection step (Stage 3) matter, or is simple averaging sufficient?*

**Expected result:** Naive averaging performs reasonably well (averaging helps), but DASH with diversity selection does better because it actively ensures the selected models cover different feature usage patterns. Without diversity selection, the top models may all be similar (since they all perform well with similar configurations).

### 5. Stochastic Retrain (`StochasticRetrainBaseline`)

**What it does:** Finds the best hyperparameter configuration, then retrains N models with that exact configuration but different random seeds.

**What it tests:** *Does deliberate hyperparameter diversity matter, or does natural stochasticity (different seeds) provide enough diversity?*

**Expected result:** Stochastic retrain captures some variability (different seeds produce slightly different models), but the diversity is much smaller than DASH's forced hyperparameter diversity. All N models use the same `colsample_bytree`, `max_depth`, etc., so they tend to make similar feature selections.

---

## Experiments

The repository includes a comprehensive experimental validation framework. Experiments can be run via:

```bash
python run_experiments.py
```

Or interactively via the demo notebooks:

- **`notebooks/demo_benchmark_1.ipynb`** -- Prototype run (M=50, K=15, 5 reps). Runs in minutes. Use for quick validation and development iteration.
- **`notebooks/demo_benchmark_2.ipynb`** -- Full run (M=500, K=30, 20 reps). Runs in ~1-2 hours. Produces the authoritative results cited throughout this README.

### Experiment 1: Synthetic Linear -- Correlation Sweep

The central experiment. Tests DASH across five levels of feature correlation.

**Data generation:**
- 5,000 observations, 50 features organized into 10 groups of 5
- Within each group, features have pairwise correlation `rho`
- The target variable is a linear combination of group means: `y = sum(z_g * beta_g) + noise`
- Ground-truth betas: `[2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0]`
- True per-feature importance: `|beta_g| / group_size` (within each group, all features are equally important)

**Sweep:** `rho` in {0.0, 0.5, 0.7, 0.9, 0.95}, with 5 repetitions at each level.

**Metrics:**
- **Stability:** Mean pairwise Spearman correlation of importance vectors across the 5 repetitions. Measures: "If I run this method twice, do I get the same ranking?"
- **Accuracy:** Spearman correlation between estimated importance and ground truth. Measures: "Is the ranking correct?"
- **Within-group equity:** Average coefficient of variation (std/mean) of importance values within each correlated group. Measures: "Do correlated features get similar importance, as they should?"

### Experiment 2: Overlapping Correlation Structure

Tests robustness when the correlation structure is not clean block-diagonal. Uses overlapping groups where features A and B are correlated, B and C are correlated, but A and C are not. This "chain" structure is common in real-world data.

### Experiment 3: Nonlinear DGP

Tests DASH on a nonlinear data-generating process:
```
y = z1^2 + 0.8 * z1 * z2 + 1.2 * sin(pi * z3) + linear terms + noise
```
where `z_g` is the mean of group `g`'s features. Accuracy against ground truth is not measured here (because Sobol indices and SHAP distribute nonlinear variance differently), so the focus is on stability and equity.

### Experiment 4: Real Data

Validates DASH on two real datasets:

- **California Housing** (8 features, regression): Natural collinearity between median income and house value, number of rooms and bedrooms, latitude and longitude.
- **Breast Cancer Wisconsin** (30 features, binary classification): Heavy collinearity -- radius, perimeter, and area are mathematically related (perimeter ~ 2*pi*radius, area ~ pi*radius^2). Mean, standard error, and worst-case versions of each measurement create ~21 feature pairs with |r| > 0.9.

---

## Benchmark Results

The repository includes two benchmark notebooks that validate DASH at different scales. The prototype run confirms the method works directionally; the full run provides the authoritative evidence. All numbers cited in narrative sections of this README come from the full run.

### Two Benchmark Configurations

| Parameter | Prototype | Full Run |
|-----------|-----------|----------|
| Population size (M) | 50 | 500 |
| Selected models (K) | 15 | 30 |
| Repetitions (N_REPS) | 5 | 20 |
| Notebook | `demo_benchmark_1.ipynb` | `demo_benchmark_2.ipynb` |
| Runtime | Minutes | ~1-2 hours |

Both notebooks run the same experiments: a proof of concept at high collinearity, a full baseline comparison, a stability test across repetitions, a correlation sweep from ρ=0.0 to ρ=0.95, and a real-data validation on Breast Cancer. The full run uses 10x the population, 2x the selected models, and 4x the repetitions, producing tighter confidence intervals and more decisive results.

### Baseline Comparison at ρ=0.95

The most demanding test: 50 features in 10 correlated groups at ρ=0.95, averaged across 20 repetitions. Each method is run independently from scratch at each repetition, and stability measures the mean pairwise Spearman correlation of importance rankings across runs.

```
Method                Stability   Accuracy (ρ)    Equity (CV)
=============================================================
Single Best              0.9527         0.9757         0.2411
Large Single Model       0.9301         0.9641         0.2708
DASH (MaxMin)            0.9806         0.9901         0.1618
```

DASH (MaxMin) leads on all three metrics at the highest collinearity level:

- **Stability:** 0.9806 vs. 0.9527 for Single Best (+0.0279). DASH produces nearly identical importance rankings across 20 independent runs. Single Best's ranking shifts substantially depending on which arbitrary feature choices a given seed produces.

- **Accuracy:** 0.9901 vs. 0.9757 for Single Best. DASH's consensus ranking is closer to the known ground truth because averaging cancels the arbitrary feature-selection noise that biases any single model's ranking.

- **Equity:** CV of 0.1618 vs. 0.2411 for Single Best (33% lower) and 0.2708 for Large Single Model (40% lower). Within each correlated group, DASH distributes importance fairly across all members rather than concentrating it on whichever feature one model happened to grab.

The Large Single Model -- which matches DASH's total compute budget in a single sequential ensemble -- performs worst on every metric. This is the direct evidence for sequential residual dependency: more trees in a single ensemble amplifies the first-mover bias rather than correcting it.

### The Correlation Sweep

The central result: how each method performs as correlation increases from 0.0 to 0.95 (full run, 20 repetitions per level).

**Stability:**
```
ρ=0.0:    DASH=0.9778   SB=0.9744   LSM=0.9650   (comparable -- safety check)
ρ=0.5:    DASH=0.9815   SB=0.9781   LSM=0.9702   (small advantage)
ρ=0.7:    DASH=0.9781   SB=0.9699   LSM=0.9635   (advantage emerges)
ρ=0.9:    DASH=0.9810   SB=0.9603   LSM=0.9396   (clear separation)
ρ=0.95:   DASH=0.9806   SB=0.9527   LSM=0.9301   (largest gap)
```

**Equity (within-group CV, lower is better):**
```
ρ=0.0:    DASH=0.1511   SB=0.1539   LSM=0.1532   (comparable)
ρ=0.5:    DASH=0.1583   SB=0.1667   LSM=0.1799   (advantage begins)
ρ=0.7:    DASH=0.1719   SB=0.1980   LSM=0.2021   (growing gap)
ρ=0.9:    DASH=0.1655   SB=0.2182   LSM=0.2539   (24% better than SB)
ρ=0.95:   DASH=0.1618   SB=0.2411   LSM=0.2708   (33% better than SB)
```

DASH stability is effectively flat across all correlation levels (0.9778-0.9815), while Single Best degrades from 0.9744 to 0.9527 and Large Single Model degrades from 0.9650 to 0.9301. DASH is immune to the correlation-induced instability that plagues single-model explanations because its independent models make their arbitrary choices independently, and averaging cancels the arbitrariness.

### Key Conclusions

1. **DASH's advantage is specifically about collinearity.** The stability gap widens from +0.0034 at ρ=0.0 to +0.0279 at ρ=0.95. At zero correlation, all methods perform similarly -- DASH is a targeted fix, not a blunt hammer.

2. **Bigger models make explanations worse, not better.** The Large Single Model -- matching DASH's total compute in a single sequential ensemble -- achieves the worst stability (0.9301), worst accuracy (0.9641), and worst equity (0.2708) of any method at ρ=0.95. Model independence, not model size, is what matters.

3. **DASH distributes credit fairly across correlated features.** At ρ=0.95, DASH's within-group CV of 0.1618 is 33% lower than Single Best (0.2411) and 40% lower than Large Single Model (0.2708). Where single models arbitrarily concentrate importance on one member of a correlated group, DASH's consensus reflects the group's collective contribution.

4. **DASH is safe when collinearity is absent.** At ρ=0.0, the accuracy gap between DASH and Single Best is 0.0019 -- effectively zero. DASH does not degrade performance on uncorrelated data.

5. **Results strengthen with statistical power.** The prototype (5 reps) shows 4/5 stability wins and 4/5 equity wins across correlation levels. The full run (20 reps) shows 5/5 on both. The directional findings are consistent; the full run eliminates sampling noise.

### Success Criteria

The benchmark defines four formal success criteria. Both configurations pass all four, with the full run passing more decisively:

| Criterion | Prototype (5 reps) | Full Run (20 reps) | Threshold |
|-----------|--------------------|--------------------|-----------|
| Stability wins (DASH > Single Best) | 4/5 ρ levels | **5/5** ρ levels | >= 80% |
| Accuracy at ρ=0.9 | 0.9824 | **0.9901** | >= 0.90 |
| Equity wins (DASH CV < Single Best CV) | 4/5 ρ levels | **5/5** ρ levels | Most |
| Safety at ρ=0 (accuracy gap) | 0.0006 | 0.0019 | < 0.1 |

The prototype is a reliable proxy for the full run: both reach the same qualitative conclusions and pass the same criteria. The full run's improvements are quantitative refinements (tighter CIs, perfect 5/5 instead of 4/5), not qualitative reversals.

### Breast Cancer Real-Data Results

The Breast Cancer dataset is a natural showcase for DASH because it contains 30 features with 21 pairs having |r| > 0.9. Features like `mean radius`, `mean perimeter`, and `mean area` are mathematically related and nearly interchangeable.

**DASH pipeline results:**
- 220 of 500 models pass the performance filter (AUC within 0.02 of best), showing that the Breast Cancer classification task is well-handled by most configurations.
- 30 models selected by MaxMin diversity selection.
- Top 5 features identified by consensus importance.

**The IS Plot reveals the correlation structure unsupervised:**
- Features like `worst concave points` and `worst perimeter` appear as **Robust Drivers** (Quadrant I) -- high importance, low FSI, consistently important across all models.
- Features like `mean radius` and `mean perimeter` appear as **Collinear Cluster Members** (Quadrant II) -- high importance but high FSI, because different models attribute importance to different members of this correlated trio.
- Many of the "SE" (standard error) features appear as **Confirmed Unimportant** (Quadrant III).

**The Local Disagreement Map** for a high-variance patient shows which feature attributions are trustworthy (narrow error bars, e.g., texture and concavity features) and which are model-dependent (wide error bars, e.g., radius vs. perimeter). In a clinical setting, this tells the physician which parts of the explanation are reliable versus uncertain.

---

## API Reference

### `DASHPipeline`

The main entry point for using DASH.

```python
from dash.core.pipeline import DASHPipeline

pipeline = DASHPipeline(
    M=200,                              # Number of models in the population
    K=20,                               # Number of models to select for consensus
    epsilon=0.02,                       # Performance filter threshold
    selection_method="maxmin",          # "maxmin", "cluster", or "dedup"
    delta=0.1,                          # Diversity threshold (maxmin)
    tau=0.3,                            # Cluster distance threshold (cluster)
    task="regression",                  # "regression", "binary", "multiclass"
    search_space=None,                  # Custom hyperparameter search space (dict)
    preliminary_importance_method="gain",  # "gain" or "shap_subsample"
    background_size=100,                # SHAP background data size
    n_jobs=-1,                          # Parallel jobs (-1 = all cores)
    seed=42,                            # Random seed
    verbose=True,                       # Print progress
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X_train, y_train, X_val, y_val, X_ref=None, feature_names=None)` | self | Runs all 5 stages. `X_ref` defaults to `X_val`. |
| `get_fsi()` | `FeatureStabilityIndex` | Feature Stability Index object with quadrant labels. |
| `plot_importance_stability(groups=None, **kwargs)` | matplotlib Figure | Generates the IS Plot. Pass `groups` to color by feature group. |
| `get_importance_ranking()` | np.array | Feature indices sorted by descending importance. |
| `get_consensus_ensemble_predictions(X)` | np.array | Mean predictions across selected models. |

**Attributes (available after `fit`):**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `models_` | dict | All M trained models |
| `val_scores_` | dict | Validation scores for all M models |
| `filtered_indices_` | list | Indices passing performance filter |
| `selected_indices_` | list | Indices selected for consensus |
| `consensus_matrix_` | (N', P) | Consensus SHAP values |
| `all_shap_matrices_` | (K, N', P) | Individual SHAP matrices |
| `global_importance_` | (P,) | Mean absolute consensus SHAP per feature |
| `fsi_` | (P,) | Feature Stability Index values |
| `variance_matrix_` | (N', P) | Cross-model variance of SHAP values |
| `timing_` | dict | Execution time per stage |

### `FeatureStabilityIndex`

```python
fsi_obj = pipeline.get_fsi()

# Get quadrant labels for each feature
labels = fsi_obj.get_quadrant_labels(
    importance_threshold=None,  # Default: median importance
    fsi_threshold=None,         # Default: median FSI
)

# Print summary of top features
print(fsi_obj.summary(top_k=10))
```

### `ImportanceStabilityPlot`

```python
from dash.core.diagnostics import ImportanceStabilityPlot

fig = ImportanceStabilityPlot.plot(
    global_importance,              # Array of importance values
    fsi,                            # Array of FSI values
    feature_names=None,             # Optional feature names
    groups=None,                    # Optional group assignments (for coloring)
    importance_threshold=None,      # Quadrant threshold (default: median)
    fsi_threshold=None,             # Quadrant threshold (default: median)
    title="Importance-Stability Plot",
    figsize=(10, 7),
    annotate_top_k=5,               # Label the top-k most important features
)
```

### `local_disagreement_map`

```python
from dash.core.diagnostics import local_disagreement_map

fig = local_disagreement_map(
    all_shap_matrices,    # (K, N', P) array
    observation_idx=0,    # Which observation to explain
    feature_names=None,   # Optional feature names
    top_k=15,             # Number of features to show
    figsize=(10, 6),
)
```

### Synthetic Data Generators

```python
from dash.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear

# Linear DGP with controllable correlation
X_train, y_train, X_val, y_val, X_test, y_test, groups, true_importance, meta = \
    generate_synthetic_linear(
        N=5000,           # Total observations
        P=50,             # Number of features
        group_size=5,     # Features per correlated group
        rho=0.9,          # Within-group correlation
        sigma_noise=0.5,  # Noise standard deviation
        seed=42,
        test_size=0.15,
        val_size=0.15,
        structure="block", # "block" or "overlapping"
    )

# Nonlinear DGP (same interface, adds quadratic/interaction/sin terms)
X_train, y_train, X_val, y_val, X_test, y_test, groups, true_importance, meta = \
    generate_synthetic_nonlinear(N=5000, P=50, group_size=5, rho=0.9)
```

### Evaluation Metrics

```python
from dash.evaluation import importance_accuracy, importance_stability, within_group_equity

# Accuracy vs. ground truth
spearman_rho, mse = importance_accuracy(estimated_importance, true_importance)

# Stability across repetitions
stability = importance_stability([importance_run1, importance_run2, ...])

# Within-group equity
mean_cv = within_group_equity(importance_vector, group_assignments)
```

---

## Project Structure

```
dash-shap/
├── dash/                              # Main Python package
│   ├── __init__.py                    # Package init with lazy imports
│   ├── core/                          # Core DASH pipeline
│   │   ├── pipeline.py                # DASHPipeline: end-to-end orchestrator
│   │   ├── population.py              # Stage 1: model population generation
│   │   ├── filtering.py               # Stage 2: performance filtering
│   │   ├── diversity.py               # Stage 3: diversity-aware selection
│   │   ├── consensus.py               # Stage 4: consensus SHAP aggregation
│   │   └── diagnostics.py             # Stage 5: FSI, IS plots, disagreement maps
│   ├── baselines/                     # Comparison methods
│   │   ├── single_best.py             # Standard practice baseline
│   │   ├── large_single.py            # Sequential dependency test
│   │   ├── ensemble_shap.py           # Single large ensemble
│   │   ├── naive_averaging.py         # Top-N without diversity
│   │   └── stochastic_retrain.py      # Same config, different seeds
│   ├── experiments/
│   │   └── synthetic.py               # Linear & nonlinear data generators
│   ├── evaluation/
│   │   └── __init__.py                # Metrics: accuracy, stability, equity
│   └── utils/
│       ├── __init__.py                # Utils package init
│       ├── io.py                      # I/O utilities
│       └── shap_helpers.py            # SHAP computation helpers
├── notebooks/                         # Interactive demo notebooks
│   ├── demo_benchmark.ipynb           # Interactive demo notebook
│   ├── demo_benchmark_1.ipynb         # Prototype benchmark (M=50, K=15, 5 reps)
│   └── demo_benchmark_2.ipynb         # Full benchmark (M=500, K=30, 20 reps)
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_evaluation.py             # Evaluation metrics tests
│   ├── test_pipeline.py               # Pipeline integration tests
│   └── test_synthetic.py              # Synthetic data generator tests
├── run_experiments.py                 # Full experiment runner (all 4 experiments)
├── EXPERIMENT_GUIDE.md                # Detailed experimental design documentation
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package metadata
└── setup_dash.sh                      # Setup script
```

---

## Requirements

- **Python** >= 3.9
- **OS:** Linux, macOS, or Windows
- **Hardware:** No GPU required. Parallel model training benefits from multiple CPU cores.

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```
