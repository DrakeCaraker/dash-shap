# DASH: Diversified Aggregation of SHAP

Machine learning models can make predictions, but they can also tell you *why* -- which factors mattered most. That explanation should be consistent: if you build the model a second time on the same data and get equally good predictions, the explanation should be the same. But it isn't. Small, meaningless changes in setup -- like the arbitrary starting point of a randomized algorithm -- can completely change which factors the model calls "most important," even when the predictions don't change at all. This is especially bad when your data contains related measurements that carry overlapping information. DASH fixes this by combining explanations from many independently built models into a single stable consensus.

> Caraker, Arnold, Rhoads (2026)

---

## Table of Contents

- [The Explanation Problem](#the-explanation-problem)
- [Where This Matters](#where-this-matters)
- [How DASH Fixes It](#how-dash-fixes-it)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Problem in Detail](#the-problem-in-detail)
- [Why Bigger Models Make It Worse](#why-bigger-models-make-it-worse)
- [How DASH Works -- Technical Details](#how-dash-works----technical-details)
- [Beyond XGBoost and SHAP](#beyond-xgboost-and-shap)
- [The Five-Stage Pipeline](#the-five-stage-pipeline)
- [Diagnostics: Importance-Stability Plots and FSI](#diagnostics-importance-stability-plots-and-fsi)
- [Baseline Methods](#baseline-methods)
- [Experiments](#experiments)
- [Key Results](#key-results)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Research Roadmap](#research-roadmap)

---

## The Explanation Problem

Imagine a hospital builds a model to predict whether a breast tumor is malignant. The model uses 30 measurements taken from a biopsy -- things like the tumor's radius, perimeter, and area. These measurements are closely related to each other (if you know the radius, you can roughly calculate the perimeter and area). The model makes good predictions. But when you ask it *which measurements mattered most*, something strange happens:

- **Monday's model** says: *"The tumor's radius was the biggest factor in this prediction."*
- **Tuesday's model** -- built on the same data, with the same method, differing only in an arbitrary setup choice (a "random seed" that controls tie-breaking inside the algorithm) -- says: *"Actually, the tumor's perimeter was the biggest factor."*

Both models are equally accurate. Both identify the right group of measurements. But they give a pathologist different answers about which specific measurement to focus on. A researcher reading Monday's explanation might publish that radius is "the" key biomarker, when in reality perimeter and area carry the same information and are equally valid.

This isn't a hypothetical. We tested this on the real Wisconsin Breast Cancer dataset (30 measurements, 21 pairs that are highly correlated with each other). The standard approach produced explanations that changed dramatically between runs -- a consistency score of just 0.534 out of 1.0. DASH raised that to 0.933, nearly doubling the reliability of the explanation.

The core issue is simple: when two measurements carry the same information, the model has to pick one to give credit to. Which one it picks is essentially a coin flip. But the explanation treats that coin flip as if it were a meaningful finding. **DASH makes this problem go away.**

---

## Where This Matters

Unstable explanations aren't just an academic concern. In any domain where decisions depend on *why* a model made a prediction -- not just *what* it predicted -- this instability has real consequences.

**Medical research and clinical decision support.** Predictive models increasingly guide treatment decisions and drug development. Lab tests like ALT and AST both measure liver damage. Systolic and diastolic blood pressure naturally move together. If a clinical decision support tool tells one physician that ALT is the primary risk factor and tells another that AST is, clinicians lose trust in the system -- or worse, pursue the wrong intervention. Regulatory bodies like the FDA increasingly require explanations for AI-based medical devices. Those explanations need to be reproducible.

**Lending and credit decisions.** In the US, lenders are legally required to explain why a loan application was denied (Equal Credit Opportunity Act adverse action notices). If a model uses related inputs like income, debt-to-income ratio, and credit utilization, the explanation might cite "insufficient income" one time and "high credit utilization" another -- for the same applicant, with the same data. The applicant receives different advice depending on which version of the model generated their notice, and a regulator auditing the system sees inconsistent reasoning.

**Hiring and talent analytics.** Companies increasingly use machine learning to screen job candidates. Inputs like years of experience, number of past roles, and tenure at previous jobs are related to each other. If the model's explanation says "years of experience" drove a rejection one time but "short tenure at previous jobs" another, the company faces fairness concerns and legal exposure -- especially when one of those inputs correlates with a protected characteristic like age.

In all of these cases, the model's predictions are fine. It's the *explanation* that's unreliable. And in regulated, high-stakes, or ethically sensitive domains, an unreliable explanation can be worse than no explanation at all.

---

## How DASH Fixes It

Instead of asking one model for an explanation, DASH builds hundreds of small models, each one independently from scratch. Each model is deliberately forced to look at a different subset of the available measurements -- so if radius and perimeter are related, some models rely on radius while others rely on perimeter.

Each model still makes its own arbitrary choice about which related measurement to give credit to. But because the models are built independently, they make *different* arbitrary choices. The noise points in different directions.

DASH averages all their explanations. The arbitrary noise cancels out. What remains is the signal: which *group* of related measurements actually matters, with credit distributed fairly across the group. Instead of "radius is #1 and perimeter doesn't matter," you get "radius *and* perimeter are both important -- they're part of the same underlying signal."

*That's the intuition. The sections below cover installation, usage, and the full technical details.*

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
    epsilon=0.08,               # Keep models within 0.08 of best score
    selection_method="maxmin",  # Greedy diversity selection
    task="regression",          # "regression", "binary", or "multiclass"
)

# Fit on your data (use a held-out explain set as X_ref, separate from X_test)
pipeline.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

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

## The Problem in Detail

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

The natural response to "my explanations are unstable" is "train a more powerful model." We tested that. The result is the most counterintuitive finding of this work: **a single large model with the same total compute as DASH produces the worst explanations of any method tested** -- worse than a simple tuned model, worse than naive averaging, worse on stability, DGP agreement, and equity simultaneously.

This happens because of a specific mechanism we call **sequential residual dependency**.

In gradient boosting, each tree fits the errors (residuals) left by all previous trees. If tree 1 picks feature A from a correlated pair (A, B), it partially removes A's contribution from the residuals. Tree 2 now sees residuals where A looks less useful -- but B also looks less useful, because B carries the same signal that A already partially captured. The net effect: whichever feature gets picked first gets reinforced across thousands of subsequent trees. More trees means more reinforcement of an arbitrary initial choice.

Low `colsample_bytree` makes this worse, not better. When each tree sees fewer features, the initial selection becomes even more path-dependent -- a tree that happens not to see feature B in its random column sample will default to feature A, further concentrating importance on A.

A single XGBoost model with 10,000-15,000 trees and the same low `colsample_bytree` that DASH uses -- matching DASH's total compute budget in a single sequential model -- amplifies this path dependence to its extreme. In our full benchmark (20 repetitions at ρ=0.95), this Large Single Model configuration achieved:

- **Worst stability** (0.9301 vs. 0.9819 for DASH)
- **Worst DGP agreement** (0.9641 vs. 0.9907 for DASH)
- **Worst equity** (CV of 0.2708 vs. 0.1585 for DASH)

The ranking that emerges from a large sequential model is not a better version of the truth. It is a more committed version of an arbitrary initial choice.

---

## How DASH Works -- Technical Details

The core insight behind DASH is that the instability of SHAP values under collinearity is a property of individual models, not of the data. If you train enough diverse models and average their SHAP values, the arbitrary choices cancel out and the consensus explanation reflects the true underlying importance structure.

DASH achieves this through three key design decisions:

### 1. Forced Feature Restriction

Each model in the DASH population is trained with a low `colsample_bytree` parameter (randomly sampled from 0.1 to 0.5). This means each tree in each model can only see 10-50% of the available features. Different models are forced to rely on different features, even from the same correlated group.

### 2. Model Independence

Unlike a single large XGBoost ensemble (where trees are trained sequentially on residuals), DASH trains each model completely independently from scratch. This is critical because sequential boosting creates a **first-mover bias**: the first tree picks feature `f1` from a correlated group, which modifies the residuals, which makes subsequent trees less likely to pick `f2` from the same group. This path-dependent bias concentrates importance on whichever feature happened to be selected first. Training independent models eliminates this bias entirely.

### 3. Diversity-Aware Selection

Not all models in the population are equally useful. DASH filters for high-performing models (so explanations come from accurate models) and then selects a diverse subset (so the selected models collectively cover different feature usage patterns). This is more effective than simply averaging all models or picking the top performers.

### What DASH Shows

DASH's 0.982 stability, 0.991 DGP agreement, and 0.159 equity numbers at high collinearity (ρ=0.95, 20 repetitions) demonstrate that independence helps. The lasting contribution is the identification that explanation instability has a specific mechanistic cause -- sequential path dependence in iterative optimization -- and a structural mitigation -- independence between explained models. Because DASH's independent models make their arbitrary choices independently, averaging cancels the arbitrariness. The consensus reflects what the data supports -- which *group* of correlated features matters -- rather than which individual feature one optimization path happened to favor. This reframes the problem from "SHAP is noisy" to "single-model explanations are fundamentally limited."

**Important caveats:** (1) The simplest form of independence -- Stochastic Retrain (same hyperparameters, different seeds) -- achieves stability within 0.001 of DASH at ρ=0.9, and the difference is not statistically significant. DASH's marginal value over seed averaging lies in its diagnostics (FSI, IS Plot) and equity improvements rather than raw stability. (2) The "DGP agreement" metric (formerly "accuracy") uses a ground-truth definition (uniform within-group importance) that presupposes equitable credit distribution, making it partially circular with the equity metric. It is reported as a sanity check, not the primary evaluation criterion. (3) Under nonlinear DGPs, DASH shows marginally lower stability than Single Best at low correlation (ρ ≤ 0.5), so it should be applied when moderate-to-high correlation is suspected. (4) Stability confidence intervals use bias-corrected and accelerated (BCa) bootstrap, which corrects for both bias and skewness in the bootstrap distribution. (5) Synthetic experiments use a four-way data split (train / val / explain / test) so that the SHAP reference set (`X_explain`) is separate from the RMSE evaluation set (`X_test`).

---

## Beyond XGBoost and SHAP

The sequential residual dependency that DASH exposes is not a quirk of XGBoost. It is a property of iterative optimization. Any model where step N+1 depends on the outcome of step N has the potential for the same path-dependent feature utilization:

- **Gradient boosting** (XGBoost, LightGBM, CatBoost): Each tree fits residuals from previous trees. Early feature choices shape all subsequent trees.
- **Neural networks trained by gradient descent**: If early gradient updates latch onto one member of a correlated feature group, the learned representations organize around that choice. Different random initializations produce different first movers.
- **Linear models via coordinate descent**: The order in which coordinates are updated determines which correlated feature absorbs the signal first.

This means the instability problem is potentially not confined to XGBoost and SHAP. Every saliency map, every attention visualization, every feature attribution from any iteratively trained model may inherit the same arbitrary path dependence. Different initializations produce different first movers, which produce different explanations from models with identical performance.

We demonstrated the problem rigorously for one model class (XGBoost) and one explanation method (SHAP). The generalization to other model-explanation pairs is an open question -- but the underlying mechanism (sequential dependence creating path-dependent feature utilization) is shared across iterative optimization methods. The structural cure is the same: explanations derived from independent models, whose arbitrary choices cancel under aggregation.

This implementation focuses on gradient-boosted decision tree models (XGBoost) and SHAP explanations, where the sequential residual dependency mechanism is well-characterized and empirically validated. However, the independence principle -- aggregate explanations from independently trained models to cancel path-dependent noise -- generalizes to any model class with iterative optimization. Deep neural networks are next: Paper 4 of the research roadmap (see [Research Roadmap](ROADMAP.md)) extends DASH to MLPs and integrated gradients, testing whether the same first-mover dynamics appear in neural network training and whether the same independence-based fix resolves them.

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

Not every model in the population performs well -- some random hyperparameter combinations produce poor models. Stage 2 removes these by keeping only models whose validation score is within `epsilon` (default: 0.08) of the best model's score.

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

Alternate aggregation strategies were considered but not tested in this work to keep scope manageable. These include weighted averaging (weighting each model's SHAP matrix by its validation performance or diversity score), median aggregation (more robust to outlier models), trimmed means (discarding extreme SHAP values before averaging), and rank-based fusion such as Borda count (averaging feature *rankings* rather than raw SHAP values). Element-wise mean was chosen for simplicity and interpretability -- it preserves the additive SHAP property that values sum to the model's prediction. Exploring these alternatives is future work.

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

DASH is benchmarked against 9 baseline methods spanning single models, ensembles, and resampling strategies. Each baseline isolates a specific hypothesis about why DASH works (e.g., "Is it the independence? The low colsample_bytree? The diversity selection?").

Key baselines: **Single Best** (standard practice), **Large Single Model** (matched compute in one sequential ensemble -- tests whether independence matters), **Stochastic Retrain** (same config, different seeds -- tests whether deliberate diversity matters), **Random Selection** (random instead of MaxMin -- isolates diversity selection value).

See [Experiment Guide: The 10 Methods](EXPERIMENT_GUIDE.md#the-10-methods-and-what-each-tests) for full descriptions of all methods and what each tests.

---

## Experiments

The repository includes 8 experiments across synthetic and real-world datasets (10 methods, 20 repetitions per condition, 11 success criteria). Experiments can be run via `python run_experiments.py` or interactively through the demo notebooks:

- **`notebooks/demo_benchmark_6.ipynb`** -- **Authoritative run** (M=200, K=30, 20 reps, checkpointed). Supersedes all prior notebooks.
- **`notebooks/demo_benchmark_1.ipynb`** -- Prototype run (M=50, K=15, 5 reps). Runs in minutes.
- **`notebooks/demo_benchmark_4_checkpointed.ipynb`** / **`demo_benchmark_2.ipynb`** -- Historical reference.

Experiments cover: correlation sweep (ρ=0.0-0.95), overlapping correlation structure, nonlinear DGP, real-world datasets (California Housing, Breast Cancer, Superconductor), epsilon sensitivity, ablation studies, variance decomposition, and statistical significance tests.

See [Experiment Guide](EXPERIMENT_GUIDE.md) for detailed descriptions of all experiments, metrics, methodology fixes, and success criteria.

---

## Key Results

At ρ=0.95 (50 features, 10 correlated groups, 20 repetitions):

```
Method                Stability   DGP Agreement (ρ)  Equity (CV)
=============================================================
Single Best              0.9529         0.9755         0.2421
Large Single Model       0.9301         0.9641         0.2708
DASH (MaxMin)            0.9819         0.9907         0.1585
```

- **Stability is flat across correlation levels**: DASH ranges 0.976-0.982 from ρ=0.0 to ρ=0.95; Single Best degrades from 0.976 to 0.953; Large Single Model from 0.965 to 0.930.
- **Bigger models make it worse**: The Large Single Model (matched compute budget in one sequential ensemble) performs worst on every metric -- confirming sequential residual dependency as the mechanism.
- **34% better equity**: DASH's within-group CV of 0.159 vs. Single Best's 0.242 at ρ=0.95.
- **Breast Cancer**: DASH nearly doubles stability (0.933 vs. 0.534) on 30 features with 21 pairs at |r| > 0.9.
- **Statistically significant** at ρ≥0.7 (Wilcoxon, Bonferroni-corrected, Cohen's d > 1.0).
- **Robust to hyperparameters**: <0.001 stability variation across a 3x range of ε values.

Full benchmark results (correlation sweep, real-world datasets, nonlinear DGP, success criteria): **[Benchmark Results](docs/BENCHMARK_RESULTS.md)**

Methodology details (A-series fixes, v6 improvements, design decision tags): **[Experiment Guide](EXPERIMENT_GUIDE.md#methodology-fixes-applied)**

---

## API Reference

The main entry point is `DASHPipeline`:

```python
from dash.core.pipeline import DASHPipeline

pipeline = DASHPipeline(M=200, K=20, epsilon=0.08, selection_method="maxmin")
pipeline.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

importance = pipeline.global_importance_       # Mean |SHAP| per feature
ranking = pipeline.get_importance_ranking()    # Feature indices, descending
fsi = pipeline.get_fsi()                       # Feature Stability Index
fig = pipeline.plot_importance_stability()     # IS Plot
preds = pipeline.get_consensus_ensemble_predictions(X_test)
```

Full API documentation (all classes, methods, attributes, data generators, evaluation metrics): **[API Reference](docs/API_REFERENCE.md)**

---

## Project Structure

```
dash-shap/
├── dash/                              # Main Python package
│   ├── core/                          # Core DASH pipeline
│   │   ├── pipeline.py                # DASHPipeline: end-to-end orchestrator
│   │   ├── population.py              # Stage 1: model population generation
│   │   ├── filtering.py               # Stage 2: performance filtering
│   │   ├── diversity.py               # Stage 3: diversity-aware selection
│   │   ├── consensus.py               # Stage 4: consensus SHAP aggregation
│   │   └── diagnostics.py             # Stage 5: FSI, IS plots, disagreement maps
│   ├── baselines/                     # Comparison methods (9 baselines)
│   ├── experiments/
│   │   └── synthetic.py               # Linear & nonlinear data generators
│   ├── evaluation/
│   │   └── __init__.py                # Metrics: DGP agreement, stability, equity
│   └── utils/                         # I/O, SHAP helpers
├── notebooks/                         # Interactive demo notebooks
│   ├── demo_benchmark_6.ipynb         # Authoritative benchmark (M=200, K=30, 20 reps)
│   ├── demo_benchmark_4_checkpointed.ipynb  # Prior benchmark (historical)
│   ├── demo_benchmark_1.ipynb         # Prototype benchmark (M=50, K=15, 5 reps)
│   └── explore_experiment_results.ipynb     # Results analysis notebook
├── docs/                              # Detailed documentation
│   ├── BENCHMARK_RESULTS.md           # Full benchmark results and tables
│   └── API_REFERENCE.md              # Complete API documentation
├── tests/                             # Test suite
├── run_experiments.py                 # Full experiment runner
├── EXPERIMENT_GUIDE.md                # Experimental design, methods, methodology fixes
├── ROADMAP.md                         # Five-paper research program
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

---

## Research Roadmap

DASH is Paper 1 of a five-paper research program building from a practical tool toward a framework for trustworthy feature attribution. Papers 2-5 extend DASH to partial orders, prove an impossibility result for total-order importance under collinearity, generalize path dependence to neural networks, and propose explanation-aware model selection as a new paradigm.

See **[Research Roadmap](ROADMAP.md)** for the full plan, timeline, and decision gates.
