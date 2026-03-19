# API Reference

## `DASHPipeline`

The main entry point for using DASH.

```python
from dash_shap.core.pipeline import DASHPipeline

pipeline = DASHPipeline(
    M=200,                              # Number of models in the population
    K=30,                               # Number of models to select for consensus
    epsilon=0.08,                       # Performance filter threshold
    epsilon_mode="absolute",            # "absolute", "relative", or "quantile"
    selection_method="maxmin",          # "maxmin", "cluster", or "dedup"
    delta=0.05,                         # Diversity threshold (maxmin)
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

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X_train, y_train, X_val, y_val, X_ref=None, feature_names=None)` | self | Runs all 5 stages. `X_ref` defaults to `X_val`. **Recommended:** pass a dedicated explain set (e.g., `X_explain`) as `X_ref`, separate from `X_test`, so that SHAP reference data and RMSE evaluation data do not overlap. |
| `get_fsi()` | `FeatureStabilityIndex` | Feature Stability Index object with quadrant labels. |
| `plot_importance_stability(groups=None, **kwargs)` | matplotlib Figure | Generates the IS Plot. Pass `groups` to color by feature group. |
| `get_importance_ranking()` | np.array | Feature indices sorted by descending importance. |
| `get_consensus_ensemble_predictions(X)` | np.array | Mean predictions across selected models. |

### Attributes (available after `fit`)

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

---

## `FeatureStabilityIndex`

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

---

## `ImportanceStabilityPlot`

```python
from dash_shap.core.diagnostics import ImportanceStabilityPlot

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

---

## `local_disagreement_map`

```python
from dash_shap.core.diagnostics import local_disagreement_map

fig = local_disagreement_map(
    all_shap_matrices,    # (K, N', P) array
    observation_idx=0,    # Which observation to explain
    feature_names=None,   # Optional feature names
    top_k=15,             # Number of features to show
    figsize=(10, 6),
)
```

---

## Synthetic Data Generators

```python
from dash_shap.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear

# Linear DGP with controllable correlation (11-tuple return with four-way split)
X_train, y_train, X_val, y_val, X_explain, y_explain, X_test, y_test, \
    groups, true_importance, meta = \
    generate_synthetic_linear(
        N=5000,           # Total observations
        P=50,             # Number of features
        group_size=5,     # Features per correlated group
        rho=0.9,          # Within-group correlation
        sigma_noise=0.5,  # Noise standard deviation
        seed=42,
        test_size=0.15,
        val_size=0.15,
        explain_size=0.10, # Dedicated SHAP reference set
        structure="block", # "block" or "overlapping"
    )

# Nonlinear DGP (same interface, adds quadratic/interaction/sin terms)
X_train, y_train, X_val, y_val, X_explain, y_explain, X_test, y_test, \
    groups, true_importance, meta = \
    generate_synthetic_nonlinear(N=5000, P=50, group_size=5, rho=0.9)
```

The four-way split ensures `X_explain` (used as `X_ref` for SHAP) is separate from `X_test` (used only for RMSE evaluation).

---

## Evaluation Metrics

```python
from dash_shap.evaluation import (
    dgp_agreement,             # formerly importance_accuracy (alias retained)
    group_level_accuracy,
    group_level_mse,
    importance_stability,
    stability_bootstrap_ci,    # BCa bootstrap CI for stability
    within_group_equity,
)

# DGP agreement vs. ground truth (sanity check, not primary criterion)
spearman_rho, mse = dgp_agreement(estimated_importance, true_importance)

# Group-level accuracy — Spearman of group-level sums vs true group betas
# Note (C8): saturates at 1.0 when true group betas are well-separated
gacc = group_level_accuracy(estimated_importance, true_importance, groups)

# Group-level MSE — normalized MSE of group-level proportions vs true proportions
# Discriminative complement to gacc when rank order is trivially correct
gmse = group_level_mse(estimated_importance, true_importance, groups)

# Stability across repetitions
stability = importance_stability([importance_run1, importance_run2, ...])

# BCa bootstrap confidence interval for stability
point, se, ci_lo, ci_hi = stability_bootstrap_ci(
    [importance_run1, importance_run2, ...], n_boot=1000, ci=0.95
)

# Within-group equity (optionally score zero-importance groups)
mean_cv = within_group_equity(importance_vector, group_assignments,
                              include_zero_groups=False)
```

> `importance_accuracy` is retained as a backward-compatible alias for `dgp_agreement`.

---

## Statistical Testing

```python
from dash_shap.evaluation import (
    cohens_d,
    compare_methods,
    friedman_test,
    holm_bonferroni,
    feature_ablation_score,
)

# Cohen's d effect size between two groups of scores
d = cohens_d(dash_scores, baseline_scores)

# Wilcoxon signed-rank test between paired scores
stat, pval = compare_methods(dash_scores, baseline_scores)

# Friedman chi-square test across multiple methods
stat, pval = friedman_test(method1_scores, method2_scores, method3_scores)

# Holm-Bonferroni step-down correction for multiple comparisons
adjusted_pvals = holm_bonferroni(raw_p_values)

# Feature ablation score — proxy for explanation quality on real data
# Zeros out top-K features and measures prediction degradation
degradation = feature_ablation_score(
    model, X, y, importance, top_k=5, metric_fn=None  # default: RMSE
)
```
