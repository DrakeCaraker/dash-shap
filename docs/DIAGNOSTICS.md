# Interpreting DASH Diagnostics

DASH provides three diagnostic tools that let you audit feature importance explanations **without ground truth**. They answer: *which features have stable attributions and which are affected by first-mover bias?*

Under feature collinearity, gradient-boosted models arbitrarily assign credit to whichever correlated feature appears first in a tree split — the "first-mover bias." Different random seeds produce different winners, making SHAP explanations unstable. DASH's diagnostics quantify this instability at both the feature and observation level.

---

## Worked Example

After fitting a `DASHPipeline`, diagnostics are available immediately:

```python
from dash_shap.core.pipeline import DASHPipeline
from dash_shap.core.diagnostics import local_disagreement_map
import numpy as np

pipeline = DASHPipeline(M=200, K=30, epsilon=0.08, seed=42)
pipeline.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

# 1. Importance-Stability plot — visual overview
fig = pipeline.plot_importance_stability(
    title="IS Plot — My Dataset",
    annotate_top_k=8,
)

# 2. FSI summary table — numerical view
fsi = pipeline.get_fsi()
print(fsi.summary(top_k=10))

# 3. Quadrant labels — classify each feature
labels = fsi.get_quadrant_labels()
for name, label in zip(fsi.feature_names, labels):
    print(f"  {name}: {label}")

# 4. Local disagreement map — drill into one observation
var_per_obs = np.mean(pipeline.variance_matrix_, axis=1)
fig = local_disagreement_map(
    pipeline.all_shap_matrices_, np.argmax(var_per_obs),
    feature_names=pipeline.feature_names_, top_k=15,
)
```

---

## The Importance-Stability Plot

A 2D scatter where each point is a feature:

- **X-axis**: Consensus importance — `mean(|SHAP|)` averaged across K selected models
- **Y-axis**: Feature Stability Index (FSI) — cross-model disagreement relative to signal strength

Dashed lines split the plot into four quadrants at adaptive thresholds (median importance; median FSI among high-importance features). Points are colored by quadrant.

**Key parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `groups` | None | Color points by known feature group instead of quadrant |
| `annotate_top_k` | 5 | Number of top features to label on the plot |
| `importance_threshold` | median | Custom vertical split |
| `fsi_threshold` | median of high-importance FSI | Custom horizontal split |
| `figsize` | (10, 7) | Figure dimensions |
| `title` | "Importance-Stability Plot" | Plot title |

When `groups` is provided (e.g., from a synthetic DGP or domain knowledge), the plot colors by group membership instead of quadrant, making it easy to confirm that collinear features cluster together.

---

## Quadrant Interpretation Guide

| Quadrant | Importance | FSI | Label | Color |
|----------|-----------|-----|-------|-------|
| I | High | Low | Robust Drivers | Green |
| II | High | High | Collinear Cluster | Red |
| III | Low | Low | Confirmed Unimportant | Gray |
| IV | Low | High | Fragile Interactions | Orange |

### I: Robust Drivers
Models agree these features are important. Trust them. Report directly to stakeholders.

### II: Collinear Cluster
Important but unstable — models disagree on which correlated feature gets credit. **Interpret as a group, not individually.** Check the correlation matrix to identify the cluster. The `within_group_equity` metric quantifies how evenly credit is distributed within a group.

### III: Confirmed Unimportant
Low importance and consistent across models. Safe to omit from reports.

### IV: Fragile Interactions
Low importance but high relative variance. Often interaction terms or weak signals amplified by specific model configurations. Investigate further — if importance is near zero, likely noise; if FSI is extreme, the signal may be real but model-dependent.

---

## Feature Stability Index (FSI) In Depth

**Definition:**

```
FSI_j = mean_std_j / (mean_abs_consensus_j + epsilon)
```

where `mean_std_j` is the average (across observations) of the cross-model standard deviation for feature j, and `mean_abs_consensus_j` is the average absolute consensus SHAP value. This is a coefficient-of-variation-style ratio: disagreement relative to signal strength.

**Interpretation:**

- **FSI = 0** — Perfect agreement across all K models
- **FSI < 0.3** — Highly stable; models agree on this feature's role
- **FSI 0.3–0.7** — Moderate instability; worth investigating
- **FSI > 0.7** — High instability; interpret with caution, likely collinear
- **FSI > 1.0** — Disagreement exceeds signal; do not interpret individually

These thresholds are rough guidelines derived from synthetic and real-world experiments. The IS plot's adaptive median-based quadrant splits are the primary mechanism — they adjust automatically to each dataset.

**Accessing FSI values:**

```python
fsi_obj = pipeline.get_fsi()
fsi_obj.fsi                # (P,) raw FSI array
fsi_obj.global_importance  # (P,) importance array
fsi_obj.feature_names      # list of feature names
fsi_obj.summary(top_k=10)  # formatted table string
```

---

## Local Disagreement Maps

A horizontal bar chart showing **one observation's** SHAP values (consensus across K models) with error bars representing cross-model standard deviation.

**What it shows:**
- Bar length = consensus SHAP value (red = positive, blue = negative)
- Error bars = ±1 standard deviation across K models
- Wide error bars = models disagree on that feature's contribution for this observation

**When to use:**
After the IS plot reveals QII or QIV features, drill into specific observations to see *where* and *how* models disagree.

**How to pick observations:**

```python
# Highest overall disagreement
var_per_obs = np.mean(pipeline.variance_matrix_, axis=1)
high_var_idx = np.argmax(var_per_obs)

# Highest disagreement on a specific feature
feature_idx = 3  # e.g., feature of interest
high_var_idx = np.argmax(pipeline.variance_matrix_[:, feature_idx])

# Domain-interesting cases (outliers, edge cases, misclassified)
high_var_idx = 42
```

**Parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `observation_idx` | (required) | Index into the explain set |
| `top_k` | 15 | Number of features shown (sorted by |SHAP|) |
| `feature_names` | f0, f1, ... | Feature name labels |
| `figsize` | (10, 6) | Figure dimensions |
| `title` | auto-generated | Custom plot title |

---

## Common Patterns

**Correlated feature pairs splitting credit.** In datasets with high collinearity (e.g., Breast Cancer with 21 feature pairs at |r| > 0.9), many features appear in QII. The IS plot shows a cluster of red points in the upper-right. Action: report group-level importance rather than individual feature rankings.

**Clean quadrant separation in synthetic data.** With known ground truth and moderate correlation (rho >= 0.7), QI features correspond to truly important features and QIII to noise. QII appears when correlated features compete for the same signal.

**High-variance observations at distribution boundaries.** The observations with the highest cross-model disagreement tend to lie near decision boundaries or in sparse regions of feature space. Local disagreement maps on these observations reveal which features are most contested.

**Feature groups for visual confirmation.** When you have domain knowledge or synthetic group assignments, pass `groups=` to `plot_importance_stability()` to color by group rather than quadrant. This visually confirms that QII features belong to the same correlated groups.

---

## API Quick Reference

| Tool | Via pipeline | Direct import |
|------|-------------|---------------|
| IS Plot | `pipeline.plot_importance_stability(**kwargs)` | `ImportanceStabilityPlot.plot(importance, fsi, ...)` |
| FSI object | `pipeline.get_fsi()` | `FeatureStabilityIndex(fsi, importance, names)` |
| FSI summary | `pipeline.get_fsi().summary(top_k=10)` | — |
| Quadrant labels | `pipeline.get_fsi().get_quadrant_labels()` | — |
| Local disagreement | — | `local_disagreement_map(all_shap_matrices, idx, ...)` |
| Variance matrix | `pipeline.variance_matrix_` | — |
| Raw SHAP matrices | `pipeline.all_shap_matrices_` | — |

For full parameter documentation, see **[API Reference](API_REFERENCE.md)**.

---

## Going Further: Extensions Framework

The IS-plot and FSI identify *which quadrant* each feature falls into. The
**[extensions framework](EXTENSIONS_GUIDE.md)** lets you act on that classification:

| Question | Extension | What it does |
|---|---|---|
| "How confident is the ranking?" | `confidence_intervals` | BCa bootstrap CI for importance, FSI, and rank |
| "Which features are certified safe?" | `robust_certification` | Worst-case guarantee: top-k across *every* model |
| "Is feature A definitely more important than B?" | `partial_order` | π(A>B) — fraction of models ranking A above B |
| "Which features are collinear with each other?" | `feature_groups` | Cluster on SHAP substitutability |
| "What's the actionable label for each feature?" | `causal_flags` | Labels: robust / collinear / fragile / unimportant |
| "What triggered this observation's prediction?" | `local_uncertainty` | Per-observation K×P slice with sign-flip rate |
| "How do I pick the best features?" | `stable_feature_selection` | Composite importance+stability ranking |
| "Can I justify this to stakeholders?" | `audit_report` | Structured report with warnings |
| "What flip rate should I expect?" | `theory_bridge` | SNR, predicted flip rates, M recommendation |
| "Has my model's behavior changed?" | `DriftMonitor` | Cosine distance between baseline and current importance |
| "What's the optimal DASH config?" | `ParetoSelector` | RMSE-stability Pareto frontier |
| "Can I combine results from multiple sites?" | `federated_consensus` | Cross-site consensus without sharing data |

```python
# After pipe.fit(...):
from dash_shap.extensions import robust_certification, confidence_intervals, causal_flags

cert = robust_certification(pipe.result_)
print(cert.summary())   # which features are certified top-3?

ci = confidence_intervals(pipe.result_)
print(ci.summary())     # how wide are the importance CIs?

flags = causal_flags(pipe.result_, X_ref=X_ref)
print(flags.summary())  # robust / collinear / fragile / unimportant
```

See the **[Extensions Guide](EXTENSIONS_GUIDE.md)** for worked examples of all 12 extensions, organized by use case.

---

## Coverage Conflict

Coverage conflict is a **sign-stability** diagnostic that complements FSI's **magnitude-stability** measurement. While FSI measures how much a feature's SHAP values vary in magnitude across models, coverage conflict measures whether models even agree on the *direction* (positive or negative) of a feature's effect.

**What it measures:** For each (observation, feature) pair, coverage conflict counts how many of the K models assign a positive vs. negative SHAP value. The **minority fraction** — `min(n_positive, n_negative) / total` — quantifies the degree of sign disagreement.

**Interpreting `minority_fraction`:**

| Value | Meaning |
|-------|---------|
| 0.0 | Unanimous — all models agree on the sign |
| 0.0–0.1 | Near-consensus — one or two dissenting models |
| 0.1–0.3 | Moderate disagreement — sign is dataset-dependent |
| 0.3–0.5 | Severe disagreement — models are nearly split on direction |
| 0.5 | Perfect split — half positive, half negative (coin flip) |

The distribution-free minority fraction is a distribution-free predictor of sign instability. Its performance relative to the parametric Gaussian flip formula is regime-dependent: coverage conflict wins under weak/partial collinearity (California Housing: Spearman 0.96 vs 0.46) while the Gaussian formula wins under strong/pervasive collinearity (Breast Cancer: 0.93 vs 0.45). Use `compare_flip_predictors()` to determine which is more reliable on your dataset. This is grounded in the bilemma's all-or-nothing theorem: features are either unanimously signed or split, with a predicted dead zone in between.

**When to use coverage conflict vs. FSI:**

- Use **FSI** to identify features where the *magnitude* of importance is unstable (collinear features trading credit).
- Use **coverage conflict** to identify features where the *sign* of the effect is unstable (models disagree on whether a feature increases or decreases the prediction).
- A feature can have low FSI (stable magnitude) but high coverage conflict (models disagree on direction) — this happens when a feature has a near-zero effect that fluctuates around zero.

**Code example:**

```python
from dash_shap.core.diagnostics import coverage_conflict, compare_flip_predictors

# After fitting a DASHPipeline:
cc = coverage_conflict(pipeline.all_shap_matrices_)

# Per-feature summary
for j, name in enumerate(pipeline.feature_names_):
    rate = cc["feature_conflict_rate"][j]
    minority = cc["feature_mean_minority"][j]
    print(f"{name}: conflict_rate={rate:.2f}, mean_minority={minority:.3f}")

# Compare against Gaussian flip predictor
comparison = compare_flip_predictors(pipeline.all_shap_matrices_)
print(f"CC predictions:  {comparison['cc_prediction'][:5]}")
print(f"Gaussian predictions: {comparison['gf_prediction'][:5]}")
```

**API quick reference update:**

| Tool | Via pipeline | Direct import |
|------|-------------|---------------|
| Coverage conflict | — | `coverage_conflict(all_shap_matrices)` |
| Flip predictor comparison | — | `compare_flip_predictors(all_shap_matrices)` |
