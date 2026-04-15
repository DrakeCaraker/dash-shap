# Extensions Guide

DASH's extensions framework adds analysis capabilities on top of the core consensus. Every extension accepts a `DASHResult` and returns a result object with `.summary()` and `.plot()` methods.

All 12 extensions are organized below by the question they answer.

---

## "How confident am I in these rankings?"

### Confidence Intervals

Bootstrap confidence intervals tell you how much the importance estimate would change if a different subset of K models were selected. Wide CIs often signal collinearity.

```python
from dash_shap.extensions import confidence_intervals

ci = confidence_intervals(pipe.result_, alpha=0.05, n_boot=1000)
print(ci.summary())
ci.plot()  # side-by-side importance and FSI CIs
```

Wide CI + high FSI = collinear feature (attribution shared with a partner). Narrow CI + low FSI = robust estimate.

### Robust Certification

A stronger guarantee: feature j is **certified top-k** if it ranks in the top-k across *every* model — not just on average. This is a worst-case bound.

```python
from dash_shap.extensions import robust_certification

cert = robust_certification(pipe.result_, k_values=[3, 5, 10])
print(cert.summary())
# Certified top-3: ['worst_concave_points', 'worst_area']
# Certified top-5: ['worst_concave_points', 'worst_area', 'mean_concave_points']
```

Certification is monotone: certified top-3 implies certified top-5. Uncertified top-ranked features may be collinear (their rank varies across models).

---

## "Which features are collinear with each other?"

### Feature Groups

Groups features by SHAP substitutability: pairs with high cross-model correlation are interchangeable — any one of them could receive the group's total importance.

```python
from dash_shap.extensions import feature_groups

groups = feature_groups(pipe.result_, threshold=0.8, X_ref=X_ref)
print(groups.summary())
# Group 0: [mean_radius, mean_perimeter, mean_area]  — substitutability: 0.92
# Group 1: [worst_radius, worst_perimeter]            — substitutability: 0.87
```

High substitutability (near 1.0) means the models treat the features as interchangeable. Report the *group* importance, not the individual features.

### Causal Flags

Labels each feature with an actionable flag based on its IS-plot quadrant:

| Flag | Quadrant | Action |
|------|----------|--------|
| **robust** | QI: high importance, low FSI | Safe to use in decisions |
| **collinear** | QII: high importance, high FSI | Report the group, not the feature |
| **fragile** | QIV: low importance, high FSI | Exclude from downstream use |
| **unimportant** | QIII: low importance, low FSI | Safe to ignore |

```python
from dash_shap.extensions import causal_flags

flags = causal_flags(pipe.result_, X_ref=X_ref)
print(flags.summary())
# robust: 8 features
# collinear: 12 features
# fragile: 4 features
# unimportant: 6 features
```

Requires `X_ref` for computing feature correlations.

---

## "Is feature A definitely more important than feature B?"

### Partial Order

Computes the pairwise dominance probability: what fraction of the K models rank feature A above feature B?

```python
from dash_shap.extensions import partial_order

po = partial_order(pipe.result_, alpha=0.1)
print(po.summary())

# Check a specific pair
pi = po.confidence_matrix[0, 1]  # P(f0 > f1)
if abs(pi - 0.5) < 0.2:
    print("Attribution is split — these features are collinear")
```

When two features are collinear, pi(A > B) ~ 0.5 — neither dominates. When A is genuinely more important, pi(A > B) approaches 1.

---

## "What triggered this specific prediction?"

### Local Uncertainty

Shows how much the K models disagree about a single observation's SHAP values. Highlights features where the models give conflicting explanations.

```python
from dash_shap.extensions import local_uncertainty

local = local_uncertainty(pipe.result_, obs_idx=0, top_k=10)
print(local.summary())
local.plot()  # bar chart with cross-model error bars
```

High `sign_flip_rate` for a feature means the K models can't even agree on the *direction* of its effect for this observation.

---

## "I need to justify my feature selection to stakeholders"

### Stable Feature Selection

Selects the top-k features by a composite score that weights importance *and* stability. Quadrant I features (high importance, low FSI) naturally dominate.

```python
from dash_shap.extensions import stable_feature_selection

selection = stable_feature_selection(
    pipe.result_, k=5, importance_weight=0.7, stability_weight=0.3
)
print(selection.summary())
selection.plot()  # Pareto plot of importance vs stability
```

### Audit Report

One-stop structured summary combining all available evidence. Works with just a `DASHResult`; richer with optional enrichments.

```python
from dash_shap.extensions import audit_report, causal_flags, confidence_intervals

# Basic report
report = audit_report(pipe.result_)
print(report.summary())

# Enriched report with correlation analysis and causal flags
flags = causal_flags(pipe.result_, X_ref=X_ref)
ci = confidence_intervals(pipe.result_)
report = audit_report(pipe.result_, X_ref=X_ref, causal=flags, confidence=ci)
print(report.summary())
```

The report includes sections for: overview, top features, stability concerns, collinearity analysis, and any warnings. Use for model documentation, regulatory review, or stakeholder communication.

---

## "How do I pick optimal DASH parameters?"

### Theory Bridge

Predicts instability from theory before you run the full pipeline. Based on the Attribution Impossibility theorem (Lean 4 verified).

```python
from dash_shap.extensions import theory_bridge

tb = theory_bridge(pipe.result_)
print(tb.summary())
# Recommended M: 150 (for 5% target flip rate)
# worst_concave_points vs worst_perimeter: SNR=0.42, predicted flip=34%
tb.plot()  # SNR vs predicted flip rate with theoretical curve
```

Standalone functions (work without DASHResult):

```python
from dash_shap.extensions.theory_bridge import predict_flip_rate, recommend_M, divergence_ratio

predict_flip_rate(snr=1.5)     # 0.067 — 6.7% chance of ranking flip
divergence_ratio(rho=0.9)      # 5.26 — first-mover gets 5.3x more credit
recommend_M(importance_matrix)  # {'recommended_M': 150, ...}
```

### Pareto Selector

Finds optimal DASH configurations on the prediction quality vs. explanation stability frontier.

```python
from dash_shap.extensions import ParetoSelector

selector = ParetoSelector()
for M in [50, 100, 200]:
    pipe = DASHPipeline(M=M, K=max(10, M // 7), epsilon=0.05)
    pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)
    selector.evaluate(
        {"M": M, "K": pipe.K},
        pipe.result_, X_test, y_test,
        predict_fn=lambda x: pipe.get_consensus_ensemble_predictions(x),
    )

frontier = selector.frontier()
print(frontier.summary())
frontier.plot()  # scatter with Pareto frontier highlighted
```

Pareto-optimal configurations are those where you cannot improve RMSE without sacrificing stability (or vice versa).

---

## "Has my model's behavior changed?"

### Drift Monitor

Detects when feature importance drifts between model versions by comparing cosine distance on importance vectors.

```python
from dash_shap.extensions import DriftMonitor

# Set up monitor with a baseline
monitor = DriftMonitor(baseline_result, threshold=0.1)

# Check new model versions
alert = monitor.check(current_result, label="2026-Q2")
if alert.drifted:
    print(f"Drift detected! Distance: {alert.distance:.3f}")
    print(f"Changed features: {alert.changed_features}")

# Track over time
monitor.check(q3_result, label="2026-Q3")
monitor.check(q4_result, label="2026-Q4")
monitor.plot_timeline()  # bar chart of drift distance over time
```

Threshold guidance: 0.05 (strict — flags minor shifts), 0.1 (moderate), 0.2 (lenient — only major changes).

---

## "I want to combine results from multiple sites"

### Federated Consensus

Combines `DASHResult` objects from multiple sites (hospitals, institutions, departments) into a single consensus without sharing raw data.

```python
from dash_shap.extensions import federated_consensus

fed = federated_consensus([result_site1, result_site2, result_site3])
print(f"Cross-site agreement: {fed.cross_site_agreement:.3f}")
print(fed.summary())
fed.plot()  # heatmap of per-site importance

# The combined result works with ALL other extensions
from dash_shap.extensions import robust_certification
cert = robust_certification(fed.combined)
```

Supports optional per-site weights:

```python
fed = federated_consensus(
    [result_large_hospital, result_small_clinic],
    weights=[0.8, 0.2],
)
```

---

## Extension Summary

| Extension | Input | Key Output | Use When |
|---|---|---|---|
| `confidence_intervals` | DASHResult | CIs on importance, FSI, rank | Quantifying uncertainty |
| `robust_certification` | DASHResult | Certified top-k features | Need worst-case guarantees |
| `partial_order` | DASHResult | Pairwise dominance probabilities | Comparing specific features |
| `feature_groups` | DASHResult + X_ref | Feature clusters | Identifying collinear sets |
| `local_uncertainty` | DASHResult + obs_idx | Per-observation disagreement | Explaining individual predictions |
| `stable_feature_selection` | DASHResult | Top-k by importance+stability | Feature selection |
| `causal_flags` | DASHResult + X_ref | Per-feature labels | Actionable classification |
| `audit_report` | DASHResult (+ optional enrichments) | Structured report + warnings | Stakeholder communication |
| `theory_bridge` | DASHResult | SNR, flip rates, M recommendation | Parameter tuning |
| `DriftMonitor` | Baseline + current DASHResult | Drift alerts | Production monitoring |
| `ParetoSelector` | Multiple configs + results | Pareto frontier | Hyperparameter optimization |
| `federated_consensus` | Multiple DASHResults | Combined consensus | Multi-site analysis |
