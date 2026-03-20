# DASH Extensions — User Guide

> **Concept-first narrative.** For the full specification see [EXTENSIONS.md](EXTENSIONS.md).

The IS-plot tells you *which quadrant* each feature falls into. The extensions
framework lets you act on that classification. This guide is organized by user
question.

---

## "How confident am I in these rankings?"

### Confidence Intervals (`confidence_intervals` — Ext 1)

DASH's consensus importance is a point estimate. Confidence intervals tell you
how much that estimate would change if a different random subset of K models were
selected. Wide CIs often signal collinearity — credit is split between correlated
features, so any single model may assign it differently.

```python
from dash_shap.extensions import confidence_intervals

ci = confidence_intervals(pipe.result_, alpha=0.05, n_boot=1000)
print(ci.summary())
# Feature       Importance CI              FSI CI                  Rank CI
# f0_robust     [0.812, 0.921, 1.034]     [0.01, 0.03, 0.06]     [1.0, 1.0, 2.0]
# f1_collinear  [0.231, 0.895, 1.412]     ...
```

Wide importance CI + high FSI = collinear feature: its attribution is real but
shared with a partner. Narrow CI + low FSI = robust estimate you can trust.

See [EXTENSIONS.md §3 Ext 1](EXTENSIONS.md#extension-1-confidence-intervals-confidencepy)
for full API.

### Robust Certification (`robust_certification` — Ext 9)

A stronger guarantee: feature j is **certified top-k** if it ranks in the top-k
across *every* model in the ensemble — not just on average. This is a worst-case
bound, not a probabilistic CI.

```python
from dash_shap.extensions import robust_certification

cert = robust_certification(pipe.result_, k_values=[3, 5, 10])
print(cert.summary())
# Certified top-3: ['f0_robust', 'f2_stable']
# Certified top-5: ['f0_robust', 'f2_stable', 'f4_robust']
```

Certified features are safe to include in downstream decisions regardless of
which K models were selected. Uncertified top-ranked features may be collinear
(their rank varies across models).

Certification is monotone: certified top-3 implies certified top-5.

---

## "Which features are collinear with each other?"

### Feature Groups (`feature_groups` — Ext 4, Phase 2)

Groups features by *SHAP substitutability*: pairs with high cross-model
correlation are interchangeable — any one of them could receive the group's
total importance. This directly identifies IS-plot QII features and their
collinear partners.

```python
from dash_shap.extensions import feature_groups  # Phase 2

groups = feature_groups(pipe.result_, threshold=0.8, X_ref=X_explain)
print(groups.summary())
# Group 0: [f0, f1, f2, f3]  — mean substitutability: 0.92
# Group 1: [f5, f6]           — mean substitutability: 0.87
```

High substitutability (→ 1.0) means the models treat the features as
interchangeable. Report the *group* importance, not the individual features.

### Causal Flags (`causal_flags` — Ext 11, Phase 3)

Combines FSI and correlation to label each feature with its IS-plot quadrant:

| Flag | IS-Plot Quadrant | Action |
|------|-----------------|--------|
| `robust` | QI — high imp, low FSI | Safe to use in decisions |
| `collinear` | QII — high imp, high FSI | Report the group, not the feature |
| `fragile` | QIV — low imp, high FSI | Exclude from downstream use |
| *(unlabeled)* | QIII — low imp, low FSI | Unimportant but stable; safe to ignore |

```python
from dash_shap.extensions import causal_flags  # Phase 3

flags = causal_flags(pipe.result_, X_ref=X_explain)
print(flags.summary())
```

---

## "Is feature A definitely more important than feature B?"

### Partial Orders (`partial_order` — Ext 2)

π(A > B) = fraction of the K models that rank A above B. When two features
are collinear, π(A > B) ≈ 0.5 — neither dominates, attribution is split.
When A is genuinely more important across all models, π(A > B) → 1.

```python
from dash_shap.extensions import partial_order

po = partial_order(pipe.result_, alpha=0.1)
print(po.summary())

# Paper 2 check: within collinear group
pi = po.confidence_matrix[0, 1]  # π(f0 > f1)
if abs(pi - 0.5) < 0.2:
    print("Attribution is split — these features are collinear")
```

This is the core metric for Paper 2: within a collinear group, π ≈ 0.5.
Between groups (different true importance), π → 1.

---

## "I need to justify my feature selection to stakeholders"

### Audit Report (`audit_report` — Ext 3, Phase 3)

One-stop summary combining all available evidence into a structured report
with flagged warnings.

```python
from dash_shap.extensions import audit_report, feature_groups  # Phase 3

groups = feature_groups(pipe.result_, X_ref=X_explain)
report = audit_report(pipe.result_, X_ref=X_explain, groups=groups)
print(report.summary())
```

Works with just `DASHResult` (basic report). Richer with optional
`X_ref`, `groups`, `confidence`, and `partial_order` enrichments.

### Stable Feature Selection (`stable_feature_selection` — Ext 5, Phase 2)

Selects the top-k features by a composite score that weights importance
*and* stability. QI features (high importance, low FSI) naturally dominate.

```python
from dash_shap.extensions import stable_feature_selection  # Phase 2

selection = stable_feature_selection(
    pipe.result_, k=5, importance_weight=0.7, stability_weight=0.3
)
print(selection.summary())
```

---

## "I want to combine results from multiple sites"

### Federated Consensus (`federated_consensus` — Ext 10, Phase 4)

Combines `DASHResult` objects from multiple sites into a single consensus
result without sharing raw data. The combined result is itself a `DASHResult`,
so all Phase 1-3 extensions work on it directly.

```python
from dash_shap.extensions import federated_consensus  # Phase 4

combined = federated_consensus([result_site1, result_site2, result_site3])
print(f"Cross-site agreement: {combined.cross_site_agreement:.3f}")
cert = robust_certification(combined.combined)  # all extensions work on combined
```

---

## "Has my data distribution changed?"

### Drift Monitor (`DriftMonitor` — Ext 6, Phase 4)

Stores a baseline `DASHResult` and detects when current importance vectors
drift beyond a threshold (cosine distance).

```python
from dash_shap.extensions import DriftMonitor  # Phase 4

monitor = DriftMonitor(baseline_result, threshold=0.1)
alert = monitor.check(current_result, label="2026-Q2")
if alert.drifted:
    print(f"Drift detected! Distance: {alert.distance:.3f}")
    print(f"Changed features: {alert.changed_features}")
```

---

## Implementation Status

| Phase | Extensions | Status |
|-------|-----------|--------|
| **0** | `DASHResult`, `_base.py`, serialization, `fit_from_attributions()` | ✅ Implemented |
| **1** | CI (1), Partial Orders (2), Certification (9) | ✅ Implemented |
| **2** | Groups (4), Local (8), Selection (5) | Planned |
| **3** | Causal (11), Audit (3) | Planned |
| **4** | Drift (6), Federated (10) | Planned |
| **5** | ParetoSelector (7) | Planned (Paper 5) |

---

## Full API Reference

See [EXTENSIONS.md](EXTENSIONS.md) for complete specification including:
- All dataclass field definitions
- Dependency graph (zero hard dependencies between extensions)
- Phasing rationale
- Testing strategy and property tests
- Backward compatibility guarantees
