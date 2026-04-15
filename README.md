# DASH: Diversified Aggregation of SHAP

[![PyPI](https://img.shields.io/pypi/v/dash-shap.svg)](https://pypi.org/project/dash-shap/)
[![ArXiv](https://img.shields.io/badge/arXiv-2603.22346-b31b1b.svg)](https://arxiv.org/abs/2603.22346)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stable feature importance when your features are correlated.** SHAP rankings from a single model are partially random under collinearity — DASH fixes this by averaging explanations across independent models so the noise cancels and the signal remains. Works with any attribution method.

## Install

```bash
pip install dash-shap
```

## Quick Check — Is Your Model Affected?

```python
from dash_shap import check
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
result = check(data.data, data.target, task="binary",
               feature_names=list(data.feature_names))
print(result.report())
```

```
DASH Stability Check
========================================
Models trained: 25
Features: 30
Unstable pairs: 68
Correlated groups: 5

UNSTABLE PAIRS (rankings flip across retrains):
  worst radius vs worst perimeter: flip rate 48%, predicted 47%
  mean radius vs mean perimeter: flip rate 44%, predicted 43%
  ...

RECOMMENDATION: Use M=150 models for 5% flip rate target

TOP FEATURES (DASH consensus):
  1. worst concave points: 0.0831 (stable)
  2. worst perimeter: 0.0534 (UNSTABLE)
  3. worst radius: 0.0521 (UNSTABLE)
  ...
```

`result.plot()` shows the Importance-Stability plot. `result.dash_importance()` returns stable consensus rankings. `result.to_dataframe()` gives a full summary with SNR predictions.

## Full Pipeline

For production use with diversity selection and full diagnostics:

```python
from dash_shap import DASHPipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_temp, X_ref, y_temp, _ = train_test_split(X_temp, y_temp, test_size=0.12, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

pipe = DASHPipeline(M=100, K=20, epsilon=0.05, epsilon_mode="relative",
                    task="binary", seed=42)
pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)

# Stable consensus importance
print(pipe.global_importance_)

# Importance-Stability plot: which features are stable vs. collinear?
pipe.plot_importance_stability()

# Feature Stability Index: per-feature stability scores
pipe.get_fsi().summary(top_k=10)
```

The IS plot places features in four quadrants:

| Quadrant | Importance | Stability | Action |
|---|---|---|---|
| **I: Robust Drivers** | High | Stable | Report individually |
| **II: Collinear Cluster** | High | Unstable | Report as a group |
| **III: Unimportant** | Low | Stable | Safe to omit |
| **IV: Fragile** | Low | Unstable | Investigate or drop |

## Using with Any Attribution Method

The default pipeline uses XGBoost + TreeSHAP, but DASH works with **any** attribution source — LIME, Integrated Gradients, attention maps, KernelSHAP, or custom methods:

```python
# You have M attribution vectors from any source, shape (M, P)
# e.g., LIME importance from M independently trained neural networks
pipe = DASHPipeline(M=100, K=20, epsilon=0.05)
pipe.fit_from_attributions(attribution_matrices, val_scores)
print(pipe.global_importance_)

# Or use the lightweight stability workflow directly
from dash_shap import validate_from_attributions, consensus_from_attributions

results = validate_from_attributions(attribution_matrix)  # Z-tests + flip rates
stable = consensus_from_attributions(attribution_matrix)   # averaged importance
```

The underlying [impossibility theorem](https://github.com/DrakeCaraker/dash-impossibility-lean) is proved for all iterative optimizers (gradient boosting, Lasso, neural networks), so the resolution — independent model averaging — is equally general.

## Extensions

After fitting, `pipe.result_` is a `DASHResult` that supports additional analyses:

```python
from dash_shap.extensions import confidence_intervals, robust_certification, theory_bridge

# Bootstrap confidence intervals on importance and FSI
ci = confidence_intervals(pipe.result_)
print(ci.summary())

# Certify which features are top-k across ALL models (worst-case guarantee)
cert = robust_certification(pipe.result_)
print(cert.certified[5])  # features certified top-5

# Theory bridge: predict flip rates from impossibility theorem formulas
tb = theory_bridge(pipe.result_)
print(tb.summary())       # SNR per pair, predicted flip rates, M recommendation
```

All extensions: `confidence_intervals`, `partial_order`, `feature_groups`, `stable_feature_selection`, `local_uncertainty`, `robust_certification`, `theory_bridge`.

## How It Works

DASH is a five-stage pipeline:

1. **Population** — Train M independent models with random hyperparameters and restricted feature access, so each model explores different features
2. **Filtering** — Keep models within ε of the best validation score
3. **Diversity Selection** — Greedy MaxMin selection maximizing pairwise distance among importance vectors
4. **Consensus** — Compute attributions for each selected model, then average
5. **Diagnostics** — Feature Stability Index (FSI) and IS plots for auditing without ground truth

The key insight: **model independence** — not model size — is what cancels the arbitrary noise. See the [paper](https://arxiv.org/abs/2603.22346) for the full mechanism analysis.

## Documentation

| Resource | Description |
|---|---|
| [API Reference](docs/API_REFERENCE.md) | Complete parameter and method documentation |
| [FAQ](FAQ.md) | Usage, troubleshooting, parameter guidance |
| [Diagnostics Guide](docs/DIAGNOSTICS.md) | IS plots, FSI thresholds, disagreement maps |
| [Getting Started](docs/GETTING_STARTED.md) | Concept-first introduction with worked examples |
| [Research & Benchmarks](docs/RESEARCH.md) | Full results, method comparisons, reproduction guide |
| [Tutorials](notebooks/) | Step-by-step notebooks from basic to advanced |

## Citation

```bibtex
@misc{caraker2026firstmover,
  title={First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution},
  author={Caraker, Drake and Arnold, Bryan and Rhoads, David},
  year={2026},
  eprint={2603.22346},
  archiveprefix={arXiv},
  primaryclass={cs.LG},
  doi={10.5281/zenodo.19060132},
  url={https://arxiv.org/abs/2603.22346}
}
```

## License

[MIT](LICENSE)
