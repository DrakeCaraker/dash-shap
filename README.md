# DASH: Diversified Aggregation of SHAP

[![ArXiv](https://img.shields.io/badge/arXiv-2603.22346-b31b1b.svg)](https://arxiv.org/abs/2603.22346)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19060132.svg)](https://doi.org/10.5281/zenodo.19060132)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stable feature importance explanations under collinearity via independent model aggregation.**

> Caraker, Arnold, Rhoads (2026). *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* [arXiv:2603.22346](https://arxiv.org/abs/2603.22346) | [Zenodo](https://doi.org/10.5281/zenodo.19060132). Target venue: TMLR.

---

## The Problem

SHAP is one of the most widely used tools for explaining machine learning predictions. But it has a hidden fragility: when features are correlated, the explanation changes depending on which feature the model happens to use first. Train the same XGBoost model twice with different random seeds and you can get completely different importance rankings — even though the predictions are identical. In regulated, high-stakes, or scientific settings, this means the "explanation" is partially arbitrary.

This instability comes from a specific mechanism — **sequential residual dependency** in gradient boosting — and the intuitive fix (bigger models) makes it *worse*. A single large model with the same total tree count as DASH produces the worst explanations of any method tested, because more sequential trees means more reinforcement of an arbitrary initial feature choice.

**DASH fixes this.** It trains many small models independently, so each one makes its own arbitrary choice about which correlated feature to use. When you average their explanations, the arbitrary noise cancels out and the signal remains — which *group* of features matters, with credit distributed fairly across the group.

---

## Key Results

At high collinearity (ρ = 0.9, 50 features, 10 correlated groups, 50 repetitions):

| Method | Stability | Top-K5 | Equity (CV) |
|---|---|---|---|
| Single Best | 0.958 | 0.546 | 0.232 |
| Large Single Model | 0.938 | 0.433 | 0.258 |
| **DASH (MaxMin)** | **0.977** | **0.863** | **0.175** |

Stability on real-world datasets (50 reps):

| Dataset | Features | Single Best | DASH | SR | Improvement |
|---|---|---|---|---|---|
| Breast Cancer | 30 (21 pairs \|r\| > 0.9) | 0.376 | **0.925** | 0.862 | **+0.549** |
| Superconductor | 81 | 0.840 | **0.964** | 0.924 | +0.124 |
| California Housing | 8 | 0.969 | 0.978 | 0.977 | +0.009 (n.s.) |

*SR = Stochastic Retrain (seed averaging).*

DASH stability is flat across correlation levels (0.973–0.977 from ρ = 0.0 to ρ = 0.95). On Breast Cancer — the most extreme collinearity case — DASH outperforms Stochastic Retrain by +0.063, the largest DASH-SR gap in any experiment.

*All results from 50-rep SageMaker run (PAPER_CONFIG: M=200, K=30, ε=0.08, δ=0.05). See [Benchmark Results](docs/BENCHMARK_RESULTS.md) for full tables.*

Full results: **[Benchmark Results](docs/BENCHMARK_RESULTS.md)** | Methodology: **[Experiment Guide](EXPERIMENT_GUIDE.md)**

---

## Learning Path

**New to DASH or to feature importance under collinearity?**

| Step | Resource | What You'll Learn |
|---|---|---|
| 1 | [Getting Started](docs/GETTING_STARTED.md) | Concept-first introduction: the problem, the mechanism, how DASH solves it |
| 2 | [Quickstart Notebook](notebooks/quickstart.ipynb) | 3-minute end-to-end demo on synthetic data: fit, IS plot, FSI summary |
| 3 | [Tutorial 1: The Problem](notebooks/tutorial_01_the_problem.ipynb) | See SHAP ranking instability on Breast Cancer before DASH is introduced |
| 4 | [Tutorial 2: How DASH Works](notebooks/tutorial_02_dash_walkthrough.ipynb) | 5-stage walkthrough on Breast Cancer — inspect intermediate outputs at each stage |
| 5 | [Tutorial 3: Reading Results](notebooks/tutorial_03_interpreting_outputs.ipynb) | IS plot, FSI, local disagreement maps — quadrant action guide with clinical feature names |
| 6 | [Tutorial 4: Parameter Exploration](notebooks/tutorial_04_simulation.ipynb) | Sweep ρ, M, K, epsilon; understand why the Breast Cancer +0.549 result happens |

**Already familiar with SHAP?**

- [API Reference](docs/API_REFERENCE.md) — complete parameter and method documentation
- [Diagnostics Guide](docs/DIAGNOSTICS.md) — deep-dive on IS plots, FSI thresholds, disagreement maps
- [Experiment Guide](EXPERIMENT_GUIDE.md) — full paper methodology, 9-method comparison
- [FAQ](FAQ.md) — usage, troubleshooting, and parameter guidance

---

## When to Use DASH

### Detecting the problem

Before trusting a SHAP-based feature ranking from a gradient-boosted model, train multiple models with different random seeds and compare their importance rankings. If rankings differ substantially, your explanations are affected by **first-mover bias** — the sequential residual fitting in gradient boosting arbitrarily concentrates importance on whichever correlated feature the model happens to use first.

### When DASH is most valuable

Use DASH when:

- **Features are correlated** — The stability advantage is statistically significant at ρ ≥ 0.7 and grows with correlation severity. At ρ = 0.9, DASH improves stability from 0.958 to 0.977 over the single-best baseline.
- **Explanation stability matters** — In regulated, high-stakes, or scientific settings where explanations must be reproducible across model retrains.
- **Equitable credit distribution is needed** — DASH distributes importance proportionally across correlated feature groups (within-group CV = 0.175) rather than concentrating it on an arbitrary group member.
- **You need to audit explanations without ground truth** — DASH's Feature Stability Index (FSI) and Importance-Stability (IS) plots detect which specific features are affected by first-mover bias. Quadrant II features (high importance, high instability) should be interpreted as collinear cluster members rather than individually important features.

### When simpler alternatives suffice

**Stochastic Retrain** (same hyperparameters, different seeds) achieves stability equivalent to DASH (~0.977 at ρ = 0.9) with minimal implementation effort. Use it when diagnostics and equity are not required.

### DASH vs. other methods

| Method | Stability | Equity | Diagnostics | Notes |
|---|---|---|---|---|
| **Single Best Model** | Unstable under collinearity (0.376 on Breast Cancer) | Poor — concentrates credit arbitrarily | None | Standard workflow; unreliable when features are correlated |
| **Large Single Model** | *Worst* of all methods tested | Worst | None | More sequential trees *amplifies* first-mover bias — do not scale up single models |
| **Stochastic Retrain** | Equivalent to DASH (~0.977) | Moderate | None | Sufficient when only stability matters; lacks equity and diagnostics |
| **Stability Selection** | N/A (feature selection, not explanation) | N/A | N/A | Complementary: perturbs *data* and selects features; DASH perturbs *models* and distributes credit. Use both together. |
| **Ensemble SHAP** | Good | Moderate | None | Standard ensembles lack forced feature restriction and diversity selection, producing less diverse models |
| **DASH** | **Best or tied-best** | **Best** (lowest within-group CV) | **FSI + IS plots** | Full pipeline with diagnostics and equity |

### What NOT to do

Do not increase the size of a single model to fix instability. The Large Single Model experiment — matching DASH's total tree count in one sequential model — produces the **worst** explanations of any method tested, confirming that sequential dependency, not model capacity, drives the problem.

### Current scope

DASH currently targets **XGBoost + interventional TreeSHAP**. It averages main-effect SHAP value matrices; interaction tensor averaging is theoretically supported but not yet implemented. Extension to other model families (e.g., neural networks with KernelSHAP) is planned for future work.

---

## Installation

```bash
pip install dash-shap
```

Or install from source:

```bash
git clone https://github.com/DrakeCaraker/dash-shap.git
cd dash-shap
pip install -e .
```

**Requirements:** Python >= 3.9. No GPU needed.

Activate the pre-push hook (blocks `.pkl` files and files >10MB):

```bash
git config core.hooksPath .githooks
```

**Verify your installation:**
```bash
python -c "from dash_shap import DASHPipeline; print('DASH installation OK')"
```

| Package | Version | Purpose |
|---|---|---|
| xgboost | >= 2.0.0 | Gradient boosting models |
| shap | >= 0.44.0 | TreeExplainer for SHAP values |
| scikit-learn | >= 1.4.0 | Data splitting, metrics |
| numpy | >= 1.24.0 | Numerical computation |
| scipy | >= 1.11.0 | Statistical tests, clustering |
| joblib | >= 1.3.0 | Parallel model training |

> Dependency versions are specified in `pyproject.toml`. `requirements.txt` uses version ranges suitable for general installation. `requirements.lock` pins exact versions used to produce the paper results.

---

## Quick Start

```python
from dash_shap import DASHPipeline
from dash_shap.experiments.synthetic import generate_synthetic_linear

# Generate synthetic data (P=20 features in 4 correlated groups of 5)
(X_train, y_train, X_val, y_val, X_explain, _,
 X_test, y_test, groups, true_importance, _) = generate_synthetic_linear(
    N=2000, P=20, group_size=5, rho=0.9, seed=42
)

pipe = DASHPipeline(
    M=200,                      # Train 200 diverse models
    K=30,                       # Select up to 30 for consensus
    epsilon=0.08,               # Keep models within 0.08 of best score
    selection_method="maxmin",  # Greedy diversity selection
    task="regression",          # "regression", "binary", or "multiclass"
)

# Fit (use a held-out explain set as X_ref, separate from X_test)
pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

# Consensus feature importance
importance = pipe.global_importance_
ranking = pipe.get_importance_ranking()

# Diagnostics — see docs/DIAGNOSTICS.md for interpretation
fig = pipe.plot_importance_stability()   # IS plot: 4 quadrants of feature behavior

fsi = pipe.get_fsi()
print(fsi.summary(top_k=10))                # Importance + stability per feature

# Quadrant labels: which features are stable vs. collinear?
labels = fsi.get_quadrant_labels()
robust = [f for f, l in zip(fsi.feature_names, labels) if "Robust" in l]

# Local disagreement map for the highest-variance observation
from dash_shap.core.diagnostics import local_disagreement_map
import numpy as np
var_per_obs = np.mean(pipe.variance_matrix_, axis=1)
fig = local_disagreement_map(                # Bar chart with cross-model error bars
    pipe.all_shap_matrices_, np.argmax(var_per_obs),
    feature_names=pipe.feature_names_,
)

# Ensemble predictions
predictions = pipe.get_consensus_ensemble_predictions(X_test)
```

Diagnostic interpretation: **[Diagnostics Guide](docs/DIAGNOSTICS.md)** | Full API: **[API Reference](docs/API_REFERENCE.md)**

---

## Reproducing Paper Results

See **[REPRODUCE.md](REPRODUCE.md)** for the complete reproduction guide, including hardware requirements, estimated runtimes, and step-by-step verification instructions.

```bash
# Exact paper environment (pinned versions)
pip install -r requirements.lock
pip install -e .

# Run all 20 experiments (~6–10 hours on 72-vCPU instance)
python run_experiments_parallel.py

# Run a single experiment
python run_experiments_parallel.py --experiments linear_sweep

# Run the test suite
pytest -m "not slow"
```

**Canonical notebooks:**
- [`demo_benchmark_6.ipynb`](notebooks/demo_benchmark_6.ipynb) — ArXiv canonical results (M=200, K=30, 20 reps)
- [`demo_benchmark_7_parallel.ipynb`](notebooks/demo_benchmark_7_parallel.ipynb) — TMLR canonical notebook
- [`explore_experiment_results.ipynb`](notebooks/explore_experiment_results.ipynb) — Interactive results viewer

---

## How It Works

DASH is a five-stage pipeline:

1. **Population** — Train M independent XGBoost models with random hyperparameters and low `colsample_bytree` (0.1–0.5), forcing each model to explore different features
2. **Filtering** — Keep models within ε of the best validation score
3. **Diversity Selection** — Greedy MaxMin selection maximizing minimum pairwise cosine distance among importance vectors
4. **Consensus** — Interventional TreeSHAP on each selected model, then element-wise mean
5. **Diagnostics** — Feature Stability Index (FSI) and Importance-Stability plots for auditing explanations without ground truth

After fitting, `pipe.result_` is a `DASHResult` — a lightweight container for the
K×N'×P SHAP tensor. The **extensions framework** (`dash_shap.extensions`) adds
confidence intervals, partial orders, robust certification, and more on top of any
`DASHResult`. See the extensions module (`dash_shap/extensions/`) for details.

The key insight: model independence — not model size or count — is what cancels the arbitrary noise. Stochastic Retrain (same hyperparameters, different seeds) achieves equivalent stability, confirming that independence is the operative principle. DASH adds diversity selection and diagnostics on top.

See the **[paper](https://arxiv.org/abs/2603.22346)** for the full mechanism analysis and the Large Single Model experiment that proves bigger models make the problem worse.

---

## Project Structure

```
dash-shap/
├── dash_shap/
│   ├── core/           # Five-stage pipeline (population, filtering, diversity, consensus, diagnostics)
│   ├── extensions/     # Analysis beyond consensus mean (CI, partial orders, certification, ...)
│   ├── baselines/      # 9 comparison methods
│   ├── experiments/    # Synthetic data generators (linear & nonlinear DGP)
│   ├── evaluation/     # Metrics: stability, DGP agreement, equity, statistical tests
│   └── utils/          # I/O and SHAP helpers
├── notebooks/          # Interactive benchmarks (demo_benchmark_6 is canonical)
├── examples/           # Standalone usage examples (quickstart.py, extensions_quickstart.py)
├── docs/               # API_REFERENCE.md, BENCHMARK_RESULTS.md, DIAGNOSTICS.md
├── tests/              # pytest suite
├── paper/              # LaTeX source
├── run_experiments_parallel.py  # CLI experiment runner (canonical)
├── run_experiments.py           # Sequential runner (deprecated — retained for provenance)
├── REPRODUCE.md        # Step-by-step reproduction guide
├── requirements.lock   # Pinned exact versions for experiment reproduction
├── EXPERIMENT_GUIDE.md # Full methodology and method descriptions
├── FAQ.md              # Common questions and answers
└── CONTRIBUTING.md     # How to contribute and add extensions
```

---

## Research Program

DASH is Paper 1 of a five-paper research program on trustworthy feature attribution:

| Paper | Topic | Venue | Repo |
|---|---|---|---|
| **1. DASH** (this repo) | Method + empirical validation | TMLR | [dash-shap](https://github.com/DrakeCaraker/dash-shap) |
| **3. Attribution Impossibility** | Formally verified impossibility theorem (Lean 4) | NeurIPS 2026 | [dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean) |

The impossibility theorem proves that no single-model feature ranking can simultaneously be faithful, stable, and complete when features are collinear — formalizing the theoretical foundation for why DASH's ensemble approach is necessary.

---

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

---

## License

[MIT](LICENSE)

