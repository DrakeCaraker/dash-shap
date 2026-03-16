# DASH: Diversified Aggregation of SHAP

<!-- TODO: Replace placeholder once ArXiv ID is assigned -->
[![ArXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stable feature importance explanations under collinearity via independent model aggregation.**

> Caraker, Arnold, Rhoads (2026). *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* ArXiv pre-print (2026). Target venue: TMLR.

---

## The Problem

SHAP is one of the most widely used tools for explaining machine learning predictions. But it has a hidden fragility: when features are correlated, the explanation changes depending on which feature the model happens to use first. Train the same XGBoost model twice with different random seeds and you can get completely different importance rankings — even though the predictions are identical. In regulated, high-stakes, or scientific settings, this means the "explanation" is partially arbitrary.

This instability comes from a specific mechanism — **sequential residual dependency** in gradient boosting — and the intuitive fix (bigger models) makes it *worse*. A single large model with the same total tree count as DASH produces the worst explanations of any method tested, because more sequential trees means more reinforcement of an arbitrary initial feature choice.

**DASH fixes this.** It trains many small models independently, so each one makes its own arbitrary choice about which correlated feature to use. When you average their explanations, the arbitrary noise cancels out and the signal remains — which *group* of features matters, with credit distributed fairly across the group.

---

## Key Results

At high collinearity (ρ = 0.9, 50 features, 10 correlated groups, 20 repetitions):

| Method | Stability | DGP Agreement | Equity (CV) |
|---|---|---|---|
| Single Best | 0.958 | 0.978 | 0.224 |
| Large Single Model | 0.938 | 0.967 | 0.262 |
| **DASH (MaxMin)** | **0.977** | **0.988** | **0.176** |

Stability on real-world datasets:

| Dataset | Features | Single Best | DASH | Improvement |
|---|---|---|---|---|
| Breast Cancer | 30 (21 pairs \|r\| > 0.9) | 0.317* | 0.930 | **+0.614** |
| Superconductor | 81 | 0.830 | 0.962 | +0.132 |
| California Housing | 8 | 0.967 | 0.982 | +0.015 |

*\*Tree-count-matched Single Best (M=200): trains 200 models, keeps the best. Standard Single Best uses default hyperparameter tuning.*

DASH stability is flat across correlation levels (0.972–0.977 from ρ = 0.0 to ρ = 0.95). All 11 pre-registered success criteria pass. Statistically significant at ρ ≥ 0.7 (Wilcoxon, Holm-Bonferroni corrected, Cohen's d > 1.0).

Full results: **[Benchmark Results](docs/BENCHMARK_RESULTS.md)** | Methodology: **[Experiment Guide](EXPERIMENT_GUIDE.md)**

---

## When to Use DASH

### Detecting the problem

Before trusting a SHAP-based feature ranking from a gradient-boosted model, train multiple models with different random seeds and compare their importance rankings. If rankings differ substantially, your explanations are affected by **first-mover bias** — the sequential residual fitting in gradient boosting arbitrarily concentrates importance on whichever correlated feature the model happens to use first.

### When DASH is most valuable

Use DASH when:

- **Features are correlated** — The stability advantage is statistically significant at ρ ≥ 0.7 and grows with correlation severity. At ρ = 0.9, DASH improves stability from 0.958 to 0.977 over the single-best baseline.
- **Explanation stability matters** — In regulated, high-stakes, or scientific settings where explanations must be reproducible across model retrains.
- **Equitable credit distribution is needed** — DASH distributes importance proportionally across correlated feature groups (within-group CV = 0.176) rather than concentrating it on an arbitrary group member.
- **You need to audit explanations without ground truth** — DASH's Feature Stability Index (FSI) and Importance-Stability (IS) plots detect which specific features are affected by first-mover bias. Quadrant II features (high importance, high instability) should be interpreted as collinear cluster members rather than individually important features.

### When simpler alternatives suffice

**Stochastic Retrain** (same hyperparameters, different seeds) achieves stability equivalent to DASH (~0.977 at ρ = 0.9) with minimal implementation effort. Use it when diagnostics and equity are not required.

### DASH vs. other methods

| Method | Stability | Equity | Diagnostics | Notes |
|---|---|---|---|---|
| **Single Best Model** | Unstable under collinearity (0.317 on Breast Cancer) | Poor — concentrates credit arbitrarily | None | Standard workflow; unreliable when features are correlated |
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
git clone https://github.com/DrakeCaraker/dash-shap.git
cd dash-shap
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python >= 3.9. No GPU needed.

| Package | Version | Purpose |
|---|---|---|
| xgboost | >= 2.0.0 | Gradient boosting models |
| shap | >= 0.44.0 | TreeExplainer for SHAP values |
| scikit-learn | >= 1.4.0 | Data splitting, metrics |
| numpy | >= 1.24.0 | Numerical computation |
| scipy | >= 1.11.0 | Statistical tests, clustering |
| joblib | >= 1.3.0 | Parallel model training |

---

## Quick Start

```python
from dash.core.pipeline import DASHPipeline

pipeline = DASHPipeline(
    M=200,                      # Train 200 diverse models
    K=30,                       # Select up to 30 for consensus
    epsilon=0.08,               # Keep models within 0.08 of best score
    selection_method="maxmin",  # Greedy diversity selection
    task="regression",          # "regression", "binary", or "multiclass"
)

# Fit (use a held-out explain set as X_ref, separate from X_test)
pipeline.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

# Consensus feature importance
importance = pipeline.global_importance_
ranking = pipeline.get_importance_ranking()

# Diagnostics
fig = pipeline.plot_importance_stability()   # Importance-Stability plot
fsi = pipeline.get_fsi()                     # Feature Stability Index

# Ensemble predictions
predictions = pipeline.get_consensus_ensemble_predictions(X_test)
```

Full API: **[API Reference](docs/API_REFERENCE.md)**

---

## Reproducing Paper Results

```bash
# Run all 10 experiments (correlation sweep, real-world, ablation, etc.)
python run_experiments.py

# Run a single experiment
python run_experiments.py --experiments linear_sweep

# Run the test suite
pytest
```

**Canonical notebooks:**
- [`demo_benchmark_6.ipynb`](notebooks/demo_benchmark_6.ipynb) — Authoritative results (M=200, K=30, 20 reps)
- [`demo_benchmark_7.ipynb`](notebooks/demo_benchmark_7.ipynb) — TMLR submission (in development)
- [`explore_experiment_results.ipynb`](notebooks/explore_experiment_results.ipynb) — Interactive results viewer

---

## How It Works

DASH is a five-stage pipeline:

1. **Population** — Train M independent XGBoost models with random hyperparameters and low `colsample_bytree` (0.1–0.5), forcing each model to explore different features
2. **Filtering** — Keep models within ε of the best validation score
3. **Diversity Selection** — Greedy MaxMin selection maximizing minimum pairwise cosine distance among importance vectors
4. **Consensus** — Interventional TreeSHAP on each selected model, then element-wise mean
5. **Diagnostics** — Feature Stability Index (FSI) and Importance-Stability plots for auditing explanations without ground truth

The key insight: model independence — not model size or count — is what cancels the arbitrary noise. Stochastic Retrain (same hyperparameters, different seeds) achieves equivalent stability, confirming that independence is the operative principle. DASH adds diversity selection and diagnostics on top.

See the **[paper](https://arxiv.org/abs/XXXX.XXXXX)** for the full mechanism analysis and the Large Single Model experiment that proves bigger models make the problem worse.

---

## Project Structure

```
dash-shap/
├── dash/
│   ├── core/           # Five-stage pipeline (population, filtering, diversity, consensus, diagnostics)
│   ├── baselines/      # 9 comparison methods
│   ├── experiments/    # Synthetic data generators (linear & nonlinear DGP)
│   ├── evaluation/     # Metrics: stability, DGP agreement, equity, statistical tests
│   └── utils/          # I/O and SHAP helpers
├── notebooks/          # Interactive benchmarks (demo_benchmark_6 is canonical)
├── docs/               # API_REFERENCE.md, BENCHMARK_RESULTS.md
├── tests/              # pytest suite (~47 tests)
├── paper/              # LaTeX source
├── run_experiments.py  # CLI experiment runner
├── EXPERIMENT_GUIDE.md # Full methodology and method descriptions
└── ROADMAP.md          # Five-paper research program
```

---

## Citation

<!-- TODO: Update with ArXiv ID once assigned -->
```bibtex
@article{caraker2026firstmover,
  title={First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution},
  author={Caraker, Drake and Arnold, Bryan and Rhoads, David},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## License

[MIT](LICENSE)

## Research Roadmap

DASH is Paper 1 of a five-paper research program extending from gradient boosting to neural networks, impossibility results, and explanation-aware model selection. See **[ROADMAP.md](ROADMAP.md)**.
