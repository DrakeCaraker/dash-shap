# Research & Benchmarks

This document contains the full research context for DASH — the problem mechanism, benchmark results, method comparisons, and reproduction instructions. For the practitioner guide, see the [README](../README.md).

> Caraker, Arnold, Rhoads (2026). *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* [arXiv:2603.22346](https://arxiv.org/abs/2603.22346) | [Zenodo](https://doi.org/10.5281/zenodo.19060132). Target venue: TMLR.

---

## The Problem

Feature importance explanations are fragile under collinearity: when features are correlated, the explanation changes depending on which feature the model happens to use first. Train the same model twice with a different random seed and you can get completely different importance rankings — even though the predictions are identical. In regulated, high-stakes, or scientific settings, this means the "explanation" is partially arbitrary.

This instability comes from **sequential residual dependency** in iterative model fitting — proved for gradient boosting (XGBoost, LightGBM, CatBoost), Lasso, and neural networks. The intuitive fix (bigger models) makes it *worse*. A single large model with the same total tree count as DASH produces the worst explanations of any method tested, because more sequential fitting means more reinforcement of an arbitrary initial feature choice.

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
| Superconductor | 81 | 0.840 | 0.964 | 0.924 | +0.124 |
| California Housing | 8 | 0.969 | 0.978 | 0.977 | +0.009 (n.s.) |

*SR = Stochastic Retrain (seed averaging).*

DASH stability is flat across correlation levels (0.973–0.977 from ρ = 0.0 to ρ = 0.95). On Breast Cancer — the most extreme collinearity case — DASH outperforms Stochastic Retrain by +0.063, the largest DASH-SR gap in any experiment.

*All results from 50-rep SageMaker run (PAPER_CONFIG: M=200, K=30, ε=0.08, δ=0.05). See [Benchmark Results](BENCHMARK_RESULTS.md) for full tables.*

Full results: **[Benchmark Results](BENCHMARK_RESULTS.md)** | Methodology: **[Experiment Guide](../EXPERIMENT_GUIDE.md)**

---

## When to Use DASH

### Detecting the problem

Before trusting a feature ranking from any iterative model, train multiple models with different random seeds and compare their importance rankings. If rankings differ substantially, your explanations are affected by **first-mover bias** — the sequential residual fitting arbitrarily concentrates importance on whichever correlated feature the model happens to use first.

### When DASH is most valuable

Use DASH when:

- **Features are correlated** — The stability advantage is statistically significant at ρ ≥ 0.7 and grows with correlation severity. At ρ = 0.9, DASH improves stability from 0.958 to 0.977 over the single-best baseline.
- **Explanation stability matters** — In regulated, high-stakes, or scientific settings where explanations must be reproducible across model retrains.
- **Equitable credit distribution is needed** — DASH distributes importance proportionally across correlated feature groups (within-group CV = 0.175) rather than concentrating it on an arbitrary group member.
- **You need to audit explanations without ground truth** — DASH's Feature Stability Index (FSI) and Importance-Stability (IS) plots detect which specific features are affected by first-mover bias. Quadrant II features (high importance, high instability) should be interpreted as collinear cluster members rather than individually important features.

### When simpler alternatives suffice

**Stochastic Retrain** (same hyperparameters, different seeds) achieves stability equivalent to DASH in the linear regime (~0.977 at ρ = 0.9) with minimal implementation effort. Use it when diagnostics and equity are not required. Under nonlinear DGPs at ρ ≥ 0.9, DASH significantly outperforms SR (0.887 vs 0.857; bootstrap CIs non-overlapping) — hyperparameter diversity matters when models learn qualitatively different functional forms.

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

---

## Reproducing Paper Results

See **[REPRODUCE.md](../REPRODUCE.md)** for the complete reproduction guide, including hardware requirements, estimated runtimes, and step-by-step verification instructions.

```bash
# Exact paper environment (pinned versions)
pip install -r requirements.lock
pip install -e .

# Run all 18 default experiments (~6–10 hours on 72-vCPU instance)
python run_experiments_parallel.py

# Run a single experiment
python run_experiments_parallel.py --experiments linear_sweep

# Run the test suite
pytest -m "not slow"
```

**Canonical notebooks:**
- [`demo_benchmark_6.ipynb`](../notebooks/demo_benchmark_6.ipynb) — ArXiv canonical results (M=200, K=30, 20 reps)
- [`demo_benchmark_7_parallel.ipynb`](../notebooks/demo_benchmark_7_parallel.ipynb) — TMLR canonical notebook
- [`explore_experiment_results.ipynb`](../notebooks/explore_experiment_results.ipynb) — Interactive results viewer

---

## Research Program

DASH is Paper 1 of a five-paper research program on trustworthy feature attribution:

| Paper | Topic | Venue | Repo |
|---|---|---|---|
| **1. DASH** (this repo) | Method + empirical validation | TMLR | [dash-shap](https://github.com/DrakeCaraker/dash-shap) |
| **3. Attribution Impossibility** | Formally verified impossibility theorem (Lean 4) | NeurIPS 2026 | [dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean) |
| **4. Ostrowski Impossibility** | Bilemma for binary explanation spaces | Letters in Mathematical Physics | [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) |

The impossibility theorem (Paper 3) proves that no single-model feature ranking can simultaneously be faithful, stable, and complete when features are collinear — formalizing the theoretical foundation for why DASH's ensemble approach is necessary. The companion paper further proves:

- **Bilemma**: Even for binary sign attribution (is this feature's effect positive or negative?), faithfulness and stability cannot coexist — completeness is not needed to generate the impossibility.
- **Enrichment resolution**: Collapsed tightness (the binary sign case) can *only* be resolved by enrichment — expanding the output space to include a neutral element (ties). DASH's consensus averaging is precisely this enrichment, making it the unique structural resolution class.
- **Approximate bilemma**: The impossibility survives ε-approximation at every tolerance level, with a quantitative bound: unfaith₁ + unfaith₂ ≥ Δ − δ.
- **DASH optimality**: Consensus averaging is the minimum-variance unbiased estimator (Cramér-Rao bound σ²/M) and is Pareto-optimal among all stable attribution methods.

The Ostrowski impossibility (Paper 4) extends the bilemma to physics: Ostrowski's classification of absolute values on ℚ creates a binary partition (archimedean vs p-adic) that triggers the same impossibility for spacetime geometry explanations, resolved by the adelic framework — the physics analogue of DASH.

---

## Project Structure

```
dash-shap/
├── dash_shap/
│   ├── core/           # Five-stage pipeline (population, filtering, diversity, consensus, diagnostics)
│   ├── extensions/     # Analysis beyond consensus mean (CI, partial orders, certification, theory bridge)
│   ├── baselines/      # 9 comparison methods
│   ├── experiments/    # Synthetic data generators (linear & nonlinear DGP)
│   ├── evaluation/     # Metrics: stability, DGP agreement, equity, statistical tests
│   └── utils/          # I/O and SHAP helpers
├── notebooks/          # Interactive benchmarks (demo_benchmark_7_parallel is canonical for TMLR)
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
