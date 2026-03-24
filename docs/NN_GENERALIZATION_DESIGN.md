# NN Generalization: Research Design

> **Status:** Phase 0 (infrastructure + proof of concept)
> **Branch:** `claude/assess-nn-generalization-kq7pK`
> **Target:** Paper 3 in the DASH research program (after TMLR submission)

## 1. Hypothesis

DASH's independence principle transfers to neural networks because attribution instability under multicollinearity is a universal phenomenon — not specific to gradient boosting.

**XGBoost instability** arises from *first-mover bias*: sequential residual fitting means whichever correlated feature happens to split first captures all the signal. The remaining features appear unimportant despite being equally predictive.

**NN instability** arises from *optimization path dependence*: random initialization maps features to neurons arbitrarily. Under collinearity, which correlated feature gets higher weight depends entirely on the initialization and SGD trajectory. Different seeds → different local minima → different feature attributions for equivalent features.

**Core prediction:** Averaging attributions across independently initialized NNs should cancel path-dependent noise, just as averaging across independently trained XGBoost models cancels first-mover bias.

### Three Publishable Findings

1. **Independence principle is universal.** Model-family-agnostic proof that averaging attributions across independently trained models improves stability under collinearity. Different mechanisms, same cure.

2. **NNs have natural attribution diversity.** Trees need explicit diversity forcing (`colsample_bytree ∈ [0.1, 0.5]`) to break sequential correlation. NNs get diversity for free from random initialization → different local minima.

3. **Diversity selection may be unnecessary for NNs.** If DASH Stage 3 (MaxMin greedy selection) adds nothing over naive averaging for NNs, it confirms that NN diversity is intrinsic — a qualitatively different regime from trees.

## 2. Architecture: What's Already Model-Agnostic

DASH's existing infrastructure handles Stages 2–5 with zero modification:

| Component | Why It Works for NNs |
|-----------|---------------------|
| `fit_from_attributions()` (pipeline.py:422–494) | Accepts pre-computed `(M, n_ref, P)` matrices + val_scores |
| Filtering (filtering.py) | Operates on scalar val_scores |
| Diversity selection (diversity.py) | MaxMin on cosine distance of importance vectors |
| Consensus (consensus.py) | Simple matrix averaging |
| Diagnostics (diagnostics.py) | FSI, variance — all numerical |
| `DASHResult` container | Model-agnostic, serializable |
| Evaluation metrics (stability, DGP agreement, equity) | All rank/correlation based |
| Synthetic DGPs (experiments/synthetic.py) | Return `(X, y, groups, true_importance)` — no model coupling |

**Integration path:** Train NN population → compute attributions → feed to `fit_from_attributions()` → done.

## 3. What Needs to Be Built

### 3.1 NN Population Training (`dash_shap/core/nn_population.py`)

Trains M sklearn `MLPRegressor`/`MLPClassifier` models with randomized hyperparameters.

**Search space:**
```
hidden_layer_sizes: [(64,64), (128,64), (128,128), (256,128), (128,128,64)]
alpha:              [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
learning_rate_init: [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
activation:         [relu, tanh]
batch_size:         [32, 64, 128, 256]
```

**Feature masking** (optional ablation): Random binary mask per model, each feature kept with probability `(1 - feature_mask_fraction)`. Direct analogue of `colsample_bytree`. Allows testing whether NNs need explicit diversity forcing.

### 3.2 NN Attribution (`dash_shap/core/nn_attribution.py`)

KernelSHAP via `shap.KernelExplainer` — already a project dependency. Returns `(consensus, all_shap)` matching `compute_consensus()` signature.

Future: `method='gradient'` and `method='ig'` stubs raise `ImportError` until PyTorch+Captum are added in Phase 1.

### 3.3 NN Baselines (`dash_shap/baselines/nn_baselines.py`)

| Baseline | XGBoost Analogue | What It Measures |
|----------|-----------------|------------------|
| `SingleNNBaseline` | `SingleBestBaseline` | Standard practice: best-of-N NNs |
| `BaggedNNBaseline` | `StochasticRetrainBaseline` | Seed diversity alone (same config, different seeds) |

Phase 1 additions (not in this branch):
- `MCDropoutBaseline` — single NN, T stochastic forward passes with dropout enabled
- `DeepEnsembleBaseline` — K diverse-config NNs (Lakshminarayanan et al. 2017)

### 3.4 Phase 0 Validation Notebook (`notebooks/nn_validation_phase0.ipynb`)

Skeleton notebook for go/no-go decision. Config: M=50, N_REPS=5, ρ=0.9.

## 4. Experimental Design

### 4.1 Experiment Table

| Priority | ID | Question | Config |
|----------|-----|----------|--------|
| P0 | `nn_mvp` | Does averaging NN attributions improve stability at all? | M=50, N_REPS=5, ρ=0.9 |
| P1 | `nn_vs_xgb` | Does DASH-NN match DASH-XGBoost stability? | M=200, N_REPS=20 |
| P2 | `nn_diversity_ablation` | Does MaxMin selection help NNs, or is naive averaging sufficient? | M=200, N_REPS=20 |
| P3 | `nn_feature_mask_ablation` | Does input feature masking improve NN diversity? | M=200, N_REPS=20 |
| P4 | `nn_explainer_comparison` | KernelSHAP vs GradientSHAP vs IG — which works best? | M=100, N_REPS=10 |
| P5 | `nn_linear_sweep` | Full ρ sweep with NNs | M=200, N_REPS=50, ρ∈{0,0.5,0.7,0.9,0.95} |
| P6 | `nn_nonlinear_sweep` | Nonlinear DGP sweep | Same as P5 |
| P7 | `nn_asymmetric_dgp` | Does DASH-NN over-equalize on asymmetric DGP? | M=200, N_REPS=20 |
| P8 | `nn_real_datasets` | California Housing, Breast Cancer, Superconductor | M=200, N_REPS=20 |
| P9 | `nn_population_scaling` | How does stability scale with M? | M∈{25,50,100,200,400} |

### 4.2 Expected Results

**P0 (MVP):** DASH-NN stability at ρ=0.9 should be 0.93–0.97, vs SingleNN ~0.85–0.92. If this gap is <0.02, reassess.

**P2 (diversity ablation):** Hypothesis: MaxMin adds ≤0.01 for NNs (vs ~0.02–0.04 for trees). NNs have natural diversity from random init, so selection is redundant.

**P3 (feature mask ablation):** Hypothesis: Feature masking helps less than for trees. NNs already get feature-level diversity from the random weight initialization → gradient dynamics.

**P7 (asymmetric DGP):** DASH-NN should NOT over-equalize (same risk as DASH-XGBoost at high K). Feature masking could mitigate by reducing the number of models that see both correlated features.

### 4.3 Methods Compared Per Experiment

Each experiment compares these methods:
1. **DASH-NN** — full pipeline via `fit_from_attributions()`
2. **SingleNN** — best of N_TRIALS NNs (`SingleNNBaseline`)
3. **BaggedNN** — K same-config NNs (`BaggedNNBaseline`)
4. **DASH-XGBoost** — reference from existing experiments
5. **SingleBest-XGBoost** — reference from existing experiments

## 5. Design Decisions

### 5.1 sklearn First, PyTorch Later

Phase 0 uses `sklearn.neural_network.MLPRegressor` + `shap.KernelExplainer`. Zero new dependencies.

**Rationale:**
- sklearn is already in the dependency tree
- MLPRegressor supports all needed hyperparameter variation
- KernelSHAP is model-agnostic (only needs `.predict()`)
- Phase 1 adds PyTorch+Captum behind optional import guards for GradientSHAP/IG

### 5.2 Relative Epsilon for NN Filtering

Use `epsilon_mode='relative'` with `EPSILON=0.05` (same as real-world datasets). NN val scores have different scale/variance than XGBoost RMSE, so relative filtering is more robust.

### 5.3 Pre-Filter Optimization

Train M=200 NNs → epsilon-filter → compute attributions **only for survivors** → diversity + consensus. KernelSHAP is the bottleneck (~15–30s/model for P=50), so filtering before attribution saves ~50% compute.

Implementation: the `fit_from_attributions()` path doesn't support this (it expects all M attributions). Instead, use the explicit stage-by-stage approach:
1. `generate_nn_population()` → models, val_scores
2. `performance_filter(val_scores, epsilon)` → surviving indices
3. `compute_nn_attributions(models, surviving_indices, X_ref)` → only compute for survivors
4. Feed filtered results to pipeline

### 5.4 Input Feature Masking

Per-model random binary mask: each feature kept with probability `(1 - feature_mask_fraction)`, fraction ∈ [0.1, 0.5].

**Separate RNG stream** (seeded with `seed + i + 10000`) prevents mask randomness from perturbing model training seeds.

**Testable ablation:** Run P3 with `feature_mask_fraction=None` vs `0.3` vs `0.5`. If masking adds nothing, it confirms the natural-diversity hypothesis (Finding #2).

## 6. Compute Budget

| Component | Per-model | M=200 | Per-rep (M=200) | N_REPS=20 |
|-----------|-----------|-------|-----------------|-----------|
| NN training (sklearn MLP, P=50) | ~1–5s | ~3–15 min | ~3–15 min | ~1–5 hrs |
| KernelSHAP (P=50, nsamples=auto) | ~15–30s | ~50–100 min | ~50–100 min | ~17–33 hrs |
| Filtering + diversity + consensus | — | ~seconds | ~seconds | ~minutes |
| **Total per experiment** | — | — | **~1–2 hrs** | **~20–40 hrs** |

**Pre-filtering** cuts SHAP compute by ~50%. **Full P0–P9 suite:** ~150–300 CPU-hrs.

**Phase 1 optimization:** GradientSHAP (PyTorch) is ~10× faster than KernelSHAP, reducing full suite to ~15–30 CPU-hrs.

## 7. Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| NNs underfit on N=5000 synthetic data | Medium | Low | StandardScaler preprocessing; wider architecture search space; increase N to 10000 |
| KernelSHAP variance dominates model variance | High | Medium | Increase `nsamples`; switch to GradientSHAP (Phase 1); use larger background set |
| MC-Dropout beats DASH-NN | Medium | Low | Still publishable — shows when aggregation helps vs doesn't |
| NNs too slow for N_REPS=50 | Medium | Medium | Run P0 first (M=50, N_REPS=5); scale up only if positive signal |
| Feature masking degrades NN performance | Low | Medium | Feature masking is optional — ablation compares with vs without |
| sklearn MLP too limited (no BatchNorm, custom losses) | Low | Low | Phase 1 PyTorch migration addresses this |

## 8. Go/No-Go Criteria (Phase 0)

### Pass (proceed to full suite)
- DASH-NN stability > SingleNN stability by **≥0.02** at ρ=0.9
- DASH-NN DGP agreement within 0.05 of DASH-XGBoost

### Conditional Pass (proceed with modifications)
- DASH-NN stability improves but Δ < 0.02 → increase M or switch explainer
- DASH-NN stability ≈ BaggedNN → diversity selection unnecessary for NNs (still publishable as Finding #3)

### Fail (abandon NN extension)
- No improvement over SingleNN → independence principle doesn't transfer to NNs
- KernelSHAP variance so high that signal is lost (variance ratio >3)

## 9. Implementation Phases

### Phase 0: Proof of Concept (this branch)
- `nn_population.py` — sklearn MLP population training with feature masking
- `nn_attribution.py` — KernelSHAP computation
- `nn_baselines.py` — SingleNN, BaggedNN
- `test_nn_pipeline.py` — smoke tests
- `nn_validation_phase0.ipynb` — go/no-go validation notebook
- **Deliverable:** go/no-go decision based on P0 experiment

### Phase 1: Full Integration
- PyTorch + Captum support (optional dependency)
- GradientSHAP and Integrated Gradients methods
- MCDropout and DeepEnsemble baselines
- `run_nn_experiments.py` — standalone experiment runner
- Full experiment suite (P0–P9)

### Phase 2: Paper
- Write up results for Paper 3 in the research program
- Cross-model comparison tables (XGBoost vs NN vs RF)
- Theoretical analysis of universality

## 10. File Manifest

| File | Status | Description |
|------|--------|-------------|
| `docs/NN_GENERALIZATION_DESIGN.md` | This file | Research design document |
| `dash_shap/core/nn_population.py` | New | NN population training |
| `dash_shap/core/nn_attribution.py` | New | KernelSHAP for NNs |
| `dash_shap/baselines/nn_baselines.py` | New | SingleNN, BaggedNN baselines |
| `dash_shap/baselines/__init__.py` | Modified | Add lazy imports for NN baselines |
| `tests/test_nn_pipeline.py` | New | NN pipeline smoke tests |
| `notebooks/nn_validation_phase0.ipynb` | New | Phase 0 validation notebook |
