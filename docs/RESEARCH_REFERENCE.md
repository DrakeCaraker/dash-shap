# DASH Research Program — Comprehensive Reference

> **Purpose:** Exhaustive reference for all proved theorems, experimental results, validated findings, diagnostics, and their interconnections across the DASH research program. Written so that any session — human or AI — can understand what exists, why it matters, and where to find it.
>
> **Last updated:** 2026-04-30
>
> **Quick index:** [Theoretical Results](#theoretical-results) · [Experiments](#completed-experiments) · [Theory Bridge](#theory-bridge-validations) · [Diagnostics](#diagnostic-tools) · [Retracted](#retracted-results) · [API](#api-reference) · [Papers](#paper-drafts)

---

## The Core Claim

**SHAP feature importance rankings are partially random when features are correlated.** The randomness comes from sequential residual dependency ("first-mover bias") in iterative model fitting. DASH resolves this by averaging explanations across independently trained models. The companion papers prove this resolution is not just effective but *mathematically necessary* — no method can do better, and no other structural approach can achieve the same guarantees.

The research program spans three repositories:

| Repo | Scope | Status |
|------|-------|--------|
| [dash-shap](https://github.com/DrakeCaraker/dash-shap) | Method, experiments, PyPI package, TMLR paper | Under review |
| [dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean) | Attribution impossibility (ML-focused, Lean 4) | NeurIPS 2026 target |
| [ostrowski-impossibility](https://github.com/DrakeCaraker/ostrowski-impossibility) | Universal impossibility + physics (Lean 4) | FoP/Nature target |

---

## Theoretical Results

### The Attribution Impossibility (dash-impossibility-lean)

**357 theorems, 6 axioms, 0 sorry, 58 Lean files.**

#### The Core Theorem

`attribution_impossibility` (Trilemma.lean) — **zero domain axioms**.

No importance ranking of features (or internal components) can simultaneously be:
- **Faithful**: reflects the model's actual attributions
- **Stable**: consistent across equivalent models (Rashomon set)
- **Complete**: ranks all feature pairs

...when interchangeable components exist under the Rashomon property. This is not an empirical observation — it is a theorem that follows from definitions alone. It applies to SHAP, LIME, Integrated Gradients, attention maps, mechanistic interpretability, or any future attribution method.

**Why it matters:** Every practitioner using single-model SHAP is unknowingly choosing Family A (faithful + decisive, but unstable). The theorem says there is no Family C that avoids this tradeoff. The only escape is to change the output space — which is what DASH does.

#### The Bilemma

`bilemma_of_compatible_eq` (Bilemma.lean) — **zero axioms**.

Strengthens the trilemma for binary explanation spaces: even faithfulness + stability alone is impossible. You don't need to ask for completeness to get the impossibility. This matters because many practical tasks are binary:
- SHAP sign: is this feature's effect positive or negative?
- Feature selection: in or out?
- Counterfactual direction: increase or decrease this feature?

Three constructive ML instances are proved: `shap_sign_bilemma`, `feature_selection_bilemma`, `counterfactual_bilemma` — all using Bool/Unit types with zero domain axioms.

**Companion results:**
- `all_or_nothing` — explanations are either perfectly faithful (and maximally unstable) or perfectly stable (and maximally unfaithful). No smooth tradeoff — it's a cliff.
- `rashomon_unfaithfulness` — any stable method is wrong about at least one model in every Rashomon pair. The 50% floor is exact.

#### Tightness Classification

`tightness_dichotomy` (BeyondBinary.lean) — **zero axioms**.

Two algebraic properties of the explanation space completely determine which property pairs are achievable:

| | Neutral element exists | No neutral element |
|---|---|---|
| **Committal** | F+S achievable (trilemma tight) | F+S impossible (bilemma) |
| **Non-committal** | F+S+D achievable | Case-dependent |

Binary sign attribution is committal with no neutral element — the maximally constrained case. DASH's consensus averaging introduces the neutral element (continuous values with a zero point), moving from the bilemma cell to the trilemma cell.

**Diagnostic consequence:** `coverageConflict_implies_no_neutral` — coverage conflict is the empirical signature of collapsed tightness. If coverage conflict exists for a feature, that feature is in the bilemma regime.

#### Design Space Theorem

`design_space_theorem`, `family_a_or_family_b` (DesignSpace.lean, DesignSpaceFull.lean)

Every faithful attribution method falls into exactly one of two families:
- **Family A** (standard SHAP, single-model): Faithful + Decisive, but Unstable. You get a definite ranking that accurately reflects your model, but retrain and it changes. Empirically: ~33% of features flip across retrains.
- **Family B** (DASH): Faithful + Stable, but Indecisive. The consensus is stable across retrains, but for genuinely contested features, it reports the disagreement (ties / wide CIs) rather than forcing an arbitrary ranking.

There is no Family C. `no_complete_faithful_ranking` proves it.

**Why this matters for practitioners:** The choice isn't "SHAP vs something better." It's "accept instability (Family A) or accept indecision on contested features (Family B)." DASH makes the second choice explicit and provides diagnostics to tell you which features are contested.

#### DASH Optimality

`dash_unique_pareto_optimal`, `pareto_frontier_dichotomy` (ParetoOptimality.lean)

DASH's consensus averaging (element-wise mean across K models) is:
1. The **minimum-variance unbiased estimator** — Cramér-Rao bound σ²/M (`consensus_variance_from_independence` in VarianceDerivation.lean). No method achieves lower variance for the same number of independent models.
2. **Pareto-optimal** — no method simultaneously achieves zero within-group unfaithfulness and higher between-group stability.
3. The **unique** Pareto-optimal point — `dash_unique_pareto_optimal`. Not just good; the only non-dominated strategy.
4. Unbeatable by weighting — `weighted_variance_ge_consensus_variance`. No weighted average of model attributions beats equal weighting.
5. Variance halves with M — `double_M_halves_variance_derived`. Doubling the ensemble size halves the attribution variance.

**Provenance:** These are standard decision theory results (the mean minimizes MSE for independent samples) applied to the attribution setting. The novelty is proving they hold for the Rashomon/attribution framework specifically.

#### Model-Class Instantiation

| Model class | File | Key theorem | Rate | Implication |
|---|---|---|---|---|
| Gradient boosting (XGBoost) | General.lean, SplitGap.lean | `split_gap_exact` | 1/(1-ρ²) — divergent | Concentration grows without bound as ρ→1 |
| Lasso | Lasso.lean | — | ∞ (sign instability for all ρ>0) | Any nonzero correlation causes sign flips |
| Neural networks | NeuralNet.lean | — | Conditional, depends on initialization | Symmetry breaking via random init, not residual fitting |
| Random forests | RandomForest.lean | — | O(1/√T) — convergent | Independent by construction; bounded instability |

**Why RF is convergent but GBDT divergent:** RF trees are trained independently (bootstrap + random feature subsets). Each tree's arbitrary choice is independent of the others, so averaging cancels the noise. GBDT trees are sequential — each tree fits residuals from previous trees, creating a self-reinforcing bias toward whichever feature was selected first. The Large Single Model experiment (Experiment 8-9) confirms this empirically.

#### Quantitative Bounds

| Theorem | File | Statement | Practical use |
|---|---|---|---|
| `alpha_faithful_bound` | AlphaFaithful.lean | ε-approximate faithfulness bound | "Even approximately faithful" methods have the same impossibility |
| `stable_ranking_half_unfaithful` | AlphaFaithful.lean | Stable rankings are ≥50% unfaithful | Lower bound on error for any stable method |
| `attribution_prob_half` | UnfaithfulQuantitative.lean | Probability bound | Quantitative version of the 50% floor |
| `consensus_variance_from_independence` | VarianceDerivation.lean | Var = σ²/M | Ensemble size formula — how many models you need |
| `variance_ratio_from_independence` | VarianceDerivation.lean | Var ratio across M values | Diminishing returns quantified |

#### Extensions Beyond Attribution

| Domain | File | Key result | Status |
|---|---|---|---|
| Fairness auditing | FairnessAudit.lean | SHAP-based fairness audits fail under collinearity — audit outcome depends on which model was trained | Proved |
| Mechanistic interpretability | MechInterp.lean | ≥50% circuit disagreement for symmetric heads — activation patching inherits the impossibility | Proved |
| Model selection | ModelSelection.lean | Model selection as attribution instance | Proved |
| Causal discovery | CausalDiscovery.lean | Causal direction determination as attribution instance | Proved |
| Rashomon universality | RashomonUniversality.lean, RashomonInevitability.lean | The Rashomon property is generic, not pathological — almost all practical settings trigger it | Proved |

### The Universal Impossibility (ostrowski-impossibility)

**482 theorems, 13 axioms, 0 sorry, 38 Lean files.**

This repo proves the general framework and applies it beyond ML. Only results relevant to cross-session understanding are listed here; the full catalog is in the ostrowski repo's `docs/research-assessment.md`.

#### Approximate and Quantitative Bilemma

| Theorem | File | Statement |
|---|---|---|
| `approximate_bilemma` | ApproximateBilemma.lean | F+S incompatibility survives ε-approximation at every tolerance level |
| `quantitative_bilemma` | ApproximateBilemma.lean | unfaith₁ + unfaith₂ ≥ Δ − δ (triangle inequality, tight) |
| `exact_bilemma_from_quantitative` | ApproximateBilemma.lean | Exact bilemma is the Δ→0 limit of the quantitative version |

**Why it matters:** Preempts "what about approximately faithful methods?" The impossibility is not a knife-edge result. Relaxing faithfulness by any finite amount does not escape it.

#### Enrichment Forced Resolution

| Theorem | File | Statement |
|---|---|---|
| `forced_resolution_complete` | EnrichmentForcedResolution.lean | Collapsed tightness can ONLY be resolved by enrichment |
| `prime_collapsed_tightness` | EnrichmentForcedResolution.lean | Prime-indexed collapsed tightness |

**Why it matters:** This is the formal justification for DASH. The resolution taxonomy proves that adding a neutral element (ties/continuous values) is the *unique structural resolution class* for the bilemma. DASH's consensus averaging implements exactly this enrichment. No other approach — regularization, alternative attribution methods, conditional SHAP, etc. — can achieve F+S without enrichment.

#### MI Quantitative Bridge

| Theorem | File (universal repo) | Statement |
|---|---|---|
| `mi_implies_positive_gap` | MIQuantitativeBridge.lean | MI > 0 → ∃ Rashomon witnesses with opposite-sign attributions |
| `total_unfaithfulness_bound` | MIQuantitativeBridge.lean | Triangle inequality on unfaithfulness |
| `mi_quantitative_unfaithfulness` | MIQuantitativeBridge.lean | MI > 0 → any stable explanation has error ≥ Δ/2 |

**The chain:** Mutual information > 0 between features → Rashomon witnesses exist with opposite attributions → any stable explanation must be wrong by ≥ Δ/2 on at least one witness → this floor is unreachable (no algorithm can beat it) → DASH resolves by enrichment (the only structural escape).

**Why MI and not correlation:** MI captures ALL statistical dependence (including nonlinear). Pearson ρ and VIF miss nonlinear dependence entirely: X₂ = X₁² gives |ρ| ≈ 0, VIF ≈ 1, but MI = 1.91. The `mi_prescreen()` function in this repo implements this diagnostic.

#### Social Choice Parallels

- **Arrow's impossibility** — proved from scratch in Lean 4 (first Lean 4 proof; Lean 3 exists per Souther & Davidson 2021). IIA decomposition, 2 voters, 3 alternatives. Full tightness witnesses.
- **May's theorem** — majority rule for binary alternatives. Proved from scratch.
- **Direction theorem** — coverage conflict and neutral existence are anti-correlated. Binary tightens the bilemma but relaxes Arrow. This explains why Arrow needs ≥3 alternatives but the bilemma works with 2 — they're structurally opposite.

#### Cross-Domain Instances (Nature/FoP scope)

These results live in the ostrowski repo and are NOT part of the DASH or NeurIPS papers, but provide context:

- **Physics bilemma:** Ostrowski's classification of absolute values on ℚ creates a binary partition (archimedean/smooth vs p-adic/ultrametric) that triggers the bilemma for spacetime geometry. Resolved by the adelic framework (the physics analogue of DASH).
- **Enrichment stack:** The bilemma→enrichment pattern is recursive with proved unbounded depth (kthBit construction).
- **Gödel parallel:** Gödel's incompleteness derived from `hasGoedelProperty` (weaker than diagonal lemma). Both enrichment stack and Gödel instantiate the shared `RecursiveImpossibility` pattern.
- **Quantum contextuality:** Bell/CHSH and Kochen-Specker as impossibility instances (but NOT sheaf-theoretic contextuality — the bilemma is a simpler obstruction).
- **Navier-Stokes, circuit complexity, DPRM, Langlands:** Additional cross-domain instances demonstrating the framework's generality.

---

## Completed Experiments

All experiments use PAPER_CONFIG: M=200, K=30, N_REPS=50, ε=0.08, δ=0.05, SEED=42. Run on SageMaker ml.g5.16xlarge (64 CPU, 248 GB RAM). Results in `results/tables/*.json`. Runner: `run_experiments_parallel.py`.

### Experiment 1: Linear Correlation Sweep (THE central experiment)

**File:** `synthetic_linear_sweep.json` · **Script:** `run_experiments_parallel.py --experiments linear_sweep`

**Setup:** 50 features in 10 groups of 5, within-group correlation ρ ∈ {0.0, 0.5, 0.7, 0.9, 0.95}. Target is linear combination of group means (known ground truth). 9 methods × 5 ρ levels × 50 reps.

**Key results at ρ=0.9:**

| Method | Stability | Top-K5 | Equity (CV) |
|---|---|---|---|
| DASH (MaxMin) | **0.977** | **0.863** | **0.175** |
| Stochastic Retrain | 0.978 | 0.922 | 0.180 |
| Single Best | 0.958 | 0.546 | 0.232 |
| Large Single Model | 0.938 | 0.433 | 0.258 |

**What it proves:** DASH stability is flat (0.973-0.977) across all ρ levels. SB degrades from 0.972 to 0.952. LSM degrades from 0.955 to 0.927 — *worst* of any method despite matching DASH's total tree count. The advantage is specifically about collinearity: at ρ=0, all methods are equivalent.

**The SR equivalence:** SR ≈ DASH on stability (Δ ≤ 0.003, n.s. at every ρ). This is the paper's strongest finding — it proves the operative mechanism is model independence, not pipeline engineering. DASH's value over SR: speed (1.7×), diagnostics (FSI/IS Plot), and equity (p<0.001 vs Random Selection).

**Statistical support:** Bootstrap stability tests (p<0.001 for DASH vs SB and vs LSM). TOST equivalence confirmed for DASH vs SR. Wilcoxon signed-rank with Holm-Bonferroni correction on equity.

### Experiment 3: Nonlinear DGP Sweep

**File:** `nonlinear_sweep.json`

**Setup:** Nonlinear DGP with quadratic terms, interactions (z₁ × z₂), sinusoidal component. 5 methods × 5 ρ levels × 50 reps.

**Key result:** At ρ=0.9, DASH significantly outperforms SR (0.887 vs 0.857; bootstrap CIs non-overlapping). At ρ≤0.5, SR marginally beats DASH. This proves that under nonlinearity, *how* independence is achieved matters — hyperparameter diversity (DASH) outperforms seed-only diversity (SR) when models learn qualitatively different functional forms.

**Caveat:** Ground truth is approximate for nonlinear DGPs (C4 caveat). The nonlinear experiment evaluates primarily stability and equity, not accuracy against ground truth.

### Experiments 4a-c: Real-World Datasets

| Dataset | Features | DASH | SB | Improvement | File |
|---|---|---|---|---|---|
| **Breast Cancer** | 30 (21 pairs \|r\|>0.9) | 0.925 | 0.376 | **+0.549** | `breast_cancer.json` |
| Superconductor | 81 | 0.964 | 0.840 | +0.124 | `superconductor.json` |
| California Housing | 8 | 0.978 | 0.969 | +0.009 (n.s.) | `california_housing.json` |

**Breast Cancer** is the strongest result — the most extreme natural collinearity. DASH-SR gap is +0.063, largest in any experiment. SB(M=200) < SB(30) — model selection instability *increases* with population size under extreme collinearity.

**Superconductor:** RS (0.968) and NTN (0.976) slightly beat DASH (0.964). With 81 features, natural diversity is high enough that MaxMin selection adds diminishing value.

**California Housing:** Not statistically significant (p=0.063). Only 8 features, mild collinearity. Correctly acknowledged as a weak result.

### Experiment 13: Crossed Variance Decomposition (ANOVA)

**File:** `variance_decomposition_crossed.json`

**Setup:** 7×7 factorial (7 data seeds × 7 model seeds = 49 cells), two-way ANOVA at ρ=0.9.

**Result:**

| Method | Data % | Model % | Residual % |
|---|---|---|---|
| Single Best | 37.6 | **40.6** | 21.8 |
| DASH | **73.6** | 16.2 | 10.2 |

**Why it matters:** This is the strongest mechanistic evidence. SB's instability is dominated by model-selection noise (40.6%). DASH shifts the budget to data-dominated (73.6%) by canceling the path-dependent component. The 60% reduction in model-selection variance directly confirms the first-mover bias hypothesis.

### Experiment 15: Colsample Ablation

**File:** `colsample_ablation.json`

**Key finding:** DASH Low colsample (0.1-0.5) achieves 0.976 stability at ρ=0.9. DASH High colsample (0.5-1.0) achieves only 0.953 — comparable to Single Best (0.958). The effect disappears at ρ=0.0 (p=0.54). This isolates forced feature restriction as the operative diversity mechanism.

**Under nonlinearity:** The gap widens — Low=0.885 vs High=0.801 (Δ=+0.084). Low colsample is even more important when models learn different interaction structures.

### Other Experiments (abbreviated)

| # | Experiment | File | One-line finding |
|---|-----------|------|-----------------|
| 2 | Overlapping correlation | `overlapping.json` | DASH's largest advantage: +0.079 stability over SB |
| 5 | Epsilon sensitivity | `epsilon_sensitivity.json` | Stability varies <0.005 across ε∈{0.03-0.10}. Robust. |
| 6 | Ablation (M, K, ε, δ) | `ablation.json` | M insensitive past 100; K saturates at 20; δ sensitive above 0.05 |
| 7 | Variance decomposition (marginal) | `variance_decomposition.json` | Directional evidence consistent with crossed ANOVA |
| 8 | First-mover visualization | `first_mover_visualization.json` | SB/LSM concentrate credit; DASH distributes within group |
| 9 | First-mover bias isolation | `first_mover_bias.json` | Concentration grows with tree count, converges at M≥500 |
| 10 | Table 2 extended baselines | `table2_baselines.json` | 5 additional methods at ρ=0.9 |
| 11 | Background sensitivity | `background_sensitivity.json` | Stability Δ<0.0002 across B∈{50-500}. Not critical. |
| 12 | Asymmetric DGP | `asymmetric_dgp.json` | DASH has highest passive leak (0.089 vs SB 0.068) — equity tradeoff |
| 14 | K sweep independence | `k_sweep_independence.json` | Stability plateaus at K≈20. DASH fails at K=1. |
| 17 | High-dimensional scaling | PENDING | Deferred to future work |

---

## Theory Bridge Validations

These experiments validate predictions from the impossibility theorems against empirical data from the DASH pipeline. Scripts in `theory_bridge/`.

### Coverage Conflict (VALIDATED — HIGH confidence)

**Script:** `test_coverage_conflict.py` · **Evidence:** Spearman 0.59-0.98, 4 model classes, 3 datasets

Coverage conflict (fraction of models assigning conflicting signs to a feature) predicts per-feature sign instability. Model-class-universal: mean ρ = 0.82 (excluding degenerate California linear). The nonparametric predictor (based on coverage conflict + minority fraction) outperforms the Gaussian flip formula on weakly correlated data (California: 0.96 vs 0.46) but loses on strongly correlated data (Breast Cancer: 0.45 vs 0.93).

**Theoretical grounding:** Coverage conflict is the empirical signature of collapsed tightness (`coverageConflict_implies_no_neutral`). Features with coverage conflict are in the bilemma regime — no method can rank them stably.

### Flip-Rate Bimodality (VALIDATED synthetic, NUANCED real)

**Script:** `test_bimodality.py` · **Evidence:** Dip p < 0.002 at ρ ≥ 0.5 (synthetic); p = 0.575 on California Housing

The all-or-nothing theorem predicts SHAP value distributions should be bimodal for correlated features. Confirmed on synthetic data with permutation control (8-22% rejection rate validates the dip test isn't too liberal). NOT confirmed on California Housing — an honest negative that was reported, not hidden.

### Var[SHAP] = DASH MSE (VALIDATED — HIGH confidence)

**Script:** `test_variance_bound.py` · **Evidence:** Mathematical identity, 0/12 violations across 4 model classes

The variance of SHAP values across independent models equals DASH's mean squared error. This is standard decision theory (the mean minimizes MSE) applied to the attribution setting. It confirms that DASH optimality isn't just proved in Lean but holds empirically with zero violations.

### Model-Class Structure (VALIDATED — HIGH confidence)

**Script:** `model_class_rigorous.py` · **Evidence:** Within-family ρ = 0.79-0.94; cross-family not significant

Instability has two components: (1) a universal data-determined core (shared stable features, detectable by coverage conflict) and (2) a model-family-specific pattern (which *unstable* features flip depends on the algorithm). Trees agree with trees (XGB-RF ρ = 0.79-0.92); linear agrees with linear (Ridge-LASSO ρ = 0.94); but cross-family agreement (linear-tree ρ = 0.19-0.57) does not exceed the permutation null.

### First-Mover SHAP Correlation (VALIDATED — HIGH synthetic, MEDIUM real)

**Script:** `eta_shap_correlation.py` · **Evidence:** ρ_SHAP < 0 for substitutable features

For genuinely substitutable features (features competing for the same predictive signal), cross-model SHAP values are negatively correlated (ρ_SHAP = -0.11 to -0.24 on synthetic data; -0.19 for Breast Cancer texture mean/worst). This is the mechanism behind first-mover bias: when one model assigns high credit to feature A, a different model assigns high credit to feature B. The criterion is exchangeability (features compete for the same signal), not just correlation.

### MI Boundary Test (VALIDATED — HIGH confidence)

**Script:** `mi_only_dependence_test.py` · **Evidence:** MI = 1.91 catches X₂=X₁² that ρ=0.08 and VIF=1.008 miss

Mutual information detects nonlinear dependence that all correlation-based diagnostics miss. On Breast Cancer, MI-based pre-screening identifies 292/435 feature pairs as structurally dependent despite low linear correlation. On drug discovery (BBBP), MI reduces prediction error from 23pp to ~4pp vs correlation-only screening.

### Information-Theoretic Predictors (VALIDATED as negative — HIGH confidence)

**Script:** `info_theoretic_validation.py` · **Evidence:** 7 predictors tested, max Spearman 0.26

No data-only formula exists for predicting SHAP instability in tree models. Seven predictors tested (mutual information, conditional entropy, interaction information, etc.) — all weak (max Spearman 0.26). The extension works conceptually (redundancy → instability) but not quantitatively as a magnitude predictor. MI works as a *binary boundary* (I > 0 → instability possible) but not as a *magnitude predictor* (I does not predict how much instability).

---

## Retracted Results

These results were produced during the research but subsequently invalidated by controls. They are listed here to prevent re-use.

| Result | Why retracted | How caught |
|--------|---------------|------------|
| Entropy bimodality | 100% permutation artifact from entropy function discretization | Permutation control |
| Pairwise "audit pairs" recommendation | Marginal rates suffice; no pair-specific signal above marginal | Permutation control |
| η = 1/g from correlation thresholds | Inverts reality for XGBoost+SHAP (correlated features are MORE stable, not less) | Wrong test corrected |
| Data-only instability prediction | 7 predictors tested, max Spearman 0.26 | Systematic evaluation |
| Phase transition in stability curve | Gradual transition, not sharp | Visual inspection + quantitative check |
| "Model-class dependence" (original framing) | Confounded by λ-sweep vs seed-variation comparison | Rigorous re-test |
| p/2 average unfaithfulness bound | Corrected to p · mean_minority_fraction (≈ 14-19%, not 16.5%) | Arithmetic check |

### Nuanced Results (use with caveats)

| Result | Caveat |
|--------|--------|
| Flip-rate bimodality on real data | NOT confirmed on California Housing (p = 0.575) |
| Spectral predictors for Ridge SHAP | Tautological when using ridge-specific predictors; non-tautological ones are weak |
| Cross-family disagreement | "Not significant agreement" ≠ "proved disagreement" — absence of evidence, not evidence of absence |
| California Housing linear models | Degenerate: 7/8 features at zero flip rate |

---

## Diagnostic Tools

### Three-Level Hierarchy

| Level | Question | Tool | What it tells you |
|---|---|---|---|
| **Structure** | What compromises are forced? | Tightness classification (Lean) | F+S impossible for binary; enrichment is the only resolution |
| **Existence** | Does this dataset trigger it? | `mi_prescreen(X)` | Which feature pairs share MI — even nonlinear dependence |
| **Magnitude** | How much instability, per feature? | `coverage_conflict()` + FSI | Per-feature flip rates and importance variance |

### Pre-Pipeline Diagnostics

**`mi_prescreen(X)`** — Pairwise mutual information with permutation-based null threshold. Flags "hidden" pairs where MI > threshold but |Pearson ρ| < 0.7. These are dependencies that standard correlation diagnostics miss entirely. Returns MI matrix, correlation matrix, threshold, and list of hidden pairs.

**When to use:** Before fitting a DASH pipeline. If `n_hidden == 0`, your features may not need DASH. If `n_hidden > 0`, each hidden pair is subject to the MI quantitative bridge error floor (Δ/2).

### Post-Pipeline Diagnostics

**Feature Stability Index (FSI)** — `pipeline.get_fsi()`. Cross-model disagreement relative to signal strength (coefficient of variation). FSI < 0.3 = stable; FSI > 0.7 = likely collinear; FSI > 1.0 = disagreement exceeds signal.

**Importance-Stability (IS) Plot** — `pipeline.plot_importance_stability()`. 2D scatter: consensus importance (x) vs FSI (y). Four quadrants:

| Quadrant | Importance | Stability | Action |
|---|---|---|---|
| I: Robust Drivers | High | Stable | Report individually |
| II: Collinear Cluster | High | Unstable | Report as group |
| III: Unimportant | Low | Stable | Omit |
| IV: Fragile | Low | Unstable | Investigate |

**Coverage Conflict** — `coverage_conflict(all_shap_matrices)`. Per-(observation, feature) sign agreement across models. Minority fraction in [0, 0.5]: 0 = unanimous, 0.5 = coin flip. Distribution-free predictor of sign instability.

**SHAP Residual** — `shap_residual(all_shap_matrices, groups)`. Within-group residual from group mean (|SHAP_r|). High residuals = first-mover bias actively concentrating credit on arbitrary group members.

**Local Disagreement Map** — `local_disagreement_map(all_shap_matrices, idx)`. Per-observation SHAP values with ±1 SD error bars across K models. Shows which features are contested for a specific observation.

---

## API Reference

### Core Pipeline

| Function / Class | Module | Purpose |
|---|---|---|
| `check(X, y)` | `dash_shap` | 3-line stability check — trains M=25 models, returns CheckResult |
| `DASHPipeline` | `dash_shap.core.pipeline` | Full 5-stage pipeline: `.fit()`, `.fit_from_population()`, `.fit_from_attributions()` |
| `compute_consensus(aggregation='mean'\|'pca')` | `dash_shap.core.consensus` | Stage 4 with optional PCA aggregation for opposite-directional features |
| `generate_model_population()` | `dash_shap.core.population` | Stage 1: train M models with random hyperparameters |
| `performance_filter()` | `dash_shap.core.filtering` | Stage 2: keep models within ε of best |
| `greedy_maxmin_selection()` | `dash_shap.core.diversity` | Stage 3: MaxMin diversity selection |

### Diagnostics

| Function / Class | Module | Purpose |
|---|---|---|
| `mi_prescreen(X)` | `dash_shap.core.diagnostics` | Pre-pipeline: pairwise MI with permutation threshold |
| `FeatureStabilityIndex` | `dash_shap.core.diagnostics` | FSI computation and summary |
| `ImportanceStabilityPlot` | `dash_shap.core.diagnostics` | IS Plot with quadrant classification |
| `coverage_conflict()` | `dash_shap.core.diagnostics` | Sign-stability diagnostic |
| `compare_flip_predictors()` | `dash_shap.core.diagnostics` | CC vs Gaussian formula comparison |
| `predict_sign_instability()` | `dash_shap.core.diagnostics` | Per-feature binary stability flag |
| `has_coverage_conflict()` | `dash_shap.core.diagnostics` | Per-feature coverage conflict check |
| `shap_residual()` | `dash_shap.core.diagnostics` | Within-group |SHAP_r| metric |
| `local_disagreement_map()` | `dash_shap.core.diagnostics` | Per-observation SHAP with error bars |

### Extensions (12 total)

| Extension | Purpose | Key output |
|---|---|---|
| `confidence_intervals` | BCa bootstrap CI for importance, FSI, rank | Confidence intervals with coverage |
| `partial_order` | π(A>B) — fraction of models ranking A above B | Partial order DAG |
| `feature_groups` | Cluster features by SHAP substitutability | Group assignments |
| `stable_feature_selection` | Importance+stability composite ranking | Selected feature set |
| `local_uncertainty` | Per-observation K×P slice with sign-flip rate | Local stability map |
| `robust_certification` | Worst-case top-k guarantee across ALL models | Certified feature set |
| `theory_bridge` | SNR, predicted flip rates, M recommendation | TheoryBridgeResult |
| `causal_flags` | Label each feature: robust / collinear / fragile / unimportant | CausalResult |
| `audit_report` | Structured stakeholder report with warnings | AuditResult |
| `DriftMonitor` | Cosine distance between model versions | DriftAlert |
| `ParetoSelector` | RMSE-stability Pareto frontier optimization | ParetoFrontier |
| `federated_consensus` | Cross-site consensus without sharing data | FederatedResult |

### Baselines (11 total)

SingleBest, SingleBest(M=200), LargeSingleModel, LSM(Tuned), EnsembleSHAP, StochasticRetrain, RandomSelection, NaiveTopN, RandomForest, PermutationImportance, LightGBM.

---

## Paper Drafts

| File | Version | Status | Scope |
|------|---------|--------|-------|
| `paper/draft_v7_tmlr.tex` | v7 | TMLR submission (anonymous, under review) | Method + empirical validation |
| `paper/draft_v7_preprint.tex` | v7 | ArXiv preprint (de-anonymized) | Same content, author names visible |
| `paper/draft_v8_reviewer_response.tex` | v8 | Reviewer response revision | Adds colsample ablation, MI quantitative bridge, enrichment theory |
| `paper/draft_v1.tex` through `draft_v6_preprint.tex` | v1-v6 | Historical/frozen | Do not modify |
| `paper/arxiv_abstract.txt` | — | ArXiv abstract text | Standalone |

**Citation:** Caraker, Arnold, Rhoads (2026). *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* arXiv:2603.22346. DOI: 10.5281/zenodo.19060132.

---

## Methodology Notes

### Four-Way Data Split

All experiments use train / val / explain / test (A4 fix). SHAP is computed on the explain set (X_explain), separate from training data, validation data (used for performance filtering), and test data (used for RMSE evaluation). This prevents any overlap between model selection, explanation computation, and predictive evaluation.

### Statistical Testing

- **Stability** is computed as mean pairwise Spearman ρ across all repetition pairs — a single number, not per-rep. Cannot use Wilcoxon (requires paired values). Bootstrap CIs used instead.
- **Equity and accuracy** are per-rep values. Wilcoxon signed-rank tests with Holm-Bonferroni correction.
- **Effect sizes** via Cohen's d with direction indicators.
- **TOST equivalence** for DASH vs SR (confirmed equivalent on stability, accuracy, equity).

### Canonical Configuration

```
M = 200, K = 30, N_REPS = 50, EPSILON = 0.08, DELTA = 0.05, SEED = 42
Real-world: REAL_EPSILON = 0.05, epsilon_mode = 'relative'
```
