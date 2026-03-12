# Research Roadmap

DASH is Paper 1 of a five-paper research program that builds from a practical tool toward a new framework for trustworthy feature attribution. Each paper builds on the previous, uses the same codebase and experimental infrastructure, and targets a progressively more ambitious claim. Total timeline: 12-18 months.

## Paper 1: DASH -- The Method

**Target:** TMLR (Transactions on Machine Learning Research) | **Status:** Experimental validation in progress | **Risk:** Low

Practical tool and empirical validation. The core claim: DASH produces more stable, accurate, and equitable SHAP importance rankings than single-model or single-ensemble approaches, with the advantage growing as feature collinearity increases. The key mechanistic insight is that sequential residual dependency within a single boosting ensemble amplifies collinearity-induced instability, and only independence between models resolves it.

**Results to date:** DASH stability exceeds Single Best at 5/5 correlation levels (full run). DGP agreement at ρ=0.9: Spearman ρ = 0.990. Equity 33% better than Single Best at ρ=0.95. Large Single Model degrades faster than Single Best at all ρ levels, confirming the first-mover hypothesis. FSI correctly identifies known collinear clusters on Breast Cancer data without supervision.

**Remaining work:** Full experiments at M=200, K=20, N_REPS=10 across all ρ levels, both DGPs, and all 8 methods. UCI benchmarks (Superconductor, Communities & Crime). UCR time series + tsfresh experiments (5 datasets). Ablation studies (M, K, ε, δ, colsample range, importance proxy). Statistical testing: Friedman omnibus + Wilcoxon with Bonferroni correction.

## Paper 2: From Consensus to Partial Orders

**Target:** KDD or NeurIPS Workshop on XAI | **Timeline:** Months 2-4 | **Risk:** Low

Feature importance under collinearity is fundamentally a distributional quantity, not a point estimate. This paper extends DASH to produce importance partial orders -- directed acyclic graphs where edges represent high-confidence importance relationships and absent edges represent underdetermined orderings.

DASH already stores the full K x N' x P tensor of SHAP values across models. Currently we collapse this to a mean (consensus) and variance (FSI). The partial order lives in that tensor. For each pair of features (j, k), compute π(j>k) = the fraction of models where feature j has higher global importance than feature k. If π > 0.95, draw a confident edge j→k. If π is between 0.4 and 0.6, the ordering is underdetermined -- no edge. The resulting DAG is a partial order with calibrated confidence.

**Decision gate (month 3):** Within-group pairwise confidence π values near 0.5 (underdetermined), between-group π values near 1.0 (well-determined). If the partial order doesn't add enough over FSI, fold into Paper 1 as an additional diagnostic.

## Paper 3: The Impossibility Result

**Target:** NeurIPS or AISTATS | **Timeline:** Months 3-6 | **Risk:** Medium

No single importance ranking can simultaneously satisfy stability (invariance to model specification within the Rashomon set), accuracy (recovery of the true importance ordering), and completeness (total order over all features) when features are collinear. This is a fundamental limitation analogous to Arrow's impossibility theorem for social choice.

**Proof strategy:** In the linear Gaussian case with block-diagonal correlation, there exist models in the Rashomon set that achieve the same loss but attribute different importances to features within a correlated group. Any estimator producing a total order must violate stability, accuracy, or completeness. The constructive resolution: relax completeness to a partial order (Paper 2) and you recover stability and accuracy. The FSI identifies exactly which features require the relaxation.

**Empirical validation:** From the DASH population, plot each model's accuracy vs stability. Show they trace a Pareto frontier that no single model escapes. Show DASH's consensus sits outside this frontier because it aggregates rather than selects.

**Decision gate (month 4):** Clean proof for the linear Gaussian case with a non-trivial result. If the general case hits technical obstacles, publish the linear case with empirical evidence that it extends.

## Paper 4: Optimization Path Dependence in Explanations

**Target:** JMLR or ICML | **Timeline:** Months 6-12 | **Risk:** Medium-high

The sequential residual dependency discovered in DASH's Large Single Model experiments is an instance of a general phenomenon: any model trained by iterative optimization develops path-dependent explanations where the order in which features become "active" during training biases their final attribution.

**Experimental plan:**
- **GBDT formalization (months 6-8):** Model split selection probability as a function of residuals and collinearity. Prove positive autocorrelation in feature selection across trees.
- **Neural network experiments (months 8-10):** Same synthetic data and Breast Cancer. Train 20 identical MLPs with different random initializations. Compute integrated gradients for each. Measure stability vs ρ. Track attributions at training checkpoints to show first-mover dynamics.
- **DASH for neural networks (months 10-11):** Apply the independence principle: train 20 MLPs with different architectures or dropout patterns, average their attributions.
- **Unifying framework (months 11-12):** Iterative optimization with collinear inputs produces path-dependent feature utilization. DASH's independence principle is the general fix.

**Decision gate (month 7):** Attribution stability degrades with ρ for MLPs, same qualitative pattern as GBDTs. If MLPs don't show the effect, narrow to GBDTs only.

## Paper 5: Explanation-Aware Model Selection

**Target:** Nature Machine Intelligence | **Timeline:** Months 12-18 | **Risk:** Low execution, high acceptance

The paradigm paper. The field's standard workflow -- train the best predictor, then explain it -- is fundamentally flawed when features are correlated. Papers 1-4 collectively demonstrate this. The right paradigm is joint optimization of prediction and explanation quality: among all models with equivalent predictive performance (the Rashomon set), select based on explanation properties.

**Three strategies for navigating the prediction-explanation Pareto frontier:**
1. **Selection** (DASH's approach): train many models, select for explanation quality
2. **Regularization:** penalize instability during training
3. **Weighting:** ensemble models weighted by explanation reliability

Each opens a research direction. The paper synthesizes Papers 1-4 with minimal new experiments and builds the case that single-model explanations are fundamentally limited -- a different and larger claim than "SHAP is noisy."

---

## Timeline

| Month | Paper 1 | Paper 2 | Paper 3 | Paper 4 | Paper 5 |
|-------|---------|---------|---------|---------|---------|
| 1-2 | Full experiments, writing | -- | -- | -- | -- |
| 3 | Submit TMLR, post ArXiv | Implement partial orders | Begin proof | -- | -- |
| 4 | Revisions | Write + submit | Linear Gaussian proof | -- | -- |
| 5-6 | -- | -- | General case, write | -- | -- |
| 7 | -- | -- | Submit | Pilot MLP experiments | -- |
| 8-10 | -- | -- | Revisions | Full experiments | -- |
| 11-12 | -- | -- | -- | Write + submit | Draft |
| 13-15 | -- | -- | -- | Revisions | Write + submit |

## Decision Gates

| Gate | Timing | Test | Status |
|------|--------|------|--------|
| 1: Paper 1 proof of concept | End of week 1 | DASH > SB > LSM at M=200, ρ=0.9 | **PASSED** |
| 2: Paper 2 viability | Month 3 | Partial order confidence calibration works | Pending |
| 3: Paper 3 proof viability | Month 4 | Clean proof for linear Gaussian case | Pending |
| 4: Paper 4 neural networks | Month 7 | MLP attribution stability degrades with ρ | Pending |
