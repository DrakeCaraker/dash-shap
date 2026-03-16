# Research Roadmap

DASH is Paper 1 of a five-paper research program that builds from a practical tool toward a new framework for trustworthy feature attribution. Each paper builds on the previous, uses the same codebase and experimental infrastructure, and targets a progressively more ambitious claim. Total timeline: 12-18 months.

## Paper 1: First-Mover Bias in Gradient Boosting Explanations

**Target:** TMLR (Transactions on Machine Learning Research) | **Status:** ArXiv pre-print posted; TMLR submission in preparation | **Risk:** Low

**Title:** "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution"

The paper reframed from "DASH as a method" to "first-mover bias as a mechanism." The core claim: sequential residual dependency in gradient boosting creates first-mover bias that concentrates SHAP importance on arbitrary features under multicollinearity, and model independence is both necessary and sufficient to resolve it. DASH is one principled instantiation of this principle, but the contribution is the mechanistic insight — validated by the finding that DASH ≈ Stochastic Retrain (d=+0.05, n.s.), proving the principle generalizes beyond any specific implementation.

**Three lines of evidence:**
1. **The mechanism:** Large Single Model (same compute budget as DASH) produces *worst* stability, confirming sequential dependency is the cause
2. **The principle:** Independence between models resolves it (DASH ≈ Stochastic Retrain equivalence)
3. **Dose-response:** Effect scales with correlation — dependent methods degrade 0.973→0.951, independent methods remain flat

**Final configuration:** M=200, K=30, N_REPS=20, ε=0.08 (synthetic) / 0.05 relative (real-world).

**Results (ArXiv pre-print):** DASH stability 0.977 at ρ=0.9 vs Single Best 0.958 vs LSM 0.938. Breast Cancer: DASH 0.930 vs Single Best 0.317 (+0.613). Superconductor: DASH 0.962 vs Single Best 0.830 vs LSM 0.702. California Housing: DASH 0.982 vs Single Best 0.967. DGP agreement (Spearman ρ) = 0.988 at ρ=0.9. Safety at ρ=0: gap = 0.0003 (n.s.). 11/11 pre-registered success criteria pass. Statistical testing: Wilcoxon signed-rank + Holm-Bonferroni step-down + Cohen's d effect sizes + TOST equivalence testing.

**Nonlinear scope boundary (discovered):** Under nonlinear DGP, DASH advantage only emerges at ρ≥0.7, with overall stability ~0.87-0.88 vs ~0.93-0.98 for linear. Paper documents this as an honest limitation and scope condition.

**Descoped from original plan (scope control — 3 real-world datasets sufficient for claims):**
- Communities & Crime dataset — removed to keep paper focused
- UCR time series + tsfresh (5 datasets) — removed; not needed for the first-mover bias narrative
- California Housing added (not originally planned) as a low-collinearity safety check

**Remaining work for TMLR:** Finalize demo_benchmark_7.ipynb (TMLR canonical notebook). Submit to TMLR. Address reviewer feedback.

## Paper 2: From Consensus to Partial Orders

**Target:** KDD or NeurIPS Workshop on XAI | **Timeline:** Months 2-4 | **Risk:** Low

Feature importance under collinearity is fundamentally a distributional quantity, not a point estimate. This paper extends DASH to produce importance partial orders -- directed acyclic graphs where edges represent high-confidence importance relationships and absent edges represent underdetermined orderings.

DASH already stores the full K x N' x P tensor of SHAP values across models. Currently we collapse this to a mean (consensus) and variance (FSI). The partial order lives in that tensor. For each pair of features (j, k), compute π(j>k) = the fraction of models where feature j has higher global importance than feature k. If π > 0.95, draw a confident edge j→k. If π is between 0.4 and 0.6, the ordering is underdetermined -- no edge. The resulting DAG is a partial order with calibrated confidence.

**Decision gate (month 3):** Within-group pairwise confidence π values near 0.5 (underdetermined), between-group π values near 1.0 (well-determined). If the partial order doesn't add enough over FSI, fold into Paper 1 as an additional diagnostic.

**Note (post Paper 1 reframing):** Paper 1's shift from "DASH as method" to "first-mover bias as mechanism" makes partial orders a stronger standalone contribution. Since Paper 1 no longer centers DASH-as-tool, the partial order extension is less at risk of appearing incremental.

## Paper 3: The Impossibility Result

**Target:** NeurIPS or AISTATS | **Timeline:** Months 3-6 | **Risk:** Medium

No single importance ranking can simultaneously satisfy stability (invariance to model specification within the Rashomon set), accuracy (recovery of the true importance ordering), and completeness (total order over all features) when features are collinear. This is a fundamental limitation analogous to Arrow's impossibility theorem for social choice.

**Proof strategy:** In the linear Gaussian case with block-diagonal correlation, there exist models in the Rashomon set that achieve the same loss but attribute different importances to features within a correlated group. Any estimator producing a total order must violate stability, accuracy, or completeness. The constructive resolution: relax completeness to a partial order (Paper 2) and you recover stability and accuracy. The FSI identifies exactly which features require the relaxation.

**Empirical validation:** From the DASH population, plot each model's accuracy vs stability. Show they trace a Pareto frontier that no single model escapes. Show DASH's consensus sits outside this frontier because it aggregates rather than selects.

**Decision gate (month 4):** Clean proof for the linear Gaussian case with a non-trivial result. If the general case hits technical obstacles, publish the linear case with empirical evidence that it extends.

## Paper 4: Optimization Path Dependence in Explanations

**Target:** JMLR or ICML | **Timeline:** Months 6-12 | **Risk:** Medium-high

The sequential residual dependency discovered in DASH's Large Single Model experiments is an instance of a general phenomenon: any model trained by iterative optimization develops path-dependent explanations where the order in which features become "active" during training biases their final attribution.

**Note (post Paper 1 reframing):** Paper 1's title — "First-Mover Bias in Gradient Boosting Explanations" — directly sets up this paper. The "first-mover bias" terminology and the LSM evidence provide a tighter bridge than originally anticipated. Paper 4 generalizes the mechanism Paper 1 identifies.

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
| 3 | ~~Submit TMLR~~, post ArXiv ✓ | Implement partial orders | Begin proof | -- | -- |
| 4 | Submit TMLR, revisions | Write + submit | Linear Gaussian proof | -- | -- |
| 5-6 | -- | -- | General case, write | -- | -- |
| 7 | -- | -- | Submit | Pilot MLP experiments | -- |
| 8-10 | -- | -- | Revisions | Full experiments | -- |
| 11-12 | -- | -- | -- | Write + submit | Draft |
| 13-15 | -- | -- | -- | Revisions | Write + submit |

## Decision Gates

| Gate | Timing | Test | Status |
|------|--------|------|--------|
| 1: Paper 1 proof of concept | End of week 1 | DASH > SB > LSM at M=200, ρ=0.9 | **PASSED** — confirmed at K=30, N_REPS=20; ArXiv pre-print posted with 11/11 success criteria passing |
| 2: Paper 2 viability | Month 3 | Partial order confidence calibration works | Pending |
| 3: Paper 3 proof viability | Month 4 | Clean proof for linear Gaussian case | Pending |
| 4: Paper 4 neural networks | Month 7 | MLP attribution stability degrades with ρ | Pending |
