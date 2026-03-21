# Research Directions — DASH-SHAP Program

> **Purpose**: Consolidated catalog of all research directions, open questions, and prioritized recommendations across the DASH-SHAP program. Cross-references source documents rather than duplicating content.
>
> **Last updated**: 2026-03-21

---

## Source Document Inventory

| Document | Location | Content |
|----------|----------|---------|
| Research Roadmap | `ROADMAP.md` | 5-paper program, timeline, decision gates |
| Extensions Framework | `docs/EXTENSIONS.md` | 11 extensions, phasing, dependency graph |
| Extensions User Guide | `docs/EXTENSIONS_USER_GUIDE.md` | Practitioner-facing workflows |
| TMLR Review | `paper/REVIEW_v7.md` | Reviewer action items, analytical gaps |
| Paper §6.3–6.4 | `paper/draft_v7_preprint.tex` (lines 1490–1554) | Limitations + broader implications |
| Experiment Guide | `EXPERIMENT_GUIDE.md` | 11 experiments, methodology |

---

## 1. Immediate Priority: TMLR Submission (Paper 1)

**Status**: ArXiv posted. TMLR submission in preparation.

Critical path (from `paper/REVIEW_v7.md` §H):

| # | Item | Effort | Status |
|---|------|--------|--------|
| 1 | Run notebook 7 fully (50 reps, RF baseline, T-scaling) | High compute | Code exists, not yet run |
| 2 | Add Random Forest to all paper tables | Low | Depends on #1 |
| 3 | Update numbers from 20-rep to 50-rep | Low | Depends on #1 |
| 4 | T-scaling figure in appendix | Low | Notebook 7 cell 44–45 |
| 5 | TMLR style file conversion | Medium | Manual LaTeX |
| 6 | Bootstrap stability hypothesis test | Low | New code needed |
| 7 | LightGBM baseline at ρ=0.9 | Medium | New baseline class |
| 8 | Bibliography to .bib + vector figures | Low | Manual |

Full details: `paper/REVIEW_v7.md` §H (P0–P2 priority tiers).

---

## 2. Near-Term Research (Months 2–6)

### Paper 2: From Consensus to Partial Orders

- **Target**: KDD or NeurIPS Workshop on XAI
- **Risk**: Low
- **Core idea**: Extend DASH from consensus mean to importance partial orders (DAGs with calibrated confidence edges). For feature pair (j, k), compute π(j>k) = fraction of K models ranking j above k. High π → confident edge; π ≈ 0.5 → underdetermined ordering.
- **Decision gate** (month 3): Within-group π ≈ 0.5, between-group π → 1.0
- **Code status**: Extension 2 (`partial_order.py`) implemented in Phase 1. `PartialOrderResult` with adjacency matrix and confidence matrix exists.
- **Cross-ref**: `ROADMAP.md` §2, `docs/EXTENSIONS.md` Extension 2

### Paper 3: The Impossibility Result

- **Target**: NeurIPS or AISTATS
- **Risk**: Medium
- **Core idea**: No single importance ranking can simultaneously satisfy stability, accuracy, and completeness under collinearity. Analogous to Arrow's impossibility theorem.
- **Proof strategy**: Linear Gaussian case with block-diagonal correlation. Constructive: relaxing completeness to partial orders recovers stability and accuracy.
- **Decision gate** (month 4): Clean proof for linear Gaussian case
- **Cross-ref**: `ROADMAP.md` §3

### Extensions Phases 2–3

| Extension | Phase | Value |
|-----------|-------|-------|
| Feature Groups (4) | 2 | Identifies collinear clusters via SHAP substitutability |
| Local Uncertainty (8) | 2 | Per-observation SHAP disagreement across K models |
| Stable Feature Selection (5) | 2 | Composite importance + stability ranking |
| Causal Flags (11) | 3 | Per-feature labels: robust / collinear / fragile |
| Audit Report (3) | 3 | Structured report with optional enrichments |

**Cross-ref**: `docs/EXTENSIONS.md` §3, Phases 2–3

---

## 3. Medium-Term Research (Months 6–12)

### Paper 4: Optimization Path Dependence in Explanations

- **Target**: JMLR or ICML
- **Risk**: Medium-high
- **Core idea**: First-mover bias is an instance of a general phenomenon — any iterative optimization creates path-dependent explanations where feature activation order biases final attribution.
- **Experimental plan**:
  - Months 6–8: GBDT formalization (split selection probability as function of residuals and collinearity)
  - Months 8–10: Neural network experiments (20 MLPs, different initializations, integrated gradients, stability vs ρ)
  - Months 10–11: DASH for neural networks (independence principle applied to MLPs)
  - Months 11–12: Unifying framework
- **Decision gate** (month 7): MLP attribution stability degrades with ρ in same pattern as GBDTs
- **Cross-ref**: `ROADMAP.md` §4

### Interaction Tensor Averaging

TreeSHAP supports interaction tensors Φ_int ∈ ℝ^{N'×P×P}. Averaging element-wise across the DASH ensemble yields stable interaction estimates by the same independence argument. Cost: O(TLD²) per model (vs O(TLD) for main effects). Practical for moderate P, expensive for large feature spaces.

- **Status**: Described in paper §6.3 (lines 1500–1511). Not implemented.
- **Assessment**: Concrete, low-risk extension with clear novelty. No single-model workflow can offer this.
- **Cross-ref**: `paper/draft_v7_preprint.tex` lines 1500–1511

### LightGBM / CatBoost Generalization

Paper 1 positions first-mover bias as a property of gradient boosting generally but only tests XGBoost. A single LightGBM confirmation (leaf-wise splitting) would substantially strengthen the generalization claim.

- **Status**: Not implemented. New baseline class needed.
- **Cross-ref**: `paper/REVIEW_v7.md` §B3

---

## 4. Long-Term Research (Months 12–18)

### Paper 5: Explanation-Aware Model Selection

- **Target**: Nature Machine Intelligence
- **Risk**: Low execution, high acceptance
- **Core idea**: The standard workflow (train best predictor, then explain) is fundamentally flawed under collinearity. Papers 1–4 collectively demonstrate this. The right paradigm: joint optimization of prediction and explanation quality within the Rashomon set.
- **Three strategies**: Selection (DASH), Regularization (penalize instability during training), Weighting (ensemble weights by explanation reliability)
- **Cross-ref**: `ROADMAP.md` §5, `docs/EXTENSIONS.md` Extension 7 (ParetoSelector)

### Efficient Rashomon Set Construction

> **Note**: This direction emerged from analysis of surrogate-assisted optimization approaches. It is documented here for the first time.

The current DASH pipeline uses brute-force-then-filter: train M=200 models with random hyperparameters, filter to those within ε of best validation score, select K=30 diverse models. This is simple and sufficient for XGBoost (training is cheap, Rashomon set is generous). But for expensive model families (neural networks in Paper 4), efficient Rashomon enumeration becomes relevant.

#### Approach 1: Surrogate-Assisted Level Set Estimation

The Rashomon set R_ε = {θ : L(θ) ≤ L* + ε} is a level set of the loss function. GP-based active level set estimation (Gotovos et al. 2013) classifies regions as inside/outside/uncertain, focusing evaluations on the boundary.

**Assessment**: Theoretically sound, but GP surrogation faces impedance mismatches with tree ensemble hyperparameter spaces:
- Mixed discrete/continuous hyperparameters (max_depth ∈ ℤ, learning_rate ∈ ℝ) violate smooth kernel assumptions
- Loss surfaces for tree ensembles have plateaus, ridges, and discontinuities
- Rashomon sets can be non-convex and disconnected — multiple isolated "islands"
- The 50–100 evaluation budget suggested in the literature is likely insufficient for ~10-dimensional mixed spaces

**Recommendation**: If pursuing this, use a random forest surrogate (SMAC-style, Hutter et al. 2011) rather than a GP. Handles mixed spaces natively, doesn't assume smoothness.

#### Approach 2: Diversity-Aware Acquisition Functions

Composite acquisition: α_div(θ) = P(L(θ) ≤ L* + ε | D) · d(θ, S_found).

**Assessment**: The multiplicative form creates an uncontrollable implicit trade-off between Rashomon membership probability and distance from discovered set. A constrained formulation (maximize diversity subject to P(Rashomon) ≥ δ) is strictly better — separates the feasibility gate from the diversity objective.

**Subtlety**: P(L(θ) ≤ L* + ε) depends on knowing L*, which is itself uncertain early in the search. Requires integrating over posterior uncertainty in L* or accepting plug-in bias.

#### Approach 3: Batch BO with DPPs

Determinantal Point Processes for batch selection: kernel L_ij = q_i · q_j · S_ij trades off quality (Rashomon membership) against diversity. The determinantal structure naturally produces both high-quality and mutually dissimilar selections.

**Assessment**: Mathematically clean. MAP inference is NP-hard (greedy approximation gives (1-1/e) guarantee). Requires discretizing the continuous hyperparameter space or using continuous DPP variants (less mature). The claim that this is "plug-and-play" with a GP surrogate oversimplifies the kernel design challenges.

#### Approach 4: Quality-Diversity (MAP-Elites)

Tessellate behavior/feature space into bins, maintain an archive where each bin holds the best-performing solution.

**Assessment**: Works well for ≤ 3–4 dimensional behavior spaces. For DASH, the relevant behavior space is explanation diversity in ℝ^P (P = 10–50 features). Cannot tessellate this space — the curse of dimensionality makes this impractical without projecting to low-dimensional summaries, which discards exactly the fine-grained diversity you want.

#### The Diversity Metric Hierarchy

This is the most important design choice, more so than the search algorithm:

| Metric | Cost to Compute | Cost to Surrogate | Relevance to DASH |
|--------|----------------|-------------------|-------------------|
| Hyperparameter distance | Free | Trivial | Low |
| Prediction disagreement | Requires trained model | Moderate | Medium |
| Explanation divergence | Trained model + SHAP | Very hard | **High** |
| Decision boundary geometry | Trained model + analysis | Extremely hard | Medium-high |

**Fundamental tension**: The most relevant diversity metric (explanation divergence) requires computing SHAP values, which requires fully training the model. Any surrogate that avoids training can only optimize a proxy metric.

#### Practical Assessment

**When brute force wins**: If the Rashomon set volume fraction is > 30% of the search space (common for well-regularized ML), most random samples pass the filter and surrogate-based filtering saves < 3× compute. Run this diagnostic first: what fraction of M=200 models pass the epsilon filter in DASH experiments?

**When surrogation wins**: Scaling to expensive model families (neural nets, large-scale GBDTs) where training cost dominates, and the Rashomon set is a thin shell in hyperparameter space.

**Sharpest version** (open research question): Two-loop architecture — outer GP for loss level set estimation, inner loop for diversity selection among verified Rashomon members. The genuinely open problem is joint surrogation of loss + explanation features via multi-output GP. This faces: (a) high-dimensional output (SHAP is P-dimensional), (b) heterogeneous smoothness (loss smoother than SHAP summaries), (c) SHAP discontinuity (small hyperparameter changes → discrete tree structure changes → discontinuous SHAP). This is a separate research contribution, not a deployable tool.

### Extensions Phases 4–5

| Extension | Phase | Value |
|-----------|-------|-------|
| Drift Monitor (6) | 4 | Detect concept drift via cosine distance on importance |
| Federated Consensus (10) | 4 | Combine DASH results across sites |
| Pareto Model Selection (7) | 5 | Prediction-explanation frontier (Paper 5 core) |

**Cross-ref**: `docs/EXTENSIONS.md` §3, Phases 4–5

---

## 5. Prioritized Recommendations

### Tier 1: Do Now (blocks publication)

| # | Direction | Rationale |
|---|-----------|-----------|
| 1 | Run notebook 7 fully (50 reps) | Single action that produces RF baseline, T-scaling evidence, and publishable numbers |
| 2 | Update paper tables + TMLR format | Mechanical but required for submission |
| 3 | Bootstrap stability hypothesis test | Low effort, directly addresses anticipated reviewer concern (REVIEW_v7 §M3) |

### Tier 2: Next Quarter (high-impact research)

| # | Direction | Rationale |
|---|-----------|-----------|
| 4 | Paper 2 (Partial Orders) | Strongest risk-adjusted return. Low risk, code exists, clean standalone contribution |
| 5 | Paper 3 (Impossibility Result) | If the linear Gaussian proof works, most citable contribution in the program. Start in parallel with Paper 2 |
| 6 | LightGBM baseline | Single experiment that substantially strengthens Paper 1's generalization claim |
| 7 | Extensions Phase 2 (Groups, Local, Selection) | High practitioner value, strengthens the DASH ecosystem |

### Tier 3: This Year (builds research program)

| # | Direction | Rationale |
|---|-----------|-----------|
| 8 | Paper 4 pilot (MLP experiment) | Run pilot early to de-risk before full commitment. Decision gate at month 7 is well-designed |
| 9 | Interaction tensor averaging | Concrete, low-risk, clear novelty. No single-model workflow can offer stable interaction estimates |
| 10 | Conditional SHAP discussion | Important for completeness. Discussion paragraph sufficient for TMLR; empirical comparison only if reviewer demands it |

### Tier 4: Exploratory (research bets)

| # | Direction | Rationale |
|---|-----------|-----------|
| 11 | Efficient Rashomon construction | Over-engineered for XGBoost (training is cheap). Only pursue if scaling to neural nets in Paper 4 where training cost dominates. Start with diagnostic: measure Rashomon set volume fraction |
| 12 | Paper 5 (Explanation-Aware Selection) | Don't invest until Papers 2–3 are accepted. Low execution risk but high acceptance risk for Nature MI |
| 13 | Federated / Drift extensions | Practitioner features, not research contributions. Implement on user demand |

---

## 6. Open Research Questions

| Question | Difficulty | Relevant Paper |
|----------|-----------|----------------|
| Can loss + explanation features be jointly surrogated for efficient Rashomon enumeration? | Very hard (multi-output GP, SHAP discontinuity, heterogeneous smoothness) | Follow-up to Paper 4 |
| Does DASH scale to P > 50 features? | Medium (computational, not conceptual) | Paper 1 extension |
| How does conditional/causal SHAP interact with DASH aggregation? | Medium (orthogonal approach to correlated features) | Paper 1 related work |
| Is a crossed variance decomposition (R data seeds × R model seeds) worth the compute? | Low (nice-to-have, current marginal design is standard) | Paper 1 methodology |
| Does the impossibility result extend beyond linear Gaussian to general correlated features? | Hard (Paper 3 general case) | Paper 3 |
| Do neural networks exhibit the same qualitative stability degradation with ρ as GBDTs? | Unknown until pilot experiment | Paper 4 decision gate |
| Can interaction tensor averaging produce stable pairwise interaction rankings at scale? | Medium (O(TLD²) cost, need P ≤ ~50) | Paper 1 extension |
| What is the optimal diversity metric for Rashomon set sampling — explanation space, prediction space, or a hybrid? | Hard (explanation space is most relevant but hardest to surrogate) | Efficient Rashomon construction |

---

## Decision Gate Schedule

| Gate | Timing | Test | Status |
|------|--------|------|--------|
| 1: Paper 1 proof of concept | Week 1 | DASH > SB > LSM at M=200, ρ=0.9 | **PASSED** |
| 2: Paper 2 viability | Month 3 | Partial order confidence calibration works | Pending |
| 3: Paper 3 proof viability | Month 4 | Clean proof for linear Gaussian case | Pending |
| 4: Paper 4 neural networks | Month 7 | MLP attribution stability degrades with ρ | Pending |

Cross-ref: `ROADMAP.md` §Decision Gates
