# Peer Review: draft_v6_preprint.tex — ArXiv Readiness Assessment

**Paper:** "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution"
**Authors:** Caraker, Arnold, Rhoads (2026)
**Reviewer standard:** Rigorous ML venue peer review; ArXiv pre-print norms noted separately
**Date:** 2026-03-16
**Draft version:** v6
**Disclosure:** This review was conducted by an AI system (Claude). The full paper was read line-by-line; all claims were verified against the codebase (`dash/core/`, `dash/baselines/`, `dash/evaluation/`, `dash/experiments/`, `run_experiments.py`). Prior reviews (REVIEW_v5_arxiv.md, REVIEW_v6.md, REVIEW_v6_arxiv.md, REVIEW_v7.md) were read for context but this review is independently structured.

---

## Overall Verdict

**ArXiv: Ready to post.** This is a well-executed empirical paper with a genuine mechanistic insight, thorough experiments, and — most notably — uncommon intellectual honesty. The paper clears the ArXiv bar comfortably.

**Conference/journal equivalent: 6.5–7/10.** The contribution is primarily empirical-diagnostic. The mechanistic characterization of first-mover bias is the real value; DASH as a pipeline is secondary to the finding that model independence suffices. This is appropriate for TMLR scope.

---

## Strengths

**S1. Elegant controlled experiment design.** The LSM comparison is the paper's strongest methodological move. Same total tree count, same `colsample_bytree` restriction, only structural difference is sequential vs. independent training. This cleanly isolates the mechanism. The result — LSM produces the *worst* explanations of any method at every rho level — is striking and convincing.

**S2. Exemplary scientific honesty.** The paper makes its strongest competitor (Stochastic Retrain) the headline finding: "the most important result in this paper" (Section 5.2, ~line 1036). SR matches DASH on stability at every rho level. Most authors would bury this. Here it becomes the evidence *for* the independence principle, which is a more fundamental and useful insight than "our pipeline is best." This intellectual maturity will serve the paper well under formal review.

**S3. Clean two-tier separation.** The partition between dependent methods (stability 0.938–0.964) and independent methods (~0.977) at rho=0.9, with the inter-tier gap (0.01–0.04) dwarfing the intra-tier gap (~0.001), is compelling. The non-overlapping BCa bootstrap CIs reinforce this.

**S4. Thorough experimental coverage.** Nine methods, five correlation levels, two DGPs, three real-world datasets, 20 repetitions each, Wilcoxon tests with Holm-Bonferroni correction, effect sizes, bootstrap CIs. The four-way data split (train/val/explain/test) is carefully motivated and prevents SHAP overfitting — a subtlety many papers miss.

**S5. Accuracy/equity circularity caveat.** Section 5.1 (lines 862–872) explicitly acknowledges that the ground-truth definition presupposes equitable credit distribution, partially confounding accuracy and equity results with DASH's design intent. This is a trap most papers fall into silently. Flagging it preemptively strengthens the paper's credibility.

**S6. Practical diagnostics.** The FSI and IS Plot are genuinely useful — an unsupervised collinearity detector built from SHAP disagreement across an ensemble, without needing the correlation matrix. The four-quadrant taxonomy (robust drivers, collinear clusters, confirmed unimportant, fragile interactions) is intuitive and actionable.

**S7. Clear, disciplined writing.** The prose is concise and well-organized. The paper progresses logically: mechanism → principle → diagnostics → method → experiments → limitations. Technical claims are carefully scoped.

---

## Major Issues

### M1. The T-Scaling Hypothesis Is Formalized But Not Directly Tested

**Lines 384–396.** The paper formalizes an "empirical hypothesis" that first-mover concentration increases monotonically with tree count T, then tests it only *indirectly* via the LSM comparison (which confounds T with many other factors: model capacity, regularization interaction, etc.). A direct experiment — vary T while holding other hyperparameters fixed — is described as "available in the code repository." I verified this: the code exists and the experiment is trivial to include.

This creates a structural awkwardness: the paper elevates a claim to "hypothesis" status, formulates it mathematically, then delegates its direct test to a URL. The indirect evidence (LSM uses ~15K sequential trees and performs worst) is suggestive but not dispositive — LSM also differs from Single Best in tree count, capacity, and regularization regime.

**ArXiv acceptability:** Borderline. The indirect evidence is reasonable, but the gap between formalization and testing is noticeable. A single appendix figure showing concentration vs. T would close this cleanly at near-zero page cost.

**Recommendation:** Include the T-scaling results as an appendix figure, or soften "empirical hypothesis" to "motivating observation."

### M2. Random Forest Baseline — The Most Predictable Reviewer Question

**Lines 1458–1463.** The paper's central principle is that model independence resolves first-mover bias. Random forests train trees independently by construction. The obvious corollary: RF SHAP stability should be higher than single XGBoost. I confirmed that `dash/baselines/random_forest.py` exists and is integrated into the experiment infrastructure. This experiment is cheap and the code is ready.

Three outcomes, all informative:
- **RF matches DASH:** Validates the independence principle powerfully. Simplifies the practical recommendation to "use RF if you only need stable SHAP values."
- **RF is intermediate:** Shows that tree-level independence (RF) is weaker than model-level independence (DASH/SR), adding nuance to the principle.
- **RF is worse than expected:** Reveals that RF's constrained feature space or shallow trees interact differently with SHAP attribution.

Any outcome would significantly strengthen the paper. The current limitation acknowledgment (lines 1458–1463) is adequate for ArXiv but this will be the first question from any knowledgeable reviewer.

**ArXiv acceptability:** Acceptable as-is with the limitation flagged. But barely — this is a cheap experiment that directly tests the paper's central claim.

### M3. DASH's Unique Method Contribution Is Narrow

The paper's own results demonstrate that the operative mechanism is model independence, not DASH's specific pipeline design:
- Random Selection: stability 0.976 (vs. DASH 0.977)
- Stochastic Retrain: stability 0.977 (matches DASH)
- Naive Top-N: stability 0.976

DASH's three claimed advantages over SR:
1. **Speed (1.7x):** Real, but incidental (fewer SHAP evaluations due to early stopping in diversity selection, not from the diversity criterion itself).
2. **Diagnostics (FSI, IS Plot):** Genuinely novel and useful, but not architecturally coupled to the DASH pipeline — you could compute FSI/IS Plot from *any* multi-model ensemble, including SR.
3. **Equity (CV 0.176 vs. 0.182):** Not statistically significant (acknowledged).

The four-contribution framing (Phenomenon, Principle, Diagnostics, Method) may overweight the Method contribution. The paper's real contributions are #1 and #2 (finding + principle), with #3 (diagnostics) as a useful practical tool and #4 (pipeline) as an engineering convenience.

**ArXiv acceptability:** Acceptable. The paper already acknowledges the SR equivalence prominently and positions DASH as a "principled instantiation" rather than a statistical improvement. The honest framing is sufficient for a pre-print.

**Recommendation for journal:** Consider restructuring the abstract to lead with finding/principle rather than pipeline, and explicitly decouple FSI/IS Plot from the DASH pipeline (noting they work with any multi-model ensemble).

### M4. N_reps = 20 Limits Statistical Power for Nuanced Claims

Several comparisons that the paper reports as non-significant might resolve with more repetitions:
- DASH vs. SR on equity (d = -0.21, p = 0.622): A small effect at 20 reps could become significant at 50.
- DASH vs. Random Selection on stability (0.977 vs. 0.976): No formal test is possible with the current aggregate stability metric.
- Nonlinear DGP differences at rho=0.5 (SR 0.855 vs. DASH 0.852): The ordering reversal relative to linear DGP is potentially interesting but untestable at N=20.

The paper acknowledges this limitation (lines 1449–1452) and the TMLR version targets N=50 (notebook 7 is configured for this). But the current claims about non-significance — particularly the DASH-SR equivalence, which is framed as "the most important result" — are made on limited power.

**ArXiv acceptability:** Acceptable. N=20 is standard for this type of benchmark. The important thing is that the paper doesn't over-claim significance where there isn't any.

### M5. Interventional vs. Conditional SHAP — A Load-Bearing Choice

**Lines 1421–1428.** The paper uses interventional TreeSHAP throughout and correctly notes in the limitations that this evaluates the model at out-of-distribution feature combinations under high correlation. This is well-known (Aas et al. 2021 and Janzing et al. 2020 are cited). But for a paper specifically about SHAP instability under multicollinearity, the choice between interventional and conditional SHAP is load-bearing:

- Conditional SHAP respects the feature distribution and avoids OOD evaluations, but is more expensive and has its own issues (non-uniqueness under different causal structures, Heskes et al. 2020).
- Interventional SHAP's OOD evaluations may themselves be a source of instability under high correlation, confounding with the first-mover bias mechanism.

The paper doesn't investigate whether the instability patterns change under conditional SHAP. If conditional SHAP produces stable attributions from a single model (because it doesn't evaluate the model at OOD feature combinations), then the "problem" the paper addresses may be partially an artifact of the SHAP variant chosen.

**ArXiv acceptability:** Acceptable. This is a standard choice and the limitation is acknowledged. But a sophisticated reviewer will ask whether the first-mover bias mechanism operates through the split-selection path dependence (as claimed) or through the interaction between interventional SHAP's OOD evaluations and path-dependent feature utilization.

---

## Minor Issues

### m1. Breast Cancer Abstract Numbers Are Potentially Confusing
**Lines 140–143.** The abstract reports both comparisons: +0.40 over standard SB and +0.61 over SB(M=200). Including both is transparent but dense. The SB(M=200) comparison is the more impressive number but requires understanding the training-budget-matched distinction. Consider using only the more conservative +0.40 in the abstract.

### m2. Table 8 (Nonlinear DGP) Has Asymmetric Baseline Coverage
The nonlinear table includes LSM-T but not Random Selection, Naive Top-N, or Ensemble SHAP. The caption now includes a note about this (good), but the inclusion of LSM-T specifically (which wasn't in the linear sweep table either) is unexplained. Why LSM-T but not Random Selection?

### m3. No Convergence Analysis for Consensus Averaging
As K increases, do consensus SHAP values converge? The epsilon sensitivity table (Table 10) shows K_eff from 4 to 16 with stability varying <0.005, which implies fast convergence. A direct convergence curve (stability vs. K for fixed M and epsilon) would strengthen the argument and help practitioners choose K.

### m4. The "Pre-Specified Success Criteria" Are Suspiciously Clean
**Appendix C.** All 11 criteria pass. The paper correctly qualifies these as testing "under favorable conditions" (line 1870) with the caveat that they were not formally pre-registered. But 11/11 passes — with criteria like ">=50% of tests significant" (criterion 7) that appear calibrated to the expected result — invites skepticism about specification after the fact. The honest caveats help but don't fully address this.

**Recommendation:** Either remove this appendix (it adds little given the results are already in the main text) or add one criterion that DASH is expected to *fail* (e.g., "DASH outperforms SR on stability" — which it doesn't). Including a designed-to-fail criterion would make the pre-specification more credible.

### m5. "First-Mover Bias" Terminology
The phenomenon is essentially the well-known feature selection bias in boosted ensembles (Strobl et al. 2007 is cited) applied to post-hoc SHAP attributions. The paper correctly distinguishes its contribution from Strobl (Section 2, line 233: "we isolate and empirically characterize the specific mechanistic pathway... through which it concentrates SHAP-based feature attributions") but "coining" a new term for a known phenomenon applied to a new context may draw reviewer objections. The v3 revision already softened the "coin the term" phrasing, which helps.

### m6. Classification vs. Regression Stability Metric
The stability metric (mean pairwise Spearman on global importance vectors) is computed identically for regression and classification tasks. For Breast Cancer (classification), SHAP values are for the positive class probability. The interpretation is the same, but the paper doesn't note this distinction. A brief parenthetical would suffice.

### m7. Background Size B=100 Never Justified or Ablated
**Lines 1453–1457.** B=100 is used for all experiments. For Superconductor (81 features), this is barely above the feature dimension. The paper defers a sensitivity analysis to the journal version. For ArXiv, acceptable — B=100 is a common default. But it would be reassuring to note that TreeSHAP's exact computation doesn't depend on background size in the same way that KernelSHAP does (the background is used only for the interventional reference distribution, not for a sampling approximation).

### m8. Population Size Ablation Shows Non-Monotone Pattern
**Lines 1807–1811.** M=50 (0.9727) → M=100 (0.9719) → M=200 (0.9722) → M=500 (0.9722). The M=50→M=100 transition *decreases* stability by 0.0008. This is within noise, but the paper describes performance as "effectively invariant" and the figure caption says "invariant." A more careful phrasing would be "varies within noise (range 0.001)" rather than "invariant."

### m9. Ensemble SHAP Timing Missing
**Table 4 (line 928):** Ensemble SHAP timing is listed as "---" with the note that it "shares infrastructure with other methods and was not independently measured." This is vague. Ensemble SHAP trains a single 2000-tree model and computes SHAP once — its cost should be comparable to LSM. The "---" entry is unsatisfying.

### m10. Inline Bibliography
For ArXiv, `\begin{thebibliography}` is fine. For TMLR, this must be converted to a `.bib` file. Not an issue for current submission but worth noting for the authors' planning.

---

## Verification Against Codebase

I verified the following claims against the implementation:

| Claim | Code Location | Status |
|-------|--------------|--------|
| M=200 XGBoost models with random hyperparams | `population.py:119-172` | Verified |
| colsample_bytree in [0.1, 0.5] | `population.py:21` | Verified |
| K=30 via MaxMin greedy selection | `diversity.py:56-96` | Verified |
| Element-wise SHAP averaging | `consensus.py:54` | Verified |
| Interventional TreeSHAP | `consensus.py:47` | Verified |
| FSI = sigma / (mean\|SHAP\| + epsilon_0) | `diagnostics.py:34` | Verified |
| 4-way data split | `experiments/synthetic.py:99-111` | Verified |
| BCa bootstrap (not simple percentile) | `evaluation/__init__.py:92-144` | Verified |
| 8 baselines implemented | `baselines/__init__.py` | Verified (all 8 present) |
| Stability = mean pairwise Spearman | `evaluation/__init__.py:79-89` | Verified |
| Delta-based early stopping in diversity | `diversity.py:81` | Verified |

**No discrepancies found between paper claims and code implementation.**

---

## Numerical Spot-Check

The paper sources all results from `notebooks/demo_benchmark_6.ipynb`. The CLAUDE.md key results match the paper's tables:
- rho=0.9: DASH 0.977 vs SB 0.958 vs LSM 0.938 — **matches Table 3**
- Breast Cancer: DASH 0.930 vs SB(M=200) 0.317 — **matches Table 7**
- Superconductor: DASH 0.962 vs SB 0.830 — **matches Table 7**
- California Housing: DASH 0.982 vs SB 0.967 — **matches Table 7**

---

## Summary of Action Items by Priority

### For ArXiv (recommended but not blocking)
1. Include T-scaling figure as an appendix (closes M1 cleanly)
2. Add RF baseline at rho=0.9 if feasible (addresses M2, the most predictable reviewer question)
3. Simplify abstract to use only the conservative Breast Cancer comparison (+0.40)
4. Add one-sentence note to Table 8 caption explaining LSM-T inclusion
5. Consider removing or revising Appendix C pre-specified criteria (add a designed-to-fail criterion)

### For TMLR (future, not needed for ArXiv)
1. Increase to N_reps = 50 (M4)
2. Add RF baseline to all tables (M2)
3. Include T-scaling direct experiment (M1)
4. Add convergence curve for consensus averaging (m3)
5. Ablate background size B (m7)
6. Investigate conditional SHAP variant (M5)
7. Convert to .bib format and TMLR style (m10)
8. Produce vector figures (PDF/PGF)

### No action needed
- The SR equivalence framing is already excellent (S2)
- The accuracy/equity circularity caveat is present and well-placed (S5)
- The nonlinear scope boundary is honestly described
- The stability selection contrast is present (Section 6.1)
- Alt text on figures is good practice
- Bibliography is alphabetically sorted

---

## Final Assessment

This paper identifies a real phenomenon (first-mover bias via sequential residual dependency), demonstrates the correct remedy (model independence), and honestly characterizes the contribution hierarchy — including the finding that a trivial baseline (SR) matches the proposed method. The writing is clear, the experiments are thorough, and the limitations are forthrightly stated.

The main vulnerabilities are the missing RF baseline (M2), the untested T-scaling hypothesis (M1), and the narrow unique value of the DASH pipeline over simpler alternatives (M3). For an ArXiv pre-print, all three are acceptable given the honest limitation acknowledgments. For TMLR, M2 and M1 will need to be addressed.

**Recommendation: Post to ArXiv as-is, or with the T-scaling appendix figure added (trivial cost, closes the most noticeable gap).** The paper is above the quality bar and the intellectual honesty will serve it well in subsequent review.
