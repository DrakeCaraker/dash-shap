# Peer Review: draft_v5_preprint.tex

**Paper:** "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution"
**Authors:** Caraker, Arnold, Rhoads (2026)
**Reviewer standard:** ArXiv pre-print release readiness
**Date:** 2026-03-16

---

## Overall Assessment

**Verdict: Ready for ArXiv with minor revisions (detailed below).**

This is a well-structured, clearly written paper that makes a genuine contribution to the XAI literature. The mechanistic claim (sequential residual dependency causes SHAP instability) is cleanly articulated, the experimental design is thorough, and the writing is unusually honest about limitations (the accuracy/equity circularity caveat, the nonlinear scope boundary, the SR equivalence). The paper reads more like a careful technical report than a hype piece, which is appropriate for the claim being made.

The v5 draft has already incorporated numbers from `demo_benchmark_6.ipynb` and addressed the major issues flagged in `REVIEW_v6.md`. What remains are issues of presentation polish, a few analytical gaps, and some framing choices that could be tightened.

**Strengths:**
- Mechanistic clarity: The first-mover bias concept is well-defined, and the LSM comparison is an elegant controlled experiment
- Intellectual honesty: The SR equivalence is presented as the paper's *most important* result, not buried
- The accuracy/equity circularity caveat (Section 5.1) is exemplary scientific transparency
- Clean two-tier separation (dependent vs. independent) is compelling and reproducible
- Comprehensive experimental design with appropriate statistical testing
- Variance decomposition provides direct mechanistic evidence
- Diagnostics (FSI, IS Plot) have genuine practical value

**Weaknesses (detailed below):**
1. Some theoretical claims remain informal despite opportunities for tighter formalization
2. The "pre-registration" framing is stretched
3. Missing comparisons with closely related work (stability selection, Rashomon-set methods)
4. Nonlinear DGP results need slightly more careful handling
5. Several presentation and formatting issues

---

## Major Issues

### M1. The "Empirical Hypothesis" (Section 3.3) Should Be Stronger or Weaker

The paper states an "empirical hypothesis" (line 351-358) that first-mover bias increases with T (tree count). This is tested in the first-mover bias isolation experiment (referenced via the concentration figure), but the paper never explicitly presents those results as a formal test of this hypothesis. The hypothesis says the gap increases in T, and Figure 2 shows it for one snapshot, but the T-scaling experiment results from `run_experiments.py --experiments first_mover_bias` are not included in the paper.

**For ArXiv:** Either include the T-scaling results (even briefly) to validate the hypothesis, or weaken the statement to a "motivating observation." The current middle ground where you state a testable hypothesis and then don't directly test it in the paper body is awkward.

### M2. Pre-Registration Framing (Appendix C)

The paper says "We pre-registered eleven pass/fail criteria before running the final benchmark." The v5 header notes this was "softened" with a scope caveat, which is good. But for ArXiv: pre-registration typically implies a public, timestamped commitment to an analysis plan *before* data collection, lodged with an independent registry (e.g., OSF, AsPredicted). If the criteria were written down before running `demo_benchmark_6.ipynb` but not formally registered, this is better described as "pre-specified" or "prospectively defined." Actual pre-registration is a strong claim with a specific methodological meaning.

**Recommendation:** Change "pre-registered" to "pre-specified" unless there is an actual public registration record. This is the kind of thing reviewers at TMLR will flag.

### M3. Missing Connection to Stability Selection (Meinshausen & Buhlmann 2010)

The related work cites stability selection but doesn't engage with it deeply. Stability selection addresses a closely related problem (which features are consistently selected across subsamples?) using a strikingly similar mechanism (repeated resampling + aggregation). The key difference is that stability selection perturbs *data* while DASH perturbs *models*. This comparison deserves more than one sentence, because a reviewer will ask: "How does DASH compare to simply computing SHAP on stability-selected features?" or "Isn't this just stability selection applied to explanations?"

**For ArXiv:** A paragraph in Discussion explicitly contrasting DASH with stability selection (data perturbation vs. model perturbation, feature selection vs. feature attribution) would preempt this obvious reviewer question.

### M4. The Nonlinear Table Numbers Don't Match the Text

Table 8 (nonlinear DGP) reports:
- rho=0.0: DASH=0.934, SB=0.933
- rho=0.5: DASH=0.852, SB=0.849

But line 1290-1291 says: "At rho=0.5, DASH shows a marginal advantage (0.8520 vs. 0.8492)." These are 4-decimal values that don't match the 3-decimal table values (0.852 vs 0.849). The table appears to use different rounding. While these are close, having inline text cite numbers at higher precision than the table is confusing.

**Recommendation:** Either report consistent decimal places in both table and text, or don't cite the 4-decimal values inline when the table shows 3.

---

## Minor Issues

### m1. Inline Bibliography

The bibliography uses `\begin{thebibliography}` instead of a `.bib` file. This is fine for ArXiv but will need to be converted for TMLR submission. Note for the authors: the bibliography is not alphabetically sorted (Altmann, Meinshausen, Heskes, Strobl, Hooker, Janzing, Aas, Dong, Semenova, Covert, Molnar, Slack come after the main entries). `plainnat` typically sorts alphabetically when using `.bib`, so the current order looks slightly disorganized.

### m2. Table Numbering Mismatch with REVIEW_v6.md

The REVIEW_v6.md refers to Tables 3, 5, 6, 7, 8, 9 which don't correspond to the actual table labels in v5 (tab:sweep, tab:extended, tab:epsilon, tab:realworld, tab:nonlinear, tab:cost). This is an internal documentation issue, not a paper issue, but worth noting for consistency.

### m3. Stochastic Retrain Outperforms DASH on Stability at Most rho Levels

Looking at Table 3 carefully: SR has higher stability than DASH at rho=0.0 (0.975 vs 0.972), rho=0.5 (0.980 vs 0.977), rho=0.7 (0.980 vs 0.977), and rho=0.95 (0.979 vs 0.977). They tie only at rho=0.9 (0.977 vs 0.977). The paper's framing emphasizes the tie at rho=0.9 and the "independence principle," but a careful reader will notice SR is *consistently slightly better* on the headline metric. The text says "DASH stability is effectively flat (0.972-0.977)" but SR's is 0.975-0.980, which is both flat *and* higher.

The paper handles this honestly by positioning DASH's advantages as speed, diagnostics, and equity (Section 5.2). But the abstract's emphasis on the 0.977 tie at rho=0.9 is slightly cherry-picked. This won't block ArXiv but will likely be noted by TMLR reviewers.

**Recommendation:** Consider adding a sentence acknowledging that SR achieves marginally higher point estimates on stability across most rho levels, while noting the differences are not statistically significant. This preempts the reviewer observation and reinforces the honesty of the paper.

### m4. Breast Cancer Baseline Asymmetry

The real-world table uses SB(M=200) for Breast Cancer but standard SB for Superconductor and California Housing. This inconsistency is acknowledged in the Breast Cancer results (the M=200 dagger notation), but a reader may wonder: what is standard SB stability on Breast Cancer? If it's ~0.534 (from the earlier review document), that's still a massive improvement. The asymmetry in baseline choice across datasets deserves a brief justification in the text.

### m5. Ensemble SHAP is Underspecified

Ensemble SHAP (Paillard et al. baseline) appears only in Table 5 at rho=0.9 (stability 0.956). It's not included in the sweep table or any other analysis. Given that this method is the paper's direct foil (the paper argues *against* the Paillard et al. recommendation), it deserves sweep results. Was this computationally prohibitive? If so, a note explaining why would help.

### m6. "Diversified Aggregation of SHAP" -- the Acronym

DASH is a somewhat forced acronym ("Diversified Aggregation of SHAP" -- the 'H' is silent?). This is fine and common in ML papers, but note that `dash` as a Python package name shadows Plotly Dash, which is widely used. The import convention note in CLAUDE.md suggests this is already a known issue internally. Ensure the package naming on PyPI (if planned) doesn't conflict.

### m7. Figure Quality

The paper references several `.png` figures. For ArXiv, PNG is acceptable. For journal submission, vector formats (PDF, PGF) are strongly preferred for line plots and scatter plots. The concentration plot (Figure 2), sweep plot (Figure 3), IS plot, and disagreement map should all be vector graphics.

### m8. Missing Standard Errors on Some Metrics

Tables report stability with ±SE but accuracy, equity, and RMSE without error bars in the main sweep table. The REVIEW_v6.md noted this. For ArXiv this is acceptable (the focus is clearly on stability), but TMLR reviewers may request ±SE on all metrics. The data exists in the notebooks.

### m9. Variance Decomposition Caveats Are Buried

The variance decomposition (Section 5.2, paragraph 4) is presented as clean: "Model-selection variance accounts for 54% of Single Best's total instability but only 24% of DASH's." But the EXPERIMENT_GUIDE.md notes a significant caveat (C5): 1-stability is a proxy for variance, not actual variance; the additive decomposition doesn't hold exactly; and the marginal design provides only directional evidence. None of these caveats appear in the paper. For ArXiv, at minimum add a brief caveat that the decomposition is approximate.

### m10. Naive Top-N and Random Selection Are Buried

These methods are important ablation controls (they isolate the value of diversity selection vs. simple averaging), but they appear only in Table 5. Their story is actually quite interesting: Random Selection achieves 0.976 stability, nearly matching DASH's 0.977, suggesting that *diversity selection specifically adds almost nothing to stability*. DASH's equity advantage (0.176 vs 0.187) is modest. This undercuts the importance of Stage 3 (Diversity Selection) in the pipeline.

The paper doesn't address this directly. A reviewer will ask: "If random selection from the filtered pool achieves the same stability, why do you need MaxMin diversity selection?" The answer seems to be equity, but the difference is small and not significance-tested.

### m11. Page Count

At ~17 pages of body + 3 pages of appendix + 3 pages of references, this is a substantial paper. For ArXiv, no issue. For TMLR, check their format requirements -- this may need trimming.

---

## Statistical and Methodological Notes

### S1. Wilcoxon Tests on Accuracy/Equity but Not Stability

The paper correctly notes that stability is computed as a single aggregate across all reps, making paired per-rep testing impossible. This is a fundamental limitation of the stability metric's definition. The paper doesn't propose any alternative (e.g., split-half stability, leave-one-out stability) that would enable statistical testing. For ArXiv this is fine; for TMLR, consider whether a bootstrap comparison of stability point estimates would be informative.

### S2. Effect Size Interpretation

Cohen's d > 1.4 for DASH vs LSM is described as "large." With n=20, these effect sizes are well-powered. But Cohen's d for DASH vs SR on equity is -0.21 (small), and the paper correctly says it's not significant. Good.

### S3. Multiple Testing Scope

The 26 tests within the linear sweep are corrected via Holm-Bonferroni. But the paper also makes claims about real-world datasets, nonlinear DGP, and ablation studies without joint correction. This is standard practice (different experiment families) but worth noting.

---

## Presentation Issues

### P1. The Abstract Needs Tightening

The abstract is 200+ words and tries to convey four distinct contributions plus results. For ArXiv, this is fine but verbose. Consider: the phrase "numerically identical" is doing a lot of work (they're identical at 3 decimal places with different SEs; this is noted honestly but the abstract makes it sound like exact equality).

### P2. The Conclusion Repeats the Introduction

Lines 1443-1463 closely mirror the introduction's contribution list. This is common but the conclusion could add more forward-looking content instead of restating what was already said.

### P3. Section 5 Is Very Long

The Results section spans ~600 lines (5.1 through 5.6). Consider whether some content (epsilon sensitivity, ablation, timing) could move to an appendix for the ArXiv version, keeping the main body focused on the mechanism experiment, independence principle, and real-world validation.

### P4. Equation Numbering

Several important quantities (FSI, stability, equity) have equation numbers, which is good. But the consensus equation (Eq. 3) is the mathematical heart of the method and could be more prominently featured.

---

## Questions the Authors Should Prepare For (TMLR Submission)

1. **"Why not use random forests as the base learner?"** RF trees are independent by construction. The paper notes this in Limitations as future work, but a reviewer may insist this is a necessary baseline, not future work.

2. **"Is DASH anything more than bagging explanations?"** The paper should have a clear answer: DASH adds forced feature restriction (colsample_bytree), performance filtering, and diversity selection beyond simple model averaging. But Random Selection (which omits diversity selection) achieves nearly identical results...

3. **"How does this compare to conditional SHAP / causal SHAP?"** These methods address the correlated features problem at the Shapley value level rather than the model level. The related work mentions them but doesn't empirically compare.

4. **"20 reps seems low for a stability paper."** The paper acknowledges this in Limitations (underpowered for small effects). For TMLR, consider 50 reps (which is the PAPER_CONFIG's N_REPS=50 setting noted in CLAUDE.md, suggesting this is already planned).

5. **"The M ablation shows no benefit past M=50. Why is M=200 the default?"** The ablation results (0.9727 at M=50 vs 0.9722 at M=200) actually show M=50 performing marginally *better*. The paper says "performance plateaus early, confirming that M=200 is sufficient" -- but the reviewer will ask: sufficient for what? The data suggests M=50 is already optimal and M=200 is wastefully large.

---

## ArXiv-Specific Checklist

- [x] Compiles without errors (PDF exists at v5)
- [x] Figures referenced correctly
- [x] No broken cross-references visible
- [x] License/distribution statement: **MISSING** -- consider adding CC-BY or similar
- [x] Author emails provided
- [x] ORCID for corresponding author
- [ ] ORCIDs for co-authors: missing
- [x] Repository link provided and consistent
- [x] Abstract within 300-word limit
- [x] Keywords provided
- [ ] ACM/arXiv subject classification: not specified
- [x] No supplementary material that should be submitted separately

---

## Summary of Required Actions

**Must-fix for ArXiv (P0):**
1. Change "pre-registered" to "pre-specified" (Appendix C) unless there's an actual registration
2. Add variance decomposition caveat (approximate, not exact additive decomposition) -- one sentence in Section 5.2
3. Ensure consistent decimal precision between inline text and tables (nonlinear DGP section)

**Should-fix for ArXiv (P1):**
4. Add 1-2 sentences about SR's marginally higher stability across most rho levels to preempt reviewer observation
5. Add a paragraph in Discussion contrasting DASH with stability selection (data vs. model perturbation)
6. Briefly explain why Ensemble SHAP isn't in the sweep table
7. Add brief justification for asymmetric baseline choice across real-world datasets

**Nice-to-have for ArXiv (P2):**
8. Include T-scaling results from first_mover_bias experiment to validate the empirical hypothesis
9. Address the Random Selection near-equivalence and what it implies for diversity selection's value
10. Add arXiv subject classification
11. Consider shortening Section 5 by moving ablation/timing to appendix

**The paper is ready for ArXiv posting with the P0 items fixed.** The P1 items would strengthen it noticeably. The overall quality is high -- the mechanistic insight is real, the experimental design is thorough, and the writing is clear and honest about limitations.
