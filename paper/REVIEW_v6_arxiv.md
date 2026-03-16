# Independent Peer Review: draft_v6_preprint.tex

**Paper:** "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution"
**Authors:** Caraker, Arnold, Rhoads (2026)
**Reviewer standard:** Rigorous ML venue peer review, with ArXiv pre-print norms noted
**Date:** 2026-03-16
**Draft version:** v6 (incorporates fixes from REVIEW_v5_arxiv.md)
**Disclosure:** This review was conducted by an AI system (Claude). Prior reviews (REVIEW_v5_arxiv.md, REVIEW_v6.md, REVIEW_v7.md) were read for context but this review is independently structured.

---

## Overall Assessment

**ArXiv verdict: Ready to post.** The paper is above the quality bar for an ArXiv pre-print. The core insight is genuine, the experiments are well-designed, the writing is clear, and—critically—the paper is unusually honest about the limits of its own contribution (the SR equivalence, the accuracy/equity circularity caveat, the nonlinear scope boundary). This intellectual honesty is the paper's greatest asset and will serve it well under formal review.

**Conference equivalent rating: 6.5–7/10.** The contribution is primarily empirical-diagnostic. The mechanistic characterization (first-mover bias via sequential residual dependency) is the real value; DASH as a pipeline is secondary to the finding that model independence suffices.

---

## Strengths

**S1. Clean mechanistic framing.** The paper isolates a specific mechanism (sequential residual dependency) rather than vaguely attributing SHAP instability to "multicollinearity." The Large Single Model comparison is an elegant controlled experiment: same total tree count, same colsample_bytree, different training structure (sequential vs. independent). This is the kind of clean ablation that reviewers love.

**S2. The SR equivalence is presented as the headline result.** Most papers would bury the finding that a trivial baseline (Stochastic Retrain) matches their proposed method on the headline metric. This paper calls it "the most important result" (Section 5.2, line ~1036). This is exemplary scientific transparency and actually strengthens the paper's contribution: the claim shifts from "DASH is best" to "independence is what matters," which is a more fundamental and broadly useful insight.

**S3. Comprehensive experimental design.** Nine methods, five correlation levels, two DGPs (linear + nonlinear), three real-world datasets, 20 repetitions each, appropriate statistical tests with multiple-testing correction. The four-way data split (train/val/explain/test) is carefully motivated and prevents the most common source of SHAP overfitting.

**S4. Honest limitations section.** The paper explicitly acknowledges: the accuracy/equity circularity (Section 5.1, lines 862–872), the nonlinear scope boundary (Section 5.5), the interventional SHAP limitation under correlation (Section 6.3), the underpowered comparison between DASH and SR (Section 6.3), and the missing random forest baseline (Section 6.3). This is thorough and preemptive.

**S5. Practical diagnostics with real value.** The FSI and IS Plot are genuinely useful. An unsupervised collinearity detector that works from SHAP disagreement across an ensemble—without needing the correlation matrix—has clear practical applications. The four-quadrant taxonomy (robust drivers, collinear clusters, confirmed unimportant, fragile interactions) is intuitive and actionable.

**S6. Two-tier separation is striking.** The clean partition between dependent methods (stability 0.938–0.964) and independent methods (0.976–0.977) at rho=0.9, with inter-tier gap ~0.01–0.04 dwarfing intra-tier gap ~0.001, is compelling evidence for the independence principle. The BCa bootstrap CIs are non-overlapping, reinforcing this.

**S7. Reproducibility infrastructure.** The repository link, fixed seeds, checkpointed notebooks, CLI experiment runner, and reproducibility statement are above average for the field.

---

## Major Issues

### M1. The Theoretical Contribution Remains Informal

The "empirical hypothesis" (Section 3.3, lines 384–396) states that first-mover concentration increases with tree count T, but the paper only tests this indirectly through the LSM comparison and refers readers to the code repository for a direct T-scaling experiment. This creates an awkward gap: the paper formalizes a testable hypothesis, then doesn't directly test it in the paper body.

The hypothesis itself is intuitively obvious—more sequential iterations amplify path dependence—but formalizing it would strengthen the contribution. Even a simple probabilistic argument (e.g., under a simplified split-selection model, the probability that feature j is selected at tree t given it was selected at tree 1 increases monotonically with t) would elevate the paper from "empirical characterization" to "mechanistic explanation with theoretical grounding."

**For ArXiv:** Acceptable as-is. The empirical evidence is strong enough to support the qualitative claim. But the mismatch between "empirical hypothesis" framing and indirect testing is noticeable.

**Recommendation:** Either (a) include the T-scaling results from the codebase as an appendix figure (one panel, 5 lines of results—minimal page cost), or (b) soften "empirical hypothesis" to "motivating observation." Currently the paper occupies an uncomfortable middle ground.

### M2. DASH's Method Contribution Is Thin Given the Independence Finding

The paper's own results demonstrate that the operative mechanism is model independence, not any particular pipeline design. Random Selection achieves 0.976 stability vs. DASH's 0.977. Stochastic Retrain achieves 0.977. The three practical advantages claimed for DASH over SR are:

1. **Speed** (1.7x faster): This is real but the mechanism is somewhat incidental (fewer SHAP evaluations due to diversity selection pruning).
2. **Diagnostics** (FSI, IS Plot): These are genuinely novel and useful. But they don't require the full DASH pipeline—you could compute FSI/IS Plot from any multi-model ensemble.
3. **Equity** (CV 0.176 vs. 0.182): Not statistically significant (acknowledged).

This means the DASH pipeline's unique value over "train K models with different seeds and average their SHAP values" is primarily the diagnostic tools. The paper could be stronger if it reframed the contribution hierarchy more explicitly: (1) finding (first-mover bias mechanism), (2) principle (independence suffices), (3) diagnostics (FSI/IS Plot), (4) pipeline (DASH as convenient operationalization).

**For ArXiv:** The current framing is acceptable. The paper already acknowledges the SR equivalence prominently. But a reviewer at TMLR will press this point hard.

**Recommendation:** No change required for ArXiv, but consider restructuring the abstract for any journal submission to lead with the finding/principle rather than the pipeline.

### M3. The Nonlinear DGP Results Weaken the Generality Claim

Under the nonlinear DGP:
- At rho=0.0 and rho=0.5, DASH and SB are effectively tied (0.934 vs. 0.933 and 0.852 vs. 0.849)
- At rho=0.5, SR actually slightly outperforms DASH (0.855 vs. 0.852)
- Overall stability drops from ~0.93–0.98 (linear) to ~0.85–0.93 (nonlinear)
- DASH's clear advantage only emerges at rho >= 0.7

The paper handles this reasonably in Section 5.5 ("genuine scope boundary"), but the Discussion (Section 6.1, lines 1354–1359) is vague about the mechanism: "different models may capture qualitatively different interaction structures whose averaging introduces noise." This is hand-waving. What specifically about nonlinear interactions defeats the independence-based cancellation?

**For ArXiv:** The honesty about the scope boundary is sufficient. The mechanism for nonlinear degradation is an open question that need not be fully answered here.

**Recommendation:** Add one sentence clarifying the hypothesized mechanism: e.g., "Under nonlinear DGPs, the SHAP attribution for a feature depends on which interactions the model has learned, not just which feature was selected first. Since different models may learn qualitatively different interaction structures, averaging their SHAP values introduces model-disagreement noise that is not resolved by independence." This turns hand-waving into a testable hypothesis.

### M4. Missing Random Forest Baseline Is a Significant Gap

The paper's central claim—that model independence resolves first-mover bias—has an obvious corollary: random forests, whose trees are independent by construction, should exhibit higher baseline SHAP stability than single XGBoost models. If RF SHAP stability matches or exceeds DASH, the practical recommendation simplifies to "use random forests." If RF stability is intermediate (better than single XGBoost, worse than DASH), it provides direct evidence that independence alone is not sufficient and DASH's additional engineering (filtering, diversity selection) adds value.

The paper acknowledges this gap (Limitations, line 1452–1455) and defers to the journal version. For ArXiv, this is borderline acceptable—the paper already frames it as future work. But it's the single most predictable reviewer question, and the experiment is cheap to run.

**For ArXiv:** Flagged as a known limitation. Acceptable but barely. A reviewer might argue this should have been included.

**Recommendation:** If time permits, add RF results at rho=0.9 before ArXiv posting. If not, the limitation acknowledgment is adequate.

### M5. Stability Metric Cannot Be Directly Tested

The headline metric (pairwise Spearman stability across repetitions) is a single aggregate, making paired hypothesis testing impossible. The paper provides BCa bootstrap CIs as a partial remedy (Table 4 caption: SB [0.952, 0.963] vs. DASH [0.975, 0.978], non-overlapping). However:

1. Non-overlapping CIs is a conservative test (the actual significance threshold for two-sample comparison is wider than individual CIs suggest).
2. No formal hypothesis test is conducted on the stability difference itself.
3. The Wilcoxon tests in Table 6 are on accuracy and equity—not stability.

The paper correctly notes this limitation (lines 998–1001). For ArXiv, this is acceptable—BCa CIs with non-overlapping intervals are adequate directional evidence. For journal submission, a bootstrap permutation test on the stability difference would be needed.

**For ArXiv:** Acceptable. The non-overlapping BCa CIs are informative.

---

## Minor Issues

### m1. Abstract Length and Density (Lines 117–151)

At ~220 words, the abstract tries to convey: (a) the phenomenon, (b) the mechanism, (c) the LSM counterexample, (d) the independence principle, (e) the SR equivalence, (f) DASH numbers, (g) real-world numbers, and (h) the diagnostics. This is a lot. The result is dense but readable.

The sentence "Both our proposed method, DASH (Diversified Aggregation of SHAP), and simple seed-averaging (Stochastic Retrain) restore stability by breaking the sequential dependency chain" (lines 134–136) is arguably the paper's most important single claim. It's buried in the second paragraph. Consider leading with it.

**For ArXiv:** Fine. ArXiv abstracts tend to be long.

### m2. The LSM Tree-Count Matching Is Approximate

Line 196: "M × ~75 ≈ 15,000 trees, where ~75 is the average number of trees per population model after early stopping." The ~75 is a sample average that varies across runs. The parenthetical is long and breaks reading flow. More importantly, the "tree-count-matched" claim is approximate, and the paper acknowledges this (lines 756–759: "tree-count-matched refers here to the total number of trees... not wall-clock time or FLOPs"). The clarification is good.

**Recommendation:** No change needed. The clarification is present.

### m3. SR Consistently Outperforms DASH on Stability Point Estimates

Looking at Table 3 carefully:
- rho=0.0: SR 0.975 vs. DASH 0.972
- rho=0.5: SR 0.980 vs. DASH 0.977
- rho=0.7: SR 0.980 vs. DASH 0.977
- rho=0.9: SR 0.977 vs. DASH 0.977 (tie)
- rho=0.95: SR 0.979 vs. DASH 0.977

SR has higher or equal stability at every single rho level. The paper now acknowledges this (lines 1038–1041: "SR achieves marginally higher stability point estimates than DASH at most rho levels... though these differences are small (<=0.003) and not statistically significant"). This is honest and adequate.

**For ArXiv:** Addressed. No further action needed.

### m4. Equity Metric Definition Could Be Confusing

The equity metric (Eq. 7, line 718) is the mean coefficient of variation within correlated groups—i.e., *lower is better*. But the paper sometimes uses "equity" without clarifying directionality. Table 3 headers say "Equity (CV)" which helps. Still, a reader scanning the table might initially assume higher is better.

**Recommendation:** Consider adding "(lower is better)" to the equity column header or to the metric definition.

### m5. Breast Cancer Table Now Shows Both SB and SB(M=200)

Table 7 (lines 1196–1220) includes both SB (0.534) and SB(M=200) (0.317). The abstract uses the SB(M=200) comparison ($0.32$ to $0.93$, $+0.61$). This is the training-budget-matched comparison, which is fair. The standard SB comparison ($0.53$ to $0.93$, $+0.40$) also appears. Having both is transparent.

However, the abstract cites *both* comparisons (lines 140–143), which may confuse readers unfamiliar with the notation. Consider mentioning only the more conservative comparison (+0.40 over standard SB) in the abstract and reserving the SB(M=200) comparison for the results section.

**For ArXiv:** Minor. Both are defensible.

### m6. Nonlinear DGP: Incomplete Method Comparison

Table 8 includes SB, LSM, LSM-T, SR, and DASH—but not Random Selection, Naive Top-N, Ensemble SHAP, or SB(M=200). The paper justifies excluding some baselines from the sweep table (lines 769–772), but the nonlinear table silently includes LSM-T (which wasn't in the linear sweep table) while excluding others. The asymmetry in baseline coverage across tables is confusing.

**Recommendation:** Add a brief note to Table 8's caption explaining which methods are included and why. For example: "Five methods with distinct training structures shown; Random Selection and Naive Top-N (which performed similarly to DASH in the linear regime) are available in the code repository."

### m7. The "Pre-Specified" Framing Is Now Correct

The v6 draft uses "pre-specified" (Appendix C, line 1859) with an honest caveat: "written into the experimental notebook prior to execution, though not lodged with a formal pre-registration registry." This addresses the concern from REVIEW_v5_arxiv.md. Good.

### m8. Variance Decomposition Caveat Is Now Present

The footnote (lines 982–986) clarifies that 1–stability is a proxy for instability, not actual variance, and that the decomposition is approximate. This is adequate for ArXiv.

### m9. Bibliography Is Now Alphabetically Sorted

The v6 draft's bibliography appears alphabetically sorted (Aas, Altmann, Alvarez-Melis, Breiman, Chen, ..., Zou). This addresses the concern from REVIEW_v5_arxiv.md. Good.

### m10. Stability Selection Contrast Is Now Present

Section 6.1 (lines 1361–1378) contains a dedicated paragraph contrasting DASH with stability selection. The key distinction (data perturbation vs. model perturbation, binary selection indicators vs. continuous attributions) is clearly articulated. This addresses REVIEW_v5_arxiv.md item M3. Good.

### m11. Timing Note for Random Selection vs. DASH Is Now Present

Table 9 (lines 1847–1853) includes an italicized note explaining why Random Selection (287.2s) is slower than DASH (140.3s). However, the explanation is somewhat hand-wavy: "lack of diversity guarantees leads to higher variance in per-run K_eff and additional overhead from the random sampling procedure across the full filtered pool." The actual mechanism should be straightforward: Random Selection computes SHAP for K=30 randomly-chosen models (some of which may be near-duplicates), while DASH's diversity selection may stop early when the minimum distance falls below delta, yielding K_eff < 30. The note should state this more concretely.

**Recommendation:** Revise the timing note to: "DASH's diversity selection typically selects K_eff < K_max models (10–15 at epsilon=0.08), reducing SHAP evaluations compared to Random Selection's fixed K=30."

### m12. Figure Alt Text Is Good Practice

The paper includes alt text for all figures (e.g., line 881: "Bar chart showing per-feature importance within a correlated group..."). This is excellent accessibility practice and not commonly seen in ML papers. It also helps ArXiv's HTML rendering. Keep this.

### m13. The Conclusion Is Repetitive But Adequate

The conclusion (lines 1482–1513) restates the three lines of evidence and the real-world results. It closely mirrors the introduction and Section 5.2. For ArXiv, this is standard. For a journal version, the conclusion could add more forward-looking content (the "broader implications" paragraph in Section 6.4 is more interesting than the repetitive summary).

### m14. Missing ORCIDs for Bryan Arnold

Drake Caraker and David Rhoads have ORCIDs; Bryan Arnold does not. For ArXiv, this is fine—ORCIDs are optional. For journal submission, consider adding Arnold's ORCID if available.

---

## Statistical and Methodological Notes

### S1. Wilcoxon Tests Are Appropriate

Paired Wilcoxon signed-rank tests are the right choice for comparing methods across 20 paired repetitions. The data is paired (same data regeneration seed per repetition), non-parametric is appropriate (no assumption of normal differences), and Holm-Bonferroni correction is less conservative than Bonferroni while still controlling family-wise error rate. The "selected comparisons shown from 26 total tests" note (Table 6 caption) is transparent.

### S2. Effect Sizes Are Well-Calibrated

Cohen's d > 1.4 for DASH vs. LSM (large effect, well-powered at n=20). Cohen's d = +0.05 and -0.21 for DASH vs. SR (negligible/small, correctly reported as n.s.). The paper doesn't over-claim on small effects.

### S3. BCa Bootstrap CIs Are Correctly Applied

Bias-corrected and accelerated bootstrap CIs are the gold standard for non-normal statistics like Spearman correlations. The non-overlapping intervals (SB [0.952, 0.963] vs. DASH [0.975, 0.978]) provide strong directional evidence even without a formal hypothesis test.

### S4. Multiple Testing Scope Is Standard

Within-family correction (Holm-Bonferroni across 26 tests in the linear sweep) without cross-family correction (between linear, nonlinear, and real-world experiments) is standard practice. No issue.

### S5. 20 Repetitions: Sufficient for Large Effects, Underpowered for Small

The paper acknowledges this (Section 6.3, line 1442): "With N_reps = 20, the Wilcoxon test is underpowered for the small effect sizes between DASH and Stochastic Retrain." This is correct. For the large effects that matter (DASH vs. SB, DASH vs. LSM), 20 reps is adequate. For distinguishing DASH from SR, it is not—but the paper's claim is that they're equivalent, so underpowering actually strengthens this (failure to reject the null supports equivalence, though this is not a formal equivalence test).

### S6. No TOST Equivalence Test for DASH vs. SR

The paper claims DASH and SR are equivalent on stability/accuracy/equity, but uses failure-to-reject (n.s. Wilcoxon) rather than a formal equivalence test (e.g., Two One-Sided Tests with a pre-specified equivalence margin). This is a common practice but technically incorrect: absence of evidence is not evidence of absence. With n=20, even moderate differences might fail to reach significance.

**For ArXiv:** Acceptable. The paper is careful to say "not significant" rather than "equivalent," and the identical point estimates (0.977 vs. 0.977) provide additional evidence.

**Recommendation for journal:** Consider adding a TOST equivalence test with a meaningful margin (e.g., delta=0.01 in stability) to formally support the equivalence claim.

---

## Presentation Quality

### Writing Quality: Strong

The prose is clear, precise, and economical. Technical terms are defined on first use. The paper avoids hype and overclaiming. The caveat paragraphs (accuracy/equity circularity, variance decomposition approximation, nonlinear scope boundary) demonstrate mature scientific writing.

### Figure Quality: Adequate for ArXiv

PNG figures are standard for ArXiv. The figure descriptions in captions are detailed and informative. Alt text is included. For journal submission, vector formats (PDF/PGF) would be needed.

### Table Design: Good

Tables are clean, well-labeled, and consistent. The Dependent/Independent grouping (Tables 3, 4) with rotated labels is effective. Standard errors are provided for stability. The dagger notation for training-budget-matched baselines is clear.

### Mathematical Notation: Consistent

The paper uses rho for within-group correlation, varepsilon for the performance threshold, epsilon_0 for the FSI smoothing constant, and delta for the diversity threshold. These are defined on first use and used consistently throughout. The epsilon/varepsilon distinction (noted in the v6 header) is now clean.

---

## ArXiv-Specific Checklist

- [x] Paper compiles (PDF exists at v5; v6 should compile given the nature of changes)
- [x] All figures referenced with correct labels
- [x] No visible broken cross-references (checked: all \ref and \label pairs match)
- [x] CC-BY 4.0 license statement present (lines 1536–1539)
- [x] Author emails provided for all three authors
- [x] ORCIDs for Caraker and Rhoads (Arnold missing—optional)
- [x] Repository link provided (https://github.com/DrakeCaraker/dash-shap)
- [x] Abstract within 300-word limit (~220 words)
- [x] Keywords provided (line 154)
- [x] arXiv subject classification noted in comment (line 158: cs.LG primary, stat.ML cross-list)
- [x] Bibliography present and alphabetically sorted
- [x] Pre-specified criteria clearly labeled as not formally pre-registered (line 1861)
- [x] Reproducibility statement present (lines 1517–1528)
- [x] No supplementary material requiring separate upload

---

## Summary of Issues by Priority

### P0: Blocking for ArXiv — NONE

The paper has no blocking issues for ArXiv posting. Prior review rounds have addressed the critical problems (stale numbers, pre-registration framing, missing caveats, decimal precision).

### P1: Should Fix Before ArXiv (improves quality noticeably)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 1 | T-scaling results for empirical hypothesis | Section 3.3 / Appendix | Either include the direct T-scaling experiment from the codebase (even 2-3 lines of results), or soften "empirical hypothesis" to "motivating observation" |
| 2 | Nonlinear degradation mechanism is under-explained | Section 6.1, line ~1354 | Add 1-2 sentences clarifying why independence-based cancellation fails under nonlinear interactions |
| 3 | Timing note for Random Selection is hand-wavy | Table 9, lines 1847–1853 | Revise to explain the actual mechanism (DASH stops at K_eff < K_max) |
| 4 | Nonlinear table has no method-coverage note | Table 8 caption | Add note explaining why different methods are shown vs. Tables 3/4 |
| 5 | Equity column directionality | Tables 3, 4, 8 | Add "(lower is better)" to equity column headers |

### P2: Nice to Have (would strengthen paper but not critical for ArXiv)

| # | Issue | Location | Action |
|---|-------|----------|--------|
| 6 | Abstract leads with pipeline, not finding | Abstract | Restructure to lead with independence principle |
| 7 | Random forest baseline missing | Section 6.3 | Add RF results at rho=0.9 if time permits |
| 8 | No formal equivalence test for DASH vs. SR | Section 5.2 | Add TOST or note this caveat |
| 9 | Conclusion is repetitive | Section 7 | Add more forward-looking content |
| 10 | Bryan Arnold missing ORCID | Author block | Add if available |

### P3: Noted for Journal Submission (not relevant for ArXiv)

| # | Issue | Action |
|---|-------|--------|
| 11 | Increase to 50 repetitions (PAPER_CONFIG already set) |
| 12 | Add Random Forest baseline to all tables |
| 13 | Add LightGBM confirmation |
| 14 | Convert figures to vector format |
| 15 | Convert to TMLR style file |
| 16 | Move bibliography to .bib file |
| 17 | Add ±SE to accuracy/equity/RMSE columns |
| 18 | Bootstrap permutation test for stability differences |
| 19 | Background dataset size (B) sensitivity analysis |
| 20 | TOST equivalence test for DASH vs. SR |

---

## Verdict

**The paper is ready for ArXiv.** The v6 draft has addressed all previously identified blocking issues. The remaining P1 items are polish—they would improve the paper but their absence does not undermine the claims. The core contribution (mechanistic characterization of first-mover bias, the independence principle, and diagnostic tools) is clearly articulated, well-supported by evidence, and honestly qualified.

The paper's main vulnerability at a top venue will be the narrow method contribution (Random Selection and SR nearly match DASH), but the authors have pre-empted this by positioning the independence principle as the primary finding and DASH as a convenient operationalization with diagnostic benefits. This framing is honest and defensible.

Post to ArXiv. Then prepare the TMLR submission with 50 reps, RF baseline, and T-scaling results.
