# Peer Review: draft_v6_preprint.tex — TMLR Preparation Roadmap

**Paper:** "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution"
**Authors:** Caraker, Arnold, Rhoads (2026)
**Reviewer standard:** Top ML venue (TMLR) with ArXiv pragmatism noted
**Date:** 2026-03-16
**Materials reviewed:** draft_v6_preprint.tex (full), REVIEW_v5_arxiv.md, REVIEW_v6.md, EXPERIMENT_GUIDE.md, ROADMAP.md, BENCHMARK_RESULTS.md, notebooks/demo_benchmark_7.ipynb

---

## What Was Fixed in the Current v6 Update

The following items from the review were applied directly to `draft_v6_preprint.tex`:

1. **Algorithm pseudocode ε → ε₀** — Line 1741: consistency with Eq. 5's `$\epsilon_0 = 10^{-8}$`
2. **David Rhoads' ORCID** (0009-0005-3015-5948) added to author block
3. **Ablation figure caption** — "plateaus past M=100" → "invariant to population size" (matches body text wording)
4. **Timing discrepancy note** — Added explanation for why Random Selection (287.2s) is slower than DASH (140.3s)
5. **CC-BY 4.0 license statement** — Added after Acknowledgments

---

## Overall Assessment

**ArXiv verdict: Ready to post.** No blocking issues remain.

**TMLR verdict: Accept with revisions.** The mechanistic insight is genuine, the experiments are thorough, and the writing is unusually honest. The issues below are what TMLR reviewers will likely raise.

**Rating (conference scale): 6.5/10** — Above average; the contribution is more empirical-diagnostic than methodological, which is fine for TMLR's scope.

---

## A. MAJOR ISSUES (TMLR-targeted)

### M1. The Core Method Contribution Is Narrow

Random Selection achieves stability 0.976 vs. DASH's 0.977 at ρ=0.9. MaxMin diversity selection adds almost nothing to the headline metric. The paper honestly acknowledges this (Section 5.2), but a TMLR reviewer will ask: "Is DASH a method contribution or a finding about model independence?"

**Recommendation:** Consider whether the TMLR framing should foreground the *finding* (first-mover bias + independence principle) rather than the *tool* (DASH). The current title already leans this way, which is good. The abstract could be restructured to lead with the independence principle rather than the pipeline.

**Action:** Rewrite abstract for TMLR to lead with "model independence is sufficient" rather than "we propose DASH."

### M2. The Empirical Hypothesis Is Only Tested Indirectly

Section 3.3 formally states that the first-mover concentration gap increases with tree count T. The paper tests this "indirectly" via the LSM comparison and references a direct T-scaling experiment in the code repository (`run_experiments.py --experiments first_mover_bias`).

**Recommendation:** Include T-scaling results in the TMLR version, even as a small appendix figure. The experiment already exists in the codebase.

**Action:** Add T-scaling figure to Appendix (data from `first_mover_bias` experiment in `run_experiments.py`).

**Status in demo_benchmark_7.ipynb:** Cell 44/45 (Section 12: "First-Mover Bias Isolation") contains the T-scaling experiment code. The cell exists but has **not been run** (no outputs). When notebook 7 is executed, this will produce the direct mechanistic evidence. **Action for TMLR:** Run notebook 7, then add the T-scaling figure to the paper appendix.

### M3. No Direct Statistical Test of the Headline Stability Metric

Stability is a single aggregate across all repetitions, so paired Wilcoxon tests are impossible. BCa bootstrap CIs are provided and are non-overlapping, which is informative but not a direct hypothesis test.

**Recommendation:** Add a bootstrap permutation test of the stability difference between DASH and Single Best. Compute stability for both methods on each of (e.g.) 10,000 bootstrap resamples of the repetition pairs, then report the p-value for the null hypothesis that the difference is zero.

**Action:** Implement bootstrap stability comparison test in evaluation code; add results to significance section.

**Status in demo_benchmark_7.ipynb:** Cell 47 (Section 13) implements Wilcoxon signed-rank tests with Holm-Bonferroni correction on accuracy and equity. BCa bootstrap CIs are computed in the sweep (cell 4 output shows these). A dedicated bootstrap *hypothesis test* on the stability difference itself is **not yet implemented**. **Action for TMLR:** Add a permutation/bootstrap test for stability differences to the significance cell.

### M4. Breast Cancer Baseline Asymmetry

SB(M=200) performing *worse* than SB(30) is an interesting finding in itself (model selection instability worsens with more candidates under extreme collinearity). The comparison is now transparent with both baselines shown, but a reviewer may argue the SB(M=200) comparison is somewhat unfair since DASH *benefits* from more models while SB is *harmed*.

**Recommendation:** Add a sentence explicitly framing why SB(M=200) degrades: it's evidence of model-selection instability under extreme collinearity, not just a comparison point.

**Action:** Add 1-2 sentences to Breast Cancer paragraph explaining the SB(M=200) degradation mechanism.

---

## B. ANALYTICAL GAPS (Experiments to Add for TMLR)

### B1. Random Forest Baseline [CRITICAL]

The paper's central claim is that model independence resolves first-mover bias. Random forests train trees independently by construction. If RF already provides stable SHAP explanations, the entire DASH pipeline becomes unnecessary for practitioners willing to use RF.

**Experiment needed:** Train RF on all synthetic and real-world datasets. Compute SHAP stability. Compare with XGBoost Single Best and DASH.

**Expected outcome:** RF stability should be higher than XGBoost Single Best but potentially lower than DASH (because RF has less hyperparameter diversity). If RF matches DASH, the paper's practical contribution narrows to "use RF for stable explanations" — which would need to be acknowledged.

**Status in demo_benchmark_7.ipynb:** **ALREADY IMPLEMENTED.** `RandomForestBaseline` exists in `dash/baselines/` and is included in:
- Cell 11/12: Independence confirmation table at ρ=0.9 (listed in `independent_methods`)
- Cell 36: Nonlinear DGP sweep (listed in `nl_methods`)
- `run_experiments.py`: Included in linear sweep, nonlinear sweep, and all three real-world dataset experiments (California, Breast Cancer, Superconductor)

Notebook 7 has been partially run (cells 1, 3, 4 have outputs) but most cells with RF results have **not yet been executed**. **Action for TMLR:** Run full notebook 7 to get RF results, then add RF to all paper tables. This is the single most important experiment for the TMLR submission — RF results will either strongly confirm the independence principle (if RF stability is high) or reveal that DASH adds value beyond independence (if RF stability is moderate).

### B2. Increase to 50 Repetitions

PAPER_CONFIG specifies N_REPS=50, but current results use 20. With 20 reps, the Wilcoxon test is underpowered for small effect sizes (DASH vs. SR).

**Action:** Rerun all experiments with N_REPS=50 for the TMLR version.

**Status in demo_benchmark_7.ipynb:** **ALREADY CONFIGURED.** Cell 1 sets `PAPER_CONFIG['N_REPS'] = 50`. The notebook is configured for 50 reps. Cell 14 markdown confirms: "Full sweep ρ ∈ {0.0, 0.5, 0.7, 0.9, 0.95} with 8 methods, N_REPS=50." The notebook just needs to be fully executed. **Action for TMLR:** Run full notebook 7 (will be computationally expensive — ~2.5× longer than v6 due to 50 vs 20 reps).

### B3. LightGBM Confirmation

The paper positions first-mover bias as a property of gradient boosting generally but only tests XGBoost. A single confirmation on LightGBM (leaf-wise splitting) would strengthen the generalization claim.

**Action:** Add LightGBM baseline to at least the ρ=0.9 comparison.

**Status in demo_benchmark_7.ipynb:** **NOT IMPLEMENTED.** No LightGBM or CatBoost references found in notebook 7 or `run_experiments.py`. Would require implementing a new baseline class in `dash/baselines/`. **Priority:** P1 for TMLR — a single LightGBM comparison at ρ=0.9 would strengthen the generalization claim substantially with modest effort.

### B4. Conditional/Causal SHAP Comparison

Related work cites Heskes et al. (2020) and Aas et al. (2021) but doesn't empirically compare. These methods address correlated features at the Shapley value level rather than the model level.

**Action:** At minimum, add a discussion paragraph explaining why conditional SHAP is not compared (computational cost, causal graph requirement). Ideally, add one empirical comparison at ρ=0.9.

### B5. Background Dataset Size Sensitivity

All experiments use B=100 background samples. Under high correlation, this may be insufficient. The paper defers sensitivity analysis to the journal version (Limitations).

**Action:** Run B ∈ {50, 100, 200, 500} sensitivity at ρ=0.9.

---

## C. PRESENTATION AND FORMATTING (TMLR)

### C1. Convert to TMLR Style File

Current paper uses standard `article` class. TMLR has its own LaTeX template.

**Action:** Convert to TMLR template, adjust formatting.

### C2. Move Bibliography to .bib File

Current inline `\begin{thebibliography}` should become a proper `.bib` file for maintainability and automatic sorting.

**Action:** Extract all entries to `references.bib`.

### C3. Convert Figures to Vector Format

All figures are PNG. For journal submission, use PDF/PGF for line plots and scatter plots (concentration plot, sweep plot, IS plot, disagreement map).

**Action:** Update figure generation code to output PDF; update `\graphicspath` and `\includegraphics` calls.

### C4. Expand Bibliography

Missing relevant work:
- Ribeiro et al. (2016) — LIME; worth contrasting stability properties with SHAP
- Hooker et al. (2021) — extends the already-cited 2019 paper on permutation
- Recent Rashomon-set variable importance work (post-Fisher 2019)
- Watson et al. (2023) — testing conditional independence

**Action:** Add 4-6 additional citations; expand Related Work accordingly.

### C5. Shorten Section 5

The Results section is ~600 lines. For TMLR, consider moving ablation (epsilon sensitivity, M ablation, timing) to appendix. The main body should focus on: mechanism experiment, independence principle, real-world validation.

### C6. Strengthen the Conclusion

Currently repeats the introduction's contribution list. Should add more forward-looking content — the "broader implications" paragraph in Discussion (Section 6.4) is more interesting than the repetitive conclusion summary.

### C7. Abstract Restructuring

Consider leading with the independence principle finding rather than the pipeline description. The phrase "Both our proposed method, DASH, and simple seed-averaging (Stochastic Retrain) restore stability" is the most important sentence — bring it forward.

---

## D. SPECIFIC TEXT ISSUES

| Location | Issue | Severity |
|----------|-------|----------|
| Line ~191 | Long parenthetical "($M \times {\sim}75 \approx 15{,}000$ trees...)" breaks reading flow | Low |
| Line ~682 | LSM described as "~15K trees" in methods table but tree count matching is approximate | Low |
| Conclusion | Lines 1477-1509 closely mirror both intro and Section 5.2 — reduce repetition | Medium |
| Abstract | 219 words, dense — consider restructuring for TMLR | Medium |

---

## E. STATISTICAL AND METHODOLOGICAL NOTES

### E1. Multiple Testing Across Experiments

The 26 tests within the linear sweep are corrected via Holm-Bonferroni. Tests across different experiments (nonlinear, real-world) are not jointly corrected. This is standard practice (different experiment families), but TMLR reviewers may note it.

### E2. Missing CIs on Accuracy and Equity

Tables report stability with ±SE but accuracy, equity, and RMSE without error bars in the main sweep table. Data exists in the notebooks.

**Action:** Add ±SE to all metrics in main tables.

### E3. Variance Decomposition Limitations

The footnote caveat added in v6 is good. For TMLR, consider a proper crossed design (R data seeds × R model seeds) for exact ANOVA decomposition instead of the current marginal design.

### E4. Group-level Accuracy Saturation

EXPERIMENT_GUIDE.md notes (C8): with 10 groups and true betas spanning 20× range, group-level accuracy saturates at 1.0. The gmse metric was added as a discriminative complement. Consider reporting gmse alongside accuracy in the TMLR version.

---

## F. QUESTIONS TMLR REVIEWERS WILL ASK

1. **"Why not use random forests?"** — RF trees are independent by construction. This is the most obvious missing baseline. (See B1)

2. **"Is DASH anything more than bagging explanations?"** — Random Selection nearly matches DASH. The answer is: equity and diagnostics. Make this sharper.

3. **"How does this compare to conditional SHAP / causal SHAP?"** — These address correlated features at the Shapley value level. (See B4)

4. **"20 reps is low for a stability paper."** — Acknowledged in Limitations. Fix with 50 reps. (See B2)

5. **"The M ablation shows no benefit past M=50. Why is M=200?"** — Body text now says "invariant," which is the right framing. But reviewers will ask about computational waste.

6. **"What happens with more than 50 features?"** — All synthetic experiments use P=50. Scalability to P=500+ is untested.

---

## G. CROSS-REFERENCE WITH demo_benchmark_7.ipynb

Notebook 7 is the **authoritative TMLR notebook** (per CLAUDE.md). It has 52 cells, of which only 3 have been executed (cells 1, 3, 4 — setup, proof of concept, and the main sweep). Key findings:

### Already configured in notebook 7 (just needs execution):
| Item | Notebook Location | Status |
|------|-------------------|--------|
| **N_REPS=50** | Cell 1 (PAPER_CONFIG) | Configured, not yet run at 50 reps |
| **Random Forest baseline** | Cells 11, 12, 36; `run_experiments.py` | Code exists in baselines, included in sweep/nonlinear/real-world |
| **Permutation Importance baseline** | Cell 12 (Table 2 Extended) | Code exists in baselines |
| **DASH (Dedup) results** | Cell 12 (Table 2 Extended) | Code exists, not yet run |
| **T-scaling / first-mover bias isolation** | Cell 44-45 (Section 12) | Experiment code exists, not yet run |
| **Overlapping correlation structure** | Cell 38-39 (Section 10) | Experiment code exists, not yet run |
| **Variance decomposition** | Cell 40-41 (Section 11) | Experiment code exists, not yet run |
| **BCa bootstrap CIs** | Built into sweep infrastructure | Computed when sweep runs |

### NOT in notebook 7 (needs new implementation):
| Item | Priority | Effort |
|------|----------|--------|
| **LightGBM baseline** | P1 | Medium — new baseline class needed |
| **CatBoost baseline** | P2 | Medium — new baseline class needed |
| **Conditional/Causal SHAP** | P2 | High — different SHAP computation |
| **Bootstrap stability hypothesis test** | P1 | Low — add to significance cell |
| **Background size (B) sensitivity** | P2 | Low — parameter sweep |
| **Vector figure output (PDF)** | P1 | Low — matplotlib savefig format change |
| **TMLR style file conversion** | P0 | Medium — LaTeX reformatting |
| **Bibliography .bib extraction** | P1 | Low — manual extraction |

### Critical path for TMLR submission:
1. **Run notebook 7 fully** — this alone produces: 50-rep results, RF baseline, Permutation Importance, T-scaling evidence, expanded baselines
2. **Add RF to all paper tables** — the single most impactful new result
3. **Update all numbers** from 20-rep to 50-rep values
4. **Add T-scaling figure** to appendix (direct test of empirical hypothesis)
5. **Convert to TMLR format** (style file, .bib, vector figures)

---

## H. PRIORITY-ORDERED ACTION ITEMS FOR TMLR

### P0: Must-fix for TMLR submission
1. **Run notebook 7 fully** — produces 50-rep results, RF baseline, Permutation Importance, T-scaling evidence *[all code exists, just needs execution]*
2. **Add Random Forest to all paper tables** — results will come from notebook 7 run *[code exists in `dash/baselines/` and `run_experiments.py`]*
3. **Update all numbers** from 20-rep (v6) to 50-rep (v7) values *[after notebook 7 run]*
4. **Include T-scaling figure** in appendix (direct test of empirical hypothesis) *[notebook 7 cell 44-45]*
5. **Convert to TMLR style file** (C1) *[manual LaTeX work]*

### P1: Should-fix for TMLR
6. Add bootstrap stability hypothesis test *[new code needed in significance cell]*
7. Add LightGBM confirmation at ρ=0.9 *[new baseline class needed]*
8. Move bibliography to .bib (C2) *[manual extraction]*
9. Add ±SE to all metrics in tables (E2) *[data available from notebook 7]*
10. Convert figures to vector/PDF (C3) *[matplotlib savefig format change]*
11. Expand bibliography with 4-6 additional citations (C4)
12. Shorten Section 5 — move ablation/timing to appendix (C5)
13. Strengthen conclusion — reduce repetition, add forward-looking content (C6)
14. Add explicit framing for SB(M=200) degradation mechanism (M4)
15. Restructure abstract to lead with independence principle (C7)

### P2: Nice-to-have for TMLR
16. Background size (B) sensitivity (B5) *[parameter sweep needed]*
17. Conditional SHAP discussion or comparison (B4)
18. Crossed variance decomposition design for exact ANOVA (E3)
19. Report gmse alongside accuracy (E4) *[metric exists in evaluation code]*
20. Scalability experiments with P > 50 features
21. CatBoost baseline *[new baseline class needed]*
