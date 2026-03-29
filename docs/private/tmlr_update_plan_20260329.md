# TMLR Paper Update Strategy Based on SageMaker Results

**Date:** 2026-03-29
**Authors:** Caraker, Arnold, Rhoads
**Paper:** `paper/draft_v7_preprint.tex`
**Status:** Pre-submission analysis — no edits applied yet

---

## TL;DR — Three Decisions

| Question | Answer | Key Reason |
|----------|--------|------------|
| Clean 50-rep re-run? | **YES, but targeted** | 6 experiments remain; the 7 completed in run 20260326 have good provenance and can be kept |
| 100 reps? | **NO** | Point estimates moved <0.002 from 20→50; doubled cost for ~29% CI narrowing |
| Edit paper now? | **NO** | Wait for remaining 6 experiments; document changes needed |

---

## Current State of Experimental Data

### Two SageMaker runs exist

- **Run 20260323** (older): 8 result files, backfilled provenance, no hardware metadata
- **Run 20260326** (newer, authoritative): 7 experiments with proper provenance — real config_sha, hardware=ml.g5.16xlarge, elapsed_s captured, `_stability_tests`/`_equity_tests`/`_fsi_validation` computed, figures generated (PDF+PNG)

### Run 20260326 — completed experiments (keep these)

| Experiment | Status | Key DASH Result |
|---|---|---|
| `synthetic_linear_sweep` | ✅ 50 reps | 0.977 @ ρ=0.9 (confirms v6) |
| `nonlinear_sweep` | ✅ 50 reps | 0.887 @ ρ=0.9 (DASH beats SR by +0.030) |
| `table2_baselines` | ✅ 50 reps | Ensemble SHAP=0.956, Perm Imp=0.952, DASH-Dedup=0.975 |
| `breast_cancer` | ✅ 50 reps | DASH=0.925, SR=0.862, SB=0.376 |
| `california_housing` | ✅ 50 reps | DASH=0.978, SR=0.977, SB=0.969 |
| `superconductor` | ✅ 50 reps | DASH=0.964, RS=0.968, SB=0.840 |
| `epsilon_sensitivity` | ✅ 50 reps | Per-rep arrays stored; k_eff data present |

### Still missing (6 experiments — need run)

`ablation`, `variance_decomposition`, `variance_decomposition_crossed`, `first_mover_bias`, `background_sensitivity`, `asymmetric_dgp`

Also: `k_sweep_independence` (broken, 2/50 reps), `overlapping` (only in older run with backfilled provenance)

Estimated additional compute: **~3–4 hours on ml.g5.16xlarge**

---

## Key Numbers: v6 (20 reps) vs v7 (50 reps)

### Linear Sweep @ ρ=0.9

| Method | v6 (20 reps) | 50 reps (20260326) | Delta |
|--------|-------------|-------------------|-------|
| DASH (MaxMin) | 0.977 | 0.977 | 0.000 |
| Stochastic Retrain | 0.977 | 0.978 | +0.001 |
| Single Best | 0.958 | 0.958 | 0.000 |
| LSM | 0.938 | 0.938 | 0.000 |
| Random Selection | 0.976 | 0.977 | +0.001 |

**All claims survive identically.**

### 50-Rep Statistical Tests (ρ=0.9)

**Stability (vs DASH):**

| Comparison | diff | p | Verdict |
|---|---|---|---|
| vs SB | +0.019 | <0.001 | **Significant** |
| vs LSM | +0.039 | <0.001 | **Significant** |
| vs SR | -0.001 | 0.40 | Not significant |
| vs RS | +0.000 | 0.62 | Not significant |

**Equity (vs DASH):**

| Comparison | p | Cohen's d | Verdict |
|---|---|---|---|
| vs SB | 9.1e-11 | -1.47 | **Significant, large** |
| vs LSM | 1.8e-15 | -2.94 | **Significant, huge** |
| **vs RS** | **1.65e-6** | **-0.37** | **Significant, medium** |
| **vs Naive Top-N** | **4.88e-7** | **-0.49** | **Significant, medium** |
| vs SR | 0.44 | -0.13 | Not significant |

### What diversity selection actually does (DASH MaxMin vs Random Selection)

| Metric | Helps? | Evidence |
|---|---|---|
| Stability | No | Tied at every ρ level (±0.002) |
| Top-K5 (high ρ) | Yes | +0.036 at ρ=0.9, +0.042 at ρ=0.95 |
| Equity | **Yes (significant)** | p=1.65e-6, d=-0.37 at ρ=0.9 |
| RMSE | Yes | ~-0.010 at ρ≥0.5 |
| Accuracy | No | Tied everywhere |

### Nonlinear DGP (ρ=0.9)

| Method | Stability | Equity |
|---|---|---|
| Random Forest | **0.892** | 0.232 |
| DASH (MaxMin) | 0.887 | **0.161** |
| Stochastic Retrain | 0.857 | 0.184 |
| Single Best | 0.811 | 0.217 |

DASH beats SR by +0.030 on stability and -0.023 on equity. RF slightly beats DASH (different mechanism: internal feature bagging).

### Real-world datasets

| Dataset | DASH | SR | RS | SB | Notes |
|---------|------|-----|-----|-----|-------|
| Breast Cancer | **0.925** | 0.862 | 0.919 | 0.376 | DASH dominant (+0.063 over SR) |
| Superconductor | 0.964 | 0.924 | **0.968** | 0.840 | RS slightly beats DASH |
| Calif. Housing | 0.978 | 0.977 | **0.989** | 0.969 | RS beats DASH |

---

## Reframing Recommendation

*Developed through three rounds of self-critique. Tested against four reviewer archetypes (methodologist, practitioner, statistician, theorist). Checked against 50-rep statistical tests.*

### The core tension

Line 1097 of the current draft: *"This equivalence is our central theoretical contribution."*

This framing is honest but self-undermining for a methods paper. A TMLR reviewer will ask: *"If SR works just as well, why does DASH exist?"* The "threefold advantages" (speed, diagnostics, equity) then read as post-hoc justifications.

But the paper's honesty is also its greatest strength. The solution is not to hide results — it's to place them in the right hierarchy.

### Self-critique of aggressive reframing

After three rounds of review, I scaled back the initial recommendation:

1. **Promoting equity as a *core* advantage is risky.** DASH vs SR equity is p=0.44 (not significant). Promoting equity prominently, then qualifying "not significant vs SR," sends a contradictory signal. The precisely-scoped claim (MaxMin vs RS: p<0.001) works; a broader claim doesn't.

2. **The nonlinear advantage has real caveats.** RF beats DASH. Without RS in the experiment, we can't isolate diversity selection from population design. Overselling invites pointed questions.

3. **The paper's current honesty is well-calibrated for TMLR.** Reviewers are exhausted by overclaiming. Don't sacrifice credibility.

### Three targeted changes (high confidence)

**Change 1: Reword the "headline" claim (lines 1097-1112)**

FROM: *"This equivalence is our central theoretical contribution: it isolates model independence as the operative mechanism..."*

TO: *"This equivalence is the strongest evidence for our central claim: model independence, not any particular aggregation strategy, is the operative mechanism that neutralizes first-mover bias. DASH operationalizes this principle with additional benefits — equitable attribution (Section 5.2), nonlinear robustness (Section 5.4), and ground-truth-free diagnostics (Section 5.5) — that simple seed averaging does not provide."*

The independence principle IS the contribution. The equivalence is evidence.

**Change 2: Add the significant equity test result (lines 1017-1024)**

Currently says equity comparison "has not been formally significance-tested."

UPDATE: *"MaxMin selection significantly improves equity over Random Selection (p < 0.001, Cohen's d = −0.37 at ρ = 0.9), confirming that deliberate diversity in feature utilization produces more equitable credit distribution within the filtered-population framework. The DASH-vs-SR comparison on equity is not significant (p = 0.44), indicating that the equity benefit is specific to diversity-aware selection, not to the overall pipeline design relative to seed averaging."*

This gives the diversity mechanism a concrete, defensible, significant purpose.

**Change 3: Bridge nonlinear result into the main argument (after line ~1132)**

Add a paragraph at the end of Section 5.2:

*"The linear regime demonstrates that any form of model independence suffices for stability. Section 5.4 shows that under nonlinear data-generating processes, the form of independence matters: DASH's population-level diversity — forced feature restriction and varied hyperparameters — outperforms seed averaging by +0.030 stability at ρ = 0.9 (Table 8), likely because diverse feature subsets explore distinct nonlinear interaction pathways. Random Forest achieves comparable stability (0.892) through internal feature bagging, a related but structurally distinct mechanism."*

Creates a two-part story without restructuring.

### Three additional changes (medium confidence — assess after full data)

**Change 4: Restructure the "threefold advantages" list (lines 1117-1132)**

Current: Speed, Diagnostics, Equity → Proposed: Nonlinear robustness, Diagnostics, Equity (with significance), Speed

**Change 5: Acknowledge SR's top-k5 advantage**

*"SR achieves higher top-5 ranking stability than DASH (0.922 vs 0.863 at ρ = 0.9), likely because fixed hyperparameters preserve the feature importance landscape across seeds while DASH's diversity mechanism produces more varied individual-model rankings."*

**Change 6: Note the RF result in Discussion**

*"Random Forest achieves comparable overall stability through internal feature bagging — a related mechanism — though with substantially lower top-5 ranking agreement (0.375) and higher RMSE (0.957), suggesting stable but systematically different attributions."*

### What NOT to change

- **Don't restructure the paper.** Section order works.
- **Don't hide SR equivalence.** The honesty is a strength.
- **Don't promote equity beyond the scoped claim.** DASH vs SR is not significant.
- **Don't overstate the nonlinear advantage.** RF beats DASH; synthetic-only.
- **Don't remove "For stability alone, seed averaging suffices"** (line 1480). This builds reviewer trust.

### How this addresses each reviewer archetype

| Reviewer | Concern | How reframe addresses it |
|---|---|---|
| Methodologist | "Method doesn't beat baselines" | Principle is the contribution; method adds equity (significant), nonlinear robustness, diagnostics |
| Practitioner | "When do I use DASH vs SR?" | SR for stability only; DASH when equity, nonlinear data, or auditing matters |
| Statistician | "Are differences real?" | New equity test (p<0.001); honest non-significance for DASH vs SR |
| Theorist | "Is the principle proven?" | Unchanged — tier split, LSM evidence, dose-response |

---

## Mechanical Updates Required (after full data)

### Text changes in `draft_v7_preprint.tex`

| Location | Current | Update To | Lines |
|----------|---------|-----------|-------|
| All "20 repetitions" | "20 repetitions" | "50 repetitions" | ~15 instances |
| Table 2 (extended baselines) | v6 numbers | 50-rep numbers | ~963-990 |
| Real-world results table | v6 numbers | 50-rep numbers | ~1270-1291 |
| Significance table | v6 p-values/d | 50-rep stats | ~1080-1094 |
| Nonlinear DGP table | v6 numbers | 50-rep numbers | ~1361-1378 |
| Asymmetric DGP table | "---" placeholders | actual values | ~2091-2094 |
| Correlation sweep figure caption | "20 repetitions" | "50 repetitions" | ~1167 |
| Limitations re: underpowered | "N_reps=20 underpowered" | remove/update | ~1538-1541 |
| Source attribution | demo_benchmark_6.ipynb | run_experiments_parallel.py | ~1267 |
| Remove placeholder note | "Note: Table will be populated" | delete | ~2100-2104 |

---

## Execution Sequence

1. Complete remaining 6 experiments on SageMaker (20260326 branch)
2. Debug and re-run k_sweep_independence
3. Apply Changes 1-3 (high confidence) to `draft_v7_preprint.tex`
4. Update all "20 repetitions" → "50 repetitions" (~15 instances)
5. Update all tables with 50-rep numbers
6. Fill asymmetric DGP table (Appendix D)
7. Assess Changes 4-6 (medium confidence) with full data
8. Regenerate figures, update `docs/BENCHMARK_RESULTS.md`
9. Run `/sync-check`, compile LaTeX, verify

---

## Decision: 100 Reps

**NO.** Rationale:

- 20→50 moved point estimates <0.002
- Large effects already conclusive (DASH vs SB: d>1.4, p≈0)
- SR≈DASH equivalence already confirmed at 50 reps
- Doubles runtime and SageMaker cost for ~29% CI narrowing
- If a reviewer specifically requests it, do a targeted 100-rep run on just the contested experiment
