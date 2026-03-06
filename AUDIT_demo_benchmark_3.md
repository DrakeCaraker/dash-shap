# Audit: `notebooks/demo_benchmark_3.ipynb`

**Date**: 2026-03-06
**Scope**: Structure, rigor, reproducibility, and statistical methodology

---

## Overall Assessment

The notebook is well-organized (54 cells, 16 sections) with a clear narrative arc: proof of concept → baselines → stability → correlation sweep → real data → sensitivity/ablation → statistical tests → publication figures. Markdown cells provide good explanations of *why* each experiment exists, not just *what* it computes. The experiment design is ambitious and mostly sound.

That said, there are several rigor and methodology issues ranging from minor housekeeping to substantive concerns that could undermine reviewer confidence.

---

## Strengths

1. **Strong narrative structure.** Each section is motivated ("what it tests"), not just executed. The baseline comparison table in Section 2 explicitly states what hypothesis each method tests.
2. **Safety control at ρ=0.** The sweep includes uncorrelated data to verify DASH doesn't degrade when there's no problem to solve.
3. **Multiple DGPs.** Linear (Section 4) and nonlinear (Section 9) DGPs plus three real datasets (Breast Cancer, Superconductor, California Housing).
4. **Epsilon sensitivity (Section 7)** shares a single population per rep across ε values, isolating the filter threshold effect from training stochasticity. Good experimental design.
5. **Bonferroni correction** applied to Wilcoxon tests (Section 10). Methodological note correctly explains why stability cannot be tested with Wilcoxon.
6. **Ablation studies** (Section 13) use one-at-a-time variation with a clearly documented default, matching standard ML practice.

---

## Issues

### A. Reproducibility

| # | Severity | Issue |
|---|----------|-------|
| A1 | Medium | **No environment recording.** No cell captures `numpy.__version__`, `xgboost.__version__`, platform, Python version, or total wall-clock time. With `n_jobs=-1`, XGBoost parallelism is nondeterministic — results may not reproduce exactly across machines even with `SEED=42`. Add a versioning cell at the top and a timing summary at the bottom. |
| A2 | Low | **`warnings.filterwarnings('ignore')`** suppresses all warnings globally (Cell 1). This hides potential convergence warnings or deprecation notices. Scope it narrowly (e.g., `warnings.filterwarnings('ignore', category=FutureWarning)`) or at minimum log warnings to a file. |

### B. Hyperparameter Inconsistencies

| # | Severity | Issue |
|---|----------|-------|
| B1 | High | **Global config says M=500, K=30, N_REPS=20 but the comment says "paper uses M=200, K=20, N_REPS=10."** Cell 1 sets `M=500`, `K=30`, `N_REPS=20`, which are used in Sections 1–4. Later sections define local overrides (`EXT_N_REPS=10`, `NL_M=200`, `SC_M=200`, `ABL_N_REPS=5`). It is unclear which configuration is canonical for the paper. The mismatch between global and local settings makes the notebook fragile and confusing. Recommend: define a single `PAPER_CONFIG` dict and derive all section-level configs from it with explicit overrides documented. |
| B2 | Medium | **`SingleBestBaseline(n_trials=50)` in Sections 1–2 vs `n_trials=30` everywhere else.** The proof-of-concept and single-run baselines use a better-tuned Single Best (50 trials) than the repeated experiments (30 trials). This makes the single-run Section 2 results non-comparable with the repeated experiments. Pick one value and use it consistently. |
| B3 | Medium | **`StochasticRetrainBaseline(N=15)` in Section 11 vs global K=30.** Stochastic Retrain averages 15 models while DASH uses up to K=30. This disadvantages the baseline. Should use `N=K` for fair comparison. |
| B4 | Low | **Breast Cancer epsilon differs.** Cell 49 uses `epsilon=0.02` for Breast Cancer vs `EPSILON=0.03` globally. The reason isn't documented. |

### C. Statistical Methodology

| # | Severity | Issue |
|---|----------|-------|
| C1 | High | **Wilcoxon with N=10 has very low power.** The minimum achievable two-sided p-value with N=10 is ≈0.002. After Bonferroni correction (×20), the minimum corrected p-value is ≈0.04. This means you can barely reach significance even with a perfect separation. The notebook doesn't discuss statistical power. Recommend: (a) note the power limitation, (b) consider increasing EXT_N_REPS to at least 20 (you already have N_REPS=20 in Section 4), or (c) use a permutation test instead of Wilcoxon. |
| C2 | Medium | **Bootstrap CI on stability is methodologically questionable.** Cell 51 resamples importance vectors with replacement and computes stability on each bootstrap sample. When the same vector appears twice in a resample, its self-correlation is 1.0, inflating the pairwise mean. The bootstrap distribution is biased upward. Consider instead: (a) bootstrap *pairs* of vectors and compute correlations, or (b) use a delete-d jackknife. |
| C3 | Medium | **Cohen's d direction ambiguity for equity.** Cell 50 computes `cohens_d(dash_equity, baseline_equity)`. Since equity is CV (lower=better), a *negative* d means DASH is better. But the `interpret_d` function uses `abs(d)` and doesn't report direction. A reader seeing "large effect" doesn't know which direction. Report signed d or add a "favors" column. |
| C4 | Low | **`importance_stability` returns 1.0 for n<2.** If a filter leaves only 1 model in an epsilon sweep rep, stability defaults to 1.0, which would inflate the mean. Cell 26 handles this by checking `len(filtered) < 2` and skipping, but the `imp_runs` list could still have varying lengths across ε values, leading to unequal-n stability comparisons. |

### D. Data Handling

| # | Severity | Issue |
|---|----------|-------|
| D1 | High | **Test set never used; all evaluation on val set.** `generate_synthetic_linear` returns `X_test, y_test` but they are never used. SHAP is computed on `X_ref=X_val`, and predictive RMSE is computed on `X_val`. Since val scores are used for performance filtering (Stage 2), the filtered models are optimistically biased on val. SHAP explanations and RMSE should be evaluated on a held-out test set. |
| D2 | Medium | **Superconductor scaler leaks across re-splits.** Cell 40 fits `StandardScaler` on the original train split. Cell 41 re-splits `train+val` per rep with a different random state but applies the original scaler. The scaler's mean/std don't match the new train split. Either re-fit the scaler per rep or don't re-split. |
| D3 | Low | **Nonlinear DGP `beta_4_to_G` uses a hardcoded seed.** Groups 4+ get random coefficients from `RandomState(42)` regardless of the main `seed` parameter. The "ground truth importance" is thus identical across reps. This is fine since accuracy isn't reported for nonlinear, but it's a subtle gotcha worth a comment in the code. |

### E. Redundancy and Missing Coverage

| # | Severity | Issue |
|---|----------|-------|
| E1 | Medium | **Sections 4 and 8 are redundant.** The correlation sweep runs twice: Section 4 (N_REPS=20, no RMSE) and Section 8 (EXT_N_REPS=10, with RMSE). Section 8 is strictly more informative. Consolidate into one sweep with the higher rep count and RMSE extraction. This also clarifies which results are canonical for the paper. |
| E2 | Medium | **Publication figure (Cell 47) uses 10-rep data, not 20-rep.** Cell 47 renders from `ext_sweep` (Section 8, 10 reps), not the 20-rep data from Section 4. The main paper figure should use the highest-rep data available. |
| E3 | Medium | **Breast Cancer has no repetition analysis.** Superconductor and California Housing get 10-rep stability measurements. Breast Cancer gets a single run with a diagnostic plot. Add a repetition loop for consistency. |
| E4 | Medium | **DASH (Cluster) dropped after Section 3.** Introduced in Sections 2–3 but absent from the sweep, significance tests, and ablation. Either include it in the sweep or explain why it's excluded. |
| E5 | Low | **Naive Top-N and Stochastic Retrain missing from correlation sweep.** Only 3 methods in the sweep (Sections 4/8). Table 2 (Cell 38) only covers ρ=0.9. A reviewer may want sweep curves for all methods. |
| E6 | Low | **Section 6 success criteria are checked before Section 8 re-runs the sweep.** The Section 6 checks use `sweep_results` from Section 4, but Section 16 re-checks using `ext_sweep`. Section 6 is thus stale by the end of the notebook. Consider removing Section 6 or merging it with Section 16. |

### F. Code Quality

| # | Severity | Issue |
|---|----------|-------|
| F1 | Low | **Abbreviated imports reduce readability.** Cell 40 uses `train_test_split as tts`, `StandardScaler as SS`. These save characters but reduce clarity for a demo notebook meant for reviewers. |
| F2 | Low | **`plt.show()` should come after all print statements in each cell.** In several cells (e.g., Cell 5), `plt.show()` precedes `print()`. In non-interactive contexts (e.g., batch execution), the figure may render before the accompanying text. |
| F3 | Low | **Magic numbers.** `T_per_model=500` in `LargeSingleModelBaseline`, `n_estimators=2000` in `EnsembleSHAPBaseline`, `tau=0.3` in cluster selection, `n_trials=30/50` — these appear without justification. Add brief comments or centralize in the config. |

---

## Recommended Actions (Priority Order)

1. **Fix D1**: Evaluate on test set, not val set. This is the most substantive methodological issue.
2. **Fix B1**: Unify hyperparameter config. Define canonical values and document overrides.
3. **Fix E1/E2**: Merge Sections 4 and 8 into a single sweep with RMSE, using the higher rep count.
4. **Fix C1**: Either increase reps for significance tests or document the power limitation.
5. **Fix B2/B3**: Standardize `n_trials` and `N` across all baselines.
6. **Fix D2**: Re-fit scaler per rep in Superconductor benchmark.
7. **Fix E3**: Add repetition analysis for Breast Cancer.
8. **Fix C2**: Use a correct bootstrap scheme for stability CIs.
9. **Fix C3**: Report signed Cohen's d with direction indicator.
10. **Add A1**: Version/environment recording cell.
