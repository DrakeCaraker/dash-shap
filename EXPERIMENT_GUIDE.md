# DASH Experimental Guide

**What the experiments test, why each piece matters, and how to interpret the results.**

## The Problem DASH Solves

When you train an XGBoost model on data with correlated features, the model has to pick *one* feature from each correlated group at each split point. Which one it picks is essentially arbitrary — feature A and feature B carry the same signal, and the model grabs whichever gives a marginal gain advantage at that specific split. Change the hyperparameters slightly (tree depth, learning rate, column subsampling) and the model picks different members of the correlated group. The predictions barely change, but the SHAP values shift dramatically.

This is the Rashomon effect applied to explanations: many models fit the data equally well, but they tell completely different stories about *which features matter*. For anyone using SHAP to make decisions — feature selection, scientific hypothesis generation, regulatory audit — this is a serious problem. You're looking at an artifact of model specification, not a property of the data.

## How DASH Works

Instead of trusting one model's arbitrary feature selection, DASH deliberately trains a population of models that are *forced* to use different features. The key mechanism is restricting `colsample_bytree` to low values (0.1–0.5), so each tree only sees a small fraction of features. A model that achieves good predictive accuracy with only 20% of features visible per tree has necessarily found a *different* path through the correlated feature space than a model using a different 20%.

After filtering for performance (only keeping models that actually learned signal) and selecting for diversity (ensuring the ensemble covers different feature utilization patterns), you average their SHAP matrices element-wise. The consensus explanation fairly distributes importance across the correlated group rather than concentrating it on whichever member one model happened to grab.

---

## Experiment 1: Correlation Sweep — The Central Claim

This is the experiment that makes or breaks the paper. It tests the hypothesis that DASH produces more stable, accurate, and equitable importance rankings than baselines, and that the advantage grows with collinearity.

### Setup

The synthetic data has 50 features in 10 groups of 5, where within-group correlation is ρ. The target is a linear combination of group means, with known coefficients descending from 2.0 to 0.0. Because the DGP is linear and symmetric within groups, we know the ground-truth importance: every feature in group g should get importance |β_g|/5.

We sweep ρ ∈ {0.0, 0.5, 0.7, 0.9, 0.95} and run 5 repetitions at each level. For each repetition, we regenerate the data (same coefficients, new random draws) and run all 7 methods.

### Three Metrics

**Stability** (the headline metric): After running the full pipeline 5 times with different random seeds, how consistent are the importance rankings? Measured as the mean pairwise Spearman correlation across all pairs of runs. If a method gives ranking [A, B, C, D] one time and [C, A, D, B] the next, stability is low. DASH should produce the same ranking every time because averaging over diverse models washes out the stochastic feature selection.

**Accuracy**: Spearman correlation between estimated importance and known ground truth. This checks that stabilization doesn't come at the cost of getting the wrong answer. DASH should be more accurate because averaging fairly distributes importance across correlated features, matching the true equal-within-group structure.

**Within-group equity**: Coefficient of variation of importance within each correlated group. If features f1–f5 all contribute equally (they do, by construction), a good method assigns them similar importance. A single model might assign f1 importance 0.4 and f2–f5 importance 0.0, giving high CV. DASH should produce low CV because different models in the ensemble grab different group members, and the average spreads importance evenly.

### Expected Results

**At ρ=0** (no collinearity — the control condition): All methods should perform comparably. There's no collinearity to resolve, so DASH has no advantage — but it also shouldn't *hurt*. This is the safety check. If DASH were significantly worse at ρ=0, it would mean the method introduces unnecessary noise when there's no problem to solve.

**At ρ=0.5**: Mild collinearity. Single Best starts showing some instability. DASH should have a small advantage.

**At ρ=0.9 and ρ=0.95**: Severe collinearity. This is where DASH should dominate. Single Best stability should degrade sharply because small hyperparameter changes cause large shifts in which features get selected. DASH stability should remain high because the consensus absorbs this variation. The equity gap should be large: Single Best gives wildly uneven importance within groups, DASH distributes it fairly.

The correlation sweep plot (3 panels: stability vs ρ, accuracy vs ρ, equity vs ρ) is the paper's central figure. You want to see the lines diverge as ρ increases, with DASH methods staying high/flat while Single Best degrades.

---

## The 7 Methods and What Each Tests

**Single Best**: The standard practice. Tune one XGBoost model, compute SHAP, report importance. This is what most practitioners do today. It's the baseline to beat.

**Large Single Model** (the sequential residual dependency test): This is the sharpest baseline in the paper. It trains a single XGBoost with the *same low colsample_bytree* (0.2) that DASH uses and K×T_per_model total trees — matching DASH's total tree budget. The question it answers: does DASH's advantage come simply from using low colsample_bytree, or does it specifically require *independent* models? Within a single boosting ensemble, sequential residual dependency biases feature selection — early trees pick a feature from a correlated group, modifying residuals so subsequent trees see that feature's contribution partially removed. This creates a path-dependent "first mover" bias that concentrates importance on whichever correlated feature happened to be selected first. DASH breaks this dependency by training models from scratch independently. If DASH outperforms this baseline, it proves that breaking residual dependency matters. The gap should be largest at high ρ where the first-mover effect is strongest.

**Ensemble SHAP** (Paillard et al. baseline): Trains a single large XGBoost with *standard* high colsample_bytree (0.8) and computes SHAP. This tests the Paillard et al. (2025) argument that you should explain one big ensemble rather than aggregate explanations. Expected to behave like a more powerful Single Best — good predictions but still unstable explanations because it doesn't diversify feature selection.

**Naive Top-N**: Takes the top 15 models by validation performance from DASH's population and averages their SHAP matrices *without* diversity selection. This isolates whether diversity selection matters or whether simple averaging is enough. If Naive Top-N matches DASH, the diversity selection algorithms (MaxMin, Cluster) are unnecessary overhead. The prediction is that Naive Top-N is better than Single Best (averaging helps) but worse than DASH (because top-performing models tend to be similar — they found the same good hyperparameter region and use features the same way).

**Stochastic Retrain**: 15 models with identical hyperparameters but different random seeds. SHAP matrices averaged. This tests whether DASH's deliberate hyperparameter diversification is better than natural stochastic variation from retraining. The prediction is that stochastic retraining provides modest stability gains (some models happen to pick different features due to randomness in split tie-breaking) but much less than DASH's forced diversification.

**DASH (MaxMin)**: The recommended default. Greedy max-min dissimilarity selection ensures each added model is maximally different from all previously selected models in its feature utilization pattern. Doesn't require the correlation matrix.

**DASH (Cluster)**: Feature cluster coverage selection. Uses the actual correlation matrix to identify correlated groups, then selects models that use different representative features from each cluster. Should be strong when the correlation structure is clean (block-diagonal) but requires computing the P×P correlation matrix.

**DASH (Dedup)**: Rank correlation deduplication baseline. Removes models whose importance vectors are too similar (Spearman ρ > 0.95) but doesn't actively seek diversity. The weakest DASH variant — a sanity check that even minimal deduplication helps.

The expected ranking at high ρ: DASH (MaxMin) ≈ DASH (Cluster) > DASH (Dedup) > Naive Top-N ≈ Stochastic Retrain > Ensemble SHAP ≈ Large Single Model ≈ Single Best.

The key comparison is DASH (MaxMin) vs. Large Single Model. Both use low colsample_bytree. DASH trains independent models; LSM trains one sequential ensemble. If DASH wins, it confirms the first-mover hypothesis — sequential residual dependency within a single boosting ensemble is a distinct source of instability that only independence can resolve.

---

## Experiment 2: Overlapping Correlation Structure

The synthetic data in Experiment 1 uses perfectly clean block-diagonal correlation. Real data is messier. This experiment uses overlapping groups where features at group boundaries are correlated with *both* adjacent groups — creating chain correlations where A correlates with B and B correlates with C, but A and C are only weakly correlated.

This tests robustness. If DASH only works on the idealized structure, a reviewer will (rightly) say "but real collinearity isn't block-diagonal." The prediction is that DASH still wins, possibly with a smaller margin because the overlapping structure is inherently harder for any method to resolve cleanly. MaxMin should be particularly robust here because it doesn't assume any specific correlation structure — it just maximizes feature utilization diversity.

---

## Experiment 3: Nonlinear DGP

The nonlinear DGP has quadratic terms, interactions (z₁·z₂), and a sinusoidal component. This tests whether DASH works when the relationship between features and the target is complex and non-additive.

**Key caveat**: We can't straightforwardly measure "accuracy" against ground truth here. The approximate ground truth would be Sobol total-effect indices, but these measure total variance contribution including all interactions, while SHAP values distribute the prediction margin across features. At high collinearity, the two can diverge meaningfully. So this experiment evaluates primarily *stability* (do repeated runs agree?) and *within-group equity* (are correlated features treated symmetrically?), not accuracy.

The prediction is that DASH still improves stability and equity in the nonlinear case, though the absolute stability levels may be lower because nonlinear models have more ways to use features differently.

---

## Experiment 4: Real Data

### California Housing (8 features, regression)

Has natural collinearity — median income correlates with house value, average rooms correlates with average bedrooms, latitude/longitude are correlated with each other and with house value. No known ground truth, so we measure only stability across repetitions.

### Breast Cancer (30 features, binary classification)

Heavy natural collinearity. Radius, perimeter, and area are essentially measuring the same thing (area ∝ perimeter² ∝ radius²). Mean, standard error, and worst-case versions of each measurement are correlated. This is the ideal showcase for DASH — 30 features where maybe 8–10 are independently informative, and the rest are collinear variants.

**IS Plot interpretation**: On Breast Cancer, you should see features like "mean radius" and "mean perimeter" landing in the "Collinear Cluster Members" quadrant (high importance, high FSI), while something like "mean concavity" (a distinct geometric property) lands in "Robust Drivers" (high importance, low FSI). The IS Plot is doing unsupervised collinearity detection as a byproduct — no one told it which features are correlated, but the FSI recovers it.

**Disagreement map interpretation**: The local disagreement map picks the observation with the highest cross-model variance and shows, for that one patient, which features have reliable explanations (small error bars) versus model-specification-dependent explanations (large error bars). Clinically, this means you could tell a doctor "for this patient, texture and concavity are driving the prediction (reliable), but the specific contribution of radius vs. perimeter is uncertain (they're interchangeable)."

---

## Success Criteria

The script checks these automatically:

1. **Stability**: DASH stability > Single Best stability on ≥80% of ρ levels
2. **Accuracy**: DASH Spearman ρ ≥ 0.90 vs ground truth at ρ=0.9
3. **Equity**: DASH within-group CV < Single Best CV at all ρ levels
4. **Safety (ρ=0 control)**: DASH accuracy within 0.1 of Single Best when there's no collinearity

If all four pass, the results are publishable. Criterion 1 is the headline. Criterion 4 is the safety check. Criteria 2 and 3 demonstrate that stability comes with accuracy and fairness gains, not at their expense.

5. **Independence value** (criterion #8): DASH (MaxMin) achieves higher stability than Large Single Model on ≥70% of datasets, with the gap largest at ρ ≥ 0.9. This is the mechanistic confirmation — breaking sequential residual dependency provides value beyond internal tree diversity.

---

## What This Means for the Paper

The correlation sweep fills in Tables 1–3 (accuracy, stability, equity across ρ levels). The bar chart at ρ=0.9 becomes the summary figure. The IS Plots on real data become Figures 3–4. The disagreement map becomes Figure 6.

The overlapping structure result goes into a paragraph in Discussion confirming robustness beyond idealized settings.

The Large Single Model baseline result is the paper's mechanistic contribution. It demonstrates that collinearity-induced instability has a specific cause — sequential residual dependency within boosting ensembles — and a specific cure — training independent models that select features without path-dependent bias. This goes beyond "averaging helps" to explain *why* DASH's particular form of averaging (across independently trained, deliberately diversified models) works better than simply throwing more trees at the problem.
