# Implementation Plan: Mechanism-First Paper Repositioning + Code Changes

## Strategic Framing

**Current title**: "DASH: Diversified Aggregation of SHAP — Stable, Accurate, and Equitable Feature Importance Under Multicollinearity"

**Proposed title**: "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Diagnosis, and Resolution via Diversified Aggregation"

**Core reframe**: The primary contribution is the *mechanistic finding* (sequential residual dependency / first-mover bias). DASH is the engineered solution. Stochastic Retrain is corroborating evidence. The diagnostics (FSI, IS Plot) are the practical toolset. This positions Paper 1 as the foundation for the 5-paper program (partial orders → impossibility result → neural net generalization → paradigm paper).

---

## Part A: Paper Restructure (draft_v1.tex)

### A1. New contribution hierarchy (rewrite Introduction lines 166-179)

```
1. MECHANISM: Sequential residual dependency as the specific cause of SHAP instability
   in gradient boosting under collinearity. The "first-mover bias."
2. EVIDENCE: Both DASH (deliberate diversity) and Stochastic Retrain (seed diversity)
   fix it via model independence → confirms the mechanism, not just the method
3. DIAGNOSTICS: FSI and IS Plot as ground-truth-free auditing tools
4. METHOD: DASH as the engineered solution with practical advantages over SR
   (diagnostics, equity, forced feature diversity for harder cases)
```

### A2. Restructure Section 5 (Results) — new ordering

**Current ordering**: Correlation sweep → Extended baselines → Statistical significance → LSM → Nonlinear → Real-world → Epsilon → Population ablation

**Proposed ordering** (leads with mechanism, not leaderboard):

1. **The Mechanism Experiment** (NEW — promote LSM subsection + variance decomposition)
   - LSM vs DASH: same colsample_bytree, sequential vs independent → worst vs best
   - This is the "smoking gun" for first-mover bias
   - Variance decomposition data already exists in `experiment_variance_decomposition()`
   - Show model-selection variance dominates for Single Best; DASH reduces it

2. **Independence Resolves It** (reframed from current "Extended baselines")
   - Both DASH and Stochastic Retrain achieve ~0.98 stability
   - Gap is 0.001, not significant → this is the POINT: independence is the key
   - Random Selection (no MaxMin) still beats Single Best → confirms independence matters more than selection strategy
   - Naive Top-N shows averaging alone isn't enough without filtering

3. **The Advantage Scales with Correlation** (current correlation sweep)
   - Same data, just reframed: "the mechanism's severity grows with rho"
   - DASH stability is flat; baselines degrade → mechanism is rho-dependent

4. **Diagnostics: FSI and IS Plot** (promoted from afterthought)
   - Breast Cancer case study: FSI correctly identifies radius/perimeter/area without supervision
   - IS Plot quadrants match known correlation structure
   - These work without ground truth → practical value

5. **Real-World Validation** (current real-world section, enhanced)
   - Include California Housing (already implemented, missing from paper)
   - Add wall-clock timings table (data already captured in `elapsed_s`)

6. **Robustness** (current epsilon sensitivity + population ablation)
   - Epsilon insensitivity
   - Population size diminishing returns
   - Nonlinear DGP with scope caveat (rho >= 0.7)

### A3. Rewrite Abstract (lines 51-91)

Lead with: "We identify sequential residual dependency — a 'first-mover bias' in gradient boosting — as a specific mechanistic cause of SHAP instability under multicollinearity."

Key change: Stochastic Retrain result is framed as confirmation, not limitation:
"Both DASH and simple seed-averaging (Stochastic Retrain) resolve the instability, confirming that model independence is the key mechanism. DASH additionally provides..."

### A4. Rewrite Discussion Section 6.1 (lines 921-941)

Currently titled "Why independence matters" — this is already close to the right framing but needs to be the paper's core message, not a discussion point. Move the key insight ("the central finding is not merely that averaging helps") into the introduction.

### A5. Fix incomplete Table 1

The `---*` placeholders must be filled. Two approaches:
- Option A: Run `python run_experiments.py --experiments linear_sweep` (generates all methods across all rho levels) — this already includes Stochastic Retrain, LSM (Tuned), Random Selection at every rho level.
- Option B: If compute time is prohibitive, restructure Table 1 to show the mechanism experiment (LSM vs DASH vs SR at rho=0.9) as Table 1, and move the full sweep to Table 2.

**Recommendation**: Option A if compute available (the experiment runner already does this). If not, Option B defers the full sweep to supplementary while keeping the main tables complete.

### A6. Add wall-clock timings table (NEW)

The experiment runner already captures `elapsed_s` per method. Add a table:

```
| Method              | Time (s) per rep | Total (20 reps) |
|---------------------|------------------|-----------------|
| Single Best         |                  |                 |
| DASH (MaxMin)       |                  |                 |
| Stochastic Retrain  |                  |                 |
| Large Single Model  |                  |                 |
```

### A7. Fill remaining placeholders

- Figure 1 pipeline diagram: create a TikZ or SVG diagram of the 5-stage pipeline
- Author names and affiliations
- Paillard et al. full citation
- Fix Bonferroni → Holm-Bonferroni consistency

### A8. Tighten scope claims

- Change "stable under multicollinearity" → "stable under moderate-to-high multicollinearity"
- Add explicit scope note for nonlinear: "DASH's advantage emerges at rho >= 0.7 for nonlinear DGPs"

---

## Part B: Code Changes

### B1. Add Naive Averaging + Random Selection to paper outputs

These baselines are already in `run_experiments.py` but missing from the paper. The linear sweep already runs `Random Selection` at every rho level (line 304). Need to ensure Naive Top-N is also included in the sweep methods list.

**File**: `run_experiments.py`
**Change**: Add `'Naive Top-N'` to `sweep_methods` list (line 301-306) if not already present. Then add the corresponding if/elif block.

### B2. Add wall-clock timing output

The timing is already captured (`elapsed_s` in results JSON). Need a summary function that aggregates timings across methods.

**File**: `run_experiments.py`
**Change**: Add a `print_timing_summary()` helper after each experiment that formats the per-method timings from the results dict.

### B3. New experiment: First-Mover Bias Isolation

Create a focused experiment that demonstrates first-mover bias directly, suitable for the paper's new lead experiment. This is distinct from the existing variance decomposition (which decomposes sources) — this one shows the bias *accumulating* with tree count.

**File**: `run_experiments.py`
**New function**: `experiment_first_mover_bias()`
```
- Train a single XGBoost on synthetic data (rho=0.9) with increasing n_estimators
  [50, 100, 200, 500, 1000, 2000, 5000]
- At each checkpoint, compute SHAP importance
- Measure concentration: max(importance_within_group) / sum(importance_within_group)
- Plot: concentration vs n_estimators → show first-mover advantage growing with depth
- Compare against M independent models averaged at same total tree count
```

This produces a crisp figure showing the mechanism at work.

### B4. Fix diversity.py:38 (subsample bias)

```python
# Current (biased):
X_sub = X_ref[:min(n_subsample, len(X_ref))]

# Fixed (random):
rng = np.random.RandomState(seed)
idx = rng.choice(len(X_ref), min(n_subsample, len(X_ref)), replace=False)
X_sub = X_ref[idx]
```

### B5. Add X_ref defaulting warning in pipeline.py

```python
if X_ref is None:
    import warnings
    warnings.warn(
        "X_ref not provided; defaulting to X_val. Consider using a "
        "held-out reference set (X_explain) to avoid potential confounds.",
        UserWarning
    )
    X_ref = X_val
```

### B6. Add integration tests

**File**: `tests/test_pipeline.py` (extend)

```python
def test_dash_pipeline_end_to_end():
    """Full pipeline produces expected shapes and non-degenerate output."""
    # Small synthetic data, M=10, K=3, N_REPS=1
    # Check: consensus_shap_ shape, global_importance_ length, K_eff >= 2

def test_dash_reproducibility():
    """Same seed produces identical output."""
    # Run pipeline twice with seed=42, assert np.allclose

def test_maxmin_selection_basic():
    """MaxMin selects diverse vectors over similar ones."""
    # Construct 5 importance vectors: 3 identical, 2 orthogonal
    # MaxMin should prefer the orthogonal ones
```

### B7. Add Naive Top-N to the linear sweep

**File**: `run_experiments.py`, `experiment_linear_sweep()`
**Change**: Add `'Naive Top-N'` to `sweep_methods` and add elif block using `NaiveAveragingBaseline`.

---

## Part C: Execution Order

### Phase 1: Code fixes (no experiment re-runs needed)
1. B4: Fix diversity.py subsample bias
2. B5: Add X_ref warning in pipeline.py
3. B6: Add integration tests
4. Run `pytest` to verify nothing breaks

### Phase 2: Experiment infrastructure
5. B1 + B7: Add Naive Top-N to linear sweep
6. B2: Add timing summary helper
7. B3: Implement first-mover bias isolation experiment

### Phase 3: Paper restructure
8. A3: Rewrite abstract (mechanism-first)
9. A1: Rewrite contribution list
10. A2: Restructure results section ordering
11. A4: Strengthen discussion
12. A5: Add notes for Table 1 completion (actual values require experiment run)
13. A6: Add timing table skeleton
14. A7: Fill placeholders (Bonferroni fix, scope tightening)
15. A8: Tighten scope claims

### Phase 4: Generate results (requires compute)
16. Run `python run_experiments.py --experiments linear_sweep` (populates full Table 1)
17. Run `python run_experiments.py --experiments real_california` (adds missing dataset)
18. Run new `experiment_first_mover_bias()` (generates mechanism figure)
19. Update paper tables with actual values from `results/tables/*.json`

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Paper restructure | Low | No data changes, just reframing |
| diversity.py fix | Low | Affects subsample only, not main SHAP computation |
| New experiment (B3) | Medium | May not show clean monotonic concentration |
| Full experiment re-run | Low | Infrastructure already exists, just needs compute |
| Integration tests | Low | New tests, no existing code changes |

---

## What This Achieves

After these changes, the paper:
1. **Leads with a novel mechanistic finding** that motivates the entire 5-paper program
2. **Uses Stochastic Retrain as evidence** rather than competing with it
3. **Has complete tables** with no placeholders
4. **Includes wall-clock timings** addressing the compute cost question
5. **Adds ablation baselines** (Naive Top-N, Random Selection) that answer "which components matter?"
6. **Has a clean first-mover bias figure** that visualizes the mechanism directly
7. **Tightens scope** to avoid the nonlinear safety violation claim

The framing shift from "we propose DASH" to "we identify first-mover bias and show independence resolves it" makes the paper harder to reject: the mechanism is the contribution, the method is the solution, and Stochastic Retrain is corroboration.
