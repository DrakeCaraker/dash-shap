#!/usr/bin/env python3
"""
Pairwise interaction instability test.
Prediction: SHAP interaction flip rates > max(individual flip rates).

Tests whether SHAP interaction values are systematically more unstable
than individual SHAP values across a Rashomon set of XGBoost models.
Uses the fiber product prediction from the impossibility framework.

200 models, 50 observations, California Housing dataset.
Sensitivity analysis at 3 zero-exclusion thresholds (25%, 50%, 75%).
"""
import time
import warnings

import numpy as np
import shap
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_MODELS = 200
N_OBS = 50  # fewer than main experiments — interactions are expensive
SEED = 42
TEST_SIZE = 0.2
ZERO_THRESHOLDS = [0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# Model training (same as validate_predictions.py)
# ---------------------------------------------------------------------------
def train_models(X_train, y_train, n_models, seed):
    """Train n_models XGBoost regressors with varied hyperparameters."""
    rng = np.random.RandomState(seed)
    models = []
    for i in range(n_models):
        s = rng.randint(0, 2**31)
        params = {
            "n_estimators": rng.choice([100, 200, 300]),
            "max_depth": rng.choice([3, 4, 5, 6]),
            "learning_rate": rng.choice([0.05, 0.1, 0.2]),
            "colsample_bytree": rng.uniform(0.1, 0.5),
            "subsample": rng.uniform(0.6, 1.0),
            "random_state": s,
            "n_jobs": 1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, verbose=False)
        models.append(m)
        if (i + 1) % 50 == 0:
            print(f"  Trained {i + 1}/{n_models} models")
    return models


# ---------------------------------------------------------------------------
# SHAP interaction value computation
# ---------------------------------------------------------------------------
def compute_shap_interactions(models, X_explain, X_background):
    """Compute SHAP interaction values for all models.

    Returns
    -------
    interaction_arrays : list of ndarray, each shape (n_obs, n_features, n_features)
    shap_arrays : list of ndarray, each shape (n_obs, n_features)
        Main SHAP values (diagonal of interaction matrix).
    """
    interaction_arrays = []
    shap_arrays = []
    bg = X_background[:50]  # small background for speed

    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="tree_path_dependent")
        iv = explainer.shap_interaction_values(X_explain)
        # iv shape: (n_obs, n_features, n_features)
        interaction_arrays.append(iv)
        # Main effects are on the diagonal
        main_effects = np.diagonal(iv, axis1=1, axis2=2)  # (n_obs, n_features)
        shap_arrays.append(main_effects)

        if (i + 1) % 25 == 0:
            print(f"  SHAP interactions computed for {i + 1}/{len(models)} models")

    return interaction_arrays, shap_arrays


# ---------------------------------------------------------------------------
# Minority fraction (flip rate) computation
# ---------------------------------------------------------------------------
def minority_fraction(signs):
    """Compute minority fraction from an array of signs, excluding zeros.

    Parameters
    ----------
    signs : 1d array of {-1, 0, 1}

    Returns
    -------
    flip_rate : float in [0, 0.5], or NaN if too few nonzero
    n_zero : int, number of zeros
    """
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return np.nan, int(np.sum(signs == 0))
    n_pos = np.sum(nonzero > 0)
    n_neg = np.sum(nonzero < 0)
    total = n_pos + n_neg
    return min(n_pos, n_neg) / total, int(np.sum(signs == 0))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_interaction_experiment():
    print("=" * 70)
    print("PAIRWISE INTERACTION INSTABILITY EXPERIMENT")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  N_MODELS           = {N_MODELS}")
    print(f"  N_OBS              = {N_OBS}")
    print(f"  SEED               = {SEED}")
    print(f"  ZERO_THRESHOLDS    = {ZERO_THRESHOLDS}")
    print(f"  Background samples = 50")
    print()

    # ---- Load data ----
    print("Loading California Housing...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    n_features = X.shape[1]
    print(f"  Features ({n_features}): {feature_names}")

    # ---- Split data ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Select observations for SHAP
    rng = np.random.RandomState(SEED + 1)
    obs_idx = rng.choice(len(X_test), size=N_OBS, replace=False)
    X_explain = X_test[obs_idx]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Explain: {N_OBS}")

    # ---- Train models ----
    print(f"\nTraining {N_MODELS} models...")
    t0 = time.time()
    models = train_models(X_train, y_train, N_MODELS, SEED)
    print(f"  Trained in {time.time() - t0:.1f}s")

    # ---- Compute SHAP interaction values ----
    print(f"\nComputing SHAP interaction values ({N_MODELS} models x {N_OBS} obs)...")
    print("  This may take 10-20 minutes...")
    t0 = time.time()
    interaction_arrays, shap_arrays = compute_shap_interactions(
        models, X_explain, X_train
    )
    elapsed_shap = time.time() - t0
    print(f"  SHAP interactions complete in {elapsed_shap:.1f}s ({elapsed_shap / 60:.1f} min)")

    # Stack arrays for vectorized access
    # interactions: (N_MODELS, N_OBS, n_features, n_features)
    interactions_stack = np.stack(interaction_arrays, axis=0)
    # shap_values: (N_MODELS, N_OBS, n_features)
    shap_stack = np.stack(shap_arrays, axis=0)

    # ---- Enumerate feature pairs (upper triangle, A < B) ----
    pairs = []
    for a in range(n_features):
        for b in range(a + 1, n_features):
            pairs.append((a, b))
    n_pairs = len(pairs)
    print(f"\n  Feature pairs (upper triangle): {n_pairs}")

    # ---- Compute per-pair, per-observation metrics ----
    print("\nComputing flip rates for interactions and individuals...")

    # Store results: (n_pairs, N_OBS) for each metric
    interaction_flip_rates = np.full((n_pairs, N_OBS), np.nan)
    individual_flip_A = np.full((n_pairs, N_OBS), np.nan)
    individual_flip_B = np.full((n_pairs, N_OBS), np.nan)
    zero_fractions = np.full((n_pairs, N_OBS), np.nan)

    for pi, (a, b) in enumerate(pairs):
        for obs in range(N_OBS):
            # Interaction signs across models
            interaction_vals = interactions_stack[:, obs, a, b]
            interaction_signs = np.sign(interaction_vals)

            # Zero fraction for this pair-observation
            n_zero = np.sum(interaction_signs == 0)
            zero_fractions[pi, obs] = n_zero / N_MODELS

            # Interaction flip rate
            iflip, _ = minority_fraction(interaction_signs)
            interaction_flip_rates[pi, obs] = iflip

            # Individual flip rates
            shap_a_signs = np.sign(shap_stack[:, obs, a])
            flip_a, _ = minority_fraction(shap_a_signs)
            individual_flip_A[pi, obs] = flip_a

            shap_b_signs = np.sign(shap_stack[:, obs, b])
            flip_b, _ = minority_fraction(shap_b_signs)
            individual_flip_B[pi, obs] = flip_b

    max_individual = np.fmax(individual_flip_A, individual_flip_B)
    delta = interaction_flip_rates - max_individual

    # ---- Sensitivity analysis at different zero-exclusion thresholds ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for threshold in ZERO_THRESHOLDS:
        print(f"\n{'─' * 70}")
        print(f"ZERO-EXCLUSION THRESHOLD: {threshold:.0%}")
        print(f"  (Exclude pair-observations where >{threshold:.0%} of models have zero interaction)")
        print(f"{'─' * 70}")

        # Mask: keep pair-observations where zero fraction <= threshold
        mask = zero_fractions <= threshold
        # Also exclude NaN flip rates
        valid = mask & ~np.isnan(delta) & ~np.isnan(interaction_flip_rates) & ~np.isnan(max_individual)

        n_valid = np.sum(valid)
        n_total = n_pairs * N_OBS
        print(f"  Valid pair-observations: {n_valid}/{n_total} ({n_valid / n_total:.1%})")

        if n_valid < 10:
            print("  Too few valid observations for analysis. Skipping.")
            continue

        # Extract valid values
        delta_valid = delta[valid]
        iflip_valid = interaction_flip_rates[valid]
        max_ind_valid = max_individual[valid]

        # ---- Mean delta with 95% CI ----
        mean_delta = np.mean(delta_valid)
        se_delta = np.std(delta_valid, ddof=1) / np.sqrt(n_valid)
        ci_lo = mean_delta - 1.96 * se_delta
        ci_hi = mean_delta + 1.96 * se_delta
        print(f"\n  Mean delta (interaction_flip - max_individual):")
        print(f"    {mean_delta:.6f}  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")

        # ---- Mean interaction flip rate vs mean max individual ----
        print(f"\n  Mean interaction flip rate:  {np.mean(iflip_valid):.6f}")
        print(f"  Mean max individual flip:   {np.mean(max_ind_valid):.6f}")

        # ---- Wilcoxon signed-rank test ----
        # H0: delta = 0 (interaction flip rate = max individual flip rate)
        try:
            stat, pval = wilcoxon(delta_valid, alternative="greater")
            print(f"\n  Paired Wilcoxon signed-rank test (H1: delta > 0):")
            print(f"    statistic = {stat:.1f}")
            print(f"    p-value   = {pval:.2e}")
            if pval < 0.001:
                print(f"    Result: HIGHLY SIGNIFICANT (p < 0.001)")
            elif pval < 0.01:
                print(f"    Result: SIGNIFICANT (p < 0.01)")
            elif pval < 0.05:
                print(f"    Result: SIGNIFICANT (p < 0.05)")
            else:
                print(f"    Result: NOT SIGNIFICANT (p >= 0.05)")
        except ValueError as e:
            print(f"\n  Wilcoxon test failed: {e}")

        # ---- Fraction where interaction > max individual ----
        frac_higher = np.mean(delta_valid > 0)
        frac_equal = np.mean(delta_valid == 0)
        frac_lower = np.mean(delta_valid < 0)
        print(f"\n  Fraction where interaction_flip > max_individual: {frac_higher:.4f}")
        print(f"  Fraction where interaction_flip = max_individual: {frac_equal:.4f}")
        print(f"  Fraction where interaction_flip < max_individual: {frac_lower:.4f}")

        # ---- Per-pair summary ----
        print(f"\n  Per-pair summary:")
        print(f"  {'Pair':>20s}  {'mean_int_flip':>14s}  {'mean_max_ind':>14s}  {'mean_delta':>12s}  {'n_valid':>8s}")
        print(f"  {'─' * 20}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 8}")

        for pi, (a, b) in enumerate(pairs):
            pair_mask = valid[pi, :]
            n_pair_valid = np.sum(pair_mask)
            if n_pair_valid == 0:
                print(
                    f"  {feature_names[a]:>8s} x {feature_names[b]:<8s}"
                    f"  {'N/A':>14s}  {'N/A':>14s}  {'N/A':>12s}  {0:>8d}"
                )
                continue

            pair_iflip = interaction_flip_rates[pi, pair_mask]
            pair_max_ind = max_individual[pi, pair_mask]
            pair_delta = delta[pi, pair_mask]

            print(
                f"  {feature_names[a]:>8s} x {feature_names[b]:<8s}"
                f"  {np.mean(pair_iflip):>14.6f}"
                f"  {np.mean(pair_max_ind):>14.6f}"
                f"  {np.mean(pair_delta):>12.6f}"
                f"  {n_pair_valid:>8d}"
            )

    # ---- Overall summary ----
    print(f"\n\n{'=' * 70}")
    print("SENSITIVITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Threshold':>10s}  {'n_valid':>8s}  {'mean_delta':>12s}  {'95% CI':>24s}  {'p-value':>12s}  {'frac>':>8s}")
    print(f"  {'─' * 10}  {'─' * 8}  {'─' * 12}  {'─' * 24}  {'─' * 12}  {'─' * 8}")

    for threshold in ZERO_THRESHOLDS:
        mask = (zero_fractions <= threshold) & ~np.isnan(delta) & ~np.isnan(interaction_flip_rates) & ~np.isnan(max_individual)
        n_valid = np.sum(mask)
        if n_valid < 10:
            print(f"  {threshold:>10.0%}  {n_valid:>8d}  {'---':>12s}  {'---':>24s}  {'---':>12s}  {'---':>8s}")
            continue
        delta_v = delta[mask]
        mean_d = np.mean(delta_v)
        se_d = np.std(delta_v, ddof=1) / np.sqrt(n_valid)
        ci = f"[{mean_d - 1.96 * se_d:.6f}, {mean_d + 1.96 * se_d:.6f}]"
        try:
            _, pv = wilcoxon(delta_v, alternative="greater")
            pv_str = f"{pv:.2e}"
        except ValueError:
            pv_str = "N/A"
        frac = np.mean(delta_v > 0)
        print(f"  {threshold:>10.0%}  {n_valid:>8d}  {mean_d:>12.6f}  {ci:>24s}  {pv_str:>12s}  {frac:>8.4f}")

    # ---- Distribution of zero fractions ----
    print(f"\n\nZero fraction distribution (across all pair-observations):")
    zf_flat = zero_fractions.ravel()
    zf_flat = zf_flat[~np.isnan(zf_flat)]
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    pct_vals = np.percentile(zf_flat, percentiles)
    for p, v in zip(percentiles, pct_vals):
        print(f"  {p:>3d}th percentile: {v:.4f}")

    print(f"\n  Mean zero fraction: {np.mean(zf_flat):.4f}")
    print(f"  Fraction with zero_frac = 0: {np.mean(zf_flat == 0):.4f}")
    print(f"  Fraction with zero_frac > 0.5: {np.mean(zf_flat > 0.5):.4f}")

    return {
        "interaction_flip_rates": interaction_flip_rates,
        "max_individual": max_individual,
        "delta": delta,
        "zero_fractions": zero_fractions,
        "pairs": pairs,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    t_total = time.time()
    results = run_interaction_experiment()
    elapsed = time.time() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)
