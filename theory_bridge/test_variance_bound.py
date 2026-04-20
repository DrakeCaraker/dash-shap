#!/usr/bin/env python3
"""
Information-theoretic bound: Var[SHAP] = minimum MSE of any stable method.

The SHAP variance across the Rashomon set is the explanation quality budget.
For each feature j and observation x:

    min_{E stable} E[(E_j - SHAP_j(model, x))^2] = Var_model[SHAP_j(model, x)]

The optimal stable E is the conditional mean (= DASH, element-wise average).
Median and trimmed mean are suboptimal (MSE >= Var).

This is a mathematical identity (the mean minimizes MSE). The practical value
is that Var[SHAP_j] across the Rashomon set IS the per-feature "explanation
quality budget" — computable before choosing any method.

The variance-as-budget is the continuous analogue of the binary coverage
conflict diagnostic: coverage conflict says "is this feature's sign stable?"
(yes/no); variance says "how much does this feature's magnitude vary?"
(continuous). Together they give the full picture.
"""

import time
import warnings

import numpy as np
import shap
import xgboost as xgb
from scipy.stats import spearmanr, trim_mean
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_MODELS = 200
N_OBS = 100
SEED = 42
TEST_SIZE = 0.2
TRIM_FRACTION = 0.10  # 10% trimmed mean


# ---------------------------------------------------------------------------
# Synthetic data generation (same as validate_predictions.py)
# ---------------------------------------------------------------------------
def generate_synthetic_correlated(rho=0.9, n_samples=2000, seed=SEED):
    """California Housing augmented with synthetic rho=0.9 correlated pairs.

    Uses California Housing as the base dataset but adds 4 synthetic features
    that are rho-correlated copies of the first 4 original features, creating
    the collinearity structure needed for meaningful Rashomon instability.
    """
    data = fetch_california_housing()
    X_orig, y = data.data, data.target
    feature_names = list(data.feature_names)

    rng = np.random.RandomState(seed)

    # Add 4 correlated copies of first 4 features
    n = X_orig.shape[0]
    X_extra = np.zeros((n, 4))
    extra_names = []
    for i in range(4):
        noise_std = np.std(X_orig[:, i]) * np.sqrt(1 - rho**2)
        X_extra[:, i] = rho * X_orig[:, i] + rng.normal(0, noise_std, n)
        extra_names.append(f"{feature_names[i]}_corr")

    X = np.hstack([X_orig, X_extra])
    feature_names = feature_names + extra_names

    return X, y, feature_names


# ---------------------------------------------------------------------------
# Model training (same infrastructure as other theory_bridge scripts)
# ---------------------------------------------------------------------------
def train_rashomon_set(X_train, y_train, n_models, seed):
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


def compute_shap_matrices(models, X_explain, X_background):
    """Return list of SHAP value matrices, each shape (n_obs, n_features)."""
    shap_matrices = []
    bg = X_background[:100]
    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)
        if (i + 1) % 50 == 0:
            print(f"  SHAP computed for {i + 1}/{len(models)} models")
    return shap_matrices


# ---------------------------------------------------------------------------
# Coverage conflict computation (for correlation analysis)
# ---------------------------------------------------------------------------
def compute_coverage_conflict_rate(shap_stack):
    """Per-feature coverage conflict rate across observations.

    Args:
        shap_stack: (n_models, n_obs, n_features)

    Returns:
        conflict_rate: (n_features,) — fraction of observations with sign conflict
    """
    signs = np.sign(shap_stack)
    n_models, n_obs, n_features = signs.shape
    conflict = np.zeros((n_obs, n_features), dtype=int)

    for obs in range(n_obs):
        for feat in range(n_features):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) >= 2:
                has_pos = np.any(nonzero > 0)
                has_neg = np.any(nonzero < 0)
                conflict[obs, feat] = int(has_pos and has_neg)

    return conflict.mean(axis=0)  # per-feature rate


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_single_dataset(dataset_name, X, y, feature_names):
    """Run variance bound analysis on a single dataset. Returns results dict."""
    n_features = len(feature_names)

    print(f"\n{'=' * 74}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 74}")
    print(f"  Features ({n_features}): {feature_names}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    rng = np.random.RandomState(SEED + 1)
    obs_idx = rng.choice(len(X_test), size=min(N_OBS, len(X_test)), replace=False)
    X_explain = X_test[obs_idx]
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Explain: {len(obs_idx)}")
    print()

    # Train Rashomon set
    print(f"Training {N_MODELS} XGBoost models...")
    t0 = time.time()
    models = train_rashomon_set(X_train, y_train, N_MODELS, SEED)
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # Compute SHAP values
    print(f"Computing SHAP values ({N_MODELS} models x {len(X_explain)} obs)...")
    t0 = time.time()
    shap_matrices = compute_shap_matrices(models, X_explain, X_train)
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # Stack: (n_models, n_obs, n_features)
    shap_stack = np.stack(shap_matrices, axis=0)
    n_models, n_obs, n_feat = shap_stack.shape

    # Compute variance bound and MSE for each aggregation method
    print("Computing variance bound and MSE for each method...")
    print()

    # Per feature-observation: Var[SHAP_j] across models
    shap_var = np.var(shap_stack, axis=0)  # (n_obs, n_features)

    # DASH = mean across models
    dash_values = np.mean(shap_stack, axis=0)  # (n_obs, n_features)

    # Median across models
    median_values = np.median(shap_stack, axis=0)  # (n_obs, n_features)

    # Trimmed mean (10%) across models
    trimmed_values = np.zeros((n_obs, n_feat))
    for obs in range(n_obs):
        for feat in range(n_feat):
            trimmed_values[obs, feat] = trim_mean(shap_stack[:, obs, feat], proportiontocut=TRIM_FRACTION)

    # MSE for each method: mean across models of (SHAP_j(m) - E_j)^2
    dash_mse = np.mean((shap_stack - dash_values[np.newaxis, :, :]) ** 2, axis=0)
    median_mse = np.mean((shap_stack - median_values[np.newaxis, :, :]) ** 2, axis=0)
    trimmed_mse = np.mean((shap_stack - trimmed_values[np.newaxis, :, :]) ** 2, axis=0)

    # Ratios: MSE / Var[SHAP]
    nonzero_mask = shap_var > 1e-15

    dash_ratio = np.where(nonzero_mask, dash_mse / shap_var, np.nan)
    median_ratio = np.where(nonzero_mask, median_mse / shap_var, np.nan)
    trimmed_ratio = np.where(nonzero_mask, trimmed_mse / shap_var, np.nan)

    # ---- Report: Global ratios ----
    print("-" * 74)
    print(f"RESULT 1: MSE / Var[SHAP] RATIOS — {dataset_name}")
    print("-" * 74)
    print()

    dash_ratio_mean = np.nanmean(dash_ratio)
    dash_ratio_std = np.nanstd(dash_ratio)
    median_ratio_mean = np.nanmean(median_ratio)
    median_ratio_std = np.nanstd(median_ratio)
    trimmed_ratio_mean = np.nanmean(trimmed_ratio)
    trimmed_ratio_std = np.nanstd(trimmed_ratio)

    print(f"  {'Method':<20s}  {'Mean Ratio':>12s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}")
    print("  " + "-" * 56)
    print(
        f"  {'DASH (mean)':<20s}  {dash_ratio_mean:12.6f}  "
        f"{dash_ratio_std:10.6f}  {np.nanmin(dash_ratio):10.6f}  {np.nanmax(dash_ratio):10.6f}"
    )
    print(
        f"  {'Median':<20s}  {median_ratio_mean:12.6f}  "
        f"{median_ratio_std:10.6f}  {np.nanmin(median_ratio):10.6f}  {np.nanmax(median_ratio):10.6f}"
    )
    print(
        f"  {'Trimmed Mean (10%)':<20s}  {trimmed_ratio_mean:12.6f}  "
        f"{trimmed_ratio_std:10.6f}  {np.nanmin(trimmed_ratio):10.6f}  {np.nanmax(trimmed_ratio):10.6f}"
    )
    print()

    # Verify DASH identity (Fix 6): explicit machine-precision check
    n_valid = np.sum(nonzero_mask)
    abs_diff = np.abs(dash_mse - shap_var)
    violations_identity = np.sum(abs_diff[nonzero_mask] >= 1e-10)
    max_identity_diff = np.max(abs_diff[nonzero_mask]) if n_valid > 0 else 0.0
    print(f"  VARIANCE IDENTITY VERIFICATION (|DASH_MSE - Var| < 1e-10):")
    print(f"    {violations_identity}/{n_valid} violations at machine precision")
    print(f"    Max |DASH_MSE - Var| = {max_identity_diff:.2e}")
    if violations_identity == 0:
        print("    CONFIRMED: Var = DASH MSE exactly (0 violations)")
    else:
        print(f"    WARNING: {violations_identity} violations detected")
    print()

    max_dash_deviation = np.nanmax(np.abs(dash_ratio - 1.0))
    print(f"  DASH ratio check: max |ratio - 1.0| = {max_dash_deviation:.2e}")
    print()

    # Verify median and trimmed mean are suboptimal
    median_violations = np.nansum(median_ratio < 1.0 - 1e-10)
    trimmed_violations = np.nansum(trimmed_ratio < 1.0 - 1e-10)
    print(
        f"  Median MSE >= Var violations:       {median_violations}/{n_valid} "
        f"({'PASS' if median_violations == 0 else 'UNEXPECTED'})"
    )
    print(
        f"  Trimmed Mean MSE >= Var violations:  {trimmed_violations}/{n_valid} "
        f"({'PASS' if trimmed_violations == 0 else 'UNEXPECTED'})"
    )
    print()

    # Excess MSE
    median_excess = (median_ratio_mean - 1.0) * 100
    trimmed_excess = (trimmed_ratio_mean - 1.0) * 100
    print(f"  Median excess MSE over bound:       +{median_excess:.2f}%")
    print(f"  Trimmed mean excess MSE over bound:  +{trimmed_excess:.2f}%")
    print()

    # ---- Per-feature budget table ----
    feat_var = np.mean(shap_var, axis=0)
    feat_dash_mse = np.mean(dash_mse, axis=0)
    feat_median_mse = np.mean(median_mse, axis=0)
    feat_trimmed_mse = np.mean(trimmed_mse, axis=0)
    feat_mean_abs_shap = np.mean(np.abs(np.mean(shap_stack, axis=0)), axis=0)

    sort_idx = np.argsort(-feat_var)

    print("-" * 74)
    print(f"RESULT 2: PER-FEATURE BUDGET — {dataset_name}")
    print("-" * 74)
    print()

    print(
        f"  {'Rank':<5s} {'Feature':<15s} {'Var[SHAP]':>12s} {'DASH MSE':>12s} "
        f"{'Median MSE':>12s} {'Trim MSE':>12s} {'|DASH|':>10s} {'Var/|DASH|^2':>12s}"
    )
    print("  " + "-" * 92)

    for rank, idx in enumerate(sort_idx, 1):
        var_val = feat_var[idx]
        dash_val = feat_dash_mse[idx]
        med_val = feat_median_mse[idx]
        trim_val = feat_trimmed_mse[idx]
        abs_shap = feat_mean_abs_shap[idx]
        rel_budget = var_val / (abs_shap**2) if abs_shap > 1e-15 else np.nan
        print(
            f"  {rank:<5d} {feature_names[idx]:<15s} {var_val:12.6f} {dash_val:12.6f} "
            f"{med_val:12.6f} {trim_val:12.6f} {abs_shap:10.4f} {rel_budget:12.4f}"
        )
    print()

    # ---- Correlation with coverage conflict rate ----
    conflict_rate = compute_coverage_conflict_rate(shap_stack)

    print("-" * 74)
    print(f"RESULT 3: VARIANCE vs CONFLICT — {dataset_name}")
    print("-" * 74)
    print()

    print(f"  {'Feature':<15s} {'Var[SHAP]':>12s} {'Conflict Rate':>14s}")
    print("  " + "-" * 43)
    for idx in sort_idx:
        print(f"  {feature_names[idx]:<15s} {feat_var[idx]:12.6f} {conflict_rate[idx]:14.1%}")
    print()

    rho_abs, p_abs = spearmanr(feat_var, conflict_rate)
    print(f"  Spearman (absolute Var vs conflict): rho = {rho_abs:.4f}, p = {p_abs:.4e}")

    rel_var = np.array(
        [
            feat_var[i] / (feat_mean_abs_shap[i] ** 2) if feat_mean_abs_shap[i] > 1e-15 else np.nan
            for i in range(n_features)
        ]
    )
    valid_rel = ~np.isnan(rel_var)
    if np.sum(valid_rel) >= 3:
        rho_corr, p_val = spearmanr(rel_var[valid_rel], conflict_rate[valid_rel])
        print(f"  Spearman (relative Var/|DASH|^2 vs conflict): rho = {rho_corr:.4f}, p = {p_val:.4e}")
    else:
        rho_corr, p_val = np.nan, np.nan
        print("  Spearman (relative): too few valid features")
    print()

    return {
        "dataset_name": dataset_name,
        "dash_ratio_mean": dash_ratio_mean,
        "median_ratio_mean": median_ratio_mean,
        "trimmed_ratio_mean": trimmed_ratio_mean,
        "dash_identity_max_dev": max_dash_deviation,
        "identity_violations": int(violations_identity),
        "identity_n_checked": int(n_valid),
        "identity_max_diff": float(max_identity_diff),
        "median_violations": int(median_violations),
        "trimmed_violations": int(trimmed_violations),
        "median_excess_pct": median_excess,
        "trimmed_excess_pct": trimmed_excess,
        "var_conflict_spearman_rho_abs": rho_abs,
        "var_conflict_spearman_p_abs": p_abs,
        "var_conflict_spearman_rho_rel": rho_corr,
        "var_conflict_spearman_p_rel": p_val,
        "per_feature_var": {feature_names[i]: float(feat_var[i]) for i in range(n_features)},
        "per_feature_conflict": {feature_names[i]: float(conflict_rate[i]) for i in range(n_features)},
    }


def main():
    print("=" * 74)
    print("INFORMATION-THEORETIC BOUND: Var[SHAP] = EXPLANATION QUALITY BUDGET")
    print("=" * 74)
    print()
    print("Theorem: For any stable explanation E (constant across model choice),")
    print("  min_E E[(E_j - SHAP_j(m,x))^2] = Var_m[SHAP_j(m,x)]")
    print("The optimal E is the conditional mean = DASH (element-wise average).")
    print()

    all_results = {}

    # ------------------------------------------------------------------
    # Dataset 1: Unmodified California Housing (8 original features)
    # ------------------------------------------------------------------
    data = fetch_california_housing()
    X_orig, y_orig = data.data, data.target
    fnames_orig = list(data.feature_names)
    all_results["California Housing (original)"] = run_single_dataset(
        "California Housing (original, 8 features)", X_orig, y_orig, fnames_orig
    )

    # ------------------------------------------------------------------
    # Dataset 2: California Housing + synthetic rho=0.9 correlated copies
    # ------------------------------------------------------------------
    X_aug, y_aug, fnames_aug = generate_synthetic_correlated(rho=0.9, seed=SEED)
    all_results["California Housing (augmented)"] = run_single_dataset(
        "California Housing (augmented, 12 features, rho=0.9)", X_aug, y_aug, fnames_aug
    )

    # ------------------------------------------------------------------
    # Cross-dataset summary
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("CROSS-DATASET SUMMARY")
    print("=" * 74)
    print()

    print(f"  {'Dataset':<45s}  {'DASH ratio':>12s}  {'Med ratio':>12s}  {'Trim ratio':>12s}")
    print("  " + "-" * 85)
    for dname, res in all_results.items():
        print(
            f"  {dname:<45s}  {res['dash_ratio_mean']:12.6f}"
            f"  {res['median_ratio_mean']:12.6f}  {res['trimmed_ratio_mean']:12.6f}"
        )
    print()

    # Per-dataset variance-conflict correlation (Fix 5)
    print(f"  {'Dataset':<45s}  {'rho(abs)':>10s}  {'p(abs)':>12s}  {'rho(rel)':>10s}  {'p(rel)':>12s}")
    print("  " + "-" * 92)
    for dname, res in all_results.items():
        rho_a = res["var_conflict_spearman_rho_abs"]
        p_a = res["var_conflict_spearman_p_abs"]
        rho_r = res["var_conflict_spearman_rho_rel"]
        p_r = res["var_conflict_spearman_p_rel"]
        rho_r_str = f"{rho_r:.4f}" if not np.isnan(rho_r) else "N/A"
        p_r_str = f"{p_r:.4e}" if not np.isnan(p_r) else "N/A"
        print(f"  {dname:<45s}  {rho_a:10.4f}  {p_a:12.4e}  {rho_r_str:>10s}  {p_r_str:>12s}")
    print()

    # Per-dataset identity verification (Fix 6)
    print(f"  {'Dataset':<45s}  {'Identity violations':>20s}  {'Max |diff|':>12s}")
    print("  " + "-" * 81)
    for dname, res in all_results.items():
        print(
            f"  {dname:<45s}  {res['identity_violations']}/{res['identity_n_checked']:>14d}"
            f"  {res['identity_max_diff']:12.2e}"
        )
    print()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("=" * 74)
    print("INTERPRETATION")
    print("=" * 74)
    print()
    print("Before running any explanation method, compute the SHAP variance across")
    print("your Rashomon set. This number IS the minimum squared error any stable")
    print("method will make on that feature. It is computable, per-feature, and")
    print("cannot be improved. DASH achieves it; all alternatives are worse or equal.")
    print()
    print("The variance-as-budget is the continuous analogue of the binary coverage")
    print("conflict diagnostic:")
    print("  - Coverage conflict: 'Is this feature's sign stable?' (yes/no)")
    print("  - Variance budget:   'How much does this feature's magnitude vary?' (continuous)")
    print("Together they give the full picture of explanation stability.")
    print()
    print("=" * 74)

    return all_results


if __name__ == "__main__":
    results = main()
