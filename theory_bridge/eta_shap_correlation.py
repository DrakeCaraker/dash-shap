#!/usr/bin/env python3
"""
SHAP Cross-Correlation Test for First-Mover / η Law

Tests whether correlated features have NEGATIVE cross-model SHAP
correlation (first-mover effect) and whether the variance ratio
Var[sum]/(Var[A]+Var[B]) is < 1 for correlated pairs.

Key insight: For correlated features under first-mover bias, SHAP values
are negatively correlated across models. When model A gives high SHAP to
feature 0, it gives low SHAP to feature 1 (and vice versa). Their SUM is
stable because the negative correlation cancels variance.
"""

import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings
import sys

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

M = 200
N_OBS = 100
SEED = 42


# ===========================================================================
# Core computation functions
# ===========================================================================


def compute_shap_cross_correlation(shap_stack, i, j):
    """
    Compute the cross-model correlation between SHAP_i and SHAP_j.
    shap_stack: (M, N_obs, P)
    Returns: mean ρ_SHAP across observations
    """
    M, N_obs, P = shap_stack.shape
    correlations = []
    for obs in range(N_obs):
        si = shap_stack[:, obs, i]
        sj = shap_stack[:, obs, j]
        if np.std(si) < 1e-10 or np.std(sj) < 1e-10:
            continue
        r = np.corrcoef(si, sj)[0, 1]
        if not np.isnan(r):
            correlations.append(r)
    return np.mean(correlations) if correlations else 0.0


def compute_variance_ratio(shap_stack, i, j):
    """
    Var[SHAP_i + SHAP_j] / (Var[SHAP_i] + Var[SHAP_j])
    Variance across models, averaged across observations.
    """
    M, N_obs, P = shap_stack.shape
    var_sum = np.var(shap_stack[:, :, i] + shap_stack[:, :, j], axis=0).mean()
    var_i = np.var(shap_stack[:, :, i], axis=0).mean()
    var_j = np.var(shap_stack[:, :, j], axis=0).mean()
    denom = var_i + var_j
    return var_sum / denom if denom > 1e-20 else 1.0


def compute_feature_correlation(X, i, j):
    """Pearson correlation between features i and j in the data."""
    r = np.corrcoef(X[:, i], X[:, j])[0, 1]
    return r if not np.isnan(r) else 0.0


# ===========================================================================
# Model training + SHAP computation
# ===========================================================================


def train_and_explain(X_train, y_train, X_explain, M=200, seed=42):
    """
    Train M models, compute interventional SHAP values.
    Returns: shap_stack (M, N_obs, P)
    """
    rng = np.random.RandomState(seed)
    shap_matrices = []
    bg = X_train[:100]

    for i in range(M):
        s = rng.randint(0, 2**31)
        params = {
            "n_estimators": rng.choice([100, 200, 300]),
            "max_depth": rng.choice([3, 4, 5, 6]),
            "learning_rate": rng.choice([0.05, 0.1, 0.2]),
            "colsample_bytree": rng.uniform(0.3, 0.8),
            "subsample": rng.uniform(0.6, 1.0),
            "random_state": s,
            "n_jobs": 1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, verbose=False)

        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)

        if (i + 1) % 50 == 0:
            print(f"      Trained {i + 1}/{M} models")

    return np.stack(shap_matrices, axis=0)  # (M, N_obs, P)


# ===========================================================================
# Dataset generators
# ===========================================================================


def make_synthetic(g, rho, p=10, n=1000, seed=42):
    """
    Synthetic dataset with g correlated features (ρ) and p-g independent features.
    Target = sum of all features + noise.
    """
    rng = np.random.RandomState(seed)
    n_total = n + N_OBS

    # Correlated block
    z = rng.randn(n_total, 1)
    corr_block = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.randn(n_total, g)

    # Independent block
    indep_block = rng.randn(n_total, p - g)

    X = np.hstack([corr_block, indep_block])
    y = X.sum(axis=1) + 0.1 * rng.randn(n_total)

    X_train, X_explain = X[:n], X[n : n + N_OBS]
    y_train = y[:n]

    return X_train, y_train, X_explain


# ===========================================================================
# Main analysis
# ===========================================================================


def analyze_dataset(name, shap_stack, X, pairs, pair_labels):
    """
    Analyze a dataset: compute feature correlations, SHAP correlations,
    and variance ratios for specified pairs.
    """
    P = X.shape[1]
    print(f"\n{'=' * 70}")
    print(f"  DATASET: {name}")
    print(f"{'=' * 70}")

    # Feature correlation matrix (top corner)
    n_show = min(P, 10)
    print(f"\n  Feature Correlation Matrix (first {n_show} features):")
    feat_corr = np.corrcoef(X.T)
    print("       ", "  ".join([f"f{i:2d}" for i in range(n_show)]))
    for i in range(n_show):
        row = "  ".join([f"{feat_corr[i, j]:+.2f}" for j in range(n_show)])
        print(f"  f{i:2d}  {row}")

    # SHAP correlation matrix (top corner)
    print(f"\n  SHAP Cross-Model Correlation Matrix (first {n_show} features):")
    shap_corr = np.zeros((n_show, n_show))
    for i in range(n_show):
        for j in range(n_show):
            if i == j:
                shap_corr[i, j] = 1.0
            elif j > i:
                shap_corr[i, j] = compute_shap_cross_correlation(shap_stack, i, j)
                shap_corr[j, i] = shap_corr[i, j]
            # else already filled
    print("       ", "  ".join([f"f{i:2d}" for i in range(n_show)]))
    for i in range(n_show):
        row = "  ".join([f"{shap_corr[i, j]:+.2f}" for j in range(n_show)])
        print(f"  f{i:2d}  {row}")

    # Per-pair analysis
    results = []
    print(f"\n  Per-Pair Analysis:")
    print(f"  {'Pair':<12} {'ρ_feature':>10} {'ρ_SHAP':>10} {'VarRatio':>10} {'First-mover?':>14}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 14}")

    for (i, j), label in zip(pairs, pair_labels):
        rho_feat = compute_feature_correlation(X, i, j)
        rho_shap = compute_shap_cross_correlation(shap_stack, i, j)
        var_ratio = compute_variance_ratio(shap_stack, i, j)
        first_mover = "YES" if (rho_shap < -0.05 and var_ratio < 0.9) else "NO"

        results.append(
            {
                "dataset": name,
                "pair": label,
                "rho_feature": rho_feat,
                "rho_shap": rho_shap,
                "var_ratio": var_ratio,
                "first_mover": first_mover,
            }
        )
        print(f"  {label:<12} {rho_feat:>+10.3f} {rho_shap:>+10.3f} {var_ratio:>10.3f} {first_mover:>14}")

    return results


def main():
    print("=" * 70)
    print("  FIRST-MOVER EFFECT: SHAP CROSS-CORRELATION TEST")
    print("  M={}, N_obs={}, seed={}".format(M, N_OBS, SEED))
    print("=" * 70)

    all_results = []

    # -----------------------------------------------------------------------
    # Dataset 1: Synthetic g=2
    # -----------------------------------------------------------------------
    print("\n\n[1/4] Synthetic g=2 (ρ=0.9, features 0-1 correlated, 2-9 independent)")
    X_train, y_train, X_explain = make_synthetic(g=2, rho=0.9, p=10, seed=SEED)
    shap_stack = train_and_explain(X_train, y_train, X_explain, M=M, seed=SEED)

    pairs = [(0, 1), (0, 2), (2, 3), (4, 5)]
    labels = ["f0-f1", "f0-f2", "f2-f3", "f4-f5"]
    results = analyze_dataset("syn_g2", shap_stack, X_train, pairs, labels)
    all_results.extend(results)

    # -----------------------------------------------------------------------
    # Dataset 2: Synthetic g=3
    # -----------------------------------------------------------------------
    print("\n\n[2/4] Synthetic g=3 (ρ=0.9, features 0-2 correlated, 3-9 independent)")
    X_train, y_train, X_explain = make_synthetic(g=3, rho=0.9, p=10, seed=SEED + 1)
    shap_stack = train_and_explain(X_train, y_train, X_explain, M=M, seed=SEED + 1)

    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (5, 6)]
    labels = ["f0-f1", "f0-f2", "f1-f2", "f3-f4", "f5-f6"]
    results = analyze_dataset("syn_g3", shap_stack, X_train, pairs, labels)
    all_results.extend(results)

    # -----------------------------------------------------------------------
    # Dataset 3: California Housing
    # -----------------------------------------------------------------------
    print("\n\n[3/4] California Housing")
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    feature_names = data.feature_names
    print(f"  Features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    X_explain = X_test[:N_OBS]
    shap_stack = train_and_explain(X_train, y_train, X_explain, M=M, seed=SEED)

    # Identify feature indices
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    pairs = [
        (feat_idx["Latitude"], feat_idx["Longitude"]),
        (feat_idx["AveRooms"], feat_idx["AveBedrms"]),
        (feat_idx["MedInc"], feat_idx["Population"]),
        (feat_idx["AveRooms"], feat_idx["Population"]),
    ]
    labels = ["Lat-Long", "Rooms-Bed", "MedInc-Pop", "Rooms-Pop"]
    results = analyze_dataset("california", shap_stack, X_train, pairs, labels)
    all_results.extend(results)

    # -----------------------------------------------------------------------
    # Dataset 4: Breast Cancer
    # -----------------------------------------------------------------------
    print("\n\n[4/4] Breast Cancer")
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target.astype(float)
    feature_names = data.feature_names
    print(f"  Features (first 10): {list(feature_names[:10])}")

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    X_explain = X_test[:N_OBS]
    shap_stack = train_and_explain(X_train, y_train, X_explain, M=M, seed=SEED)

    # Breast cancer: features 0-9 are "mean", 10-19 are "se", 20-29 are "worst"
    # mean radius (0) vs worst radius (20), mean texture (1) vs worst texture (21)
    pairs = [
        (0, 20),  # mean radius vs worst radius
        (1, 21),  # mean texture vs worst texture
        (2, 22),  # mean perimeter vs worst perimeter
        (0, 1),  # mean radius vs mean texture (lower correlation)
        (5, 15),  # mean compactness vs se compactness (control)
    ]
    labels = ["rad_m-rad_w", "tex_m-tex_w", "per_m-per_w", "rad_m-tex_m", "comp_m-comp_se"]
    results = analyze_dataset("breast_cancer", shap_stack, X_train, pairs, labels)
    all_results.extend(results)

    # -----------------------------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------------------------
    print("\n\n")
    print("=" * 80)
    print("  SUMMARY: FIRST-MOVER EFFECT — SHAP CROSS-CORRELATION TEST")
    print("=" * 80)
    print(f"\n  {'Dataset':<14} {'Pair':<14} {'ρ_feature':>10} {'ρ_SHAP':>10} {'VarRatio':>10} {'First-mover?':>14}")
    print(f"  {'-' * 14} {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 14}")

    for r in all_results:
        print(
            f"  {r['dataset']:<14} {r['pair']:<14} {r['rho_feature']:>+10.3f} "
            f"{r['rho_shap']:>+10.3f} {r['var_ratio']:>10.3f} {r['first_mover']:>14}"
        )

    # -----------------------------------------------------------------------
    # Spearman anti-correlation: ρ_feature vs ρ_SHAP
    # -----------------------------------------------------------------------
    rho_features = np.array([r["rho_feature"] for r in all_results])
    rho_shaps = np.array([r["rho_shap"] for r in all_results])
    var_ratios = np.array([r["var_ratio"] for r in all_results])

    spearman_rho, spearman_p = spearmanr(rho_features, rho_shaps)
    spearman_vr, spearman_vr_p = spearmanr(rho_features, var_ratios)

    print(f"\n\n  SPEARMAN CORRELATIONS (across all {len(all_results)} pairs):")
    print(f"    Spearman(ρ_feature, ρ_SHAP)    = {spearman_rho:+.4f}  (p = {spearman_p:.4e})")
    print(f"    Spearman(ρ_feature, VarRatio)   = {spearman_vr:+.4f}  (p = {spearman_vr_p:.4e})")

    print(f"\n  PREDICTION: If first-mover effect is real:")
    print(f"    - Spearman(ρ_feature, ρ_SHAP) should be NEGATIVE (anti-correlation)")
    print(f"    - Correlated pairs (ρ_feature > 0.5): ρ_SHAP < 0, VarRatio < 0.5")
    print(f"    - Independent pairs (ρ_feature ≈ 0): ρ_SHAP ≈ 0, VarRatio ≈ 1.0")

    # Count
    n_corr_pairs = sum(1 for r in all_results if r["rho_feature"] > 0.5)
    n_first_mover = sum(1 for r in all_results if r["rho_feature"] > 0.5 and r["first_mover"] == "YES")
    n_indep_pairs = sum(1 for r in all_results if abs(r["rho_feature"]) < 0.3)
    n_indep_correct = sum(1 for r in all_results if abs(r["rho_feature"]) < 0.3 and r["first_mover"] == "NO")

    print(f"\n  VERDICT:")
    print(f"    Correlated pairs showing first-mover effect: {n_first_mover}/{n_corr_pairs}")
    print(f"    Independent pairs correctly showing no effect: {n_indep_correct}/{n_indep_pairs}")
    print(f"    Overall Spearman anti-correlation: {spearman_rho:+.4f} (p={spearman_p:.2e})")

    if spearman_rho < -0.3 and spearman_p < 0.05:
        print(f"\n    *** FIRST-MOVER EFFECT DETECTED ***")
        print(f"    Feature correlation anti-correlates with SHAP cross-model correlation.")
    else:
        print(f"\n    First-mover effect: {'WEAK' if spearman_rho < 0 else 'NOT DETECTED'}")
        print(f"    Spearman = {spearman_rho:+.4f}, p = {spearman_p:.2e}")


if __name__ == "__main__":
    main()
