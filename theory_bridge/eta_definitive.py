#!/usr/bin/env python3
"""
Definitive eta law validation — three tests per (dataset, group):

Test A: Directional — invariant (group mean) more stable than variant (within-group deviation)?
Test B: Random control — actual group mean beats random groups of same size?
Test C: Quantitative — variance fraction matches (1 + (g-1)*rho) / g?

Both interventional and tree_path_dependent (marginal) SHAP.
Synthetic g=2,3,4,5,8. Real data: California, Breast Cancer, Diabetes.
"""

import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
import warnings
import time
import sys

warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

M = 200
N_OBS = 100
SEED = 42


# ===========================================================================
# Helper functions
# ===========================================================================


def flip_rate_vector(shap_vec):
    """Compute flip rate for a (M,) vector of SHAP values across models."""
    signs = np.sign(shap_vec)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0.0
    n_pos = (nonzero > 0).sum()
    n_neg = (nonzero < 0).sum()
    return min(n_pos, n_neg) / (n_pos + n_neg)


def flip_rate_series(shap_series):
    """
    Compute mean flip rate for a (M, N_obs) series.
    Returns mean across observations.
    """
    M, N_obs = shap_series.shape
    flips = []
    for obs in range(N_obs):
        flips.append(flip_rate_vector(shap_series[:, obs]))
    return np.mean(flips), np.array(flips)


def test_a_directional(shap_stack, group_indices):
    """
    Test A: Is invariant (group mean) more stable than variant (deviations)?
    Returns: inv_flip, var_flip, ratio, p_value
    """
    group_shap = shap_stack[:, :, group_indices]  # (M, N_obs, g)
    g = len(group_indices)

    # Invariant = group mean
    invariant = group_shap.mean(axis=2)  # (M, N_obs)
    inv_flip, inv_flips = flip_rate_series(invariant)

    # Variant = deviations from mean (per feature)
    var_flips_per_obs = []
    for obs in range(shap_stack.shape[1]):
        dev_flips = []
        for j in range(g):
            dev = group_shap[:, obs, j] - invariant[:, obs]
            dev_flips.append(flip_rate_vector(dev))
        var_flips_per_obs.append(np.mean(dev_flips))
    var_flips_arr = np.array(var_flips_per_obs)
    var_flip = np.mean(var_flips_arr)

    # Wilcoxon signed-rank: paired by observation
    # H0: inv_flips == var_flips_arr
    diff = var_flips_arr - inv_flips
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) < 10:
        p_val = 1.0
    else:
        _, p_val = wilcoxon(nonzero_diff, alternative="greater")

    ratio = var_flip / inv_flip if inv_flip > 1e-10 else float("inf")
    return inv_flip, var_flip, ratio, p_val


def test_b_random_control(shap_stack, group_indices, actual_inv_flip_rate, n_random=100, seed=777):
    """
    Test B: Compare actual group-mean flip rate to random groups of same size.
    Returns: percentile rank (lower = actual is more stable than random)
    """
    M, N_obs, P = shap_stack.shape
    group_size = len(group_indices)
    rng = np.random.RandomState(seed)
    random_flips = []

    for _ in range(n_random):
        rand_idx = rng.choice(P, size=group_size, replace=False)
        rand_mean = shap_stack[:, :, rand_idx].mean(axis=2)  # (M, N_obs)
        flips = []
        for obs in range(N_obs):
            flips.append(flip_rate_vector(rand_mean[:, obs]))
        random_flips.append(np.mean(flips))

    random_flips = np.array(random_flips)
    # Percentile: fraction of random groups with flip rate <= actual
    percentile = np.mean(random_flips <= actual_inv_flip_rate) * 100
    return percentile, random_flips


def test_c_variance_fraction(shap_stack, group_indices, rho_observed):
    """
    Test C: Does invariant variance fraction match (1 + (g-1)*rho) / g?
    """
    g = len(group_indices)
    group_shap = shap_stack[:, :, group_indices]  # (M, N_obs, g)

    # Per-observation variance across models
    var_per_feature = np.var(group_shap, axis=0)  # (N_obs, g)
    total_var = var_per_feature.sum(axis=1)  # (N_obs,)

    invariant = group_shap.mean(axis=2)  # (M, N_obs) = group mean
    inv_var = np.var(invariant, axis=0) * g  # (N_obs,) scaled

    # Fraction
    mask = total_var > 1e-20
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    fraction = np.mean(inv_var[mask] / total_var[mask])
    predicted = (1 + (g - 1) * rho_observed) / g

    return fraction, predicted, abs(fraction - predicted)


def compute_observed_rho(X, group_indices):
    """Compute mean pairwise |correlation| within the group."""
    g = len(group_indices)
    if g < 2:
        return 0.0
    sub = X[:, group_indices]
    corr = np.abs(np.corrcoef(sub.T))
    # Mean off-diagonal
    mask = ~np.eye(g, dtype=bool)
    return corr[mask].mean()


# ===========================================================================
# Model training + SHAP computation
# ===========================================================================


def train_and_explain(X_train, y_train, X_explain, shap_variant, M=200, seed=42):
    """
    Train M models, compute SHAP values.
    shap_variant: 'interventional' or 'marginal'
    Returns: shap_stack (M, N_obs, P)
    """
    rng = np.random.RandomState(seed)
    shap_matrices = []

    if shap_variant == "interventional":
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

        if shap_variant == "interventional":
            explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        else:
            explainer = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")

        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)

        if (i + 1) % 50 == 0:
            print(f"      [{shap_variant}] Trained {i + 1}/{M} models")

    return np.stack(shap_matrices, axis=0)  # (M, N_obs, P)


# ===========================================================================
# Synthetic data generators
# ===========================================================================


def generate_synthetic(g, rho=0.9, n=2000, seed=42):
    """
    Generate synthetic data with g correlated features + independent features.
    Total features P = max(10, g+2).
    """
    P = max(10, g + 2)
    rng = np.random.RandomState(seed)

    # Covariance: first g features correlated at rho, rest independent
    cov = np.eye(P)
    for i in range(g):
        for j in range(g):
            if i != j:
                cov[i, j] = rho

    X = rng.multivariate_normal(np.zeros(P), cov, n)

    # Target: all features contribute, grouped ones have equal weight
    coeffs = rng.uniform(0.5, 2.0, P)
    y = X @ coeffs + rng.normal(0, 0.5, n)

    return X, y, list(range(g))


# ===========================================================================
# Real datasets
# ===========================================================================


def get_real_datasets():
    """
    Returns list of (name, X, y, groups_dict) for real datasets.
    groups_dict maps group_name -> list of feature indices.
    """
    datasets = []

    # California Housing
    cal = fetch_california_housing()
    X_cal, y_cal = cal.data, cal.target
    # Latitude=6, Longitude=7
    # AveRooms=3, AveBedrms=4
    datasets.append(
        (
            "California",
            X_cal,
            y_cal,
            {
                "Lat/Long": [6, 7],
                "Rooms/Bedrms": [3, 4],
            },
        )
    )

    # Breast Cancer
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target.astype(float)
    # radius: mean=0, se=10, worst=20
    # area: mean=3, se=13, worst=23
    datasets.append(
        (
            "BreastCancer",
            X_bc,
            y_bc,
            {
                "radius(3)": [0, 10, 20],
                "area(3)": [3, 13, 23],
            },
        )
    )

    # Diabetes
    diab = load_diabetes()
    X_d, y_d = diab.data, diab.target
    # s1=4, s2=5
    datasets.append(
        (
            "Diabetes",
            X_d,
            y_d,
            {
                "s1/s2": [4, 5],
            },
        )
    )

    return datasets


# ===========================================================================
# Main validation
# ===========================================================================


def run_validation():
    print("=" * 90)
    print("DEFINITIVE ETA LAW VALIDATION")
    print("=" * 90)
    print(f"M={M} models, N_obs={N_OBS}, seed={SEED}")
    print(f"SHAP variants: interventional, marginal (tree_path_dependent)")
    print()

    results = []
    total_time_start = time.time()

    # ---- Synthetic datasets ----
    for g in [2, 3, 4, 5, 8]:
        print(f"\n--- Synthetic g={g} (rho=0.9) ---")
        X, y, group_indices = generate_synthetic(g, rho=0.9, n=2000, seed=SEED)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
        X_explain = X_test[:N_OBS]

        rho_obs = compute_observed_rho(X_train, group_indices)
        print(f"  Observed rho={rho_obs:.3f}, group={group_indices}")

        for variant in ["interventional", "marginal"]:
            t0 = time.time()
            shap_stack = train_and_explain(X_train, y_train, X_explain, variant, M=M, seed=SEED)
            dt = time.time() - t0
            print(f"    [{variant}] done in {dt:.0f}s")

            inv_flip, var_flip, ratio, p_val = test_a_directional(shap_stack, group_indices)
            percentile, _ = test_b_random_control(shap_stack, group_indices, inv_flip)
            frac, pred, diff = test_c_variance_fraction(shap_stack, group_indices, rho_obs)

            results.append(
                {
                    "dataset": f"syn_g{g}",
                    "group": f"Z_{g}",
                    "g": g,
                    "shap": variant[:4],
                    "inv_flip": inv_flip,
                    "var_flip": var_flip,
                    "ratio": ratio,
                    "p_val": p_val,
                    "percentile": percentile,
                    "struct": percentile < 5,
                    "frac": frac,
                    "pred": pred,
                    "diff": diff,
                }
            )

    # ---- Real datasets ----
    real_datasets = get_real_datasets()
    for ds_name, X, y, groups_dict in real_datasets:
        print(f"\n--- {ds_name} ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
        X_explain = X_test[:N_OBS]

        for variant in ["interventional", "marginal"]:
            t0 = time.time()
            shap_stack = train_and_explain(X_train, y_train, X_explain, variant, M=M, seed=SEED)
            dt = time.time() - t0
            print(f"    [{variant}] done in {dt:.0f}s")

            for grp_name, group_indices in groups_dict.items():
                g = len(group_indices)
                rho_obs = compute_observed_rho(X_train, group_indices)

                inv_flip, var_flip, ratio, p_val = test_a_directional(shap_stack, group_indices)
                percentile, _ = test_b_random_control(shap_stack, group_indices, inv_flip)
                frac, pred, diff = test_c_variance_fraction(shap_stack, group_indices, rho_obs)

                results.append(
                    {
                        "dataset": ds_name,
                        "group": grp_name,
                        "g": g,
                        "shap": variant[:4],
                        "inv_flip": inv_flip,
                        "var_flip": var_flip,
                        "ratio": ratio,
                        "p_val": p_val,
                        "percentile": percentile,
                        "struct": percentile < 5,
                        "frac": frac,
                        "pred": pred,
                        "diff": diff,
                    }
                )

    total_time = time.time() - total_time_start

    # ---- Print consolidated table ----
    print("\n")
    print("=" * 120)
    print("CONSOLIDATED RESULTS")
    print("=" * 120)
    header = (
        f"{'Dataset':<14} {'Group':<12} {'g':>2} {'SHAP':<5} "
        f"{'Inv':>6} {'Var':>6} {'Ratio':>6} {'p-val':>9} "
        f"{'Pctile':>7} {'Struct':>6} "
        f"{'Frac':>6} {'Pred':>6} {'|Diff|':>7}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        row = (
            f"{r['dataset']:<14} {r['group']:<12} {r['g']:>2} {r['shap']:<5} "
            f"{r['inv_flip']:>6.3f} {r['var_flip']:>6.3f} {r['ratio']:>6.1f}x "
            f"{r['p_val']:>9.1e} "
            f"{r['percentile']:>6.1f}% {'YES' if r['struct'] else 'no':>5} "
            f"{r['frac']:>6.3f} {r['pred']:>6.3f} {r['diff']:>7.4f}"
        )
        print(row)

    # ---- Summary ----
    print("\n")
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    n_tests = len(results)
    bonferroni_alpha = 0.05 / n_tests

    # Test A
    test_a_pass = sum(1 for r in results if r["p_val"] < bonferroni_alpha)
    print(f"\nTest A (directional, Bonferroni alpha={bonferroni_alpha:.4f}):")
    print(f"  Pass: {test_a_pass}/{n_tests} ({100 * test_a_pass / n_tests:.0f}%)")

    # Test B
    test_b_pass = sum(1 for r in results if r["struct"])
    print(f"\nTest B (random control, actual < 5th percentile):")
    print(f"  Pass: {test_b_pass}/{n_tests} ({100 * test_b_pass / n_tests:.0f}%)")

    # Test C
    diffs = [r["diff"] for r in results]
    print(f"\nTest C (variance fraction):")
    print(f"  Mean |diff| from prediction: {np.mean(diffs):.4f}")
    print(f"  Max  |diff|: {np.max(diffs):.4f}")
    print(f"  Within 0.05: {sum(1 for d in diffs if d < 0.05)}/{n_tests}")

    # Marginal vs interventional
    print(f"\nMarginal vs Interventional comparison:")
    int_results = [r for r in results if r["shap"] == "inte"]
    marg_results = [r for r in results if r["shap"] == "marg"]
    if int_results and marg_results:
        int_ratios = [r["ratio"] for r in int_results]
        marg_ratios = [r["ratio"] for r in marg_results]
        print(f"  Mean ratio (int):  {np.mean(int_ratios):.2f}x")
        print(f"  Mean ratio (marg): {np.mean(marg_ratios):.2f}x")
        int_a_pass = sum(1 for r in int_results if r["p_val"] < bonferroni_alpha)
        marg_a_pass = sum(1 for r in marg_results if r["p_val"] < bonferroni_alpha)
        print(f"  Test A pass: int={int_a_pass}/{len(int_results)}, marg={marg_a_pass}/{len(marg_results)}")
        int_b_pass = sum(1 for r in int_results if r["struct"])
        marg_b_pass = sum(1 for r in marg_results if r["struct"])
        print(f"  Test B pass: int={int_b_pass}/{len(int_results)}, marg={marg_b_pass}/{len(marg_results)}")

    print(f"\nTotal wall time: {total_time / 60:.1f} minutes")
    print("=" * 90)


if __name__ == "__main__":
    run_validation()
