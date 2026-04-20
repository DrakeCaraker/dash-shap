#!/usr/bin/env python3
"""
Cross-domain validation of DASH-SHAP impossibility predictions.

ONE script, ZERO per-dataset knobs. Same M=200 models, same hyperparameters,
same analysis for every dataset. The only thing that varies is the data loader.

Datasets: 4 real (sklearn), 3 synthetic controls.
Protocol: identical for all datasets.
Correction: BH-FDR across all p-values.
"""

import time
import json
import warnings
from datetime import datetime

import numpy as np
import xgboost as xgb
import shap
import sklearn.datasets
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from diptest import diptest
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def generate_synthetic(rho: float, n: int = 2000, p: int = 8, seed: int = 12345):
    """Generate synthetic regression data with controlled feature correlation."""
    rng = np.random.RandomState(seed)
    # Build correlation matrix: all off-diag = rho
    cov = np.full((p, p), rho)
    np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    # True model: only first 4 features matter
    beta = np.zeros(p)
    beta[:4] = rng.uniform(0.5, 2.0, size=4) * rng.choice([-1, 1], size=4)
    y = X @ beta + rng.normal(0, 0.5, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "california_housing": lambda: sklearn.datasets.fetch_california_housing(return_X_y=True),
    "diabetes_regression": lambda: sklearn.datasets.load_diabetes(return_X_y=True),
    "breast_cancer": lambda: sklearn.datasets.load_breast_cancer(return_X_y=True),
    "iris_control": lambda: sklearn.datasets.load_iris(return_X_y=True),
    "synthetic_rho0": lambda: generate_synthetic(rho=0.0, n=2000, p=8),
    "synthetic_rho05": lambda: generate_synthetic(rho=0.5, n=2000, p=8),
    "synthetic_rho09": lambda: generate_synthetic(rho=0.9, n=2000, p=8),
}


# ---------------------------------------------------------------------------
# Standardized protocol (identical for every dataset)
# ---------------------------------------------------------------------------


def run_protocol(X, y, dataset_name, M=200, N_obs=100, seed=42):
    """
    Standardized validation protocol. No dataset-specific parameters.
    """
    t0 = time.time()

    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # 2. Select N_obs test observations
    rng = np.random.RandomState(seed + 1)
    obs_idx = rng.choice(len(X_test), size=min(N_obs, len(X_test)), replace=False)
    X_explain = X_test[obs_idx]

    # 3. Train M models with IDENTICAL hyperparameter distribution
    rng_models = np.random.RandomState(seed)
    models = []
    for i in range(M):
        s = rng_models.randint(0, 2**31)
        params = {
            "n_estimators": rng_models.choice([100, 200, 300]),
            "max_depth": rng_models.choice([3, 4, 5, 6]),
            "learning_rate": rng_models.choice([0.05, 0.1, 0.2]),
            "colsample_bytree": rng_models.uniform(0.3, 0.8),
            "subsample": rng_models.uniform(0.6, 1.0),
            "random_state": s,
            "n_jobs": 1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, verbose=False)
        models.append(m)
        if (i + 1) % 50 == 0:
            print(f"    [{dataset_name}] trained {i + 1}/{M} models ({time.time() - t0:.0f}s)")

    # 4. Compute SHAP values
    bg = X_train[:100]
    shap_matrices = []
    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)
        if (i + 1) % 50 == 0:
            print(f"    [{dataset_name}] SHAP {i + 1}/{M} ({time.time() - t0:.0f}s)")

    # 5. Analysis (identical for every dataset)
    stack = np.stack(shap_matrices, axis=0)  # (M, N_obs, P)
    signs = np.sign(stack)
    n_models, n_obs, n_features = signs.shape

    # 5a. Coverage conflict + flip rates
    flip_rates = np.zeros((n_obs, n_features))
    coverage_conflict = np.zeros((n_obs, n_features), dtype=int)
    for obs in range(n_obs):
        for feat in range(n_features):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                continue
            n_pos = (nonzero > 0).sum()
            n_neg = (nonzero < 0).sum()
            total = n_pos + n_neg
            flip_rates[obs, feat] = min(n_pos, n_neg) / total
            coverage_conflict[obs, feat] = int(n_pos > 0 and n_neg > 0)

    # 5b. Var[SHAP] and DASH MSE identity check
    var_shap = np.var(stack, axis=0)  # (N_obs, P)
    mean_shap = np.mean(stack, axis=0)  # (N_obs, P)
    dash_mse = np.mean((stack - mean_shap[np.newaxis, :, :]) ** 2, axis=0)  # (N_obs, P)
    identity_max_violation = float(np.max(np.abs(var_shap - dash_mse)))

    # 5c. Statistical tests
    pooled_flip = flip_rates.ravel()

    # Dip test for bimodality
    dip_stat, dip_p = diptest(pooled_flip)

    # Permutation control for dip test (100 perms)
    n_perm = 100
    perm_reject = 0
    rng_perm = np.random.RandomState(99999)
    for _ in range(n_perm):
        perm_signs = rng_perm.choice([-1, 1], size=signs.shape)
        perm_flip = np.zeros((n_obs, n_features))
        for obs in range(n_obs):
            for feat in range(n_features):
                s_perm = perm_signs[:, obs, feat]
                n_pos = (s_perm > 0).sum()
                n_neg = (s_perm < 0).sum()
                perm_flip[obs, feat] = min(n_pos, n_neg) / (n_pos + n_neg)
        _, pp = diptest(perm_flip.ravel())
        if pp < 0.05:
            perm_reject += 1
    perm_frac = perm_reject / n_perm

    # Spearman(CC, flip_rate) — per-feature aggregation
    cc_per_feature = coverage_conflict.mean(axis=0)  # (P,)
    flip_per_feature = flip_rates.mean(axis=0)  # (P,)
    if len(np.unique(cc_per_feature)) > 1 and len(np.unique(flip_per_feature)) > 1:
        spearman_r, spearman_p = spearmanr(cc_per_feature, flip_per_feature)
    else:
        spearman_r, spearman_p = float("nan"), 1.0

    # Var vs CC correlation (per-feature)
    mean_abs_shap = np.mean(np.abs(mean_shap), axis=0)  # (P,)
    relative_var = np.where(mean_abs_shap > 1e-10, np.mean(var_shap, axis=0) / (mean_abs_shap**2), 0)
    if len(np.unique(cc_per_feature)) > 1 and len(np.unique(relative_var)) > 1:
        var_cc_r, var_cc_p = spearmanr(cc_per_feature, relative_var)
    else:
        var_cc_r, var_cc_p = float("nan"), 1.0

    # Summary statistics
    cc_rate = float(coverage_conflict.mean())
    mean_minority = float(flip_rates[coverage_conflict == 1].mean()) if coverage_conflict.sum() > 0 else 0.0

    elapsed = time.time() - t0
    print(f"    [{dataset_name}] done in {elapsed:.0f}s")

    return {
        "dataset": dataset_name,
        "n_samples": len(X),
        "n_features": n_features,
        "n_models": M,
        "n_obs": min(N_obs, len(X_test)),
        "cc_rate": float(cc_rate),
        "mean_minority_fraction": float(mean_minority),
        "dip_stat": float(dip_stat),
        "dip_p": float(dip_p),
        "perm_reject_frac": float(perm_frac),
        "bimodality_valid": bool(dip_p < 0.05 and perm_frac < 0.25),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "var_cc_r": float(var_cc_r),
        "var_cc_p": float(var_cc_p),
        "identity_max_violation": float(identity_max_violation),
        "stable_pct": float(np.mean(pooled_flip < 0.05)),
        "unstable_pct": float(np.mean(pooled_flip > 0.25)),
        "elapsed_s": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("CROSS-DOMAIN VALIDATION — DASH-SHAP IMPOSSIBILITY PREDICTIONS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {len(DATASETS)} | M=200 models each | BH-FDR q=0.05")
    print("=" * 80)

    results = []
    for name, loader in DATASETS.items():
        print(f"\n>>> {name}")
        X, y = loader()
        # Ensure numpy arrays
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        print(f"    shape: X={X.shape}, y={y.shape}")

        r = run_protocol(X, y, name)
        results.append(r)

    # -----------------------------------------------------------------------
    # BH-FDR correction across all p-values
    # -----------------------------------------------------------------------
    all_pvals = []
    all_labels = []
    for r in results:
        all_pvals.extend([r["dip_p"], r["spearman_p"], r["var_cc_p"]])
        all_labels.extend(
            [
                f"{r['dataset']}_dip",
                f"{r['dataset']}_spearman",
                f"{r['dataset']}_var_cc",
            ]
        )

    # Handle NaN p-values: set to 1.0 for FDR
    pvals_clean = [p if not np.isnan(p) else 1.0 for p in all_pvals]
    reject, adjusted_p, _, _ = multipletests(pvals_clean, alpha=0.05, method="fdr_bh")

    # Map adjusted p-values back to results
    idx = 0
    for r in results:
        r["dip_p_adj"] = float(adjusted_p[idx])
        r["dip_reject"] = bool(reject[idx])
        idx += 1
        r["spearman_p_adj"] = float(adjusted_p[idx])
        r["spearman_reject"] = bool(reject[idx])
        idx += 1
        r["var_cc_p_adj"] = float(adjusted_p[idx])
        r["var_cc_reject"] = bool(reject[idx])
        idx += 1

    # -----------------------------------------------------------------------
    # Print consolidated results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CROSS-DOMAIN VALIDATION RESULTS (BH-FDR corrected, q=0.05)")
    print("=" * 120)
    header = (
        f"{'Dataset':<22s} | {'N':>6s} | {'P':>3s} | {'CC%':>6s} | "
        f"{'MinFrac':>7s} | {'Spear_r':>7s} | {'Sp_adj':>8s} | "
        f"{'Dip_p':>7s} | {'Perm%':>5s} | {'Bimod?':>6s} | "
        f"{'Var=MSE':>9s} | {'VarCC_r':>7s} | {'VC_adj':>8s} | "
        f"{'Stbl%':>5s} | {'Unstbl%':>7s} | {'Time':>5s}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        sp_sig = "*" if r["spearman_reject"] else " "
        vc_sig = "*" if r["var_cc_reject"] else " "
        bimod = "YES" if r["bimodality_valid"] else "NO"

        n_str = f"{r['n_samples']}"
        if r["n_samples"] >= 1000:
            n_str = f"{r['n_samples'] / 1000:.0f}K"

        sp_r = f"{r['spearman_r']:.3f}" if not np.isnan(r["spearman_r"]) else "  NaN"
        vc_r = f"{r['var_cc_r']:.3f}" if not np.isnan(r["var_cc_r"]) else "  NaN"

        line = (
            f"{r['dataset']:<22s} | {n_str:>6s} | {r['n_features']:>3d} | "
            f"{r['cc_rate'] * 100:>5.1f}% | "
            f"{r['mean_minority_fraction']:>7.3f} | "
            f"{sp_r:>7s} | {r['spearman_p_adj']:>7.4f}{sp_sig} | "
            f"{r['dip_p']:>7.4f} | {r['perm_reject_frac'] * 100:>4.0f}% | "
            f"{bimod:>6s} | "
            f"{r['identity_max_violation']:>9.2e} | "
            f"{vc_r:>7s} | {r['var_cc_p_adj']:>7.4f}{vc_sig} | "
            f"{r['stable_pct'] * 100:>4.1f}% | "
            f"{r['unstable_pct'] * 100:>6.1f}% | "
            f"{r['elapsed_s']:>4.0f}s"
        )
        print(line)

    print("-" * 120)
    print("* = BH-adjusted p < 0.05")
    print()

    # -----------------------------------------------------------------------
    # Prediction check summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("PREDICTION SURVIVAL SUMMARY")
    print("=" * 80)

    # Prediction 1: CC > 0 for real datasets
    real_datasets = [r for r in results if not r["dataset"].startswith("synthetic")]
    p1_pass = all(r["cc_rate"] > 0.10 for r in real_datasets)
    print(f"\nP1. Coverage conflict > 10% on all real datasets: {'PASS' if p1_pass else 'FAIL'}")
    for r in real_datasets:
        print(f"    {r['dataset']}: CC = {r['cc_rate'] * 100:.1f}%")

    # Prediction 2: Spearman(CC, flip) significantly positive
    p2_count = sum(1 for r in results if r["spearman_reject"] and r["spearman_r"] > 0)
    print(f"\nP2. Spearman(CC, flip) significant & positive: {p2_count}/{len(results)} datasets")
    for r in results:
        sig = "*" if r["spearman_reject"] else " "
        sr = f"{r['spearman_r']:.3f}" if not np.isnan(r["spearman_r"]) else "NaN"
        print(f"    {r['dataset']}: r={sr}, adj_p={r['spearman_p_adj']:.4f}{sig}")

    # Prediction 3: Var = MSE identity holds
    p3_pass = all(r["identity_max_violation"] < 1e-8 for r in results)
    print(f"\nP3. Var[SHAP] = DASH-MSE identity (max violation < 1e-8): {'PASS' if p3_pass else 'FAIL'}")
    for r in results:
        print(f"    {r['dataset']}: max |Var - MSE| = {r['identity_max_violation']:.2e}")

    # Prediction 4: Bimodality (dip test + permutation control)
    p4_count = sum(1 for r in results if r["bimodality_valid"])
    print(f"\nP4. Bimodality (dip p<0.05 & perm reject <25%): {p4_count}/{len(results)} datasets")
    for r in results:
        print(
            f"    {r['dataset']}: dip_p={r['dip_p']:.4f}, "
            f"perm_reject={r['perm_reject_frac'] * 100:.0f}%, "
            f"valid={'YES' if r['bimodality_valid'] else 'NO'}"
        )

    # Negative controls
    print(f"\n{'=' * 80}")
    print("NEGATIVE CONTROLS")
    print(f"{'=' * 80}")

    iris = next((r for r in results if r["dataset"] == "iris_control"), None)
    rho0 = next((r for r in results if r["dataset"] == "synthetic_rho0"), None)

    if iris:
        print(f"\nIris (low-dimensional, easy classification):")
        print(f"    CC rate: {iris['cc_rate'] * 100:.1f}%")
        print(f"    Stable features: {iris['stable_pct'] * 100:.1f}%")

    if rho0:
        print(f"\nSynthetic rho=0 (no feature correlation):")
        print(f"    CC rate: {rho0['cc_rate'] * 100:.1f}%")
        print(f"    Stable features: {rho0['stable_pct'] * 100:.1f}%")

    # -----------------------------------------------------------------------
    # Save raw JSON
    # -----------------------------------------------------------------------
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_domain_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
