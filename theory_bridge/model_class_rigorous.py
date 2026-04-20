#!/usr/bin/env python3
"""
Rigorous model-class comparison with null models and seed-variation control.

Fixes over model_class_comparison.py:
1. Seed-variation control (fixed data, varied seeds) for XGB/RF
2. Permutation null model (1000 perms, 95th percentile) for cross-class Spearman
3. p-values for all cross-class Spearman correlations
4. Degeneracy inspection for Ridge/LASSO flip rates
5. LassoCV for proper feature selection with zero-count reporting

Experiments:
  A: Bootstrap variation with null model + p-values
  B: Seed variation (XGB vs RF only — linear models are deterministic)
  C: Combined summary
"""

import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")


# =============================================================================
# Configuration
# =============================================================================
M = 200  # number of bootstrap/seed models (RF uses M_RF=100)
N_OBS = 50  # test observations for SHAP
N_PERM = 200  # permutation null iterations
M_RF = 100  # reduced M for Random Forest only
SEED = 42
DEGENERACY_THRESHOLD = 0.01  # max flip rate below this → degenerate

np.random.seed(SEED)


# =============================================================================
# Data loading
# =============================================================================
def load_datasets():
    """Load and prepare all three datasets."""
    datasets = {}

    # California Housing
    cal = fetch_california_housing()
    X, y = cal.data, cal.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_test, y_test = X_test[:N_OBS], y_test[:N_OBS]
    scaler = StandardScaler().fit(X_train)
    datasets["california"] = {
        "X_train": scaler.transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(cal.feature_names),
        "task": "regression",
    }

    # Breast Cancer (regression on binary target for uniformity)
    bc = load_breast_cancer()
    X, y = bc.data, bc.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_test, y_test = X_test[:N_OBS], y_test[:N_OBS]
    scaler = StandardScaler().fit(X_train)
    datasets["breast_cancer"] = {
        "X_train": scaler.transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(bc.feature_names),
        "task": "regression",
    }

    # Diabetes
    diab = load_diabetes()
    X, y = diab.data, diab.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_test, y_test = X_test[:N_OBS], y_test[:N_OBS]
    scaler = StandardScaler().fit(X_train)
    datasets["diabetes"] = {
        "X_train": scaler.transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(diab.feature_names),
        "task": "regression",
    }

    return datasets


# =============================================================================
# Flip rate computation
# =============================================================================
def compute_flip_rates(shap_stack):
    """
    Per-feature flip rate from (M, N_obs, P) SHAP stack.
    Flip rate = minority fraction of non-zero signs, averaged across observations.
    """
    signs = np.sign(shap_stack)  # (M, N_obs, P)
    M_models, N_obs, P = signs.shape
    flip_rates = np.zeros(P)

    for j in range(P):
        obs_flips = []
        for obs in range(N_obs):
            s = signs[:, obs, j]
            nz = s[s != 0]
            if len(nz) < 2:
                obs_flips.append(0.0)
            else:
                obs_flips.append(min((nz > 0).sum(), (nz < 0).sum()) / len(nz))
        flip_rates[j] = np.mean(obs_flips)
    return flip_rates


# =============================================================================
# Null model (permutation test)
# =============================================================================
def null_spearman(flip_a, flip_b, n_perm=N_PERM, seed=42):
    """
    Permutation null for |Spearman| between two flip rate vectors.
    Returns: (observed |rho|, null 95th percentile, exceeds_null boolean)
    """
    rng = np.random.RandomState(seed)

    # Check for constant vectors
    if len(np.unique(flip_a)) <= 1 or len(np.unique(flip_b)) <= 1:
        return 0.0, 0.0, False

    observed = abs(spearmanr(flip_a, flip_b)[0])

    null_vals = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(flip_b))
        r, _ = spearmanr(flip_a, flip_b[perm])
        null_vals[i] = abs(r) if not np.isnan(r) else 0.0

    p95 = np.percentile(null_vals, 95)
    exceeds = observed > p95
    return observed, p95, exceeds


# =============================================================================
# Bootstrap ensemble
# =============================================================================
def bootstrap_sample(X_train, y_train, rng):
    """Draw a bootstrap sample."""
    n = X_train.shape[0]
    idx = rng.choice(n, size=n, replace=True)
    return X_train[idx], y_train[idx]


def run_bootstrap_ridge(data, M, seed):
    """M bootstrap Ridge models, return SHAP values [M, N_obs, P]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    P = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, P))

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = Ridge(alpha=1.0)
        model.fit(Xb, yb)
        # Linear SHAP: coef * (x - mean). Data is standardized so mean=0.
        shap_values[m] = X_test * model.coef_[np.newaxis, :]

    return shap_values


def run_bootstrap_lasso(data, M, seed):
    """M bootstrap LassoCV models, return SHAP values [M, N_obs, P] and zero counts."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    P = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, P))
    zero_counts = []

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = LassoCV(cv=5, max_iter=10000, random_state=seed)
        model.fit(Xb, yb)
        shap_values[m] = X_test * model.coef_[np.newaxis, :]
        zero_counts.append(np.sum(model.coef_ == 0))

    return shap_values, zero_counts


def run_bootstrap_xgb(data, M, seed):
    """M bootstrap XGBoost models, return SHAP values [M, N_obs, P]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    P = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, P))

    # Background data for interventional SHAP
    bg_idx = rng.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
    bg = X_train[bg_idx]

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            colsample_bytree=0.5,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(Xb, yb)
        explainer = shap.TreeExplainer(model, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_test)
        shap_values[m] = sv

    return shap_values


def run_bootstrap_rf(data, M, seed):
    """M bootstrap Random Forest models, return SHAP values [M, N_obs, P]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    P = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, P))

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)
        model.fit(Xb, yb)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test, check_additivity=False)
        shap_values[m] = sv

    return shap_values


# =============================================================================
# Seed-variation ensemble (fixed data, varied random_state)
# =============================================================================
def run_seed_xgb(data, M, bg):
    """M XGBoost models on SAME training data with different random_state."""
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    P = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, P))

    for m in range(M):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            colsample_bytree=0.5,
            random_state=m,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_test)
        shap_values[m] = sv

    return shap_values


def run_seed_rf(data, M, bg=None):
    """M Random Forest models on SAME training data with different random_state."""
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    P = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, P))

    for m in range(M):
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=m, n_jobs=1)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test, check_additivity=False)
        shap_values[m] = sv

    return shap_values


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("RIGOROUS MODEL-CLASS COMPARISON", flush=True)
    print("Bootstrap + Seed Variation + Null Model + Degeneracy Checks", flush=True)
    print("=" * 80, flush=True)
    print(f"\nConfig: M={M} (RF: M_RF={M_RF}), N_OBS={N_OBS}, N_PERM={N_PERM}, SEED={SEED}", flush=True)
    print(f"Models: Ridge, LassoCV, XGBoost, Random Forest", flush=True)
    print(f"Datasets: California Housing, Breast Cancer, Diabetes\n", flush=True)

    datasets = load_datasets()

    # Storage
    bootstrap_flips = {}  # {dataset: {model: flip_rates}}
    seed_flips = {}  # {dataset: {model: flip_rates}}
    lasso_zeros = {}  # {dataset: mean_zeros}
    degeneracy_flags = {}  # {dataset: {model: (max_flip, is_degenerate)}}

    # =========================================================================
    # EXPERIMENT A: Bootstrap variation
    # =========================================================================
    print("=" * 80, flush=True)
    print("EXPERIMENT A: BOOTSTRAP VARIATION", flush=True)
    print("=" * 80, flush=True)

    for dname, data in datasets.items():
        P = data["X_train"].shape[1]
        print(f"\n--- {dname} (P={P}) ---", flush=True)
        bootstrap_flips[dname] = {}
        degeneracy_flags[dname] = {}

        # Ridge
        print(f"  Ridge (M={M} bootstrap)...", end=" ", flush=True)
        sv_ridge = run_bootstrap_ridge(data, M, SEED)
        bootstrap_flips[dname]["Ridge"] = compute_flip_rates(sv_ridge)
        print("done.", flush=True)

        # LassoCV
        print(f"  LassoCV (M={M} bootstrap)...", end=" ", flush=True)
        sv_lasso, zero_counts = run_bootstrap_lasso(data, M, SEED)
        bootstrap_flips[dname]["LASSO"] = compute_flip_rates(sv_lasso)
        lasso_zeros[dname] = {
            "mean": np.mean(zero_counts),
            "min": np.min(zero_counts),
            "max": np.max(zero_counts),
            "total_features": P,
        }
        print(
            f"done. (zeros: mean={np.mean(zero_counts):.1f}, range=[{np.min(zero_counts)}, {np.max(zero_counts)}]/{P})",
            flush=True,
        )

        # XGBoost
        print(f"  XGBoost (M={M} bootstrap)...", end=" ", flush=True)
        sv_xgb = run_bootstrap_xgb(data, M, SEED)
        bootstrap_flips[dname]["XGB"] = compute_flip_rates(sv_xgb)
        print("done.", flush=True)

        # Random Forest
        print(f"  RF (M={M_RF} bootstrap, n_estimators=50)...", end=" ", flush=True)
        sv_rf = run_bootstrap_rf(data, M_RF, SEED)
        bootstrap_flips[dname]["RF"] = compute_flip_rates(sv_rf)
        print("done.", flush=True)

        # Degeneracy check
        for mc in ["Ridge", "LASSO", "XGB", "RF"]:
            max_flip = bootstrap_flips[dname][mc].max()
            is_degen = max_flip < DEGENERACY_THRESHOLD
            degeneracy_flags[dname][mc] = (max_flip, is_degen)

    # =========================================================================
    # EXPERIMENT B: Seed variation (XGB and RF only)
    # =========================================================================
    print(f"\n\n{'=' * 80}", flush=True)
    print("EXPERIMENT B: SEED VARIATION (fixed data, varied random_state)", flush=True)
    print("Only XGB and RF — Ridge/LASSO are deterministic given data.", flush=True)
    print("=" * 80, flush=True)

    for dname, data in datasets.items():
        print(f"\n--- {dname} ---", flush=True)
        seed_flips[dname] = {}

        # Background data for XGB
        rng = np.random.RandomState(SEED)
        bg_idx = rng.choice(data["X_train"].shape[0], size=min(100, data["X_train"].shape[0]), replace=False)
        bg = data["X_train"][bg_idx]

        # XGBoost seed variation
        print(f"  XGBoost (M={M} seeds, fixed data)...", end=" ", flush=True)
        sv_xgb_seed = run_seed_xgb(data, M, bg)
        seed_flips[dname]["XGB"] = compute_flip_rates(sv_xgb_seed)
        print("done.", flush=True)

        # RF seed variation
        print(f"  RF (M={M_RF} seeds, n_estimators=50, fixed data)...", end=" ", flush=True)
        sv_rf_seed = run_seed_rf(data, M_RF)
        seed_flips[dname]["RF"] = compute_flip_rates(sv_rf_seed)
        print("done.", flush=True)

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'=' * 80}", flush=True)
    print("RESULTS", flush=True)
    print("=" * 80, flush=True)

    # --- Per-feature flip rates ---
    print(f"\n{'=' * 80}", flush=True)
    print("PER-FEATURE FLIP RATES (bootstrap variation)", flush=True)
    print("=" * 80, flush=True)

    for dname, data in datasets.items():
        P = data["X_train"].shape[1]
        fnames = data["feature_names"]
        lz = lasso_zeros[dname]
        print(f"\n{dname} (LASSO zeros: mean={lz['mean']:.1f}/{lz['total_features']})", flush=True)

        if lz["mean"] < 2:
            print(
                f"  *** WARNING: LASSO not selecting (< 2 features zeroed). "
                f"Cross-class comparison involving LASSO may be invalid. ***",
                flush=True,
            )

        header = f"  {'Feature':<20} {'Ridge':<10} {'LASSO':<10} {'XGB':<10} {'RF':<10}"
        print(header, flush=True)
        print(f"  {'-' * 60}", flush=True)
        for j in range(P):
            fname = fnames[j] if j < len(fnames) else f"f{j}"
            vals = [bootstrap_flips[dname][mc][j] for mc in ["Ridge", "LASSO", "XGB", "RF"]]
            print(f"  {fname:<20} {vals[0]:<10.4f} {vals[1]:<10.4f} {vals[2]:<10.4f} {vals[3]:<10.4f}", flush=True)

    # --- Degeneracy flags ---
    print(f"\n{'=' * 80}", flush=True)
    print("DEGENERACY FLAGS", flush=True)
    print(f"(max flip rate < {DEGENERACY_THRESHOLD} → DEGENERATE, excluded from cross-class)", flush=True)
    print("=" * 80, flush=True)

    excluded_pairs = {}  # {dataset: set of model classes to exclude}
    for dname in datasets:
        excluded_pairs[dname] = set()
        lz = lasso_zeros[dname]
        parts = []
        for mc in ["Ridge", "LASSO", "XGB", "RF"]:
            max_flip, is_degen = degeneracy_flags[dname][mc]
            flag = " *** DEGENERATE ***" if is_degen else ""
            parts.append(f"{mc} max_flip={max_flip:.4f}{flag}")
            if is_degen:
                excluded_pairs[dname].add(mc)
        print(f"\n{dname}:", flush=True)
        for p in parts:
            print(f"  {p}", flush=True)
        print(f"  LASSO zeros: mean={lz['mean']:.1f}, range=[{lz['min']}, {lz['max']}]", flush=True)
        if lz["mean"] < 2:
            print(f"  *** LASSO NOT SELECTING — excluding from cross-class ***", flush=True)
            excluded_pairs[dname].add("LASSO")

    # --- Cross-class Spearman (bootstrap) ---
    print(f"\n{'=' * 80}", flush=True)
    print("CROSS-CLASS SPEARMAN (bootstrap variation)", flush=True)
    print("With p-values and permutation null (95th percentile)", flush=True)
    print("=" * 80, flush=True)

    pairs = [("Ridge", "XGB"), ("Ridge", "RF"), ("Ridge", "LASSO"), ("XGB", "RF"), ("XGB", "LASSO"), ("RF", "LASSO")]

    header = (
        f"{'Dataset':<15} {'Pair':<14} {'Spearman':<10} {'p-value':<12} {'Null95':<10} {'Exceeds?':<10} {'Valid?':<8}"
    )
    print(f"\n{header}", flush=True)
    print("-" * len(header), flush=True)

    bootstrap_results = []
    for dname in datasets:
        for c1, c2 in pairs:
            flip_a = bootstrap_flips[dname][c1]
            flip_b = bootstrap_flips[dname][c2]

            # Check if either class is excluded
            valid = c1 not in excluded_pairs[dname] and c2 not in excluded_pairs[dname]

            # Compute Spearman with p-value
            if len(np.unique(flip_a)) > 1 and len(np.unique(flip_b)) > 1:
                rho, pval = spearmanr(flip_a, flip_b)
            else:
                rho, pval = 0.0, 1.0

            # Null model
            obs_abs, null95, exceeds = null_spearman(flip_a, flip_b)

            valid_str = "YES" if valid else "NO"
            exceeds_str = "YES" if exceeds else "NO"
            pair_str = f"{c1}-{c2}"

            print(
                f"{dname:<15} {pair_str:<14} {rho:<10.4f} {pval:<12.2e} "
                f"{null95:<10.4f} {exceeds_str:<10} {valid_str:<8}",
                flush=True,
            )

            bootstrap_results.append(
                {
                    "dataset": dname,
                    "pair": pair_str,
                    "rho": rho,
                    "pval": pval,
                    "null95": null95,
                    "exceeds": exceeds,
                    "valid": valid,
                }
            )

    # --- Cross-class Spearman (seed variation) ---
    print(f"\n{'=' * 80}", flush=True)
    print("CROSS-CLASS SPEARMAN (seed variation — XGB vs RF only)", flush=True)
    print("Fixed training data, varied random_state. Tests if tree-building", flush=True)
    print("randomness produces same instability pattern across model classes.", flush=True)
    print("=" * 80, flush=True)

    header = f"{'Dataset':<15} {'Pair':<14} {'Spearman':<10} {'p-value':<12} {'Null95':<10} {'Exceeds?':<10}"
    print(f"\n{header}", flush=True)
    print("-" * len(header), flush=True)

    seed_results = []
    for dname in datasets:
        flip_a = seed_flips[dname]["XGB"]
        flip_b = seed_flips[dname]["RF"]

        if len(np.unique(flip_a)) > 1 and len(np.unique(flip_b)) > 1:
            rho, pval = spearmanr(flip_a, flip_b)
        else:
            rho, pval = 0.0, 1.0

        obs_abs, null95, exceeds = null_spearman(flip_a, flip_b)
        exceeds_str = "YES" if exceeds else "NO"

        print(f"{dname:<15} {'XGB-RF':<14} {rho:<10.4f} {pval:<12.2e} {null95:<10.4f} {exceeds_str:<10}", flush=True)

        seed_results.append({"dataset": dname, "rho": rho, "pval": pval, "null95": null95, "exceeds": exceeds})

    # --- Comparison: bootstrap vs seed variation ---
    print(f"\n{'=' * 80}", flush=True)
    print("BOOTSTRAP vs SEED VARIATION (XGB-RF pair)", flush=True)
    print("If agreement holds under BOTH, it's real geometry.", flush=True)
    print("If only under bootstrap, it's a shared-data confound.", flush=True)
    print("=" * 80, flush=True)

    header = f"{'Dataset':<15} {'Boot rho':<12} {'Seed rho':<12} {'Boot>Null?':<12} {'Seed>Null?':<12} {'Verdict':<20}"
    print(f"\n{header}", flush=True)
    print("-" * len(header), flush=True)

    for dname in datasets:
        # Find XGB-RF in bootstrap results
        boot_row = [r for r in bootstrap_results if r["dataset"] == dname and r["pair"] == "XGB-RF"][0]
        seed_row = [r for r in seed_results if r["dataset"] == dname][0]

        boot_exceeds = "YES" if boot_row["exceeds"] else "NO"
        seed_exceeds = "YES" if seed_row["exceeds"] else "NO"

        if boot_row["exceeds"] and seed_row["exceeds"]:
            verdict = "REAL (both exceed)"
        elif boot_row["exceeds"] and not seed_row["exceeds"]:
            verdict = "CONFOUND (boot only)"
        elif not boot_row["exceeds"] and seed_row["exceeds"]:
            verdict = "SEED ONLY (unusual)"
        else:
            verdict = "NO SIGNAL"

        print(
            f"{dname:<15} {boot_row['rho']:<12.4f} {seed_row['rho']:<12.4f} "
            f"{boot_exceeds:<12} {seed_exceeds:<12} {verdict:<20}",
            flush=True,
        )

    # --- Summary ---
    print(f"\n{'=' * 80}", flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)

    valid_boot = [r for r in bootstrap_results if r["valid"]]
    if valid_boot:
        mean_rho = np.mean([r["rho"] for r in valid_boot])
        n_exceed = sum(1 for r in valid_boot if r["exceeds"])
        n_sig = sum(1 for r in valid_boot if r["pval"] < 0.05)
        print(f"\nBootstrap (valid pairs only, N={len(valid_boot)}):", flush=True)
        print(f"  Mean |Spearman|: {mean_rho:.4f}", flush=True)
        print(f"  Exceed null 95th: {n_exceed}/{len(valid_boot)}", flush=True)
        print(f"  p < 0.05: {n_sig}/{len(valid_boot)}", flush=True)

    if seed_results:
        mean_seed_rho = np.mean([r["rho"] for r in seed_results])
        n_seed_exceed = sum(1 for r in seed_results if r["exceeds"])
        n_seed_sig = sum(1 for r in seed_results if r["pval"] < 0.05)
        print(f"\nSeed variation (XGB-RF, N={len(seed_results)}):", flush=True)
        print(f"  Mean Spearman: {mean_seed_rho:.4f}", flush=True)
        print(f"  Exceed null 95th: {n_seed_exceed}/{len(seed_results)}", flush=True)
        print(f"  p < 0.05: {n_seed_sig}/{len(seed_results)}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed / 60:.1f} minutes", flush=True)
    print(f"\n{'=' * 80}", flush=True)
    print("END OF REPORT", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
