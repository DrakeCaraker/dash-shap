#!/usr/bin/env python3
"""
Spectral Predictor Validation — definitive test of model-class dependence hypothesis.

Three questions:
1. Do GLOBAL spectral predictors (eigenvalue loading, leverage) predict per-feature
   instability better than pairwise predictors?
2. Does ridge regression SHAP show the spectral pattern that XGBoost SHAP doesn't?
3. Is the barrier model-class dependence (spectral works for ridge, fails for XGBoost)?

Design: 7 spectral/leverage predictors × 2 model classes (XGBoost, Ridge) × 4 datasets.
"""

import numpy as np
import xgboost as xgb
import shap
from numpy.linalg import eigh, inv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings
import time
import sys

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

M = 200
N_OBS = 100
SEED = 42


# ===========================================================================
# Spectral predictors (data-only, per-feature)
# ===========================================================================


def compute_spectral_predictors(X, y):
    """All spectral/leverage predictors for P features."""
    P = X.shape[1]

    # Standardize for eigendecomposition
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ys = (y - y.mean()) / (y.std() + 1e-10)

    # 1. Feature correlation eigendecomposition
    corr = np.corrcoef(Xs.T)  # P×P
    eigenvalues, eigenvectors = eigh(corr)  # sorted ascending

    # 2. Per-feature loading on SMALLEST eigenvalue
    small_loading = eigenvectors[:, 0] ** 2

    # 3. Per-feature loading on bottom-k eigenvalues (k = P//3)
    k = max(1, P // 3)
    bottom_k_loading = np.sum(eigenvectors[:, :k] ** 2, axis=1)

    # 4. VIF
    try:
        vif = np.diag(inv(corr))
    except np.linalg.LinAlgError:
        vif = np.ones(P)

    # 5. Ridge coefficient stability proxy
    ridge_coefs = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        r = Ridge(alpha=alpha)
        r.fit(Xs, ys)
        ridge_coefs.append(r.coef_)
    ridge_coefs = np.array(ridge_coefs)  # (n_alpha, P)
    ridge_sign_consistency = np.mean(np.sign(ridge_coefs) == np.sign(ridge_coefs[0]), axis=0)
    ridge_coef_cv = np.std(ridge_coefs, axis=0) / (np.abs(np.mean(ridge_coefs, axis=0)) + 1e-10)

    # 6. Marginal R²
    marginal_r2 = np.array([np.corrcoef(Xs[:, j], ys)[0, 1] ** 2 for j in range(P)])

    # 7. Importance-weighted small-eigenvalue loading
    weighted_loading = bottom_k_loading / np.maximum(marginal_r2, 0.01)

    return {
        "small_eig_loading": small_loading,
        "bottom_k_loading": bottom_k_loading,
        "vif": vif,
        "ridge_sign_consistency": ridge_sign_consistency,
        "ridge_coef_cv": ridge_coef_cv,
        "marginal_r2": marginal_r2,
        "weighted_loading": weighted_loading,
        "eigenvalues": eigenvalues,
    }


# ===========================================================================
# Ridge SHAP instability
# ===========================================================================


def ridge_shap_instability(X_train, y_train, X_explain, M=200, seed=42):
    """
    Train M ridge models with varied lambda, compute SHAP sign flip rates.
    For linear models: SHAP_ij = coef_j * (x_j - E[x_j]).
    After standardization E[x_j]=0, so SHAP_ij = coef_j * x_j.
    """
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_explain = scaler.transform(X_explain)
    ys_train = (y_train - y_train.mean()) / (y_train.std() + 1e-10)

    rng = np.random.RandomState(seed)
    coefs = []
    for i in range(M):
        alpha = 10 ** rng.uniform(-2, 3)  # log-uniform from 0.01 to 1000
        r = Ridge(alpha=alpha)
        r.fit(Xs_train, ys_train)
        coefs.append(r.coef_)
    coefs = np.stack(coefs)  # (M, P)

    P = Xs_train.shape[1]
    N_obs = Xs_explain.shape[0]

    flip_rates = np.zeros(P)
    for j in range(P):
        feat_flips = []
        for obs in range(N_obs):
            shap_vals = coefs[:, j] * Xs_explain[obs, j]  # (M,)
            signs = np.sign(shap_vals)
            nonzero = signs[signs != 0]
            if len(nonzero) < 2:
                feat_flips.append(0.0)
            else:
                n_pos = (nonzero > 0).sum()
                n_neg = (nonzero < 0).sum()
                feat_flips.append(min(n_pos, n_neg) / (n_pos + n_neg))
        flip_rates[j] = np.mean(feat_flips)

    return flip_rates, coefs


# ===========================================================================
# XGBoost SHAP instability
# ===========================================================================


def xgboost_shap_instability(X_train, y_train, X_explain, M=200, seed=42):
    """
    Train M XGBoost models with varied hyperparameters, compute SHAP sign flip rates.
    """
    rng = np.random.RandomState(seed)
    P = X_train.shape[1]
    N_obs = X_explain.shape[0]
    bg = X_train[:100]

    # Collect SHAP matrices
    shap_stack = np.zeros((M, N_obs, P))

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
        shap_stack[i] = sv

        if (i + 1) % 50 == 0:
            print(f"      [XGB] Trained {i + 1}/{M} models")

    # Per-feature flip rates
    flip_rates = np.zeros(P)
    for j in range(P):
        feat_flips = []
        for obs in range(N_obs):
            signs = np.sign(shap_stack[:, obs, j])
            nonzero = signs[signs != 0]
            if len(nonzero) < 2:
                feat_flips.append(0.0)
            else:
                n_pos = (nonzero > 0).sum()
                n_neg = (nonzero < 0).sum()
                feat_flips.append(min(n_pos, n_neg) / (n_pos + n_neg))
        flip_rates[j] = np.mean(feat_flips)

    return flip_rates, shap_stack


# ===========================================================================
# Synthetic data generator
# ===========================================================================


def generate_synthetic(g=2, rho=0.9, n=2000, seed=42):
    """Generate synthetic data with g correlated features + independent features."""
    P = max(10, g + 2)
    rng = np.random.RandomState(seed)

    cov = np.eye(P)
    for i in range(g):
        for j in range(g):
            if i != j:
                cov[i, j] = rho

    X = rng.multivariate_normal(np.zeros(P), cov, n)
    coeffs = rng.uniform(0.5, 2.0, P)
    y = X @ coeffs + rng.normal(0, 0.5, n)

    return X, y


# ===========================================================================
# Dataset loading
# ===========================================================================


def load_datasets():
    """Load all 4 datasets, return dict of (X_train, y_train, X_explain)."""
    datasets = {}

    # 1. Synthetic g=2, rho=0.9
    X, y = generate_synthetic(g=2, rho=0.9)
    X_tr, X_ex, y_tr, _ = train_test_split(X, y, test_size=N_OBS, random_state=SEED)
    datasets["syn"] = (X_tr, y_tr, X_ex[:N_OBS])

    # 2. California Housing
    cal = fetch_california_housing()
    X_tr, X_ex, y_tr, _ = train_test_split(cal.data, cal.target, test_size=N_OBS, random_state=SEED)
    datasets["cal"] = (X_tr, y_tr, X_ex[:N_OBS])

    # 3. Breast Cancer
    bc = load_breast_cancer()
    X_tr, X_ex, y_tr, _ = train_test_split(bc.data, bc.target, test_size=N_OBS, random_state=SEED)
    datasets["bc"] = (X_tr, y_tr, X_ex[:N_OBS])

    # 4. Diabetes
    diab = load_diabetes()
    X_tr, X_ex, y_tr, _ = train_test_split(diab.data, diab.target, test_size=N_OBS, random_state=SEED)
    datasets["diab"] = (X_tr, y_tr, X_ex[:N_OBS])

    return datasets


# ===========================================================================
# Main validation
# ===========================================================================


def safe_spearman(x, y):
    """Spearman correlation, handling constant arrays."""
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    rho, _ = spearmanr(x, y)
    return rho if np.isfinite(rho) else 0.0


def run_validation():
    print("=" * 72)
    print("SPECTRAL PREDICTOR VALIDATION")
    print("Tests whether data-level spectral quantities predict SHAP instability")
    print("and whether the barrier is model-class dependence.")
    print("=" * 72)
    print()

    datasets = load_datasets()
    ds_names = ["syn", "cal", "bc", "diab"]

    predictor_names = [
        "small_eig_loading",
        "bottom_k_loading",
        "vif",
        "ridge_sign_consistency",
        "ridge_coef_cv",
        "marginal_r2",
        "weighted_loading",
    ]

    # Storage: predictor -> dataset -> (xgb_spearman, ridge_spearman)
    results = {p: {} for p in predictor_names}
    model_class_corr = {}

    total_start = time.time()

    for ds_name in ds_names:
        print(f"\n{'─' * 72}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─' * 72}")
        X_train, y_train, X_explain = datasets[ds_name]
        P = X_train.shape[1]
        print(f"  N_train={X_train.shape[0]}, P={P}, N_explain={X_explain.shape[0]}")

        # 1. Spectral predictors
        t0 = time.time()
        predictors = compute_spectral_predictors(X_train, y_train)
        print(f"  Spectral predictors computed ({time.time() - t0:.1f}s)")
        print(f"    Eigenvalue range: [{predictors['eigenvalues'][0]:.4f}, {predictors['eigenvalues'][-1]:.4f}]")
        print(f"    Condition number: {predictors['eigenvalues'][-1] / max(predictors['eigenvalues'][0], 1e-10):.1f}")

        # 2. XGBoost instability
        t0 = time.time()
        print(f"  Training {M} XGBoost models...")
        xgb_flip, _ = xgboost_shap_instability(X_train, y_train, X_explain, M=M, seed=SEED)
        print(f"  XGBoost done ({time.time() - t0:.1f}s)")
        print(
            f"    XGB flip rates: mean={xgb_flip.mean():.4f}, "
            f"std={xgb_flip.std():.4f}, range=[{xgb_flip.min():.4f}, {xgb_flip.max():.4f}]"
        )

        # 3. Ridge instability
        t0 = time.time()
        print(f"  Training {M} Ridge models...")
        ridge_flip, _ = ridge_shap_instability(X_train, y_train, X_explain, M=M, seed=SEED)
        print(f"  Ridge done ({time.time() - t0:.1f}s)")
        print(
            f"    Ridge flip rates: mean={ridge_flip.mean():.4f}, "
            f"std={ridge_flip.std():.4f}, range=[{ridge_flip.min():.4f}, {ridge_flip.max():.4f}]"
        )

        # 4. Model-class correlation
        rho_mc = safe_spearman(xgb_flip, ridge_flip)
        model_class_corr[ds_name] = rho_mc
        print(f"  Spearman(XGB_flip, Ridge_flip) = {rho_mc:.4f}")

        # 5. Predictor correlations
        for pred_name in predictor_names:
            pred_vals = predictors[pred_name]
            # For ridge_sign_consistency, LOWER consistency → HIGHER instability
            # So we expect NEGATIVE correlation (or flip sign)
            xgb_rho = safe_spearman(pred_vals, xgb_flip)
            ridge_rho = safe_spearman(pred_vals, ridge_flip)
            results[pred_name][ds_name] = (xgb_rho, ridge_rho)

    elapsed = time.time() - total_start
    print(f"\n\nTotal runtime: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ===========================================================================
    # Results table
    # ===========================================================================
    print("\n\n")
    print("=" * 90)
    print("RESULTS: Spearman correlation between predictor and flip rate")
    print("=" * 90)
    print()

    # Header
    header = f"{'Predictor':<22}"
    header += "| XGBoost FlipRate Spearman      | Ridge FlipRate Spearman"
    print(header)
    sub_header = f"{'':22}"
    sub_header += f"| {'syn':>6} {'cal':>6} {'bc':>6} {'diab':>6}"
    sub_header += f" | {'syn':>6} {'cal':>6} {'bc':>6} {'diab':>6}"
    print(sub_header)
    print("─" * 90)

    for pred_name in predictor_names:
        row = f"{pred_name:<22}|"
        # XGB columns
        for ds in ds_names:
            val = results[pred_name][ds][0]
            row += f" {val:>6.3f}"
        row += " |"
        # Ridge columns
        for ds in ds_names:
            val = results[pred_name][ds][1]
            row += f" {val:>6.3f}"
        print(row)

    print()
    print("─" * 90)
    print("MODEL-CLASS COMPARISON: Spearman(XGB_flip, Ridge_flip) per dataset")
    print("─" * 90)
    for ds in ds_names:
        print(f"  {ds:>6}: {model_class_corr[ds]:>7.4f}")

    # ===========================================================================
    # Interpretation
    # ===========================================================================
    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    # Check if spectral predictors work for Ridge
    ridge_works = []
    for pred_name in ["small_eig_loading", "bottom_k_loading", "vif", "weighted_loading"]:
        ridge_corrs = [abs(results[pred_name][ds][1]) for ds in ds_names]
        ridge_works.append(np.mean(ridge_corrs))
    ridge_mean = np.mean(ridge_works)

    # Check if spectral predictors work for XGBoost
    xgb_works = []
    for pred_name in ["small_eig_loading", "bottom_k_loading", "vif", "weighted_loading"]:
        xgb_corrs = [abs(results[pred_name][ds][0]) for ds in ds_names]
        xgb_works.append(np.mean(xgb_corrs))
    xgb_mean = np.mean(xgb_works)

    print(f"\n  Mean |Spearman| for spectral predictors (4 key ones):")
    print(f"    Ridge:   {ridge_mean:.4f}")
    print(f"    XGBoost: {xgb_mean:.4f}")
    print(f"    Gap:     {ridge_mean - xgb_mean:.4f}")

    mc_mean = np.mean(list(model_class_corr.values()))
    print(f"\n  Mean model-class correlation: {mc_mean:.4f}")

    print(f"\n  VERDICT:")
    if ridge_mean > 0.4 and xgb_mean < 0.3:
        print("    → BARRIER IS MODEL-CLASS DEPENDENCE")
        print("      Spectral predictors work for Ridge but NOT XGBoost.")
        print("      Tree splits break the spectral structure.")
    elif ridge_mean > 0.4 and xgb_mean > 0.4:
        print("    → SPECTRAL THEORY WORKS GENERALLY")
        print("      Spectral predictors predict instability for BOTH model classes.")
    elif ridge_mean < 0.3 and xgb_mean < 0.3:
        print("    → SPECTRAL APPROACH FAILS ENTIRELY")
        print("      Spectral predictors do not predict instability for either class.")
    else:
        print(f"    → MIXED RESULT (Ridge={ridge_mean:.3f}, XGB={xgb_mean:.3f})")
        print("      Partial spectral signal exists. Further investigation needed.")

    if mc_mean > 0.5:
        print(f"\n    Model-class correlation is HIGH ({mc_mean:.3f})")
        print("    → Instability patterns are largely data-determined (model-independent)")
    elif mc_mean < 0.2:
        print(f"\n    Model-class correlation is LOW ({mc_mean:.3f})")
        print("    → Instability patterns are model-class-dependent")
    else:
        print(f"\n    Model-class correlation is MODERATE ({mc_mean:.3f})")
        print("    → Partial model-class dependence")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_validation()
