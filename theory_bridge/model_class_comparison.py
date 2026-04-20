"""
Unified Model-Class Comparison — Bootstrap Standardized Across 4 Model Classes

Design: EVERY model class tested with the SAME source of variation (bootstrap
resampling of training data). Fixed hyperparameters per class. The only variation
is which bootstrap sample is drawn.

Model classes: Ridge, LASSO, XGBoost, Random Forest
Datasets: California Housing, Breast Cancer, Diabetes
M = 200 bootstrap models per class per dataset
N_obs = 100 test observations
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.linear_model import Ridge, Lasso
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
M = 200  # number of bootstrap models
N_OBS = 100  # test observations
SEED = 42

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
    # Take N_OBS test points
    X_test, y_test = X_test[:N_OBS], y_test[:N_OBS]
    scaler = StandardScaler().fit(X_train)
    datasets["california"] = {
        "X_train": scaler.transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": cal.feature_names,
        "train_mean": np.zeros(X_train.shape[1]),  # already standardized
        "task": "regression",
    }

    # Breast Cancer (regression on target for uniformity)
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
        "feature_names": bc.feature_names,
        "train_mean": np.zeros(X_train.shape[1]),
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
        "feature_names": diab.feature_names,
        "train_mean": np.zeros(X_train.shape[1]),
        "task": "regression",
    }

    return datasets


# =============================================================================
# Bootstrap + SHAP computation per model class
# =============================================================================
def bootstrap_sample(X_train, y_train, rng):
    """Draw a bootstrap sample (sample with replacement, same size)."""
    n = X_train.shape[0]
    idx = rng.choice(n, size=n, replace=True)
    return X_train[idx], y_train[idx]


def compute_shap_linear(model, X_test, train_mean):
    """SHAP for linear model: coef_j * (x_j - mean_j)."""
    coefs = model.coef_
    # train_mean is 0 because data is standardized
    return X_test * coefs[np.newaxis, :]


def run_bootstrap_ridge(data, M, seed):
    """Run M bootstrap Ridge models, return SHAP values [M, N_obs, n_features]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    n_features = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, n_features))

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = Ridge(alpha=1.0)
        model.fit(Xb, yb)
        shap_values[m] = compute_shap_linear(model, X_test, data["train_mean"])

    return shap_values


def run_bootstrap_lasso(data, M, seed):
    """Run M bootstrap LASSO models, return SHAP values [M, N_obs, n_features]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    n_features = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, n_features))

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = Lasso(alpha=0.01, max_iter=10000)
        model.fit(Xb, yb)
        shap_values[m] = compute_shap_linear(model, X_test, data["train_mean"])

    return shap_values


def run_bootstrap_xgb(data, M, seed):
    """Run M bootstrap XGBoost models, return SHAP values [M, N_obs, n_features]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    n_features = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, n_features))

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
    """Run M bootstrap Random Forest models, return SHAP values [M, N_obs, n_features]."""
    rng = np.random.RandomState(seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    n_features = X_train.shape[1]
    shap_values = np.zeros((M, N_OBS, n_features))

    for m in range(M):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=1)
        model.fit(Xb, yb)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        shap_values[m] = sv

    return shap_values


# =============================================================================
# Metrics
# =============================================================================
def compute_flip_rate(shap_values):
    """
    Per-feature flip rate: minority fraction of SHAP signs across M models.
    shap_values: [M, N_obs, n_features]
    Returns: [n_features] — averaged across observations.
    """
    # For each observation and feature, compute the minority sign fraction
    pos_frac = (shap_values > 0).mean(axis=0)  # [N_obs, n_features]
    flip_rate = np.minimum(pos_frac, 1 - pos_frac)  # [N_obs, n_features]
    return flip_rate.mean(axis=0)  # [n_features]


def compute_var_shap(shap_values):
    """
    Per-feature variance of SHAP across bootstrap models.
    Returns: [n_features] — averaged across observations.
    """
    var = shap_values.var(axis=0)  # [N_obs, n_features]
    return var.mean(axis=0)  # [n_features]


def compute_coverage_conflict_rate(shap_values):
    """
    Per-feature coverage conflict rate: fraction of observations where
    both positive and negative SHAP signs appear across bootstrap models.
    Returns: [n_features]
    """
    # For each obs, feature: does both a positive and negative sign appear?
    has_pos = (shap_values > 0).any(axis=0)  # [N_obs, n_features]
    has_neg = (shap_values < 0).any(axis=0)  # [N_obs, n_features]
    conflict = (has_pos & has_neg).astype(float)  # [N_obs, n_features]
    return conflict.mean(axis=0)  # [n_features]


def compute_dash_mse(shap_values):
    """
    DASH MSE = E[(phi - E[phi])^2] = Var[phi] per feature per obs, averaged.
    This should equal Var[SHAP] by mathematical identity.
    """
    # Mean SHAP per obs per feature
    mean_shap = shap_values.mean(axis=0)  # [N_obs, n_features]
    # MSE = mean over M of (shap - mean_shap)^2
    mse = ((shap_values - mean_shap[np.newaxis, :, :]) ** 2).mean(axis=0)  # [N_obs, n_features]
    return mse.mean(axis=0)  # [n_features]


# =============================================================================
# Level 2 Predictors
# =============================================================================
def compute_small_eig_loading(X_train):
    """
    Small-eigenvalue loading: for each feature, its loading on the
    smallest eigenvalue directions of the correlation matrix.
    """
    corr = np.corrcoef(X_train, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Take bottom 25% of eigenvalues
    n_small = max(1, len(eigenvalues) // 4)
    small_vecs = eigenvectors[:, :n_small]  # columns are eigenvectors
    # Loading = sum of squared loadings on small eigenvectors
    loading = (small_vecs**2).sum(axis=1)
    return loading


def compute_lasso_boundary(data, seed):
    """
    LASSO boundary proximity: |coef_j| for features near the L1 threshold.
    Small |coef| means near the selection boundary → unstable.
    We invert: boundary_proximity = 1 / (|coef| + eps) so higher = more unstable.
    """
    model = Lasso(alpha=0.01, max_iter=10000)
    model.fit(data["X_train"], data["y_train"])
    coefs = np.abs(model.coef_)
    # Proximity to boundary: inverse of |coef|
    boundary_prox = 1.0 / (coefs + 1e-8)
    return boundary_prox


def compute_xgb_importance_var(data, seed):
    """
    XGBoost: feature importance variance across trees within a single model.
    """
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, colsample_bytree=0.5, random_state=seed, verbosity=0
    )
    model.fit(data["X_train"], data["y_train"])

    # Get per-tree feature importances via booster
    booster = model.get_booster()
    n_features = data["X_train"].shape[1]

    # Get gain-based importance per tree
    trees = booster.get_dump()
    n_trees = len(trees)

    # Use SHAP-based approach: get feature importances from the model
    # and compute variance across subsets of trees
    importances = []
    chunk_size = max(1, n_trees // 10)
    for i in range(0, n_trees - chunk_size + 1, chunk_size):
        # Train a model on subset of iterations
        sub_model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, colsample_bytree=0.5, random_state=seed, verbosity=0
        )
        sub_model.fit(data["X_train"], data["y_train"])
        imp = sub_model.feature_importances_
        importances.append(imp)

    # Simpler approach: use the built-in feature importance with noise
    # Actually, compute importance variance via bootstrap of the single model's predictions
    rng = np.random.RandomState(seed)
    importances = np.zeros((20, n_features))
    for i in range(20):
        Xb, yb = bootstrap_sample(data["X_train"], data["y_train"], rng)
        m = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            colsample_bytree=0.5,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        m.fit(Xb, yb)
        importances[i] = m.feature_importances_

    return importances.var(axis=0)


def compute_rf_importance_var(data, seed):
    """
    RF: feature importance variance across trees within a single model.
    """
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=seed)
    model.fit(data["X_train"], data["y_train"])

    # Get per-tree importances
    n_features = data["X_train"].shape[1]
    tree_importances = np.zeros((len(model.estimators_), n_features))
    for i, tree in enumerate(model.estimators_):
        tree_importances[i] = tree.feature_importances_

    return tree_importances.var(axis=0)


# =============================================================================
# Main
# =============================================================================
def main():
    print("Loading datasets...")
    datasets = load_datasets()

    model_classes = ["Ridge", "LASSO", "XGB", "RF"]
    runners = {
        "Ridge": run_bootstrap_ridge,
        "LASSO": run_bootstrap_lasso,
        "XGB": run_bootstrap_xgb,
        "RF": run_bootstrap_rf,
    }

    # Storage
    all_shap = {}  # {dataset: {model_class: shap_values}}
    all_flip = {}  # {dataset: {model_class: flip_rate per feature}}
    all_var = {}  # {dataset: {model_class: var_shap per feature}}
    all_cc = {}  # {dataset: {model_class: coverage_conflict per feature}}
    all_dash = {}  # {dataset: {model_class: dash_mse per feature}}

    for dname, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dname} ({data['X_train'].shape[1]} features)")
        print(f"{'=' * 60}")

        all_shap[dname] = {}
        all_flip[dname] = {}
        all_var[dname] = {}
        all_cc[dname] = {}
        all_dash[dname] = {}

        for mc in model_classes:
            print(f"  Running {mc} (M={M} bootstrap models)...", end=" ", flush=True)
            sv = runners[mc](data, M, SEED)
            all_shap[dname][mc] = sv
            all_flip[dname][mc] = compute_flip_rate(sv)
            all_var[dname][mc] = compute_var_shap(sv)
            all_cc[dname][mc] = compute_coverage_conflict_rate(sv)
            all_dash[dname][mc] = compute_dash_mse(sv)
            print("done.")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n\n")
    print("=" * 77)
    print("UNIFIED MODEL-CLASS COMPARISON (bootstrap resampling, M=200)")
    print("=" * 77)

    # --- A. Cross-class instability correlation ---
    print("\nA. CROSS-CLASS INSTABILITY CORRELATION (Spearman of per-feature flip rates)")
    pairs = [("Ridge", "XGB"), ("Ridge", "RF"), ("Ridge", "LASSO"), ("XGB", "RF"), ("XGB", "LASSO"), ("RF", "LASSO")]
    pair_labels = ["Ridge-XGB", "Ridge-RF", "Ridge-LASSO", "XGB-RF", "XGB-LASSO", "RF-LASSO"]

    header = f"{'Dataset':<15}" + "".join(f"{pl:<12}" for pl in pair_labels)
    print(header)
    print("-" * len(header))

    table_a = {}
    for dname in datasets:
        row = []
        for c1, c2 in pairs:
            r, p = spearmanr(all_flip[dname][c1], all_flip[dname][c2])
            row.append(r)
        table_a[dname] = row
        print(f"{dname:<15}" + "".join(f"{v:<12.3f}" for v in row))

    # --- B. Coverage Conflict Universality ---
    print(f"\nB. COVERAGE CONFLICT UNIVERSALITY (Spearman with flip rate)")
    header = f"{'Dataset':<15}" + "".join(f"{mc:<10}" for mc in model_classes)
    print(header)
    print("-" * len(header))

    table_b = {}
    for dname in datasets:
        row = []
        for mc in model_classes:
            r, p = spearmanr(all_cc[dname][mc], all_flip[dname][mc])
            row.append(r)
        table_b[dname] = row
        print(f"{dname:<15}" + "".join(f"{v:<10.3f}" for v in row))

    # --- C. Level 2 Predictor × Model Class Matrix ---
    print(f"\nC. LEVEL 2 PREDICTOR × MODEL CLASS MATRIX")
    print("   (Spearman correlation of predictor with flip_rate, averaged across datasets)")

    # Compute Level 2 predictors per dataset
    l2_predictors = {}
    for dname, data in datasets.items():
        l2_predictors[dname] = {
            "small_eig_loading": compute_small_eig_loading(data["X_train"]),
            "lasso_boundary": compute_lasso_boundary(data, SEED),
            "xgb_importance_var": compute_xgb_importance_var(data, SEED),
            "rf_importance_var": compute_rf_importance_var(data, SEED),
        }

    predictor_names = ["small_eig_loading", "lasso_boundary", "xgb_importance_var", "rf_importance_var"]
    header = f"{'Predictor':<22}" + "".join(f"{'→ ' + mc:<10}" for mc in model_classes)
    print(header)
    print("-" * len(header))

    table_c = {}
    for pred_name in predictor_names:
        row = []
        for mc in model_classes:
            corrs = []
            for dname in datasets:
                pred = l2_predictors[dname][pred_name]
                flip = all_flip[dname][mc]
                r, p = spearmanr(pred, flip)
                if not np.isnan(r):
                    corrs.append(r)
            row.append(np.mean(corrs) if corrs else np.nan)
        table_c[pred_name] = row
        print(f"{pred_name:<22}" + "".join(f"{v:<10.3f}" for v in row))

    # --- D. Var[SHAP] = DASH MSE ---
    print(f"\nD. VAR[SHAP] = DASH MSE (max relative violation per class per dataset)")
    header = f"{'Dataset':<15}" + "".join(f"{mc:<10}" for mc in model_classes)
    print(header)
    print("-" * len(header))

    table_d = {}
    for dname in datasets:
        row = []
        for mc in model_classes:
            var_shap = all_var[dname][mc]
            dash_mse = all_dash[dname][mc]
            # Max absolute difference relative to var_shap
            denom = np.maximum(var_shap, 1e-12)
            max_violation = np.max(np.abs(var_shap - dash_mse) / denom)
            row.append(max_violation)
        table_d[dname] = row
        print(f"{dname:<15}" + "".join(f"{v:<10.2e}" for v in row))

    # --- Summary ---
    print(f"\n{'=' * 77}")
    print("SUMMARY")
    print(f"{'=' * 77}")

    # A summary
    mean_cross = np.mean([v for row in table_a.values() for v in row])
    print(f"\nA. Mean cross-class Spearman: {mean_cross:.3f}")
    print(
        f"   → {'High' if mean_cross > 0.5 else 'Moderate' if mean_cross > 0.3 else 'Low'} "
        f"agreement on which features are unstable across model classes."
    )

    # B summary
    mean_cc = np.mean([v for row in table_b.values() for v in row])
    print(f"\nB. Mean CC-flip Spearman: {mean_cc:.3f}")
    print(f"   → Coverage conflict {'IS' if mean_cc > 0.7 else 'partially'} universal as Level 3 predictor.")

    # C summary
    print(f"\nC. Level 2 predictor specificity:")
    for i, pred_name in enumerate(predictor_names):
        own_class_idx = i  # small_eig→Ridge, lasso_boundary→LASSO, xgb_var→XGB, rf_var→RF
        own = table_c[pred_name][own_class_idx]
        others = [table_c[pred_name][j] for j in range(4) if j != own_class_idx]
        mean_others = np.mean(others)
        print(
            f"   {pred_name}: own={own:.3f}, others_mean={mean_others:.3f} "
            f"({'SPECIFIC' if own > mean_others + 0.1 else 'TRANSFERS'})"
        )

    # D summary
    max_violation_all = max(v for row in table_d.values() for v in row)
    print(f"\nD. Max violation across all: {max_violation_all:.2e}")
    print(
        f"   → Var[SHAP] = DASH MSE identity {'HOLDS' if max_violation_all < 1e-6 else 'APPROXIMATELY HOLDS' if max_violation_all < 0.01 else 'VIOLATED'} for all classes."
    )

    print(f"\n{'=' * 77}")
    print("END OF REPORT")
    print(f"{'=' * 77}")


if __name__ == "__main__":
    main()
