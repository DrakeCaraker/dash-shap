#!/usr/bin/env python3
"""
Coverage conflict diagnostic for SHAP sign instability.

For each feature, checks whether both positive and negative SHAP signs
appear across the Rashomon set (coverage_conflict = binary indicator).
Compares this against actual flip_rate (continuous) and reports ROC-AUC
of coverage_conflict predicting flip_rate > 10%.

Self-contained: trains 50 XGBoost models on California Housing,
computes SHAP for 100 test observations.
"""

import time
import warnings

import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_MODELS = 50
N_OBS = 100
SEED = 42
TEST_SIZE = 0.2
FLIP_THRESHOLD = 0.10


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
            "colsample_bytree": rng.uniform(0.3, 0.8),
            "subsample": rng.uniform(0.6, 1.0),
            "random_state": s,
            "n_jobs": 1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, verbose=False)
        models.append(m)
    return models


def compute_shap_matrices(models, X_explain, X_background):
    """Return list of SHAP value matrices, each shape (n_obs, n_features)."""
    shap_matrices = []
    bg = X_background[:100]
    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)
        if (i + 1) % 10 == 0:
            print(f"  SHAP computed for {i + 1}/{len(models)} models")
    return shap_matrices


def compute_flip_rates_and_conflicts(shap_matrices):
    """
    Returns:
        flip_rates: (n_obs, n_features) - fraction of models in minority sign
        coverage_conflict: (n_obs, n_features) - binary, 1 if both +/- signs present
    """
    stack = np.stack(shap_matrices, axis=0)  # (n_models, n_obs, n_features)
    signs = np.sign(stack)
    n_models, n_obs, n_features = signs.shape

    flip_rates = np.zeros((n_obs, n_features))
    coverage_conflict = np.zeros((n_obs, n_features), dtype=int)

    for obs in range(n_obs):
        for feat in range(n_features):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                flip_rates[obs, feat] = 0.0
                coverage_conflict[obs, feat] = 0
                continue
            n_pos = np.sum(nonzero > 0)
            n_neg = np.sum(nonzero < 0)
            total = n_pos + n_neg
            flip_rates[obs, feat] = min(n_pos, n_neg) / total
            coverage_conflict[obs, feat] = int(n_pos > 0 and n_neg > 0)

    return flip_rates, coverage_conflict


def main():
    print("=" * 70)
    print("COVERAGE CONFLICT DIAGNOSTIC")
    print("Does sign coverage conflict predict SHAP flip rate?")
    print("=" * 70)
    print()

    # Load data
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"  Features: {feature_names}")
    print()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    rng = np.random.RandomState(SEED + 1)
    obs_idx = rng.choice(len(X_test), size=min(N_OBS, len(X_test)), replace=False)
    X_explain = X_test[obs_idx]
    print(f"Selected {len(obs_idx)} test observations")
    print()

    # Train Rashomon set
    print(f"Training {N_MODELS} XGBoost models...")
    t0 = time.time()
    models = train_rashomon_set(X_train, y_train, N_MODELS, SEED)
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # Compute SHAP
    print(f"Computing SHAP values ({N_MODELS} models x {len(X_explain)} obs)...")
    t0 = time.time()
    shap_matrices = compute_shap_matrices(models, X_explain, X_train)
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # Compute flip rates and coverage conflicts
    print("Computing flip rates and coverage conflicts...")
    flip_rates, coverage_conflict = compute_flip_rates_and_conflicts(shap_matrices)
    print()

    # Pooled analysis
    fr_flat = flip_rates.ravel()
    cc_flat = coverage_conflict.ravel()
    high_flip = (fr_flat > FLIP_THRESHOLD).astype(int)

    print("POOLED STATISTICS:")
    print(f"  Total feature-observation pairs: {len(fr_flat)}")
    print(f"  Coverage conflicts:              {cc_flat.sum()} ({cc_flat.mean():.1%})")
    print(f"  High flip rate (>{FLIP_THRESHOLD:.0%}):       {high_flip.sum()} ({high_flip.mean():.1%})")
    print()

    # Confusion-matrix-style breakdown
    tp = np.sum((cc_flat == 1) & (high_flip == 1))
    fp = np.sum((cc_flat == 1) & (high_flip == 0))
    tn = np.sum((cc_flat == 0) & (high_flip == 0))
    fn = np.sum((cc_flat == 0) & (high_flip == 1))

    print("CONTINGENCY TABLE (coverage_conflict vs flip_rate > 10%):")
    print(f"                     high_flip=0   high_flip=1")
    print(f"  conflict=0         {tn:10d}   {fn:10d}")
    print(f"  conflict=1         {fp:10d}   {tp:10d}")
    print()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print()

    # ROC-AUC
    if len(np.unique(high_flip)) < 2:
        print("WARNING: only one class in high_flip — ROC-AUC undefined")
        auc = float("nan")
    else:
        auc = roc_auc_score(high_flip, cc_flat)
    print(f"ROC-AUC (coverage_conflict predicting flip_rate > {FLIP_THRESHOLD:.0%}): {auc:.4f}")
    print()

    # Per-feature breakdown
    print("PER-FEATURE BREAKDOWN:")
    print("-" * 70)
    print(f"  {'Feature':15s}  {'mean_flip':>10s}  {'conflict_%':>10s}  {'high_flip_%':>12s}")
    print("-" * 70)
    for i, name in enumerate(feature_names):
        mf = flip_rates[:, i].mean()
        cc_pct = coverage_conflict[:, i].mean()
        hf_pct = (flip_rates[:, i] > FLIP_THRESHOLD).mean()
        print(f"  {name:15s}  {mf:10.4f}  {cc_pct:10.1%}  {hf_pct:12.1%}")
    print()

    # Per-feature AUC
    print("PER-FEATURE ROC-AUC (coverage_conflict -> flip > 10%):")
    print("-" * 55)
    for i, name in enumerate(feature_names):
        fr_f = flip_rates[:, i]
        cc_f = coverage_conflict[:, i]
        hf_f = (fr_f > FLIP_THRESHOLD).astype(int)
        if len(np.unique(hf_f)) < 2 or len(np.unique(cc_f)) < 2:
            print(f"  {name:15s}  AUC = N/A (single class)")
        else:
            a = roc_auc_score(hf_f, cc_f)
            print(f"  {name:15s}  AUC = {a:.4f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Pooled ROC-AUC:  {auc:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(
        f"  Interpretation:  Coverage conflict is {'a strong' if auc > 0.7 else 'a moderate' if auc > 0.5 else 'a weak'} predictor of SHAP instability"
    )
    print("=" * 70)

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


if __name__ == "__main__":
    results = main()
