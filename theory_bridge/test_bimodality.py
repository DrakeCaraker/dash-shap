#!/usr/bin/env python3
"""
Bimodality test for SHAP sign flip rates across a Rashomon set.

Tests the prediction from the all-or-nothing theorem (Ostrowski impossibility):
per-feature SHAP sign flip rates should be bimodal — concentrated near 0%
(stable features) and >=50% (Rashomon-caught features) — with a dead zone
between ~5-45%.

Self-contained: trains 50 XGBoost models on California Housing,
computes SHAP for 100 test observations, then runs Hartigan's dip test.
"""

import time
import warnings

import numpy as np
import shap
import xgboost as xgb
from diptest import diptest
from sklearn.datasets import fetch_california_housing
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
DEAD_ZONE_LO = 0.05
DEAD_ZONE_HI = 0.45


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
    bg = X_background[:100]  # small background for speed
    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)
        if (i + 1) % 10 == 0:
            print(f"  SHAP computed for {i + 1}/{len(models)} models")
    return shap_matrices


def compute_flip_rates(shap_matrices):
    """
    For each observation and feature, compute the fraction of model pairs
    that disagree on the sign of SHAP value.

    Returns array of shape (n_obs, n_features) with flip rates in [0, 1].
    """
    # Stack: (n_models, n_obs, n_features)
    stack = np.stack(shap_matrices, axis=0)
    signs = np.sign(stack)  # -1, 0, +1

    n_obs, n_features = stack.shape[1], stack.shape[2]
    flip_rates = np.zeros((n_obs, n_features))

    for obs in range(n_obs):
        for feat in range(n_features):
            s = signs[:, obs, feat]
            # Count pairs with disagreeing signs (ignoring zeros)
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                flip_rates[obs, feat] = 0.0
                continue
            n_pos = np.sum(nonzero > 0)
            n_neg = np.sum(nonzero < 0)
            total_pairs = n_pos + n_neg
            # Flip rate = fraction that are in the minority sign
            flip_rates[obs, feat] = min(n_pos, n_neg) / total_pairs
    return flip_rates


def main():
    print("=" * 70)
    print("BIMODALITY TEST: SHAP Sign Flip Rates Across Rashomon Set")
    print("Prediction: all-or-nothing theorem => bimodal distribution")
    print("=" * 70)
    print()

    # Load data
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"  Features: {feature_names}")
    print(f"  Shape: {X.shape}")
    print()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Select observations for SHAP
    rng = np.random.RandomState(SEED + 1)
    obs_idx = rng.choice(len(X_test), size=min(N_OBS, len(X_test)), replace=False)
    X_explain = X_test[obs_idx]
    print(f"Selected {len(obs_idx)} test observations for SHAP explanation")
    print()

    # Train Rashomon set
    print(f"Training {N_MODELS} XGBoost models with varied hyperparameters...")
    t0 = time.time()
    models = train_rashomon_set(X_train, y_train, N_MODELS, SEED)
    t_train = time.time() - t0
    print(f"  Training complete in {t_train:.1f}s")
    print()

    # Compute SHAP values
    print(f"Computing SHAP values ({N_MODELS} models x {len(X_explain)} obs)...")
    t0 = time.time()
    shap_matrices = compute_shap_matrices(models, X_explain, X_train)
    t_shap = time.time() - t0
    print(f"  SHAP complete in {t_shap:.1f}s")
    print()

    # Compute flip rates
    print("Computing per-feature sign flip rates...")
    flip_rates = compute_flip_rates(shap_matrices)
    pooled = flip_rates.ravel()
    print(f"  Pooled distribution: {len(pooled)} values (obs x features)")
    print(f"  Mean flip rate: {pooled.mean():.4f}")
    print(f"  Median flip rate: {np.median(pooled):.4f}")
    print(f"  Std flip rate: {pooled.std():.4f}")
    print()

    # Zone analysis
    below_lo = np.mean(pooled < DEAD_ZONE_LO)
    above_hi = np.mean(pooled > DEAD_ZONE_HI)
    in_dead_zone = np.mean((pooled >= DEAD_ZONE_LO) & (pooled <= DEAD_ZONE_HI))
    print("ZONE ANALYSIS:")
    print(f"  Stable zone    (< {DEAD_ZONE_LO:.0%}):       {below_lo:.4f} ({below_lo:.1%})")
    print(f"  Dead zone      ({DEAD_ZONE_LO:.0%}-{DEAD_ZONE_HI:.0%}):    {in_dead_zone:.4f} ({in_dead_zone:.1%})")
    print(f"  Unstable zone  (> {DEAD_ZONE_HI:.0%}):      {above_hi:.4f} ({above_hi:.1%})")
    print(f"  Bimodal mass   (stable+unstable): {below_lo + above_hi:.4f} ({below_lo + above_hi:.1%})")
    print()

    # Hartigan's dip test
    print("HARTIGAN'S DIP TEST:")
    dip_stat, p_value = diptest(pooled)
    print(f"  Dip statistic: {dip_stat:.6f}")
    print(f"  p-value:       {p_value:.6f}")
    if p_value < 0.05:
        print(f"  Result: REJECT unimodality (p < 0.05) => bimodal distribution supported")
    else:
        print(f"  Result: FAIL to reject unimodality (p >= 0.05)")
    print()

    # Per-feature summary
    print("PER-FEATURE FLIP RATE SUMMARY (averaged across observations):")
    print("-" * 55)
    mean_per_feature = flip_rates.mean(axis=0)
    for i, name in enumerate(feature_names):
        rate = mean_per_feature[i]
        if rate < DEAD_ZONE_LO:
            zone = "STABLE"
        elif rate > DEAD_ZONE_HI:
            zone = "UNSTABLE"
        else:
            zone = "DEAD ZONE"
        print(f"  {name:15s}  flip_rate={rate:.4f}  [{zone}]")
    print()

    # Histogram summary (text-based)
    print("FLIP RATE HISTOGRAM (pooled):")
    bins = np.linspace(0, 0.5, 11)
    counts, _ = np.histogram(pooled, bins=bins)
    for i in range(len(counts)):
        lo, hi = bins[i], bins[i + 1]
        bar = "#" * (counts[i] * 40 // max(counts.max(), 1))
        print(f"  [{lo:.2f}-{hi:.2f}]  {counts[i]:5d}  {bar}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Dip statistic:         {dip_stat:.6f}")
    print(f"  Dip p-value:           {p_value:.6f}")
    print(f"  Bimodality rejected:   {'YES' if p_value < 0.05 else 'NO'} (alpha=0.05)")
    print(f"  Stable zone fraction:  {below_lo:.1%}")
    print(f"  Dead zone fraction:    {in_dead_zone:.1%}")
    print(f"  Unstable zone fraction:{above_hi:.1%}")
    print("=" * 70)

    return {
        "dip_stat": dip_stat,
        "p_value": p_value,
        "below_lo": below_lo,
        "above_hi": above_hi,
        "in_dead_zone": in_dead_zone,
        "mean_flip_rates": mean_per_feature.tolist(),
        "feature_names": list(feature_names),
    }


if __name__ == "__main__":
    results = main()
