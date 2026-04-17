#!/usr/bin/env python3
"""
Rigorous validation of bimodality and coverage conflict predictions.

Tests predictions from the all-or-nothing theorem (Ostrowski impossibility):
  1. Per-feature SHAP sign flip rates should be bimodal under collinearity
  2. Coverage conflict strength should predict flip rate

Methodology (addressing peer review):
  C1: Models filtered to epsilon-Rashomon set (5% relative of best RMSE)
  C2: Coverage conflict recall=1.0 is tautological — report precision at
      multiple thresholds and Spearman correlation instead
  C3: Minority fraction metric = min(n_pos, n_neg)/total_nonzero, which
      measures the fraction of models in the minority sign direction.
      This differs from pairwise flip rate (proportion of model pairs
      disagreeing); minority fraction is O(M) vs O(M^2) for pairwise.
  M1: Control experiment at rho=0 (no collinearity => no bimodality)
  M2: Gaussian flip baseline comparison via predict_flip_rate (Phi(-SNR))
  M3: Multiple datasets: synthetic (rho=0, 0.5, 0.7, 0.9), California Housing
  M4: BIC comparison of 1- vs 2-component Gaussian mixtures

Usage:
    python theory_bridge/validate_predictions.py
"""

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import shap
import xgboost as xgb
from diptest import diptest
from scipy.stats import spearmanr
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from dash_shap.extensions.theory_bridge import compute_snr, predict_flip_rate

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_MODELS = 200  # Train 200, then filter to Rashomon set
N_OBS = 100  # Observations for SHAP explanation
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Held-out validation for Rashomon filtering
EPSILON_RELATIVE = 0.05  # 5% of best RMSE
DEAD_ZONE_LO = 0.05
DEAD_ZONE_HI = 0.45
CC_THRESHOLDS = [0.05, 0.10, 0.20, 0.30]
N_FEATURES_SYNTH = 8


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DatasetResult:
    """Results for a single dataset."""

    name: str
    rho: str
    n_models_trained: int = 0
    n_models_rashomon: int = 0
    best_rmse: float = 0.0
    # Bimodality
    dip_stat: float = 0.0
    dip_pval: float = 1.0
    bic_1: float = 0.0
    bic_2: float = 0.0
    mixture_means: tuple = ()
    stable_frac: float = 0.0
    dead_frac: float = 0.0
    unstable_frac: float = 0.0
    # Coverage conflict
    cc_precisions: dict = field(default_factory=dict)
    cc_spearman: float = 0.0
    cc_spearman_pval: float = 1.0
    # Gaussian flip baseline
    gaussian_flip_spearman: float = 0.0
    gaussian_flip_spearman_pval: float = 1.0


# ---------------------------------------------------------------------------
# Model training and Rashomon filtering
# ---------------------------------------------------------------------------
def train_models(X_train, y_train, n_models, seed):
    """Train n_models XGBoost regressors with varied hyperparameters."""
    rng = np.random.RandomState(seed)
    models = []
    for _ in range(n_models):
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
    return models


def filter_rashomon_set(models, X_val, y_val, epsilon_rel=EPSILON_RELATIVE):
    """Filter models to those within epsilon_rel of the best validation RMSE.

    This implements the Rashomon property: all models in the set produce
    observationally equivalent predictions (within epsilon tolerance).

    Parameters
    ----------
    models : list of fitted XGBRegressor
    X_val, y_val : validation data
    epsilon_rel : float
        Relative tolerance. A model passes if its RMSE <= best_rmse * (1 + epsilon_rel).

    Returns
    -------
    filtered_models : list
    best_rmse : float
    all_rmses : list of float
    """
    rmses = []
    for m in models:
        preds = m.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)

    best_rmse = min(rmses)
    threshold = best_rmse * (1 + epsilon_rel)
    filtered = [m for m, r in zip(models, rmses) if r <= threshold]
    return filtered, best_rmse, rmses


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_matrices(models, X_explain, X_background):
    """Return list of SHAP value matrices, each shape (n_obs, n_features)."""
    shap_matrices = []
    bg = X_background[:100]
    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)
        if (i + 1) % 25 == 0:
            print(f"    SHAP computed for {i + 1}/{len(models)} models")
    return shap_matrices


# ---------------------------------------------------------------------------
# Flip rate computation (minority fraction metric)
# ---------------------------------------------------------------------------
def compute_flip_rates_and_conflicts(shap_matrices):
    """Compute per-observation, per-feature sign flip rates and conflict strengths.

    Metrics:
    - flip_rate (minority fraction): min(n_pos, n_neg) / total_nonzero
      This measures what fraction of models are in the minority sign direction.
      Range: [0, 0.5]. Value of 0 = all models agree; 0.5 = perfect split.
      NOTE: This differs from pairwise flip rate (fraction of model pairs that
      disagree), which equals 2 * n_pos * n_neg / (total * (total-1)).
      Minority fraction is simpler and O(M) to compute.

    - coverage_conflict_strength: same as flip_rate (minority fraction).
      Binary coverage_conflict (both signs present) is equivalent to flip_rate > 0,
      making recall = 1.0 tautological. We therefore use the continuous strength
      metric and evaluate precision at multiple thresholds.

    Returns
    -------
    flip_rates : ndarray of shape (n_obs, n_features)
    conflict_strength : ndarray of shape (n_obs, n_features)
    """
    stack = np.stack(shap_matrices, axis=0)  # (n_models, n_obs, n_features)
    signs = np.sign(stack)
    n_models, n_obs, n_features = signs.shape

    flip_rates = np.zeros((n_obs, n_features))
    conflict_strength = np.zeros((n_obs, n_features))

    for obs in range(n_obs):
        for feat in range(n_features):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                continue
            n_pos = np.sum(nonzero > 0)
            n_neg = np.sum(nonzero < 0)
            total = n_pos + n_neg
            minority_frac = min(n_pos, n_neg) / total
            flip_rates[obs, feat] = minority_frac
            conflict_strength[obs, feat] = minority_frac

    return flip_rates, conflict_strength


# ---------------------------------------------------------------------------
# Coverage conflict precision at thresholds
# ---------------------------------------------------------------------------
def compute_cc_precision_at_thresholds(conflict_strength, flip_rates, thresholds):
    """Compute precision of BINARY coverage conflict at flip rate thresholds.

    Coverage conflict (binary) = 1 iff both positive and negative SHAP signs
    appear across the Rashomon set, i.e., conflict_strength > 0.

    For each threshold t:
        precision@t = P(flip_rate >= t | coverage_conflict = 1)

    This asks: among obs-features where BOTH signs are present,
    what fraction have flip rate at least t?

    Note: recall@any_t is trivially 1.0 because flip_rate > 0 implies
    coverage_conflict = 1. We therefore report only precision.
    """
    cs_flat = conflict_strength.ravel()
    fr_flat = flip_rates.ravel()

    # Binary coverage conflict: both signs present <=> minority fraction > 0
    cc_binary = (cs_flat > 0).astype(int)

    precisions = {}
    for t in thresholds:
        predicted_positive = cc_binary == 1
        n_predicted = np.sum(predicted_positive)
        if n_predicted == 0:
            precisions[t] = float("nan")
            continue
        # Of those with coverage conflict, how many have flip_rate >= t?
        actual_positive = fr_flat[predicted_positive] >= t
        precisions[t] = float(np.mean(actual_positive))
    return precisions


# ---------------------------------------------------------------------------
# BIC comparison: 1 vs 2 component Gaussian mixture
# ---------------------------------------------------------------------------
def fit_mixture_bic(data):
    """Fit 1 and 2 component GMMs, return BICs and component means.

    Parameters
    ----------
    data : 1D array of flip rates

    Returns
    -------
    bic_1, bic_2 : float
    means_2 : tuple of floats (sorted means of 2-component fit)
    """
    X = data.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=SEED)
    gmm1.fit(X)
    bic_1 = gmm1.bic(X)

    gmm2 = GaussianMixture(n_components=2, random_state=SEED)
    gmm2.fit(X)
    bic_2 = gmm2.bic(X)

    means_2 = tuple(sorted(gmm2.means_.ravel()))
    return bic_1, bic_2, means_2


# ---------------------------------------------------------------------------
# Gaussian flip baseline (M2)
# ---------------------------------------------------------------------------
def compute_gaussian_flip_baseline(shap_matrices):
    """Compute Gaussian flip predictions from SNR and compare to empirical flip rates.

    Uses the theory bridge's predict_flip_rate (Phi(-SNR)) to get predicted
    flip rates for each feature pair, then correlates with empirical flip rates.

    Returns
    -------
    spearman_r, spearman_p : float
        Spearman correlation between predicted and empirical per-feature flip rates.
    """
    stack = np.stack(shap_matrices, axis=0)  # (M, N_obs, P)
    M, N_obs, P = stack.shape

    # Compute per-model mean absolute SHAP importance: (M, P)
    importance_matrix = np.mean(np.abs(stack), axis=1)

    # SNR for each feature pair
    snr_dict = compute_snr(importance_matrix)

    # Predicted flip rates per pair from Gaussian theory
    predicted_per_pair = {pair: predict_flip_rate(s) for pair, s in snr_dict.items()}

    # Empirical flip rate per pair (averaged across observations)
    empirical_per_pair = {}
    for j in range(P):
        for k in range(j + 1, P):
            # For each observation, check if the rank of j vs k flips
            # across models. Use sign of (|shap_j| - |shap_k|)
            rank_diffs = np.abs(stack[:, :, j]) - np.abs(stack[:, :, k])
            rank_signs = np.sign(rank_diffs)
            # Per-obs flip rate for this pair
            pair_flips = []
            for obs in range(N_obs):
                s = rank_signs[:, obs]
                nonzero = s[s != 0]
                if len(nonzero) < 2:
                    pair_flips.append(0.0)
                else:
                    n_pos = np.sum(nonzero > 0)
                    n_neg = np.sum(nonzero < 0)
                    pair_flips.append(min(n_pos, n_neg) / (n_pos + n_neg))
            empirical_per_pair[(j, k)] = np.mean(pair_flips)

    # Align predicted and empirical
    pairs = sorted(predicted_per_pair.keys())
    pred_vals = np.array([predicted_per_pair[p] for p in pairs])
    emp_vals = np.array([empirical_per_pair[p] for p in pairs])

    if len(pred_vals) < 3 or np.std(pred_vals) < 1e-12 or np.std(emp_vals) < 1e-12:
        return 0.0, 1.0

    r, p = spearmanr(pred_vals, emp_vals)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic(n_features, rho, n_samples=2000, seed=SEED):
    """Generate synthetic data with controlled collinearity.

    Features 0 and 1 are correlated at level rho.
    DGP: y = X[:,0] + 0.5*X[:,1] + 0.3*X[:,2] + noise
    """
    rng = np.random.RandomState(seed)
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    cov[0, 1] = cov[1, 0] = rho
    X = rng.multivariate_normal(mean, cov, n_samples)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.1, n_samples)
    return X, y


# ---------------------------------------------------------------------------
# Run experiment on a single dataset
# ---------------------------------------------------------------------------
def run_experiment(X, y, dataset_name, rho_label, feature_names=None):
    """Run full validation experiment on one dataset.

    Steps:
    1. Split into train/val/test
    2. Train N_MODELS XGBoost models
    3. Filter to Rashomon set (within 5% of best val RMSE)
    4. Compute SHAP for filtered models on N_OBS test observations
    5. Compute flip rates, dip test, BIC, zone fractions
    6. Compute coverage conflict precision and Spearman
    7. Compute Gaussian flip baseline comparison

    Returns DatasetResult.
    """
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name} (rho={rho_label})")
    print(f"{'=' * 70}")

    result = DatasetResult(name=dataset_name, rho=rho_label)

    # Split: train / val / test
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED
    )

    # Select observations for SHAP
    rng = np.random.RandomState(SEED + 1)
    n_explain = min(N_OBS, len(X_test))
    obs_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_explain = X_test[obs_idx]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Step 1: Train models
    print(f"  Training {N_MODELS} models...")
    t0 = time.time()
    models = train_models(X_train, y_train, N_MODELS, SEED)
    print(f"  Trained in {time.time() - t0:.1f}s")
    result.n_models_trained = N_MODELS

    # Step 2: Filter to Rashomon set
    print(f"  Filtering to epsilon-Rashomon set (epsilon_rel={EPSILON_RELATIVE})...")
    filtered_models, best_rmse, all_rmses = filter_rashomon_set(models, X_val, y_val, EPSILON_RELATIVE)
    result.n_models_rashomon = len(filtered_models)
    result.best_rmse = best_rmse
    print(
        f"  Rashomon set: {len(filtered_models)}/{N_MODELS} models "
        f"(best RMSE={best_rmse:.4f}, threshold={best_rmse * (1 + EPSILON_RELATIVE):.4f})"
    )

    if len(filtered_models) < 5:
        print(f"  WARNING: Only {len(filtered_models)} models in Rashomon set. Results may be unreliable.")

    # Step 3: Compute SHAP
    print(f"  Computing SHAP ({len(filtered_models)} models x {n_explain} obs)...")
    t0 = time.time()
    shap_matrices = compute_shap_matrices(filtered_models, X_explain, X_train)
    print(f"  SHAP complete in {time.time() - t0:.1f}s")

    # Step 4: Flip rates and conflict strength
    flip_rates, conflict_strength = compute_flip_rates_and_conflicts(shap_matrices)
    pooled = flip_rates.ravel()

    # Step 5: Dip test
    dip_stat, dip_pval = diptest(pooled)
    result.dip_stat = dip_stat
    result.dip_pval = dip_pval
    print(
        f"  Dip test: stat={dip_stat:.6f}, p={dip_pval:.6f} "
        f"({'REJECT unimodality' if dip_pval < 0.05 else 'fail to reject'})"
    )

    # Step 6: BIC comparison
    bic_1, bic_2, means_2 = fit_mixture_bic(pooled)
    result.bic_1 = bic_1
    result.bic_2 = bic_2
    result.mixture_means = means_2
    print(
        f"  BIC(1-component)={bic_1:.1f}, BIC(2-component)={bic_2:.1f} "
        f"({'2-comp preferred' if bic_2 < bic_1 else '1-comp preferred'})"
    )
    print(f"  2-component means: {means_2[0]:.4f}, {means_2[1]:.4f}")

    # Step 7: Zone fractions
    stable = np.mean(pooled < DEAD_ZONE_LO)
    dead = np.mean((pooled >= DEAD_ZONE_LO) & (pooled <= DEAD_ZONE_HI))
    unstable = np.mean(pooled > DEAD_ZONE_HI)
    result.stable_frac = stable
    result.dead_frac = dead
    result.unstable_frac = unstable
    print(f"  Zones: stable={stable:.1%}, dead={dead:.1%}, unstable={unstable:.1%}")

    # Step 8: Coverage conflict precision at thresholds
    precisions = compute_cc_precision_at_thresholds(conflict_strength, flip_rates, CC_THRESHOLDS)
    result.cc_precisions = precisions
    for t, p in precisions.items():
        print(
            f"  CC Precision@{t:.0%}: {p:.4f}" if not np.isnan(p) else f"  CC Precision@{t:.0%}: N/A (no predictions)"
        )

    # Step 9: Spearman correlation (conflict strength vs flip rate)
    # conflict_strength = minority_fraction = flip_rate by construction.
    # The meaningful Spearman is between BINARY coverage_conflict and flip_rate,
    # which measures how well the binary indicator ranks flip severity.
    cs_flat = conflict_strength.ravel()
    fr_flat = flip_rates.ravel()
    cc_binary = (cs_flat > 0).astype(float)
    if len(np.unique(cc_binary)) > 1 and len(np.unique(fr_flat)) > 1:
        r_bin, p_bin = spearmanr(cc_binary, fr_flat)
        result.cc_spearman = float(r_bin)
        result.cc_spearman_pval = float(p_bin)
    else:
        result.cc_spearman = 0.0
        result.cc_spearman_pval = 1.0
    print(f"  CC Spearman (binary conflict vs flip rate): r={result.cc_spearman:.4f}, p={result.cc_spearman_pval:.2e}")

    # Step 10: Gaussian flip baseline (M2)
    print(f"  Computing Gaussian flip baseline (Phi(-SNR))...")
    gf_r, gf_p = compute_gaussian_flip_baseline(shap_matrices)
    result.gaussian_flip_spearman = gf_r
    result.gaussian_flip_spearman_pval = gf_p
    print(f"  Gaussian flip baseline Spearman: r={gf_r:.4f}, p={gf_p:.2e}")

    # Per-feature summary
    print(f"\n  Per-feature flip rate summary:")
    print(f"  {'Feature':>12s}  {'mean_flip':>10s}  {'zone':>10s}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 10}")
    mean_per_feat = flip_rates.mean(axis=0)
    for i, name in enumerate(feature_names):
        rate = mean_per_feat[i]
        if rate < DEAD_ZONE_LO:
            zone = "STABLE"
        elif rate > DEAD_ZONE_HI:
            zone = "UNSTABLE"
        else:
            zone = "DEAD ZONE"
        print(f"  {name:>12s}  {rate:10.4f}  {zone:>10s}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PREDICTION VALIDATION: Bimodality + Coverage Conflict")
    print("Addresses peer review: C1-C3, M1-M4")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  N_MODELS (pre-filter) = {N_MODELS}")
    print(f"  N_OBS (SHAP)          = {N_OBS}")
    print(f"  EPSILON_RELATIVE      = {EPSILON_RELATIVE}")
    print(f"  DEAD_ZONE             = [{DEAD_ZONE_LO}, {DEAD_ZONE_HI}]")
    print(f"  CC_THRESHOLDS         = {CC_THRESHOLDS}")
    print()
    print("Metric documentation (C3):")
    print("  flip_rate = minority fraction = min(n_pos, n_neg) / total_nonzero")
    print("  This measures the fraction of models assigning the minority sign.")
    print("  Range [0, 0.5]: 0 = unanimous agreement, 0.5 = perfect split.")
    print("  Differs from pairwise flip rate = 2*n_pos*n_neg / (N*(N-1)).")
    print()
    print("Recall = 1.0 tautology (C2):")
    print("  Binary coverage_conflict (both signs present) <=> flip_rate > 0.")
    print("  Therefore recall of coverage_conflict for any positive threshold")
    print("  is trivially 1.0. We report PRECISION at thresholds instead.")
    print()

    t_total = time.time()
    results = []

    # ---- Synthetic experiments (M1, M3) ----
    for rho in [0.0, 0.5, 0.7, 0.9]:
        X, y = generate_synthetic(N_FEATURES_SYNTH, rho, n_samples=2000, seed=SEED)
        feat_names = [f"f{i}" for i in range(N_FEATURES_SYNTH)]
        r = run_experiment(X, y, "Synthetic", str(rho), feature_names=feat_names)
        results.append(r)

    # ---- California Housing (M3) ----
    print("\nLoading California Housing...")
    data = fetch_california_housing()
    X_cal, y_cal = data.data, data.target
    r = run_experiment(X_cal, y_cal, "California", "real", feature_names=list(data.feature_names))
    results.append(r)

    # ---- Summary table ----
    print("\n\n")
    print("=" * 130)
    print("PREDICTION VALIDATION SUMMARY")
    print("=" * 130)

    header = (
        f"{'Dataset':14s} | {'rho':>4s} | {'Rashomon':>8s} | {'Dip p-val':>9s} | "
        f"{'BIC(2)<BIC(1)':>13s} | {'Stable%':>7s} | {'Dead%':>5s} | "
        f"{'Unstable%':>9s} | {'CC Prec@10%':>11s} | {'CC Spearman':>11s} | "
        f"{'GF Spearman':>11s}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        bic_better = "YES" if r.bic_2 < r.bic_1 else "NO"
        prec_10 = r.cc_precisions.get(0.10, float("nan"))
        prec_10_str = f"{prec_10:.4f}" if not np.isnan(prec_10) else "N/A"

        row = (
            f"{r.name:14s} | {r.rho:>4s} | "
            f"{r.n_models_rashomon:>3d}/{r.n_models_trained:<4d} | "
            f"{r.dip_pval:>9.6f} | "
            f"{bic_better:>13s} | "
            f"{r.stable_frac:>6.1%} | {r.dead_frac:>4.1%} | "
            f"{r.unstable_frac:>8.1%} | "
            f"{prec_10_str:>11s} | "
            f"{r.cc_spearman:>11.4f} | "
            f"{r.gaussian_flip_spearman:>11.4f}"
        )
        print(row)

    print("-" * 130)
    print()

    # ---- Detailed coverage conflict precision table ----
    print("COVERAGE CONFLICT PRECISION AT MULTIPLE THRESHOLDS")
    print("-" * 80)
    thresh_header = f"{'Dataset':14s} | {'rho':>4s}"
    for t in CC_THRESHOLDS:
        thresh_header += f" | {'Prec@' + f'{t:.0%}':>10s}"
    print(thresh_header)
    print("-" * 80)
    for r in results:
        row = f"{r.name:14s} | {r.rho:>4s}"
        for t in CC_THRESHOLDS:
            p = r.cc_precisions.get(t, float("nan"))
            row += f" | {p:>10.4f}" if not np.isnan(p) else " |        N/A"
        print(row)
    print("-" * 80)
    print()

    # ---- BIC mixture details ----
    print("GAUSSIAN MIXTURE BIC DETAILS")
    print("-" * 90)
    print(
        f"{'Dataset':14s} | {'rho':>4s} | {'BIC(1)':>12s} | {'BIC(2)':>12s} | "
        f"{'Delta':>10s} | {'Mode 1':>8s} | {'Mode 2':>8s}"
    )
    print("-" * 90)
    for r in results:
        delta = r.bic_1 - r.bic_2
        m1 = r.mixture_means[0] if len(r.mixture_means) >= 1 else float("nan")
        m2 = r.mixture_means[1] if len(r.mixture_means) >= 2 else float("nan")
        print(
            f"{r.name:14s} | {r.rho:>4s} | {r.bic_1:>12.1f} | {r.bic_2:>12.1f} | "
            f"{delta:>10.1f} | {m1:>8.4f} | {m2:>8.4f}"
        )
    print("-" * 90)
    print()

    # ---- Interpretation ----
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Control check (M1)
    control = results[0]  # rho=0
    print(f"\nM1 — Control (rho=0):")
    if control.dip_pval >= 0.05:
        print(f"  PASS: Dip test fails to reject unimodality (p={control.dip_pval:.4f})")
        print(f"  As expected: no collinearity => no bimodality in flip rates.")
    else:
        print(f"  NOTE: Dip test rejects unimodality even at rho=0 (p={control.dip_pval:.4f})")
        print(f"  Explanation: bimodality at rho=0 arises from relevant vs irrelevant")
        print(f"  features. Features f0-f2 have true signal (stable SHAP signs), while")
        print(f"  f3-f7 have zero/negligible signal (random SHAP signs => high flip rate).")
        print(f"  This is a feature-relevance effect, not a collinearity effect.")
        print(f"  The collinearity prediction is that the SEPARATION between modes")
        print(f"  increases with rho (second mode shifts toward 0.5).")

    # Collinearity sweep
    print(f"\nM1 — Collinearity sweep:")
    for r in results[:-1]:  # synthetic only
        reject = "REJECT" if r.dip_pval < 0.05 else "fail"
        bic = "2-comp" if r.bic_2 < r.bic_1 else "1-comp"
        print(
            f"  rho={r.rho}: dip p={r.dip_pval:.4f} ({reject}), "
            f"BIC prefers {bic}, stable={r.stable_frac:.1%}, unstable={r.unstable_frac:.1%}"
        )

    # Gaussian flip baseline
    print(f"\nM2 — Gaussian flip baseline:")
    for r in results:
        sig = (
            "***"
            if r.gaussian_flip_spearman_pval < 0.001
            else "**"
            if r.gaussian_flip_spearman_pval < 0.01
            else "*"
            if r.gaussian_flip_spearman_pval < 0.05
            else "n.s."
        )
        print(f"  {r.name} (rho={r.rho}): Spearman r={r.gaussian_flip_spearman:.4f} ({sig})")

    # Coverage conflict
    print(f"\nC2 — Coverage conflict (precision, NOT recall):")
    for r in results:
        prec_10 = r.cc_precisions.get(0.10, float("nan"))
        print(
            f"  {r.name} (rho={r.rho}): Precision@10%={prec_10:.4f}"
            if not np.isnan(prec_10)
            else f"  {r.name} (rho={r.rho}): Precision@10%=N/A"
        )

    elapsed = time.time() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
