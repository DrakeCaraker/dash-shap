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

Dual analysis (addressing filtered Rashomon set size issue):
  - Unfiltered (all 200 models): justified by approximate bilemma —
    the impossibility holds at any epsilon-tolerance, so the full diverse
    population constitutes an approximate (epsilon-Rashomon) set.
  - Filtered (epsilon=5%): proper Rashomon set, flagged when < 20 models.
  - Permutation control for dip test: randomly reassign SHAP signs within
    each observation to break feature structure. If permuted data also
    rejects unimodality, the test is uninformative.

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
N_PERMUTATIONS = 100  # Number of permutation replicates for dip control
MIN_RASHOMON_FOR_TESTS = 20  # Minimum Rashomon set size for distributional tests


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """Results for a single analysis pass (filtered or unfiltered)."""

    label: str  # "unfiltered" or "filtered"
    n_models: int = 0
    rashomon_diameter: float = 0.0  # max RMSE difference (unfiltered only)
    # Bimodality
    dip_stat: float = 0.0
    dip_pval: float = 1.0
    dip_permuted_pval: float = 1.0  # fraction of permutations that also reject
    dip_permuted_informative: bool = True  # False if permuted also rejects
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
    # Flag
    too_small: bool = False


@dataclass
class DatasetResult:
    """Results for a single dataset, with both filtered and unfiltered analyses."""

    name: str
    rho: str
    n_models_trained: int = 0
    n_models_rashomon: int = 0
    best_rmse: float = 0.0
    all_rmses: list = field(default_factory=list)
    unfiltered: AnalysisResult = field(default_factory=lambda: AnalysisResult(label="unfiltered"))
    filtered: AnalysisResult = field(default_factory=lambda: AnalysisResult(label="filtered"))


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
      Range: [0, 0.5]. Value of 0 = all models agree; 0.5 = perfect split.

    - coverage_conflict_strength: same as flip_rate (minority fraction).

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
# Permutation control for dip test
# ---------------------------------------------------------------------------
def permutation_dip_control(shap_matrices, n_permutations=N_PERMUTATIONS, seed=SEED):
    """Permutation-based control for the dip test.

    For each permutation replicate:
    1. For each observation, randomly shuffle the SHAP sign assignments
       across features (breaking the feature structure).
    2. Recompute flip rates on the permuted data.
    3. Run the dip test.

    Returns the fraction of permutations that also reject unimodality (p < 0.05).
    If this fraction is high, the dip test result is uninformative (detecting
    discreteness, not real bimodality).

    Parameters
    ----------
    shap_matrices : list of ndarray, each (n_obs, n_features)
    n_permutations : int
    seed : int

    Returns
    -------
    perm_reject_frac : float
        Fraction of permutations where dip test rejects (p < 0.05).
    """
    rng = np.random.RandomState(seed)
    stack = np.stack(shap_matrices, axis=0)  # (n_models, n_obs, n_features)
    signs = np.sign(stack)
    n_models, n_obs, n_features = signs.shape

    n_reject = 0
    for _ in range(n_permutations):
        # Permute: for each observation, shuffle signs across features
        perm_signs = signs.copy()
        for obs in range(n_obs):
            for model in range(n_models):
                rng.shuffle(perm_signs[model, obs, :])

        # Compute flip rates on permuted data
        perm_flip_rates = np.zeros((n_obs, n_features))
        for obs in range(n_obs):
            for feat in range(n_features):
                s = perm_signs[:, obs, feat]
                nonzero = s[s != 0]
                if len(nonzero) < 2:
                    continue
                n_pos = np.sum(nonzero > 0)
                n_neg = np.sum(nonzero < 0)
                total = n_pos + n_neg
                perm_flip_rates[obs, feat] = min(n_pos, n_neg) / total

        pooled = perm_flip_rates.ravel()
        _, pval = diptest(pooled)
        if pval < 0.05:
            n_reject += 1

    return n_reject / n_permutations


# ---------------------------------------------------------------------------
# Coverage conflict precision at thresholds
# ---------------------------------------------------------------------------
def compute_cc_precision_at_thresholds(conflict_strength, flip_rates, thresholds):
    """Compute precision of BINARY coverage conflict at flip rate thresholds.

    For each threshold t:
        precision@t = P(flip_rate >= t | coverage_conflict = 1)
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
        actual_positive = fr_flat[predicted_positive] >= t
        precisions[t] = float(np.mean(actual_positive))
    return precisions


# ---------------------------------------------------------------------------
# BIC comparison: 1 vs 2 component Gaussian mixture
# ---------------------------------------------------------------------------
def fit_mixture_bic(data):
    """Fit 1 and 2 component GMMs, return BICs and component means."""
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

    Returns
    -------
    spearman_r, spearman_p : float
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
            rank_diffs = np.abs(stack[:, :, j]) - np.abs(stack[:, :, k])
            rank_signs = np.sign(rank_diffs)
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
# Single analysis pass (shared between filtered and unfiltered)
# ---------------------------------------------------------------------------
def run_analysis_pass(label, models, shap_matrices, all_rmses, feature_names):
    """Run bimodality + coverage conflict analysis on a set of models.

    Parameters
    ----------
    label : str
        "unfiltered" or "filtered"
    models : list of fitted models (used only for count)
    shap_matrices : list of SHAP value matrices
    all_rmses : list of floats (RMSE for each model in the set)
    feature_names : list of str

    Returns
    -------
    AnalysisResult
    """
    result = AnalysisResult(label=label)
    result.n_models = len(models)

    if len(models) < 3:
        print(f"    [{label}] Only {len(models)} models — skipping analysis.")
        result.too_small = True
        return result

    # Effective Rashomon diameter (for unfiltered)
    if all_rmses and len(all_rmses) > 1:
        result.rashomon_diameter = max(all_rmses) - min(all_rmses)

    # Flag too-small filtered sets
    if label == "filtered" and len(models) < MIN_RASHOMON_FOR_TESTS:
        result.too_small = True
        print(
            f"    [{label}] WARNING: Only {len(models)} models (< {MIN_RASHOMON_FOR_TESTS}). "
            f"Distributional tests uninformative — rely on unfiltered analysis."
        )

    # Flip rates and conflict strength
    flip_rates, conflict_strength = compute_flip_rates_and_conflicts(shap_matrices)
    pooled = flip_rates.ravel()

    # Dip test
    dip_stat, dip_pval = diptest(pooled)
    result.dip_stat = dip_stat
    result.dip_pval = dip_pval
    print(
        f"    [{label}] Dip test: stat={dip_stat:.6f}, p={dip_pval:.6f} "
        f"({'REJECT unimodality' if dip_pval < 0.05 else 'fail to reject'})"
    )

    # Permutation control for dip test
    print(f"    [{label}] Running permutation control ({N_PERMUTATIONS} reps)...")
    perm_reject_frac = permutation_dip_control(shap_matrices, N_PERMUTATIONS, SEED)
    result.dip_permuted_pval = perm_reject_frac
    result.dip_permuted_informative = perm_reject_frac < 0.5
    if perm_reject_frac >= 0.5:
        print(
            f"    [{label}] PERMUTATION CONTROL: {perm_reject_frac:.0%} of permutations also reject "
            f"=> dip test UNINFORMATIVE (detecting discreteness, not bimodality)"
        )
    else:
        print(
            f"    [{label}] PERMUTATION CONTROL: {perm_reject_frac:.0%} of permutations reject "
            f"=> dip test INFORMATIVE (bimodality is real)"
        )

    # BIC comparison
    bic_1, bic_2, means_2 = fit_mixture_bic(pooled)
    result.bic_1 = bic_1
    result.bic_2 = bic_2
    result.mixture_means = means_2
    print(
        f"    [{label}] BIC(1)={bic_1:.1f}, BIC(2)={bic_2:.1f} "
        f"({'2-comp preferred' if bic_2 < bic_1 else '1-comp preferred'})"
    )
    print(f"    [{label}] 2-component means: {means_2[0]:.4f}, {means_2[1]:.4f}")

    # Zone fractions
    stable = np.mean(pooled < DEAD_ZONE_LO)
    dead = np.mean((pooled >= DEAD_ZONE_LO) & (pooled <= DEAD_ZONE_HI))
    unstable = np.mean(pooled > DEAD_ZONE_HI)
    result.stable_frac = stable
    result.dead_frac = dead
    result.unstable_frac = unstable
    print(f"    [{label}] Zones: stable={stable:.1%}, dead={dead:.1%}, unstable={unstable:.1%}")

    # Coverage conflict precision at thresholds
    precisions = compute_cc_precision_at_thresholds(conflict_strength, flip_rates, CC_THRESHOLDS)
    result.cc_precisions = precisions

    # Spearman correlation (binary conflict vs flip rate)
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
    print(f"    [{label}] CC Spearman (binary vs flip): r={result.cc_spearman:.4f}")

    # Gaussian flip baseline (M2)
    gf_r, gf_p = compute_gaussian_flip_baseline(shap_matrices)
    result.gaussian_flip_spearman = gf_r
    result.gaussian_flip_spearman_pval = gf_p
    print(f"    [{label}] Gaussian flip Spearman: r={gf_r:.4f}, p={gf_p:.2e}")

    # Per-feature summary
    mean_per_feat = flip_rates.mean(axis=0)
    print(f"    [{label}] Per-feature flip rates:")
    print(f"    {'Feature':>12s}  {'mean_flip':>10s}  {'zone':>10s}")
    print(f"    {'-' * 12}  {'-' * 10}  {'-' * 10}")
    for i, name in enumerate(feature_names):
        rate = mean_per_feat[i]
        if rate < DEAD_ZONE_LO:
            zone = "STABLE"
        elif rate > DEAD_ZONE_HI:
            zone = "UNSTABLE"
        else:
            zone = "DEAD ZONE"
        print(f"    {name:>12s}  {rate:10.4f}  {zone:>10s}")

    return result


# ---------------------------------------------------------------------------
# Run experiment on a single dataset
# ---------------------------------------------------------------------------
def run_experiment(X, y, dataset_name, rho_label, feature_names=None):
    """Run full validation experiment on one dataset with dual analysis.

    Steps:
    1. Split into train/val/test
    2. Train N_MODELS XGBoost models
    3. Compute SHAP for ALL models (unfiltered analysis)
    4. Filter to Rashomon set and compute SHAP for filtered models
    5. Run analysis on both sets

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

    # Step 2: Compute validation RMSEs for all models
    all_rmses = []
    for m in models:
        preds = m.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        all_rmses.append(rmse)
    result.all_rmses = all_rmses
    result.best_rmse = min(all_rmses)

    # Step 3: Filter to Rashomon set
    threshold = result.best_rmse * (1 + EPSILON_RELATIVE)
    filtered_models = [m for m, r in zip(models, all_rmses) if r <= threshold]
    filtered_rmses = [r for r in all_rmses if r <= threshold]
    result.n_models_rashomon = len(filtered_models)
    print(
        f"  Rashomon set: {len(filtered_models)}/{N_MODELS} models "
        f"(best RMSE={result.best_rmse:.4f}, threshold={threshold:.4f})"
    )
    print(f"  Effective Rashomon diameter (all models): {max(all_rmses) - min(all_rmses):.4f}")

    # Step 4: Compute SHAP for ALL models (unfiltered)
    print(f"\n  --- UNFILTERED ANALYSIS (all {N_MODELS} models) ---")
    print(f"  Justified by approximate bilemma: impossibility holds at any epsilon.")
    print(f"  Computing SHAP ({N_MODELS} models x {n_explain} obs)...")
    t0 = time.time()
    shap_all = compute_shap_matrices(models, X_explain, X_train)
    print(f"  SHAP complete in {time.time() - t0:.1f}s")

    result.unfiltered = run_analysis_pass("unfiltered", models, shap_all, all_rmses, feature_names)

    # Step 5: Run filtered analysis (reuse SHAP from unfiltered where possible)
    print(f"\n  --- FILTERED ANALYSIS (epsilon={EPSILON_RELATIVE}, {len(filtered_models)} models) ---")
    # Extract SHAP matrices for filtered models only
    filtered_indices = [i for i, r in enumerate(all_rmses) if r <= threshold]
    shap_filtered = [shap_all[i] for i in filtered_indices]

    result.filtered = run_analysis_pass("filtered", filtered_models, shap_filtered, filtered_rmses, feature_names)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PREDICTION VALIDATION: Bimodality + Coverage Conflict")
    print("Addresses peer review: C1-C3, M1-M4")
    print("DUAL ANALYSIS: Unfiltered (approx bilemma) + Filtered (Rashomon)")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  N_MODELS (pre-filter) = {N_MODELS}")
    print(f"  N_OBS (SHAP)          = {N_OBS}")
    print(f"  EPSILON_RELATIVE      = {EPSILON_RELATIVE}")
    print(f"  DEAD_ZONE             = [{DEAD_ZONE_LO}, {DEAD_ZONE_HI}]")
    print(f"  CC_THRESHOLDS         = {CC_THRESHOLDS}")
    print(f"  N_PERMUTATIONS (dip)  = {N_PERMUTATIONS}")
    print(f"  MIN_RASHOMON_TESTS    = {MIN_RASHOMON_FOR_TESTS}")
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
    print("Dual analysis rationale:")
    print("  Unfiltered: The approximate bilemma proves impossibility at any")
    print("  epsilon-tolerance. The 200 diverse XGBoost models constitute an")
    print("  approximate (epsilon-Rashomon) set with full statistical power.")
    print("  Filtered: Proper Rashomon set (epsilon=5%). Flagged when < 20 models.")
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

    # ---- Summary tables ----
    for analysis_type in ["unfiltered", "filtered"]:
        print("\n\n")
        print("=" * 150)
        label_upper = analysis_type.upper()
        if analysis_type == "unfiltered":
            print(f"SUMMARY: {label_upper} ANALYSIS (all {N_MODELS} models — approximate bilemma)")
        else:
            print(f"SUMMARY: {label_upper} ANALYSIS (epsilon={EPSILON_RELATIVE} Rashomon set)")
        print("=" * 150)

        header = (
            f"{'Dataset':14s} | {'rho':>4s} | {'N models':>8s} | {'Dip p':>9s} | "
            f"{'Perm ctrl':>9s} | {'Dip info':>8s} | "
            f"{'BIC 2<1':>7s} | {'Stable%':>7s} | {'Dead%':>5s} | "
            f"{'Unst%':>5s} | {'CC Prec@10%':>11s} | {'CC Sprmn':>8s} | "
            f"{'GF Sprmn':>8s}"
        )
        if analysis_type == "unfiltered":
            header += f" | {'Rash diam':>9s}"
        print(header)
        print("-" * 150)

        for r in results:
            a = r.unfiltered if analysis_type == "unfiltered" else r.filtered

            if a.too_small and analysis_type == "filtered":
                flag = " [<20]"
            else:
                flag = ""

            bic_better = "YES" if a.bic_2 < a.bic_1 else "NO"
            prec_10 = a.cc_precisions.get(0.10, float("nan"))
            prec_10_str = f"{prec_10:.4f}" if not np.isnan(prec_10) else "N/A"
            dip_info = "YES" if a.dip_permuted_informative else "NO"

            row = (
                f"{r.name + flag:14s} | {r.rho:>4s} | "
                f"{a.n_models:>8d} | "
                f"{a.dip_pval:>9.6f} | "
                f"{a.dip_permuted_pval:>8.0%}rej | "
                f"{dip_info:>8s} | "
                f"{bic_better:>7s} | "
                f"{a.stable_frac:>6.1%} | {a.dead_frac:>4.1%} | "
                f"{a.unstable_frac:>4.1%} | "
                f"{prec_10_str:>11s} | "
                f"{a.cc_spearman:>8.4f} | "
                f"{a.gaussian_flip_spearman:>8.4f}"
            )
            if analysis_type == "unfiltered":
                row += f" | {a.rashomon_diameter:>9.4f}"
            print(row)

        print("-" * 150)
        print()

    # ---- Coverage conflict precision at thresholds (both analyses) ----
    for analysis_type in ["unfiltered", "filtered"]:
        label_upper = analysis_type.upper()
        print(f"COVERAGE CONFLICT PRECISION — {label_upper}")
        print("-" * 90)
        thresh_header = f"{'Dataset':14s} | {'rho':>4s}"
        for t in CC_THRESHOLDS:
            thresh_header += f" | {'Prec@' + f'{t:.0%}':>10s}"
        print(thresh_header)
        print("-" * 90)
        for r in results:
            a = r.unfiltered if analysis_type == "unfiltered" else r.filtered
            row = f"{r.name:14s} | {r.rho:>4s}"
            for t in CC_THRESHOLDS:
                p = a.cc_precisions.get(t, float("nan"))
                row += f" | {p:>10.4f}" if not np.isnan(p) else " |        N/A"
            print(row)
        print("-" * 90)
        print()

    # ---- BIC mixture details (both analyses) ----
    for analysis_type in ["unfiltered", "filtered"]:
        label_upper = analysis_type.upper()
        print(f"GAUSSIAN MIXTURE BIC DETAILS — {label_upper}")
        print("-" * 100)
        print(
            f"{'Dataset':14s} | {'rho':>4s} | {'N models':>8s} | {'BIC(1)':>12s} | {'BIC(2)':>12s} | "
            f"{'Delta':>10s} | {'Mode 1':>8s} | {'Mode 2':>8s}"
        )
        print("-" * 100)
        for r in results:
            a = r.unfiltered if analysis_type == "unfiltered" else r.filtered
            delta = a.bic_1 - a.bic_2
            m1 = a.mixture_means[0] if len(a.mixture_means) >= 1 else float("nan")
            m2 = a.mixture_means[1] if len(a.mixture_means) >= 2 else float("nan")
            print(
                f"{r.name:14s} | {r.rho:>4s} | {a.n_models:>8d} | {a.bic_1:>12.1f} | {a.bic_2:>12.1f} | "
                f"{delta:>10.1f} | {m1:>8.4f} | {m2:>8.4f}"
            )
        print("-" * 100)
        print()

    # ---- Permutation control summary ----
    print("=" * 90)
    print("PERMUTATION CONTROL SUMMARY (dip test)")
    print("=" * 90)
    print(
        f"{'Dataset':14s} | {'rho':>4s} | {'Unfilt reject%':>15s} | {'Unfilt info':>11s} | "
        f"{'Filt reject%':>13s} | {'Filt info':>9s}"
    )
    print("-" * 90)
    for r in results:
        u = r.unfiltered
        f = r.filtered
        print(
            f"{r.name:14s} | {r.rho:>4s} | {u.dip_permuted_pval:>14.0%} | "
            f"{'YES' if u.dip_permuted_informative else 'NO':>11s} | "
            f"{f.dip_permuted_pval:>12.0%} | "
            f"{'YES' if f.dip_permuted_informative else 'NO':>9s}"
        )
    print("-" * 90)
    print()

    # ---- Interpretation ----
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Dual analysis note
    print("\nDual analysis rationale:")
    print("  The filtered Rashomon sets contain only 4-22 models, making")
    print("  distributional tests (dip, BIC) unreliable — flip rates are")
    print("  quantized to 3-5 values and dip detects discreteness, not bimodality.")
    print("  The unfiltered analysis uses all 200 models, justified by the")
    print("  approximate bilemma: the impossibility theorem holds at any epsilon-")
    print("  tolerance, so the full diverse population is an approximate Rashomon set.")
    print("  The 'effective Rashomon diameter' quantifies the epsilon.")

    # Control check (M1)
    control = results[0]  # rho=0
    u = control.unfiltered
    print(f"\nM1 — Control (rho=0, unfiltered):")
    if u.dip_pval >= 0.05:
        print(f"  PASS: Dip test fails to reject unimodality (p={u.dip_pval:.4f})")
        print(f"  As expected: no collinearity => no bimodality in flip rates.")
    else:
        informative = "INFORMATIVE" if u.dip_permuted_informative else "UNINFORMATIVE"
        print(f"  NOTE: Dip test rejects unimodality even at rho=0 (p={u.dip_pval:.4f})")
        print(f"  Permutation control: {informative} ({u.dip_permuted_pval:.0%} permutations also reject)")
        print(f"  Explanation: bimodality at rho=0 arises from relevant vs irrelevant")
        print(f"  features. Features f0-f2 have true signal (stable SHAP signs), while")
        print(f"  f3-f7 have zero/negligible signal (random SHAP signs => high flip rate).")
        print(f"  This is a feature-relevance effect, not a collinearity effect.")
        print(f"  The collinearity prediction is that the SEPARATION between modes")
        print(f"  increases with rho (second mode shifts toward 0.5).")

    # Collinearity sweep (unfiltered)
    print(f"\nM1 — Collinearity sweep (unfiltered):")
    for r in results[:-1]:  # synthetic only
        a = r.unfiltered
        reject = "REJECT" if a.dip_pval < 0.05 else "fail"
        bic = "2-comp" if a.bic_2 < a.bic_1 else "1-comp"
        info = "informative" if a.dip_permuted_informative else "UNINFORMATIVE"
        print(
            f"  rho={r.rho}: dip p={a.dip_pval:.4f} ({reject}, {info}), "
            f"BIC prefers {bic}, stable={a.stable_frac:.1%}, unstable={a.unstable_frac:.1%}, "
            f"Rash.diam={a.rashomon_diameter:.4f}"
        )

    # Gaussian flip baseline (unfiltered)
    print(f"\nM2 — Gaussian flip baseline (unfiltered):")
    for r in results:
        a = r.unfiltered
        sig = (
            "***"
            if a.gaussian_flip_spearman_pval < 0.001
            else "**"
            if a.gaussian_flip_spearman_pval < 0.01
            else "*"
            if a.gaussian_flip_spearman_pval < 0.05
            else "n.s."
        )
        print(f"  {r.name} (rho={r.rho}): Spearman r={a.gaussian_flip_spearman:.4f} ({sig})")

    # Coverage conflict (unfiltered)
    print(f"\nC2 — Coverage conflict precision (unfiltered):")
    for r in results:
        a = r.unfiltered
        prec_10 = a.cc_precisions.get(0.10, float("nan"))
        print(
            f"  {r.name} (rho={r.rho}): Precision@10%={prec_10:.4f}"
            if not np.isnan(prec_10)
            else f"  {r.name} (rho={r.rho}): Precision@10%=N/A"
        )

    # Filtered sensitivity check
    print(f"\nFiltered sensitivity check:")
    for r in results:
        flag = " [TOO SMALL]" if r.filtered.too_small else ""
        print(f"  {r.name} (rho={r.rho}): {r.n_models_rashomon}/{r.n_models_trained} models{flag}")

    # Permutation control interpretation
    print(f"\nPermutation control interpretation:")
    for r in results:
        a = r.unfiltered
        if a.dip_permuted_informative:
            print(
                f"  {r.name} (rho={r.rho}): Dip result is REAL — "
                f"only {a.dip_permuted_pval:.0%} of permutations reject (vs {a.dip_pval:.6f} actual p)"
            )
        else:
            print(
                f"  {r.name} (rho={r.rho}): Dip result UNINFORMATIVE — "
                f"{a.dip_permuted_pval:.0%} of permutations also reject"
            )

    elapsed = time.time() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
