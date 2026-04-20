#!/usr/bin/env python3
"""
Eta law validation: does dim(V^G)/dim(V) predict per-feature instability?

For each dataset:
1. Compute feature correlation matrix
2. Identify correlation groups at threshold |rho| > c
3. Compute eta = 1/g_i for each feature in group of size g_i (eta=1 for ungrouped)
4. Compute predicted instability = 1 - eta per feature
5. Compute observed instability = mean flip rate per feature (from 200 XGBoost models)
6. Correlation between predicted and observed = the eta law's R^2
"""

import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                               load_breast_cancer, load_iris)
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import fcluster, linkage
import warnings, time

warnings.filterwarnings("ignore")

M = 200
N_OBS = 100
SEED = 42
THRESHOLDS = [0.5, 0.7, 0.9]  # test eta at multiple correlation thresholds


def identify_groups(X, threshold):
    """
    Identify feature correlation groups at given threshold.
    Returns: group_id per feature (0-indexed), group_sizes dict
    """
    from scipy.spatial.distance import squareform

    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)
    # Use hierarchical clustering on 1 - |correlation| as distance
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    # condensed distance matrix
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='complete')
    # Cut at distance = 1 - threshold (features with |corr| > threshold are in same cluster)
    labels = fcluster(Z, t=1 - threshold, criterion='distance')
    return labels


def compute_eta_per_feature(group_labels):
    """
    For each feature, compute eta = 1/g where g is its group size.
    Ungrouped features (group size 1) have eta = 1.
    """
    unique, counts = np.unique(group_labels, return_counts=True)
    group_size = dict(zip(unique, counts))
    eta = np.array([1.0 / group_size[g] for g in group_labels])
    return eta


def compute_observed_instability(X_train, y_train, X_explain, M, seed):
    """
    Train M models, compute SHAP, return per-feature mean flip rate.
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
            print(f"    Trained and explained {i + 1}/{M} models")

    stack = np.stack(shap_matrices, axis=0)  # (M, N_obs, P)
    signs = np.sign(stack)

    # Per-feature flip rate (averaged across observations)
    n_obs, n_features = stack.shape[1], stack.shape[2]
    flip_rates = np.zeros(n_features)
    for feat in range(n_features):
        feat_flips = []
        for obs in range(n_obs):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                feat_flips.append(0.0)
                continue
            n_pos = (nonzero > 0).sum()
            n_neg = (nonzero < 0).sum()
            feat_flips.append(min(n_pos, n_neg) / (n_pos + n_neg))
        flip_rates[feat] = np.mean(feat_flips)

    return flip_rates


def generate_synthetic(rho, n=2000, p=8, seed=42):
    """Generate synthetic data with features 0,1 correlated at rho."""
    rng = np.random.RandomState(seed)
    cov = np.eye(p)
    cov[0, 1] = cov[1, 0] = rho
    X = rng.multivariate_normal(np.zeros(p), cov, n)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.5, n)
    return X, y


def run_eta_validation(X, y, dataset_name, feature_names=None):
    """Run full eta law validation on one dataset."""
    P = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(P)]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    rng = np.random.RandomState(SEED + 1)
    obs_idx = rng.choice(len(X_test), size=min(N_OBS, len(X_test)), replace=False)
    X_explain = X_test[obs_idx]

    # Compute observed instability
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name} (N={len(X)}, P={P})")
    print(f"{'='*60}")
    print(f"Training {M} models and computing SHAP...")
    t0 = time.time()
    observed = compute_observed_instability(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Test eta law at multiple thresholds
    results = []
    for threshold in THRESHOLDS:
        groups = identify_groups(X_train, threshold)
        eta = compute_eta_per_feature(groups)
        predicted = 1 - eta  # predicted instability

        # Correlation between predicted and observed
        if len(np.unique(predicted)) > 1:
            r_spearman, p_spearman = spearmanr(predicted, observed)
            r_pearson, p_pearson = pearsonr(predicted, observed)
            r2 = r_pearson ** 2
        else:
            r_spearman, p_spearman = float('nan'), 1.0
            r_pearson, p_pearson = float('nan'), 1.0
            r2 = float('nan')

        n_groups = len(set(groups))
        max_group = max(np.bincount(groups)[1:]) if max(groups) > 0 else 1

        print(f"\n  Threshold |rho| > {threshold}:")
        print(f"    Groups: {n_groups} (largest: {max_group})")
        print(f"    Spearman(predicted, observed): r={r_spearman:.3f}, p={p_spearman:.4f}")
        print(f"    Pearson R^2: {r2:.3f}")

        # Per-feature detail
        print(f"    {'Feature':>15s}  {'Group':>5s}  {'g':>3s}  {'eta':>5s}  {'Pred':>6s}  {'Obs':>6s}")
        for i in range(P):
            g = np.sum(groups == groups[i])
            print(f"    {feature_names[i]:>15s}  {groups[i]:>5d}  {g:>3d}  {eta[i]:>5.2f}  {predicted[i]:>6.3f}  {observed[i]:>6.3f}")

        results.append({
            "threshold": threshold,
            "n_groups": n_groups,
            "max_group_size": int(max_group),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "pearson_r2": float(r2),
        })

    return results, observed


# ==================== MAIN ====================
if __name__ == "__main__":
    datasets = [
        ("california_housing", *fetch_california_housing(return_X_y=True),
         list(fetch_california_housing().feature_names)),
        ("diabetes", *load_diabetes(return_X_y=True),
         list(load_diabetes().feature_names)),
        ("breast_cancer", *load_breast_cancer(return_X_y=True),
         list(load_breast_cancer().feature_names)),
        ("synthetic_rho09", *generate_synthetic(0.9), None),
        ("synthetic_rho0", *generate_synthetic(0.0), None),
    ]

    all_results = {}
    for name, X, y, fnames in datasets:
        results, observed = run_eta_validation(X, y, name, fnames)
        all_results[name] = results

    # Summary table
    print(f"\n\n{'='*80}")
    print("ETA LAW VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':>20s}  {'Threshold':>9s}  {'Groups':>6s}  {'MaxG':>4s}  {'Spearman':>8s}  {'p-value':>8s}  {'R^2':>5s}")
    print("-" * 80)
    for name, results in all_results.items():
        for r in results:
            print(f"{name:>20s}  {r['threshold']:>9.1f}  {r['n_groups']:>6d}  {r['max_group_size']:>4d}  {r['spearman_r']:>8.3f}  {r['spearman_p']:>8.4f}  {r['pearson_r2']:>5.3f}")
