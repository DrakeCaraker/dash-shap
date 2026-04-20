#!/usr/bin/env python3
"""
Information-Theoretic Predictor Comparison for SHAP Instability

Computes 7 data-only predictors on all (P choose 2) feature pairs and correlates
each with 3 model-based instability outcomes (rho_SHAP, VarRatio, FlipRate).

Predictors:
  A1. |rho_feature| — raw absolute Pearson correlation
  A2. Partial R^2 — max conditional predictive power (Gaussian)
  A3. Co-Information (Gaussian form)
  A4. VIF — max VIF of pair
  B5. Spearman Partial R^2 — rank-based
  C6. kNN Co-Information — nonparametric MI
  D7. XGB Conditional — model-light nonparametric

Outcomes:
  rho_SHAP — cross-model SHAP correlation (from eta_shap_correlation.py)
  VarRatio — Var[sum] / (Var[A] + Var[B])
  FlipRate — fraction of models where feature rank flips

Datasets:
  1. Synthetic g=2 (rho=0.9)
  2. Synthetic g=3 (rho=0.9)
  3. California Housing
  4. Breast Cancer
"""

import numpy as np
import time
import warnings
import sys
from itertools import combinations

import xgboost as xgb
import shap
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr, rankdata
from numpy.linalg import inv

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ===========================================================================
# Configuration
# ===========================================================================
M = 200  # models per SHAP stack
N_OBS = 100  # observations for SHAP explanation
SEED = 42


# ===========================================================================
# SHAP stack computation (reused from eta_shap_correlation.py)
# ===========================================================================


def train_and_explain(X_train, y_train, X_explain, M=200, seed=42):
    """Train M diverse XGBoost models, compute interventional SHAP values."""
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
            print(f"      SHAP models: {i + 1}/{M}")

    return np.stack(shap_matrices, axis=0)  # (M, N_obs, P)


# ===========================================================================
# Outcome computations
# ===========================================================================


def compute_rho_shap(shap_stack, i, j):
    """Mean cross-model SHAP correlation for pair (i, j)."""
    M, N_obs, P = shap_stack.shape
    corrs = []
    for obs in range(N_obs):
        si = shap_stack[:, obs, i]
        sj = shap_stack[:, obs, j]
        if np.std(si) > 1e-10 and np.std(sj) > 1e-10:
            r = np.corrcoef(si, sj)[0, 1]
            if not np.isnan(r):
                corrs.append(r)
    return np.mean(corrs) if corrs else 0.0


def compute_var_ratio(shap_stack, i, j):
    """Var[SHAP_i + SHAP_j] / (Var[SHAP_i] + Var[SHAP_j]) across models."""
    var_sum = np.var(shap_stack[:, :, i] + shap_stack[:, :, j], axis=0).mean()
    var_i = np.var(shap_stack[:, :, i], axis=0).mean()
    var_j = np.var(shap_stack[:, :, j], axis=0).mean()
    denom = var_i + var_j
    return var_sum / denom if denom > 1e-20 else 1.0


def compute_flip_rate(shap_stack, i, j):
    """Fraction of model pairs where importance rank of i vs j flips."""
    # Per-model mean |SHAP| importance
    imp_i = np.mean(np.abs(shap_stack[:, :, i]), axis=1)  # (M,)
    imp_j = np.mean(np.abs(shap_stack[:, :, j]), axis=1)  # (M,)
    # i is more important in some models, j in others
    i_wins = np.sum(imp_i > imp_j)
    j_wins = np.sum(imp_j > imp_i)
    # Flip rate = min(wins) / total (0 = always same order, 0.5 = maximal instability)
    total = i_wins + j_wins
    if total == 0:
        return 0.0
    return min(i_wins, j_wins) / total


# ===========================================================================
# Predictor computations
# ===========================================================================


def compute_all_predictors(X, y, feature_pairs):
    """
    Compute all 7 predictors for the given feature pairs.
    Returns dict: predictor_name -> {(i,j): value}
    """
    P = X.shape[1]
    n = X.shape[0]

    # Train/test split for XGB
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # --- A1: Raw absolute feature correlation ---
    print("    Computing |rho_feature|...")
    corr_matrix = np.corrcoef(X.T)
    raw_corr = {(i, j): abs(corr_matrix[i, j]) for (i, j) in feature_pairs}

    # --- A2 & A3: Partial R^2 and Co-Information (Gaussian) ---
    print("    Computing Partial R^2 and Co-I (Gaussian)...")
    R2_i = {}
    for feat in range(P):
        lr = LinearRegression()
        lr.fit(X[:, [feat]], y)
        R2_i[feat] = max(lr.score(X[:, [feat]], y), 0.0)

    R2_ij = {}
    for i, j in feature_pairs:
        lr = LinearRegression()
        lr.fit(X[:, [i, j]], y)
        R2_ij[(i, j)] = max(lr.score(X[:, [i, j]], y), 0.0)

    partial_R2 = {}
    co_info = {}
    for i, j in feature_pairs:
        # Partial R^2: how much does i add beyond j, and vice versa
        pr_i_given_j = (R2_ij[(i, j)] - R2_i[j]) / max(1 - R2_i[j], 1e-10)
        pr_j_given_i = (R2_ij[(i, j)] - R2_i[i]) / max(1 - R2_i[i], 1e-10)
        partial_R2[(i, j)] = max(pr_i_given_j, pr_j_given_i)

        # Co-Information (Gaussian): -0.5 * log((1-R2_i)(1-R2_j) / (1-R2_ij))
        num = (1 - R2_i[i]) * (1 - R2_i[j])
        den = max(1 - R2_ij[(i, j)], 1e-10)
        co_info[(i, j)] = -0.5 * np.log(max(num / den, 1e-10))

    # --- A4: VIF ---
    print("    Computing VIF...")
    try:
        # Regularize slightly to avoid singular matrix
        corr_reg = corr_matrix + 1e-6 * np.eye(P)
        vif_diag = np.diag(inv(corr_reg))
    except Exception:
        vif_diag = np.ones(P)
    vif_pair = {(i, j): max(vif_diag[i], vif_diag[j]) for (i, j) in feature_pairs}

    # --- B5: Spearman Partial R^2 ---
    print("    Computing Spearman Partial R^2...")
    X_ranked = np.apply_along_axis(rankdata, 0, X)
    y_ranked = rankdata(y)

    R2_rank_i = {}
    for feat in range(P):
        lr = LinearRegression()
        lr.fit(X_ranked[:, [feat]], y_ranked)
        R2_rank_i[feat] = max(lr.score(X_ranked[:, [feat]], y_ranked), 0.0)

    R2_rank_ij = {}
    for i, j in feature_pairs:
        lr = LinearRegression()
        lr.fit(X_ranked[:, [i, j]], y_ranked)
        R2_rank_ij[(i, j)] = max(lr.score(X_ranked[:, [i, j]], y_ranked), 0.0)

    spearman_partial_R2 = {}
    for i, j in feature_pairs:
        pr_i = (R2_rank_ij[(i, j)] - R2_rank_i[j]) / max(1 - R2_rank_i[j], 1e-10)
        pr_j = (R2_rank_ij[(i, j)] - R2_rank_i[i]) / max(1 - R2_rank_i[i], 1e-10)
        spearman_partial_R2[(i, j)] = max(pr_i, pr_j)

    # --- C6: kNN Co-Information ---
    print("    Computing kNN Co-Information...")
    mi_each = {}
    for feat in range(P):
        mi_each[feat] = mutual_info_regression(X[:, [feat]], y, random_state=SEED)[0]

    knn_co_info = {}
    for i, j in feature_pairs:
        mi_ij = mutual_info_regression(X[:, [i, j]], y, random_state=SEED)[0]
        knn_co_info[(i, j)] = mi_each[i] + mi_each[j] - mi_ij

    # --- D7: XGB Conditional ---
    print("    Computing XGB Conditional...")
    xgb_params = {"n_estimators": 100, "max_depth": 4, "random_state": SEED, "n_jobs": 1}

    R2_xgb_i = {}
    for feat in range(P):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr[:, [feat]], y_tr, verbose=False)
        R2_xgb_i[feat] = m.score(X_te[:, [feat]], y_te)

    xgb_conditional = {}
    for i, j in feature_pairs:
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_tr[:, [i, j]], y_tr, verbose=False)
        R2_xgb_ij = m.score(X_te[:, [i, j]], y_te)

        cond_i = (R2_xgb_ij - R2_xgb_i[j]) / max(1 - R2_xgb_i[j], 1e-10)
        cond_j = (R2_xgb_ij - R2_xgb_i[i]) / max(1 - R2_xgb_i[i], 1e-10)
        xgb_conditional[(i, j)] = max(cond_i, cond_j)

    return {
        "|rho_feature|": raw_corr,
        "Partial R^2": partial_R2,
        "Co-I (Gauss)": co_info,
        "VIF": vif_pair,
        "Spearman pR^2": spearman_partial_R2,
        "kNN Co-I": knn_co_info,
        "XGB conditional": xgb_conditional,
    }


# ===========================================================================
# Dataset generators
# ===========================================================================


def make_synthetic(g, rho, p=8, n=1000, seed=42):
    """Synthetic dataset with g correlated features and p-g independent."""
    rng = np.random.RandomState(seed)
    n_total = n + N_OBS

    z = rng.randn(n_total, 1)
    corr_block = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.randn(n_total, g)
    indep_block = rng.randn(n_total, p - g)

    X = np.hstack([corr_block, indep_block])
    y = X.sum(axis=1) + 0.1 * rng.randn(n_total)

    X_train, X_explain = X[:n], X[n : n + N_OBS]
    y_train = y[:n]

    return X_train, y_train, X_explain


# ===========================================================================
# Main
# ===========================================================================


def run_dataset(name, X_train, y_train, X_explain, P):
    """Run full analysis for one dataset. Returns (predictors, outcomes, pairs)."""
    print(f"\n  [{name}] Training SHAP stack (M={M})...")
    shap_stack = train_and_explain(X_train, y_train, X_explain, M=M, seed=SEED)

    # All (P choose 2) pairs
    pairs = list(combinations(range(P), 2))
    n_pairs = len(pairs)
    print(f"  [{name}] {n_pairs} feature pairs")

    # Outcomes
    print(f"  [{name}] Computing outcomes...")
    rho_shap = np.array([compute_rho_shap(shap_stack, i, j) for (i, j) in pairs])
    var_ratio = np.array([compute_var_ratio(shap_stack, i, j) for (i, j) in pairs])
    flip_rate = np.array([compute_flip_rate(shap_stack, i, j) for (i, j) in pairs])

    # Predictors
    print(f"  [{name}] Computing predictors...")
    predictors = compute_all_predictors(X_train, y_train, pairs)

    # Convert to arrays aligned with pairs
    predictor_arrays = {}
    for pred_name, pred_dict in predictors.items():
        predictor_arrays[pred_name] = np.array([pred_dict[(i, j)] for (i, j) in pairs])

    outcomes = {
        "rho_SHAP": rho_shap,
        "VarRatio": var_ratio,
        "FlipRate": flip_rate,
    }

    return predictor_arrays, outcomes, pairs


def main():
    t0 = time.time()

    print("=" * 78)
    print("  INFORMATION-THEORETIC PREDICTOR COMPARISON FOR SHAP INSTABILITY")
    print(f"  M={M}, N_obs={N_OBS}, seed={SEED}")
    print("=" * 78)

    datasets = {}

    # --- Dataset 1: Synthetic g=2 ---
    print("\n\n[1/4] Synthetic g=2 (rho=0.9, 8 features)")
    X_train, y_train, X_explain = make_synthetic(g=2, rho=0.9, p=8, seed=SEED)
    datasets["syn_g2"] = run_dataset("syn_g2", X_train, y_train, X_explain, P=8)

    # --- Dataset 2: Synthetic g=3 ---
    print("\n\n[2/4] Synthetic g=3 (rho=0.9, 8 features)")
    X_train, y_train, X_explain = make_synthetic(g=3, rho=0.9, p=8, seed=SEED + 1)
    datasets["syn_g3"] = run_dataset("syn_g3", X_train, y_train, X_explain, P=8)

    # --- Dataset 3: California Housing ---
    print("\n\n[3/4] California Housing (8 features)")
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    X_explain = X_test[:N_OBS]
    datasets["calif"] = run_dataset("calif", X_train, y_train, X_explain, P=8)

    # --- Dataset 4: Breast Cancer ---
    print("\n\n[4/4] Breast Cancer (30 features)")
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)
    X_explain = X_test[:N_OBS]
    datasets["breast"] = run_dataset("breast", X_train, y_train, X_explain, P=30)

    # ===================================================================
    # Compute Spearman correlations: predictor vs outcome, per dataset
    # ===================================================================
    predictor_names = [
        "|rho_feature|",
        "Partial R^2",
        "Co-I (Gauss)",
        "VIF",
        "Spearman pR^2",
        "kNN Co-I",
        "XGB conditional",
    ]
    outcome_names = ["rho_SHAP", "VarRatio", "FlipRate"]
    dataset_names = ["syn_g2", "syn_g3", "calif", "breast"]

    # results_matrix[pred][outcome][dataset] = (spearman_rho, p_value)
    results_matrix = {}
    for pred in predictor_names:
        results_matrix[pred] = {}
        for outcome in outcome_names:
            results_matrix[pred][outcome] = {}
            for ds in dataset_names:
                pred_arr, outcomes, _ = datasets[ds]
                x = pred_arr[pred]
                y_out = outcomes[outcome]
                # Handle constant arrays
                if np.std(x) < 1e-10 or np.std(y_out) < 1e-10:
                    results_matrix[pred][outcome][ds] = (0.0, 1.0)
                else:
                    rho, p = spearmanr(x, y_out)
                    results_matrix[pred][outcome][ds] = (rho, p)

    # ===================================================================
    # Print results
    # ===================================================================
    elapsed = time.time() - t0

    print("\n\n")
    print("=" * 100)
    print("  INFORMATION-THEORETIC PREDICTOR COMPARISON — RESULTS")
    print("=" * 100)

    for outcome in outcome_names:
        print(f"\n  Spearman correlation with {outcome}:")
        print(f"  {'Predictor':<18}", end="")
        for ds in dataset_names:
            print(f"  {ds:>8}", end="")
        print(f"  {'|avg|':>8}")
        print(f"  {'-' * 18}", end="")
        for _ in dataset_names:
            print(f"  {'-' * 8}", end="")
        print(f"  {'-' * 8}")

        for pred in predictor_names:
            print(f"  {pred:<18}", end="")
            vals = []
            for ds in dataset_names:
                rho, p = results_matrix[pred][outcome][ds]
                vals.append(rho)
                marker = "*" if p < 0.05 else " "
                print(f"  {rho:>+7.3f}{marker}", end="")
            avg_abs = np.mean(np.abs(vals))
            print(f"  {avg_abs:>7.3f}")

    # ===================================================================
    # Overall ranking
    # ===================================================================
    print(f"\n\n  OVERALL RANKING (average |Spearman| across all outcomes and datasets):")
    print(f"  {'Predictor':<18} {'avg |Spearman|':>14} {'avg vs rho_SHAP':>16}")
    print(f"  {'-' * 18} {'-' * 14} {'-' * 16}")

    overall_scores = {}
    rho_shap_scores = {}
    for pred in predictor_names:
        all_abs = []
        rho_abs = []
        for outcome in outcome_names:
            for ds in dataset_names:
                rho, _ = results_matrix[pred][outcome][ds]
                all_abs.append(abs(rho))
            # rho_SHAP specific
        for ds in dataset_names:
            rho, _ = results_matrix[pred]["rho_SHAP"][ds]
            rho_abs.append(abs(rho))
        overall_scores[pred] = np.mean(all_abs)
        rho_shap_scores[pred] = np.mean(rho_abs)

    ranked = sorted(overall_scores.items(), key=lambda x: -x[1])
    for i, (pred, score) in enumerate(ranked):
        marker = " <-- WINNER" if i == 0 else (" <-- RUNNER-UP" if i == 1 else "")
        print(f"  {pred:<18} {score:>14.4f} {rho_shap_scores[pred]:>16.4f}{marker}")

    # ===================================================================
    # Summary
    # ===================================================================
    winner = ranked[0][0]
    runner_up = ranked[1][0]

    # Check |Spearman| > 0.7 threshold for rho_SHAP
    print(f"\n\n  THRESHOLD CHECK: Any predictor with |Spearman| > 0.7 vs rho_SHAP?")
    for pred in predictor_names:
        for ds in dataset_names:
            rho, p = results_matrix[pred]["rho_SHAP"][ds]
            if abs(rho) > 0.7:
                print(f"    YES: {pred} on {ds}: Spearman = {rho:+.4f} (p = {p:.2e})")

    # Check if any predictor achieves > 0.7 on average
    any_above = any(v > 0.7 for v in rho_shap_scores.values())
    if not any_above:
        print(f"    No predictor achieves avg |Spearman| > 0.7 with rho_SHAP across all datasets.")

    print(f"\n\n  {'=' * 60}")
    print(f"  WINNER:     {winner} (avg |Spearman| = {overall_scores[winner]:.4f})")
    print(f"  RUNNER-UP:  {runner_up} (avg |Spearman| = {overall_scores[runner_up]:.4f})")
    print(f"  COMPUTATION TIME: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  {'=' * 60}")

    # Full detail for the winner
    print(f"\n  Winner detail ({winner}):")
    for outcome in outcome_names:
        for ds in dataset_names:
            rho, p = results_matrix[winner][outcome][ds]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {outcome} x {ds}: rho={rho:+.4f}, p={p:.2e} {sig}")


if __name__ == "__main__":
    main()
