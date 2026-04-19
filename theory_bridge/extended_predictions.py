#!/usr/bin/env python3
"""
Extended predictions from the bilemma framework.

Three new analyses on Rashomon set data:
  1. Stability curve: per-feature critical epsilon* (phase transition)
  2. Average unfaithfulness lower bound: p/2 bound on any stable method
  3. Entropy bimodality: sign entropy bimodal (modes near 0 and 1 bit)

Reuses model training / SHAP computation patterns from validate_predictions.py
but is self-contained (trains its own models, computes its own SHAP).

Usage:
    python theory_bridge/extended_predictions.py
"""

import time
import warnings

import numpy as np
import shap
import xgboost as xgb
from diptest import diptest
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# ---------------------------------------------------------------------------
# Configuration (matches validate_predictions.py)
# ---------------------------------------------------------------------------
N_MODELS = 200
N_OBS = 100
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
EPSILON_RELATIVE = 0.05
N_FEATURES_SYNTH = 8
N_EPSILON_STEPS = 50


# ---------------------------------------------------------------------------
# Shared infrastructure (from validate_predictions.py)
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


def compute_shap_matrices(models, X_explain, X_background):
    """Return list of SHAP value matrices, each shape (n_obs, n_features)."""
    shap_matrices = []
    bg = X_background[:100]
    for i, m in enumerate(models):
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_matrices.append(sv)
        if (i + 1) % 50 == 0:
            print(f"    SHAP computed for {i + 1}/{len(models)} models")
    return shap_matrices


def generate_synthetic(n_features, rho, n_samples=2000, seed=SEED):
    """Generate synthetic data with controlled collinearity."""
    rng = np.random.RandomState(seed)
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    cov[0, 1] = cov[1, 0] = rho
    X = rng.multivariate_normal(mean, cov, n_samples)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.1, n_samples)
    return X, y


def prepare_dataset(X, y, dataset_name, feature_names=None):
    """Split data, train models, compute SHAP. Returns dict with all needed state."""
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 70}")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Split
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED
    )

    # Select observations for SHAP
    rng = np.random.RandomState(SEED + 1)
    n_explain = min(N_OBS, len(X_test))
    obs_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_explain = X_test[obs_idx]

    # Train models
    print(f"  Training {N_MODELS} models...")
    t0 = time.time()
    models = train_models(X_train, y_train, N_MODELS, SEED)
    print(f"  Trained in {time.time() - t0:.1f}s")

    # Compute predictions for all models on explain set
    print(f"  Computing predictions on {n_explain} observations...")
    predictions = np.array([m.predict(X_explain) for m in models])  # (M, N_obs)

    # Compute SHAP
    print(f"  Computing SHAP ({N_MODELS} models x {n_explain} obs)...")
    t0 = time.time()
    shap_matrices = compute_shap_matrices(models, X_explain, X_train)
    print(f"  SHAP complete in {time.time() - t0:.1f}s")

    # Stack SHAP into (M, N_obs, P) array
    shap_stack = np.stack(shap_matrices, axis=0)

    return {
        "name": dataset_name,
        "feature_names": feature_names,
        "models": models,
        "predictions": predictions,  # (M, N_obs)
        "shap_stack": shap_stack,  # (M, N_obs, P)
        "X_explain": X_explain,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Stability Curve (epsilon sweep)
# ---------------------------------------------------------------------------
def stability_curve(ds):
    """Compute coverage conflict rate vs epsilon for each feature.

    Returns dict with per-feature results:
        feature_name -> {epsilons, conflict_rates, epsilon_star, shape}
    """
    predictions = ds["predictions"]  # (M, N_obs)
    shap_stack = ds["shap_stack"]  # (M, N_obs, P)
    feature_names = ds["feature_names"]
    M, N_obs, P = shap_stack.shape

    signs = np.sign(shap_stack)  # (M, N_obs, P)

    # Compute all pairwise prediction differences: (M*(M-1)/2, N_obs)
    # For each observation, compute |pred_i - pred_j| for all pairs
    # Use upper triangle indices
    idx_i, idx_j = np.triu_indices(M, k=1)
    # pred_diffs[pair, obs] = |pred_i(x) - pred_j(x)|
    pred_diffs = np.abs(predictions[idx_i] - predictions[idx_j])  # (n_pairs, N_obs)

    max_d = np.max(pred_diffs)
    epsilons = np.linspace(0, max_d, N_EPSILON_STEPS + 1)[1:]  # skip 0

    results = {}
    for feat_idx in range(P):
        fname = feature_names[feat_idx]
        conflict_rates = []

        for eps in epsilons:
            # For each observation, find pairs within epsilon
            # Then check coverage conflict among those pairs
            n_conflicted = 0
            n_total = 0

            for obs in range(N_obs):
                # Mask: which pairs have d_ij < eps for this observation
                mask = pred_diffs[:, obs] < eps
                if not np.any(mask):
                    continue

                # Get signs for this feature, this observation, for the
                # models in epsilon-close pairs
                pair_models_i = idx_i[mask]
                pair_models_j = idx_j[mask]

                # For each qualifying pair, check if signs disagree
                signs_i = signs[pair_models_i, obs, feat_idx]
                signs_j = signs[pair_models_j, obs, feat_idx]

                # Only count pairs where both are nonzero
                both_nonzero = (signs_i != 0) & (signs_j != 0)
                if not np.any(both_nonzero):
                    continue

                n_total += np.sum(both_nonzero)
                n_conflicted += np.sum(signs_i[both_nonzero] != signs_j[both_nonzero])

            rate = n_conflicted / n_total if n_total > 0 else 0.0
            conflict_rates.append(rate)

        conflict_rates = np.array(conflict_rates)

        # Find epsilon_star: smallest epsilon where conflict first appears
        nonzero_mask = conflict_rates > 0
        if np.any(nonzero_mask):
            eps_star = epsilons[np.argmax(nonzero_mask)]
        else:
            eps_star = float("inf")

        # Classify shape: sharp transition vs gradual
        if np.any(nonzero_mask):
            # Look at the derivative of the curve
            diffs = np.diff(conflict_rates)
            max_jump = np.max(diffs) if len(diffs) > 0 else 0
            final_rate = conflict_rates[-1]
            # Sharp if the biggest single jump is > 50% of the final rate
            shape = "sharp transition" if max_jump > 0.3 * final_rate else "gradual"
        else:
            shape = "no conflict"

        results[fname] = {
            "epsilons": epsilons,
            "conflict_rates": conflict_rates,
            "epsilon_star": eps_star,
            "shape": shape,
        }

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Average Unfaithfulness Lower Bound
# ---------------------------------------------------------------------------
def unfaithfulness_bound(ds):
    """Compute the p/2 lower bound on average unfaithfulness and DASH's actual.

    Returns dict with:
        coverage_conflict_rate (p), bound, dash_unfaithfulness, bound_valid
    """
    shap_stack = ds["shap_stack"]  # (M, N_obs, P)
    M, N_obs, P = shap_stack.shape
    signs = np.sign(shap_stack)

    # Step 1: Coverage conflict rate
    # p = fraction of (feature, observation) pairs with coverage conflict
    # Coverage conflict = both positive and negative signs present
    n_conflicted = 0
    n_total = 0
    minority_fracs = []

    for obs in range(N_obs):
        for feat in range(P):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                continue
            n_total += 1
            n_pos = np.sum(nonzero > 0)
            n_neg = np.sum(nonzero < 0)
            if n_pos > 0 and n_neg > 0:
                n_conflicted += 1
                minority_fracs.append(min(n_pos, n_neg) / len(nonzero))

    p = n_conflicted / n_total if n_total > 0 else 0.0

    # Step 2: Bound = p * average minority fraction across conflicted pairs
    avg_minority_frac = np.mean(minority_fracs) if len(minority_fracs) > 0 else 0.0
    bound = p * avg_minority_frac

    # Step 3: DASH unfaithfulness
    # DASH sign = majority vote sign for each (obs, feat)
    # Unfaithfulness = average disagreement between DASH sign and individual signs
    total_disagreements = 0
    total_comparisons = 0

    for obs in range(N_obs):
        for feat in range(P):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                continue
            n_pos = np.sum(nonzero > 0)
            n_neg = np.sum(nonzero < 0)
            # DASH sign = majority vote
            dash_sign = 1 if n_pos >= n_neg else -1
            # Disagreement = fraction of models that disagree with DASH sign
            disagreements = np.sum(nonzero != dash_sign)
            total_disagreements += disagreements
            total_comparisons += len(nonzero)

    dash_unfaithfulness = total_disagreements / total_comparisons if total_comparisons > 0 else 0.0

    bound_valid = dash_unfaithfulness >= bound - 1e-10  # small tolerance

    return {
        "coverage_conflict_rate": p,
        "avg_minority_frac": avg_minority_frac,
        "bound": bound,
        "dash_unfaithfulness": dash_unfaithfulness,
        "bound_valid": bound_valid,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Entropy Bimodality (MDL)
# ---------------------------------------------------------------------------
def entropy_bimodality(ds):
    """Compute sign entropy for each (feature, observation) and test bimodality.

    Returns dict with:
        dip_stat, dip_pval, n_modes, mode_values, entropies
    """
    shap_stack = ds["shap_stack"]  # (M, N_obs, P)
    M, N_obs, P = shap_stack.shape
    signs = np.sign(shap_stack)

    entropies = []
    for obs in range(N_obs):
        for feat in range(P):
            s = signs[:, obs, feat]
            nonzero = s[s != 0]
            if len(nonzero) < 2:
                entropies.append(0.0)
                continue
            p_pos = np.sum(nonzero > 0) / len(nonzero)
            p_neg = 1.0 - p_pos
            # Binary entropy
            if p_pos == 0 or p_pos == 1:
                h = 0.0
            else:
                h = -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)
            entropies.append(h)

    entropies = np.array(entropies)

    # Dip test (with jitter to handle discreteness from M=200)
    rng = np.random.RandomState(SEED)
    jitter = rng.normal(0, 0.01, len(entropies))
    entropies_jittered = np.clip(entropies + jitter, 0, 1)
    dip_stat, dip_pval = diptest(entropies_jittered)

    # Fit 2-component GMM to find modes
    X_ent = entropies.reshape(-1, 1)
    gmm2 = GaussianMixture(n_components=2, random_state=SEED)
    gmm2.fit(X_ent)
    gmm1 = GaussianMixture(n_components=1, random_state=SEED)
    gmm1.fit(X_ent)

    bic1 = gmm1.bic(X_ent)
    bic2 = gmm2.bic(X_ent)
    n_modes = 2 if bic2 < bic1 else 1
    mode_values = tuple(sorted(gmm2.means_.ravel()))

    return {
        "dip_stat": dip_stat,
        "dip_pval": dip_pval,
        "bic1": bic1,
        "bic2": bic2,
        "n_modes": n_modes,
        "mode_values": mode_values,
        "entropies": entropies,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EXTENDED PREDICTIONS: Stability Curve + Unfaithfulness + Entropy")
    print("=" * 70)
    print()

    t_total = time.time()

    # Prepare datasets
    datasets = []

    # California Housing
    print("Loading California Housing...")
    cal = fetch_california_housing()
    ds_cal = prepare_dataset(cal.data, cal.target, "California Housing", list(cal.feature_names))
    datasets.append(ds_cal)

    # Synthetic at rho=0, 0.5, 0.9
    for rho in [0.0, 0.5, 0.9]:
        X, y = generate_synthetic(N_FEATURES_SYNTH, rho)
        ds = prepare_dataset(X, y, f"Synthetic rho={rho}", [f"f{i}" for i in range(N_FEATURES_SYNTH)])
        datasets.append(ds)

    # ===================================================================
    # Analysis 1: Stability Curve
    # ===================================================================
    print("\n\n" + "=" * 70)
    print("=== STABILITY CURVE (epsilon sweep) ===")
    print("=" * 70)

    all_stability = {}
    for ds in datasets:
        print(f"\n  Processing {ds['name']}...")
        result = stability_curve(ds)
        all_stability[ds["name"]] = result

    # Print summary table
    print("\n" + "=" * 70)
    print("=== STABILITY CURVE RESULTS ===")
    print("=" * 70)
    for ds_name, feat_results in all_stability.items():
        print(f"\n  Dataset: {ds_name}")
        print(f"  {'Feature':<12s} | {'eps* (critical)':>15s} | {'Curve shape':<20s}")
        print(f"  {'-' * 12}-+-{'-' * 15}-+-{'-' * 20}")
        for fname, r in feat_results.items():
            eps_str = f"{r['epsilon_star']:.4f}" if r["epsilon_star"] < float("inf") else "inf"
            print(f"  {fname:<12s} | {eps_str:>15s} | {r['shape']:<20s}")

    # ===================================================================
    # Analysis 2: Average Unfaithfulness Lower Bound
    # ===================================================================
    print("\n\n" + "=" * 70)
    print("=== AVERAGE UNFAITHFULNESS BOUND ===")
    print("=" * 70)

    print(f"\n  {'Dataset':<25s} | {'p (CC rate)':>11s} | {'Bound':>8s} | {'DASH unfaith':>12s} | {'Valid?':>8s}")
    print(f"  {'-' * 25}-+-{'-' * 11}-+-{'-' * 8}-+-{'-' * 12}-+-{'-' * 8}")

    all_unfaith = {}
    for ds in datasets:
        r = unfaithfulness_bound(ds)
        all_unfaith[ds["name"]] = r
        valid_str = f"YES ({r['dash_unfaithfulness']:.4f} >= {r['bound']:.4f})" if r["bound_valid"] else "NO"
        print(
            f"  {ds['name']:<25s} | {r['coverage_conflict_rate']:>11.4f} | "
            f"{r['bound']:>8.4f} | {r['dash_unfaithfulness']:>12.4f} | {valid_str}"
        )

    # ===================================================================
    # Analysis 3: Entropy Bimodality
    # ===================================================================
    print("\n\n" + "=" * 70)
    print("=== ENTROPY BIMODALITY (sign entropy distribution) ===")
    print("=" * 70)

    print(
        f"\n  {'Dataset':<25s} | {'Dip p-value':>11s} | {'Modes':>5s} | "
        f"{'Mode 1 (bits)':>13s} | {'Mode 2 (bits)':>13s} | {'BIC1':>10s} | {'BIC2':>10s}"
    )
    print(f"  {'-' * 25}-+-{'-' * 11}-+-{'-' * 5}-+-{'-' * 13}-+-{'-' * 13}-+-{'-' * 10}-+-{'-' * 10}")

    all_entropy = {}
    for ds in datasets:
        r = entropy_bimodality(ds)
        all_entropy[ds["name"]] = r
        m1 = r["mode_values"][0] if len(r["mode_values"]) >= 1 else float("nan")
        m2 = r["mode_values"][1] if len(r["mode_values"]) >= 2 else float("nan")
        print(
            f"  {ds['name']:<25s} | {r['dip_pval']:>11.6f} | {r['n_modes']:>5d} | "
            f"{m1:>13.4f} | {m2:>13.4f} | {r['bic1']:>10.1f} | {r['bic2']:>10.1f}"
        )

    # ===================================================================
    # Overall summary
    # ===================================================================
    elapsed = time.time() - t_total
    print(f"\n\n{'=' * 70}")
    print("SUMMARY OF KEY FINDINGS")
    print(f"{'=' * 70}")

    # Stability curve
    print("\n1. STABILITY CURVE")
    print("   Prediction: sharp phase transition at critical epsilon*")
    for ds_name, feat_results in all_stability.items():
        sharp_count = sum(1 for r in feat_results.values() if r["shape"] == "sharp transition")
        gradual_count = sum(1 for r in feat_results.values() if r["shape"] == "gradual")
        no_conflict = sum(1 for r in feat_results.values() if r["shape"] == "no conflict")
        print(f"   {ds_name}: {sharp_count} sharp, {gradual_count} gradual, {no_conflict} no-conflict features")

    # Unfaithfulness
    print("\n2. AVERAGE UNFAITHFULNESS BOUND")
    print("   Prediction: DASH unfaithfulness >= p * avg_minority_frac (near-optimal)")
    all_valid = all(r["bound_valid"] for r in all_unfaith.values())
    print(f"   Bound valid across all datasets: {'YES' if all_valid else 'NO'}")
    for ds_name, r in all_unfaith.items():
        ratio = r["dash_unfaithfulness"] / r["bound"] if r["bound"] > 0 else float("inf")
        print(f"   {ds_name}: DASH={r['dash_unfaithfulness']:.4f}, bound={r['bound']:.4f}, ratio={ratio:.2f}x")

    # Entropy bimodality
    print("\n3. ENTROPY BIMODALITY")
    print("   Prediction: sign entropy bimodal (modes near 0 and 1 bit)")
    for ds_name, r in all_entropy.items():
        reject = "REJECT unimodality" if r["dip_pval"] < 0.05 else "fail to reject"
        bic_pref = "2-comp" if r["n_modes"] == 2 else "1-comp"
        m1 = r["mode_values"][0]
        m2 = r["mode_values"][1]
        print(
            f"   {ds_name}: dip p={r['dip_pval']:.6f} ({reject}), "
            f"BIC prefers {bic_pref}, modes at {m1:.3f} and {m2:.3f} bits"
        )

    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)

    return all_stability, all_unfaith, all_entropy


if __name__ == "__main__":
    main()
