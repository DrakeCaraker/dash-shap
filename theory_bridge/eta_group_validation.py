#!/usr/bin/env python3
"""
Eta law validation: GROUP MEAN stability vs WITHIN-GROUP instability.

The η law predicts:
- Invariant projection (group mean) has LOW flip rate
- Variant projection (within-group deviation) has HIGH flip rate
- Fraction of stable components = η = 1/g

This tests the ACTUAL prediction, not the naive "individual features in
groups are unstable" (which the previous test showed to be INVERTED).
"""

import numpy as np
import xgboost as xgb
import shap
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                               load_breast_cancer)
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, wilcoxon
import warnings, time

warnings.filterwarnings("ignore")

M = 200
N_OBS = 100
SEED = 42


def train_and_shap(X_train, y_train, X_explain, M, seed):
    """Train M models, compute SHAP, return (M, N_obs, P) array."""
    rng = np.random.RandomState(seed)
    bg = X_train[:100]
    shap_stack = []
    for i in range(M):
        s = rng.randint(0, 2**31)
        params = {
            "n_estimators": rng.choice([100, 200, 300]),
            "max_depth": rng.choice([3, 4, 5, 6]),
            "learning_rate": rng.choice([0.05, 0.1, 0.2]),
            "colsample_bytree": rng.uniform(0.3, 0.8),
            "subsample": rng.uniform(0.6, 1.0),
            "random_state": s, "n_jobs": 1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train, verbose=False)
        explainer = shap.TreeExplainer(m, bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_explain)
        shap_stack.append(sv)
    return np.stack(shap_stack, axis=0)


def flip_rate(values_across_models):
    """
    Compute flip rate for a 1D array of values across M models.
    Flip rate = minority fraction of signs (ignoring zeros).
    """
    signs = np.sign(values_across_models)
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0.0
    n_pos = (nonzero > 0).sum()
    n_neg = (nonzero < 0).sum()
    return min(n_pos, n_neg) / (n_pos + n_neg)


def test_group(shap_stack, group_indices, group_name):
    """
    Test the η law for one group of features.

    shap_stack: (M, N_obs, P) array
    group_indices: list of feature indices in the group

    Computes:
    - invariant_flip: flip rate of the GROUP MEAN across group features
    - variant_flip: flip rate of WITHIN-GROUP DEVIATIONS (each feature minus group mean)
    - η_predicted = 1/g
    - η_observed = fraction of components with flip rate < 0.05
    """
    g = len(group_indices)
    M_models, N_obs, P = shap_stack.shape

    # Extract group SHAP values: (M, N_obs, g)
    group_shap = shap_stack[:, :, group_indices]

    # Invariant projection: group mean (M, N_obs)
    invariant = group_shap.mean(axis=2)

    # Variant projections: deviations from mean (M, N_obs, g)
    variant = group_shap - invariant[:, :, np.newaxis]

    # Compute flip rates
    inv_flips = []
    var_flips = []
    for obs in range(N_obs):
        inv_flips.append(flip_rate(invariant[:, obs]))
        for j in range(g):
            var_flips.append(flip_rate(variant[:, obs, j]))

    mean_inv_flip = np.mean(inv_flips)
    mean_var_flip = np.mean(var_flips)

    # η observed: fraction of components (1 invariant + g variant) with low flip rate
    # η = 1/g means 1 out of g dimensions is stable.
    # The invariant is 1 dimension; the variant is g-1 dimensions.
    # So η = 1/g.

    eta_predicted = 1.0 / g

    # Is invariant more stable than variant?
    # Compare inv_flips (N_obs values) to var_flips averaged per obs
    var_flips_per_obs = []
    for obs in range(N_obs):
        obs_var_flips = [flip_rate(variant[:, obs, j]) for j in range(g)]
        var_flips_per_obs.append(np.mean(obs_var_flips))

    deltas = np.array(var_flips_per_obs) - np.array(inv_flips)
    if np.any(deltas != 0):
        stat, p_val = wilcoxon(deltas, alternative='greater')
    else:
        stat, p_val = 0, 1.0

    return {
        "group": group_name,
        "g": g,
        "eta_predicted": eta_predicted,
        "mean_invariant_flip": mean_inv_flip,
        "mean_variant_flip": mean_var_flip,
        "ratio": mean_var_flip / max(mean_inv_flip, 1e-10),
        "wilcoxon_p": p_val,
        "invariant_more_stable": mean_inv_flip < mean_var_flip,
    }


def generate_synthetic_groups(rho, group_sizes, n=2000, seed=42):
    """
    Generate synthetic data with specified correlation group structure.
    group_sizes: list of group sizes. E.g., [2, 3] = one pair + one triple.
    Total features = sum(group_sizes) + (extra independent features up to 10).
    """
    rng = np.random.RandomState(seed)
    p_total = max(10, sum(group_sizes) + 2)
    cov = np.eye(p_total)
    idx = 0
    for gs in group_sizes:
        for i in range(idx, idx + gs):
            for j in range(idx, idx + gs):
                if i != j:
                    cov[i, j] = rho
        idx += gs
    X = rng.multivariate_normal(np.zeros(p_total), cov, n)
    # Target depends on first few features
    y = sum(X[:, i] * (1.0 / (i + 1)) for i in range(min(5, p_total)))
    y += rng.normal(0, 0.5, n)
    return X, y, group_sizes


# ==================== MAIN ====================
if __name__ == "__main__":
    all_results = []

    # === SYNTHETIC TESTS (known groups by construction) ===

    print("=" * 70)
    print("SYNTHETIC TESTS: Known groups by construction")
    print("=" * 70)

    # Test 1: Z/2Z (one pair, rho=0.9)
    print("\n--- Synthetic Z/2Z (pair at rho=0.9) ---")
    X, y, gs = generate_synthetic_groups(0.9, [2], n=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rng = np.random.RandomState(SEED+1)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")
    r = test_group(shap_stack, [0, 1], "Z/2Z pair (f0,f1)")
    all_results.append(("synthetic_Z2Z", r))
    print(f"  Invariant (mean) flip: {r['mean_invariant_flip']:.4f}")
    print(f"  Variant (diff) flip:   {r['mean_variant_flip']:.4f}")
    print(f"  Ratio var/inv:         {r['ratio']:.1f}x")
    print(f"  Wilcoxon p:            {r['wilcoxon_p']:.2e}")
    print(f"  eta predicted: {r['eta_predicted']:.2f}, invariant more stable: {r['invariant_more_stable']}")

    # Test 2: S_3 (triple at rho=0.9)
    print("\n--- Synthetic S_3 (triple at rho=0.9) ---")
    X, y, gs = generate_synthetic_groups(0.9, [3], n=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")
    r = test_group(shap_stack, [0, 1, 2], "S_3 triple (f0,f1,f2)")
    all_results.append(("synthetic_S3", r))
    print(f"  Invariant (mean) flip: {r['mean_invariant_flip']:.4f}")
    print(f"  Variant (dev) flip:    {r['mean_variant_flip']:.4f}")
    print(f"  Ratio var/inv:         {r['ratio']:.1f}x")
    print(f"  Wilcoxon p:            {r['wilcoxon_p']:.2e}")

    # Test 3: (Z/2Z)^2 (two independent pairs at rho=0.9)
    print("\n--- Synthetic (Z/2Z)^2 (two pairs at rho=0.9) ---")
    X, y, gs = generate_synthetic_groups(0.9, [2, 2], n=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")
    for pair_idx, pair_name in enumerate(["pair1 (f0,f1)", "pair2 (f2,f3)"]):
        indices = [pair_idx*2, pair_idx*2+1]
        r = test_group(shap_stack, indices, pair_name)
        all_results.append(("synthetic_Z2Z_sq_" + pair_name, r))
        print(f"  {pair_name}: inv_flip={r['mean_invariant_flip']:.4f}, var_flip={r['mean_variant_flip']:.4f}, ratio={r['ratio']:.1f}x, p={r['wilcoxon_p']:.2e}")

    # Test 4: S_4 (4 correlated features)
    print("\n--- Synthetic S_4 (quad at rho=0.9) ---")
    X, y, gs = generate_synthetic_groups(0.9, [4], n=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")
    r = test_group(shap_stack, [0, 1, 2, 3], "S_4 quad (f0-f3)")
    all_results.append(("synthetic_S4", r))
    print(f"  Invariant (mean) flip: {r['mean_invariant_flip']:.4f}")
    print(f"  Variant (dev) flip:    {r['mean_variant_flip']:.4f}")
    print(f"  Ratio var/inv:         {r['ratio']:.1f}x")
    print(f"  Wilcoxon p:            {r['wilcoxon_p']:.2e}")

    # === REAL DATA TESTS (domain-identified groups) ===

    print("\n" + "=" * 70)
    print("REAL DATA TESTS: Domain-identified groups")
    print("=" * 70)

    # Test 5: California Housing
    print("\n--- California Housing ---")
    data = fetch_california_housing()
    X, y = data.data, data.target
    fnames = list(data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Domain groups: Lat/Long (geographic), AveRooms/AveBedrms (room counts)
    lat_idx = fnames.index("Latitude")
    lon_idx = fnames.index("Longitude")
    rooms_idx = fnames.index("AveRooms")
    bedrms_idx = fnames.index("AveBedrms")

    r = test_group(shap_stack, [lat_idx, lon_idx], "Lat/Long")
    all_results.append(("california_LatLong", r))
    print(f"  Lat/Long: inv_flip={r['mean_invariant_flip']:.4f}, var_flip={r['mean_variant_flip']:.4f}, ratio={r['ratio']:.1f}x, p={r['wilcoxon_p']:.2e}")

    r = test_group(shap_stack, [rooms_idx, bedrms_idx], "Rooms/Bedrms")
    all_results.append(("california_RoomsBedrms", r))
    print(f"  Rooms/Bedrms: inv_flip={r['mean_invariant_flip']:.4f}, var_flip={r['mean_variant_flip']:.4f}, ratio={r['ratio']:.1f}x, p={r['wilcoxon_p']:.2e}")

    # Test 6: Breast Cancer (mean/se/worst triples)
    print("\n--- Breast Cancer ---")
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(float)
    fnames = list(data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Domain groups: 10 base measurements x {mean, se, worst}
    # Feature naming: "mean radius", "mean texture", ..., "radius error", ..., "worst radius", ...
    # Indices: 0-9 = mean, 10-19 = se, 20-29 = worst
    base_measurements = ["radius", "texture", "perimeter", "area", "smoothness",
                         "compactness", "concavity", "concave points", "symmetry",
                         "fractal dimension"]
    for bm_idx, bm in enumerate(base_measurements[:5]):  # test first 5
        indices = [bm_idx, bm_idx + 10, bm_idx + 20]  # mean, se, worst
        r = test_group(shap_stack, indices, f"{bm} (mean/se/worst)")
        all_results.append((f"breast_{bm}", r))
        print(f"  {bm}: inv_flip={r['mean_invariant_flip']:.4f}, var_flip={r['mean_variant_flip']:.4f}, ratio={r['ratio']:.1f}x, p={r['wilcoxon_p']:.2e}")

    # Test 7: Diabetes with correlation-derived groups
    print("\n--- Diabetes (correlation-derived groups) ---")
    data = load_diabetes()
    X, y = data.data, data.target
    fnames = list(data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_explain = X_test[rng.choice(len(X_test), min(N_OBS, len(X_test)), replace=False)]
    print(f"Training {M} models...")
    t0 = time.time()
    shap_stack = train_and_shap(X_train, y_train, X_explain, M, SEED)
    print(f"  Done in {time.time()-t0:.0f}s")

    # Find correlation groups at |rho| > 0.5
    corr = np.corrcoef(X.T)
    print(f"  Feature names: {fnames}")
    print(f"  Correlation pairs with |rho| > 0.5:")
    used = set()
    diabetes_groups = []
    for i in range(len(fnames)):
        for j in range(i+1, len(fnames)):
            if abs(corr[i, j]) > 0.5 and i not in used and j not in used:
                print(f"    {fnames[i]}-{fnames[j]}: rho={corr[i,j]:.3f}")
                diabetes_groups.append((i, j, f"{fnames[i]}/{fnames[j]}"))
                used.add(i)
                used.add(j)

    for i, j, name in diabetes_groups:
        r = test_group(shap_stack, [i, j], name)
        all_results.append((f"diabetes_{name}", r))
        print(f"  {name}: inv_flip={r['mean_invariant_flip']:.4f}, var_flip={r['mean_variant_flip']:.4f}, ratio={r['ratio']:.1f}x, p={r['wilcoxon_p']:.2e}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("ETA LAW VALIDATION SUMMARY (GROUP MEAN vs WITHIN-GROUP)")
    print("=" * 70)
    print(f"{'Test':>35s}  {'g':>3s}  {'eta':>5s}  {'Inv flip':>9s}  {'Var flip':>9s}  {'Ratio':>6s}  {'p-value':>9s}  {'Pass?':>5s}")
    print("-" * 95)
    for name, r in all_results:
        passed = "YES" if r['invariant_more_stable'] and r['wilcoxon_p'] < 0.05 else "NO"
        print(f"{r['group']:>35s}  {r['g']:>3d}  {r['eta_predicted']:>5.2f}  {r['mean_invariant_flip']:>9.4f}  {r['mean_variant_flip']:>9.4f}  {r['ratio']:>6.1f}x  {r['wilcoxon_p']:>9.2e}  {passed:>5s}")

    n_pass = sum(1 for _, r in all_results if r['invariant_more_stable'] and r['wilcoxon_p'] < 0.05)
    print(f"\nPassed: {n_pass}/{len(all_results)}")
    print("\nInterpretation:")
    print("  - 'Inv flip' = flip rate of the G-invariant component (group mean of SHAP values)")
    print("  - 'Var flip' = flip rate of the G-variant components (within-group deviations)")
    print("  - 'Ratio' = Var/Inv; higher means stronger separation (eta law predicts >> 1)")
    print("  - 'Pass' = invariant strictly more stable AND Wilcoxon p < 0.05")
    print("  - eta = 1/g = fraction of stable dimensions in the group representation")
