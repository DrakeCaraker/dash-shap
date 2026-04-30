#!/usr/bin/env python3
"""
MI-Only Dependence Test
========================
Critical test: does MI-only dependence (rho≈0, I>0) cause SHAP instability
that correlation-based methods miss?

Setup:
  X1 ~ N(0,1)
  X2 = X1² + noise  (MI-dependent on X1, Pearson rho ≈ 0)
  X3..X6 ~ N(0,1) independent
  Y = X1² + X3 + noise  (X2 is redundant with nonlinear component)

If MI boundary matters:
  X1-X2 pair should have HIGH instability (competing for nonlinear credit)
  X1-X3 pair should have LOW instability (independent, different roles)
  Correlation predicts BOTH pairs are stable (rho ≈ 0 for both)
  MI predicts X1-X2 unstable (I > 0) but X1-X3 stable (I ≈ 0)
"""

import numpy as np
import xgboost as xgb
import shap
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr, pearsonr
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
N = 2000
M = 100

# Create dataset
X1 = np.random.randn(N)
X2 = X1**2 + 0.1 * np.random.randn(N)
X3 = np.random.randn(N)
X4 = np.random.randn(N)
X5 = np.random.randn(N)
X6 = np.random.randn(N)

X = np.column_stack([X1, X2, X3, X4, X5, X6])
Y = X1**2 + X3 + 0.5 * np.random.randn(N)

# Check correlation and MI for key pairs
rho_12 = abs(pearsonr(X[:, 0], X[:, 1])[0])
mi_12 = mutual_info_regression(X[:, 0].reshape(-1, 1), X[:, 1], random_state=42)[0]
rho_13 = abs(pearsonr(X[:, 0], X[:, 2])[0])
mi_13 = mutual_info_regression(X[:, 0].reshape(-1, 1), X[:, 2], random_state=42)[0]

print("=" * 60)
print("MI-ONLY DEPENDENCE TEST")
print("=" * 60)
print(f"\nX1-X2: |Pearson| = {rho_12:.4f}, MI = {mi_12:.4f}")
print(f"X1-X3: |Pearson| = {rho_13:.4f}, MI = {mi_13:.4f}")
print(f"\nX1-X2: MI >> 0 but rho ~ 0 (nonlinear dependence)")
print(f"X1-X3: MI ~ 0 and rho ~ 0 (independent)")

# Train M models, compute SHAP
print(f"\nTraining {M} XGBoost models...")
shap_matrices = []
for i in range(M):
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{M}")
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=i,
        verbosity=0,
    )
    model.fit(X[:1600], Y[:1600])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[1600:1700])
    shap_matrices.append(sv)

shap_stack = np.array(shap_matrices)  # (M, 100, 6)

# Measure instability for ALL pairs
P = 6
names = ["X1", "X2", "X3", "X4", "X5", "X6"]

print(f"\n{'Pair':<10} {'|rho|':>8} {'MI':>8} {'FlipRate':>10} {'|SHAP_r|':>10}")
print("-" * 50)

results = {}
for j in range(P):
    for k in range(j + 1, P):
        rho_jk = abs(pearsonr(X[:, j], X[:, k])[0])
        mi_jk = mutual_info_regression(X[:, j].reshape(-1, 1), X[:, k], random_state=42)[0]

        # FlipRate: fraction of test points where models disagree on |SHAP_j| vs |SHAP_k|
        abs_j = np.abs(shap_stack[:, :, j])
        abs_k = np.abs(shap_stack[:, :, k])
        j_wins = (abs_j > abs_k).astype(float)
        flip_rate = np.mean(np.std(j_wins, axis=0) > 0.1)

        # Mean absolute SHAP cross-model correlation
        shap_corrs = []
        for obs in range(50):
            sj = shap_stack[:, obs, j]
            sk = shap_stack[:, obs, k]
            if np.std(sj) > 1e-10 and np.std(sk) > 1e-10:
                r, _ = spearmanr(sj, sk)
                shap_corrs.append(abs(r))
        shap_rho = np.mean(shap_corrs) if shap_corrs else 0.0

        pair = f"{names[j]}-{names[k]}"
        tag = ""
        if mi_jk > 0.1 and rho_jk < 0.15:
            tag = " <-- MI-ONLY"
        results[pair] = (rho_jk, mi_jk, flip_rate, shap_rho)
        print(f"{pair:<10} {rho_jk:>8.4f} {mi_jk:>8.4f} {flip_rate:>10.4f} {shap_rho:>10.4f}{tag}")

# Key comparison
print("\n" + "=" * 60)
print("KEY COMPARISON: X1-X2 vs X1-X3")
print("=" * 60)

r12 = results["X1-X2"]
r13 = results["X1-X3"]
print(f"\n  X1-X2 (MI-only):   |rho|={r12[0]:.4f}  MI={r12[1]:.4f}  FlipRate={r12[2]:.4f}")
print(f"  X1-X3 (independent): |rho|={r13[0]:.4f}  MI={r13[1]:.4f}  FlipRate={r13[2]:.4f}")
print(f"\n  Correlation predicts: both stable (|rho| ~ 0 for both)")

if r12[1] > 0.1 and r13[1] < 0.1:
    print(f"  MI predicts:         X1-X2 unstable (MI={r12[1]:.3f}), X1-X3 stable (MI={r13[1]:.3f})")
else:
    print(f"  MI:                  X1-X2 MI={r12[1]:.3f}, X1-X3 MI={r13[1]:.3f}")

if r12[2] > r13[2] + 0.05:
    print(f"\n  RESULT: MI-only dependence DOES cause higher instability")
    print(f"  FlipRate gap: {r12[2] - r13[2]:.4f} (X1-X2 minus X1-X3)")
    print(f"  MI CORRECTLY predicts instability that correlation misses")
elif r12[2] > r13[2]:
    print(f"\n  RESULT: Small difference in expected direction")
    print(f"  FlipRate gap: {r12[2] - r13[2]:.4f}")
else:
    print(f"\n  RESULT: MI-only dependence does NOT cause higher instability")
    print(f"  FlipRate gap: {r12[2] - r13[2]:.4f}")
    print(f"  The binary boundary (MI > 0) is correct but does not manifest")
    print(f"  as measurable instability in this setup")
