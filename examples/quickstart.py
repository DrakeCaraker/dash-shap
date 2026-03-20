"""DASH Quickstart — Fit DASH on synthetic correlated data.

Run this script to verify your installation and see DASH in action:
    python examples/quickstart.py

Demonstrates: data generation, pipeline fitting, global importance,
IS plot, FSI interpretation, and comparison with a single-best baseline.
Runtime: < 2 minutes (M=20, K=8 for speed).
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless environments

import numpy as np
from dash_shap import DASHPipeline
from dash_shap.experiments.synthetic import generate_synthetic_linear
from dash_shap.baselines import SingleBestBaseline

# 1. Generate synthetic data (P=20 features in 4 correlated groups of 5)
print("Generating synthetic dataset (N=2000, P=20, rho=0.9)...")
(X_train, y_train, X_val, y_val, X_explain, _,
 X_test, y_test, groups, true_importance, meta) = generate_synthetic_linear(
    N=2000, P=20, group_size=5, rho=0.9, seed=42
)

# 2. Fit DASH (use M=20, K=8 for a quick demo; paper uses M=200, K=30)
print("\nFitting DASH pipeline (M=20, K=8)...")
pipe = DASHPipeline(M=20, K=8, epsilon=0.10, seed=42, verbose=True)
pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

# 3. Inspect global importance
print("\n--- Global Feature Importance (top 5) ---")
ranking = pipe.get_importance_ranking()
for rank, feat_idx in enumerate(ranking[:5]):
    print(f"  {rank+1}. f{feat_idx}: {pipe.global_importance_[feat_idx]:.4f}")

# 4. Feature Stability Index
fsi = pipe.get_fsi()
print("\n--- FSI Summary ---")
print(fsi.summary(top_k=10))

# 5. Save IS plot
fig = pipe.plot_importance_stability(groups=groups)
fig.savefig("quickstart_is_plot.png", dpi=100, bbox_inches="tight")
print("\nIS plot saved to quickstart_is_plot.png")

# 6. Compare with Single Best baseline
print("\nFitting SingleBest baseline for comparison...")
sb = SingleBestBaseline(n_trials=10, seed=42)
sb.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

print(f"\nDASH global importance (first 5): {np.round(pipe.global_importance_[:5], 3)}")
print(f"SingleBest importance (first 5): {np.round(sb.global_importance_[:5], 3)}")
print("\nQuickstart complete! See quickstart_is_plot.png for the IS plot.")
