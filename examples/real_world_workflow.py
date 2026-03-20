"""Real-world workflow: DASH + Extensions on sklearn Breast Cancer dataset.

Demonstrates empirical verifiability of the extensions framework on a
publicly available dataset with known collinearity structure (21 feature
pairs with |r| > 0.9).

Run: python examples/real_world_workflow.py

Requires: scikit-learn, xgboost, shap
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from dash_shap import DASHPipeline
from dash_shap.extensions import (
    feature_groups,
    causal_flags,
    audit_report,
    robust_certification,
    confidence_intervals,
)


def main():
    print("=" * 65)
    print("DASH Real-World Workflow: Breast Cancer Dataset")
    print("=" * 65)
    print("Known structure: 30 features, 21 pairs with |r| > 0.9")
    print()

    # ------------------------------------------------------------------ #
    # 1. Load and split data
    # ------------------------------------------------------------------ #
    data = load_breast_cancer()
    X, y = data.data.astype(float), data.target.astype(float)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.20, random_state=42)
    X_train, X_explain, y_train, _ = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Explain: {X_explain.shape}, Test: {X_test.shape}")
    print()

    # ------------------------------------------------------------------ #
    # 2. Fit DASHPipeline
    # ------------------------------------------------------------------ #
    print("[1] Fitting DASHPipeline (M=50, K=15) ...")
    pipe = DASHPipeline(
        M=50,
        K=15,
        epsilon=0.05,
        epsilon_mode="relative",
        seed=42,
        verbose=False,
    )
    pipe.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        X_ref=X_explain,
        feature_names=list(data.feature_names),
    )
    result = pipe.result_
    print(f"    DASHResult: K={result.K}, P={result.P}")
    print()

    # ------------------------------------------------------------------ #
    # 3. Robust Certification — which features are definitively top-5?
    # ------------------------------------------------------------------ #
    print("[2] Robust Certification ...")
    cert = robust_certification(result, k_values=[3, 5, 10])
    print(cert.summary())
    print()

    # ------------------------------------------------------------------ #
    # 4. Confidence Intervals — how wide are the top features' CIs?
    # ------------------------------------------------------------------ #
    print("[3] Confidence Intervals (95%) ...")
    ci = confidence_intervals(result, alpha=0.05, n_boot=300, seed=42)
    # Show top-5 by importance
    top5 = np.argsort(-result.global_importance)[:5]
    print(f"{'Feature':<35} {'Importance CI':>30}")
    print("-" * 67)
    for i in top5:
        lo, pt, hi = ci.importance_ci[i]
        print(f"{result.feature_names[i]:<35} [{lo:.4f}, {pt:.4f}, {hi:.4f}]")
    print()

    # Note: Phase 2 and 3 extensions (feature_groups, causal_flags, audit_report)
    # are not yet implemented. The imports above will raise NotImplementedError
    # once stubs are added. Remove the try/except when Phase 2/3 land.

    print("[4] Feature Groups (Phase 2 — not yet implemented) ...")
    try:
        grps = feature_groups(result, threshold=0.8, X_ref=X_explain)
        print(grps.summary())
    except (NotImplementedError, AttributeError, ImportError) as e:
        print(f"    (Skipped: {e})")
    print()

    print("[5] Causal Flags (Phase 3 — not yet implemented) ...")
    try:
        flags = causal_flags(result, X_ref=X_explain)
        print(flags.summary())
    except (NotImplementedError, AttributeError, ImportError) as e:
        print(f"    (Skipped: {e})")
    print()

    print("[6] Audit Report (Phase 3 — not yet implemented) ...")
    try:
        report = audit_report(result, X_ref=X_explain)
        print(report.summary())
    except (NotImplementedError, AttributeError, ImportError) as e:
        print(f"    (Skipped: {e})")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
