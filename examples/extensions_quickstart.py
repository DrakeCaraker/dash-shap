"""Extensions Quickstart — Phase 1 Extensions Demo.

Mirrors examples/quickstart.py but shows the extensions framework:
  result_ → confidence_intervals, robust_certification, partial_order

Run: python examples/extensions_quickstart.py
"""
import numpy as np

from dash_shap import DASHPipeline
from dash_shap.experiments.synthetic import generate_synthetic_linear
from dash_shap.extensions import (
    confidence_intervals,
    robust_certification,
    partial_order,
)


def main():
    print("=" * 60)
    print("DASH Extensions Quickstart")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Generate data and fit pipeline
    # ------------------------------------------------------------------ #
    print("\n[1] Generating synthetic data (N=1000, P=8, rho=0.9) ...")
    (X_train, y_train, X_val, y_val, X_explain, _,
     X_test, y_test, groups, true_importance, meta) = generate_synthetic_linear(
        N=1000, P=8, group_size=4, rho=0.9, seed=42
    )

    print("[2] Fitting DASHPipeline (M=20, K=10) ...")
    pipe = DASHPipeline(M=20, K=10, epsilon=0.10, seed=42, verbose=False)
    pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    result = pipe.result_
    print(f"    DASHResult: K={result.K}, n_ref={result.n_ref}, P={result.P}")
    print(f"    Top features by importance: {list(np.argsort(-result.global_importance))}")

    # ------------------------------------------------------------------ #
    # 2. Robust Certification (Extension 9)
    # ------------------------------------------------------------------ #
    print("\n[3] Robust Certification ...")
    cert = robust_certification(result)
    print(cert.summary())

    # ------------------------------------------------------------------ #
    # 3. Confidence Intervals (Extension 1)
    # ------------------------------------------------------------------ #
    print("\n[4] Confidence Intervals (95%, 200 bootstrap replicates) ...")
    ci = confidence_intervals(result, alpha=0.05, n_boot=200, seed=42)
    print(ci.summary())

    # ------------------------------------------------------------------ #
    # 4. Partial Order (Extension 2)
    # ------------------------------------------------------------------ #
    print("\n[5] Partial Order (method='fraction') ...")
    po = partial_order(result, alpha=0.1, method="fraction")
    print(po.summary())

    top2 = list(np.argsort(-result.global_importance)[:2])
    pi_top2 = po.confidence_matrix[top2[0], top2[1]]
    pi_top2_rev = po.confidence_matrix[top2[1], top2[0]]
    f0, f1 = result.feature_names[top2[0]], result.feature_names[top2[1]]
    print(f"\nPaper 2 check — within-group pair ({f0}, {f1}):")
    print(f"  π({f0} > {f1}) = {pi_top2:.3f}")
    print(f"  π({f1} > {f0}) = {pi_top2_rev:.3f}")
    if abs(pi_top2 - 0.5) < 0.2:
        print("  → π ≈ 0.5 — attribution split between collinear features (expected!)")
    else:
        print(f"  → π = {pi_top2:.3f} (one feature dominates)")

    # ------------------------------------------------------------------ #
    # 5. Serialize result for reuse
    # ------------------------------------------------------------------ #
    import pathlib
    import tempfile
    print("\n[6] Serialization round-trip ...")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "demo_result"
        result.save(path)
        loaded = type(result).load(path)
    print(f"    Saved and reloaded — shapes match: {loaded.all_shap_matrices.shape}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
