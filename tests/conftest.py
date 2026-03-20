"""Shared fixtures for the dash_shap test suite."""

import numpy as np
import pytest
from dash_shap.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear


@pytest.fixture(scope="session")
def synthetic_linear():
    """Session-scoped synthetic linear dataset (N=500, P=20, rho=0.5)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    return {
        "X_train": Xtr,
        "y_train": ytr,
        "X_val": Xv,
        "y_val": yv,
        "X_explain": Xexp,
        "y_explain": yexp,
        "X_test": Xte,
        "y_test": yte,
        "groups": grps,
        "true_importance": true_imp,
        "meta": meta,
    }


@pytest.fixture(scope="session")
def synthetic_nonlinear():
    """Session-scoped synthetic nonlinear dataset (N=500, P=20, rho=0.7)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_nonlinear(
        N=500, P=20, group_size=5, rho=0.7, seed=42
    )
    return {
        "X_train": Xtr,
        "y_train": ytr,
        "X_val": Xv,
        "y_val": yv,
        "X_explain": Xexp,
        "y_explain": yexp,
        "X_test": Xte,
        "y_test": yte,
        "groups": grps,
        "true_importance": true_imp,
        "meta": meta,
    }


@pytest.fixture(scope="session")
def synthetic_small():
    """Session-scoped small synthetic dataset for fast tests (N=200, P=10)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.5, seed=42
    )
    return {
        "X_train": Xtr,
        "y_train": ytr,
        "X_val": Xv,
        "y_val": yv,
        "X_explain": Xexp,
        "y_explain": yexp,
        "X_test": Xte,
        "y_test": yte,
        "groups": grps,
        "true_importance": true_imp,
        "meta": meta,
    }


@pytest.fixture(scope="session")
def trained_population(synthetic_linear):
    """Session-scoped pre-trained DASH population (M=10) for reuse across tests."""
    from dash_shap.core.pipeline import DASHPipeline

    d = synthetic_linear
    pipe = DASHPipeline(
        M=10,
        K=5,
        epsilon=0.15,
        delta=0.01,
        seed=42,
        verbose=False,
        n_jobs=1,
    )
    pipe.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
    return pipe


def _make_4quadrant_shap_matrices(K=12, n_ref=50, seed=0):
    """Construct a (K, n_ref, 4) tensor spanning all four IS-plot quadrants.

    f0: high importance, low FSI   → QI  (Robust Driver)
    f1: high importance, high FSI  → QII (Collinear Cluster)
    f2: low importance,  low FSI   → QIII (Unimportant)
    f3: low importance,  high FSI  → QIV (Fragile)

    FSI = mean_std / (global_importance + eps)
      where mean_std   = mean(sqrt(var(shap, axis=0, ddof=1)), axis=0)
            global_imp = mean(|mean(shap, axis=0)|, axis=0)

    Design targets per feature:
      f0: global_imp ≈ 1.0, mean_std ≈ 0.03  → FSI ≈ 0.03
      f1: global_imp ≈ 1.0, mean_std ≈ 0.80  → FSI ≈ 0.80
      f2: global_imp ≈ 0.1, mean_std ≈ 0.01  → FSI ≈ 0.10
      f3: global_imp ≈ 0.1, mean_std ≈ 0.40  → FSI ≈ 4.00
    """
    rng = np.random.default_rng(seed)
    matrices = np.zeros((K, n_ref, 4), dtype=np.float64)

    # f0: all models agree on high importance
    matrices[:, :, 0] = 1.0 + rng.normal(0, 0.03, size=(K, n_ref))

    # f1: models disagree widely on high-importance feature
    model_offsets_f1 = rng.normal(0, 0.8, size=(K, 1))
    matrices[:, :, 1] = 1.0 + model_offsets_f1 + rng.normal(0, 0.02, size=(K, n_ref))

    # f2: all models agree on low importance
    matrices[:, :, 2] = 0.1 + rng.normal(0, 0.01, size=(K, n_ref))

    # f3: models disagree widely on low-importance feature
    model_offsets_f3 = rng.normal(0, 0.4, size=(K, 1))
    matrices[:, :, 3] = 0.1 + model_offsets_f3 + rng.normal(0, 0.01, size=(K, n_ref))

    return matrices


@pytest.fixture(scope="session")
def dash_result():
    """DASHResult with 4 features spanning all 4 IS-plot quadrants.

    f0: high importance, low FSI   (QI:   Robust Driver)
    f1: high importance, high FSI  (QII:  Collinear Cluster)
    f2: low importance,  low FSI   (QIII: Unimportant)
    f3: low importance,  high FSI  (QIV:  Fragile)

    K=12 satisfies the K>=10 requirement for bootstrap-based extensions.
    """
    from dash_shap.core.result import DASHResult

    matrices = _make_4quadrant_shap_matrices(K=12, n_ref=50, seed=0)
    result = DASHResult.from_shap_matrices(
        matrices,
        feature_names=["f0_robust", "f1_collinear", "f2_unimportant", "f3_fragile"],
    )

    # Self-verify the fixture's quadrant claims
    assert result.fsi[0] < 0.3, f"f0 should be QI (low FSI), got FSI={result.fsi[0]:.3f}"
    assert result.fsi[1] > 0.5, f"f1 should be QII (high FSI), got FSI={result.fsi[1]:.3f}"
    assert result.global_importance[0] > result.global_importance[2], "f0 importance should exceed f2 importance"
    assert result.global_importance[3] < result.fsi[3], "f3 should be QIV: FSI exceeds importance"

    return result
