"""Shared fixtures for the dash_shap test suite."""
import numpy as np
import pytest
from dash_shap.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear


@pytest.fixture(scope="session")
def synthetic_linear():
    """Session-scoped synthetic linear dataset (N=500, P=20, rho=0.5)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = \
        generate_synthetic_linear(N=500, P=20, group_size=5, rho=0.5, seed=42)
    return {
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xv, "y_val": yv,
        "X_explain": Xexp, "y_explain": yexp,
        "X_test": Xte, "y_test": yte,
        "groups": grps, "true_importance": true_imp, "meta": meta,
    }


@pytest.fixture(scope="session")
def synthetic_nonlinear():
    """Session-scoped synthetic nonlinear dataset (N=500, P=20, rho=0.7)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = \
        generate_synthetic_nonlinear(N=500, P=20, group_size=5, rho=0.7, seed=42)
    return {
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xv, "y_val": yv,
        "X_explain": Xexp, "y_explain": yexp,
        "X_test": Xte, "y_test": yte,
        "groups": grps, "true_importance": true_imp, "meta": meta,
    }


@pytest.fixture(scope="session")
def synthetic_small():
    """Session-scoped small synthetic dataset for fast tests (N=200, P=10)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = \
        generate_synthetic_linear(N=200, P=10, group_size=5, rho=0.5, seed=42)
    return {
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xv, "y_val": yv,
        "X_explain": Xexp, "y_explain": yexp,
        "X_test": Xte, "y_test": yte,
        "groups": grps, "true_importance": true_imp, "meta": meta,
    }


@pytest.fixture(scope="session")
def trained_population(synthetic_linear):
    """Session-scoped pre-trained DASH population (M=10) for reuse across tests."""
    from dash_shap.core.pipeline import DASHPipeline
    d = synthetic_linear
    pipe = DASHPipeline(
        M=10, K=5, epsilon=0.15, delta=0.01,
        seed=42, verbose=False, n_jobs=1,
    )
    pipe.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
    return pipe


@pytest.fixture(scope="session")
def dash_result():
    """DASHResult with 4 features spanning all 4 IS-plot quadrants.

    f0: high importance, low FSI   (QI:   Robust Driver)
    f1: high importance, high FSI  (QII:  Collinear Cluster)
    f2: low importance,  low FSI   (QIII: Unimportant)
    f3: low importance,  high FSI  (QIV:  Fragile)
    """
    from dash_shap.core.result import DASHResult

    rng = np.random.default_rng(42)
    K, n_ref, P = 5, 20, 4

    shap = np.zeros((K, n_ref, P))

    # f0: consistent high SHAP across models → high importance, low FSI
    for k in range(K):
        shap[k, :, 0] = 5.0 + rng.normal(0, 0.1, n_ref)

    # f1: high but variable SHAP across models → high importance, high FSI
    for k in range(K):
        shap[k, :, 1] = rng.normal(0, 1, n_ref) * (3.0 + 4.0 * rng.random())

    # f2: consistent near-zero → low importance, low FSI
    for k in range(K):
        shap[k, :, 2] = rng.normal(0, 0.05, n_ref)

    # f3: low mean, high variance across models → low importance, high FSI
    for k in range(K):
        sign = rng.choice([-1, 1])
        shap[k, :, 3] = sign * rng.uniform(0.5, 2.0) + rng.normal(0, 0.1, n_ref)

    return DASHResult.from_shap_matrices(
        shap,
        feature_names=["f0", "f1", "f2", "f3"],
        val_scores=np.array([0.90, 0.88, 0.92, 0.85, 0.91]),
    )
