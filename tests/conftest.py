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
