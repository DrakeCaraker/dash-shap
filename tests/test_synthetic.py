"""Tests for dash.experiments.synthetic data generators."""
import numpy as np
from dash.experiments.synthetic import (
    make_correlation_matrix,
    generate_synthetic_linear,
    generate_synthetic_nonlinear,
)


def test_correlation_matrix_identity():
    Sigma = make_correlation_matrix(P=10, group_size=5, rho=0.0)
    np.testing.assert_array_almost_equal(Sigma, np.eye(10))


def test_correlation_matrix_block():
    Sigma = make_correlation_matrix(P=10, group_size=5, rho=0.9)
    assert Sigma.shape == (10, 10)
    # Within-group off-diagonal should be 0.9
    assert abs(Sigma[0, 1] - 0.9) < 1e-10
    # Between-group should be 0
    assert abs(Sigma[0, 5]) < 1e-10


def test_generate_synthetic_linear_shapes():
    result = generate_synthetic_linear(N=200, P=10, group_size=5, rho=0.5, seed=0)
    X_train, y_train, X_val, y_val, X_explain, y_explain, X_test, y_test, groups, true_imp, meta = result

    assert X_train.shape[1] == 10
    assert len(y_train) == X_train.shape[0]
    assert len(groups) == 10
    assert len(true_imp) == 10
    assert len(np.unique(groups)) == 2  # 10 features / 5 per group = 2 groups
    assert meta["dgp"] == "linear"


def test_generate_synthetic_linear_splits_sum():
    N = 1000
    result = generate_synthetic_linear(N=N, P=10, group_size=5, seed=0)
    X_train, _, X_val, _, X_explain, _, X_test, _, _, _, _ = result
    total = X_train.shape[0] + X_val.shape[0] + X_explain.shape[0] + X_test.shape[0]
    assert total == N


def test_generate_synthetic_nonlinear_shapes():
    # Nonlinear DGP needs at least 3 groups (z1, z2, z3)
    result = generate_synthetic_nonlinear(N=200, P=20, group_size=5, rho=0.5, seed=0)
    X_train, y_train, X_val, y_val, X_explain, y_explain, X_test, y_test, groups, true_imp, meta = result

    assert X_train.shape[1] == 20
    assert len(y_train) == X_train.shape[0]
    assert len(groups) == 20
    assert len(true_imp) == 20
    assert meta["dgp"] == "nonlinear"


def test_generate_synthetic_linear_overlapping():
    result = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.9, seed=0, structure="overlapping",
    )
    X_train, _, _, _, _, _, _, _, groups, true_imp, meta = result
    assert meta["structure"] == "overlapping"
    assert X_train.shape[1] == 10
