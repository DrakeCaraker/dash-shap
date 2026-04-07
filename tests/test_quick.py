"""Tests for the quick check() API."""

import numpy as np
import pytest
from dash_shap import check


def _make_correlated_data(n=500, rho=0.9, seed=42):
    """Simple correlated data for testing."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n)
    A = z + rng.randn(n) * np.sqrt(1 - rho**2)
    B = rho * z + rng.randn(n) * np.sqrt(1 - rho**2)
    C = rng.randn(n)  # independent noise
    X = np.column_stack([A, B, C])
    y = A + 0.5 * rng.randn(n)
    return X, y


def test_check_returns_result():
    X, y = _make_correlated_data()
    result = check(X, y, M=5, verbose=False)
    assert result.n_models == 5
    assert result.n_features == 3
    assert len(result.consensus_importance) == 3
    assert len(result.fsi) == 3


def test_check_detects_correlation():
    X, y = _make_correlated_data(rho=0.95)
    result = check(X, y, M=10, correlation_threshold=0.5, verbose=False)
    assert len(result.correlated_groups) >= 1
    # Features 0 and 1 should be in a correlated group
    group_members = set()
    for g in result.correlated_groups:
        group_members.update(g)
    assert 0 in group_members and 1 in group_members


def test_check_report_is_string():
    X, y = _make_correlated_data()
    result = check(X, y, M=5, verbose=False)
    report = result.report()
    assert isinstance(report, str)
    assert "DASH Stability Check" in report
    assert "Models trained: 5" in report


def test_check_dash_importance():
    X, y = _make_correlated_data()
    result = check(X, y, M=5, verbose=False)
    imp = result.dash_importance()
    assert isinstance(imp, dict)
    assert len(imp) == 3


def test_check_to_dataframe():
    X, y = _make_correlated_data()
    result = check(X, y, M=5, verbose=False)
    df = result.to_dataframe()
    assert len(df) == 3
    assert "importance" in df.columns
    assert "fsi" in df.columns
    assert "stable" in df.columns


def test_check_auto_task_detection():
    # Regression
    X, y = _make_correlated_data()
    result = check(X, y, M=3, verbose=False)
    assert result._task == "regression"

    # Binary classification
    y_binary = (y > np.median(y)).astype(int)
    result = check(X, y_binary, M=3, verbose=False)
    assert result._task == "binary"


def test_check_with_feature_names():
    X, y = _make_correlated_data()
    result = check(X, y, M=5, feature_names=["A", "B", "noise"], verbose=False)
    report = result.report()
    assert "A" in report or "B" in report or "noise" in report
    imp = result.dash_importance()
    assert "A" in imp
