"""Smoke tests for DASHPipeline and core modules."""
import numpy as np
import pytest
from dash.core.population import sample_configurations, DEFAULT_SEARCH_SPACE
from dash.core.filtering import performance_filter
from dash.core.diagnostics import compute_diagnostics
from dash.utils.shap_helpers import compute_global_importance


def test_sample_configurations_count():
    configs = sample_configurations(DEFAULT_SEARCH_SPACE, M=10, seed=42)
    assert len(configs) == 10
    assert all(isinstance(c, dict) for c in configs)


def test_sample_configurations_grid():
    small_space = {"a": [1, 2], "b": [3, 4]}
    configs = sample_configurations(small_space, M=10, seed=42, strategy="grid")
    assert len(configs) == 4  # 2x2 = 4 total combos


def test_performance_filter_basic():
    scores = {0: 0.90, 1: 0.89, 2: 0.80, 3: 0.85}
    filtered = performance_filter(scores, epsilon=0.02, verbose=False)
    assert 0 in filtered
    assert 1 in filtered
    assert 2 not in filtered


def test_compute_diagnostics_shapes():
    K, N, P = 3, 10, 5
    all_shap = np.random.randn(K, N, P)
    consensus, variance, fsi, importance = compute_diagnostics(all_shap)
    assert consensus.shape == (N, P)
    assert variance.shape == (N, P)
    assert fsi.shape == (P,)
    assert importance.shape == (P,)


def test_compute_global_importance_2d():
    sv = np.array([[1.0, -2.0], [3.0, -1.0]])
    imp = compute_global_importance(sv)
    np.testing.assert_array_almost_equal(imp, [2.0, 1.5])


def test_compute_global_importance_list():
    sv = [
        np.array([[1.0, -2.0], [3.0, -1.0]]),
        np.array([[2.0, -1.0], [1.0, -3.0]]),
    ]
    imp = compute_global_importance(sv)
    assert imp.shape == (2,)
    assert all(imp > 0)
