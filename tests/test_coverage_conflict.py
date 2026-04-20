"""Tests for coverage_conflict and compare_flip_predictors diagnostics."""

import numpy as np
from dash_shap.core.diagnostics import (
    coverage_conflict,
    compare_flip_predictors,
    predict_sign_instability,
    has_coverage_conflict,
)


def test_coverage_conflict_unanimous():
    """All models agree on sign → no conflict, minority fraction = 0."""
    shap = np.array([[[1.0, -2.0], [0.5, -1.0]]] * 5)  # (5, 2, 2)
    result = coverage_conflict(shap)
    assert result["minority_fraction"].shape == (2, 2)
    np.testing.assert_array_equal(result["minority_fraction"], 0.0)
    np.testing.assert_array_equal(result["has_conflict"], False)
    np.testing.assert_array_equal(result["feature_conflict_rate"], 0.0)


def test_coverage_conflict_split():
    """50/50 split → minority fraction = 0.5."""
    # 4 models: 2 positive, 2 negative for feature 0
    shap = np.zeros((4, 1, 2))
    shap[0, 0, 0] = 1.0
    shap[1, 0, 0] = 1.0
    shap[2, 0, 0] = -1.0
    shap[3, 0, 0] = -1.0
    # Feature 1: all positive
    shap[:, 0, 1] = 1.0
    result = coverage_conflict(shap)
    assert result["minority_fraction"][0, 0] == 0.5
    assert result["minority_fraction"][0, 1] == 0.0
    assert result["has_conflict"][0, 0] is np.True_
    assert result["has_conflict"][0, 1] is np.False_


def test_coverage_conflict_asymmetric():
    """3 positive, 1 negative → minority fraction = 0.25."""
    shap = np.zeros((4, 1, 1))
    shap[0, 0, 0] = 1.0
    shap[1, 0, 0] = 2.0
    shap[2, 0, 0] = 0.5
    shap[3, 0, 0] = -1.0
    result = coverage_conflict(shap)
    assert abs(result["minority_fraction"][0, 0] - 0.25) < 1e-10


def test_coverage_conflict_zeros_ignored():
    """Exact zero SHAP values are excluded from sign counting."""
    shap = np.zeros((5, 1, 1))
    shap[0, 0, 0] = 1.0
    shap[1, 0, 0] = -1.0
    # Models 2-4 have exactly 0 → excluded
    result = coverage_conflict(shap)
    assert result["minority_fraction"][0, 0] == 0.5  # 1 pos, 1 neg out of 2


def test_coverage_conflict_feature_level():
    """Feature-level aggregation works correctly."""
    rng = np.random.RandomState(42)
    # Feature 0: always positive (no conflict expected)
    # Feature 1: mixed signs (conflict expected)
    shap = np.zeros((10, 20, 2))
    shap[:, :, 0] = np.abs(rng.randn(10, 20))
    shap[:, :, 1] = rng.randn(10, 20)
    result = coverage_conflict(shap)
    assert result["feature_conflict_rate"][0] == 0.0
    assert result["feature_conflict_rate"][1] > 0.5


def test_compare_flip_predictors_shape():
    """compare_flip_predictors returns correct shapes."""
    rng = np.random.RandomState(42)
    shap = rng.randn(10, 20, 5)
    result = compare_flip_predictors(shap)
    assert result["cc_prediction"].shape == (5,)
    assert result["gf_prediction"].shape == (5,)
    assert result["cc_conflict_rate"].shape == (5,)


def test_compare_flip_predictors_with_importance():
    """Passing explicit importance_matrix works."""
    rng = np.random.RandomState(42)
    shap = rng.randn(10, 20, 3)
    imp = np.mean(np.abs(shap), axis=1)
    result = compare_flip_predictors(shap, importance_matrix=imp)
    assert result["gf_prediction"].shape == (3,)
    assert all(0 <= v <= 0.5 for v in result["gf_prediction"])


def test_compare_flip_predictors_stable_feature():
    """A feature with unanimous signs has low cc_prediction and low gf_prediction."""
    shap = np.zeros((20, 10, 2))
    shap[:, :, 0] = np.abs(np.random.RandomState(42).randn(20, 10)) + 0.1  # always positive
    shap[:, :, 1] = np.random.RandomState(43).randn(20, 10)  # mixed
    result = compare_flip_predictors(shap)
    assert result["cc_prediction"][0] < result["cc_prediction"][1]


def test_predict_sign_instability_basic():
    """Mixed signs → some features flagged as unstable."""
    rng = np.random.RandomState(42)
    shap = np.zeros((20, 10, 3))
    shap[:, :, 0] = np.abs(rng.randn(20, 10)) + 0.1  # always positive → stable
    shap[:, :, 1] = rng.randn(20, 10)  # mixed → unstable
    shap[:, :, 2] = -(np.abs(rng.randn(20, 10)) + 0.1)  # always negative → stable
    result = predict_sign_instability(shap, threshold=0.1)
    assert result["unstable"][1] is np.True_  # feature 1 is unstable
    assert result["unstable"][0] is np.False_  # feature 0 is stable
    assert result["unstable"][2] is np.False_  # feature 2 is stable
    assert result["n_unstable"] == 1


def test_predict_sign_instability_threshold():
    """Higher threshold → fewer features flagged."""
    rng = np.random.RandomState(42)
    shap = rng.randn(20, 10, 5)
    low = predict_sign_instability(shap, threshold=0.05)
    high = predict_sign_instability(shap, threshold=0.4)
    assert low["n_unstable"] >= high["n_unstable"]


def test_has_coverage_conflict_true():
    """Feature with both positive and negative SHAP → has conflict."""
    shap = np.zeros((4, 2, 1))
    shap[0, 0, 0] = 1.0
    shap[1, 0, 0] = -1.0
    shap[2, 0, 0] = 1.0
    shap[3, 0, 0] = 1.0
    assert has_coverage_conflict(shap, 0) is True


def test_has_coverage_conflict_false():
    """Feature with only positive SHAP → no conflict."""
    shap = np.ones((5, 3, 2))
    assert has_coverage_conflict(shap, 0) is False
    assert has_coverage_conflict(shap, 1) is False
