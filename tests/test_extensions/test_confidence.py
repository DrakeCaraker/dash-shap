"""Tests for Extension 1: Confidence Intervals."""
import numpy as np
import pytest

from dash_shap.core.result import DASHResult
from dash_shap.extensions.confidence import confidence_intervals, ConfidenceResult


def _make_result(K=12, n_ref=30, P=4, seed=0):
    rng = np.random.default_rng(seed)
    return DASHResult.from_shap_matrices(rng.standard_normal((K, n_ref, P)))


class TestConfidenceIntervals:
    def test_returns_confidence_result(self, dash_result):
        ci = confidence_intervals(dash_result, n_boot=50)
        assert isinstance(ci, ConfidenceResult)

    def test_shapes(self, dash_result):
        P = dash_result.P
        ci = confidence_intervals(dash_result, n_boot=50)
        assert ci.importance_ci.shape == (P, 3)
        assert ci.fsi_ci.shape == (P, 3)
        assert ci.ranking_ci.shape == (P, 3)

    def test_interval_contains_point_estimate(self, dash_result):
        """Lower ≤ point estimate ≤ upper for all features."""
        ci = confidence_intervals(dash_result, n_boot=100)
        assert np.all(ci.importance_ci[:, 0] <= ci.importance_ci[:, 1] + 1e-10)
        assert np.all(ci.importance_ci[:, 1] <= ci.importance_ci[:, 2] + 1e-10)
        assert np.all(ci.fsi_ci[:, 0] <= ci.fsi_ci[:, 1] + 1e-10)
        assert np.all(ci.fsi_ci[:, 1] <= ci.fsi_ci[:, 2] + 1e-10)

    def test_wider_at_lower_alpha(self, dash_result):
        """90% CI should be narrower than 50% CI (i.e., lower alpha = wider interval)."""
        ci_narrow = confidence_intervals(dash_result, alpha=0.1, n_boot=200)
        ci_wide = confidence_intervals(dash_result, alpha=0.5, n_boot=200)
        width_narrow = ci_narrow.importance_ci[:, 2] - ci_narrow.importance_ci[:, 0]
        width_wide = ci_wide.importance_ci[:, 2] - ci_wide.importance_ci[:, 0]
        # On average, 90% CI is wider than 50% CI
        assert np.mean(width_narrow) >= np.mean(width_wide) - 1e-10

    def test_point_estimates_match_result(self, dash_result):
        """Point estimates (column 1) match result.global_importance."""
        ci = confidence_intervals(dash_result, n_boot=50)
        np.testing.assert_allclose(ci.importance_ci[:, 1], dash_result.global_importance)

    def test_feature_names_preserved(self, dash_result):
        ci = confidence_intervals(dash_result, n_boot=50)
        assert ci.feature_names == list(dash_result.feature_names)

    def test_warns_when_k_small(self):
        rng = np.random.default_rng(0)
        small_result = DASHResult.from_shap_matrices(rng.standard_normal((5, 20, 3)))
        with pytest.warns(UserWarning, match="K=5 < 10"):
            confidence_intervals(small_result, n_boot=20)

    def test_summary_returns_string(self, dash_result):
        ci = confidence_intervals(dash_result, n_boot=50)
        s = ci.summary()
        assert isinstance(s, str)
        assert "ConfidenceResult" in s

    def test_ranking_ci_values_are_float(self, dash_result):
        """ranking_ci stores float values, not integers."""
        ci = confidence_intervals(dash_result, n_boot=50)
        assert ci.ranking_ci.dtype in (np.float32, np.float64)
