"""Tests for confidence intervals extension."""

import numpy as np
import pytest

from dash_shap.extensions.confidence import confidence_intervals


class TestConfidenceIntervals:
    def test_ci_contains_point_estimate(self, dash_result):
        cr = confidence_intervals(dash_result, alpha=0.05, n_boot=500, seed=42)
        for j in range(dash_result.P):
            lo, pt, hi = cr.importance_ci[j]
            assert lo <= pt <= hi, f"f{j}: {lo} <= {pt} <= {hi}"

    def test_fsi_ci_contains_point(self, dash_result):
        cr = confidence_intervals(dash_result, alpha=0.05, n_boot=500, seed=42)
        for j in range(dash_result.P):
            lo, pt, hi = cr.fsi_ci[j]
            assert lo - 1e-12 <= pt <= hi + 1e-12

    def test_robust_driver_tighter_than_collinear(self, dash_result):
        cr = confidence_intervals(dash_result, alpha=0.05, n_boot=500, seed=42)
        width_f0 = cr.importance_ci[0, 2] - cr.importance_ci[0, 0]
        width_f1 = cr.importance_ci[1, 2] - cr.importance_ci[1, 0]
        assert width_f0 < width_f1, (
            f"f0 (robust) width {width_f0:.4f} should be < f1 (collinear) {width_f1:.4f}"
        )

    def test_deterministic_with_seed(self, dash_result):
        cr1 = confidence_intervals(dash_result, seed=123, n_boot=200)
        cr2 = confidence_intervals(dash_result, seed=123, n_boot=200)
        np.testing.assert_array_equal(cr1.importance_ci, cr2.importance_ci)

    def test_shapes(self, dash_result):
        cr = confidence_intervals(dash_result, n_boot=100)
        assert cr.importance_ci.shape == (dash_result.P, 3)
        assert cr.fsi_ci.shape == (dash_result.P, 3)
        assert cr.ranking_ci.shape == (dash_result.P, 3)

    def test_summary_returns_string(self, dash_result):
        cr = confidence_intervals(dash_result, n_boot=100)
        s = cr.summary()
        assert isinstance(s, str)
        assert "f0" in s
