"""Tests for Extension: Theory Bridge — impossibility-theorem-grounded diagnostics."""

import math

import numpy as np
import pytest

from dash_shap.extensions.theory_bridge import (
    TheoryBridgeResult,
    compute_snr,
    divergence_ratio,
    predict_flip_rate,
    recommend_M,
    theory_bridge,
)


# ── Standalone function tests ────────────────────────────────────────────


class TestComputeSNR:
    def test_symmetric_features_low_snr(self):
        """Symmetric features should have SNR near 0."""
        rng = np.random.default_rng(42)
        M, P = 100, 2
        # Both features drawn from same distribution
        imp = rng.standard_normal((M, P)) + 1.0
        snr = compute_snr(imp)
        # SNR should be low (mean diff ≈ 0, std > 0)
        assert snr[(0, 1)] < 0.5

    def test_identical_features_snr_zero(self):
        """Identical features should have SNR=0 (coin flip), not inf."""
        imp = np.ones((20, 3))
        snr = compute_snr(imp)
        assert snr[(0, 1)] == 0.0
        assert snr[(0, 2)] == 0.0

    def test_asymmetric_features_high_snr(self):
        """One dominant feature should produce high SNR."""
        rng = np.random.default_rng(42)
        M = 50
        imp = np.column_stack(
            [
                rng.normal(5.0, 0.1, M),  # feature 0: strong
                rng.normal(1.0, 0.1, M),  # feature 1: weak
            ]
        )
        snr = compute_snr(imp)
        assert snr[(0, 1)] > 10.0

    def test_returns_all_pairs(self):
        """Should return C(P, 2) pairs."""
        imp = np.ones((10, 5))
        snr = compute_snr(imp)
        assert len(snr) == 10  # C(5, 2) = 10

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            compute_snr(np.ones(10))


class TestPredictFlipRate:
    def test_snr_zero_gives_half(self):
        """Φ(0) = 0.5 — coin flip for indistinguishable features."""
        assert predict_flip_rate(0.0) == pytest.approx(0.5)

    def test_snr_large_gives_near_zero(self):
        """High SNR → very low flip rate."""
        assert predict_flip_rate(3.0) < 0.002

    def test_monotone_decreasing(self):
        """Flip rate should decrease as SNR increases."""
        rates = [predict_flip_rate(s) for s in [0.0, 0.5, 1.0, 2.0, 3.0]]
        for i in range(len(rates) - 1):
            assert rates[i] > rates[i + 1]

    def test_rejects_negative_snr(self):
        with pytest.raises(ValueError, match="non-negative"):
            predict_flip_rate(-1.0)

    def test_known_values(self):
        """Check against scipy.stats.norm.cdf(-1.645) ≈ 0.05."""
        assert predict_flip_rate(1.645) == pytest.approx(0.05, abs=0.001)


class TestRecommendM:
    def test_increases_with_variance(self):
        """Higher variance → larger M recommendation."""
        rng = np.random.default_rng(42)
        M = 30
        imp_low_var = np.column_stack(
            [
                rng.normal(5.0, 0.1, M),
                rng.normal(4.0, 0.1, M),
            ]
        )
        imp_high_var = np.column_stack(
            [
                rng.normal(5.0, 1.0, M),
                rng.normal(4.0, 1.0, M),
            ]
        )
        rec_low = recommend_M(imp_low_var)
        rec_high = recommend_M(imp_high_var)
        assert rec_high["recommended_M"] >= rec_low["recommended_M"]

    def test_returns_required_keys(self):
        imp = np.random.default_rng(0).standard_normal((20, 4)) + 1.0
        rec = recommend_M(imp)
        assert "recommended_M" in rec
        assert "worst_pair" in rec
        assert "worst_pair_snr" in rec
        assert "worst_pair_flip" in rec
        assert "z_critical" in rec
        assert "note" in rec

    def test_respects_min_max(self):
        imp = np.random.default_rng(0).standard_normal((20, 4))
        rec = recommend_M(imp, min_M=50, max_M=100)
        assert 50 <= rec["recommended_M"] <= 100

    def test_indistinguishable_features_returns_max(self):
        """Indistinguishable features (delta≈0) should recommend max_M."""
        imp = np.ones((20, 3)) + np.random.default_rng(0).normal(0, 1e-15, (20, 3))
        rec = recommend_M(imp, max_M=500)
        assert rec["recommended_M"] == 500

    def test_rejects_invalid_target(self):
        imp = np.ones((10, 2))
        with pytest.raises(ValueError, match="target_flip_rate"):
            recommend_M(imp, target_flip_rate=0.0)
        with pytest.raises(ValueError, match="target_flip_rate"):
            recommend_M(imp, target_flip_rate=0.5)


class TestDivergenceRatio:
    def test_zero_correlation(self):
        assert divergence_ratio(0.0) == 1.0

    def test_moderate_correlation(self):
        assert divergence_ratio(0.5) == pytest.approx(1.0 / 0.75)

    def test_high_correlation(self):
        assert divergence_ratio(0.9) == pytest.approx(1.0 / 0.19, rel=0.01)

    def test_increases_with_rho(self):
        """Monotonically increasing in |ρ|."""
        vals = [divergence_ratio(r) for r in [0.0, 0.3, 0.5, 0.7, 0.9]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_perfect_correlation_is_inf(self):
        assert divergence_ratio(1.0) == float("inf")
        assert divergence_ratio(-1.0) == float("inf")

    def test_negative_correlation(self):
        """Ratio depends on ρ², so sign doesn't matter."""
        assert divergence_ratio(0.5) == divergence_ratio(-0.5)


# ── DASHResult extension test ────────────────────────────────────────────


class TestTheoryBridge:
    def test_returns_result(self, dash_result):
        tb = theory_bridge(dash_result)
        assert isinstance(tb, TheoryBridgeResult)

    def test_snr_dict_populated(self, dash_result):
        tb = theory_bridge(dash_result)
        P = dash_result.P
        expected_pairs = P * (P - 1) // 2
        assert len(tb.snr) == expected_pairs

    def test_predicted_flip_rates_match_snr(self, dash_result):
        """Each predicted flip rate should be Φ(-SNR) of its SNR."""
        tb = theory_bridge(dash_result)
        for pair, flip in tb.predicted_flip_rates.items():
            expected = predict_flip_rate(tb.snr[pair])
            assert flip == pytest.approx(expected)

    def test_recommended_M_positive(self, dash_result):
        tb = theory_bridge(dash_result)
        assert tb.recommended_M >= 10

    def test_unstable_pairs_above_threshold(self, dash_result):
        tb = theory_bridge(dash_result, unstable_threshold=0.10)
        for j, k in tb.unstable_pairs:
            assert tb.predicted_flip_rates[(j, k)] > 0.10

    def test_summary_returns_string(self, dash_result):
        tb = theory_bridge(dash_result)
        s = tb.summary()
        assert isinstance(s, str)
        assert "TheoryBridgeResult" in s

    def test_plot_returns_figure(self, dash_result):
        import matplotlib

        matplotlib.use("Agg")

        tb = theory_bridge(dash_result)
        fig = tb.plot()
        assert fig is not None

    def test_feature_names_preserved(self, dash_result):
        tb = theory_bridge(dash_result)
        assert tb.feature_names == list(dash_result.feature_names)
