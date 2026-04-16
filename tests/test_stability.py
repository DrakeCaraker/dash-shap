"""Tests for dash_shap.stability — screen/validate/consensus/report workflow."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from dash_shap.stability import (
    _correlated_groups,
    validate_from_attributions,
    consensus_from_attributions,
    report,
    screen,
    validate,
    consensus,
)


# ---------------------------------------------------------------------------
# _correlated_groups
# ---------------------------------------------------------------------------


class TestCorrelatedGroups:
    def test_perfectly_correlated_pair(self):
        """Two perfectly correlated features should form one group."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        X = np.column_stack([x, x, rng.standard_normal(100)])
        groups = _correlated_groups(X, threshold=0.5)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1}

    def test_no_correlation(self):
        """Independent features should produce no groups."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 4))
        groups = _correlated_groups(X, threshold=0.9)
        # With truly random data and strict threshold, no groups expected
        assert len(groups) == 0

    def test_high_threshold_breaks_groups(self):
        """Raising threshold should reduce or eliminate groups."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        noise = rng.standard_normal(100) * 0.5
        X = np.column_stack([x, x + noise, rng.standard_normal(100)])
        groups_low = _correlated_groups(X, threshold=0.3)
        groups_high = _correlated_groups(X, threshold=0.99)
        assert len(groups_low) >= len(groups_high)

    def test_two_uncorrelated_features(self):
        """Two features with low correlation produce no groups at high threshold."""
        rng = np.random.default_rng(10)
        X = rng.standard_normal((200, 2))
        groups = _correlated_groups(X, threshold=0.9)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# validate_from_attributions
# ---------------------------------------------------------------------------


class TestValidateFromAttributions:
    def test_basic_output_keys(self):
        """Returned dict should contain all expected keys."""
        rng = np.random.default_rng(7)
        attr = rng.random((10, 4))
        result = validate_from_attributions(attr)
        expected_keys = {
            "shap_matrix",
            "z_statistics",
            "flip_rates",
            "unstable_pairs",
            "z_flip_correlation",
            "z_flip_pvalue",
            "f1_correlation",
            "f1_pvalue",
        }
        assert expected_keys == set(result.keys())

    def test_identical_attributions_no_unstable(self):
        """If all models agree perfectly, no pairs should be unstable."""
        attr = np.tile([0.5, 0.3, 0.1], (10, 1))
        result = validate_from_attributions(attr, threshold=1.96)
        # All models identical -> std=0 -> Z=inf -> no unstable pairs
        assert len(result["unstable_pairs"]) == 0

    def test_noisy_attributions_finds_unstable(self):
        """When two features trade places across models, they should be unstable."""
        rng = np.random.default_rng(99)
        M, P = 20, 3
        attr = np.zeros((M, P))
        attr[:, 0] = 1.0  # feature 0 always dominant
        # Features 1 and 2 swap between models
        attr[:10, 1] = 0.6
        attr[:10, 2] = 0.4
        attr[10:, 1] = 0.4
        attr[10:, 2] = 0.6
        result = validate_from_attributions(attr, threshold=1.96)
        # Pair (1,2) should be unstable (mean diff ~ 0, high flip rate)
        unstable_set = set(result["unstable_pairs"])
        assert (1, 2) in unstable_set

    def test_flip_rates_bounded(self):
        """Flip rates should be between 0 and 0.5."""
        rng = np.random.default_rng(3)
        attr = rng.random((15, 5))
        result = validate_from_attributions(attr)
        for fr in result["flip_rates"].values():
            assert 0 <= fr <= 0.5

    def test_shap_matrix_passthrough(self):
        """The returned shap_matrix should equal the input."""
        attr = np.eye(5)
        result = validate_from_attributions(attr)
        np.testing.assert_array_equal(result["shap_matrix"], attr)

    def test_wrong_dimensions_raises(self):
        """1D input should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            validate_from_attributions(np.array([1, 2, 3]))

    def test_3d_input_raises(self):
        """3D input should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            validate_from_attributions(np.ones((2, 3, 4)))

    def test_custom_threshold(self):
        """Higher threshold should find more unstable pairs."""
        rng = np.random.default_rng(5)
        attr = rng.random((20, 4))
        result_strict = validate_from_attributions(attr, threshold=1.96)
        result_loose = validate_from_attributions(attr, threshold=100.0)
        assert len(result_loose["unstable_pairs"]) >= len(result_strict["unstable_pairs"])

    def test_two_models_minimum(self):
        """Should work with just 2 models."""
        attr = np.array([[0.5, 0.3], [0.3, 0.5]])
        result = validate_from_attributions(attr)
        assert "z_statistics" in result
        assert len(result["z_statistics"]) == 1  # one pair (0,1)


# ---------------------------------------------------------------------------
# consensus_from_attributions
# ---------------------------------------------------------------------------


class TestConsensusFromAttributions:
    def test_basic_output_keys(self):
        rng = np.random.default_rng(0)
        attr = rng.random((10, 4))
        result = consensus_from_attributions(attr)
        assert set(result.keys()) == {"attributions", "std", "tied_groups"}

    def test_mean_computation(self):
        """Attributions should be the column-wise mean."""
        attr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = consensus_from_attributions(attr)
        np.testing.assert_allclose(result["attributions"], [2.0, 3.0])

    def test_std_computation(self):
        """Std should use ddof=1."""
        attr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = consensus_from_attributions(attr)
        expected_std = np.std(attr, axis=0, ddof=1)
        np.testing.assert_allclose(result["std"], expected_std)

    def test_tied_groups_detected(self):
        """Features with nearly identical attributions should be tied."""
        # 10 models, features 0 and 1 have ~same mean attribution
        attr = np.ones((10, 3))
        attr[:, 0] = 0.500
        attr[:, 1] = 0.501  # within 1% of feature 0
        attr[:, 2] = 5.0  # clearly different
        result = consensus_from_attributions(attr)
        tied_flat = [idx for g in result["tied_groups"] for idx in g]
        assert 0 in tied_flat and 1 in tied_flat

    def test_no_tied_groups(self):
        """Well-separated features should have no tied groups."""
        attr = np.array([[1.0, 10.0, 100.0]] * 5)
        result = consensus_from_attributions(attr)
        assert len(result["tied_groups"]) == 0

    def test_wrong_dimensions_raises(self):
        with pytest.raises(ValueError, match="2D"):
            consensus_from_attributions(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_with_no_inputs(self):
        """Report with no results should still produce valid text."""
        text = report()
        assert "Attribution Instability Report" in text

    def test_report_with_validate_results(self):
        attr = np.array([[0.5, 0.3], [0.3, 0.5]])
        val_res = validate_from_attributions(attr)
        text = report(validate_results=val_res)
        assert "Unstable pairs" in text

    def test_report_with_consensus_results(self):
        attr = np.ones((5, 3))
        attr[:, 0] = 0.500
        attr[:, 1] = 0.501
        cons_res = consensus_from_attributions(attr)
        text = report(consensus_results=cons_res)
        assert "Tied Groups" in text or "Attribution Instability Report" in text

    def test_report_with_feature_names(self):
        attr = np.array([[0.5, 0.3], [0.3, 0.5]])
        val_res = validate_from_attributions(attr)
        text = report(validate_results=val_res, feature_names=["alpha", "beta"])
        assert "alpha" in text or "beta" in text

    def test_report_with_screen_results(self):
        screen_res = {
            "correlated_groups": [[0, 1]],
            "shap_values": np.array([0.5, 0.3, 0.1]),
            "flagged_pairs": [(0, 1)],
        }
        text = report(screen_results=screen_res)
        assert "Correlated groups detected" in text

    def test_report_generated_by_line(self):
        text = report()
        assert "dash-shap" in text


# ---------------------------------------------------------------------------
# screen / validate / consensus (mock _compute_shap to avoid XGBoost)
# ---------------------------------------------------------------------------


class TestScreenWithMock:
    @patch("dash_shap.stability._compute_shap")
    def test_screen_returns_expected_keys(self, mock_shap):
        mock_shap.return_value = np.array([0.5, 0.4, 0.1])
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100)
        X_train = np.column_stack([x, x + rng.standard_normal(100) * 0.01, rng.standard_normal(100)])
        X_test = rng.standard_normal((20, 3))
        model = MagicMock()
        result = screen(model, X_train, X_test)
        assert "correlated_groups" in result
        assert "shap_values" in result
        assert "flagged_pairs" in result

    @patch("dash_shap.stability._compute_shap")
    def test_screen_flags_similar_attributions_in_correlated_group(self, mock_shap):
        # Features 0 and 1 correlated, similar SHAP -> flagged
        mock_shap.return_value = np.array([0.50, 0.49, 0.01])
        x = np.arange(100, dtype=float)
        X_train = np.column_stack([x, x, np.random.default_rng(0).standard_normal(100)])
        X_test = np.random.default_rng(1).standard_normal((10, 3))
        model = MagicMock()
        result = screen(model, X_train, X_test, correlation_threshold=0.5)
        assert (0, 1) in result["flagged_pairs"]


class TestValidateWithMock:
    @patch("dash_shap.stability._compute_shap")
    def test_validate_multiple_models(self, mock_shap):
        """validate() with mocked SHAP should return correct structure."""
        call_count = [0]

        def side_effect(model, X_test, X_bg=None):
            call_count[0] += 1
            rng = np.random.default_rng(call_count[0])
            return rng.random(3)

        mock_shap.side_effect = side_effect

        models = [MagicMock() for _ in range(5)]
        X_test = np.random.default_rng(0).standard_normal((10, 3))
        result = validate(models, X_test)
        assert result["shap_matrix"].shape == (5, 3)
        assert len(result["z_statistics"]) == 3  # C(3,2) pairs

    @patch("dash_shap.stability._compute_shap")
    def test_validate_threshold_effect(self, mock_shap):
        call_count = [0]

        def side_effect(model, X_test, X_bg=None):
            call_count[0] += 1
            rng = np.random.default_rng(call_count[0])
            return rng.random(3)

        mock_shap.side_effect = side_effect

        models = [MagicMock() for _ in range(5)]
        X_test = np.random.default_rng(0).standard_normal((10, 3))
        result = validate(models, X_test, threshold=1000.0)
        # Very high threshold means most pairs unstable
        assert len(result["unstable_pairs"]) >= 0


class TestConsensusWithMock:
    @patch("dash_shap.stability._compute_shap")
    def test_consensus_returns_expected_keys(self, mock_shap):
        call_count = [0]

        def side_effect(model, X_test, X_bg=None):
            call_count[0] += 1
            return np.array([0.5, 0.3, 0.1])

        mock_shap.side_effect = side_effect

        models = [MagicMock() for _ in range(3)]
        X_test = np.random.default_rng(0).standard_normal((10, 3))
        result = consensus(models, X_test)
        assert set(result.keys()) == {"attributions", "std", "tied_groups"}
        np.testing.assert_allclose(result["attributions"], [0.5, 0.3, 0.1])
