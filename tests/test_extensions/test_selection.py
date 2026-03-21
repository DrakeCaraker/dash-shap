"""Tests for Extension 5: Stable Feature Selection."""

import numpy as np
import pytest

from dash_shap.core.result import DASHResult
from dash_shap.extensions.selection import SelectionResult, stable_feature_selection


def _make_result(K=10, n_ref=30, P=6, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((K, n_ref, P))
    return DASHResult.from_shap_matrices(m, feature_names=[f"f{i}" for i in range(P)])


def _make_quadrant_result():
    """Construct a result with known quadrant membership.

    QI  (high importance, low FSI):  f0
    QIV (low importance, high FSI):  f5

    f0 is set to large stable SHAP values across all K models.
    f5 is set to tiny, noisy SHAP values.
    """
    rng = np.random.default_rng(7)
    K, n_ref, P = 15, 40, 6
    m = rng.standard_normal((K, n_ref, P)) * 0.05  # tiny baseline noise

    # f0: QI — large, consistent SHAP
    m[:, :, 0] = 5.0 + rng.standard_normal((K, n_ref)) * 0.01

    # f5: QIV — small, noisy SHAP (high FSI)
    m[:, :, 5] = rng.standard_normal((K, n_ref)) * 0.001

    names = [f"f{i}" for i in range(P)]
    return DASHResult.from_shap_matrices(m, feature_names=names)


class TestStableFeatureSelection:
    def test_returns_selection_result(self):
        result = _make_result()
        sr = stable_feature_selection(result, k=3)
        assert isinstance(sr, SelectionResult)

    def test_selected_features_length(self):
        result = _make_result(P=6)
        sr = stable_feature_selection(result, k=4)
        assert len(sr.selected_features) == 4

    def test_selected_features_are_valid_names(self):
        result = _make_result(P=6)
        sr = stable_feature_selection(result, k=3)
        assert all(f in result.feature_names for f in sr.selected_features)

    def test_scores_shape(self):
        result = _make_result(P=6)
        sr = stable_feature_selection(result, k=3)
        assert sr.scores.shape == (6,)

    def test_importance_stability_ranks_shape_and_range(self):
        result = _make_result(P=6)
        sr = stable_feature_selection(result, k=3)
        assert sr.importance_ranks.shape == (6,)
        assert sr.stability_ranks.shape == (6,)
        assert set(sr.importance_ranks.tolist()) == set(range(1, 7))
        assert set(sr.stability_ranks.tolist()) == set(range(1, 7))

    def test_k_stored(self):
        result = _make_result()
        sr = stable_feature_selection(result, k=2)
        assert sr.k == 2

    def test_feature_names_preserved(self):
        result = _make_result()
        sr = stable_feature_selection(result, k=3)
        assert sr.feature_names == list(result.feature_names)

    def test_qi_feature_always_selected(self):
        """QI feature (high importance, low FSI) must appear in selection."""
        result = _make_quadrant_result()
        sr = stable_feature_selection(result, k=3)
        assert "f0" in sr.selected_features

    def test_qiv_feature_not_selected_at_small_k(self):
        """QIV feature (low importance, high FSI) should not appear at k < P."""
        result = _make_quadrant_result()
        sr = stable_feature_selection(result, k=3)
        assert "f5" not in sr.selected_features

    def test_importance_weight_1_matches_importance_order(self):
        """importance_weight=1 should select top-k by global_importance."""
        result = _make_result(P=6, seed=1)
        sr = stable_feature_selection(result, k=3, importance_weight=1.0, stability_weight=0.0)
        imp_order = np.argsort(-result.global_importance)[:3]
        expected = sorted(result.feature_names[i] for i in imp_order)
        assert sorted(sr.selected_features) == expected

    def test_stability_weight_1_matches_fsi_order(self):
        """stability_weight=1 should select features with lowest FSI."""
        result = _make_result(P=6, seed=2)
        sr = stable_feature_selection(result, k=3, importance_weight=0.0, stability_weight=1.0)
        fsi_order = np.argsort(result.fsi)[:3]
        expected = sorted(result.feature_names[i] for i in fsi_order)
        assert sorted(sr.selected_features) == expected

    def test_k_greater_than_p_raises(self):
        result = _make_result(P=4)
        with pytest.raises(ValueError):
            stable_feature_selection(result, k=5)

    def test_k_zero_raises(self):
        result = _make_result()
        with pytest.raises(ValueError):
            stable_feature_selection(result, k=0)

    def test_k_equals_1_returns_single_feature(self):
        result = _make_result()
        sr = stable_feature_selection(result, k=1)
        assert len(sr.selected_features) == 1

    def test_unbalanced_weights_warns(self):
        result = _make_result()
        with pytest.warns(UserWarning, match="≠ 1.0"):
            stable_feature_selection(result, k=2, importance_weight=0.5, stability_weight=0.5 + 1e-5)

    def test_summary_returns_string(self):
        result = _make_result()
        sr = stable_feature_selection(result, k=3)
        assert isinstance(sr.summary(), str)

    def test_plot_pareto_returns_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = _make_result()
        sr = stable_feature_selection(result, k=3)
        fig = sr.plot_pareto()
        assert isinstance(fig, plt.Figure)
        plt.close("all")
