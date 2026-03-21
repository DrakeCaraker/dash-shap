"""Tests for Extension 4: Feature Groups."""

import numpy as np
import pytest

from dash_shap.core.result import DASHResult
from dash_shap.extensions.groups import GroupResult, feature_groups


def _make_result(K=10, n_ref=40, P=6, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((K, n_ref, P))
    return DASHResult.from_shap_matrices(m, feature_names=[f"f{i}" for i in range(P)])


def _make_correlated_result():
    """Two perfectly correlated pairs + two independent singletons.

    f0 and f1: identical SHAP values → highly substitutable
    f2 and f3: identical SHAP values → highly substitutable
    f4, f5   : independent noise → singletons
    """
    rng = np.random.default_rng(42)
    K, n_ref, P = 10, 50, 6
    m = np.zeros((K, n_ref, P))

    base01 = rng.standard_normal((K, n_ref))
    m[:, :, 0] = base01
    m[:, :, 1] = base01  # perfect correlation with f0

    base23 = rng.standard_normal((K, n_ref))
    m[:, :, 2] = base23
    m[:, :, 3] = base23  # perfect correlation with f2

    m[:, :, 4] = rng.standard_normal((K, n_ref))
    m[:, :, 5] = rng.standard_normal((K, n_ref))

    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    return DASHResult.from_shap_matrices(m, feature_names=names)


class TestFeatureGroups:
    def test_returns_group_result(self):
        result = _make_result()
        gr = feature_groups(result)
        assert isinstance(gr, GroupResult)

    def test_n_groups_equals_len_groups(self):
        result = _make_result()
        gr = feature_groups(result)
        assert gr.n_groups == len(gr.groups)

    def test_labels_length_equals_p(self):
        result = _make_result(P=6)
        gr = feature_groups(result)
        assert gr.labels.shape == (6,)

    def test_every_feature_in_exactly_one_group(self):
        result = _make_result(P=6)
        gr = feature_groups(result)
        all_assigned = [f for group in gr.groups for f in group]
        assert sorted(all_assigned) == sorted(result.feature_names)

    def test_substitutability_matrix_shape(self):
        result = _make_result(P=6)
        gr = feature_groups(result)
        assert gr.substitutability_matrix.shape == (6, 6)

    def test_substitutability_matrix_diagonal_is_one(self):
        result = _make_result(P=6)
        gr = feature_groups(result)
        np.testing.assert_allclose(np.diag(gr.substitutability_matrix), 1.0, atol=1e-6)

    def test_substitutability_matrix_symmetric(self):
        result = _make_result(P=6)
        gr = feature_groups(result)
        np.testing.assert_allclose(
            gr.substitutability_matrix, gr.substitutability_matrix.T, atol=1e-10
        )

    def test_perfectly_correlated_pairs_grouped_together(self):
        """f0,f1 and f2,f3 must land in the same group at high threshold."""
        result = _make_correlated_result()
        gr = feature_groups(result, threshold=0.95)

        label_f0 = gr.labels[result.feature_names.index("f0")]
        label_f1 = gr.labels[result.feature_names.index("f1")]
        assert label_f0 == label_f1, "f0 and f1 (identical SHAP) must be in same group"

        label_f2 = gr.labels[result.feature_names.index("f2")]
        label_f3 = gr.labels[result.feature_names.index("f3")]
        assert label_f2 == label_f3, "f2 and f3 (identical SHAP) must be in same group"

    def test_independent_features_in_own_groups(self):
        """f4 and f5 (independent noise) must NOT share a group."""
        result = _make_correlated_result()
        gr = feature_groups(result, threshold=0.95)

        label_f4 = gr.labels[result.feature_names.index("f4")]
        label_f5 = gr.labels[result.feature_names.index("f5")]
        assert label_f4 != label_f5, "Independent features f4, f5 should be in different groups"

    def test_threshold_invalid_raises(self):
        result = _make_result()
        with pytest.raises(ValueError):
            feature_groups(result, threshold=0.0)
        with pytest.raises(ValueError):
            feature_groups(result, threshold=1.1)

    def test_unknown_method_warns(self):
        result = _make_result()
        with pytest.warns(UserWarning, match="not supported"):
            feature_groups(result, method="unknown_method")

    def test_threshold_stored(self):
        result = _make_result()
        gr = feature_groups(result, threshold=0.7)
        assert gr.threshold == 0.7

    def test_feature_names_preserved(self):
        result = _make_result()
        gr = feature_groups(result)
        assert gr.feature_names == list(result.feature_names)

    def test_summary_returns_string(self):
        result = _make_result()
        gr = feature_groups(result)
        assert isinstance(gr.summary(), str)
