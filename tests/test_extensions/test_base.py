"""Tests for dash_shap.extensions._base utilities."""

import numpy as np

from dash_shap.core.result import DASHResult
from dash_shap.extensions._base import (
    per_model_importance,
    per_model_rankings,
    bootstrap_over_models,
)


def _make_result(K=8, n_ref=20, P=4, seed=0):
    rng = np.random.default_rng(seed)
    return DASHResult.from_shap_matrices(rng.standard_normal((K, n_ref, P)))


class TestPerModelImportance:
    def test_shape(self):
        result = _make_result(K=8, n_ref=20, P=4)
        imp = per_model_importance(result)
        assert imp.shape == (8, 4)

    def test_non_negative(self):
        result = _make_result()
        assert np.all(per_model_importance(result) >= 0)


class TestPerModelRankings:
    def test_shape(self):
        result = _make_result(K=8, n_ref=20, P=4)
        ranks = per_model_rankings(result)
        assert ranks.shape == (8, 4)

    def test_ranks_are_1_to_P(self):
        result = _make_result(K=5, n_ref=15, P=6)
        ranks = per_model_rankings(result)
        assert ranks.min() == 1
        assert ranks.max() == 6

    def test_each_row_is_permutation(self):
        result = _make_result(K=4, n_ref=10, P=5)
        ranks = per_model_rankings(result)
        for k in range(4):
            assert sorted(ranks[k]) == list(range(1, 6))

    def test_most_important_feature_ranked_1(self):
        """The feature with the highest mean |SHAP| should rank 1."""
        rng = np.random.default_rng(42)
        # Manually make feature 2 dominant
        m = rng.standard_normal((5, 20, 4)) * 0.1
        m[:, :, 2] += 10.0  # feature 2 overwhelmingly important
        result = DASHResult.from_shap_matrices(m)
        ranks = per_model_rankings(result)
        assert np.all(ranks[:, 2] == 1)


class TestBootstrapOverModels:
    def test_output_shape(self):
        result = _make_result(K=10, n_ref=20, P=4)
        stat_fn = lambda x: np.mean(np.abs(x), axis=(0, 1))  # noqa: E731
        boots = bootstrap_over_models(result, stat_fn, n_boot=50)
        assert boots.shape == (50, 4)

    def test_reproducible_with_seed(self):
        result = _make_result(K=10)
        stat_fn = lambda x: np.mean(np.abs(x), axis=(0, 1))  # noqa: E731
        b1 = bootstrap_over_models(result, stat_fn, n_boot=20, seed=0)
        b2 = bootstrap_over_models(result, stat_fn, n_boot=20, seed=0)
        np.testing.assert_array_equal(b1, b2)

    def test_different_seeds_differ(self):
        result = _make_result(K=10)
        stat_fn = lambda x: np.mean(np.abs(x), axis=(0, 1))  # noqa: E731
        b1 = bootstrap_over_models(result, stat_fn, n_boot=20, seed=0)
        b2 = bootstrap_over_models(result, stat_fn, n_boot=20, seed=1)
        assert not np.allclose(b1, b2)
