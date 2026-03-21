"""Tests for Extension 8: Local Uncertainty."""

import numpy as np
import pytest

from dash_shap.core.result import DASHResult
from dash_shap.extensions.local import LocalResult, local_uncertainty


def _make_result(K=10, n_ref=30, P=5, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((K, n_ref, P))
    return DASHResult.from_shap_matrices(m, feature_names=[f"f{i}" for i in range(P)])


def _make_perfect_agreement_result(K=10, n_ref=30, P=5, seed=0):
    """All K models produce identical SHAP values → std=0, sign_flip_rate=0."""
    rng = np.random.default_rng(seed)
    single = rng.standard_normal((1, n_ref, P))
    m = np.repeat(single, K, axis=0)
    return DASHResult.from_shap_matrices(m, feature_names=[f"f{i}" for i in range(P)])


class TestLocalUncertainty:
    def test_returns_local_result(self):
        result = _make_result()
        lr = local_uncertainty(result, obs_idx=0)
        assert isinstance(lr, LocalResult)

    def test_shapes(self):
        result = _make_result(K=10, n_ref=30, P=5)
        lr = local_uncertainty(result, obs_idx=0)
        assert lr.mean_shap.shape == (5,)
        assert lr.std_shap.shape == (5,)
        assert lr.sign_flip_rate.shape == (5,)

    def test_std_non_negative(self):
        result = _make_result()
        lr = local_uncertainty(result, obs_idx=0)
        assert np.all(lr.std_shap >= 0.0)

    def test_sign_flip_rate_in_unit_interval(self):
        result = _make_result()
        lr = local_uncertainty(result, obs_idx=0)
        assert np.all(lr.sign_flip_rate >= 0.0)
        assert np.all(lr.sign_flip_rate <= 1.0)

    def test_perfect_agreement_gives_zero_std_and_flip_rate(self):
        """When all K models agree exactly, std=0 and sign_flip_rate=0."""
        result = _make_perfect_agreement_result()
        lr = local_uncertainty(result, obs_idx=5)
        np.testing.assert_allclose(lr.std_shap, 0.0, atol=1e-10)
        np.testing.assert_allclose(lr.sign_flip_rate, 0.0, atol=1e-10)

    def test_feature_names_preserved(self):
        result = _make_result()
        lr = local_uncertainty(result, obs_idx=0)
        assert lr.feature_names == list(result.feature_names)

    def test_obs_idx_stored(self):
        result = _make_result()
        lr = local_uncertainty(result, obs_idx=7)
        assert lr.obs_idx == 7

    def test_top_k_stored_and_clipped(self):
        result = _make_result(P=5)
        lr = local_uncertainty(result, obs_idx=0, top_k=3)
        assert lr.top_k == 3
        # top_k > P should be clipped to P
        lr2 = local_uncertainty(result, obs_idx=0, top_k=100)
        assert lr2.top_k == 5

    def test_obs_idx_out_of_range_raises(self):
        result = _make_result(n_ref=30)
        with pytest.raises((ValueError, IndexError)):
            local_uncertainty(result, obs_idx=30)
        with pytest.raises((ValueError, IndexError)):
            local_uncertainty(result, obs_idx=-1)

    def test_last_valid_obs_idx(self):
        result = _make_result(n_ref=30)
        lr = local_uncertainty(result, obs_idx=29)
        assert lr.obs_idx == 29

    def test_summary_returns_string(self):
        result = _make_result()
        lr = local_uncertainty(result, obs_idx=0)
        assert isinstance(lr.summary(), str)

    def test_mean_shap_matches_manual(self):
        result = _make_result(K=8, n_ref=20, P=4)
        obs = 3
        lr = local_uncertainty(result, obs_idx=obs)
        expected_mean = np.mean(result.all_shap_matrices[:, obs, :], axis=0)
        np.testing.assert_allclose(lr.mean_shap, expected_mean)
