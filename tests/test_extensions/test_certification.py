"""Tests for Extension 9: Robust Certification."""

import numpy as np

from dash_shap.core.result import DASHResult
from dash_shap.extensions.certification import robust_certification, CertificationResult


def _make_stable_result(seed=0):
    """All K models agree: f0 rank 1, f1 rank 2, f2 rank 3, f3 rank 4."""
    rng = np.random.default_rng(seed)
    K, n_ref, P = 12, 30, 4
    base = np.array([4.0, 3.0, 2.0, 1.0])  # descending importance
    noise = rng.standard_normal((K, n_ref, P)) * 0.02
    m = base[np.newaxis, np.newaxis, :] + noise
    return DASHResult.from_shap_matrices(m, feature_names=["rank1", "rank2", "rank3", "rank4"])


class TestRobustCertification:
    def test_returns_certification_result(self, dash_result):
        cert = robust_certification(dash_result)
        assert isinstance(cert, CertificationResult)

    def test_max_ranks_shape(self, dash_result):
        cert = robust_certification(dash_result)
        assert cert.max_ranks.shape == (dash_result.P,)

    def test_max_ranks_at_least_1(self, dash_result):
        cert = robust_certification(dash_result)
        assert np.all(cert.max_ranks >= 1)

    def test_monotone_in_k(self, dash_result):
        """certified top-k ⊆ certified top-(k+1)."""
        cert = robust_certification(dash_result, k_values=[1, 2, 3, 4])
        for k1, k2 in zip([1, 2, 3], [2, 3, 4]):
            set_k1 = set(cert.certified[k1])
            set_k2 = set(cert.certified[k2])
            assert set_k1.issubset(set_k2), f"top-{k1} {set_k1} should be ⊆ top-{k2} {set_k2}"

    def test_stable_result_certifies_correctly(self):
        """With all K models agreeing on rank order, f0 should be top-1."""
        result = _make_stable_result()
        cert = robust_certification(result, k_values=[1, 2, 3, 4])
        assert "rank1" in cert.certified[1]
        assert "rank1" in cert.certified[2]  # monotone
        assert "rank2" in cert.certified[2]
        assert "rank4" in cert.certified[4]  # all features certified at P=4

    def test_custom_k_values(self, dash_result):
        cert = robust_certification(dash_result, k_values=[2, 4])
        assert set(cert.k_values) == {2, 4}
        assert 2 in cert.certified
        assert 4 in cert.certified

    def test_all_features_certified_at_k_equals_P(self, dash_result):
        """Every feature is certified top-P (all must appear in some ranking)."""
        P = dash_result.P
        cert = robust_certification(dash_result, k_values=[P])
        assert set(cert.certified[P]) == set(dash_result.feature_names)

    def test_feature_names_preserved(self, dash_result):
        cert = robust_certification(dash_result)
        assert cert.feature_names == list(dash_result.feature_names)

    def test_summary_returns_string(self, dash_result):
        cert = robust_certification(dash_result)
        s = cert.summary()
        assert isinstance(s, str)
        assert "Certified" in s

    def test_max_rank_determines_certification(self, dash_result):
        """certified[k] should exactly match features with max_rank <= k."""
        cert = robust_certification(dash_result, k_values=[1, 2, 3])
        for k in [1, 2, 3]:
            expected = {dash_result.feature_names[j] for j in range(dash_result.P) if cert.max_ranks[j] <= k}
            assert set(cert.certified[k]) == expected
