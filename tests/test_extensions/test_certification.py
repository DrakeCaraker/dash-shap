"""Tests for robust certification extension."""

import numpy as np
import pytest

from dash_shap.extensions.certification import robust_certification


class TestRobustCertification:
    def test_monotone_in_k(self, dash_result):
        """certified_at_k ⊆ certified_at_(k+1)."""
        cr = robust_certification(dash_result)
        k_vals = sorted(cr.certified)
        for i in range(len(k_vals) - 1):
            k_small = k_vals[i]
            k_large = k_vals[i + 1]
            assert set(cr.certified[k_small]) <= set(cr.certified[k_large]), (
                f"Top-{k_small} {cr.certified[k_small]} not subset of "
                f"Top-{k_large} {cr.certified[k_large]}"
            )

    def test_robust_driver_certified_early(self, dash_result):
        """f0 (robust driver) should be certified at the smallest k."""
        cr = robust_certification(dash_result)
        # f0 should appear in some small k
        min_k = min(k for k, feats in cr.certified.items() if "f0" in feats)
        assert min_k <= 2, f"f0 not certified until k={min_k}"

    def test_noise_not_certified_small_k(self, dash_result):
        """f2 (unimportant) should not be certified at k=1."""
        cr = robust_certification(dash_result)
        assert "f2" not in cr.certified[1]

    def test_default_k_values(self, dash_result):
        cr = robust_certification(dash_result)
        assert sorted(cr.certified) == list(range(1, dash_result.P + 1))

    def test_custom_k_values(self, dash_result):
        cr = robust_certification(dash_result, k_values=[1, 3])
        assert sorted(cr.certified) == [1, 3]

    def test_all_certified_at_P(self, dash_result):
        """All features must be certified at k=P."""
        cr = robust_certification(dash_result)
        assert len(cr.certified[dash_result.P]) == dash_result.P

    def test_max_ranks_shape(self, dash_result):
        cr = robust_certification(dash_result)
        assert cr.max_ranks.shape == (dash_result.P,)

    def test_summary_returns_string(self, dash_result):
        cr = robust_certification(dash_result)
        s = cr.summary()
        assert "Top-" in s
