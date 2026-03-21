"""Tests for partial order extension."""

import numpy as np
import pytest

from dash_shap.extensions.partial_order import partial_order


class TestPartialOrder:
    def test_known_feature_beats_noise(self, dash_result):
        """f0 (robust driver) should be confidently > f2 (unimportant)."""
        po = partial_order(dash_result, alpha=0.05)
        assert po.adjacency[0, 2], "f0 should be confidently > f2"

    def test_not_symmetric(self, dash_result):
        """If i > j then not j > i."""
        po = partial_order(dash_result, alpha=0.05)
        P = dash_result.P
        for i in range(P):
            for j in range(P):
                if i != j and po.adjacency[i, j]:
                    assert not po.adjacency[j, i], f"Both {i}>{j} and {j}>{i}"

    def test_count_consistency(self, dash_result):
        po = partial_order(dash_result, alpha=0.05)
        P = dash_result.P
        assert po.n_determined + po.n_undetermined == P * (P - 1) // 2

    def test_confidence_matrix_range(self, dash_result):
        po = partial_order(dash_result, alpha=0.05)
        assert np.all(po.confidence_matrix >= 0)
        assert np.all(po.confidence_matrix <= 1)

    def test_confidence_matrix_complement(self, dash_result):
        """π(i>j) + π(j>i) ≈ 1 (with ties accounting for the gap)."""
        po = partial_order(dash_result, alpha=0.05)
        P = dash_result.P
        for i in range(P):
            for j in range(P):
                if i != j:
                    total = po.confidence_matrix[i, j] + po.confidence_matrix[j, i]
                    # Ties mean total can be < 1
                    assert total <= 1.0 + 1e-10

    def test_diagonal_false(self, dash_result):
        po = partial_order(dash_result, alpha=0.05)
        assert not np.any(np.diag(po.adjacency))

    def test_summary_returns_string(self, dash_result):
        po = partial_order(dash_result, alpha=0.05)
        s = po.summary()
        assert "determined" in s

    def test_bootstrap_method(self, dash_result):
        po = partial_order(dash_result, alpha=0.05, method="bootstrap")
        assert po.n_determined + po.n_undetermined == dash_result.P * (dash_result.P - 1) // 2

    def test_invalid_method(self, dash_result):
        with pytest.raises(ValueError, match="Unknown method"):
            partial_order(dash_result, method="invalid")
