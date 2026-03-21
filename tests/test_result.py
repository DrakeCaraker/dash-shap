"""Tests for DASHResult dataclass."""

import numpy as np
import pytest

from dash_shap.core.result import DASHResult
from dash_shap.core.diagnostics import compute_diagnostics


class TestConstruction:
    def test_valid_input(self, dash_result):
        assert dash_result.K == 5
        assert dash_result.n_ref == 20
        assert dash_result.P == 4

    def test_from_shap_matrices_auto_names(self):
        m = np.random.default_rng(0).standard_normal((3, 10, 4))
        r = DASHResult.from_shap_matrices(m)
        assert r.feature_names == ["f0", "f1", "f2", "f3"]

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            DASHResult.from_shap_matrices(np.zeros((10, 4)))

    def test_k_lt_2_raises(self):
        with pytest.raises(ValueError, match="K >= 2"):
            DASHResult(np.zeros((1, 10, 4)), ["a", "b", "c", "d"])

    def test_wrong_feature_names_length(self):
        with pytest.raises(ValueError, match="feature_names length"):
            DASHResult(np.zeros((3, 10, 4)), ["a", "b"])

    def test_wrong_val_scores_shape(self):
        with pytest.raises(ValueError, match="val_scores shape"):
            DASHResult(np.zeros((3, 10, 4)), ["a", "b", "c", "d"],
                       val_scores=np.ones(5))


class TestComputedFields:
    def test_consensus_matches_mean(self, dash_result):
        expected = np.mean(dash_result.all_shap_matrices, axis=0)
        np.testing.assert_allclose(dash_result.consensus, expected)

    def test_fsi_matches_diagnostics(self, dash_result):
        _, _, fsi_diag, gi_diag = compute_diagnostics(
            dash_result.all_shap_matrices
        )
        np.testing.assert_allclose(dash_result.fsi, fsi_diag)
        np.testing.assert_allclose(dash_result.global_importance, gi_diag)

    def test_variance_ddof1(self, dash_result):
        expected = np.var(dash_result.all_shap_matrices, axis=0, ddof=1)
        np.testing.assert_allclose(dash_result.variance, expected)


class TestImmutability:
    def test_shap_locked(self, dash_result):
        with pytest.raises(ValueError):
            dash_result.all_shap_matrices[0, 0, 0] = 999.0

    def test_consensus_locked(self, dash_result):
        with pytest.raises(ValueError):
            dash_result.consensus[0, 0] = 999.0

    def test_fsi_locked(self, dash_result):
        with pytest.raises(ValueError):
            dash_result.fsi[0] = 999.0

    def test_val_scores_locked(self, dash_result):
        with pytest.raises(ValueError):
            dash_result.val_scores[0] = 999.0


class TestProperties:
    def test_memory_bytes_positive(self, dash_result):
        assert dash_result.memory_bytes > 0

    def test_memory_bytes_sum(self, dash_result):
        expected = (
            dash_result.all_shap_matrices.nbytes
            + dash_result.consensus.nbytes
            + dash_result.variance.nbytes
            + dash_result.global_importance.nbytes
            + dash_result.fsi.nbytes
            + dash_result.val_scores.nbytes
        )
        assert dash_result.memory_bytes == expected


class TestSerialization:
    def test_round_trip(self, dash_result, tmp_path):
        path = tmp_path / "test_result"
        dash_result.save(path)
        loaded = DASHResult.load(path)

        np.testing.assert_allclose(
            loaded.all_shap_matrices, dash_result.all_shap_matrices
        )
        np.testing.assert_allclose(loaded.consensus, dash_result.consensus)
        np.testing.assert_allclose(loaded.fsi, dash_result.fsi)
        np.testing.assert_allclose(
            loaded.global_importance, dash_result.global_importance
        )
        np.testing.assert_allclose(loaded.val_scores, dash_result.val_scores)
        assert loaded.feature_names == dash_result.feature_names

    def test_round_trip_no_val_scores(self, tmp_path):
        m = np.random.default_rng(0).standard_normal((3, 10, 4))
        r = DASHResult.from_shap_matrices(m)
        path = tmp_path / "no_scores"
        r.save(path)
        loaded = DASHResult.load(path)
        assert loaded.val_scores is None
        np.testing.assert_allclose(loaded.fsi, r.fsi)
