"""Tests for DASHResult: construction, computed fields, serialization, integration."""

import importlib.util
import pathlib
import tempfile

import numpy as np
import pytest

from dash_shap.core.result import DASHResult, VersionError

_xgboost_available = importlib.util.find_spec("xgboost") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(K=5, n_ref=20, P=4, seed=0):
    rng = np.random.default_rng(seed)
    matrices = rng.standard_normal((K, n_ref, P))
    return DASHResult.from_shap_matrices(
        matrices,
        feature_names=[f"feat_{i}" for i in range(P)],
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self):
        result = _make_result()
        assert result.K == 5
        assert result.n_ref == 20
        assert result.P == 4

    def test_auto_feature_names(self):
        rng = np.random.default_rng(0)
        m = rng.standard_normal((3, 10, 2))
        result = DASHResult.from_shap_matrices(m)
        assert result.feature_names == ["f0", "f1"]

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="3D"):
            DASHResult.from_shap_matrices(np.ones((5, 10)))

    def test_rejects_k_less_than_2(self):
        with pytest.raises(ValueError, match="K must be"):
            DASHResult.from_shap_matrices(np.ones((1, 10, 3)))

    def test_rejects_wrong_feature_names_length(self):
        with pytest.raises(ValueError, match="feature_names"):
            DASHResult(np.ones((3, 10, 4)), feature_names=["a", "b"])

    def test_val_scores_list_converted_to_ndarray(self):
        rng = np.random.default_rng(0)
        m = rng.standard_normal((4, 10, 3))
        result = DASHResult.from_shap_matrices(m, val_scores=[0.9, 0.8, 0.85, 0.7])
        assert isinstance(result.val_scores, np.ndarray)
        assert result.val_scores.shape == (4,)

    def test_val_scores_wrong_length(self):
        rng = np.random.default_rng(0)
        m = rng.standard_normal((4, 10, 3))
        with pytest.raises(ValueError, match="val_scores"):
            DASHResult.from_shap_matrices(m, val_scores=[0.9, 0.8])


# ---------------------------------------------------------------------------
# Computed fields
# ---------------------------------------------------------------------------


class TestComputedFields:
    def test_consensus_shape(self):
        result = _make_result(K=5, n_ref=20, P=4)
        assert result.consensus.shape == (20, 4)

    def test_variance_shape(self):
        result = _make_result()
        assert result.variance.shape == (20, 4)

    def test_global_importance_shape(self):
        result = _make_result()
        assert result.global_importance.shape == (4,)

    def test_fsi_shape(self):
        result = _make_result()
        assert result.fsi.shape == (4,)

    def test_fsi_non_negative(self):
        result = _make_result()
        assert np.all(result.fsi >= 0)

    def test_arrays_are_read_only(self):
        result = _make_result()
        with pytest.raises((ValueError, TypeError)):
            result.all_shap_matrices[0, 0, 0] = 999.0

    def test_fsi_formula_matches_diagnostics(self):
        """DASHResult.fsi must match compute_diagnostics() exactly."""
        from dash_shap.core.diagnostics import compute_diagnostics

        result = _make_result(K=8, n_ref=30, P=6)
        _, _, fsi_ref, _ = compute_diagnostics(result.all_shap_matrices)
        np.testing.assert_allclose(result.fsi, fsi_ref)

    def test_memory_bytes_positive(self):
        result = _make_result()
        assert result.memory_bytes > 0


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_basic_round_trip(self):
        result = _make_result(K=5, n_ref=20, P=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "test_result"
            result.save(path)
            loaded = DASHResult.load(path)
        assert loaded.K == result.K
        assert loaded.n_ref == result.n_ref
        assert loaded.P == result.P
        np.testing.assert_array_equal(loaded.all_shap_matrices, result.all_shap_matrices)
        assert loaded.feature_names == result.feature_names

    def test_large_array_round_trip(self):
        """K=30, n_ref=200, P=81 — no shape corruption or dtype change."""
        rng = np.random.default_rng(7)
        m = rng.standard_normal((30, 200, 81))
        result = DASHResult.from_shap_matrices(m)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "large"
            result.save(path)
            loaded = DASHResult.load(path)
        assert loaded.all_shap_matrices.shape == (30, 200, 81)
        assert loaded.all_shap_matrices.dtype == np.float64
        np.testing.assert_array_equal(loaded.all_shap_matrices, result.all_shap_matrices)

    def test_val_scores_round_trip(self):
        rng = np.random.default_rng(0)
        m = rng.standard_normal((4, 10, 3))
        result = DASHResult.from_shap_matrices(m, val_scores=[0.9, 0.8, 0.85, 0.7])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "with_scores"
            result.save(path)
            loaded = DASHResult.load(path)
        np.testing.assert_allclose(loaded.val_scores, result.val_scores)

    def test_version_error_on_future_version(self):
        """Load raises VersionError if format_version > current."""
        import json

        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "future"
            result.save(path)
            # Tamper with version
            meta_path = pathlib.Path(str(path) + ".json")
            meta = json.loads(meta_path.read_text())
            meta["format_version"] = 999
            meta_path.write_text(json.dumps(meta))
            with pytest.raises(VersionError):
                DASHResult.load(path)

    def test_sidecar_contains_format_version(self):
        import json

        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "check"
            result.save(path)
            meta = json.loads((pathlib.Path(str(path) + ".json")).read_text())
        assert "format_version" in meta
        assert meta["format_version"] == 1


# ---------------------------------------------------------------------------
# fit_from_attributions() integration (B12)
# ---------------------------------------------------------------------------


class TestFitFromAttributions:
    @pytest.mark.skipif(not _xgboost_available, reason="xgboost not installed")
    def test_produces_valid_dash_result(self):
        """Pre-computed (M, n_ref, P) matrices produce a valid DASHResult."""
        from dash_shap.core.pipeline import DASHPipeline

        rng = np.random.default_rng(42)
        M, n_ref, P = 20, 30, 5
        matrices = rng.standard_normal((M, n_ref, P))
        val_scores = {i: float(rng.uniform(0.7, 0.9)) for i in range(M)}

        pipe = DASHPipeline(M=M, K=5, epsilon=0.15, delta=0.01, seed=42, verbose=False)
        pipe.fit_from_attributions(matrices, val_scores=val_scores)

        assert pipe.result_ is not None
        assert isinstance(pipe.result_, DASHResult)
        assert pipe.result_.K == len(pipe.selected_indices_)
        assert pipe.result_.P == P


# ---------------------------------------------------------------------------
# Pipeline integration (B12)
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """pipe.result_ must match pipe.fsi_ and pipe.global_importance_ exactly."""

    @pytest.mark.slow
    @pytest.mark.skipif(not _xgboost_available, reason="xgboost not installed")
    def test_result_matches_pipeline_diagnostics(self, trained_population):
        pipe = trained_population
        assert pipe.result_ is not None
        np.testing.assert_allclose(pipe.result_.fsi, pipe.fsi_)
        np.testing.assert_allclose(pipe.result_.global_importance, pipe.global_importance_)
        np.testing.assert_array_equal(pipe.result_.all_shap_matrices, pipe.all_shap_matrices_)
