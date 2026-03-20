"""Tests for dash_shap.core.consensus module (Stage 4: SHAP aggregation)."""

import numpy as np
from xgboost import XGBRegressor
from dash_shap.core.consensus import compute_consensus, _compute_shap_for_model


def _train_models(X_train, y_train, n=3, seed=42):
    """Train a small set of XGBoost models for testing."""
    models = {}
    for i in range(n):
        m = XGBRegressor(
            n_estimators=20,
            max_depth=3,
            colsample_bytree=0.5,
            random_state=seed + i,
        )
        m.fit(X_train, y_train)
        models[i] = m
    return models


class TestComputeShapForModel:
    """Tests for the per-model SHAP helper."""

    def test_output_shape(self, synthetic_small):
        d = synthetic_small
        m = XGBRegressor(n_estimators=10, random_state=42)
        m.fit(d["X_train"], d["y_train"])
        bg = d["X_explain"][:20]
        sv = _compute_shap_for_model(m, bg, d["X_explain"])
        assert sv.shape == d["X_explain"].shape

    def test_values_finite(self, synthetic_small):
        d = synthetic_small
        m = XGBRegressor(n_estimators=10, random_state=42)
        m.fit(d["X_train"], d["y_train"])
        bg = d["X_explain"][:20]
        sv = _compute_shap_for_model(m, bg, d["X_explain"])
        assert np.all(np.isfinite(sv))


class TestComputeConsensus:
    """Tests for the consensus aggregation function."""

    def test_output_shapes(self, synthetic_small):
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=3)
        selected = [0, 1, 2]
        consensus, all_shap = compute_consensus(
            models,
            selected,
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
        )
        N, P = d["X_explain"].shape
        assert consensus.shape == (N, P)
        assert all_shap.shape == (3, N, P)

    def test_consensus_is_mean_of_all_shap(self, synthetic_small):
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=3)
        consensus, all_shap = compute_consensus(
            models,
            [0, 1, 2],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
        )
        expected = np.mean(all_shap, axis=0)
        np.testing.assert_array_almost_equal(consensus, expected)

    def test_single_model_consensus_equals_raw(self, synthetic_small):
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=1)
        consensus, all_shap = compute_consensus(
            models,
            [0],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
        )
        np.testing.assert_array_almost_equal(consensus, all_shap[0])

    def test_deterministic_with_seed(self, synthetic_small):
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=3)
        c1, _ = compute_consensus(
            models,
            [0, 1, 2],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
        )
        c2, _ = compute_consensus(
            models,
            [0, 1, 2],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
        )
        np.testing.assert_array_almost_equal(c1, c2)

    def test_no_seed_uses_first_rows(self, synthetic_small):
        """Without seed, background is X_ref[:background_size] (deterministic)."""
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=2)
        c1, _ = compute_consensus(
            models,
            [0, 1],
            d["X_explain"],
            background_size=20,
            seed=None,
            verbose=False,
        )
        c2, _ = compute_consensus(
            models,
            [0, 1],
            d["X_explain"],
            background_size=20,
            seed=None,
            verbose=False,
        )
        np.testing.assert_array_almost_equal(c1, c2)

    def test_parallel_matches_sequential(self, synthetic_small):
        """n_jobs=-1 produces the same result as n_jobs=1."""
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=3)
        c_seq, _ = compute_consensus(
            models,
            [0, 1, 2],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        c_par, _ = compute_consensus(
            models,
            [0, 1, 2],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
            n_jobs=-1,
        )
        np.testing.assert_array_almost_equal(c_seq, c_par)

    def test_background_size_capped(self, synthetic_small):
        """background_size > N_prime should not error."""
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=2)
        consensus, all_shap = compute_consensus(
            models,
            [0, 1],
            d["X_explain"],
            background_size=9999,
            seed=42,
            verbose=False,
        )
        assert consensus.shape == d["X_explain"].shape

    def test_subset_selection(self, synthetic_small):
        """Only selected_indices models contribute to consensus."""
        d = synthetic_small
        models = _train_models(d["X_train"], d["y_train"], n=5)
        consensus, all_shap = compute_consensus(
            models,
            [1, 3],
            d["X_explain"],
            background_size=20,
            seed=42,
            verbose=False,
        )
        assert all_shap.shape[0] == 2  # only 2 models
