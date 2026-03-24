"""Tests for previously untested baselines: SingleBest, EnsembleSHAP,
RandomSelection, NaiveAveraging, StochasticRetrain."""

import numpy as np
import pytest
from dash_shap.baselines import (
    SingleBestBaseline,
    EnsembleSHAPBaseline,
    RandomSelectionBaseline,
    NaiveAveragingBaseline,
    StochasticRetrainBaseline,
)


class TestSingleBestBaseline:
    def test_fit_shapes(self, synthetic_linear):
        d = synthetic_linear
        m = SingleBestBaseline(n_trials=10, seed=42)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert np.all(m.global_importance_ >= 0)
        assert m.model_ is not None

    def test_fit_from_population(self, trained_population, synthetic_linear):
        d = synthetic_linear
        pipe = trained_population
        m = SingleBestBaseline(n_trials=10, seed=42)
        m.fit_from_population(pipe.models_, pipe.val_scores_, d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert np.all(m.global_importance_ >= 0)


@pytest.mark.slow
class TestEnsembleSHAPBaseline:
    def test_fit_shapes(self, synthetic_linear):
        d = synthetic_linear
        m = EnsembleSHAPBaseline(n_estimators=50, seed=42)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert np.all(m.global_importance_ >= 0)
        assert m.model_ is not None
        assert np.all(np.isnan(m.fsi_))

    def test_predictions_reasonable(self, synthetic_linear):
        d = synthetic_linear
        m = EnsembleSHAPBaseline(n_estimators=50, seed=42)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        preds = m.model_.predict(d["X_test"])
        corr = np.corrcoef(preds, d["y_test"])[0, 1]
        assert corr > 0.5


@pytest.mark.slow
class TestRandomSelectionBaseline:
    def test_fit_shapes(self, synthetic_linear):
        d = synthetic_linear
        m = RandomSelectionBaseline(M=10, K=5, epsilon=0.15, seed=42)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert np.all(m.global_importance_ >= 0)
        assert len(m.selected_indices_) <= 5

    def test_fit_from_population(self, trained_population, synthetic_linear):
        d = synthetic_linear
        pipe = trained_population
        m = RandomSelectionBaseline(M=10, K=5, epsilon=0.15, seed=42)
        m.fit_from_population(pipe.models_, pipe.val_scores_, d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert m.fsi_ is not None

    def test_has_timing_info(self, synthetic_linear):
        d = synthetic_linear
        m = RandomSelectionBaseline(M=10, K=5, epsilon=0.15, seed=42)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        assert hasattr(m, "timing_")

    def test_predictions_shape(self, synthetic_linear):
        d = synthetic_linear
        m = RandomSelectionBaseline(M=10, K=5, epsilon=0.15, seed=42, verbose=False)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        preds = m.get_consensus_ensemble_predictions(d["X_test"])
        assert preds.shape[0] == d["X_test"].shape[0]

    def test_reproducibility(self, synthetic_linear):
        d = synthetic_linear
        kwargs = dict(M=10, K=5, epsilon=0.15, seed=42, verbose=False)
        m1 = RandomSelectionBaseline(**kwargs)
        m1.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        m2 = RandomSelectionBaseline(**kwargs)
        m2.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        np.testing.assert_array_equal(m1.global_importance_, m2.global_importance_)
        assert m1.selected_indices_ == m2.selected_indices_

    def test_fit_from_population_predictions(self, trained_population, synthetic_linear):
        d = synthetic_linear
        pipe = trained_population
        m = RandomSelectionBaseline(M=5, K=3, epsilon=0.15, seed=42, verbose=False)
        m.fit_from_population(pipe.models_, pipe.val_scores_, d["X_explain"])
        preds = m.get_consensus_ensemble_predictions(d["X_test"])
        assert preds.shape[0] == d["X_test"].shape[0]

    def test_fewer_candidates_than_k(self, synthetic_linear):
        d = synthetic_linear
        # epsilon=0.001 with M=10 should leave very few models passing filter
        # Use K=100 to ensure K > len(filtered) so all candidates are selected
        m = RandomSelectionBaseline(M=10, K=100, epsilon=0.15, seed=42, verbose=False)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        # All filtered models should be selected since K > filtered count
        assert len(m.selected_indices_) == len(m.filtered_indices_)
        assert len(m.selected_indices_) <= 10


class TestNaiveAveragingBaseline:
    def test_fit_from_population(self, trained_population, synthetic_linear):
        d = synthetic_linear
        pipe = trained_population
        m = NaiveAveragingBaseline(N=5)
        m.fit_from_population(pipe.models_, pipe.val_scores_, d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert np.all(m.global_importance_ >= 0)
        assert m.fsi_ is not None
        assert len(m.selected_indices_) <= 5


@pytest.mark.slow
class TestStochasticRetrainBaseline:
    def test_fit_shapes(self, synthetic_linear):
        d = synthetic_linear
        m = StochasticRetrainBaseline(N=3, seed=42)
        m.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], X_ref=d["X_explain"])
        assert m.global_importance_.shape == (20,)
        assert np.all(m.global_importance_ >= 0)
        assert m.fsi_ is not None
