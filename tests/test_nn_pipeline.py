from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dash_shap.baselines.nn_baselines import BaggedNNBaseline, SingleNNBaseline
from dash_shap.core.nn_attribution import (
    _compute_kernel_shap_for_model,
    compute_nn_attributions,
)
from dash_shap.core.nn_population import generate_nn_population, train_single_nn
from dash_shap.core.pipeline import DASHPipeline
from dash_shap.experiments.synthetic import generate_synthetic_linear


@pytest.mark.slow
def test_nn_population_training():
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.5, seed=42
    )
    models, scores, configs = generate_nn_population(Xtr, ytr, Xv, yv, M=10, seed=42, n_jobs=1)
    assert len(models) == 10
    assert all(s <= 0 for s in scores.values())
    assert len(configs) == 10


def test_nn_population_feature_masking():
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=100, P=10, group_size=5, rho=0.5, seed=42
    )
    models, scores, configs = generate_nn_population(
        Xtr, ytr, Xv, yv, M=5, seed=42, n_jobs=1, feature_mask_fraction=0.3
    )
    assert len(models) == 5
    assert all(np.isfinite(s) for s in scores.values())


@pytest.mark.slow
def test_nn_attribution_kernel_shap():
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.5, seed=42
    )
    models, scores, configs = generate_nn_population(Xtr, ytr, Xv, yv, M=5, seed=42, n_jobs=1)
    consensus, all_shap = compute_nn_attributions(models, list(range(5)), Xexp[:20], seed=42, verbose=False)
    assert consensus.shape == (20, 10)
    assert all_shap.shape == (5, 20, 10)
    assert not np.all(consensus == 0)


@pytest.mark.slow
def test_nn_fit_from_attributions():
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.5, seed=42
    )
    models, scores, configs = generate_nn_population(Xtr, ytr, Xv, yv, M=10, seed=42, n_jobs=1)
    consensus, all_shap = compute_nn_attributions(models, list(range(10)), Xexp[:20], seed=42, verbose=False)

    pipe = DASHPipeline(K=5, seed=42)
    pipe.fit_from_attributions(all_shap, scores)

    assert pipe.global_importance_.shape == (10,)
    assert pipe.fsi_.shape == (10,)
    assert pipe.consensus_matrix_.shape[1] == 10


@pytest.mark.slow
def test_single_nn_baseline():
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.5, seed=42
    )
    m = SingleNNBaseline(n_trials=5, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp[:20])
    assert m.global_importance_ is not None
    assert m.global_importance_.shape == (10,)
    assert m.model_ is not None


@pytest.mark.slow
def test_bagged_nn_baseline():
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=200, P=10, group_size=5, rho=0.5, seed=42
    )
    m = BaggedNNBaseline(N=3, seed=42, n_jobs=1)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp[:20])
    assert m.global_importance_ is not None
    assert m.global_importance_.shape == (10,)
    assert m.fsi_ is not None


# ---------------------------------------------------------------------------
# Fast (non-slow) tests using mocks to cover remaining uncovered lines
# ---------------------------------------------------------------------------


def _make_mock_model(n_features: int = 4) -> MagicMock:
    """Return a mock model whose .predict() returns zeros."""
    model = MagicMock()
    model.predict.side_effect = lambda X: np.zeros(len(X))
    return model


def _make_small_data(n: int = 20, p: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    X = rng.randn(n, p).astype(np.float32)
    y = rng.randn(n).astype(np.float32)
    return X, y


# --- nn_attribution.py coverage ---


def test_compute_kernel_shap_for_model_direct():
    """Cover _compute_kernel_shap_for_model (lines 15-19) via mocked KernelExplainer."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.core.nn_attribution.shap.KernelExplainer") as mock_ke:
        instance = mock_ke.return_value
        instance.shap_values.return_value = fake_sv
        result = _compute_kernel_shap_for_model(model, X[:5], X)

    assert result.shape == (10, 4)
    np.testing.assert_array_equal(result, fake_sv)


def test_compute_kernel_shap_for_model_list_output():
    """Cover the list-output branch (line 18) in _compute_kernel_shap_for_model."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    sv_class0 = np.ones((10, 4))
    sv_class1 = np.zeros((10, 4))

    with patch("dash_shap.core.nn_attribution.shap.KernelExplainer") as mock_ke:
        instance = mock_ke.return_value
        instance.shap_values.return_value = [sv_class0, sv_class1]
        result = _compute_kernel_shap_for_model(model, X[:5], X)

    assert result.shape == (10, 4)
    np.testing.assert_array_almost_equal(result, np.mean([sv_class0, sv_class1], axis=0))


def test_compute_nn_attributions_gradient_raises():
    """Cover method='gradient' ImportError branch (line 66)."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model}
    with pytest.raises(ImportError, match="captum"):
        compute_nn_attributions(models, [0], X, method="gradient")


def test_compute_nn_attributions_ig_raises():
    """Cover method='ig' ImportError branch (line 68)."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model}
    with pytest.raises(ImportError, match="captum"):
        compute_nn_attributions(models, [0], X, method="ig")


def test_compute_nn_attributions_unknown_method_raises():
    """Cover unknown method ValueError branch (line 70)."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model}
    with pytest.raises(ValueError, match="Unknown attribution method"):
        compute_nn_attributions(models, [0], X, method="foobar")


def test_compute_nn_attributions_sequential_no_seed(capsys):
    """Cover sequential path without seed (line 82) and verbose output (lines 104-107)."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model, 1: model}
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.core.nn_attribution._compute_kernel_shap_for_model", return_value=fake_sv):
        consensus, all_shap = compute_nn_attributions(models, [0, 1], X, seed=None, n_jobs=1, verbose=True)

    assert consensus.shape == (10, 4)
    assert all_shap.shape == (2, 10, 4)
    captured = capsys.readouterr()
    assert "Top 5 features" in captured.out


def test_compute_nn_attributions_sequential_with_seed(capsys):
    """Cover sequential path with seed (lines 77-80) and verbose=False."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model}
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.core.nn_attribution._compute_kernel_shap_for_model", return_value=fake_sv):
        consensus, all_shap = compute_nn_attributions(models, [0], X, seed=7, n_jobs=1, verbose=False)

    assert consensus.shape == (10, 4)
    assert all_shap.shape == (1, 10, 4)
    # verbose=False: no output
    captured = capsys.readouterr()
    assert captured.out == ""


def test_compute_nn_attributions_parallel(capsys):
    """Cover parallel path (lines 93-100) and verbose parallel message."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model, 1: model}
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.core.nn_attribution._compute_kernel_shap_for_model", return_value=fake_sv):
        consensus, all_shap = compute_nn_attributions(models, [0, 1], X, seed=42, n_jobs=2, verbose=True)

    assert consensus.shape == (10, 4)
    assert all_shap.shape == (2, 10, 4)
    captured = capsys.readouterr()
    assert "n_jobs=2" in captured.out


def test_compute_nn_attributions_shape_mismatch_sequential():
    """Cover shape validation ValueError in sequential loop (line 91)."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model}
    bad_sv = np.ones((5, 4))  # wrong first dim

    with patch("dash_shap.core.nn_attribution._compute_kernel_shap_for_model", return_value=bad_sv):
        with pytest.raises(ValueError, match="expected shape"):
            compute_nn_attributions(models, [0], X, n_jobs=1, verbose=False)


def test_compute_nn_attributions_shape_mismatch_parallel():
    """Cover shape validation ValueError in parallel path (line 99)."""
    X, _ = _make_small_data(10, 4)
    model = _make_mock_model(4)
    models = {0: model, 1: model}
    bad_sv = np.ones((5, 4))  # wrong first dim

    with patch("dash_shap.core.nn_attribution._compute_kernel_shap_for_model", return_value=bad_sv):
        with pytest.raises(ValueError, match="expected shape"):
            compute_nn_attributions(models, [0, 1], X, n_jobs=2, verbose=False)


# --- nn_population.py coverage ---


def test_train_single_nn_binary():
    """Cover binary classification path (lines 93-106) in train_single_nn."""
    rng = np.random.RandomState(1)
    X_tr = rng.randn(30, 4).astype(np.float64)
    y_tr = rng.randint(0, 2, size=30)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randint(0, 2, size=10)

    config = {
        "hidden_layer_sizes": (16,),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "activation": "relu",
        "batch_size": 16,
    }
    model, score = train_single_nn(config, X_tr, y_tr, X_v, y_v, task="binary", max_iter=5, seed=0)
    assert 0.0 <= score <= 1.0


def test_train_single_nn_unknown_task():
    """Cover unknown task ValueError (lines 108-109) in train_single_nn."""
    rng = np.random.RandomState(2)
    X = rng.randn(10, 4)
    y = rng.randn(10)
    config = {
        "hidden_layer_sizes": (16,),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "activation": "relu",
        "batch_size": 16,
    }
    with pytest.raises(ValueError, match="Unknown task"):
        train_single_nn(config, X, y, X, y, task="multiclass", max_iter=5, seed=0)


def test_generate_nn_population_mask_all_false():
    """Cover edge case where mask is all-False and must enable at least one feature (line 143)."""
    rng = np.random.RandomState(3)
    X_tr = rng.randn(20, 4).astype(np.float64)
    y_tr = rng.randn(20)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randn(10)

    # feature_mask_fraction=1.0 means all features are masked (random >= 1.0 is never True),
    # so every model triggers the "ensure at least one feature" branch.
    models, scores, configs = generate_nn_population(
        X_tr, y_tr, X_v, y_v, M=3, seed=0, n_jobs=1, feature_mask_fraction=1.0, verbose=False
    )
    assert len(models) == 3
    assert all(np.isfinite(s) for s in scores.values())


# --- nn_baselines.py coverage ---


def test_single_nn_baseline_init():
    """Cover SingleNNBaseline.__init__ attribute assignments (lines 29-34)."""
    m = SingleNNBaseline(n_trials=5, task="binary", n_jobs=2, seed=99)
    assert m.n_trials == 5
    assert m.task == "binary"
    assert m.n_jobs == 2
    assert m.seed == 99
    assert m.model_ is None
    assert m.global_importance_ is None


def test_bagged_nn_baseline_init():
    """Cover BaggedNNBaseline.__init__ attribute assignments (lines 119-124)."""
    m = BaggedNNBaseline(N=10, task="binary", n_jobs=4, seed=7)
    assert m.N == 10
    assert m.task == "binary"
    assert m.n_jobs == 4
    assert m.seed == 7
    assert m.global_importance_ is None
    assert m.fsi_ is None


def test_single_nn_baseline_compute_shap_with_seed():
    """Cover _compute_shap with seed (lines 38-47) for SingleNNBaseline."""
    X, _ = _make_small_data(20, 4)
    model = _make_mock_model(4)
    fake_sv = np.ones((20, 4))
    m = SingleNNBaseline(n_trials=1, seed=0)

    with patch("dash_shap.baselines.nn_baselines.shap.KernelExplainer") as mock_ke:
        instance = mock_ke.return_value
        instance.shap_values.return_value = fake_sv
        m._compute_shap(model, X, background_size=5, seed=42)

    assert m.global_importance_ is not None
    assert m.global_importance_.shape == (4,)


def test_single_nn_baseline_compute_shap_no_seed():
    """Cover _compute_shap without seed (else branch line 44)."""
    X, _ = _make_small_data(20, 4)
    model = _make_mock_model(4)
    fake_sv = np.ones((20, 4))
    m = SingleNNBaseline(n_trials=1, seed=0)

    with patch("dash_shap.baselines.nn_baselines.shap.KernelExplainer") as mock_ke:
        instance = mock_ke.return_value
        instance.shap_values.return_value = fake_sv
        m._compute_shap(model, X, background_size=5, seed=None)

    assert m.global_importance_ is not None


def test_single_nn_baseline_fit_sequential_mock():
    """Cover SingleNNBaseline.fit sequential path (lines 68-108) with mocked training."""
    rng = np.random.RandomState(0)
    X_tr = rng.randn(20, 4).astype(np.float64)
    y_tr = rng.randn(20)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randn(10)

    fake_model = _make_mock_model(4)
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.baselines.nn_baselines.sample_nn_configurations", return_value=[{}]):
        with patch("dash_shap.baselines.nn_baselines.train_single_nn", return_value=(fake_model, -0.5)):
            with patch("dash_shap.baselines.nn_baselines.shap.KernelExplainer") as mock_ke:
                instance = mock_ke.return_value
                instance.shap_values.return_value = fake_sv
                m = SingleNNBaseline(n_trials=1, n_jobs=1, seed=0)
                result = m.fit(X_tr, y_tr, X_v, y_v)

    assert result is m
    assert m.model_ is fake_model
    assert m.global_importance_ is not None


def test_single_nn_baseline_fit_xref_none():
    """Cover X_ref=None branch (line 68-69) in SingleNNBaseline.fit."""
    rng = np.random.RandomState(0)
    X_tr = rng.randn(20, 4).astype(np.float64)
    y_tr = rng.randn(20)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randn(10)

    fake_model = _make_mock_model(4)
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.baselines.nn_baselines.sample_nn_configurations", return_value=[{}]):
        with patch("dash_shap.baselines.nn_baselines.train_single_nn", return_value=(fake_model, -0.5)):
            with patch("dash_shap.baselines.nn_baselines.shap.KernelExplainer") as mock_ke:
                instance = mock_ke.return_value
                instance.shap_values.return_value = fake_sv
                m = SingleNNBaseline(n_trials=1, n_jobs=1, seed=0)
                m.fit(X_tr, y_tr, X_v, y_v, X_ref=None)

    assert m.model_ is fake_model


def test_single_nn_baseline_fit_parallel_mock():
    """Cover SingleNNBaseline.fit parallel path (lines 91-104) with mocked training."""
    rng = np.random.RandomState(0)
    X_tr = rng.randn(20, 4).astype(np.float64)
    y_tr = rng.randn(20)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randn(10)

    fake_model = _make_mock_model(4)
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.baselines.nn_baselines.sample_nn_configurations", return_value=[{}, {}]):
        with patch("dash_shap.baselines.nn_baselines.train_single_nn", return_value=(fake_model, -0.3)):
            with patch("dash_shap.baselines.nn_baselines.shap.KernelExplainer") as mock_ke:
                instance = mock_ke.return_value
                instance.shap_values.return_value = fake_sv
                m = SingleNNBaseline(n_trials=2, n_jobs=2, seed=0)
                m.fit(X_tr, y_tr, X_v, y_v, X_ref=X_v)

    assert m.model_ is not None


def test_bagged_nn_baseline_fit_with_best_config_mock():
    """Cover BaggedNNBaseline.fit with a pre-provided best_config (lines 136-186)."""
    rng = np.random.RandomState(0)
    X_tr = rng.randn(20, 4).astype(np.float64)
    y_tr = rng.randn(20)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randn(10)

    best_config = {
        "hidden_layer_sizes": (16,),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "activation": "relu",
        "batch_size": 16,
    }
    fake_model = _make_mock_model(4)
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.baselines.nn_baselines.train_single_nn", return_value=(fake_model, -0.5)):
        with patch("dash_shap.baselines.nn_baselines.compute_nn_attributions") as mock_attr:
            mock_attr.return_value = (fake_sv, np.stack([fake_sv]))
            with patch("dash_shap.baselines.nn_baselines.compute_diagnostics") as mock_diag:
                mock_diag.return_value = (None, None, np.ones(4), np.ones(4))
                m = BaggedNNBaseline(N=2, seed=0, n_jobs=1)
                result = m.fit(X_tr, y_tr, X_v, y_v, X_ref=X_v, best_config=best_config)

    assert result is m
    assert m.global_importance_ is not None
    assert m.fsi_ is not None


def test_bagged_nn_baseline_fit_no_best_config_mock():
    """Cover BaggedNNBaseline.fit path without best_config (lines 139-158): runs search."""
    rng = np.random.RandomState(0)
    X_tr = rng.randn(20, 4).astype(np.float64)
    y_tr = rng.randn(20)
    X_v = rng.randn(10, 4).astype(np.float64)
    y_v = rng.randn(10)

    fake_model = _make_mock_model(4)
    fake_sv = np.ones((10, 4))

    with patch("dash_shap.baselines.nn_baselines.sample_nn_configurations", return_value=[{}]):
        with patch("dash_shap.baselines.nn_baselines.train_single_nn", return_value=(fake_model, -0.5)):
            with patch("dash_shap.baselines.nn_baselines.compute_nn_attributions") as mock_attr:
                mock_attr.return_value = (fake_sv, np.stack([fake_sv]))
                with patch("dash_shap.baselines.nn_baselines.compute_diagnostics") as mock_diag:
                    mock_diag.return_value = (None, None, np.ones(4), np.ones(4))
                    m = BaggedNNBaseline(N=1, seed=0, n_jobs=1)
                    result = m.fit(X_tr, y_tr, X_v, y_v, best_config=None)

    assert result is m
