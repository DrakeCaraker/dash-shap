from __future__ import annotations

import numpy as np
import pytest

from dash_shap.baselines.nn_baselines import BaggedNNBaseline, SingleNNBaseline
from dash_shap.core.nn_attribution import compute_nn_attributions
from dash_shap.core.nn_population import generate_nn_population
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
