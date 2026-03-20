"""Tests for dash_shap.baselines module."""

import numpy as np
import pytest
from dash_shap.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear
from dash_shap.baselines import LargeSingleModelBaseline, RandomForestBaseline, PermutationImportanceBaseline

try:
    import lightgbm  # noqa: F401
    from dash_shap.baselines import LightGBMSingleBestBaseline

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


def test_large_single_model_fit_shapes():
    """Verify LSM fits on synthetic linear data and produces correct importance shape."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    m = LargeSingleModelBaseline(K=5, T_per_model=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    assert m.model_ is not None


def test_large_single_model_nonlinear():
    """Verify LSM works on nonlinear synthetic data (validates A8 fix)."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, _, meta = generate_synthetic_nonlinear(
        N=500, P=20, group_size=5, rho=0.7, seed=42
    )
    m = LargeSingleModelBaseline(K=5, T_per_model=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    preds = m.model_.predict(Xte)
    assert preds.shape[0] == Xte.shape[0]


def test_large_single_model_predictions():
    """Verify LSM produces reasonable predictions."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    m = LargeSingleModelBaseline(K=5, T_per_model=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    preds = m.model_.predict(Xte)
    # Should have some correlation with actual values
    corr = np.corrcoef(preds, yte)[0, 1]
    assert corr > 0.5


def test_random_forest_baseline_fit():
    """Verify RF baseline fits on synthetic linear data and produces correct importance shape."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    m = RandomForestBaseline(n_estimators=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    assert m.model_ is not None
    assert np.all(np.isnan(m.fsi_))


def test_random_forest_baseline_predictions():
    """Verify RF baseline produces reasonable predictions."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    m = RandomForestBaseline(n_estimators=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    preds = m.model_.predict(Xte)
    corr = np.corrcoef(preds, yte)[0, 1]
    assert corr > 0.5


def test_permutation_importance_baseline_fit():
    """Verify PermutationImportance baseline fits and produces correct importance shape."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    m = PermutationImportanceBaseline(n_trials=10, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, y_ref=yexp)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    assert m.model_ is not None
    assert np.all(np.isnan(m.fsi_))


@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
def test_lightgbm_baseline_fit():
    """Verify LightGBM baseline fits on synthetic linear data and produces correct importance shape."""
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
        N=500, P=20, group_size=5, rho=0.5, seed=42
    )
    m = LightGBMSingleBestBaseline(n_estimators=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    assert m.model_ is not None
    assert np.all(np.isnan(m.fsi_))
