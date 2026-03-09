"""Tests for dash.baselines module."""
import numpy as np
from dash.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear
from dash.baselines import LargeSingleModelBaseline


def test_large_single_model_fit_shapes():
    """Verify LSM fits on synthetic linear data and produces correct importance shape."""
    Xtr, ytr, Xv, yv, Xte, yte, grps, true_imp, meta = \
        generate_synthetic_linear(N=500, P=20, group_size=5, rho=0.5, seed=42)
    m = LargeSingleModelBaseline(K=5, T_per_model=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xte)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    assert m.model_ is not None


def test_large_single_model_nonlinear():
    """Verify LSM works on nonlinear synthetic data (validates A8 fix)."""
    Xtr, ytr, Xv, yv, Xte, yte, grps, _, meta = \
        generate_synthetic_nonlinear(N=500, P=20, group_size=5, rho=0.7, seed=42)
    m = LargeSingleModelBaseline(K=5, T_per_model=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xte)
    assert m.global_importance_.shape == (20,)
    assert np.all(m.global_importance_ >= 0)
    preds = m.model_.predict(Xte)
    assert preds.shape[0] == Xte.shape[0]


def test_large_single_model_predictions():
    """Verify LSM produces reasonable predictions."""
    Xtr, ytr, Xv, yv, Xte, yte, grps, true_imp, meta = \
        generate_synthetic_linear(N=500, P=20, group_size=5, rho=0.5, seed=42)
    m = LargeSingleModelBaseline(K=5, T_per_model=50, seed=42)
    m.fit(Xtr, ytr, Xv, yv, X_ref=Xte)
    preds = m.model_.predict(Xte)
    # Should have some correlation with actual values
    corr = np.corrcoef(preds, yte)[0, 1]
    assert corr > 0.5
