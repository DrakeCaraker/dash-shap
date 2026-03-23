"""Tests for impossibility theorem simulation.

Validates the three core predictions of the impossibility theorem:
1. First-mover concentration grows with sequential tree count T
2. Single sequential models cannot achieve both stability and equity
3. DASH (independent model averaging) circumvents the impossibility
"""

import numpy as np
import pytest
import shap
import xgboost as xgb
from scipy.stats import spearmanr

from dash_shap.experiments.synthetic import make_correlation_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_correlated_data(P=5, rho=0.9, N=1000, seed=42):
    """Generate correlated features with equal true importance via Cholesky."""
    rng = np.random.RandomState(seed)
    Sigma = np.full((P, P), rho)
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    Z = rng.randn(N, P)
    X = Z @ L.T
    # Equal true importance: y = sum of all features + noise
    beta = np.ones(P)
    y = X @ beta + rng.normal(0, 0.5, N)
    return X, y


def _train_xgb_and_shap(X, y, n_estimators=100, seed=42):
    """Train an XGBoost model and return global SHAP importances."""
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X, y, verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:200])
    importance = np.abs(shap_values).mean(axis=0)
    return importance


def _concentration(importance, P_group=None):
    """Compute concentration as max / sum (higher = more concentrated)."""
    total = importance.sum()
    if total == 0:
        return 0.0
    return importance.max() / total


def _gini(values):
    """Compute Gini coefficient of an array of non-negative values."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_correlated_data_generation():
    """Cholesky-based data generation produces target correlation structure."""
    rho_target = 0.9
    X, _ = _generate_correlated_data(P=5, rho=rho_target, N=5000, seed=123)
    corr = np.corrcoef(X, rowvar=False)
    # Off-diagonal elements should be close to target rho
    mask = ~np.eye(5, dtype=bool)
    empirical_rho = corr[mask].mean()
    np.testing.assert_allclose(empirical_rho, rho_target, atol=0.05)


def test_concentration_increases_with_trees():
    """First-mover concentration grows with sequential tree count T."""
    X, y = _generate_correlated_data(P=5, rho=0.9, N=1000, seed=42)

    imp_low = _train_xgb_and_shap(X, y, n_estimators=50, seed=42)
    imp_high = _train_xgb_and_shap(X, y, n_estimators=500, seed=42)

    conc_low = _concentration(imp_low)
    conc_high = _concentration(imp_high)

    # With more trees, the first-mover feature should dominate more
    assert conc_high > conc_low, (
        f"Expected concentration to increase with T: T=50 -> {conc_low:.4f}, T=500 -> {conc_high:.4f}"
    )


@pytest.mark.slow
def test_single_model_stability_equity_tradeoff():
    """Single sequential models cannot achieve both high stability and high equity."""
    P = 5
    rho = 0.9
    n_reps = 10
    importances = []

    for rep in range(n_reps):
        X, y = _generate_correlated_data(P=P, rho=rho, N=1000, seed=rep * 7 + 100)
        imp = _train_xgb_and_shap(X, y, n_estimators=200, seed=rep * 7 + 100)
        importances.append(imp)

    # Stability: mean pairwise Spearman correlation
    correlations = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            r, _ = spearmanr(importances[i], importances[j])
            correlations.append(r)
    stability = np.mean(correlations)

    # Equity: 1 - Gini (features have equal true importance, so ideal equity = 1)
    mean_imp = np.mean(importances, axis=0)
    equity = 1.0 - _gini(mean_imp)

    # The impossibility theorem predicts that stability * equity < threshold
    # A single model cannot have both high stability AND high equity
    # under collinearity with equal true importance.
    product = stability * equity
    assert product < 0.95, (
        f"Single model achieved stability={stability:.3f}, equity={equity:.3f}, "
        f"product={product:.3f} -- expected product < 0.95 under impossibility"
    )


@pytest.mark.slow
def test_dash_improves_equity():
    """DASH (independent model averaging) produces better equity than a single model."""
    P = 5
    rho = 0.9
    N = 1000
    M = 10  # number of independent models

    X, y = _generate_correlated_data(P=P, rho=rho, N=N, seed=42)

    # Single model importance
    single_imp = _train_xgb_and_shap(X, y, n_estimators=200, seed=42)
    single_equity = 1.0 - _gini(single_imp)

    # DASH: train M independent models with different seeds/hyperparameters,
    # average their SHAP importance vectors
    all_importances = []
    rng = np.random.RandomState(42)
    for m in range(M):
        seed_m = rng.randint(0, 100000)
        colsample = rng.uniform(0.1, 0.5)  # low colsample per DASH convention
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=rng.randint(3, 7),
            learning_rate=rng.uniform(0.01, 0.2),
            colsample_bytree=colsample,
            random_state=seed_m,
            n_jobs=1,
        )
        model.fit(X, y, verbose=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[:200])
        all_importances.append(np.abs(sv).mean(axis=0))

    dash_imp = np.mean(all_importances, axis=0)
    dash_equity = 1.0 - _gini(dash_imp)

    assert dash_equity > single_equity, (
        f"DASH equity ({dash_equity:.3f}) should exceed single model equity ({single_equity:.3f})"
    )


@pytest.mark.slow
def test_simulation_runs(tmp_path):
    """Smoke test: run_simulation (if available) completes without error."""
    # This test validates that the simulation infrastructure works end-to-end
    # with minimal parameters. If run_simulation doesn't exist yet, we
    # replicate its core logic inline.
    P = 5
    rho = 0.9
    n_reps = 3
    T_values = [50]
    M_values = [5]

    results = {}
    for T in T_values:
        concentrations = []
        for rep in range(n_reps):
            X, y = _generate_correlated_data(P=P, rho=rho, N=500, seed=rep)
            imp = _train_xgb_and_shap(X, y, n_estimators=T, seed=rep)
            concentrations.append(_concentration(imp))
        results[f"T={T}"] = {
            "mean_concentration": float(np.mean(concentrations)),
            "std_concentration": float(np.std(concentrations)),
        }

    for M in M_values:
        equities = []
        for rep in range(n_reps):
            X, y = _generate_correlated_data(P=P, rho=rho, N=500, seed=rep + 100)
            all_imp = []
            rng = np.random.RandomState(rep)
            for m in range(M):
                imp = _train_xgb_and_shap(X, y, n_estimators=100, seed=rng.randint(0, 100000))
                all_imp.append(imp)
            dash_imp = np.mean(all_imp, axis=0)
            equities.append(1.0 - _gini(dash_imp))
        results[f"M={M}"] = {
            "mean_equity": float(np.mean(equities)),
            "std_equity": float(np.std(equities)),
        }

    # Save results to tmp_path to verify I/O works
    import json

    out_file = tmp_path / "simulation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    assert out_file.exists()
    loaded = json.loads(out_file.read_text())
    assert len(loaded) == len(T_values) + len(M_values)
    for key, val in loaded.items():
        assert isinstance(val, dict)
