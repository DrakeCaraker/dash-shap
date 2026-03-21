"""Smoke tests for run_experiments.py experiment functions."""

import pytest


@pytest.mark.slow
def test_k_sweep_independence_smoke():
    """Smoke test: k_sweep_independence runs with tiny settings."""
    from run_experiments import experiment_k_sweep_independence

    results = experiment_k_sweep_independence(k_values=[1, 5], n_reps=2, seed=0)
    assert set(results.keys()) == {1, 5}
    for k_val, kdata in results.items():
        assert "DASH" in kdata and "SR" in kdata
        assert "stability" in kdata["DASH"]
        assert "accuracy_mean" in kdata["SR"]


@pytest.mark.slow
def test_asymmetric_dgp_smoke():
    """Smoke test: asymmetric_dgp runs with 1 rep."""
    from run_experiments import experiment_asymmetric_dgp

    results = experiment_asymmetric_dgp(n_reps=1)
    assert 0.5 in results
    assert "DASH" in results[0.5]
    assert "stability" in results[0.5]["DASH"]
