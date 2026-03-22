"""Smoke tests for experiment functions in the parallel runner."""

import pytest


@pytest.mark.slow
def test_k_sweep_independence_smoke():
    """Smoke test: k_sweep_independence runs with tiny settings via parallel runner."""
    from run_experiments_parallel import experiment_k_sweep_independence

    results = experiment_k_sweep_independence(resume=False)
    assert len(results) > 0
    # Verify per-method structure
    for k_val, k_data in results.items():
        for method, mdata in k_data.items():
            if isinstance(mdata, dict):
                assert "stability" in mdata


@pytest.mark.slow
def test_asymmetric_dgp_smoke():
    """Smoke test: asymmetric_dgp uses parallel schema (not old sequential schema)."""
    from run_experiments_parallel import experiment_asymmetric_dgp

    results = experiment_asymmetric_dgp(resume=False)
    assert 0.5 in results or "0.5" in results
    # Assert parallel schema — NOT the old sequential keys
    for rho_data in results.values():
        for method, mdata in rho_data.items():
            if isinstance(mdata, dict):
                assert "bias_f0" in mdata, "old sequential key 'bias_mean' schema detected"
                assert "bias_mean" not in mdata, "old sequential key present"
