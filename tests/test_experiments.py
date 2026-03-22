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
def test_asymmetric_dgp_schema():
    """Parallel runner returns bias_f0/passive_leak_f1, not old sequential bias_mean/bias_std."""
    from run_experiments_parallel import experiment_asymmetric_dgp

    results = experiment_asymmetric_dgp(resume=False)
    assert len(results) > 0
    for rho_key, rho_data in results.items():
        float(rho_key)  # must be parseable
        for method, mdata in rho_data.items():
            if not isinstance(mdata, dict):
                continue
            assert "bias_f0" in mdata, f"missing 'bias_f0' for {method}@rho={rho_key}"
            assert "passive_leak_f1" in mdata
            assert "bias_mean" not in mdata, "old sequential schema key found"
            assert "bias_std" not in mdata, "old sequential schema key found"


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


@pytest.mark.slow
def test_variance_decomposition_crossed_schema():
    """ANOVA fractions must be present and sum to ~1.0."""
    from run_experiments_parallel import experiment_variance_decomposition_crossed

    results = experiment_variance_decomposition_crossed(resume=False)
    assert len(results) > 0
    for method, metrics in results.items():
        assert "data_var_frac" in metrics
        assert "model_var_frac" in metrics
        assert "residual_var_frac" in metrics
        total = sum(metrics[k] for k in ["data_var_frac", "model_var_frac", "residual_var_frac"])
        assert abs(total - 1.0) < 0.02, f"{method}: fracs sum to {total:.4f}, not 1.0"


@pytest.mark.slow
def test_k_sweep_independence_schema():
    """k_sweep_independence migrated to parallel runner returns expected structure."""
    from run_experiments_parallel import experiment_k_sweep_independence

    results = experiment_k_sweep_independence(resume=False)
    assert len(results) > 0
    for k_val, k_data in results.items():
        for method, mdata in k_data.items():
            if isinstance(mdata, dict):
                assert "stability" in mdata


@pytest.mark.slow
def test_variance_decomposition_basic_schema():
    from run_experiments_parallel import experiment_variance_decomposition

    results = experiment_variance_decomposition(resume=False)
    assert len(results) > 0
