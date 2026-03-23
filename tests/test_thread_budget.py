"""Tests for dash_shap.utils.thread_budget."""

import pytest

from dash_shap.utils.thread_budget import (
    get_available_cores,
    compute_thread_budget,
    compute_rep_worker_budget,
    ThreadBudget,
)


def test_get_available_cores_returns_positive():
    cores = get_available_cores()
    assert cores >= 1


def test_get_available_cores_env_override(monkeypatch):
    monkeypatch.setenv("DASH_MAX_THREADS", "8")
    assert get_available_cores() == 8


def test_get_available_cores_env_override_minimum(monkeypatch):
    monkeypatch.setenv("DASH_MAX_THREADS", "0")
    assert get_available_cores() == 1


def test_compute_thread_budget_basic():
    budget = compute_thread_budget(n_outer=4, total_cores=72)
    assert budget.n_outer == 4
    assert budget.n_inner == 18
    assert budget.nthread == 1
    assert budget.n_outer * budget.n_inner * budget.nthread <= 72


def test_compute_thread_budget_three_levels():
    budget = compute_thread_budget(n_outer=7, n_inner=10, total_cores=72)
    assert budget.n_outer == 7
    assert budget.n_inner == 10
    assert budget.nthread == 1  # 72 // 70 = 1
    assert budget.n_outer * budget.n_inner * budget.nthread <= 72


def test_compute_thread_budget_small_machine():
    budget = compute_thread_budget(n_outer=7, total_cores=4)
    assert budget.n_outer == 4  # clamped to total_cores
    assert budget.n_outer * budget.n_inner * budget.nthread <= 4


def test_compute_thread_budget_single_outer():
    budget = compute_thread_budget(n_outer=1, total_cores=72)
    assert budget.n_outer == 1
    assert budget.n_inner == 72
    assert budget.nthread == 1


def test_compute_thread_budget_generous_nthread():
    """When outer * inner is small, nthread gets the remaining budget."""
    budget = compute_thread_budget(n_outer=2, n_inner=3, total_cores=72)
    assert budget.nthread == 12  # 72 // (2 * 3) = 12
    assert budget.n_outer * budget.n_inner * budget.nthread <= 72


def test_thread_budget_is_named_tuple():
    budget = compute_thread_budget(n_outer=2, total_cores=8)
    assert isinstance(budget, ThreadBudget)
    n_outer, n_inner, nthread = budget
    assert n_outer * n_inner * nthread <= 8


def test_budget_product_never_exceeds_cores():
    """Property-style: product should never exceed total_cores."""
    for total in [1, 2, 4, 8, 16, 32, 64, 72, 128]:
        for n_out in range(1, total + 1):
            budget = compute_thread_budget(n_outer=n_out, total_cores=total)
            assert budget.n_outer * budget.n_inner * budget.nthread <= total, (
                f"Oversubscription: {budget} with total_cores={total}"
            )


def test_budget_all_fields_positive():
    """All fields should be >= 1."""
    for total in [1, 4, 72]:
        for n_out in [1, 3, 7, 100]:
            budget = compute_thread_budget(n_outer=n_out, total_cores=total)
            assert budget.n_outer >= 1
            assert budget.n_inner >= 1
            assert budget.nthread >= 1


###############################################################################
# Tests for compute_rep_worker_budget
###############################################################################


def test_compute_rep_worker_budget_returns_positive():
    result = compute_rep_worker_budget(n_work=10, total_cores=4)
    assert result >= 1


def test_compute_rep_worker_budget_never_exceeds_work(monkeypatch):
    """Result must never exceed the number of work items."""
    monkeypatch.delenv("DASH_MAX_PARALLEL_REPS", raising=False)
    monkeypatch.setenv("DASH_MAX_THREADS", "100")
    result = compute_rep_worker_budget(n_work=3, total_cores=100)
    assert result <= 3


def test_compute_rep_worker_budget_cpu_cap(monkeypatch):
    """Result must never exceed available cores."""
    monkeypatch.delenv("DASH_MAX_PARALLEL_REPS", raising=False)
    monkeypatch.setenv("DASH_MAX_THREADS", "4")
    result = compute_rep_worker_budget(n_work=250, total_cores=4)
    assert result <= 4


def test_compute_rep_worker_budget_env_override(monkeypatch):
    """DASH_MAX_PARALLEL_REPS overrides all other caps."""
    monkeypatch.setenv("DASH_MAX_PARALLEL_REPS", "2")
    result = compute_rep_worker_budget(n_work=250, total_cores=72)
    assert result == 2


def test_compute_rep_worker_budget_env_override_capped_to_work(monkeypatch):
    """DASH_MAX_PARALLEL_REPS is capped to n_work even if larger."""
    monkeypatch.setenv("DASH_MAX_PARALLEL_REPS", "100")
    result = compute_rep_worker_budget(n_work=3, total_cores=72)
    assert result == 3


def test_compute_rep_worker_budget_single_core(monkeypatch):
    """On a 1-core machine the result should be 1."""
    monkeypatch.delenv("DASH_MAX_PARALLEL_REPS", raising=False)
    monkeypatch.setenv("DASH_MAX_THREADS", "1")
    result = compute_rep_worker_budget(n_work=50, total_cores=1)
    assert result == 1


def test_compute_rep_worker_budget_zero_env_override_clamped(monkeypatch):
    """DASH_MAX_PARALLEL_REPS=0 should clamp to 1."""
    monkeypatch.setenv("DASH_MAX_PARALLEL_REPS", "0")
    result = compute_rep_worker_budget(n_work=10)
    assert result >= 1
