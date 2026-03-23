"""Thread budget utilities for controlling nested parallelism.

On multi-core machines, triple-nested parallelism (outer joblib x inner joblib
x XGBoost internal threads) causes massive oversubscription. This module
provides a centralized budget that divides available cores across all layers.
"""

from __future__ import annotations

import os
from typing import NamedTuple

__all__ = ["get_available_cores", "compute_thread_budget", "compute_rep_worker_budget", "ThreadBudget"]


class ThreadBudget(NamedTuple):
    """Thread allocation across parallelism layers."""

    n_outer: int
    n_inner: int
    nthread: int


def get_available_cores() -> int:
    """Detect available CPU cores.

    Priority: DASH_MAX_THREADS env var > sched_getaffinity > cpu_count > 1.
    """
    env_limit = os.environ.get("DASH_MAX_THREADS")
    if env_limit is not None:
        return max(1, int(env_limit))
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


def compute_thread_budget(
    n_outer: int,
    n_inner: int | None = None,
    total_cores: int | None = None,
) -> ThreadBudget:
    """Compute thread allocation so ``n_outer * n_inner * nthread <= total_cores``.

    Parameters
    ----------
    n_outer : int
        Number of outer parallel workers (e.g., rho levels).
    n_inner : int or None
        Number of inner parallel workers per outer worker. If ``None``,
        computed as ``total_cores // n_outer``.
    total_cores : int or None
        Total available cores. If ``None``, detected via
        :func:`get_available_cores`.

    Returns
    -------
    ThreadBudget
        Named tuple ``(n_outer, n_inner, nthread)``.
    """
    if total_cores is None:
        total_cores = get_available_cores()

    n_outer = max(1, min(n_outer, total_cores))

    if n_inner is None:
        n_inner = max(1, total_cores // n_outer)
    else:
        n_inner = max(1, min(n_inner, total_cores // n_outer))

    nthread = max(1, total_cores // (n_outer * n_inner))

    return ThreadBudget(n_outer=n_outer, n_inner=n_inner, nthread=nthread)


_MEMORY_PER_REP_MB = 200  # conservative upper bound (200 XGBoost models + SHAP arrays)


def compute_rep_worker_budget(
    n_work: int,
    memory_per_worker_mb: int = _MEMORY_PER_REP_MB,
    total_cores: int | None = None,
) -> int:
    """Compute safe number of concurrent (rho, rep) workers.

    Caps based on available CPU cores, available RAM, total work items,
    and the DASH_MAX_PARALLEL_REPS environment variable override.

    Parameters
    ----------
    n_work : int
        Total number of (rho, rep) pairs to run.
    memory_per_worker_mb : int
        Estimated peak memory per worker in MB.
    total_cores : int or None
        Available CPU cores; auto-detected if None.

    Returns
    -------
    int
        Number of concurrent workers (≥ 1).
    """
    # Explicit override (useful for shared machines / local dev)
    env_cap = os.environ.get("DASH_MAX_PARALLEL_REPS")
    if env_cap is not None:
        return max(1, min(int(env_cap), n_work))

    if total_cores is None:
        total_cores = get_available_cores()

    cpu_workers = min(total_cores, n_work)

    # Memory cap — use psutil if available, else skip
    try:
        import psutil  # type: ignore[import]

        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        usable_mb = available_mb * 0.75  # leave 25% headroom
        memory_workers = max(1, int(usable_mb / memory_per_worker_mb))
        workers = min(cpu_workers, memory_workers)
    except ImportError:
        workers = cpu_workers  # no psutil; trust CPU cap

    return max(1, workers)
