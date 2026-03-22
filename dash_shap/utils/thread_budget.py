"""Thread budget utilities for controlling nested parallelism.

On multi-core machines, triple-nested parallelism (outer joblib x inner joblib
x XGBoost internal threads) causes massive oversubscription. This module
provides a centralized budget that divides available cores across all layers.
"""

from __future__ import annotations

import os
from typing import NamedTuple

__all__ = ["get_available_cores", "compute_thread_budget", "ThreadBudget"]


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
