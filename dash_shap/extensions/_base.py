"""Shared utilities for DASH extensions.

Exactly 3 functions. Single-extension utilities stay in their own module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["per_model_importance", "per_model_rankings", "bootstrap_over_models"]


def per_model_importance(result: "DASHResult") -> np.ndarray:
    """Mean absolute SHAP value per model per feature.

    Returns
    -------
    ndarray of shape (K, P)
    """
    return np.mean(np.abs(result.all_shap_matrices), axis=1)  # mean over n_ref


def per_model_rankings(result: "DASHResult") -> np.ndarray:
    """Rank features within each model (1 = most important).

    Returns
    -------
    ndarray of shape (K, P), dtype int
    """
    importance = per_model_importance(result)  # (K, P)
    # argsort descending, then convert to 1-based ranks
    order = np.argsort(-importance, axis=1)    # (K, P)
    K, P = importance.shape
    ranks = np.empty_like(order)
    for k in range(K):
        ranks[k, order[k]] = np.arange(1, P + 1)
    return ranks


def bootstrap_over_models(
    result: "DASHResult",
    stat_fn: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Resample K model indices and apply stat_fn to each bootstrap sample.

    Parameters
    ----------
    result : DASHResult
    stat_fn : callable
        Receives a (K_boot, n_ref, P) SHAP sub-tensor; returns any array.
    n_boot : int
        Number of bootstrap replicates.
    seed : int

    Returns
    -------
    ndarray of shape (n_boot, *stat_fn_output_shape)
    """
    rng = np.random.default_rng(seed)
    K = result.K
    results = []
    for _ in range(n_boot):
        idx = rng.integers(0, K, size=K)
        sample = result.all_shap_matrices[idx]  # (K, n_ref, P)
        results.append(stat_fn(sample))
    return np.stack(results, axis=0)
