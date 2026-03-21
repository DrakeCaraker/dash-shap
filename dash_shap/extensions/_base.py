"""Shared utilities for DASH extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["per_model_importance", "per_model_rankings", "bootstrap_over_models"]


def per_model_importance(result: DASHResult) -> np.ndarray:
    """Mean absolute SHAP value per model per feature.

    Returns
    -------
    np.ndarray
        Shape ``(K, P)``.
    """
    return np.mean(np.abs(result.all_shap_matrices), axis=1)


def per_model_rankings(result: DASHResult) -> np.ndarray:
    """Rank features within each model (1 = most important).

    Returns
    -------
    np.ndarray
        Shape ``(K, P)`` with integer ranks.
    """
    imp = per_model_importance(result)
    # rankdata on -imp so that largest importance gets rank 1
    rankings = np.empty_like(imp, dtype=np.intp)
    for k in range(imp.shape[0]):
        rankings[k] = rankdata(-imp[k], method="min")
    return rankings


def bootstrap_over_models(
    result: DASHResult,
    stat_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Resample K model indices and apply *stat_fn* to each bootstrap sample.

    Parameters
    ----------
    result : DASHResult
        Source tensor.
    stat_fn : callable
        ``stat_fn(shap_matrices) -> np.ndarray`` where *shap_matrices* has
        shape ``(K', n_ref, P)`` (K' = resampled K, may contain repeats).
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n_boot, *stat_fn_output_shape)``.
    """
    rng = np.random.default_rng(seed)
    K = result.K
    tensor = result.all_shap_matrices

    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, K, size=K)
        samples.append(stat_fn(tensor[idx]))
    return np.array(samples)
