"""Extension 2: Partial orders over feature importance rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dash_shap.extensions._base import per_model_importance, bootstrap_over_models

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = ["partial_order", "PartialOrderResult"]


@dataclass
class PartialOrderResult:
    """Partial ordering of features by importance confidence.

    ``adjacency[i, j] = True`` means feature *i* is confidently more important
    than feature *j*.
    """

    adjacency: np.ndarray
    confidence_matrix: np.ndarray
    n_determined: int
    n_undetermined: int
    feature_names: list[str]

    def summary(self) -> str:
        total = self.n_determined + self.n_undetermined
        return (
            f"Partial Order: {self.n_determined} of {total} pairs determined "
            f"({self.n_undetermined} undetermined)"
        )

    def plot(self) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(self.confidence_matrix, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(self.feature_names)))
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        ax.set_xlabel("Feature j")
        ax.set_ylabel("Feature i")
        ax.set_title("π(i > j): Fraction of models ranking i above j")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig


def partial_order(
    result,
    alpha: float = 0.05,
    method: str = "fraction",
) -> PartialOrderResult:
    """Compute a partial order over features from the DASH ensemble.

    Parameters
    ----------
    result : DASHResult
        DASH output tensor.
    alpha : float
        Significance threshold. A pair (i, j) is determined when
        ``confidence_matrix[i, j] > 1 - alpha`` or ``< alpha``.
    method : {"fraction", "bootstrap"}
        ``"fraction"`` uses the raw fraction of K models.
        ``"bootstrap"`` uses bootstrap resampling for the test.

    Returns
    -------
    PartialOrderResult
    """
    if method == "fraction":
        return _fraction_method(result, alpha)
    elif method == "bootstrap":
        return _bootstrap_method(result, alpha)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'fraction' or 'bootstrap'.")


def _fraction_method(result, alpha: float) -> PartialOrderResult:
    imp = per_model_importance(result)  # (K, P)
    K, P = imp.shape
    threshold = 1 - alpha

    # confidence_matrix[i, j] = fraction of models where imp_i > imp_j
    # Vectorized: (K, P, 1) > (K, 1, P) → (K, P, P) bool, mean over K
    confidence_matrix = np.mean(imp[:, :, None] > imp[:, None, :], axis=0)

    adjacency = confidence_matrix > threshold
    np.fill_diagonal(adjacency, False)

    n_pairs = P * (P - 1) // 2
    # Count pairs where either direction is determined
    n_determined = 0
    for i in range(P):
        for j in range(i + 1, P):
            if adjacency[i, j] or adjacency[j, i]:
                n_determined += 1

    return PartialOrderResult(
        adjacency=adjacency,
        confidence_matrix=confidence_matrix,
        n_determined=n_determined,
        n_undetermined=n_pairs - n_determined,
        feature_names=result.feature_names,
    )


def _bootstrap_method(result, alpha: float, n_boot: int = 1000, seed: int = 42) -> PartialOrderResult:
    imp = per_model_importance(result)  # (K, P)
    _, P = imp.shape

    # Observed fractions
    confidence_matrix = np.mean(imp[:, :, None] > imp[:, None, :], axis=0)

    # Bootstrap: resample model indices and recompute fractions
    def _conf_matrix(tensor):
        imp_boot = np.mean(np.abs(tensor), axis=1)  # (K, P)
        return np.mean(imp_boot[:, :, None] > imp_boot[:, None, :], axis=0)

    boot_samples = bootstrap_over_models(result, _conf_matrix, n_boot, seed)
    # boot_samples: (n_boot, P, P)

    # A pair is determined if the bootstrap CI for π excludes 0.5
    lo = np.percentile(boot_samples, 100 * (alpha / 2), axis=0)

    # i > j determined if lower bound of π(i>j) > 0.5
    adjacency = lo > 0.5
    np.fill_diagonal(adjacency, False)

    n_pairs = P * (P - 1) // 2
    n_determined = 0
    for i in range(P):
        for j in range(i + 1, P):
            if adjacency[i, j] or adjacency[j, i]:
                n_determined += 1

    return PartialOrderResult(
        adjacency=adjacency,
        confidence_matrix=confidence_matrix,
        n_determined=n_determined,
        n_undetermined=n_pairs - n_determined,
        feature_names=result.feature_names,
    )
