"""Extension 2: Partial Orders — Paper 2 Core.

Answers: "Is feature i confidently more important than feature j?"
The confidence_matrix values directly answer "within-group π ≈ 0.5?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["partial_order", "PartialOrderResult"]


@dataclass
class PartialOrderResult:
    """Result of partial_order().

    Attributes
    ----------
    adjacency : ndarray (P, P) bool
        adjacency[i, j] = True if feature i is confidently more important than j.
    confidence_matrix : ndarray (P, P) float
        confidence_matrix[i, j] = π(i > j), the fraction of models ranking i above j.
        For method='bootstrap', this is the bootstrap probability estimate.
        Paper 2 decision gate: within-group π ≈ 0.5.
    n_determined : int
        Number of (i, j) pairs where adjacency[i, j] is True.
    n_undetermined : int
        Number of (i, j) pairs where neither adjacency[i, j] nor adjacency[j, i] is True.
    feature_names : list[str]
    method : str — 'fraction' or 'bootstrap'
    alpha : float
    """

    adjacency: np.ndarray  # (P, P) bool
    confidence_matrix: np.ndarray  # (P, P) float — π(i>j)
    n_determined: int
    n_undetermined: int
    feature_names: list
    method: str
    alpha: float

    def summary(self) -> str:
        P = len(self.feature_names)
        n_pairs = P * (P - 1) // 2
        lines = [
            f"PartialOrderResult: P={P}, method='{self.method}', alpha={self.alpha}",
            f"  Determined: {self.n_determined} / {n_pairs * 2} directed pairs",
            f"  Undetermined: {self.n_undetermined} / {n_pairs} unordered pairs",
            "",
            "Confidence matrix (π(row > col)):",
            "  " + "  ".join(f"{n:>10}" for n in self.feature_names),
        ]
        for i, name in enumerate(self.feature_names):
            row = "  ".join(f"{'---':>10}" if i == j else f"{self.confidence_matrix[i, j]:>10.3f}" for j in range(P))
            lines.append(f"{name:>10}  {row}")
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        P = len(self.feature_names)
        fig, ax = plt.subplots(figsize=(max(6, P), max(5, P * 0.8)))
        mat = self.confidence_matrix.copy()
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="π(row > col)")
        ax.set_xticks(range(P))
        ax.set_yticks(range(P))
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")
        ax.set_yticklabels(self.feature_names)
        ax.set_title(f"Partial Order Confidence Matrix (method={self.method})")
        fig.tight_layout()
        return fig


def partial_order(
    result: "DASHResult",
    alpha: float = 0.05,
    method: Literal["fraction", "bootstrap"] = "fraction",
    n_boot: int = 1000,
    seed: int = 42,
) -> PartialOrderResult:
    """Compute a partial order over features by confidence in importance ranking.

    Parameters
    ----------
    result : DASHResult
    alpha : float
        Significance level. For method='fraction', adjacency[i,j] = True when
        π(i>j) > 1 - alpha. For method='bootstrap', adjacency[i,j] = True when
        the bootstrap CI for π(i>j) excludes alpha.
    method : 'fraction' or 'bootstrap'
        'fraction' — raw fraction of K models ranking i above j (fast).
            Preferred when K < 10.
        'bootstrap' — bootstrap test on π. Requires K ≥ 10 for reliable coverage.
            With K < 10, bootstrap confidence intervals have poor coverage.
    n_boot : int
        Bootstrap replicates (only used when method='bootstrap').
    seed : int

    Returns
    -------
    PartialOrderResult

    Notes
    -----
    Standalone — does NOT depend on the CI extension.
    The raw confidence_matrix values directly answer the Paper 2 question
    "within-group π ≈ 0.5?".
    """
    from dash_shap.extensions._base import per_model_rankings

    K = result.K
    P = result.P

    if method == "bootstrap" and K < 10:
        import warnings

        warnings.warn(
            f"K={K} < 10: bootstrap partial order may have poor coverage. Prefer method='fraction' when K < 10.",
            UserWarning,
            stacklevel=2,
        )

    ranks: np.ndarray = per_model_rankings(result).astype(float)  # (K, P)

    # confidence_matrix[i, j] = fraction of K models ranking i above j
    # (i.e., rank_i < rank_j)
    confidence_matrix: np.ndarray = np.zeros((P, P), dtype=float)
    for i in range(P):
        for j in range(P):
            if i != j:
                confidence_matrix[i, j] = np.mean(ranks[:, i] < ranks[:, j])

    if method == "fraction":
        adjacency = confidence_matrix > (1 - alpha)

    elif method == "bootstrap":
        rng = np.random.default_rng(seed)
        # Bootstrap CI: for each replicate, compute π_b(i>j) = fraction of
        # resampled models where rank_i < rank_j. adjacency[i,j] = True iff
        # the lower alpha-quantile of {π_b(i>j)} > 0.5 (one-sided bootstrap test).
        boot_confidence = np.zeros((n_boot, P, P))
        for b in range(n_boot):
            idx = rng.integers(0, K, size=K)
            boot_ranks = ranks[idx, :]  # (K, P) resampled
            # Vectorized: comparison across all (i, j) pairs at once
            boot_confidence[b] = np.mean(
                boot_ranks[:, :, np.newaxis] < boot_ranks[:, np.newaxis, :], axis=0
            )
        lower_ci = np.quantile(boot_confidence, alpha, axis=0)
        adjacency = lower_ci > 0.5

    else:
        raise ValueError(f"method must be 'fraction' or 'bootstrap', got {method!r}")

    # Count determined pairs
    n_determined = int(np.sum(adjacency))

    # n_undetermined = pairs where neither i>j nor j>i is determined
    n_undetermined = 0
    for i in range(P):
        for j in range(i + 1, P):
            if not adjacency[i, j] and not adjacency[j, i]:
                n_undetermined += 1

    return PartialOrderResult(
        adjacency=adjacency,
        confidence_matrix=confidence_matrix,
        n_determined=n_determined,
        n_undetermined=n_undetermined,
        feature_names=list(result.feature_names),
        method=method,
        alpha=alpha,
    )
