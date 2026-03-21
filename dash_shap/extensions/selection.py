"""Extension 5: Stable Feature Selection — rank features by importance + stability."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["stable_feature_selection", "SelectionResult"]


@dataclass
class SelectionResult:
    """Result of stable_feature_selection().

    Attributes
    ----------
    selected_features : list[str]
        Top-k feature names (ascending composite score order).
    scores : ndarray (P,)
        Composite score per feature. Lower is better.
    importance_ranks : ndarray (P,) int
        1-based rank by global importance (rank 1 = most important).
    stability_ranks : ndarray (P,) int
        1-based rank by FSI ascending (rank 1 = most stable / lowest FSI).
    k : int
    feature_names : list[str]
    """

    selected_features: list
    scores: np.ndarray          # (P,)
    importance_ranks: np.ndarray  # (P,) int
    stability_ranks: np.ndarray   # (P,) int
    k: int
    feature_names: list

    def summary(self) -> str:
        lines = [
            f"SelectionResult: k={self.k}, P={len(self.feature_names)}",
            f"Selected: {self.selected_features}",
            "",
            f"{'Feature':<20} {'imp_rank':>9} {'stab_rank':>10} {'score':>8}",
            "-" * 52,
        ]
        order = np.argsort(self.scores)
        for i in order:
            marker = " *" if self.feature_names[i] in self.selected_features else ""
            lines.append(
                f"{self.feature_names[i]:<20} "
                f"{self.importance_ranks[i]:>9} "
                f"{self.stability_ranks[i]:>10} "
                f"{self.scores[i]:>8.3f}{marker}"
            )
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        P = len(self.feature_names)
        order = np.argsort(self.scores)
        names = [self.feature_names[i] for i in order]
        scores = self.scores[order]
        selected_set = set(self.selected_features)
        colors = ["tab:green" if n in selected_set else "tab:gray" for n in names]

        fig, ax = plt.subplots(figsize=(8, max(4, P * 0.35)))
        y = np.arange(P)
        ax.barh(y, scores, color=colors, alpha=0.8, align="center")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("Composite score (lower = better)")
        ax.set_title(f"Stable Feature Selection (top-{self.k} in green)")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig


def stable_feature_selection(
    result: "DASHResult",
    k: int = 10,
    importance_weight: float = 0.7,
    stability_weight: float = 0.3,
) -> SelectionResult:
    """Select top-k features by a composite importance + stability score.

    Score formula (lower is better)::

        score[j] = importance_weight * importance_rank[j]
                 + stability_weight  * stability_rank[j]

    where ``importance_rank`` is 1-based descending by global importance and
    ``stability_rank`` is 1-based ascending by FSI (low FSI = most stable = rank 1).

    Parameters
    ----------
    result : DASHResult
    k : int
        Number of features to select (1 ≤ k ≤ P).
    importance_weight : float
        Weight for importance ranking (default 0.7).
    stability_weight : float
        Weight for stability ranking (default 0.3).

    Returns
    -------
    SelectionResult
    """
    P = result.P

    if not (1 <= k <= P):
        raise ValueError(f"k must be in [1, {P}], got k={k}")

    if abs(importance_weight + stability_weight - 1.0) > 1e-6:
        warnings.warn(
            f"importance_weight + stability_weight = {importance_weight + stability_weight:.4f} ≠ 1.0. "
            "Scores are unnormalized.",
            UserWarning,
            stacklevel=2,
        )

    # importance_ranks: descending global_importance → 1-based rank
    imp_order = np.argsort(-result.global_importance)  # most important first
    importance_ranks = np.empty(P, dtype=float)
    importance_ranks[imp_order] = np.arange(1, P + 1)

    # stability_ranks: ascending FSI → 1-based rank (low FSI = rank 1)
    fsi_order = np.argsort(result.fsi)  # lowest FSI first
    stability_ranks = np.empty(P, dtype=float)
    stability_ranks[fsi_order] = np.arange(1, P + 1)

    scores = importance_weight * importance_ranks + stability_weight * stability_ranks
    top_k_idx = np.argsort(scores)[:k]
    selected_features = [result.feature_names[i] for i in top_k_idx]

    return SelectionResult(
        selected_features=selected_features,
        scores=scores,
        importance_ranks=importance_ranks.astype(int),
        stability_ranks=stability_ranks.astype(int),
        k=k,
        feature_names=list(result.feature_names),
    )
