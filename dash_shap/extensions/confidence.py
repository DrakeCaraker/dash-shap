"""Extension 1: Confidence intervals for DASH importance and FSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dash_shap.extensions._base import bootstrap_over_models

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = ["confidence_intervals", "ConfidenceResult"]


@dataclass
class ConfidenceResult:
    """Bootstrap confidence intervals for importance, FSI, and rankings.

    Each ``*_ci`` array has shape ``(P, 3)`` — [lower, point_estimate, upper].
    """

    importance_ci: np.ndarray
    fsi_ci: np.ndarray
    ranking_ci: np.ndarray
    feature_names: list[str]

    def summary(self) -> str:
        lines = ["Feature Importance Confidence Intervals", "=" * 50]
        for j, name in enumerate(self.feature_names):
            lo, pt, hi = self.importance_ci[j]
            lines.append(f"  {name:>12s}: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
        return "\n".join(lines)

    def plot(self) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(self.feature_names))))
        y = np.arange(len(self.feature_names))
        points = self.importance_ci[:, 1]
        lower = points - self.importance_ci[:, 0]
        upper = self.importance_ci[:, 2] - points
        ax.barh(y, points, xerr=[lower, upper], capsize=3, color="steelblue")
        ax.set_yticks(y)
        ax.set_yticklabels(self.feature_names)
        ax.set_xlabel("Global Importance")
        ax.set_title("DASH Importance with Bootstrap CI")
        fig.tight_layout()
        return fig


def confidence_intervals(
    result,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 42,
) -> ConfidenceResult:
    """Compute bootstrap confidence intervals over the K model ensemble.

    Parameters
    ----------
    result : DASHResult
        DASH output tensor.
    alpha : float
        Significance level (default 0.05 → 95% CI).
    n_boot : int
        Number of bootstrap replicates.
    seed : int
        RNG seed.

    Returns
    -------
    ConfidenceResult
    """
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)

    def _importance(tensor):
        consensus = np.mean(tensor, axis=0)
        return np.mean(np.abs(consensus), axis=0)

    def _fsi(tensor):
        consensus = np.mean(tensor, axis=0)
        variance = np.var(tensor, axis=0, ddof=1)
        gi = np.mean(np.abs(consensus), axis=0)
        ms = np.mean(np.sqrt(variance), axis=0)
        return ms / (gi + 1e-8)

    def _rankings(tensor):
        consensus = np.mean(tensor, axis=0)
        gi = np.mean(np.abs(consensus), axis=0)
        from scipy.stats import rankdata

        return rankdata(-gi, method="min")

    imp_boots = bootstrap_over_models(result, _importance, n_boot, seed)
    fsi_boots = bootstrap_over_models(result, _fsi, n_boot, seed)
    rank_boots = bootstrap_over_models(result, _rankings, n_boot, seed)

    importance_ci = np.column_stack([
        np.percentile(imp_boots, lo_pct, axis=0),
        result.global_importance,
        np.percentile(imp_boots, hi_pct, axis=0),
    ])

    fsi_ci = np.column_stack([
        np.percentile(fsi_boots, lo_pct, axis=0),
        result.fsi,
        np.percentile(fsi_boots, hi_pct, axis=0),
    ])

    ranking_ci = np.column_stack([
        np.min(rank_boots, axis=0),
        np.median(rank_boots, axis=0),
        np.max(rank_boots, axis=0),
    ])

    return ConfidenceResult(
        importance_ci=importance_ci,
        fsi_ci=fsi_ci,
        ranking_ci=ranking_ci,
        feature_names=result.feature_names,
    )
