"""Extension 8: Local Uncertainty — per-observation SHAP disagreement across K models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["local_uncertainty", "LocalResult"]


@dataclass
class LocalResult:
    """Result of local_uncertainty().

    Attributes
    ----------
    mean_shap : ndarray (P,)
        Mean SHAP value across K models for the given observation.
    std_shap : ndarray (P,)
        Std deviation across K models (ddof=1). Zero when all K models agree.
    sign_flip_rate : ndarray (P,)
        Fraction of K models whose SHAP sign disagrees with the majority sign
        (sign of mean_shap). In [0, 1].
    feature_names : list[str]
    obs_idx : int
    top_k : int
    """

    mean_shap: np.ndarray  # (P,)
    std_shap: np.ndarray   # (P,)
    sign_flip_rate: np.ndarray  # (P,)
    feature_names: list
    obs_idx: int
    top_k: int

    def summary(self) -> str:
        P = len(self.feature_names)
        # Sort by |mean_shap| descending, show top_k
        order = np.argsort(-np.abs(self.mean_shap))[:self.top_k]
        lines = [
            f"LocalResult: obs_idx={self.obs_idx}, P={P}, top_k={self.top_k}",
            "",
            f"{'Feature':<20} {'mean_shap':>12} {'std_shap':>12} {'sign_flip':>10}",
            "-" * 58,
        ]
        for i in order:
            lines.append(
                f"{self.feature_names[i]:<20} "
                f"{self.mean_shap[i]:>12.4f} "
                f"{self.std_shap[i]:>12.4f} "
                f"{self.sign_flip_rate[i]:>10.3f}"
            )
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        order = np.argsort(-np.abs(self.mean_shap))[:self.top_k]
        names = [self.feature_names[i] for i in order]
        means = self.mean_shap[order]
        stds = self.std_shap[order]

        fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.4)))
        y = np.arange(len(order))
        colors = ["tab:red" if v < 0 else "tab:blue" for v in means]
        ax.barh(y, means, xerr=stds, color=colors, alpha=0.7, capsize=4, align="center")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (mean ± std across K models)")
        ax.set_title(f"Local Uncertainty — obs_idx={self.obs_idx}")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig


def local_uncertainty(
    result: "DASHResult",
    obs_idx: int,
    top_k: int = 15,
) -> LocalResult:
    """Compute per-feature SHAP uncertainty for a single observation.

    Summarises how much the K selected models disagree on the SHAP attribution
    for observation ``obs_idx``.

    Parameters
    ----------
    result : DASHResult
    obs_idx : int
        Index into the reference set (0 ≤ obs_idx < result.n_ref).
    top_k : int
        Number of top features to highlight in .summary() and .plot().

    Returns
    -------
    LocalResult

    Notes
    -----
    Memory-safe: slices a single observation from the tensor rather than
    copying the full (K, n_ref, P) array.
    """
    n_ref = result.n_ref
    if not (0 <= obs_idx < n_ref):
        raise ValueError(f"obs_idx={obs_idx} is out of range [0, {n_ref})")

    top_k = min(top_k, result.P)

    shap_obs = result.all_shap_matrices[:, obs_idx, :]  # (K, P)
    mean_shap = np.mean(shap_obs, axis=0)               # (P,)
    std_shap = np.std(shap_obs, axis=0, ddof=1)         # (P,)

    majority_sign = np.sign(mean_shap)
    # sign_flip_rate: fraction of K models whose sign ≠ majority sign
    # np.sign returns 0 for exactly-zero values; treat 0 as neither flipped nor not
    sign_flip_rate = np.mean(np.sign(shap_obs) != majority_sign, axis=0)  # (P,)

    return LocalResult(
        mean_shap=mean_shap,
        std_shap=std_shap,
        sign_flip_rate=sign_flip_rate,
        feature_names=list(result.feature_names),
        obs_idx=obs_idx,
        top_k=top_k,
    )
