"""Extension 4: Feature Groups — cluster features by SHAP substitutability."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["feature_groups", "GroupResult"]


@dataclass
class GroupResult:
    """Result of feature_groups().

    Attributes
    ----------
    groups : list[list[str]]
        Each inner list is a cluster of feature names.
    substitutability_matrix : ndarray (P, P) in [-1, 1]
        Mean Pearson correlation of SHAP values across K models.
        sub[i, j] ≈ 1 means features i and j are nearly interchangeable.
    labels : ndarray (P,) int
        0-based cluster label per feature.
    n_groups : int
    threshold : float
    feature_names : list[str]
    """

    groups: list
    substitutability_matrix: np.ndarray  # (P, P)
    labels: np.ndarray                   # (P,) int
    n_groups: int
    threshold: float
    feature_names: list

    def summary(self) -> str:
        lines = [
            f"GroupResult: P={len(self.feature_names)}, n_groups={self.n_groups}, threshold={self.threshold}",
            "",
        ]
        for gi, group in enumerate(self.groups):
            lines.append(f"  Group {gi}: {group}")
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        P = len(self.feature_names)
        fig, ax = plt.subplots(figsize=(max(6, P * 0.6), max(5, P * 0.5)))
        # Reorder features by cluster label for a block-diagonal appearance
        order = np.argsort(self.labels)
        mat = self.substitutability_matrix[np.ix_(order, order)]
        names = [self.feature_names[i] for i in order]

        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label="SHAP substitutability")
        ax.set_xticks(range(P))
        ax.set_yticks(range(P))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title(f"Feature Groups (threshold={self.threshold}, n_groups={self.n_groups})")
        fig.tight_layout()
        return fig


def feature_groups(
    result: "DASHResult",
    threshold: float = 0.8,
    method: str = "shap_substitutability",
) -> GroupResult:
    """Cluster features by SHAP substitutability across K models.

    Two features are substitutable when their SHAP values are highly correlated
    across observations, averaged over all K models. Features in the same group
    carry redundant explanatory information.

    Parameters
    ----------
    result : DASHResult
    threshold : float
        Substitutability threshold in (0, 1]. Features with mean SHAP correlation
        ≥ threshold are grouped together. Higher threshold → more, smaller groups.
    method : str
        Currently only ``"shap_substitutability"`` is supported.

    Returns
    -------
    GroupResult
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if not (0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    if method != "shap_substitutability":
        warnings.warn(
            f"method={method!r} is not supported; using 'shap_substitutability'.",
            UserWarning,
            stacklevel=2,
        )

    K = result.K
    P = result.P

    # sub[i, j] = mean over K models of Pearson corr(shap_k[:, i], shap_k[:, j])
    sub = np.zeros((P, P))
    for k in range(K):
        shap_k = result.all_shap_matrices[k]  # (n_ref, P)
        with np.errstate(invalid="ignore"):
            c = np.corrcoef(shap_k.T)          # (P, P)
        c = np.nan_to_num(c, nan=0.0)
        sub += c
    sub /= K

    # Enforce exact symmetry (numerical noise from float accumulation)
    sub = (sub + sub.T) / 2.0
    np.fill_diagonal(sub, 1.0)

    # Convert to distance matrix, clip for numerical stability
    dist = np.clip(1.0 - sub, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    # fcluster returns 1-based labels; convert to 0-based
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance") - 1

    # Build groups: list of lists of feature names
    n_groups = int(labels.max()) + 1
    groups: list = [[] for _ in range(n_groups)]
    for fi, gi in enumerate(labels):
        groups[gi].append(result.feature_names[fi])

    return GroupResult(
        groups=groups,
        substitutability_matrix=sub,
        labels=labels,
        n_groups=n_groups,
        threshold=threshold,
        feature_names=list(result.feature_names),
    )
