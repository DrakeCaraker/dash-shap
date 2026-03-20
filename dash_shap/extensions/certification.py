"""Extension 9: Robust Certification.

Feature j is certified top-k if its maximum rank across all K models is < k.
Monotone: certified top-3 implies certified top-4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["robust_certification", "CertificationResult"]


@dataclass
class CertificationResult:
    """Result of robust_certification().

    Attributes
    ----------
    certified : dict {int: list[str]}
        certified[k] = list of feature names certified as top-k.
    max_ranks : ndarray (P,)
        Maximum rank across all K models for each feature.
        Feature j is certified top-k iff max_ranks[j] <= k.
    k_values : list[int]
        k values that were tested.
    feature_names : list[str]
    """

    certified: dict           # {k: [feature_names]}
    max_ranks: np.ndarray     # (P,) — worst-case rank per feature
    k_values: list
    feature_names: list

    def summary(self) -> str:
        lines = [
            f"CertificationResult: P={len(self.feature_names)}, "
            f"K-values tested={self.k_values}",
            "",
        ]
        for k in self.k_values:
            names = self.certified.get(k, [])
            lines.append(f"  Certified top-{k}: {names or '(none)'}")
        lines.append("")
        lines.append("Max ranks (worst-case across K models):")
        for i, name in enumerate(self.feature_names):
            lines.append(f"  {name}: max_rank={self.max_ranks[i]}")
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        P = len(self.feature_names)
        fig, ax = plt.subplots(figsize=(max(6, len(self.k_values) * 1.5), max(4, P * 0.5)))

        data = np.zeros((P, len(self.k_values)), dtype=int)
        for ki, k in enumerate(self.k_values):
            certified_set = set(self.certified.get(k, []))
            for fi, name in enumerate(self.feature_names):
                data[fi, ki] = 1 if name in certified_set else 0

        ax.imshow(data, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(self.k_values)))
        ax.set_xticklabels([f"top-{k}" for k in self.k_values])
        ax.set_yticks(range(P))
        ax.set_yticklabels(self.feature_names)
        ax.set_title("Robust Certification (green = certified)")
        fig.tight_layout()
        return fig


def robust_certification(
    result: "DASHResult",
    k_values: Optional[list] = None,
) -> CertificationResult:
    """Certify which features are robustly top-k across all K models.

    Feature j is certified top-k if its rank is ≤ k in every one of the K models
    (equivalently, max_rank[j] ≤ k). This is a worst-case guarantee.

    Monotone: certified top-3 implies certified top-4.

    Parameters
    ----------
    result : DASHResult
    k_values : list[int] or None
        k values to certify. Defaults to [1, 2, 3, 5, 10] ∩ [1..P].

    Returns
    -------
    CertificationResult

    Notes
    -----
    ~5 lines of core logic using per_model_rankings(). Standalone — does NOT
    depend on the Partial Orders extension.
    """
    from dash_shap.extensions._base import per_model_rankings

    P = result.P
    if k_values is None:
        k_values = [k for k in [1, 2, 3, 5, 10] if k <= P]
    k_values = sorted(set(k_values))

    ranks = per_model_rankings(result)  # (K, P) — 1-based
    max_ranks = np.max(ranks, axis=0)  # (P,) — worst-case rank per feature

    certified = {}
    for k in k_values:
        certified[k] = [
            result.feature_names[j] for j in range(P) if max_ranks[j] <= k
        ]

    return CertificationResult(
        certified=certified,
        max_ranks=max_ranks,
        k_values=k_values,
        feature_names=list(result.feature_names),
    )
