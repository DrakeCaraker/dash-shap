"""Extension 9: Robust certification of top-k feature sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dash_shap.extensions._base import per_model_rankings

if TYPE_CHECKING:
    import matplotlib.figure

__all__ = ["robust_certification", "CertificationResult"]


@dataclass
class CertificationResult:
    """Certification of which features are robustly in the top-k.

    Feature *j* is **certified top-k** if ``max_ranks[j] <= k`` — meaning
    every one of the K models ranks it within the top k.
    """

    max_ranks: np.ndarray
    certified: dict[int, list[str]]
    feature_names: list[str]

    def summary(self) -> str:
        lines = ["Robust Certification", "=" * 40]
        for k in sorted(self.certified):
            features = self.certified[k]
            if features:
                lines.append(f"  Top-{k}: {', '.join(features)}")
            else:
                lines.append(f"  Top-{k}: (none certified)")
        return "\n".join(lines)

    def plot(self) -> matplotlib.figure.Figure:
        import matplotlib.pyplot as plt

        k_vals = sorted(self.certified)
        counts = [len(self.certified[k]) for k in k_vals]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(k_vals, counts, color="steelblue")
        ax.set_xlabel("k")
        ax.set_ylabel("Number of certified features")
        ax.set_title("Robust Top-k Certification")
        ax.set_xticks(k_vals)
        fig.tight_layout()
        return fig


def robust_certification(
    result,
    k_values: list[int] | None = None,
) -> CertificationResult:
    """Certify which features are robustly ranked in the top-k across all models.

    Parameters
    ----------
    result : DASHResult
        DASH output tensor.
    k_values : list[int] | None
        Values of k to certify. Defaults to ``range(1, P + 1)``.

    Returns
    -------
    CertificationResult
    """
    rankings = per_model_rankings(result)  # (K, P)
    max_ranks = rankings.max(axis=0)  # (P,)

    P = result.P
    if k_values is None:
        k_values = list(range(1, P + 1))

    certified = {}
    for k in k_values:
        certified[k] = [
            result.feature_names[j] for j in range(P) if max_ranks[j] <= k
        ]

    return CertificationResult(
        max_ranks=max_ranks,
        certified=certified,
        feature_names=result.feature_names,
    )
