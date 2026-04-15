"""Extension: Federated Consensus — combine results from multiple sites.

Aggregates DASHResult objects from multiple sites (e.g., hospitals,
institutions) into a single consensus without sharing raw data. Each
site runs DASH independently; this extension combines the results.

The combined result is a DASHResult, so all other extensions
(confidence intervals, certification, theory bridge, etc.) work on it.

Usage:
    result = federated_consensus([result_site1, result_site2, result_site3])
    print(f"Cross-site agreement: {result.cross_site_agreement:.3f}")
    cert = robust_certification(result.combined)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["federated_consensus", "FederatedResult"]


@dataclass
class FederatedResult:
    """Result of federated_consensus().

    Attributes
    ----------
    combined : DASHResult
        The combined consensus result. Usable by all extensions.
        The K axis represents *sites*, not individual models within a site.
    per_site_importance : ndarray of shape (n_sites, P)
        Global importance from each site's DASHResult.
    cross_site_agreement : float
        Mean pairwise Spearman correlation of importance vectors across sites.
        High (>0.9) means sites agree on what's important.
    n_sites : int
    feature_names : list of str
    """

    combined: "DASHResult"
    per_site_importance: np.ndarray
    cross_site_agreement: float
    n_sites: int
    feature_names: list

    def summary(self) -> str:
        lines = [
            f"FederatedResult: {self.n_sites} sites, {len(self.feature_names)} features",
            f"Cross-site agreement: {self.cross_site_agreement:.3f}",
            "",
            "Per-site importance (top 5 features by consensus):",
        ]

        consensus_imp = self.combined.global_importance
        top5 = np.argsort(-consensus_imp)[:5]
        header = f"{'Feature':<20}" + "".join(f"{'Site ' + str(i):>10}" for i in range(self.n_sites))
        lines.append(header)
        lines.append("-" * (20 + 10 * self.n_sites))
        for j in top5:
            name = self.feature_names[j]
            vals = "".join(f"{self.per_site_importance[s, j]:>10.4f}" for s in range(self.n_sites))
            lines.append(f"{name:<20}{vals}")

        return "\n".join(lines)

    def plot(self):
        """Heatmap of per-site importance for top features."""
        import matplotlib.pyplot as plt

        consensus_imp = self.combined.global_importance
        top_k = min(15, len(self.feature_names))
        top_idx = np.argsort(-consensus_imp)[:top_k]

        data = self.per_site_importance[:, top_idx].T
        names = [self.feature_names[j] for j in top_idx]

        fig, ax = plt.subplots(figsize=(max(6, self.n_sites * 1.5), max(4, top_k * 0.4)))
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names)
        ax.set_xticks(range(self.n_sites))
        ax.set_xticklabels([f"Site {i}" for i in range(self.n_sites)])
        ax.set_title("Per-Site Feature Importance")
        fig.colorbar(im, ax=ax, label="Mean |attribution|")
        fig.tight_layout()
        return fig


def federated_consensus(
    results: list,
    weights: np.ndarray | list | None = None,
) -> FederatedResult:
    """Combine DASHResult objects from multiple sites.

    Each site's consensus SHAP matrix (n_ref × P) is treated as one
    "model" in a new DASHResult. The combined result's K axis represents
    sites, not individual models.

    Parameters
    ----------
    results : list of DASHResult
        One per site. All must have the same P (feature count) and
        feature names.
    weights : array-like of shape (n_sites,) or None
        Per-site weights for the consensus. Default: equal weights.
        Weights are normalized to sum to 1.

    Returns
    -------
    FederatedResult
    """
    from dash_shap.core.result import DASHResult

    if len(results) < 2:
        raise ValueError(f"Need at least 2 sites, got {len(results)}")

    P = results[0].P
    feature_names = list(results[0].feature_names)
    for i, r in enumerate(results):
        if r.P != P:
            raise ValueError(f"Feature count mismatch: site 0 has P={P}, site {i} has P={r.P}")
        if list(r.feature_names) != feature_names:
            raise ValueError(f"Feature names mismatch between site 0 and site {i}")

    n_sites = len(results)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_sites,):
            raise ValueError(f"weights must have shape ({n_sites},), got {weights.shape}")
        weights = weights / weights.sum()
    else:
        weights = np.ones(n_sites) / n_sites

    # Stack per-site consensus matrices as (n_sites, n_ref, P)
    # Use the consensus (mean SHAP) from each site, shape (n_ref_i, P)
    # If n_ref differs, use global_importance as (1, P) fallback
    min_n_ref = min(r.n_ref for r in results)

    matrices = np.stack(
        [
            r.all_shap_matrices[:, :min_n_ref, :].mean(axis=0)  # (n_ref, P) — site consensus
            for r in results
        ],
        axis=0,
    )  # (n_sites, min_n_ref, P)

    # Apply weights by scaling each site's matrix
    for s in range(n_sites):
        matrices[s] *= weights[s] * n_sites  # scale so mean = weighted mean

    combined = DASHResult.from_shap_matrices(matrices, feature_names=feature_names)

    # Per-site importance
    per_site_imp = np.array([r.global_importance for r in results])  # (n_sites, P)

    # Cross-site agreement: mean pairwise Spearman
    from scipy.stats import spearmanr

    correlations = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            r, _ = spearmanr(per_site_imp[i], per_site_imp[j])
            correlations.append(r)
    agreement = float(np.mean(correlations)) if correlations else 1.0

    return FederatedResult(
        combined=combined,
        per_site_importance=per_site_imp,
        cross_site_agreement=agreement,
        n_sites=n_sites,
        feature_names=feature_names,
    )
