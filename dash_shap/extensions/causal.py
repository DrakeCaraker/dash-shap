"""Extension: Causal Flags — per-feature stability labels from FSI + correlation.

Labels each feature with an actionable flag based on its position in the
Importance-Stability (IS) plot:

- **robust**: High importance, low FSI → safe to use in decisions
- **collinear**: High importance, high FSI → report the group, not the feature
- **fragile**: Low importance, high FSI → exclude from downstream use
- **unimportant**: Low importance, low FSI → safe to ignore

Requires ``X_ref`` to compute feature correlations. Optionally accepts
a ``GroupResult`` for richer group-level context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["causal_flags", "CausalResult"]


@dataclass
class CausalResult:
    """Result of causal_flags().

    Attributes
    ----------
    flags : list of str
        Per-feature flag: "robust", "collinear", "fragile", or "unimportant".
    feature_names : list of str
    correlated_pairs : list of (int, int)
        Feature pairs with |correlation| above threshold.
    importance_threshold : float
    fsi_threshold : float
    """

    flags: list
    feature_names: list
    correlated_pairs: list
    importance_threshold: float
    fsi_threshold: float

    def summary(self, top_k: int = 20) -> str:
        lines = [
            f"CausalResult: {len(self.feature_names)} features, {len(self.correlated_pairs)} correlated pairs",
            "",
        ]
        counts: dict[str, int] = {}
        for f in self.flags:
            counts[f] = counts.get(f, 0) + 1
        for flag in ["robust", "collinear", "fragile", "unimportant"]:
            lines.append(f"  {flag}: {counts.get(flag, 0)} features")
        lines.append("")

        lines.append(f"{'Feature':<25} {'Flag':<15}")
        lines.append("-" * 40)
        for i, (name, flag) in enumerate(zip(self.feature_names, self.flags)):
            if i >= top_k:
                lines.append(f"  ... and {len(self.feature_names) - top_k} more")
                break
            lines.append(f"{name:<25} {flag:<15}")
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        flag_colors = {
            "robust": "tab:green",
            "collinear": "tab:orange",
            "fragile": "tab:red",
            "unimportant": "tab:gray",
        }
        colors = [flag_colors.get(f, "tab:gray") for f in self.flags]

        fig, ax = plt.subplots(figsize=(8, max(4, len(self.feature_names) * 0.3)))
        y = np.arange(len(self.feature_names))
        ax.barh(y, [1] * len(self.feature_names), color=colors, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(self.feature_names)
        ax.set_xlabel("Flag")
        ax.set_title("Causal Flags")

        from matplotlib.patches import Patch

        legend = [Patch(color=c, label=f) for f, c in flag_colors.items()]
        ax.legend(handles=legend, loc="lower right")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig


def causal_flags(
    result: "DASHResult",
    X_ref: np.ndarray,
    *,
    correlation_threshold: float = 0.5,
    importance_threshold: float | None = None,
    fsi_threshold: float | None = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Label each feature with an actionable stability flag.

    Combines the Feature Stability Index (FSI) from DASH with feature
    correlations from ``X_ref`` to produce per-feature flags.

    Parameters
    ----------
    result : DASHResult
    X_ref : ndarray of shape (n_samples, P)
        Reference data for computing feature correlations.
    correlation_threshold : float
        Minimum |correlation| to flag a pair as correlated (default 0.5).
    importance_threshold : float or None
        Threshold for "high importance" (default: median).
    fsi_threshold : float or None
        Threshold for "high FSI" (default: median of high-importance features).
    alpha : float
        Reserved for future statistical testing (default 0.05).

    Returns
    -------
    CausalResult
    """
    X_ref = np.asarray(X_ref)
    P = result.P
    imp = result.global_importance
    fsi = result.fsi

    if importance_threshold is None:
        importance_threshold = float(np.median(imp))
    if fsi_threshold is None:
        high_imp_mask = imp > importance_threshold
        fsi_threshold = float(np.median(fsi[high_imp_mask]) if np.any(high_imp_mask) else np.median(fsi))

    # Compute correlated pairs
    corr = np.abs(np.corrcoef(X_ref.T))
    correlated_pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            if corr[i, j] > correlation_threshold:
                correlated_pairs.append((i, j))

    # Assign flags based on IS-plot quadrant
    flags = []
    for j in range(P):
        high_imp = imp[j] > importance_threshold
        high_fsi = fsi[j] > fsi_threshold
        if high_imp and not high_fsi:
            flags.append("robust")
        elif high_imp and high_fsi:
            flags.append("collinear")
        elif not high_imp and high_fsi:
            flags.append("fragile")
        else:
            flags.append("unimportant")

    return CausalResult(
        flags=flags,
        feature_names=list(result.feature_names),
        correlated_pairs=correlated_pairs,
        importance_threshold=importance_threshold,
        fsi_threshold=fsi_threshold,
    )
