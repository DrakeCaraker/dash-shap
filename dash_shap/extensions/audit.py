"""Extension: Audit Report — structured explanation audit with warnings.

Aggregates all available DASH diagnostics into a single report suitable
for model documentation, regulatory review, or stakeholder communication.

Works with just a DASHResult (basic report). Richer with optional
enrichments: X_ref (correlation analysis), groups, confidence intervals,
partial orders, and causal flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["audit_report", "AuditResult"]


@dataclass
class AuditResult:
    """Result of audit_report().

    Attributes
    ----------
    sections : dict of {str: str}
        Named report sections (overview, importance, stability, warnings, etc.).
    warnings : list of str
        Actionable warnings flagged during the audit.
    feature_names : list of str
    K : int
    P : int
    """

    sections: dict
    warnings: list
    feature_names: list
    K: int
    P: int

    def summary(self) -> str:
        lines = []
        for title, content in self.sections.items():
            lines.append(f"## {title}")
            lines.append(content)
            lines.append("")
        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    def plot(self):
        """Plot importance with stability-colored bars."""
        import matplotlib.pyplot as plt

        from dash_shap.core.diagnostics import ImportanceStabilityPlot

        # Delegate to IS plot — the most informative single visualization
        return ImportanceStabilityPlot.plot(
            np.array([float(x) for x in self.sections.get("_importance_array", [])])
            if "_importance_array" in self.sections
            else np.zeros(self.P),
            np.array([float(x) for x in self.sections.get("_fsi_array", [])])
            if "_fsi_array" in self.sections
            else np.zeros(self.P),
            feature_names=self.feature_names,
        )


def audit_report(
    result: "DASHResult",
    X_ref: np.ndarray | None = None,
    *,
    confidence: Any = None,
    partial_order: Any = None,
    groups: Any = None,
    causal: Any = None,
) -> AuditResult:
    """Generate a structured explanation audit report.

    Parameters
    ----------
    result : DASHResult
    X_ref : ndarray or None
        Reference data for correlation analysis. Adds collinearity section.
    confidence : ConfidenceResult or None
        From confidence_intervals(). Adds CI section.
    partial_order : PartialOrderResult or None
        From partial_order(). Adds ranking certainty section.
    groups : GroupResult or None
        From feature_groups(). Adds group analysis section.
    causal : CausalResult or None
        From causal_flags(). Adds flag section.

    Returns
    -------
    AuditResult
    """
    sections = {}
    warnings = []
    P = result.P
    K = result.K
    names = list(result.feature_names)
    imp = result.global_importance
    fsi = result.fsi

    # --- Overview ---
    sections["Overview"] = (
        f"DASH audit: {P} features, K={K} models in consensus.\n"
        f"Consensus computed from {K} independently trained models with "
        f"diversity selection."
    )

    # --- Importance ranking ---
    ranking = np.argsort(-imp)
    top_lines = []
    for rank, j in enumerate(ranking[:10], 1):
        top_lines.append(f"  {rank}. {names[j]}: {imp[j]:.4f} (FSI={fsi[j]:.3f})")
    sections["Top Features"] = "\n".join(top_lines)

    # Store arrays for plot (not displayed in summary)
    sections["_importance_array"] = imp.tolist()
    sections["_fsi_array"] = fsi.tolist()

    # --- Stability analysis ---
    median_fsi = float(np.median(fsi))
    high_fsi = [names[j] for j in range(P) if fsi[j] > 2 * median_fsi]
    if high_fsi:
        sections["Stability Concerns"] = f"Features with FSI > 2× median ({2 * median_fsi:.3f}):\n" + "\n".join(
            f"  - {n}" for n in high_fsi[:10]
        )
        warnings.append(
            f"{len(high_fsi)} features have high FSI (>2× median) — "
            f"these are likely collinear cluster members. "
            f"Report group importance, not individual features."
        )

    # --- Collinearity (if X_ref provided) ---
    if X_ref is not None:
        X_ref = np.asarray(X_ref)
        corr = np.abs(np.corrcoef(X_ref.T))
        n_high = 0
        pairs_str: list[str] = []
        for i in range(P):
            for j in range(i + 1, P):
                if corr[i, j] > 0.9:
                    n_high += 1
                    if len(pairs_str) < 5:
                        pairs_str.append(f"  - {names[i]} / {names[j]}: |r|={corr[i, j]:.3f}")
        if n_high > 0:
            sections["Collinearity"] = (
                f"{n_high} feature pairs with |r| > 0.9:\n"
                + "\n".join(pairs_str)
                + (f"\n  ... and {n_high - 5} more" if n_high > 5 else "")
            )
            warnings.append(
                f"{n_high} feature pairs have |r| > 0.9 — "
                f"individual feature rankings within these pairs are unreliable."
            )
        else:
            sections["Collinearity"] = "No feature pairs with |r| > 0.9 detected."

    # --- Optional enrichments ---
    if confidence is not None:
        ci = confidence.importance_ci
        wide = []
        for j in range(P):
            width = ci[j, 2] - ci[j, 0]
            if width > imp[j] * 0.5 and imp[j] > median_fsi:
                wide.append(names[j])
        if wide:
            sections["Confidence Intervals"] = f"Features with wide CIs (>50% of point estimate):\n" + "\n".join(
                f"  - {n}" for n in wide[:10]
            )

    if groups is not None:
        n_groups = len(groups.groups)
        sections["Feature Groups"] = f"{n_groups} feature groups detected via SHAP substitutability."

    if causal is not None:
        from collections import Counter

        counts = Counter(causal.flags)
        sections["Causal Flags"] = (
            f"  robust: {counts.get('robust', 0)}, "
            f"collinear: {counts.get('collinear', 0)}, "
            f"fragile: {counts.get('fragile', 0)}, "
            f"unimportant: {counts.get('unimportant', 0)}"
        )

    if partial_order is not None:
        sections["Ranking Certainty"] = (
            f"Partial order computed with {K} models. See partial_order.summary() for pairwise dominance probabilities."
        )

    # --- Model adequacy ---
    if K < 10:
        warnings.append(
            f"K={K} is below the recommended minimum of 10 for reliable diagnostics. Consider increasing M and K."
        )

    return AuditResult(
        sections={k: v for k, v in sections.items() if not k.startswith("_")},
        warnings=warnings,
        feature_names=names,
        K=K,
        P=P,
    )
