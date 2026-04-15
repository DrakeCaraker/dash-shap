"""Extension: Drift Monitor — detect importance drift between model versions.

Compares a baseline DASHResult against current results using cosine
distance on global importance vectors. Tracks a timeline of checks
for monitoring explanation stability over deployments.

Usage:
    monitor = DriftMonitor(baseline_result, threshold=0.1)
    alert = monitor.check(current_result, label="2026-Q2")
    if alert.drifted:
        print(f"Drift detected: {alert.distance:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["DriftMonitor", "DriftAlert"]


@dataclass
class DriftAlert:
    """Result of DriftMonitor.check().

    Attributes
    ----------
    drifted : bool
        True if cosine distance exceeds the monitor's threshold.
    distance : float
        Cosine distance between baseline and current global importance.
    changed_features : list of str
        Features whose importance rank changed by >= 2 positions.
    label : str or None
        User-provided label for this check (e.g., date or version).
    """

    drifted: bool
    distance: float
    changed_features: list
    label: str | None = None

    def summary(self) -> str:
        status = "DRIFT DETECTED" if self.drifted else "No drift"
        lines = [
            f"DriftAlert: {status} (distance={self.distance:.4f})",
        ]
        if self.label:
            lines[0] += f" [{self.label}]"
        if self.changed_features:
            lines.append(f"Changed features (rank shift >= 2): {', '.join(self.changed_features)}")
        return "\n".join(lines)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors. Returns 0 for identical, 1 for orthogonal."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return 1.0 - float(dot / (norm_a * norm_b))


def _rank(arr: np.ndarray) -> np.ndarray:
    """Return 1-based ranks (1 = most important)."""
    order = np.argsort(-arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


class DriftMonitor:
    """Monitor explanation drift between model versions.

    Parameters
    ----------
    baseline : DASHResult
        The reference result to compare against.
    threshold : float
        Cosine distance threshold for flagging drift (default 0.1).
        Lower = more sensitive. Typical values: 0.05 (strict), 0.1 (moderate), 0.2 (lenient).
    rank_shift : int
        Minimum rank change to flag a feature as "changed" (default 2).
    """

    def __init__(self, baseline: "DASHResult", threshold: float = 0.1, rank_shift: int = 2):
        self.baseline = baseline
        self.threshold = threshold
        self.rank_shift = rank_shift
        self._baseline_imp = baseline.global_importance.copy()
        self._baseline_ranks = _rank(self._baseline_imp)
        self._timeline: list[DriftAlert] = []

    def check(self, current: "DASHResult", label: str | None = None) -> DriftAlert:
        """Compare current result against baseline.

        Parameters
        ----------
        current : DASHResult
            The new result to compare.
        label : str or None
            Optional label (e.g., "2026-Q2", "v3.1").

        Returns
        -------
        DriftAlert
        """
        if current.P != self.baseline.P:
            raise ValueError(f"Feature count mismatch: baseline has {self.baseline.P}, current has {current.P}")

        current_imp = current.global_importance
        distance = _cosine_distance(self._baseline_imp, current_imp)

        current_ranks = _rank(current_imp)
        rank_changes = np.abs(self._baseline_ranks.astype(int) - current_ranks.astype(int))
        changed = [self.baseline.feature_names[j] for j in range(self.baseline.P) if rank_changes[j] >= self.rank_shift]

        alert = DriftAlert(
            drifted=distance > self.threshold,
            distance=distance,
            changed_features=changed,
            label=label,
        )
        self._timeline.append(alert)
        return alert

    def plot_timeline(self):
        """Plot drift distance over time for all checks."""
        import matplotlib.pyplot as plt

        if not self._timeline:
            raise ValueError("No checks recorded yet. Call .check() first.")

        distances = [a.distance for a in self._timeline]
        labels = [a.label or f"check_{i}" for i, a in enumerate(self._timeline)]
        colors = ["tab:red" if a.drifted else "tab:blue" for a in self._timeline]

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(distances))
        ax.bar(x, distances, color=colors, alpha=0.7)
        ax.axhline(self.threshold, color="gray", linestyle="--", label=f"threshold={self.threshold}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Cosine distance from baseline")
        ax.set_title("Explanation Drift Timeline")
        ax.legend()
        fig.tight_layout()
        return fig
