"""Extension: Theory Bridge — impossibility-theorem-grounded diagnostics.

Implements formulas from the Attribution Impossibility theorem
(Caraker et al., 2026; Lean 4 verified) as practical diagnostics:

- compute_snr(): signal-to-noise ratio per feature pair
- predict_flip_rate(): Φ(-SNR) flip probability prediction
- recommend_M(): minimum ensemble size for target stability
- divergence_ratio(): 1/(1-ρ²) attribution divergence bound

References:
    FlipRate.lean — exact binary flip rate = 1/2
    EnsembleBound.lean — M_min = ⌈1.645² · σ²/Δ²⌉ = ⌈2.71 · σ²/Δ²⌉
    snr_calibration.py — empirical validation (OOS R² ≈ 0.85)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = [
    "theory_bridge",
    "TheoryBridgeResult",
    "compute_snr",
    "predict_flip_rate",
    "recommend_M",
    "divergence_ratio",
]

# ── Standalone functions (usable without DASHResult) ──────────────────────


def compute_snr(importance_matrix: np.ndarray) -> dict:
    """Compute signal-to-noise ratio for each feature pair.

    For features j, k:
        SNR_{jk} = |mean(φ_j - φ_k)| / std(φ_j - φ_k)

    where φ_j is mean |SHAP| for feature j across models.

    Parameters
    ----------
    importance_matrix : ndarray of shape (M, P)
        Per-model mean absolute SHAP importance. Rows = models, columns = features.

    Returns
    -------
    dict of {(j, k): float}
        SNR for each feature pair (j < k). Higher SNR = more distinguishable.

    Notes
    -----
    Complexity is O(M·P²) — iterates over all C(P,2) pairs. Fast for
    P ≤ 100 (typical); for P > 200, consider filtering to correlated
    pairs first.
    """
    importance_matrix = np.asarray(importance_matrix, dtype=float)
    if importance_matrix.ndim != 2:
        raise ValueError(f"importance_matrix must be 2D (M, P), got {importance_matrix.ndim}D")
    M, P = importance_matrix.shape
    snr = {}
    for j in range(P):
        for k in range(j + 1, P):
            diffs = importance_matrix[:, j] - importance_matrix[:, k]
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            if std_diff > 1e-12:
                snr[(j, k)] = abs(float(mean_diff)) / float(std_diff)
            elif abs(mean_diff) < 1e-12:
                snr[(j, k)] = 0.0  # indistinguishable features → coin flip
            else:
                snr[(j, k)] = float("inf")  # perfectly separated
    return snr


def predict_flip_rate(snr: float) -> float:
    """Predict feature-pair flip rate from SNR using Φ(-SNR).

    The theoretical prediction from FlipRate.lean: for symmetric features
    under the Rashomon property, the flip rate follows the Gaussian CDF
    at -SNR. Validated empirically across multiple datasets with
    OOS R² ≈ 0.85 (snr_calibration.py in dash-impossibility-lean;
    gaussian_flip_cv.py in universal-explanation-impossibility).

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio (non-negative). SNR=0 → flip rate 0.5 (coin flip).

    Returns
    -------
    float
        Predicted flip rate in [0, 0.5].
    """
    if snr < 0:
        raise ValueError(f"SNR must be non-negative, got {snr}")
    return float(norm.cdf(-snr))


def recommend_M(
    importance_matrix: np.ndarray,
    target_flip_rate: float = 0.05,
    min_M: int = 10,
    max_M: int = 1000,
) -> dict:
    """Recommend minimum ensemble size M for target stability.

    Uses the formula from EnsembleBound.lean (Cauchy-Schwarz optimality):
        M_min = ⌈z² · σ² / Δ²⌉
    where z = Φ⁻¹(1 - target_flip_rate), σ² = variance of importance
    differences across models, and Δ = mean importance gap.

    For target_flip_rate=0.05: z=1.645, z²=2.71.

    Parameters
    ----------
    importance_matrix : ndarray of shape (M_pilot, P)
        Per-model importance from a pilot run (e.g., M=25 from check()).
    target_flip_rate : float
        Desired maximum flip rate for the most contested pair (default 0.05).
    min_M : int
        Floor on recommendation (default 10).
    max_M : int
        Ceiling on recommendation (default 1000).

    Returns
    -------
    dict with keys:
        recommended_M : int
            Recommended ensemble size.
        worst_pair : tuple (j, k)
            Feature pair with highest variance (driving the recommendation).
        worst_pair_snr : float
            Current SNR for the worst pair.
        worst_pair_flip : float
            Current predicted flip rate for the worst pair.
        z_critical : float
            The z-score threshold used.
        note : str
            Caveat about the formula's assumptions.
    """
    importance_matrix = np.asarray(importance_matrix, dtype=float)
    if importance_matrix.ndim != 2:
        raise ValueError(f"importance_matrix must be 2D (M, P), got {importance_matrix.ndim}D")
    M_pilot, P = importance_matrix.shape

    if target_flip_rate <= 0 or target_flip_rate >= 0.5:
        raise ValueError(f"target_flip_rate must be in (0, 0.5), got {target_flip_rate}")

    z_critical = float(norm.ppf(1 - target_flip_rate))
    z_sq = z_critical**2

    # Find worst pair: highest σ²/Δ² ratio
    worst_ratio = 0.0
    worst_pair = (0, 1)
    worst_snr = float("inf")

    for j in range(P):
        for k in range(j + 1, P):
            diffs = importance_matrix[:, j] - importance_matrix[:, k]
            sigma_sq = float(np.var(diffs, ddof=1))
            delta = abs(float(np.mean(diffs)))

            if delta < 1e-12:
                # Features are indistinguishable — infinite M needed
                # (these are genuinely tied features; report max_M)
                ratio = float("inf")
            else:
                ratio = sigma_sq / (delta**2)

            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_pair = (j, k)
                std_diff = np.std(diffs, ddof=1)
                worst_snr = abs(float(np.mean(diffs))) / float(std_diff) if std_diff > 1e-12 else 0.0

    if worst_ratio == float("inf"):
        M_rec = max_M
    else:
        M_rec = int(math.ceil(z_sq * worst_ratio))

    M_rec = max(min_M, min(M_rec, max_M))

    return {
        "recommended_M": M_rec,
        "worst_pair": worst_pair,
        "worst_pair_snr": worst_snr,
        "worst_pair_flip": predict_flip_rate(worst_snr),
        "z_critical": z_critical,
        "note": (
            "Based on ensemble_bound_formula in EnsembleBound.lean (Lean 4 verified). "
            "Assumes Gaussian importance differences and equal-weight averaging "
            "(proved optimal via Cauchy-Schwarz in same file). "
            "The formula is theoretically derived but not yet experimentally "
            "validated end-to-end; treat as a principled starting point. "
            f"Pilot run used M={M_pilot}; estimate improves with larger pilots."
        ),
    }


def divergence_ratio(rho: float) -> float:
    """Theoretical attribution divergence ratio 1/(1-ρ²).

    From Ratio.lean: the ratio of first-mover to non-first-mover
    attribution diverges as 1/(1-ρ²) when correlation ρ → 1.

    Parameters
    ----------
    rho : float
        Feature correlation in [-1, 1]. |ρ| < 1 required.

    Returns
    -------
    float
        Divergence ratio. Approaches infinity as |ρ| → 1.
    """
    if abs(rho) >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - rho**2)


# ── DASHResult extension ─────────────────────────────────────────────────


@dataclass
class TheoryBridgeResult:
    """Result of theory_bridge().

    Attributes
    ----------
    snr : dict of {(j, k): float}
        Per-pair signal-to-noise ratio.
    predicted_flip_rates : dict of {(j, k): float}
        Φ(-SNR) predicted flip rate per pair.
    recommended_M : int
        Suggested ensemble size for 5% target flip rate.
    recommendation_details : dict
        Full output from recommend_M().
    unstable_pairs : list of (j, k)
        Pairs with predicted flip rate > 0.10 (likely unstable).
    feature_names : list of str
    K : int
        Number of models in the DASHResult.
    """

    snr: dict
    predicted_flip_rates: dict
    recommended_M: int
    recommendation_details: dict
    unstable_pairs: list
    feature_names: list
    K: int

    def summary(self, top_k: int = 10) -> str:
        """Human-readable theory bridge summary."""
        lines = [
            f"TheoryBridgeResult: {len(self.feature_names)} features, K={self.K}",
            f"Recommended M: {self.recommended_M} (for 5% target flip rate)",
            "",
        ]

        # Sort pairs by predicted flip rate (highest first)
        sorted_pairs = sorted(self.predicted_flip_rates.items(), key=lambda x: -x[1])

        if sorted_pairs:
            # Compute dynamic column width from feature names
            display_pairs = sorted_pairs[:top_k]
            max_pair_len = 4  # minimum: "Pair"
            for (j, k), _ in display_pairs:
                nj = self.feature_names[j] if j < len(self.feature_names) else f"f{j}"
                nk = self.feature_names[k] if k < len(self.feature_names) else f"f{k}"
                max_pair_len = max(max_pair_len, len(f"{nj} vs {nk}"))
            col_w = max_pair_len + 2

            lines.append(f"{'Pair':<{col_w}} {'SNR':>6} {'Predicted Flip':>14}")
            lines.append("-" * (col_w + 23))
            for (j, k), flip in display_pairs:
                nj = self.feature_names[j] if j < len(self.feature_names) else f"f{j}"
                nk = self.feature_names[k] if k < len(self.feature_names) else f"f{k}"
                snr_val = self.snr.get((j, k), float("inf"))
                flag = " ***" if flip > 0.10 else ""
                pair_str = f"{nj} vs {nk}"
                lines.append(f"{pair_str:<{col_w}} {snr_val:>6.2f} {flip:>13.1%}{flag}")

        if self.unstable_pairs:
            lines.append("")
            lines.append(f"*** {len(self.unstable_pairs)} pairs predicted unstable (flip > 10%)")

        lines.append("")
        lines.append(f"Note: {self.recommendation_details['note']}")
        return "\n".join(lines)

    def plot(self):
        """SNR vs predicted flip rate scatter plot with Φ(-SNR) curve."""
        import matplotlib.pyplot as plt

        snr_vals = np.array(list(self.snr.values()))
        flip_vals = np.array(list(self.predicted_flip_rates.values()))

        # Filter out inf SNR
        mask = np.isfinite(snr_vals)
        snr_plot = snr_vals[mask]
        flip_plot = flip_vals[mask]

        fig, ax = plt.subplots(figsize=(8, 5))

        # Theoretical curve
        x_max = float(np.max(snr_plot) * 1.1) if len(snr_plot) > 0 else 5.0
        x = np.linspace(0, max(5.0, x_max), 200)
        ax.plot(x, norm.cdf(-x), "k-", linewidth=1.5, label="Φ(−SNR) theory", zorder=1)

        # Empirical points (may be empty if all pairs are perfectly separated)
        if len(snr_plot) > 0:
            colors = ["tab:red" if f > 0.10 else "tab:blue" for f in flip_plot]
            ax.scatter(snr_plot, flip_plot, c=colors, s=30, alpha=0.7, zorder=2, edgecolors="none")
        else:
            ax.text(
                0.5,
                0.3,
                "All feature pairs perfectly separated\n(no finite SNR values)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
            )

        ax.axhline(0.10, color="gray", linestyle="--", alpha=0.5, label="10% flip threshold")
        ax.set_xlabel("SNR = |E[Δφ]| / SD(Δφ)")
        ax.set_ylabel("Predicted flip rate")
        ax.set_title("Theory Bridge: SNR → Flip Rate")
        ax.legend()
        ax.set_ylim(-0.02, 0.55)
        fig.tight_layout()
        return fig


def theory_bridge(
    result: "DASHResult",
    target_flip_rate: float = 0.05,
    unstable_threshold: float = 0.10,
) -> TheoryBridgeResult:
    """Apply impossibility-theorem diagnostics to a DASHResult.

    Computes per-pair SNR, predicts flip rates via Φ(-SNR), and recommends
    ensemble size M for a target stability level.

    Parameters
    ----------
    result : DASHResult
    target_flip_rate : float
        Target flip rate for ensemble size recommendation (default 0.05).
    unstable_threshold : float
        Pairs with predicted flip rate above this are flagged (default 0.10).

    Returns
    -------
    TheoryBridgeResult
    """
    from dash_shap.extensions._base import per_model_importance

    importance = per_model_importance(result)  # (K, P)

    snr = compute_snr(importance)
    predicted = {pair: predict_flip_rate(s) for pair, s in snr.items()}
    rec = recommend_M(importance, target_flip_rate=target_flip_rate)
    unstable = [(j, k) for (j, k), f in predicted.items() if f > unstable_threshold]

    return TheoryBridgeResult(
        snr=snr,
        predicted_flip_rates=predicted,
        recommended_M=rec["recommended_M"],
        recommendation_details=rec,
        unstable_pairs=unstable,
        feature_names=list(result.feature_names),
        K=result.K,
    )
