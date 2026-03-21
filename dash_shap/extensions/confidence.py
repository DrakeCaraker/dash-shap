"""Extension 1: Confidence Intervals for feature importance, FSI, and rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["confidence_intervals", "ConfidenceResult"]


def _bca_ci(bootstrap_samples: np.ndarray, observed: float, alpha: float) -> tuple:
    """BCa bootstrap confidence interval for a scalar.

    Falls back to percentile CI when BCa quantities are numerically unstable
    (e.g., all bootstrap samples identical, or K too small for jackknife).

    Returns
    -------
    (lower, point, upper)
    """
    from scipy.stats import norm

    bootstrap_samples = np.asarray(bootstrap_samples, dtype=float)
    n_boot = len(bootstrap_samples)

    # If all bootstrap samples are essentially the same, fall back to percentile
    if np.ptp(bootstrap_samples) < 1e-12:
        val = float(bootstrap_samples[0])
        return val, observed, val

    # Bias correction
    prop = float(np.mean(bootstrap_samples < observed))
    prop = np.clip(prop, 1e-6, 1 - 1e-6)
    z0 = float(norm.ppf(prop))

    # Acceleration (jackknife) — use a subsample for speed
    n_jk = min(n_boot, 200)
    jk = np.array([np.mean(np.delete(bootstrap_samples[:n_jk], i)) for i in range(n_jk)])
    jk_mean = np.mean(jk)
    num: float = float(np.sum((jk_mean - jk) ** 3))
    denom = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    accel = float(num / denom) if abs(denom) > 1e-12 else 0.0

    # Clip accel to avoid division by zero in 1 - accel * (z0 + z)
    accel = np.clip(accel, -10.0, 10.0)

    z_alpha = norm.ppf(alpha / 2)
    z_1malpha = norm.ppf(1 - alpha / 2)

    def _adj_pct(z_target):
        denom_val = 1.0 - accel * (z0 + z_target)
        if abs(denom_val) < 1e-8:
            return alpha / 2 if z_target < 0 else 1 - alpha / 2
        return float(np.clip(norm.cdf(z0 + (z0 + z_target) / denom_val), 1e-4, 1 - 1e-4))

    a1 = _adj_pct(z_alpha)
    a2 = _adj_pct(z_1malpha)

    if a1 >= a2:
        # Degenerate case — fall back to simple percentile CI
        a1 = alpha / 2
        a2 = 1 - alpha / 2

    lower = float(np.percentile(bootstrap_samples, 100 * a1))
    upper = float(np.percentile(bootstrap_samples, 100 * a2))
    # Ensure CI contains the point estimate (numerical safety)
    lower = min(lower, observed)
    upper = max(upper, observed)
    return lower, observed, upper


@dataclass
class ConfidenceResult:
    """Result of confidence_intervals().

    Attributes
    ----------
    importance_ci : ndarray (P, 3) — lower, point, upper for global importance
    fsi_ci : ndarray (P, 3) — lower, point, upper for FSI
    ranking_ci : ndarray (P, 3) — lower, point, upper for mean rank
        Ranks are treated as continuous scores via the bootstrap (Spearman's rank
        treated as a smooth quantity for CI construction); values are floats.
    feature_names : list[str]
    alpha : float — significance level used
    n_boot : int — number of bootstrap replicates used
    """

    importance_ci: np.ndarray  # (P, 3)
    fsi_ci: np.ndarray  # (P, 3)
    ranking_ci: np.ndarray  # (P, 3) — float, not int
    feature_names: list
    alpha: float
    n_boot: int

    def summary(self) -> str:
        lines = [
            f"ConfidenceResult: P={len(self.feature_names)}, alpha={self.alpha}, n_boot={self.n_boot}",
            "",
            f"{'Feature':<20} {'Importance CI':>22} {'FSI CI':>22} {'Rank CI':>22}",
            "-" * 90,
        ]
        for i, name in enumerate(self.feature_names):
            imp = self.importance_ci[i]
            fsi = self.fsi_ci[i]
            rnk = self.ranking_ci[i]
            lines.append(
                f"{name:<20} "
                f"[{imp[0]:.3f}, {imp[1]:.3f}, {imp[2]:.3f}]  "
                f"[{fsi[0]:.3f}, {fsi[1]:.3f}, {fsi[2]:.3f}]  "
                f"[{rnk[0]:.1f}, {rnk[1]:.1f}, {rnk[2]:.1f}]"
            )
        return "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        P = len(self.feature_names)
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, P * 0.4)))

        for ax, ci_mat, title in [
            (axes[0], self.importance_ci, "Global Importance CI"),
            (axes[1], self.fsi_ci, "FSI CI"),
        ]:
            y = np.arange(P)
            centers = ci_mat[:, 1]
            xerr = np.array([centers - ci_mat[:, 0], ci_mat[:, 2] - centers])
            ax.barh(y, centers, xerr=xerr, align="center", height=0.6, color="steelblue", alpha=0.7, capsize=4)
            ax.set_yticks(y)
            ax.set_yticklabels(self.feature_names)
            ax.set_title(title)
            ax.invert_yaxis()

        fig.tight_layout()
        return fig


def confidence_intervals(
    result: "DASHResult",
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 42,
) -> ConfidenceResult:
    """Compute BCa bootstrap confidence intervals for feature importance, FSI, and rankings.

    Memory-safe: resamples K model indices without copying the full tensor.

    Parameters
    ----------
    result : DASHResult
    alpha : float
        Significance level (default 0.05 → 95% CI).
    n_boot : int
        Number of bootstrap replicates.
    seed : int

    Returns
    -------
    ConfidenceResult

    Notes
    -----
    Requires K ≥ 10 for reliable BCa coverage. With K < 10, results may be
    unreliable — consider a simpler percentile CI.
    """
    from dash_shap.extensions._base import bootstrap_over_models, per_model_rankings

    eps = 1e-8
    K = result.K

    if K < 10:
        import warnings

        warnings.warn(
            f"K={K} < 10: BCa confidence intervals may have poor coverage. Consider K >= 10 for reliable results.",
            UserWarning,
            stacklevel=2,
        )

    def _importance_stat(shap_sample):
        consensus = np.mean(shap_sample, axis=0)  # (n_ref, P)
        return np.mean(np.abs(consensus), axis=0)  # (P,)

    def _fsi_stat(shap_sample):
        consensus = np.mean(shap_sample, axis=0)
        variance = np.var(shap_sample, axis=0, ddof=1)
        global_imp = np.mean(np.abs(consensus), axis=0)
        mean_std = np.mean(np.sqrt(variance), axis=0)
        return mean_std / (global_imp + eps)

    def _ranking_stat(shap_sample):
        # Use per-model importance then average ranks
        imp = np.mean(np.abs(shap_sample), axis=1)  # (K_boot, P)
        order = np.argsort(-imp, axis=1)
        P = imp.shape[1]
        ranks = np.empty_like(order, dtype=float)
        for k in range(len(shap_sample)):
            ranks[k, order[k]] = np.arange(1, P + 1)
        return np.mean(ranks, axis=0)  # (P,)

    boot_importance = bootstrap_over_models(result, _importance_stat, n_boot, seed)
    boot_fsi = bootstrap_over_models(result, _fsi_stat, n_boot, seed + 1)
    boot_ranking = bootstrap_over_models(result, _ranking_stat, n_boot, seed + 2)

    P = result.P
    importance_ci = np.zeros((P, 3))
    fsi_ci = np.zeros((P, 3))
    ranking_ci = np.zeros((P, 3))

    all_rankings: np.ndarray = per_model_rankings(result).astype(float)  # (K, P) — computed once
    for p in range(P):
        importance_ci[p] = _bca_ci(boot_importance[:, p], float(result.global_importance[p]), alpha)
        fsi_ci[p] = _bca_ci(boot_fsi[:, p], float(result.fsi[p]), alpha)
        # Point estimate for ranking: mean rank across K models
        mean_rank_p = float(np.mean(all_rankings[:, p]))
        ranking_ci[p] = _bca_ci(boot_ranking[:, p], mean_rank_p, alpha)

    return ConfidenceResult(
        importance_ci=importance_ci,
        fsi_ci=fsi_ci,
        ranking_ci=ranking_ci,
        feature_names=list(result.feature_names),
        alpha=alpha,
        n_boot=n_boot,
    )
