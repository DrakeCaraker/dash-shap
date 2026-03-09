"""Evaluation metrics for feature importance methods."""
import numpy as np
from scipy.stats import spearmanr, wilcoxon, norm
from typing import List, Tuple

__all__ = [
    "dgp_agreement",
    "importance_accuracy",
    "importance_stability",
    "stability_bootstrap_ci",
    "within_group_equity",
    "cohens_d",
    "compare_methods",
    "friedman_test",
]


def dgp_agreement(estimated, true):
    """Compute Spearman correlation and normalized MSE vs DGP-derived importance.

    Note: this metric presupposes that features contributing equally to the DGP
    should receive equal importance.  Under collinearity, SHAP may legitimately
    distribute credit unevenly depending on the model's internal structure.
    Report as a sanity check alongside stability and equity, not as the primary
    evaluation criterion.
    """
    rho, _ = spearmanr(estimated, true)
    est_norm = estimated / (estimated.sum() + 1e-10)
    true_norm = true / (true.sum() + 1e-10)
    mse = np.mean((est_norm - true_norm) ** 2)
    return rho, mse


# Backward-compatible alias
importance_accuracy = dgp_agreement


def importance_stability(vectors):
    """Compute mean pairwise Spearman correlation across importance vectors."""
    n = len(vectors)
    if n < 2:
        return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(vectors[i], vectors[j])
            corrs.append(rho)
    return float(np.mean(corrs))


def stability_bootstrap_ci(vectors, n_boot=1000, ci=0.95, seed=42):
    """BCa bootstrap confidence interval for importance_stability.

    Uses bias-corrected and accelerated (BCa) bootstrap instead of the
    percentile method, which corrects for both bias and skewness in the
    bootstrap distribution (A3 fix for publication).

    Returns (point, se, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    n = len(vectors)
    if n < 2:
        return 1.0, 0.0, 1.0, 1.0
    point = importance_stability(vectors)

    # Bootstrap distribution
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = [vectors[i] for i in idx]
        boots.append(importance_stability(sample))
    boots = np.array(boots)
    se = float(np.std(boots, ddof=1))

    # BCa bias-correction factor z0
    prop_below = np.mean(boots < point)
    # Clamp to avoid ±inf from ppf at exactly 0 or 1
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_below)

    # BCa acceleration factor from jackknife
    jack_stats = []
    for i in range(n):
        loo = [vectors[j] for j in range(n) if j != i]
        jack_stats.append(importance_stability(loo))
    jack_stats = np.array(jack_stats)
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2)) ** 1.5
    a = num / den if den != 0 else 0.0

    # Adjusted percentiles
    alpha = 1 - ci
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def _bca_percentile(z_alpha):
        adjusted = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return norm.cdf(adjusted) * 100

    ci_lo = float(np.percentile(boots, np.clip(_bca_percentile(z_lo), 0, 100)))
    ci_hi = float(np.percentile(boots, np.clip(_bca_percentile(z_hi), 0, 100)))
    return point, se, ci_lo, ci_hi


def within_group_equity(importance_vector, groups, include_zero_groups=False):
    """Compute mean coefficient of variation within feature groups.

    Groups whose mean absolute importance is near zero are excluded by default.
    Set ``include_zero_groups=True`` to score them: CV=0 when all values are
    near-zero (perfect equity), or ``np.inf`` otherwise (A2 robustness check).
    """
    cvs = []
    for g in np.unique(groups):
        gi = importance_vector[groups == g]
        if np.abs(gi.mean()) > 1e-10:
            cvs.append(gi.std() / np.abs(gi.mean()))
        elif include_zero_groups:
            cvs.append(0.0 if gi.std() < 1e-10 else float('inf'))
    return float(np.mean(cvs)) if cvs else 0.0


def cohens_d(g1, g2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
        / (n1 + n2 - 2)
    )
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-10 else 0.0


def compare_methods(a, b):
    """Wilcoxon signed-rank test between two sets of scores."""
    if np.allclose(a, b):
        return 0.0, 1.0
    stat, pval = wilcoxon(a, b)
    return float(stat), float(pval)


def friedman_test(*method_scores):
    """Friedman chi-square test across multiple methods."""
    from scipy.stats import friedmanchisquare
    stat, pval = friedmanchisquare(*method_scores)
    return float(stat), float(pval)
