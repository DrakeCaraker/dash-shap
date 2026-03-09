"""Evaluation metrics for feature importance methods."""
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from typing import List, Tuple

__all__ = [
    "importance_accuracy",
    "importance_stability",
    "stability_bootstrap_ci",
    "within_group_equity",
    "cohens_d",
    "compare_methods",
    "friedman_test",
]


def importance_accuracy(estimated, true):
    """Compute Spearman correlation and normalized MSE between estimated and true importance."""
    rho, _ = spearmanr(estimated, true)
    est_norm = estimated / (estimated.sum() + 1e-10)
    true_norm = true / (true.sum() + 1e-10)
    mse = np.mean((est_norm - true_norm) ** 2)
    return rho, mse


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
    """Bootstrap confidence interval for importance_stability.

    Resamples the list of importance vectors with replacement, recomputes
    stability on each bootstrap sample, and returns (point, se, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    n = len(vectors)
    if n < 2:
        return 1.0, 0.0, 1.0, 1.0
    point = importance_stability(vectors)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = [vectors[i] for i in idx]
        boots.append(importance_stability(sample))
    boots = np.array(boots)
    se = float(np.std(boots, ddof=1))
    alpha = 1 - ci
    ci_lo = float(np.percentile(boots, 100 * alpha / 2))
    ci_hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, se, ci_lo, ci_hi


def within_group_equity(importance_vector, groups):
    """Compute mean coefficient of variation within feature groups.

    Groups whose mean absolute importance is near zero are excluded from
    the average rather than being scored as perfect equity (CV=0).
    """
    cvs = []
    for g in np.unique(groups):
        gi = importance_vector[groups == g]
        if np.abs(gi.mean()) > 1e-10:
            cvs.append(gi.std() / np.abs(gi.mean()))
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
