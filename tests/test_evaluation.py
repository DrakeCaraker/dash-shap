"""Tests for dash.evaluation metrics."""
import numpy as np
from dash.evaluation import (
    importance_accuracy,
    importance_stability,
    within_group_equity,
    cohens_d,
    compare_methods,
)


def test_importance_accuracy_perfect():
    true = np.array([1.0, 0.5, 0.25, 0.1])
    rho, mse = importance_accuracy(true, true)
    assert rho > 0.99
    assert mse < 1e-10


def test_importance_accuracy_inverted():
    true = np.array([1.0, 0.5, 0.25, 0.1])
    inverted = true[::-1]
    rho, _ = importance_accuracy(inverted, true)
    assert rho < 0


def test_importance_stability_identical():
    vectors = [np.array([1.0, 0.5, 0.3])] * 3
    stab = importance_stability(vectors)
    assert stab > 0.99


def test_importance_stability_single():
    vectors = [np.array([1.0, 0.5, 0.3])]
    assert importance_stability(vectors) == 1.0


def test_within_group_equity_uniform():
    importance = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
    groups = np.array([0, 0, 0, 1, 1, 1])
    eq = within_group_equity(importance, groups)
    assert eq < 1e-10


def test_within_group_equity_nonuniform():
    importance = np.array([1.0, 0.1, 0.5, 0.3, 0.3, 0.3])
    groups = np.array([0, 0, 0, 1, 1, 1])
    eq = within_group_equity(importance, groups)
    assert eq > 0


def test_cohens_d_identical():
    g = np.array([1.0, 2.0, 3.0])
    assert cohens_d(g, g) == 0.0


def test_cohens_d_different():
    g1 = np.array([10.0, 10.5, 11.0])
    g2 = np.array([1.0, 1.5, 2.0])
    d = cohens_d(g1, g2)
    assert d > 0


def test_compare_methods_identical():
    a = np.array([1.0, 2.0, 3.0])
    stat, pval = compare_methods(a, a)
    assert pval == 1.0


def test_bootstrap_stability_ci_basic():
    """Verify bootstrap CI returns lo <= mean <= hi."""
    from scipy.stats import spearmanr, norm
    from numpy.random import RandomState

    def bootstrap_stability_ci(imp_vectors, n_bootstrap=200, alpha=0.05, seed=42):
        rng = RandomState(seed)
        n = len(imp_vectors)
        obs_stat = importance_stability(imp_vectors)
        boot_stabs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            unique_pairs_corrs = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if idx[i] != idx[j]:
                        rho, _ = spearmanr(imp_vectors[idx[i]], imp_vectors[idx[j]])
                        unique_pairs_corrs.append(rho)
            if unique_pairs_corrs:
                boot_stabs.append(float(np.mean(unique_pairs_corrs)))
        boot_stabs = np.array(boot_stabs)
        z0 = norm.ppf(np.mean(boot_stabs < obs_stat))
        jack_stats = []
        for i in range(n):
            loo = [imp_vectors[j] for j in range(n) if j != i]
            jack_stats.append(importance_stability(loo))
        jack_stats = np.array(jack_stats)
        jack_mean = np.mean(jack_stats)
        num = np.sum((jack_mean - jack_stats) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2)) ** 1.5
        a = num / den if den != 0 else 0.0
        z_alpha_lo = norm.ppf(alpha / 2)
        z_alpha_hi = norm.ppf(1 - alpha / 2)
        p_lo = norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
        p_hi = norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))
        lo = np.percentile(boot_stabs, 100 * p_lo)
        hi = np.percentile(boot_stabs, 100 * p_hi)
        return np.mean(boot_stabs), lo, hi

    rng = np.random.RandomState(123)
    vectors = [rng.randn(10) for _ in range(10)]
    mean_s, lo, hi = bootstrap_stability_ci(vectors)
    assert lo <= mean_s <= hi
    assert lo < hi  # Meaningful interval


def test_bootstrap_stability_ci_tight_for_identical():
    """Verify CI is tight when importance vectors are identical."""
    from scipy.stats import spearmanr, norm
    from numpy.random import RandomState

    def bootstrap_stability_ci(imp_vectors, n_bootstrap=200, alpha=0.05, seed=42):
        rng = RandomState(seed)
        n = len(imp_vectors)
        obs_stat = importance_stability(imp_vectors)
        boot_stabs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            unique_pairs_corrs = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if idx[i] != idx[j]:
                        rho, _ = spearmanr(imp_vectors[idx[i]], imp_vectors[idx[j]])
                        unique_pairs_corrs.append(rho)
            if unique_pairs_corrs:
                boot_stabs.append(float(np.mean(unique_pairs_corrs)))
        boot_stabs = np.array(boot_stabs)
        z0 = norm.ppf(np.mean(boot_stabs < obs_stat))
        jack_stats = []
        for i in range(n):
            loo = [imp_vectors[j] for j in range(n) if j != i]
            jack_stats.append(importance_stability(loo))
        jack_stats = np.array(jack_stats)
        jack_mean = np.mean(jack_stats)
        num = np.sum((jack_mean - jack_stats) ** 3)
        den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2)) ** 1.5
        a = num / den if den != 0 else 0.0
        z_alpha_lo = norm.ppf(alpha / 2)
        z_alpha_hi = norm.ppf(1 - alpha / 2)
        p_lo = norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
        p_hi = norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))
        lo = np.percentile(boot_stabs, 100 * p_lo)
        hi = np.percentile(boot_stabs, 100 * p_hi)
        return np.mean(boot_stabs), lo, hi

    base = np.array([1.0, 0.5, 0.3, 0.1, 0.8])
    # Near-identical vectors (small noise) should produce high stability and tight CI
    rng = np.random.RandomState(99)
    vectors = [base + rng.randn(5) * 0.001 for _ in range(10)]
    mean_s, lo, hi = bootstrap_stability_ci(vectors)
    assert mean_s > 0.95
    assert hi - lo < 0.05  # Tight interval
