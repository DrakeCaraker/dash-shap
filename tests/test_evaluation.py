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
    assert np.isnan(importance_stability(vectors))


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


def test_group_level_accuracy_perfect():
    """Group-level accuracy should be 1.0 when group sums match ranking."""
    from dash.evaluation import group_level_accuracy
    groups = np.array([0, 0, 0, 1, 1, 1])
    true = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    # Estimated has different within-group distribution but correct group sums
    estimated = np.array([0.8, 0.3, 0.4, 0.05, 0.15, 0.1])
    rho = group_level_accuracy(estimated, true, groups)
    assert rho > 0.99


def test_group_level_accuracy_inverted():
    """Group-level accuracy should be negative when ranking is reversed."""
    from dash.evaluation import group_level_accuracy
    groups = np.array([0, 0, 1, 1])
    true = np.array([1.0, 1.0, 0.1, 0.1])
    estimated = np.array([0.05, 0.05, 0.9, 0.9])
    rho = group_level_accuracy(estimated, true, groups)
    assert rho < 0


def test_group_level_mse_nonzero_when_proportions_differ():
    """Group-level MSE should be > 0 when group proportions don't match."""
    from dash.evaluation import group_level_mse
    groups = np.array([0, 0, 0, 1, 1, 1])
    true = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    # Group sums: est=[1.5, 0.1], true=[1.5, 0.3] — proportions differ
    estimated = np.array([0.3, 0.4, 0.8, 0.05, 0.02, 0.03])
    mse = group_level_mse(estimated, true, groups)
    assert mse > 0


def test_group_level_mse_zero_when_proportions_match():
    """Group-level MSE should be ~0 when group proportions match."""
    from dash.evaluation import group_level_mse
    groups = np.array([0, 0, 1, 1])
    true = np.array([0.6, 0.6, 0.2, 0.2])
    # Same group proportions (3:1 ratio)
    estimated = np.array([0.9, 0.9, 0.3, 0.3])
    mse = group_level_mse(estimated, true, groups)
    assert mse < 1e-10


def test_holm_bonferroni_basic():
    """Holm-Bonferroni adjusted p-values are >= raw and smallest survives."""
    from dash.evaluation import holm_bonferroni
    p_values = [0.01, 0.04, 0.03, 0.005]
    adjusted = holm_bonferroni(p_values)
    assert all(a >= p for a, p in zip(adjusted, p_values))
    assert adjusted[3] < 0.05  # Most significant should survive at alpha=0.05


def test_holm_bonferroni_less_conservative_than_bonferroni():
    """Holm-Bonferroni should be less conservative than Bonferroni."""
    from dash.evaluation import holm_bonferroni
    p_values = [0.01, 0.04, 0.03, 0.005]
    adjusted = holm_bonferroni(p_values)
    bonferroni = [min(p * len(p_values), 1.0) for p in p_values]
    # At least one adjusted value should be strictly less than Bonferroni
    assert any(a < b for a, b in zip(adjusted, bonferroni))


def test_tost_equivalence_small_diff():
    """Small differences with large delta should be equivalent."""
    from dash.evaluation import tost_equivalence
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    t1, p1, t2, p2, equiv = tost_equivalence(a, b, delta=0.5)
    assert isinstance(equiv, bool)
    assert equiv is True


def test_tost_equivalence_large_diff():
    """Large differences with small delta should NOT be equivalent."""
    from dash.evaluation import tost_equivalence
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    c = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    _, _, _, _, equiv = tost_equivalence(a, c, delta=0.1)
    assert equiv is False


def test_bootstrap_stability_test_significant():
    """Clearly different importance vectors should yield significant p-value."""
    from dash.evaluation import bootstrap_stability_test
    rng = np.random.RandomState(42)
    # Method A: stable importance (same order every time)
    imp_a = [np.array([5.0, 3.0, 1.0, 0.5, 0.1]) + rng.randn(5) * 0.01 for _ in range(20)]
    # Method B: noisy importance (shuffled)
    imp_b = [rng.permutation([5.0, 3.0, 1.0, 0.5, 0.1]) for _ in range(20)]
    diff, pval, ci_lo, ci_hi = bootstrap_stability_test(imp_a, imp_b, n_bootstrap=1000)
    assert diff > 0  # A is more stable than B
    assert pval < 0.05


def test_bootstrap_stability_test_equivalent():
    """Nearly identical vectors should yield non-significant p-value."""
    from dash.evaluation import bootstrap_stability_test
    rng = np.random.RandomState(42)
    base = np.array([5.0, 3.0, 1.0, 0.5, 0.1])
    imp_a = [base + rng.randn(5) * 0.01 for _ in range(20)]
    imp_b = [base + rng.randn(5) * 0.01 for _ in range(20)]
    diff, pval, ci_lo, ci_hi = bootstrap_stability_test(imp_a, imp_b, n_bootstrap=1000)
    assert abs(diff) < 0.1  # small difference
    assert pval > 0.01  # not significant


def test_topk_overlap_stability_perfect():
    """Identical rankings should give Jaccard = 1.0."""
    from dash.evaluation import topk_overlap_stability
    v = [np.array([10, 8, 6, 1, 1]), np.array([9, 7, 5, 1, 1])]
    assert topk_overlap_stability(v, k=3) == 1.0


def test_topk_overlap_stability_random():
    """Random vectors should have lower overlap than identical ones."""
    from dash.evaluation import topk_overlap_stability
    rng = np.random.RandomState(42)
    v = [rng.rand(50) for _ in range(10)]
    assert topk_overlap_stability(v, k=5) < 0.8


def test_topk_overlap_stability_single():
    """Single vector should return 1.0."""
    from dash.evaluation import topk_overlap_stability
    assert topk_overlap_stability([np.array([1, 2, 3])], k=2) == 1.0


def test_fsi_collinearity_correlation_known():
    """High FSI should correlate with high collinearity."""
    from dash.evaluation import fsi_collinearity_correlation
    fsi = np.array([0.1, 0.1, 0.8, 0.9, 0.05])
    rho = np.array([0.0, 0.0, 0.9, 0.9, 0.0])
    result = fsi_collinearity_correlation(fsi, rho)
    assert result['feature_spearman'] > 0.5
    assert result['feature_pvalue'] < 0.2


def test_fsi_collinearity_correlation_with_groups():
    """Group-level correlation should also work."""
    from dash.evaluation import fsi_collinearity_correlation
    fsi = np.array([0.1, 0.1, 0.8, 0.9, 0.05, 0.85])
    rho = np.array([0.0, 0.0, 0.9, 0.9, 0.0, 0.9])
    groups = np.array([0, 0, 1, 1, 2, 1])
    result = fsi_collinearity_correlation(fsi, rho, groups=groups)
    assert 'group_spearman' in result
    assert 'group_pvalue' in result


def test_performance_filter_relative():
    """Relative epsilon mode filters based on fraction of best score."""
    from dash.core.filtering import performance_filter
    # Best score is -0.5 (higher is better for negated RMSE)
    scores = {0: -0.5, 1: -0.55, 2: -1.0, 3: -0.52}
    # relative epsilon=0.1 means within 10% of |best|=0.5 → threshold=0.05
    filtered = performance_filter(scores, epsilon=0.1, mode='relative', verbose=False)
    assert 0 in filtered  # best
    assert 3 in filtered  # within 0.02 of best
    assert 2 not in filtered  # -1.0 is 0.5 away, >> 0.05


def test_performance_filter_quantile():
    """Quantile mode keeps top fraction of models."""
    from dash.core.filtering import performance_filter
    scores = {i: float(i) for i in range(10)}
    # Keep top 30% → 3 models (scores 7, 8, 9)
    filtered = performance_filter(scores, epsilon=0.3, mode='quantile', verbose=False)
    assert 9 in filtered
    assert 8 in filtered
    assert 0 not in filtered


def test_feature_ablation_score_basic():
    """Ablating important features should increase error more than unimportant ones."""
    from dash.evaluation import feature_ablation_score
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.RandomState(42)
    # Feature 0 is the signal, features 1-4 are noise
    X = rng.randn(200, 5)
    y = 3.0 * X[:, 0] + rng.randn(200) * 0.1
    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    # Correct importance: feature 0 dominates
    correct_imp = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    score_correct = feature_ablation_score(model, X, y, correct_imp, top_k=1)
    # Wrong importance: ablate a noise feature
    wrong_imp = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    score_wrong = feature_ablation_score(model, X, y, wrong_imp, top_k=1)
    # Ablating the true signal should degrade more
    assert score_correct > score_wrong


def test_feature_ablation_score_nonnegative():
    """Ablation score should be >= 0 (removing features shouldn't help)."""
    from dash.evaluation import feature_ablation_score
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = X[:, 0] + X[:, 1] + rng.randn(100) * 0.1
    model = GradientBoostingRegressor(n_estimators=30, random_state=42)
    model.fit(X, y)
    imp = np.array([0.5, 0.5, 0.0])
    score = feature_ablation_score(model, X, y, imp, top_k=2)
    assert score >= 0
