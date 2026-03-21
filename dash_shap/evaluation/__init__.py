"""Evaluation metrics for feature importance methods."""

import numpy as np
from scipy.stats import spearmanr, wilcoxon, norm
from typing import List, Tuple

__all__ = [
    "dgp_agreement",
    "importance_accuracy",
    "group_level_accuracy",
    "group_level_mse",
    "importance_stability",
    "topk_overlap_stability",
    "topk_stability_bootstrap_ci",
    "stability_bootstrap_ci",
    "within_group_equity",
    "cohens_d",
    "compare_methods",
    "holm_bonferroni",
    "feature_ablation_score",
    "tost_equivalence",
    "bootstrap_stability_test",
    "fsi_collinearity_correlation",
    "anova_decomposition",
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


def group_level_accuracy(estimated, true_importance, groups):
    """Accuracy at group level — sums importance within groups, Spearman vs group betas.

    This avoids confounding accuracy with within-group equity: dgp_agreement
    penalises unequal within-group attribution even when the group-level ranking
    is correct.  group_level_accuracy measures only the ranking of groups.

    Note (C8): this metric saturates to 1.0 when true group betas are
    well-separated (e.g., spanning a 20x range across 10 groups), because
    Spearman rank correlation is trivially preserved even by poor models.
    Use group_level_mse for a discriminative complement.
    """
    group_ids = np.unique(groups)
    est_group = np.array([np.sum(estimated[groups == g]) for g in group_ids])
    true_group = np.array([np.sum(true_importance[groups == g]) for g in group_ids])
    rho, _ = spearmanr(est_group, true_group)
    return float(rho)


def group_level_mse(estimated, true_importance, groups):
    """Normalized MSE at group level — measures magnitude accuracy, not just ranking.

    Unlike group_level_accuracy (Spearman), this captures how well the estimated
    group-level importance magnitudes match the true values, not just their rank
    order.  Discriminates between methods even when rank order is trivially correct
    (e.g., when true betas are well-separated).
    """
    group_ids = np.unique(groups)
    est_group = np.array([np.sum(estimated[groups == g]) for g in group_ids])
    true_group = np.array([np.sum(true_importance[groups == g]) for g in group_ids])
    # Normalize to proportions (same approach as dgp_agreement)
    est_norm = est_group / (est_group.sum() + 1e-10)
    true_norm = true_group / (true_group.sum() + 1e-10)
    return float(np.mean((est_norm - true_norm) ** 2))


def importance_stability(vectors):
    """Compute mean pairwise Spearman correlation across importance vectors.

    Uses a vectorized implementation via pre-computed rank matrix and
    np.corrcoef. For bootstrap confidence intervals, use
    stability_bootstrap_ci() instead.
    """
    n = len(vectors)
    if n < 2:
        return float("nan")
    return _stability_from_rank_matrix(_rank_matrix(vectors))


def _rank_matrix(vectors):
    """Rank-transform all vectors into an (n, P) matrix."""
    from scipy.stats import rankdata

    return np.array([rankdata(v) for v in vectors])


def _stability_from_rank_matrix(rank_matrix):
    """Compute mean pairwise Spearman from pre-computed ranks.

    Spearman correlation equals Pearson correlation on ranks.
    Uses np.corrcoef for vectorized computation.
    """
    n = rank_matrix.shape[0]
    if n < 2:
        return float("nan")
    corr = np.corrcoef(rank_matrix)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(np.mean(corr[mask]))


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
        return float("nan"), 0.0, float("nan"), float("nan")

    # Pre-compute rank matrix once for all bootstrap/jackknife iterations
    rank_matrix = _rank_matrix(vectors)
    point = _stability_from_rank_matrix(rank_matrix)

    # Bootstrap distribution
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[b] = _stability_from_rank_matrix(rank_matrix[idx])
    se = float(np.std(boots, ddof=1))

    # BCa bias-correction factor z0
    prop_below = np.mean(boots < point)
    # Clamp to avoid ±inf from ppf at exactly 0 or 1
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_below)

    # BCa acceleration factor from jackknife
    all_idx = np.arange(n)
    jack_stats = np.empty(n)
    for i in range(n):
        loo_idx = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
        jack_stats[i] = _stability_from_rank_matrix(rank_matrix[loo_idx])
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
            cvs.append(0.0 if gi.std() < 1e-10 else float("inf"))
    return float(np.mean(cvs)) if cvs else 0.0


def cohens_d(g1, g2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-10 else 0.0


def compare_methods(a, b):
    """Wilcoxon signed-rank test between two sets of scores."""
    if np.allclose(a, b):
        return 0.0, 1.0
    stat, pval = wilcoxon(a, b)
    return float(stat), float(pval)


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni while still controlling the family-wise
    error rate.  Returns an array of adjusted p-values (same length as input).
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(p_values[idx] * (n - rank), 1.0)
    # Enforce monotonicity (adjusted values must be non-decreasing in sorted order)
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def tost_equivalence(a, b, delta=None):
    """Two One-Sided T-test for equivalence (paired).

    Tests H0: |mean(a) - mean(b)| >= delta vs H1: |mean(a) - mean(b)| < delta.

    When delta is None, computes it adaptively as 5% of the pooled mean of
    absolute values, with a lower bound of 0.01. For stability scores
    (range 0–1), a reasonable delta is 0.02 (2 percentage points).

    Parameters
    ----------
    a, b : array-like
        Paired observations.
    delta : float or None
        Equivalence margin. If None, computed as max(0.01, 0.05 * pooled_mean),
        where pooled_mean is the mean of absolute values across both arrays.
        Default: None (adaptive).

    Returns
    -------
    t1 : float
        Test statistic for upper bound test.
    p1 : float
        One-sided p-value for upper bound test.
    t2 : float
        Test statistic for lower bound test.
    p2 : float
        One-sided p-value for lower bound test.
    equivalent : bool
        True if both one-sided tests reject at alpha=0.05.
    """
    from scipy.stats import ttest_rel

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    # Compute adaptive delta if not provided
    if delta is None:
        pooled_mean = (np.mean(np.abs(a)) + np.mean(np.abs(b))) / 2.0
        delta = max(0.01, 0.05 * pooled_mean)

    # Test 1: H0: mean(a-b) >= delta  vs  H1: mean(a-b) < delta
    t1, p1_two = ttest_rel(a - delta, b)
    p1 = p1_two / 2 if t1 < 0 else 1 - p1_two / 2

    # Test 2: H0: mean(a-b) <= -delta  vs  H1: mean(a-b) > -delta
    t2, p2_two = ttest_rel(a + delta, b)
    p2 = p2_two / 2 if t2 > 0 else 1 - p2_two / 2

    equivalent = bool(max(p1, p2) < 0.05)
    return float(t1), float(p1), float(t2), float(p2), equivalent


def bootstrap_stability_test(imp_runs_a, imp_runs_b, n_bootstrap=10000, seed=42):
    """Bootstrap permutation test for stability difference between two methods.

    Resamples repetition indices with replacement, recomputes stability for
    both methods on each bootstrap sample, and reports a two-sided p-value
    for H0: stability_a == stability_b.

    Parameters
    ----------
    imp_runs_a, imp_runs_b : list of arrays
        Per-repetition importance vectors for method A and B (same length).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    observed_diff : float
        stability_a - stability_b (point estimate).
    p_value : float
        Two-sided bootstrap p-value.
    ci_lo, ci_hi : float
        95% percentile CI on the stability difference.
    """
    rng = np.random.RandomState(seed)
    n = len(imp_runs_a)
    assert len(imp_runs_b) == n, "Both methods must have the same number of reps"

    rank_a = _rank_matrix(imp_runs_a)
    rank_b = _rank_matrix(imp_runs_b)
    observed_diff = _stability_from_rank_matrix(rank_a) - _stability_from_rank_matrix(rank_b)

    boot_diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[b] = _stability_from_rank_matrix(rank_a[idx]) - _stability_from_rank_matrix(rank_b[idx])

    # Two-sided p-value: fraction of bootstrap diffs at least as extreme as 0
    # under the null (centered at observed_diff)
    centered = boot_diffs - observed_diff
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    return observed_diff, p_value, ci_lo, ci_hi


def feature_ablation_score(model, X, y, importance, top_k=5, metric_fn=None):
    """Zero out top-K important features and measure prediction degradation.

    A proxy for explanation quality on real data without ground truth.
    Returns the increase in error when ablating the top-K features identified
    by ``importance``.  Higher values mean the explanation correctly identified
    important features.

    Parameters
    ----------
    model : fitted model with .predict()
    X : array-like (n_samples, n_features)
    y : array-like (n_samples,)
    importance : array-like (n_features,), absolute feature importance
    top_k : int, number of features to ablate
    metric_fn : callable(y_true, y_pred) → float, default RMSE
    """
    if metric_fn is None:
        from sklearn.metrics import root_mean_squared_error

        metric_fn = root_mean_squared_error
    baseline_score = metric_fn(y, model.predict(X))
    top_features = np.argsort(importance)[-top_k:][::-1]
    X_ablated = np.array(X, dtype=float, copy=True)
    X_ablated[:, top_features] = 0.0
    ablated_score = metric_fn(y, model.predict(X_ablated))
    return float(ablated_score - baseline_score)


def topk_overlap_stability(vectors, k=5):
    """Mean pairwise Jaccard similarity of top-k feature sets across runs.

    Complements full-rank Spearman stability by measuring practitioner-facing
    top-k membership consistency.

    Parameters
    ----------
    vectors : list of array-like
        Importance vectors from repeated runs (absolute values used).
    k : int
        Number of top features to compare.

    Returns
    -------
    float
        Mean pairwise Jaccard similarity in [0, 1].
    """
    from itertools import combinations

    top_sets = [set(np.argsort(np.abs(v))[-k:]) for v in vectors]
    n = len(top_sets)
    if n < 2:
        return 1.0

    similarities = []
    for i, j in combinations(range(n), 2):
        intersection = len(top_sets[i] & top_sets[j])
        union = len(top_sets[i] | top_sets[j])
        similarities.append(intersection / union if union > 0 else 1.0)

    return float(np.mean(similarities))


def topk_stability_bootstrap_ci(vectors, k=5, n_boot=1000, ci=0.95, seed=42):
    """BCa bootstrap confidence interval for top-k overlap stability.

    Mirrors ``stability_bootstrap_ci`` but for the Jaccard-based top-k metric.

    Returns (point, se, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    n = len(vectors)
    if n < 2:
        return 1.0, 0.0, 1.0, 1.0

    point = topk_overlap_stability(vectors, k=k)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[b] = topk_overlap_stability([vectors[i] for i in idx], k=k)
    se = float(np.std(boots, ddof=1))

    # BCa bias-correction factor z0
    prop_below = np.clip(np.mean(boots < point), 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_below)

    # BCa acceleration from jackknife
    all_idx = np.arange(n)
    jack_stats = np.empty(n)
    for i in range(n):
        loo = [vectors[j] for j in range(n) if j != i]
        jack_stats[i] = topk_overlap_stability(loo, k=k)
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2)) ** 1.5
    a = num / den if den != 0 else 0.0

    alpha = 1 - ci
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def _bca_pct(z_alpha):
        adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return norm.cdf(adj) * 100

    ci_lo = float(np.percentile(boots, np.clip(_bca_pct(z_lo), 0, 100)))
    ci_hi = float(np.percentile(boots, np.clip(_bca_pct(z_hi), 0, 100)))
    return point, se, ci_lo, ci_hi


def fsi_collinearity_correlation(fsi_values, feature_rho, groups=None):
    """Spearman correlation between FSI values and feature-level collinearity.

    Validates that high-FSI features correspond to members of highly
    correlated groups, as expected under the first-mover bias hypothesis.

    Parameters
    ----------
    fsi_values : array-like, shape (n_features,)
        Feature Stability Index values from DASH diagnostics.
    feature_rho : array-like, shape (n_features,)
        Per-feature collinearity level. For synthetic data with group
        structure, this is typically the within-group correlation rho
        for each feature (0 for independent features).
    groups : array-like, optional
        Group labels per feature. If provided, computes group-level
        correlation (mean FSI per group vs group rho) in addition to
        feature-level.

    Returns
    -------
    dict
        'feature_spearman': Spearman rho at feature level
        'feature_pvalue': p-value for feature-level correlation
        'group_spearman': Spearman rho at group level (if groups provided)
        'group_pvalue': p-value for group-level correlation (if groups provided)
    """
    fsi_values = np.asarray(fsi_values, dtype=float)
    feature_rho = np.asarray(feature_rho, dtype=float)

    rho, pval = spearmanr(fsi_values, feature_rho)
    result = {
        "feature_spearman": float(rho),
        "feature_pvalue": float(pval),
    }

    if groups is not None:
        groups = np.asarray(groups)
        group_ids = np.unique(groups)
        group_fsi = np.array([np.mean(fsi_values[groups == g]) for g in group_ids])
        group_rho_vals = np.array([np.mean(feature_rho[groups == g]) for g in group_ids])
        g_rho, g_pval = spearmanr(group_fsi, group_rho_vals)
        result["group_spearman"] = float(g_rho)
        result["group_pvalue"] = float(g_pval)

    return result


def anova_decomposition(importances_grid):
    """Exact two-way variance decomposition using a fully crossed R×R design.

    Decomposes total importance variance into data-sampling variance,
    model-selection variance, and residual (interaction) variance using
    a balanced two-way ANOVA without replication.

    Parameters
    ----------
    importances_grid : dict
        Mapping ``(data_idx, model_idx) -> importance_vector`` where
        data_idx ∈ {0, …, R_d-1} and model_idx ∈ {0, …, R_m-1}.
        The grid must be fully crossed (all combinations present).

    Returns
    -------
    dict with keys:
        data_var_frac : float
            Fraction of SS_total attributable to data-sampling variation.
        model_var_frac : float
            Fraction of SS_total attributable to model-selection variation.
        residual_var_frac : float
            Residual/interaction fraction (SS_total - SS_data - SS_model).
        ss_data, ss_model, ss_error, ss_total : float
            Raw sum-of-squares values (summed across all features).

    Notes
    -----
    Replaces the approximate ``1 - stability`` proxy used in
    ``experiment_variance_decomposition``.  Unlike the proxy, the ANOVA
    decomposition satisfies the additive identity
    SS_data + SS_model + SS_error = SS_total exactly.
    """
    keys = list(importances_grid.keys())
    d_indices = sorted({k[0] for k in keys})
    m_indices = sorted({k[1] for k in keys})
    R_d = len(d_indices)
    R_m = len(m_indices)

    d_map = {v: i for i, v in enumerate(d_indices)}
    m_map = {v: i for i, v in enumerate(m_indices)}

    P = len(next(iter(importances_grid.values())))
    grid = np.zeros((R_d, R_m, P))
    for (di, mi), imp in importances_grid.items():
        grid[d_map[di], m_map[mi], :] = np.asarray(imp, dtype=float)

    grand_mean = grid.mean(axis=(0, 1))  # (P,)
    row_means = grid.mean(axis=1)  # (R_d, P) — data-seed means
    col_means = grid.mean(axis=0)  # (R_m, P) — model-seed means

    ss_total = float(np.sum((grid - grand_mean) ** 2))
    ss_data = float(R_m * np.sum((row_means - grand_mean) ** 2))
    ss_model = float(R_d * np.sum((col_means - grand_mean) ** 2))
    ss_error = ss_total - ss_data - ss_model

    if ss_total > 0:
        data_var_frac = ss_data / ss_total
        model_var_frac = ss_model / ss_total
        residual_var_frac = ss_error / ss_total
    else:
        data_var_frac = model_var_frac = residual_var_frac = 0.0

    return {
        "data_var_frac": float(data_var_frac),
        "model_var_frac": float(model_var_frac),
        "residual_var_frac": float(residual_var_frac),
        "ss_data": ss_data,
        "ss_model": ss_model,
        "ss_error": ss_error,
        "ss_total": ss_total,
        "R_d": R_d,
        "R_m": R_m,
    }
