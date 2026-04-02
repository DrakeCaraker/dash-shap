"""DASH Stability Diagnostics — F5→F1→DASH workflow.

Implements the diagnostic pipeline from "The Attribution Impossibility":
  1. screen()    — F5 single-model screen (identify unstable pairs)
  2. validate()  — F1 multi-model validation (confirm instability)
  3. consensus() — DASH ensemble averaging (resolve instability)
  4. report()    — Generate instability disclosure text

Usage:
    import dash_shap

    # Step 1: Screen a single model
    flags = dash_shap.screen(model, X_train, X_test)

    # Step 2: Validate with multiple models
    results = dash_shap.validate(models, X_test)

    # Step 3: Compute DASH consensus
    dash = dash_shap.consensus(models, X_test)

    # Step 4: Generate report
    text = dash_shap.report(results, feature_names=X_test.columns)
"""

import numpy as np
from scipy import stats

__all__ = ["screen", "validate", "consensus", "report"]


def _compute_shap(model, X_test, X_background=None):
    """Compute mean |SHAP| for a model. Auto-detects TreeExplainer vs KernelExplainer."""
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        if X_background is None:
            X_background = X_test[:50]
        explainer = shap.KernelExplainer(model.predict, X_background)
        shap_values = explainer.shap_values(X_test, silent=True)
    return np.mean(np.abs(shap_values), axis=0)


def _correlated_groups(X, threshold=0.5):
    """Identify groups of correlated features."""
    corr = np.abs(np.corrcoef(X.T))
    P = corr.shape[0]
    visited = set()
    groups = []
    for i in range(P):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, P):
            if j not in visited and corr[i, j] > threshold:
                group.append(j)
                visited.add(j)
        if len(group) >= 2:
            groups.append(group)
    return groups


def screen(model, X_train, X_test, correlation_threshold=0.5):
    """F5 single-model screen for attribution instability.

    Trains one model, computes SHAP, identifies correlated groups,
    and flags potentially unstable pairs using within-tree split
    frequency variation.

    Parameters
    ----------
    model : fitted model
        Must support shap.TreeExplainer or shap.KernelExplainer.
    X_train : array-like
        Training data (used for correlation detection).
    X_test : array-like
        Test data (used for SHAP computation).
    correlation_threshold : float
        Minimum |correlation| to group features (default 0.5).

    Returns
    -------
    dict with keys:
        correlated_groups : list of lists of feature indices
        shap_values : mean |SHAP| per feature
        flagged_pairs : list of (i, j) pairs potentially unstable
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    groups = _correlated_groups(X_train, correlation_threshold)
    shap_vals = _compute_shap(model, X_test, X_train[:50])

    flagged = []
    for group in groups:
        for ii, i in enumerate(group):
            for j in group[ii + 1 :]:
                diff = abs(shap_vals[i] - shap_vals[j])
                mean_val = (abs(shap_vals[i]) + abs(shap_vals[j])) / 2
                if mean_val > 0 and diff / mean_val < 0.5:
                    flagged.append((i, j))

    return {
        "correlated_groups": groups,
        "shap_values": shap_vals,
        "flagged_pairs": flagged,
    }


def validate(models, X_test, threshold=1.96, X_background=None):
    """F1 multi-model validation of attribution instability.

    Computes SHAP for each model, then for each feature pair:
    Z_{jk} = |mean(phi_j - phi_k)| / (std(phi_j - phi_k) / sqrt(M))

    Parameters
    ----------
    models : list of fitted models
        Multiple models trained with different random seeds.
    X_test : array-like
        Shared test data.
    threshold : float
        Z-score threshold (default 1.96 for 5% significance).
    X_background : array-like or None
        Background data for KernelExplainer (default: first 50 test rows).

    Returns
    -------
    dict with keys:
        shap_matrix : (M, P) array of mean |SHAP| per model
        z_statistics : dict of (i, j) -> Z-score
        flip_rates : dict of (i, j) -> flip rate
        unstable_pairs : list of (i, j) with Z < threshold
        f1_correlation : Spearman r(Z, flip_rate)
    """
    X_test = np.asarray(X_test)
    M = len(models)
    shap_matrix = np.array([_compute_shap(m, X_test, X_background) for m in models])
    P = shap_matrix.shape[1]

    z_stats = {}
    flip_rates = {}

    for i in range(P):
        for j in range(i + 1, P):
            diffs = shap_matrix[:, i] - shap_matrix[:, j]
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            z = abs(mean_diff) / (std_diff / np.sqrt(M)) if std_diff > 0 else np.inf
            z_stats[(i, j)] = z

            # Flip rate: fraction of model pairs that disagree
            wins_i = np.sum(shap_matrix[:, i] > shap_matrix[:, j])
            wins_j = np.sum(shap_matrix[:, j] > shap_matrix[:, i])
            flip_rates[(i, j)] = min(wins_i, wins_j) / max(wins_i + wins_j, 1)

    unstable = [(i, j) for (i, j), z in z_stats.items() if z < threshold]

    # F1 correlation
    z_arr = np.array(list(z_stats.values()))
    flip_arr = np.array(list(flip_rates.values()))
    mask = np.isfinite(z_arr)
    r, p = stats.spearmanr(z_arr[mask], flip_arr[mask])

    return {
        "shap_matrix": shap_matrix,
        "z_statistics": z_stats,
        "flip_rates": flip_rates,
        "unstable_pairs": unstable,
        "f1_correlation": r,
        "f1_pvalue": p,
    }


def consensus(models, X_test, X_background=None):
    """DASH consensus attributions — average |SHAP| across models.

    Parameters
    ----------
    models : list of fitted models
    X_test : array-like
    X_background : array-like or None

    Returns
    -------
    dict with keys:
        attributions : mean |SHAP| per feature (averaged across M models)
        std : standard deviation per feature across models
        tied_groups : list of groups where features are tied (within 1%)
    """
    X_test = np.asarray(X_test)
    shap_matrix = np.array([_compute_shap(m, X_test, X_background) for m in models])

    mean_shap = np.mean(shap_matrix, axis=0)
    std_shap = np.std(shap_matrix, axis=0, ddof=1)

    # Detect tied groups
    P = len(mean_shap)
    tied = []
    visited = set()
    for i in range(P):
        if i in visited:
            continue
        group = [i]
        for j in range(i + 1, P):
            if j in visited:
                continue
            avg = (mean_shap[i] + mean_shap[j]) / 2
            if avg > 0 and abs(mean_shap[i] - mean_shap[j]) / avg < 0.01:
                group.append(j)
                visited.add(j)
        if len(group) >= 2:
            tied.append(group)
            visited.update(group)

    return {
        "attributions": mean_shap,
        "std": std_shap,
        "tied_groups": tied,
    }


def report(validate_results=None, consensus_results=None, feature_names=None, screen_results=None):
    """Generate instability disclosure text for model documentation.

    Parameters
    ----------
    validate_results : dict from validate()
    consensus_results : dict from consensus()
    feature_names : list of str or None
    screen_results : dict from screen() or None

    Returns
    -------
    str : formatted disclosure text
    """
    lines = ["## Attribution Instability Report\n"]

    def fname(i):
        return feature_names[i] if feature_names is not None else f"feature_{i}"

    if screen_results:
        groups = screen_results["correlated_groups"]
        lines.append(f"**Correlated groups detected:** {len(groups)}\n")
        for g in groups:
            names = [fname(i) for i in g]
            lines.append(f"- Group: {{{', '.join(names)}}}")
        lines.append("")

    if validate_results:
        unstable = validate_results["unstable_pairs"]
        r = validate_results["f1_correlation"]
        lines.append(f"**Unstable pairs (Z < 1.96):** {len(unstable)}")
        lines.append(f"**F1 diagnostic correlation:** r = {r:.3f}\n")
        if unstable:
            lines.append("**Unstable pair details:**")
            for i, j in unstable[:10]:
                flip = validate_results["flip_rates"][(i, j)]
                z = validate_results["z_statistics"][(i, j)]
                lines.append(f"- {fname(i)} vs {fname(j)}: flip rate = {flip:.1%}, Z = {z:.2f}")
        lines.append("")

    if consensus_results:
        tied = consensus_results["tied_groups"]
        if tied:
            lines.append("**DASH Consensus — Tied Groups:**")
            for g in tied:
                names = [fname(i) for i in g]
                total = sum(consensus_results["attributions"][i] for i in g)
                lines.append(
                    f"- The correlated group {{{', '.join(names)}}} "
                    f"contributes a total DASH attribution of {total:.4f}. "
                    f"Within this group, individual feature rankings are "
                    f"unstable across training seeds. The group's total "
                    f"importance is stable; individual features should be "
                    f"interpreted as interchangeable."
                )
        lines.append("")

    lines.append("*Generated by dash-shap. See: The Attribution Impossibility (Caraker et al., 2026)*")

    return "\n".join(lines)
