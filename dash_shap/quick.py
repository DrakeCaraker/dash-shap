"""Quick stability check for SHAP explanations.

The simplest way to detect and resolve first-mover bias:

    from dash_shap import check
    result = check(model, X_train, y_train, X_test)
    print(result.report())

Trains M independent models, computes SHAP, detects instability,
and provides stable consensus rankings — all in one call.
"""

import numpy as np

__all__ = ["check", "CheckResult"]


class CheckResult:
    """Result of a DASH stability check.

    Attributes
    ----------
    unstable_pairs : list of tuples
        Feature pairs with unstable rankings (Z < 1.96).
    stable_features : list of int
        Feature indices with stable, high-importance attributions (Quadrant I).
    consensus_importance : ndarray
        DASH consensus importance (averaged across M models).
    fsi : ndarray
        Feature Stability Index for each feature.
    n_models : int
        Number of models trained.
    feature_names : list of str or None
    """

    def __init__(
        self,
        shap_matrix,
        consensus_importance,
        fsi,
        unstable_pairs,
        flip_rates,
        z_statistics,
        correlated_groups,
        feature_names=None,
        task="regression",
    ):
        self.shap_matrix = shap_matrix
        self.consensus_importance = consensus_importance
        self.fsi = fsi
        self.unstable_pairs = unstable_pairs
        self.flip_rates = flip_rates
        self.z_statistics = z_statistics
        self.correlated_groups = correlated_groups
        self.feature_names = feature_names
        self.n_models = shap_matrix.shape[0]
        self.n_features = shap_matrix.shape[1]
        self._task = task

        # Compute stable features (Quadrant I: high importance, low FSI)
        imp_threshold = np.median(consensus_importance)
        fsi_threshold = np.median(fsi[consensus_importance > imp_threshold]) if np.any(consensus_importance > imp_threshold) else np.median(fsi)
        self.stable_features = [
            j for j in range(self.n_features)
            if consensus_importance[j] > imp_threshold and fsi[j] < fsi_threshold
        ]

    def _fname(self, i):
        if self.feature_names is not None:
            return self.feature_names[i]
        return f"feature_{i}"

    def report(self):
        """Human-readable stability report."""
        lines = []
        lines.append("DASH Stability Check")
        lines.append("=" * 40)
        lines.append(f"Models trained: {self.n_models}")
        lines.append(f"Features: {self.n_features}")
        lines.append(f"Unstable pairs: {len(self.unstable_pairs)}")
        lines.append(f"Correlated groups: {len(self.correlated_groups)}")
        lines.append("")

        if self.unstable_pairs:
            lines.append("UNSTABLE PAIRS (rankings flip across retrains):")
            for i, j in self.unstable_pairs[:10]:
                flip = self.flip_rates.get((i, j), self.flip_rates.get((j, i), 0))
                lines.append(f"  {self._fname(i)} vs {self._fname(j)}: "
                             f"flip rate {flip:.0%}")
            if len(self.unstable_pairs) > 10:
                lines.append(f"  ... and {len(self.unstable_pairs) - 10} more")
            lines.append("")
            lines.append("FIX: Use result.dash_importance() for stable rankings,")
            lines.append("     or result.plot() to visualize the instability.")
        else:
            lines.append("No unstable pairs detected. SHAP rankings appear stable.")

        lines.append("")
        lines.append("TOP FEATURES (DASH consensus):")
        ranking = np.argsort(-self.consensus_importance)
        for rank, j in enumerate(ranking[:10], 1):
            stability = "stable" if self.fsi[j] < np.median(self.fsi) else "UNSTABLE"
            lines.append(f"  {rank}. {self._fname(j)}: "
                         f"{self.consensus_importance[j]:.4f} ({stability})")

        return "\n".join(lines)

    def dash_importance(self):
        """Return DASH consensus importance as a dict {feature_name: importance}."""
        return {
            self._fname(j): self.consensus_importance[j]
            for j in np.argsort(-self.consensus_importance)
        }

    def plot(self, figsize=(10, 7), top_k=8):
        """Importance-Stability (IS) Plot.

        Returns a matplotlib Figure showing each feature's importance
        vs FSI, colored by quadrant.
        """
        from dash_shap.core.diagnostics import ImportanceStabilityPlot
        return ImportanceStabilityPlot.plot(
            self.consensus_importance,
            self.fsi,
            feature_names=self.feature_names,
            figsize=figsize,
            annotate_top_k=top_k,
        )

    def to_dataframe(self):
        """Return results as a pandas DataFrame."""
        import pandas as pd
        data = {
            "feature": [self._fname(j) for j in range(self.n_features)],
            "importance": self.consensus_importance,
            "fsi": self.fsi,
            "std": np.std(self.shap_matrix, axis=0, ddof=1),
            "stable": [j in self.stable_features for j in range(self.n_features)],
        }
        df = pd.DataFrame(data).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)

    def __repr__(self):
        return (
            f"CheckResult(models={self.n_models}, features={self.n_features}, "
            f"unstable_pairs={len(self.unstable_pairs)})"
        )


def check(
    X,
    y,
    *,
    M=25,
    task="auto",
    feature_names=None,
    correlation_threshold=0.5,
    seed=42,
    verbose=True,
):
    """Check SHAP explanation stability by training M independent models.

    This is the simplest entry point to DASH. It trains M XGBoost models
    with diverse hyperparameters, computes SHAP values for each, and
    identifies which feature rankings are stable vs unstable.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target variable.
    M : int, default=25
        Number of independent models to train. 25 is sufficient for
        detection; use M=200 with DASHPipeline for publication-quality
        consensus rankings.
    task : str, default="auto"
        "regression", "binary", "multiclass", or "auto" (inferred from y).
    feature_names : list of str or None
        Feature names for display. If None, uses feature_0, feature_1, ...
    correlation_threshold : float, default=0.5
        Minimum |correlation| to group features.
    seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    CheckResult
        Object with .report(), .plot(), .dash_importance(), .to_dataframe()

    Examples
    --------
    >>> from dash_shap import check
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> result = check(X, y, task="binary")
    >>> print(result.report())
    >>> result.plot()
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    X = np.asarray(X)
    y = np.asarray(y)

    # Auto-detect task
    if task == "auto":
        unique_vals = np.unique(y)
        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
            task = "binary" if len(unique_vals) == 2 else "multiclass"
        else:
            task = "regression"

    if verbose:
        print(f"DASH check: training {M} models ({task})...")

    # Split into train/test for SHAP
    rng = np.random.RandomState(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Hyperparameter search space (matches paper)
    search_space = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2, 0.3],
        "colsample_bytree": [0.2, 0.3, 0.4, 0.5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "n_estimators": [100],
    }

    objective = {
        "regression": "reg:squarederror",
        "binary": "binary:logistic",
        "multiclass": "multi:softprob",
    }[task]

    # Train M models with diverse hyperparameters
    models = []
    for i in range(M):
        params = {k: rng.choice(v) for k, v in search_space.items()}
        params["objective"] = objective
        params["random_state"] = seed + i
        params["verbosity"] = 0

        if task == "multiclass":
            params["num_class"] = len(np.unique(y))

        model = xgb.XGBClassifier(**params) if task != "regression" else xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        models.append(model)

    if verbose:
        print(f"  {M} models trained. Computing SHAP...")

    # Compute SHAP for all models
    import shap
    shap_matrix = np.zeros((M, X_test.shape[0], X_test.shape[1]))
    for i, model in enumerate(models):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]  # class 1 for binary
        shap_matrix[i] = sv

    # Consensus importance (mean |SHAP| averaged across models)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=1)  # (M, P)
    consensus_importance = np.mean(mean_abs_shap, axis=0)  # (P,)

    # FSI: cross-model std / mean importance
    model_importances = mean_abs_shap  # (M, P)
    cross_model_std = np.std(model_importances, axis=0, ddof=1)
    fsi = cross_model_std / (consensus_importance + 1e-8)

    # Correlated groups
    groups = []
    corr = np.abs(np.corrcoef(X_train.T))
    P = X_train.shape[1]
    visited = set()
    for i in range(P):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, P):
            if j not in visited and corr[i, j] > correlation_threshold:
                group.append(j)
                visited.add(j)
        if len(group) >= 2:
            groups.append(group)

    # Z-statistics and flip rates for all pairs in correlated groups
    z_stats = {}
    flip_rates = {}
    unstable = []

    for group in groups:
        for ii, fi in enumerate(group):
            for fj in group[ii + 1:]:
                diffs = model_importances[:, fi] - model_importances[:, fj]
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs, ddof=1)
                z = abs(mean_diff) / (std_diff / np.sqrt(M)) if std_diff > 0 else np.inf
                z_stats[(fi, fj)] = z

                wins_i = np.sum(model_importances[:, fi] > model_importances[:, fj])
                wins_j = np.sum(model_importances[:, fj] > model_importances[:, fi])
                flip_rates[(fi, fj)] = min(wins_i, wins_j) / max(wins_i + wins_j, 1)

                if z < 1.96:
                    unstable.append((fi, fj))

    if verbose:
        print(f"  Done. {len(unstable)} unstable pairs found "
              f"in {len(groups)} correlated groups.")

    return CheckResult(
        shap_matrix=mean_abs_shap,
        consensus_importance=consensus_importance,
        fsi=fsi,
        unstable_pairs=unstable,
        flip_rates=flip_rates,
        z_statistics=z_stats,
        correlated_groups=groups,
        feature_names=feature_names,
        task=task,
    )
