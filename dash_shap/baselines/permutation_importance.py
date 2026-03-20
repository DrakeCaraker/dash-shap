"""Baseline: Permutation Importance — tests whether instability is SHAP-specific."""

import numpy as np
from sklearn.inspection import permutation_importance as sklearn_perm_importance

from dash_shap.core.population import DEFAULT_SEARCH_SPACE, sample_configurations, train_single_model

__all__ = ["PermutationImportanceBaseline"]


class PermutationImportanceBaseline:
    def __init__(self, n_trials=100, task="regression", seed=42, n_repeats=10):
        self.n_trials = n_trials
        self.task = task
        self.seed = seed
        self.n_repeats = n_repeats
        self.model_ = None
        self.global_importance_ = None
        self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, y_ref=None, background_size=100, seed=None):
        """Fit the single-best XGBoost model and compute permutation importance.

        Parameters
        ----------
        y_ref : array-like or None
            Labels for X_ref, required for permutation importance.
            Falls back to y_val if not provided.
        seed : int or None
            Random state for permutation importance.
        """
        if X_ref is None:
            X_ref = X_val
        if y_ref is None:
            y_ref = y_val

        # Train single best model (same as SingleBestBaseline)
        configs = sample_configurations(
            DEFAULT_SEARCH_SPACE,
            self.n_trials,
            seed=self.seed,
        )
        best_score, best_model = -np.inf, None
        for i, config in enumerate(configs):
            model, score = train_single_model(
                config,
                X_train,
                y_train,
                X_val,
                y_val,
                task=self.task,
                seed=self.seed + i,
            )
            if score > best_score:
                best_score, best_model = score, model

        self.model_ = best_model

        # Permutation importance instead of SHAP
        result = sklearn_perm_importance(
            self.model_,
            X_ref,
            y_ref,
            n_repeats=self.n_repeats,
            random_state=seed if seed is not None else self.seed,
            n_jobs=-1,
        )
        self.global_importance_ = np.maximum(result.importances_mean, 0)
        # FSI is undefined for single-model baselines.
        self.fsi_ = np.full_like(self.global_importance_, np.nan)
        return self
