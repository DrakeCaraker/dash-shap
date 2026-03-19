"""Baseline: Random Forest — tests whether RF's independent trees resolve first-mover bias."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import shap

from dash_shap.utils.shap_helpers import compute_global_importance

__all__ = ["RandomForestBaseline"]


class RandomForestBaseline:
    def __init__(self, n_estimators=500, task="regression", seed=42):
        self.n_estimators = n_estimators
        self.task = task
        self.seed = seed
        self.model_ = None
        self.global_importance_ = None
        self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None,
            background_size=100, seed=None):
        """Fit a Random Forest and compute interventional TreeSHAP importance.

        Parameters
        ----------
        seed : int or None
            If provided, randomly samples SHAP background rows from X_ref.
        """
        if X_ref is None:
            X_ref = X_val

        if self.task == "regression":
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=self.seed,
                n_jobs=-1,
            )
        else:
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=self.seed,
                n_jobs=-1,
            )

        self.model_.fit(X_train, y_train)

        n_bg = min(background_size, len(X_ref))
        if seed is not None:
            rng = np.random.RandomState(seed)
            bg_idx = rng.choice(len(X_ref), size=n_bg, replace=False)
            bg = X_ref[bg_idx]
        else:
            bg = X_ref[:n_bg]

        explainer = shap.TreeExplainer(
            self.model_, data=bg, feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref, check_additivity=False)
        self.global_importance_ = compute_global_importance(sv)
        # FSI is undefined for single-model baselines (no inter-model variation).
        self.fsi_ = np.full_like(self.global_importance_, np.nan)
        return self
