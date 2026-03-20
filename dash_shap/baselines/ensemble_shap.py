"""Baseline: Ensemble SHAP — single large ensemble with standard colsample."""

import numpy as np
import xgboost as xgb
import shap

from dash_shap.utils.shap_helpers import compute_global_importance

__all__ = ["EnsembleSHAPBaseline"]


class EnsembleSHAPBaseline:
    def __init__(self, n_estimators=2000, task="regression", seed=42):
        self.n_estimators = n_estimators
        self.task = task
        self.seed = seed
        self.model_ = None
        self.global_importance_ = None
        self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, seed=None):
        """Fit the baseline.

        Parameters
        ----------
        seed : int or None
            If provided, randomly samples SHAP background rows from X_ref.
        """
        if X_ref is None:
            X_ref = X_val

        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                colsample_bytree=0.8,
                subsample=0.8,
                early_stopping_rounds=50,
                eval_metric="rmse",
                random_state=self.seed,
                verbosity=0,
            )
        else:
            self.model_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                colsample_bytree=0.8,
                subsample=0.8,
                early_stopping_rounds=50,
                eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed,
                verbosity=0,
            )

        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        n_bg = min(100, len(X_ref))
        if seed is not None:
            rng = np.random.RandomState(seed)
            bg_idx = rng.choice(len(X_ref), size=n_bg, replace=False)
            bg = X_ref[bg_idx]
        else:
            bg = X_ref[:n_bg]
        explainer = shap.TreeExplainer(
            self.model_,
            data=bg,
            feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref, check_additivity=False)
        self.global_importance_ = compute_global_importance(sv)
        # FSI is undefined for single-model baselines (no inter-model variation).
        # Set to NaN rather than zero to avoid misinterpretation as perfect agreement.
        self.fsi_ = np.full_like(self.global_importance_, np.nan)
        return self
