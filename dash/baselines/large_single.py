"""Baseline: Large Single Model — tests sequential residual dependency hypothesis."""
import numpy as np
import xgboost as xgb
import shap

from dash.utils.shap_helpers import compute_global_importance

__all__ = ["LargeSingleModelBaseline"]


class LargeSingleModelBaseline:
    def __init__(self, K=20, T_per_model=500, colsample_bytree=0.2, task="regression", seed=42):
        self.K = K
        self.T_per_model = T_per_model
        self.colsample_bytree = colsample_bytree
        self.task = task
        self.seed = seed
        self.model_ = None
        self.global_importance_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None:
            X_ref = X_val

        total_trees = self.K * self.T_per_model

        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(
                n_estimators=total_trees,
                colsample_bytree=self.colsample_bytree,
                max_depth=6, learning_rate=0.1,
                early_stopping_rounds=50, eval_metric="rmse",
                random_state=self.seed, verbosity=0,
            )
        else:
            self.model_ = xgb.XGBClassifier(
                n_estimators=total_trees,
                colsample_bytree=self.colsample_bytree,
                max_depth=6, learning_rate=0.1,
                early_stopping_rounds=50, eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed, verbosity=0,
            )

        self.model_.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], verbose=False,
        )

        bg = X_ref[:min(100, len(X_ref))]
        explainer = shap.TreeExplainer(
            self.model_, data=bg, feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref)
        self.global_importance_ = compute_global_importance(sv)
        return self
