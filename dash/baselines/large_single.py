"""Baseline: Large Single Model — tests sequential residual dependency hypothesis.

The LSM trains a single XGBoost model with many trees (K * T_per_model) to test
whether a single deep boosting chain amplifies SHAP instability via sequential
residual dependency.

When ``tune=True``, the LSM performs a hyperparameter search over max_depth and
learning_rate (matching the search budget given to other baselines) so the
comparison is fair.  When ``tune=False`` (legacy default), it uses fixed
hyperparameters as an illustrative worst-case anti-pattern.
"""
import numpy as np
import xgboost as xgb
import shap

from dash.utils.shap_helpers import compute_global_importance

__all__ = ["LargeSingleModelBaseline"]


class LargeSingleModelBaseline:
    def __init__(
        self,
        K=20,
        T_per_model=500,
        colsample_bytree=0.2,
        task="regression",
        seed=42,
        tune=False,
    ):
        self.K = K
        self.T_per_model = T_per_model
        self.colsample_bytree = colsample_bytree
        self.task = task
        self.seed = seed
        self.tune = tune
        self.model_ = None
        self.global_importance_ = None
        self.best_params_ = None

    def _build_model(self, n_estimators, max_depth, learning_rate):
        if self.task == "regression":
            return xgb.XGBRegressor(
                n_estimators=n_estimators,
                colsample_bytree=self.colsample_bytree,
                max_depth=max_depth,
                learning_rate=learning_rate,
                early_stopping_rounds=50,
                eval_metric="rmse",
                random_state=self.seed,
                verbosity=0,
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                colsample_bytree=self.colsample_bytree,
                max_depth=max_depth,
                learning_rate=learning_rate,
                early_stopping_rounds=50,
                eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed,
                verbosity=0,
            )

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None:
            X_ref = X_val

        total_trees = self.K * self.T_per_model

        if self.tune:
            # Grid search over max_depth and learning_rate for fair comparison
            depths = [3, 4, 5, 6, 8, 10]
            lrs = [0.01, 0.03, 0.05, 0.1, 0.2]
            best_score = np.inf if self.task == "regression" else -np.inf
            best_md, best_lr = 6, 0.1

            for md in depths:
                for lr in lrs:
                    m = self._build_model(total_trees, md, lr)
                    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    if self.task == "regression":
                        score = m.best_score
                        if score < best_score:
                            best_score, best_md, best_lr = score, md, lr
                    else:
                        score = m.best_score
                        if score > best_score:
                            best_score, best_md, best_lr = score, md, lr

            self.best_params_ = {"max_depth": best_md, "learning_rate": best_lr}
            self.model_ = self._build_model(total_trees, best_md, best_lr)
        else:
            # Legacy fixed hyperparameters (illustrative anti-pattern)
            self.best_params_ = {"max_depth": 6, "learning_rate": 0.1}
            self.model_ = self._build_model(total_trees, 6, 0.1)

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
