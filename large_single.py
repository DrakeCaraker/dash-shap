"""
Baseline: Large Single Model.

A single XGBoost with low colsample_bytree (matching DASH) and K*T total
trees. Tests whether internal tree diversity within a single boosting
ensemble matches DASH's inter-model diversity.

The key hypothesis: within a single boosting ensemble, sequential residual
dependency biases feature selection. Early trees select a feature from a
correlated group; subsequent trees see modified residuals with that feature's
contribution partially removed, creating a path-dependent bias that
concentrates importance on "first mover" features. DASH breaks this
dependency by training independent models from scratch.

If DASH still outperforms this baseline, it demonstrates that breaking
sequential residual dependency provides value beyond what internal tree
diversity can achieve. The gap should be largest at high correlation (rho>=0.9)
where the first-mover effect is strongest.
"""

import numpy as np
import xgboost as xgb
import shap
from typing import Optional


class LargeSingleModelBaseline:
    """Single XGBoost with low colsample_bytree and K*T_per_model trees.

    Parameters
    ----------
    K : int
        Number of DASH models (to match total tree count).
    T_per_model : int
        Average trees per DASH model.
    colsample_bytree : float
        Same low rate used in DASH population.
    task : str
        'regression', 'binary', or 'multiclass'.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        K: int = 20,
        T_per_model: int = 500,
        colsample_bytree: float = 0.2,
        task: str = "regression",
        seed: int = 42,
    ):
        self.K = K
        self.T_per_model = T_per_model
        self.colsample_bytree = colsample_bytree
        self.task = task
        self.seed = seed
        self.model_ = None
        self.global_importance_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_ref: Optional[np.ndarray] = None,
    ) -> "LargeSingleModelBaseline":
        if X_ref is None:
            X_ref = X_val

        total_trees = self.K * self.T_per_model

        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(
                n_estimators=total_trees,
                colsample_bytree=self.colsample_bytree,
                max_depth=6,
                learning_rate=0.1,
                early_stopping_rounds=50,
                eval_metric="rmse",
                random_state=self.seed,
                verbosity=0,
            )
            self.model_.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model_ = xgb.XGBClassifier(
                n_estimators=total_trees,
                colsample_bytree=self.colsample_bytree,
                max_depth=6,
                learning_rate=0.1,
                early_stopping_rounds=50,
                eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed,
                verbosity=0,
            )
            self.model_.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        # Compute SHAP with interventional TreeSHAP
        bg_data = X_ref[:min(100, len(X_ref))]
        explainer = shap.TreeExplainer(
            self.model_,
            data=bg_data,
            feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
            self.global_importance_ = np.mean(sv, axis=0)
        else:
            self.global_importance_ = np.mean(np.abs(sv), axis=0)

        return self
