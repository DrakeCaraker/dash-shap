"""Baseline: Ensemble SHAP — single large ensemble with standard colsample."""
import numpy as np
import xgboost as xgb
import shap

class EnsembleSHAPBaseline:
    def __init__(self, n_estimators=2000, task="regression", seed=42):
        self.n_estimators, self.task, self.seed = n_estimators, task, seed
        self.model_ = self.global_importance_ = self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None: X_ref = X_val
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(n_estimators=self.n_estimators, max_depth=6, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, early_stopping_rounds=50, eval_metric="rmse", random_state=self.seed, verbosity=0)
        else:
            self.model_ = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=6, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, early_stopping_rounds=50, eval_metric="auc", use_label_encoder=False, random_state=self.seed, verbosity=0)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        explainer = shap.TreeExplainer(self.model_)
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
            self.global_importance_ = np.mean(sv, axis=0)
        else:
            self.global_importance_ = np.mean(np.abs(sv), axis=0)
        self.fsi_ = np.zeros_like(self.global_importance_)
        return self
