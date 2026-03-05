"""Baseline: Single Best Model."""
import numpy as np
import shap
from dash.core.population import DEFAULT_SEARCH_SPACE, _sample_configurations, _train_single_model

class SingleBestBaseline:
    def __init__(self, n_trials=100, task="regression", seed=42):
        self.n_trials, self.task, self.seed = n_trials, task, seed
        self.model_ = self.global_importance_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None: X_ref = X_val
        configs = _sample_configurations(DEFAULT_SEARCH_SPACE, self.n_trials, seed=self.seed)
        best_score, best_model = -np.inf, None
        for i, config in enumerate(configs):
            model, score = _train_single_model(config, X_train, y_train, X_val, y_val, task=self.task, seed=self.seed + i)
            if score > best_score:
                best_score, best_model = score, model
        self.model_ = best_model
        explainer = shap.TreeExplainer(best_model)
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
            self.global_importance_ = np.mean(sv, axis=0)
        else:
            self.global_importance_ = np.mean(np.abs(sv), axis=0)
        return self
