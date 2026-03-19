"""Baseline: Naive Top-N Averaging (no diversity selection)."""
import numpy as np

from dash_shap.core.consensus import compute_consensus
from dash_shap.core.diagnostics import compute_diagnostics

__all__ = ["NaiveAveragingBaseline"]


class NaiveAveragingBaseline:
    def __init__(self, N=20, task="regression", n_jobs=1):
        self.N = N
        self.task = task
        self.n_jobs = n_jobs
        self.global_importance_ = None
        self.fsi_ = None
        self.models_ = None
        self.selected_indices_ = None

    def fit_from_population(self, models, val_scores, X_ref):
        sorted_idx = sorted(
            val_scores.keys(), key=lambda i: val_scores[i], reverse=True,
        )
        top_n = sorted_idx[:self.N]
        self.models_ = models
        self.selected_indices_ = top_n
        consensus, all_shap = compute_consensus(
            models, top_n, X_ref, verbose=False, n_jobs=self.n_jobs,
        )
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self

    def get_consensus_ensemble_predictions(self, X):
        preds = []
        for idx in self.selected_indices_:
            model = self.models_[idx]
            if self.task == "regression":
                preds.append(model.predict(X))
            else:
                preds.append(model.predict_proba(X))
        return np.mean(preds, axis=0)
