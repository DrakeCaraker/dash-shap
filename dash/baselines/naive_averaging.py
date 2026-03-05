"""Baseline: Naive Top-N Averaging (no diversity selection)."""
import numpy as np

from dash.core.consensus import compute_consensus
from dash.core.diagnostics import compute_diagnostics

__all__ = ["NaiveAveragingBaseline"]


class NaiveAveragingBaseline:
    def __init__(self, N=20, task="regression"):
        self.N = N
        self.task = task
        self.global_importance_ = None
        self.fsi_ = None

    def fit_from_population(self, models, val_scores, X_ref):
        sorted_idx = sorted(
            val_scores.keys(), key=lambda i: val_scores[i], reverse=True,
        )
        top_n = sorted_idx[:self.N]
        consensus, all_shap = compute_consensus(
            models, top_n, X_ref, verbose=False,
        )
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self
