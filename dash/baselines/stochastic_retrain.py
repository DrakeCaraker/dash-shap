"""Baseline: Stochastic Retrain Averaging."""
import numpy as np
from joblib import Parallel, delayed

from dash.core.population import train_single_model, DEFAULT_SEARCH_SPACE, sample_configurations
from dash.core.consensus import compute_consensus
from dash.core.diagnostics import compute_diagnostics

__all__ = ["StochasticRetrainBaseline"]


class StochasticRetrainBaseline:
    def __init__(self, N=20, task="regression", n_jobs=-1, seed=42):
        self.N = N
        self.task = task
        self.n_jobs = n_jobs
        self.seed = seed
        self.global_importance_ = None
        self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, best_config=None):
        if X_ref is None:
            X_ref = X_val

        if best_config is None:
            configs = sample_configurations(
                DEFAULT_SEARCH_SPACE, 100, seed=self.seed,
            )
            best_score, best_config = -np.inf, configs[0]
            for i, config in enumerate(configs):
                _, score = train_single_model(
                    config, X_train, y_train, X_val, y_val,
                    task=self.task, seed=self.seed + i,
                )
                if score > best_score:
                    best_score, best_config = score, config

        def _train(i):
            model, score = train_single_model(
                best_config, X_train, y_train, X_val, y_val,
                task=self.task, seed=self.seed + 1000 + i,
            )
            return i, model, score

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_train)(i) for i in range(self.N)
        )
        models = {i: model for i, model, _ in results}

        consensus, all_shap = compute_consensus(
            models, list(models.keys()), X_ref, verbose=False,
        )
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self
