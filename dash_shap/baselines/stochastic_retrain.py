"""Baseline: Stochastic Retrain Averaging."""

import numpy as np
from joblib import Parallel, delayed

from dash_shap.core.population import train_single_model, DEFAULT_SEARCH_SPACE, sample_configurations
from dash_shap.core.consensus import compute_consensus
from dash_shap.core.diagnostics import compute_diagnostics

__all__ = ["StochasticRetrainBaseline"]


class StochasticRetrainBaseline:
    def __init__(self, N=20, task="regression", n_jobs=-1, seed=42, nthread=None, colsample_range=None):
        self.N = N
        self.task = task
        self.n_jobs = n_jobs
        self.seed = seed
        self.nthread = nthread
        self.colsample_range = colsample_range  # None = DEFAULT_SEARCH_SPACE
        self.global_importance_ = None
        self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, best_config=None, seed=None):
        if X_ref is None:
            X_ref = X_val

        if best_config is None:
            search_space = dict(DEFAULT_SEARCH_SPACE)
            if self.colsample_range is not None:
                search_space["colsample_bytree"] = list(self.colsample_range)
            configs = sample_configurations(
                search_space,
                100,
                seed=self.seed,
            )
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(train_single_model)(
                    config,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    task=self.task,
                    seed=self.seed + i,
                    nthread=self.nthread,
                )
                for i, config in enumerate(configs)
            )
            best_idx = max(range(len(results)), key=lambda i: results[i][1])
            best_config = configs[best_idx]

        def _train(i):
            model, score = train_single_model(
                best_config,
                X_train,
                y_train,
                X_val,
                y_val,
                task=self.task,
                seed=self.seed + 1000 + i,
                nthread=self.nthread,
            )
            return i, model, score

        results = Parallel(n_jobs=self.n_jobs)(delayed(_train)(i) for i in range(self.N))
        models = {i: model for i, model, _ in results}
        self.models_ = models
        self.selected_indices_ = list(models.keys())

        consensus, all_shap = compute_consensus(
            models,
            list(models.keys()),
            X_ref,
            seed=self.seed,
            verbose=False,
            n_jobs=self.n_jobs,
        )
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self

    def get_consensus_ensemble_predictions(self, X):
        """Average predictions across all retrained models."""
        preds = []
        for idx in self.selected_indices_:
            model = self.models_[idx]
            if self.task == "regression":
                preds.append(model.predict(X))
            else:
                preds.append(model.predict_proba(X))
        return np.mean(preds, axis=0)
