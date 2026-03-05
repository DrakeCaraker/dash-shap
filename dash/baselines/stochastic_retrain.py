"""Baseline: Stochastic Retrain Averaging."""
import numpy as np
from joblib import Parallel, delayed
from dash.core.population import _train_single_model, DEFAULT_SEARCH_SPACE, _sample_configurations
from dash.core.consensus import compute_consensus, compute_diagnostics

class StochasticRetrainBaseline:
    def __init__(self, N=20, task="regression", n_jobs=-1, seed=42):
        self.N, self.task, self.n_jobs, self.seed = N, task, n_jobs, seed
        self.global_importance_ = self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, best_config=None):
        if X_ref is None: X_ref = X_val
        if best_config is None:
            configs = _sample_configurations(DEFAULT_SEARCH_SPACE, 100, seed=self.seed)
            best_score, best_config = -np.inf, configs[0]
            for i, config in enumerate(configs):
                _, score = _train_single_model(config, X_train, y_train, X_val, y_val, task=self.task, seed=self.seed + i)
                if score > best_score:
                    best_score, best_config = score, config
        def _train(i):
            model, score = _train_single_model(best_config, X_train, y_train, X_val, y_val, task=self.task, seed=self.seed + 1000 + i)
            return i, model, score
        results = Parallel(n_jobs=self.n_jobs)(delayed(_train)(i) for i in range(self.N))
        models = {i: model for i, model, _ in results}
        consensus, all_shap = compute_consensus(models, list(models.keys()), X_ref, verbose=False)
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self
