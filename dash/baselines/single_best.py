"""Baseline: Single Best Model."""
import numpy as np
import shap
from joblib import Parallel, delayed

from dash.core.population import DEFAULT_SEARCH_SPACE, sample_configurations, train_single_model
from dash.utils.shap_helpers import compute_global_importance

__all__ = ["SingleBestBaseline"]


class SingleBestBaseline:
    def __init__(self, n_trials=100, task="regression", seed=42, n_jobs=1):
        self.n_trials = n_trials
        self.task = task
        self.seed = seed
        self.n_jobs = n_jobs
        self.model_ = None
        self.global_importance_ = None

    def _compute_shap(self, model, X_ref, background_size=100, seed=None):
        """Compute SHAP values for a single model."""
        n_bg = min(background_size, len(X_ref))
        if seed is not None:
            rng = np.random.RandomState(seed)
            bg_idx = rng.choice(len(X_ref), size=n_bg, replace=False)
            bg = X_ref[bg_idx]
        else:
            bg = X_ref[:n_bg]
        explainer = shap.TreeExplainer(
            model, data=bg, feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref)
        self.global_importance_ = compute_global_importance(sv)

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None,
            background_size=100, seed=None):
        """Fit the baseline.

        Parameters
        ----------
        seed : int or None
            If provided, randomly samples SHAP background rows from X_ref
            (matching ``compute_consensus`` behaviour).  If None, uses the
            first ``background_size`` rows deterministically (legacy).
        """
        if X_ref is None:
            X_ref = X_val

        configs = sample_configurations(
            DEFAULT_SEARCH_SPACE, self.n_trials, seed=self.seed,
        )

        if self.n_jobs == 1:
            best_score, best_model = -np.inf, None
            for i, config in enumerate(configs):
                model, score = train_single_model(
                    config, X_train, y_train, X_val, y_val,
                    task=self.task, seed=self.seed + i,
                )
                if score > best_score:
                    best_score, best_model = score, model
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(train_single_model)(
                    config, X_train, y_train, X_val, y_val,
                    task=self.task, seed=self.seed + i,
                )
                for i, config in enumerate(configs)
            )
            best_model, best_score = max(results, key=lambda x: x[1])

        self.model_ = best_model
        self._compute_shap(best_model, X_ref, background_size, seed)
        return self

    def fit_from_population(self, models, val_scores, X_ref,
                            background_size=100, seed=None):
        """Pick the best model from a pre-trained population.

        Reuses an existing model population (e.g., from DASHPipeline)
        instead of training from scratch.
        """
        best_idx = max(val_scores, key=val_scores.get)
        self.model_ = models[best_idx]
        self._compute_shap(self.model_, X_ref, background_size, seed)
        return self
