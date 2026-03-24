"""Baseline methods for NN comparison with DASH."""

from __future__ import annotations

import numpy as np
import shap
from joblib import Parallel, delayed

from dash_shap.core.nn_population import (
    NN_SEARCH_SPACE,
    sample_nn_configurations,
    train_single_nn,
)
from dash_shap.core.nn_attribution import compute_nn_attributions
from dash_shap.core.diagnostics import compute_diagnostics
from dash_shap.utils.shap_helpers import compute_global_importance

__all__ = ["SingleNNBaseline", "BaggedNNBaseline"]


class SingleNNBaseline:
    def __init__(
        self,
        n_trials: int = 30,
        task: str = "regression",
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.n_trials = n_trials
        self.task = task
        self.n_jobs = n_jobs
        self.seed = seed
        self.model_ = None
        self.global_importance_ = None

    def _compute_shap(self, model, X_ref: np.ndarray, background_size: int = 100, seed: int | None = None) -> None:
        """Compute KernelSHAP for a single model."""
        n_bg = min(background_size, len(X_ref))
        if seed is not None:
            rng = np.random.RandomState(seed)
            bg_idx = rng.choice(len(X_ref), size=n_bg, replace=False)
            bg = X_ref[bg_idx]
        else:
            bg = X_ref[:n_bg]
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(X_ref, nsamples="auto", silent=True)
        self.global_importance_ = compute_global_importance(sv)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_ref: np.ndarray | None = None,
        background_size: int = 100,
        seed: int | None = None,
    ) -> SingleNNBaseline:
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

        configs = sample_nn_configurations(
            NN_SEARCH_SPACE,
            self.n_trials,
            seed=self.seed,
        )

        if self.n_jobs == 1:
            best_score, best_model = -np.inf, None
            for i, config in enumerate(configs):
                model, score = train_single_nn(
                    config,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    task=self.task,
                    seed=self.seed + i,
                )
                if score > best_score:
                    best_score, best_model = score, model
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(train_single_nn)(
                    config,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    task=self.task,
                    seed=self.seed + i,
                )
                for i, config in enumerate(configs)
            )
            best_model, best_score = max(results, key=lambda x: x[1])

        self.model_ = best_model
        self._compute_shap(best_model, X_ref, background_size, seed)
        return self


class BaggedNNBaseline:
    def __init__(
        self,
        N: int = 20,
        task: str = "regression",
        n_jobs: int = -1,
        seed: int = 42,
    ):
        self.N = N
        self.task = task
        self.n_jobs = n_jobs
        self.seed = seed
        self.global_importance_ = None
        self.fsi_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_ref: np.ndarray | None = None,
        best_config: dict | None = None,
        seed: int | None = None,
    ) -> BaggedNNBaseline:
        if X_ref is None:
            X_ref = X_val

        if best_config is None:
            configs = sample_nn_configurations(
                NN_SEARCH_SPACE,
                100,
                seed=self.seed,
            )
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(train_single_nn)(
                    config,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    task=self.task,
                    seed=self.seed + i,
                )
                for i, config in enumerate(configs)
            )
            best_idx = max(range(len(results)), key=lambda i: results[i][1])
            best_config = configs[best_idx]

        def _train(i):
            model, score = train_single_nn(
                best_config,
                X_train,
                y_train,
                X_val,
                y_val,
                task=self.task,
                seed=self.seed + 1000 + i,
            )
            return i, model, score

        results = Parallel(n_jobs=self.n_jobs)(delayed(_train)(i) for i in range(self.N))
        models = {i: model for i, model, _ in results}
        self.models_ = models
        self.selected_indices_ = list(models.keys())

        consensus, all_shap = compute_nn_attributions(
            models,
            list(models.keys()),
            X_ref,
            seed=self.seed,
            verbose=False,
            n_jobs=self.n_jobs,
        )
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self
