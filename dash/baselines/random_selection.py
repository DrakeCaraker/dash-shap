"""Baseline: Random Selection from DASH population.

Isolates the value of the MaxMin diversity selection mechanism.  Uses the
same population generation and performance filtering as DASH, but selects
K models randomly from the filtered set instead of using greedy MaxMin.
"""
import numpy as np
import time

from dash.core.population import generate_model_population, DEFAULT_SEARCH_SPACE
from dash.core.filtering import performance_filter
from dash.core.consensus import compute_consensus
from dash.core.diagnostics import compute_diagnostics

__all__ = ["RandomSelectionBaseline"]


class RandomSelectionBaseline:
    def __init__(
        self,
        M=200,
        K=20,
        epsilon=0.08,
        epsilon_mode="absolute",
        delta=0.1,
        task="regression",
        search_space=None,
        background_size=100,
        n_jobs=-1,
        seed=42,
        verbose=True,
    ):
        self.M = M
        self.K = K
        self.epsilon = epsilon
        self.epsilon_mode = epsilon_mode
        self.task = task
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.background_size = background_size
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # Fitted attributes
        self.models_ = None
        self.val_scores_ = None
        self.filtered_indices_ = None
        self.selected_indices_ = None
        self.consensus_matrix_ = None
        self.all_shap_matrices_ = None
        self.global_importance_ = None
        self.fsi_ = None
        self.variance_matrix_ = None
        self.timing_ = {}

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None,
            feature_names=None):
        if X_ref is None:
            X_ref = X_val

        # Stage 1: Same population generation as DASH
        t0 = time.time()
        self.models_, self.val_scores_, _ = generate_model_population(
            X_train, y_train, X_val, y_val,
            M=self.M, task=self.task, search_space=self.search_space,
            n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose,
        )
        self.timing_["stage1_training"] = time.time() - t0

        # Stage 2: Same performance filtering as DASH
        t0 = time.time()
        self.filtered_indices_ = performance_filter(
            self.val_scores_, epsilon=self.epsilon,
            higher_is_better=True, mode=self.epsilon_mode,
            verbose=self.verbose,
        )
        self.timing_["stage2_filtering"] = time.time() - t0
        if len(self.filtered_indices_) < 2:
            raise ValueError(
                f"Only {len(self.filtered_indices_)} models passed filter. "
                f"Increase epsilon."
            )

        # Stage 3: RANDOM selection (no MaxMin)
        t0 = time.time()
        rng = np.random.RandomState(self.seed + 999)
        if len(self.filtered_indices_) > self.K:
            self.selected_indices_ = list(
                rng.choice(self.filtered_indices_, size=self.K, replace=False)
            )
        else:
            self.selected_indices_ = list(self.filtered_indices_)
        if self.verbose:
            print(
                f"Random selection: {len(self.selected_indices_)} models "
                f"from {len(self.filtered_indices_)} candidates"
            )
        self.timing_["stage3_selection"] = time.time() - t0

        # Stage 4: Consensus SHAP (same as DASH)
        t0 = time.time()
        self.consensus_matrix_, self.all_shap_matrices_ = compute_consensus(
            self.models_, self.selected_indices_, X_ref,
            background_size=self.background_size, seed=self.seed,
            verbose=self.verbose,
        )
        self.timing_["stage4_shap"] = time.time() - t0

        # Stage 5: Diagnostics (same as DASH)
        t0 = time.time()
        _, self.variance_matrix_, self.fsi_, self.global_importance_ = (
            compute_diagnostics(self.all_shap_matrices_)
        )
        self.timing_["stage5_diagnostics"] = time.time() - t0

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
