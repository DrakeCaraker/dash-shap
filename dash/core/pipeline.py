"""DASHPipeline: End-to-end orchestration of all five DASH stages."""
import numpy as np
import time
from typing import Dict, List, Optional

from dash.core.population import generate_model_population, DEFAULT_SEARCH_SPACE
from dash.core.filtering import performance_filter
from dash.core.diversity import (
    get_preliminary_importance,
    greedy_maxmin_selection,
    cluster_coverage_selection,
    deduplication_selection,
)
from dash.core.consensus import compute_consensus
from dash.core.diagnostics import (
    compute_diagnostics,
    FeatureStabilityIndex,
    ImportanceStabilityPlot,
)

__all__ = ["DASHPipeline"]


class DASHPipeline:
    def __init__(
        self,
        M=200,
        K=20,
        epsilon=0.02,
        selection_method="maxmin",
        delta=0.1,
        tau=0.3,
        task="regression",
        search_space=None,
        preliminary_importance_method="gain",
        background_size=100,
        n_jobs=-1,
        seed=42,
        verbose=True,
    ):
        self.M = M
        self.K = K
        self.epsilon = epsilon
        self.selection_method = selection_method
        self.delta = delta
        self.tau = tau
        self.task = task
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.preliminary_importance_method = preliminary_importance_method
        self.background_size = background_size
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # Fitted attributes
        self.models_ = None
        self.val_scores_ = None
        self.configs_ = None
        self.filtered_indices_ = None
        self.selected_indices_ = None
        self.consensus_matrix_ = None
        self.all_shap_matrices_ = None
        self.fsi_ = None
        self.global_importance_ = None
        self.variance_matrix_ = None
        self.timing_ = {}

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, feature_names=None):
        if X_ref is None:
            X_ref = X_val
        self.feature_names_ = feature_names or [
            f"f{i}" for i in range(X_train.shape[1])
        ]

        # Stage 1: Population Generation
        t0 = time.time()
        if self.verbose:
            print("=" * 60)
            print("DASH Stage 1: Population Generation")
            print("=" * 60)
        self.models_, self.val_scores_, self.configs_ = generate_model_population(
            X_train, y_train, X_val, y_val,
            M=self.M, task=self.task, search_space=self.search_space,
            n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose,
        )
        self.timing_["stage1_training"] = time.time() - t0

        # Stage 2: Performance Filtering
        t0 = time.time()
        if self.verbose:
            print(f"\nDASH Stage 2: Performance Filtering (epsilon={self.epsilon})")
        self.filtered_indices_ = performance_filter(
            self.val_scores_, epsilon=self.epsilon,
            higher_is_better=True, verbose=self.verbose,
        )
        self.timing_["stage2_filtering"] = time.time() - t0
        if len(self.filtered_indices_) < 2:
            raise ValueError(
                f"Only {len(self.filtered_indices_)} models passed filter. "
                f"Increase epsilon."
            )

        # Stage 3: Diversity Selection
        t0 = time.time()
        if self.verbose:
            print(f"\nDASH Stage 3: Diversity Selection ({self.selection_method})")
        imp_vecs = get_preliminary_importance(
            self.models_, self.filtered_indices_, X_ref,
            method=self.preliminary_importance_method,
        )
        filt_scores = {i: self.val_scores_[i] for i in self.filtered_indices_}

        if self.selection_method == "maxmin":
            self.selected_indices_ = greedy_maxmin_selection(
                imp_vecs, filt_scores, K=self.K,
                delta=self.delta, verbose=self.verbose,
            )
        elif self.selection_method == "cluster":
            self.selected_indices_ = cluster_coverage_selection(
                imp_vecs, filt_scores, X_train,
                tau=self.tau, K=self.K, verbose=self.verbose,
            )
        elif self.selection_method == "dedup":
            self.selected_indices_ = deduplication_selection(
                imp_vecs, filt_scores, verbose=self.verbose,
            )
            if len(self.selected_indices_) > self.K:
                self.selected_indices_ = sorted(
                    self.selected_indices_,
                    key=lambda i: self.val_scores_[i],
                    reverse=True,
                )[:self.K]
        self.timing_["stage3_selection"] = time.time() - t0

        # Stage 4: Consensus SHAP
        t0 = time.time()
        if self.verbose:
            print(f"\nDASH Stage 4: Consensus SHAP (K={len(self.selected_indices_)})")
        self.consensus_matrix_, self.all_shap_matrices_ = compute_consensus(
            self.models_, self.selected_indices_, X_ref,
            background_size=self.background_size, seed=self.seed,
            verbose=self.verbose,
        )
        self.timing_["stage4_shap"] = time.time() - t0

        # Stage 5: Stability Diagnostics
        t0 = time.time()
        if self.verbose:
            print("\nDASH Stage 5: Stability Diagnostics")
        _, self.variance_matrix_, self.fsi_, self.global_importance_ = (
            compute_diagnostics(self.all_shap_matrices_)
        )
        self.timing_["stage5_diagnostics"] = time.time() - t0

        if self.verbose:
            total = sum(self.timing_.values())
            print(
                f"\nPipeline complete in {total:.1f}s "
                f"(Training: {self.timing_['stage1_training']:.1f}s, "
                f"SHAP: {self.timing_['stage4_shap']:.1f}s)"
            )
        return self

    def get_fsi(self):
        return FeatureStabilityIndex(
            self.fsi_, self.global_importance_, self.feature_names_,
        )

    def plot_importance_stability(self, groups=None, **kwargs):
        return ImportanceStabilityPlot.plot(
            self.global_importance_, self.fsi_,
            feature_names=self.feature_names_, groups=groups, **kwargs,
        )

    def get_importance_ranking(self):
        return np.argsort(self.global_importance_)[::-1]

    def get_consensus_ensemble_predictions(self, X):
        preds = []
        for idx in self.selected_indices_:
            model = self.models_[idx]
            if self.task == "regression":
                preds.append(model.predict(X))
            else:
                preds.append(model.predict_proba(X))
        return np.mean(preds, axis=0)
