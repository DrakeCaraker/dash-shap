"""DASHPipeline: End-to-end orchestration of all five DASH stages."""

import numpy as np
import time

from dash_shap.core.population import generate_model_population, DEFAULT_SEARCH_SPACE
from dash_shap.core.filtering import performance_filter
from dash_shap.core.diversity import (
    get_preliminary_importance,
    greedy_maxmin_selection,
    cluster_coverage_selection,
    deduplication_selection,
)
from dash_shap.core.consensus import compute_consensus
from dash_shap.core.diagnostics import (
    compute_diagnostics,
    FeatureStabilityIndex,
    ImportanceStabilityPlot,
)

__all__ = ["DASHPipeline"]


class DASHPipeline:
    def __init__(
        self,
        M=200,
        K=30,
        epsilon=0.08,
        epsilon_mode="absolute",
        selection_method="maxmin",
        delta=0.05,
        tau=0.3,
        task="regression",
        search_space=None,
        preliminary_importance_method="gain",
        background_size=100,
        n_jobs=-1,
        seed=42,
        verbose=True,
        # Surrogate search parameters (only used when selection_method="surrogate")
        surrogate_batch_size=10,
        surrogate_n_initial=30,
        surrogate_acquisition="diverse_rashomon",
        surrogate_n_candidates=500,
    ):
        """Create a DASH pipeline for stable feature importance under collinearity.

        DASH trains a population of M independent XGBoost models (each forced
        to use different features via low colsample_bytree), filters out weak
        performers, selects K maximally-diverse models, and averages their SHAP
        matrices to produce a stable consensus attribution.

        Parameters
        ----------
        M : int, default=200
            Population size — number of XGBoost models to train with randomly
            sampled hyperparameters. Larger M gives more diversity but slower
            training. M=50 is sufficient for exploratory work; M=200 for paper.
        K : int, default=30
            Number of diverse models to select for consensus. Must be < M after
            filtering. K=10 works for most datasets; K=30 for publication results.
        epsilon : float, default=0.08
            Performance filter threshold. Models whose validation score falls
            more than ``epsilon`` below the best model are discarded.
            Meaning depends on ``epsilon_mode``:
            - "absolute": raw score units (e.g., 0.08 RMSE above best)
            - "relative": fraction of best score (e.g., 8% worse than best)
            - "quantile": keep top (1-epsilon) fraction of models
        epsilon_mode : {"absolute", "relative", "quantile"}, default="absolute"
            How ``epsilon`` is interpreted (see above). Use "relative" for
            real-world datasets where absolute score scale is unknown.
        selection_method : {"maxmin", "cluster", "dedup", "surrogate"}, default="maxmin"
            Diversity selection algorithm:
            - "maxmin": greedy MaxMin — maximizes minimum pairwise cosine
              distance. Default; best for most use cases.
            - "cluster": cluster-coverage selection — selects models that
              cover different feature-correlation clusters. Requires X_train.
            - "dedup": deduplication only — removes near-duplicate models
              (Spearman > delta) without active diversity maximization.
            - "surrogate": GP-surrogate-assisted Rashomon set search.
              Uses a Gaussian Process to model the loss surface and
              iteratively acquires configs that are both likely near-optimal
              and diverse. Final K selection uses importance-space MaxMin.
              See ``surrogate_*`` parameters below.
        delta : float, default=0.05
            Minimum cosine distance between selected models (maxmin) or
            Spearman correlation threshold for deduplication (dedup).
            Larger delta -> more diverse but fewer models selected.
        tau : float, default=0.3
            Cluster distance threshold for selection_method="cluster".
            Ignored for other selection methods.
        task : {"regression", "binary", "multiclass"}, default="regression"
            Prediction task type. Controls XGBoost objective and eval metric.
        search_space : dict or None, default=None
            Hyperparameter search space for population generation. If None,
            uses DEFAULT_SEARCH_SPACE (colsample_bytree in [0.1, 0.5]).
            Keys must match XGBoost parameter names.
        preliminary_importance_method : {"gain", "shap_subsample"}, default="gain"
            Fast importance estimate used for diversity selection (Stage 3).
            "gain" is fast and exact; "shap_subsample" is slower but SHAP-based.
        background_size : int, default=100
            Number of rows to use as SHAP background (TreeExplainer background).
            Reduce to 50 for faster computation; increase for smoother attributions.
        n_jobs : int, default=-1
            Number of parallel jobs for population training and SHAP computation.
            -1 uses all available CPU cores. Set to 1 to disable parallelism.
        seed : int, default=42
            Master random seed. Controls population generation, background
            sampling, and all stochastic stages.
        verbose : bool, default=True
            If True, print progress for each of the 5 pipeline stages.
        surrogate_batch_size : int, default=10
            Models to acquire per GP iteration (surrogate mode only).
        surrogate_n_initial : int, default=30
            Random models to train before fitting the first surrogate.
        surrogate_acquisition : str, default="diverse_rashomon"
            Acquisition function: "diverse_rashomon", "rashomon_probability",
            or "level_set_boundary".
        surrogate_n_candidates : int, default=500
            Random candidates to score per acquisition batch.

        Examples
        --------
        >>> from dash_shap import DASHPipeline
        >>> from dash_shap.experiments.synthetic import generate_synthetic_linear
        >>> (X_train, y_train, X_val, y_val, X_explain,
        ...  _, X_test, _, groups, _, _) = generate_synthetic_linear(
        ...     N=2000, P=20, rho=0.9, seed=42)
        >>> pipe = DASHPipeline(M=50, K=10, epsilon=0.08, seed=42, verbose=False)
        >>> pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)
        >>> pipe.global_importance_.shape
        (20,)
        """
        self.M = M
        self.K = K
        self.epsilon = epsilon
        self.epsilon_mode = epsilon_mode
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
        self.surrogate_batch_size = surrogate_batch_size
        self.surrogate_n_initial = surrogate_n_initial
        self.surrogate_acquisition = surrogate_acquisition
        self.surrogate_n_candidates = surrogate_n_candidates

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
        self.result_ = None  # DASHResult — set after Stage 5; additive
        self.surrogate_info_ = None  # set when selection_method="surrogate"
        self.timing_ = {}

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, feature_names=None):
        """Fit all five DASH stages on the provided data.

        DASH requires a four-way data split. Use ``X_ref`` (also called
        ``X_explain``) as a held-out set for SHAP background computation,
        kept separate from both ``X_val`` (used for model selection) and
        ``X_test`` (reserved for final evaluation). This avoids data leakage
        and ensures SHAP values are not influenced by the validation objective.

        Parameters
        ----------
        X_train : array-like of shape (n_train, n_features)
            Training features for XGBoost model fitting.
        y_train : array-like of shape (n_train,)
            Training targets.
        X_val : array-like of shape (n_val, n_features)
            Validation features for early stopping and model scoring.
        y_val : array-like of shape (n_val,)
            Validation targets.
        X_ref : array-like of shape (n_ref, n_features) or None
            Reference set used as SHAP background (X_explain in the paper).
            Should be a held-out split — NOT X_test or X_train.
            If None, defaults to X_val with a UserWarning.
        feature_names : list of str or None
            Optional feature names for plots and diagnostics. If None,
            features are named f0, f1, ..., f{P-1}.

        Returns
        -------
        self : DASHPipeline
            Fitted pipeline. All attributes below are set after calling fit().

        Attributes Set After Fitting
        ----------------------------
        models_ : dict {int: XGBModel}
            All M trained models, keyed by population index.
        val_scores_ : dict {int: float}
            Validation scores (R^2 for regression) for all M models.
        filtered_indices_ : list of int
            Indices of models that passed the epsilon performance filter.
        selected_indices_ : list of int
            Indices of the K diverse models selected for consensus.
        consensus_matrix_ : ndarray of shape (n_ref, n_features)
            Element-wise mean SHAP matrix across the K selected models.
        all_shap_matrices_ : ndarray of shape (K, n_ref, n_features)
            Individual SHAP matrices for each of the K selected models.
        global_importance_ : ndarray of shape (n_features,)
            Mean absolute SHAP value per feature (consensus summary).
        fsi_ : ndarray of shape (n_features,)
            Feature Stability Index — ratio of inter-model SHAP std to mean.
            High FSI indicates unstable attribution due to collinearity.
        variance_matrix_ : ndarray of shape (n_ref, n_features)
            Per-observation, per-feature SHAP variance across selected models.
        timing_ : dict
            Wall-clock seconds per pipeline stage.
        feature_names_ : list of str
            Feature names (provided or auto-generated).

        Notes
        -----
        If ``len(filtered_indices_) < 2``, fit() raises ValueError. Increase
        ``epsilon`` or switch to ``epsilon_mode="quantile"`` if this occurs.
        """
        if X_ref is None:
            import warnings

            warnings.warn(
                "X_ref not provided; defaulting to X_val. Consider using a "
                "held-out reference set (X_explain) to avoid potential confounds.",
                UserWarning,
            )
            X_ref = X_val
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        if self.selection_method == "surrogate":
            # Surrogate-assisted search replaces Stages 1-3
            t0 = time.time()
            if self.verbose:
                print("=" * 60)
                print("DASH Stages 1-3: Surrogate-Assisted Rashomon Search")
                print("=" * 60)
            from dash_shap.core.rashomon_search import rashomon_search

            (
                self.models_,
                self.val_scores_,
                self.configs_,
                self.selected_indices_,
                self.surrogate_info_,
            ) = rashomon_search(
                X_train,
                y_train,
                X_val,
                y_val,
                X_ref=X_ref,
                search_space=self.search_space,
                budget=self.M,
                batch_size=self.surrogate_batch_size,
                n_initial=self.surrogate_n_initial,
                K=self.K,
                epsilon=self.epsilon,
                epsilon_mode=self.epsilon_mode,
                acquisition=self.surrogate_acquisition,
                delta=self.delta,
                task=self.task,
                n_candidates=self.surrogate_n_candidates,
                seed=self.seed,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
            self.filtered_indices_ = list(range(len(self.models_)))  # all trained
            self.timing_["stages1_3_surrogate"] = time.time() - t0
        else:
            # Standard Stages 1-3: Population → Filter → Diversity
            # Stage 1: Population Generation
            t0 = time.time()
            if self.verbose:
                print("=" * 60)
                print("DASH Stage 1: Population Generation")
                print("=" * 60)
            self.models_, self.val_scores_, self.configs_ = generate_model_population(
                X_train,
                y_train,
                X_val,
                y_val,
                M=self.M,
                task=self.task,
                search_space=self.search_space,
                n_jobs=self.n_jobs,
                seed=self.seed,
                verbose=self.verbose,
            )
            self.timing_["stage1_training"] = time.time() - t0

            # Stage 2: Performance Filtering
            t0 = time.time()
            if self.verbose:
                print(f"\nDASH Stage 2: Performance Filtering (epsilon={self.epsilon})")
            self.filtered_indices_ = performance_filter(
                self.val_scores_,
                epsilon=self.epsilon,
                higher_is_better=True,
                mode=self.epsilon_mode,
                verbose=self.verbose,
            )
            self.timing_["stage2_filtering"] = time.time() - t0
            n_filtered = len(self.filtered_indices_)
            if n_filtered < self.K:
                import warnings

                warnings.warn(
                    f"Only {n_filtered} models passed the performance filter (K={self.K}). "
                    f"Consider increasing epsilon (current: {self.epsilon}) or switching to "
                    f"epsilon_mode='quantile' to guarantee at least K candidates.",
                    UserWarning,
                )
            if n_filtered < 2:
                raise ValueError(f"Only {n_filtered} models passed filter. Increase epsilon.")

            # Stage 3: Diversity Selection
            t0 = time.time()
            if self.verbose:
                print(f"\nDASH Stage 3: Diversity Selection ({self.selection_method})")
            imp_vecs = get_preliminary_importance(
                self.models_,
                self.filtered_indices_,
                X_ref,
                method=self.preliminary_importance_method,
                seed=self.seed,
            )
            filt_scores = {i: self.val_scores_[i] for i in self.filtered_indices_}

            if self.selection_method == "maxmin":
                self.selected_indices_ = greedy_maxmin_selection(
                    imp_vecs,
                    filt_scores,
                    K=self.K,
                    delta=self.delta,
                    verbose=self.verbose,
                )
            elif self.selection_method == "cluster":
                self.selected_indices_ = cluster_coverage_selection(
                    imp_vecs,
                    filt_scores,
                    X_train,
                    tau=self.tau,
                    K=self.K,
                    verbose=self.verbose,
                )
            elif self.selection_method == "dedup":
                self.selected_indices_ = deduplication_selection(
                    imp_vecs,
                    filt_scores,
                    verbose=self.verbose,
                )
                if len(self.selected_indices_) > self.K:
                    self.selected_indices_ = sorted(
                        self.selected_indices_,
                        key=lambda i: self.val_scores_[i],
                        reverse=True,
                    )[: self.K]
            self.timing_["stage3_selection"] = time.time() - t0

        # Stage 4: Consensus SHAP
        t0 = time.time()
        if self.verbose:
            print(f"\nDASH Stage 4: Consensus SHAP (K={len(self.selected_indices_)})")
        self.consensus_matrix_, self.all_shap_matrices_ = compute_consensus(
            self.models_,
            self.selected_indices_,
            X_ref,
            background_size=self.background_size,
            seed=self.seed,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        self.timing_["stage4_shap"] = time.time() - t0

        # Stage 5: Stability Diagnostics
        t0 = time.time()
        if self.verbose:
            print("\nDASH Stage 5: Stability Diagnostics")
        _, self.variance_matrix_, self.fsi_, self.global_importance_ = compute_diagnostics(self.all_shap_matrices_)
        self.timing_["stage5_diagnostics"] = time.time() - t0

        # Build DASHResult (additive — all existing attributes unchanged)
        from dash_shap.core.result import DASHResult

        self.result_ = DASHResult.from_shap_matrices(
            self.all_shap_matrices_,
            feature_names=self.feature_names_,
            val_scores=[self.val_scores_[i] for i in self.selected_indices_],
        )

        if self.verbose:
            total = sum(self.timing_.values())
            if "stages1_3_surrogate" in self.timing_:
                print(
                    f"\nPipeline complete in {total:.1f}s "
                    f"(Surrogate search: {self.timing_['stages1_3_surrogate']:.1f}s, "
                    f"SHAP: {self.timing_['stage4_shap']:.1f}s)"
                )
            else:
                print(
                    f"\nPipeline complete in {total:.1f}s "
                    f"(Training: {self.timing_['stage1_training']:.1f}s, "
                    f"SHAP: {self.timing_['stage4_shap']:.1f}s)"
                )
        return self

    def fit_from_attributions(self, attribution_matrices, val_scores, feature_names=None):
        """Run stages 2–5 on pre-computed (M, n_ref, P) attribution matrices.

        Works with neural nets, linear models, LIME, external SHAP, or any source
        of feature attributions. Follows the ``fit_from_population()`` pattern used
        by the baselines. Absorbs what would have been Extension 11 (Neural) —
        no separate module needed.

        Parameters
        ----------
        attribution_matrices : ndarray of shape (M, n_ref, P)
            One attribution matrix per candidate model.
        val_scores : dict {int: float} or array-like of shape (M,)
            Validation score for each of the M candidate models.
        feature_names : list[str] or None
            Auto-generated if None.

        Returns
        -------
        self
        """
        attribution_matrices = np.asarray(attribution_matrices, dtype=float)
        if attribution_matrices.ndim != 3:
            raise ValueError(f"attribution_matrices must be 3D (M, n_ref, P), got {attribution_matrices.ndim}D")
        M_in, n_ref, P = attribution_matrices.shape

        # Normalise val_scores to dict {int: float}
        if not isinstance(val_scores, dict):
            scores_arr = np.asarray(val_scores, dtype=float)
            val_scores = {i: float(scores_arr[i]) for i in range(len(scores_arr))}

        self.feature_names_ = feature_names or [f"f{i}" for i in range(P)]
        self.models_ = None  # no underlying estimators
        self.val_scores_ = val_scores
        self.configs_ = {}

        # Stage 2: Performance Filtering
        self.filtered_indices_ = performance_filter(
            self.val_scores_,
            epsilon=self.epsilon,
            higher_is_better=True,
            mode=self.epsilon_mode,
            verbose=self.verbose,
        )

        # Stage 3: Diversity Selection
        prelim_importance = np.mean(np.abs(attribution_matrices), axis=1)  # (M, P)
        filt_scores = {i: self.val_scores_[i] for i in self.filtered_indices_}
        self.selected_indices_ = greedy_maxmin_selection(
            {i: prelim_importance[i] for i in self.filtered_indices_},
            filt_scores,
            K=self.K,
            delta=self.delta,
            verbose=self.verbose,
        )

        # Stage 4: Consensus (just average the selected attribution matrices)
        selected = attribution_matrices[list(self.selected_indices_)]  # (K, n_ref, P)
        self.all_shap_matrices_ = selected
        self.consensus_matrix_ = np.mean(selected, axis=0)  # (n_ref, P)

        # Stage 5: Diagnostics
        _, self.variance_matrix_, self.fsi_, self.global_importance_ = compute_diagnostics(self.all_shap_matrices_)

        # Build DASHResult
        from dash_shap.core.result import DASHResult

        self.result_ = DASHResult.from_shap_matrices(
            self.all_shap_matrices_,
            feature_names=self.feature_names_,
            val_scores=[self.val_scores_[i] for i in self.selected_indices_],
        )
        return self

    @property
    def selected_models_(self):
        """Return list of selected models (convenience property)."""
        if self.selected_indices_ is None:
            return None
        return [self.models_[i] for i in self.selected_indices_]

    def get_fsi(self):
        return FeatureStabilityIndex(
            self.fsi_,
            self.global_importance_,
            self.feature_names_,
        )

    def plot_importance_stability(self, groups=None, **kwargs):
        return ImportanceStabilityPlot.plot(
            self.global_importance_,
            self.fsi_,
            feature_names=self.feature_names_,
            groups=groups,
            **kwargs,
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
