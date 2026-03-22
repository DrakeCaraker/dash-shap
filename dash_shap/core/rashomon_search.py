"""Surrogate-assisted Rashomon set search for efficient diverse model discovery.

Uses a Gaussian Process surrogate to model the loss surface over hyperparameter
space, then iteratively acquires model configurations that are both likely to be
in the Rashomon set (near-optimal) and diverse from already-discovered models.

Two-phase diversity:
  1. Acquisition phase: hyperparameter-space distance spreads proposals cheaply.
  2. Final selection: importance-space cosine distance (via greedy_maxmin_selection)
     refines the K models for consensus, matching the standard DASH criterion.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from joblib import Parallel, delayed

from dash_shap.core.population import (
    DEFAULT_SEARCH_SPACE,
    train_single_model,
)
from dash_shap.core.filtering import performance_filter
from dash_shap.core.diversity import get_preliminary_importance, greedy_maxmin_selection

__all__ = [
    "encode_config",
    "decode_config",
    "RashomonSurrogate",
    "rashomon_probability_acquisition",
    "diverse_rashomon_acquisition",
    "level_set_boundary_acquisition",
    "rashomon_search",
]


# ---------------------------------------------------------------------------
# Hyperparameter encoding
# ---------------------------------------------------------------------------


def _build_param_map(search_space: dict) -> dict:
    """Build ordinal mapping for each parameter: value → [0, 1] position."""
    param_map = {}
    for key, values in search_space.items():
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            mapping = {sorted_vals[0]: 0.5}
        else:
            mapping = {v: i / (n - 1) for i, v in enumerate(sorted_vals)}
        param_map[key] = {"values": sorted_vals, "to_unit": mapping}
    return param_map


def encode_config(config: dict, search_space: "dict | None" = None) -> np.ndarray:
    """Encode a hyperparameter config to a point in [0, 1]^d.

    Parameters
    ----------
    config : dict
        Hyperparameter configuration with keys matching ``search_space``.
    search_space : dict or None
        Search space definition. If None, uses ``DEFAULT_SEARCH_SPACE``.

    Returns
    -------
    np.ndarray of shape (d,)
        Encoded point in the unit hypercube.
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    param_map = _build_param_map(search_space)
    encoded = []
    for key in search_space:
        val = config[key]
        mapping = param_map[key]["to_unit"]
        # Snap to nearest known value if not exact match
        if val in mapping:
            encoded.append(mapping[val])
        else:
            sorted_vals = param_map[key]["values"]
            closest = min(sorted_vals, key=lambda v: abs(v - val))
            encoded.append(mapping[closest])
    return np.array(encoded, dtype=float)


def decode_config(encoded: np.ndarray, search_space: "dict | None" = None) -> dict:
    """Decode a [0, 1]^d point back to a hyperparameter config.

    Snaps each dimension to the nearest grid value.

    Parameters
    ----------
    encoded : np.ndarray of shape (d,)
        Point in the unit hypercube.
    search_space : dict or None
        Search space definition. If None, uses ``DEFAULT_SEARCH_SPACE``.

    Returns
    -------
    dict
        Hyperparameter configuration with grid-snapped values.
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    param_map = _build_param_map(search_space)
    config = {}
    for i, key in enumerate(search_space):
        sorted_vals = param_map[key]["values"]
        n = len(sorted_vals)
        if n == 1:
            config[key] = sorted_vals[0]
        else:
            # Map [0, 1] → index, snap to nearest
            idx = int(np.round(encoded[i] * (n - 1)))
            idx = max(0, min(n - 1, idx))
            val = sorted_vals[idx]
            # Preserve original types
            if isinstance(val, (int, np.integer)):
                config[key] = int(val)
            else:
                config[key] = float(val)
    return config


def encode_configs(configs: "list[dict]", search_space: "dict | None" = None) -> np.ndarray:
    """Encode a list of configs to an (N, d) array."""
    return np.array([encode_config(c, search_space) for c in configs])


# ---------------------------------------------------------------------------
# GP Surrogate
# ---------------------------------------------------------------------------


class RashomonSurrogate:
    """Gaussian Process surrogate for the validation score surface.

    Wraps ``sklearn.gaussian_process.GaussianProcessRegressor`` with a
    Matérn-2.5 kernel (standard for Bayesian optimization).

    Parameters
    ----------
    n_restarts : int, default=5
        Number of optimizer restarts for kernel hyperparameter fitting.
    """

    def __init__(self, n_restarts: int = 5):
        kernel = Matern(nu=2.5, length_scale=np.ones(1), length_scale_bounds=(1e-3, 10.0)) + WhiteKernel(
            noise_level=1e-3, noise_level_bounds=(1e-5, 1.0)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RashomonSurrogate":
        """Fit the GP on observed (encoded_config, val_score) pairs.

        Parameters
        ----------
        X : np.ndarray of shape (n_obs, d)
            Encoded hyperparameter configurations.
        y : np.ndarray of shape (n_obs,)
            Observed validation scores (higher is better).
        """
        self.gp.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at query points.

        Returns
        -------
        mu : np.ndarray of shape (n_query,)
        sigma : np.ndarray of shape (n_query,)
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu, sigma

    def rashomon_probability(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """P(f(x) >= threshold) under the GP posterior.

        Parameters
        ----------
        X : np.ndarray of shape (n_query, d)
        threshold : float
            Rashomon set lower bound (best_score - epsilon).

        Returns
        -------
        np.ndarray of shape (n_query,)
            Probability of Rashomon membership for each point.
        """
        mu, sigma = self.predict(X)
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-10)
        return norm.sf(threshold, loc=mu, scale=sigma)  # 1 - CDF(threshold)

    def level_set_entropy(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Entropy of Rashomon membership — high at the level set boundary.

        Useful for active level set estimation: focus evaluations on the
        boundary where membership is most uncertain.

        Returns
        -------
        np.ndarray of shape (n_query,)
            Binary entropy H(p) where p = P(f(x) >= threshold).
        """
        p = self.rashomon_probability(X, threshold)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def rashomon_probability_acquisition(
    surrogate: RashomonSurrogate,
    X_candidates: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Acquire points most likely to be in the Rashomon set.

    Parameters
    ----------
    surrogate : RashomonSurrogate
        Fitted GP surrogate.
    X_candidates : np.ndarray of shape (N, d)
        Candidate encoded configs to score.
    threshold : float
        Rashomon threshold (best_score - epsilon).

    Returns
    -------
    np.ndarray of shape (N,)
        Acquisition scores (higher = more promising).
    """
    return surrogate.rashomon_probability(X_candidates, threshold)


def diverse_rashomon_acquisition(
    surrogate: RashomonSurrogate,
    X_candidates: np.ndarray,
    threshold: float,
    X_found: np.ndarray,
    length_scale: float = 1.0,
) -> np.ndarray:
    """Acquire points that are both Rashomon members and diverse.

    Combines P(Rashomon membership) with a soft distance penalty that
    repels proposals from already-discovered configurations in
    hyperparameter space.

    Parameters
    ----------
    surrogate : RashomonSurrogate
        Fitted GP surrogate.
    X_candidates : np.ndarray of shape (N, d)
        Candidate encoded configs.
    threshold : float
        Rashomon threshold.
    X_found : np.ndarray of shape (n_found, d)
        Already-discovered encoded configs (those in R_ε).
    length_scale : float, default=1.0
        Controls how quickly the diversity term saturates with distance.
        Smaller values make the repulsion more local.

    Returns
    -------
    np.ndarray of shape (N,)
        Acquisition scores (higher = more promising).
    """
    prob = surrogate.rashomon_probability(X_candidates, threshold)
    if len(X_found) == 0:
        return prob
    dists = cdist(X_candidates, X_found)
    min_dist = dists.min(axis=1)
    diversity = 1.0 - np.exp(-min_dist / length_scale)
    return prob * diversity


def level_set_boundary_acquisition(
    surrogate: RashomonSurrogate,
    X_candidates: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Acquire points on the Rashomon set boundary (maximum membership entropy).

    Useful for refining the GP's understanding of where R_ε begins and ends,
    rather than just sampling inside it.

    Parameters
    ----------
    surrogate : RashomonSurrogate
        Fitted GP surrogate.
    X_candidates : np.ndarray of shape (N, d)
        Candidate encoded configs.
    threshold : float
        Rashomon threshold.

    Returns
    -------
    np.ndarray of shape (N,)
        Acquisition scores (higher = more uncertain about membership).
    """
    return surrogate.level_set_entropy(X_candidates, threshold)


# ---------------------------------------------------------------------------
# Iterative search loop
# ---------------------------------------------------------------------------


def _generate_random_candidates(
    n_candidates: int,
    search_space: dict,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, list[dict]]:
    """Generate random candidate configs and their encodings."""
    configs = []
    for _ in range(n_candidates):
        config = {}
        for key, values in search_space.items():
            val = rng.choice(values)
            if isinstance(val, np.floating):
                val = float(val)
            elif isinstance(val, np.integer):
                val = int(val)
            config[key] = val
        configs.append(config)
    X_encoded = encode_configs(configs, search_space)
    return X_encoded, configs


def rashomon_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_ref: "np.ndarray | None" = None,
    search_space: "dict | None" = None,
    budget: int = 200,
    batch_size: int = 10,
    n_initial: int = 30,
    K: int = 30,
    epsilon: float = 0.08,
    epsilon_mode: str = "absolute",
    acquisition: str = "diverse_rashomon",
    delta: float = 0.05,
    task: str = "regression",
    n_candidates: int = 500,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 20,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[dict, dict, list, list, dict]:
    """Surrogate-assisted search for diverse Rashomon set members.

    Iteratively trains models guided by a GP surrogate, balancing
    Rashomon membership probability with diversity in hyperparameter space.
    Final K selection uses importance-space MaxMin (same as standard DASH).

    Parameters
    ----------
    X_train, y_train : array-like
        Training data for XGBoost models.
    X_val, y_val : array-like
        Validation data for scoring.
    X_ref : array-like or None
        Reference data for importance computation. Defaults to X_val.
    search_space : dict or None
        Hyperparameter search space. Defaults to ``DEFAULT_SEARCH_SPACE``.
    budget : int, default=200
        Total model evaluations (for fair comparison with M=200).
    batch_size : int, default=10
        Models to acquire per surrogate iteration.
    n_initial : int, default=30
        Random models to train before fitting the first surrogate.
    K : int, default=30
        Number of diverse models to select for consensus.
    epsilon : float, default=0.08
        Rashomon threshold.
    epsilon_mode : str, default="absolute"
        How epsilon is interpreted ("absolute", "relative", "quantile").
    acquisition : str, default="diverse_rashomon"
        Acquisition function name.
    delta : float, default=0.05
        MaxMin diversity stopping threshold.
    task : str, default="regression"
        XGBoost task type.
    n_candidates : int, default=500
        Random candidates to score per acquisition batch.
    n_estimators : int, default=1000
        Max boosting rounds per model.
    early_stopping_rounds : int, default=20
        Early stopping patience.
    seed : int, default=42
        Master random seed.
    n_jobs : int, default=-1
        Parallel jobs for model training.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    models : dict {int: XGBModel}
        All trained models, keyed by index.
    val_scores : dict {int: float}
        Validation scores for all trained models.
    configs : list[dict]
        Hyperparameter configs for all trained models.
    selected_indices : list[int]
        Indices of the K diverse Rashomon members.
    info : dict
        Search diagnostics (n_rashomon, hit_rate, n_iterations, etc.).
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    if X_ref is None:
        X_ref = X_val

    rng = np.random.RandomState(seed)
    n_initial = min(n_initial, budget)

    # ------------------------------------------------------------------
    # Phase 1: Random initial batch
    # ------------------------------------------------------------------
    if verbose:
        print(f"Rashomon search: training {n_initial} initial models...")

    _, init_configs = _generate_random_candidates(n_initial, search_space, rng)

    def _train(i: int, config: dict):
        model, score = train_single_model(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            task=task,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed + i,
        )
        return i, model, score

    results = Parallel(n_jobs=n_jobs)(delayed(_train)(i, c) for i, c in enumerate(init_configs))

    models: dict = {}
    val_scores: dict = {}
    configs: list = list(init_configs)
    for i, model, score in results:
        models[i] = model
        val_scores[i] = score

    n_trained = n_initial

    # ------------------------------------------------------------------
    # Phase 2: Iterative surrogate-guided acquisition
    # ------------------------------------------------------------------
    surrogate = RashomonSurrogate()
    valid_acquisitions = {"rashomon_probability", "diverse_rashomon", "level_set_boundary"}
    if acquisition not in valid_acquisitions:
        raise ValueError(f"Unknown acquisition function '{acquisition}'. Choose from: {sorted(valid_acquisitions)}")

    n_iterations = 0
    while n_trained < budget:
        n_iterations += 1
        actual_batch = min(batch_size, budget - n_trained)

        # Fit surrogate on all observations so far
        X_obs = encode_configs(configs, search_space)
        y_obs = np.array([val_scores[i] for i in range(len(configs))])
        surrogate.fit(X_obs, y_obs)

        # Compute Rashomon threshold
        best_score = max(val_scores.values())
        if epsilon_mode == "absolute":
            threshold = best_score - epsilon
        elif epsilon_mode == "relative":
            threshold = best_score - epsilon * abs(best_score)
        else:
            # For quantile mode, use the quantile as threshold
            all_scores = np.array(list(val_scores.values()))
            threshold = float(np.percentile(all_scores, (1 - epsilon) * 100))

        # Generate candidates and score with acquisition function
        X_cand, cand_configs = _generate_random_candidates(n_candidates, search_space, rng)

        if acquisition == "diverse_rashomon":
            # Find which existing models are in the Rashomon set
            rashomon_mask = y_obs >= threshold
            X_found = X_obs[rashomon_mask] if rashomon_mask.any() else np.empty((0, X_obs.shape[1]))
            scores = diverse_rashomon_acquisition(surrogate, X_cand, threshold, X_found)
        elif acquisition == "level_set_boundary":
            scores = level_set_boundary_acquisition(surrogate, X_cand, threshold)
        else:
            scores = rashomon_probability_acquisition(surrogate, X_cand, threshold)

        # Select top candidates
        top_idx = np.argsort(scores)[::-1][:actual_batch]
        batch_configs = [cand_configs[j] for j in top_idx]

        # Train the batch
        batch_results = Parallel(n_jobs=n_jobs)(delayed(_train)(n_trained + j, c) for j, c in enumerate(batch_configs))

        for j, (idx, model, score) in enumerate(batch_results):
            models[idx] = model
            val_scores[idx] = score
        configs.extend(batch_configs)
        n_trained += actual_batch

        if verbose:
            n_rashomon = sum(1 for s in val_scores.values() if s >= threshold)
            print(
                f"  Iteration {n_iterations}: {n_trained}/{budget} trained, "
                f"{n_rashomon} in Rashomon set, best={best_score:.4f}, threshold={threshold:.4f}"
            )

    # ------------------------------------------------------------------
    # Phase 3: Final selection via importance-space MaxMin
    # ------------------------------------------------------------------
    best_score = max(val_scores.values())
    filtered_indices = performance_filter(
        val_scores,
        epsilon=epsilon,
        higher_is_better=True,
        mode=epsilon_mode,
        verbose=verbose,
    )

    if len(filtered_indices) < 2:
        raise ValueError(
            f"Only {len(filtered_indices)} models in Rashomon set after surrogate search. Increase budget or epsilon."
        )

    # Compute gain-based importance for Rashomon members
    imp_vecs = get_preliminary_importance(
        models,
        filtered_indices,
        X_ref,
        method="gain",
        seed=seed,
    )
    filt_scores = {i: val_scores[i] for i in filtered_indices}

    selected_indices = greedy_maxmin_selection(
        imp_vecs,
        filt_scores,
        K=K,
        delta=delta,
        verbose=verbose,
    )

    # Diagnostics
    n_rashomon = len(filtered_indices)
    hit_rate = n_rashomon / n_trained if n_trained > 0 else 0.0
    info = {
        "n_trained": n_trained,
        "n_rashomon": n_rashomon,
        "n_selected": len(selected_indices),
        "hit_rate": hit_rate,
        "n_iterations": n_iterations,
        "best_score": best_score,
    }

    if verbose:
        print(
            f"Rashomon search complete: {n_trained} trained, "
            f"{n_rashomon} in Rashomon set (hit rate {hit_rate:.1%}), "
            f"{len(selected_indices)} selected"
        )

    return models, val_scores, configs, selected_indices, info
