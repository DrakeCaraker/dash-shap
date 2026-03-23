"""Stage 1 (NN variant): Neural Network Population Generation."""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from tqdm import tqdm

__all__ = [
    "NN_SEARCH_SPACE",
    "sample_nn_configurations",
    "train_single_nn",
    "generate_nn_population",
]

NN_SEARCH_SPACE: dict[str, list] = {
    "hidden_layer_sizes": [
        (64, 64),
        (128, 64),
        (128, 128),
        (256, 128),
        (128, 128, 64),
    ],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "learning_rate_init": [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
    "activation": ["relu", "tanh"],
    "batch_size": [32, 64, 128, 256],
}


def sample_nn_configurations(
    search_space: dict[str, list],
    M: int,
    seed: int = 42,
) -> list[dict]:
    """Sample M hyperparameter configurations from the search space."""
    rng = np.random.RandomState(seed)
    configs: list[dict] = []
    for _ in range(M):
        config = {k: rng.choice(v) for k, v in search_space.items()}
        config = {
            k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in config.items()
        }
        # hidden_layer_sizes must be a tuple, not a numpy array
        if "hidden_layer_sizes" in config and not isinstance(config["hidden_layer_sizes"], tuple):
            config["hidden_layer_sizes"] = tuple(config["hidden_layer_sizes"])
        configs.append(config)
    return configs


def train_single_nn(
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "regression",
    max_iter: int = 500,
    seed: int = 42,
    feature_mask: np.ndarray | None = None,
) -> tuple:
    """Train a single neural network model and return (model, validation_score)."""
    # Apply feature mask if provided (zero out masked columns on copies)
    if feature_mask is not None:
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_train[:, ~feature_mask] = 0.0
        X_val[:, ~feature_mask] = 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if task == "regression":
            model = MLPRegressor(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                activation=config["activation"],
                batch_size=config["batch_size"],
                early_stopping=False,
                max_iter=max_iter,
                random_state=seed,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            val_score = -root_mean_squared_error(y_val, preds)

        elif task == "binary":
            model = MLPClassifier(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                activation=config["activation"],
                batch_size=config["batch_size"],
                early_stopping=False,
                max_iter=max_iter,
                random_state=seed,
            )
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, preds)

        else:
            raise ValueError(f"Unknown task: {task}")

    return model, val_score


def generate_nn_population(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    M: int = 200,
    task: str = "regression",
    seed: int = 42,
    search_space: dict[str, list] | None = None,
    n_jobs: int = 1,
    feature_mask_fraction: float | None = None,
    verbose: bool = True,
) -> tuple[dict, dict, list[dict]]:
    """Train M diverse neural network models and return (models, val_scores, configs)."""
    if search_space is None:
        search_space = NN_SEARCH_SPACE

    configs = sample_nn_configurations(search_space, M, seed=seed)

    n_features = X_train.shape[1]

    def _train(i: int, config: dict) -> tuple[int, object, float]:
        # Generate per-model feature mask if requested
        mask = None
        if feature_mask_fraction is not None:
            mask_rng = np.random.RandomState(seed + i + 10000)
            mask = mask_rng.random(n_features) >= feature_mask_fraction
            # Ensure at least one feature is kept
            if not mask.any():
                mask[mask_rng.randint(n_features)] = True

        model, score = train_single_nn(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            task=task,
            max_iter=500,
            seed=seed + i,
            feature_mask=mask,
        )
        return i, model, score

    if verbose:
        print(f"Training {M} NN models with {n_jobs} parallel jobs...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_train)(i, config) for i, config in enumerate(tqdm(configs, disable=not verbose, desc="Training NNs"))
    )

    models: dict = {}
    val_scores: dict = {}
    for i, model, score in results:
        models[i] = model
        val_scores[i] = score

    if verbose:
        scores = list(val_scores.values())
        print(f"NN population trained. Best: {max(scores):.4f}, Worst: {min(scores):.4f}, Mean: {np.mean(scores):.4f}")

    return models, val_scores, configs
