"""Stage 1: Diversified Model Population Generation."""

import numpy as np
import xgboost as xgb
from itertools import product

from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from tqdm import tqdm

__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "sample_configurations",
    "train_single_model",
    "generate_model_population",
]

DEFAULT_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "colsample_bytree": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0, 5.0, 10.0],
    "reg_lambda": [0, 0.01, 0.1, 1.0, 5.0, 10.0],
    "min_child_weight": [1, 3, 5, 10, 20],
}


def sample_configurations(search_space, M, seed=42, strategy="random"):
    """Sample M hyperparameter configurations from the search space."""
    rng = np.random.RandomState(seed)

    if strategy == "grid":
        keys = list(search_space.keys())
        vals = [search_space[k] for k in keys]
        all_combos = list(product(*vals))
        if len(all_combos) > M:
            indices = rng.choice(len(all_combos), size=M, replace=False)
            all_combos = [all_combos[i] for i in indices]
        return [dict(zip(keys, combo)) for combo in all_combos]

    configs = []
    for _ in range(M):
        config = {k: rng.choice(v) for k, v in search_space.items()}
        config = {
            k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in config.items()
        }
        configs.append(config)
    return configs


def train_single_model(
    config,
    X_train,
    y_train,
    X_val,
    y_val,
    task="regression",
    n_estimators=1000,
    early_stopping_rounds=20,
    seed=42,
    nthread=None,
):
    """Train a single XGBoost model and return (model, validation_score)."""
    thread_kw = {"nthread": nthread} if nthread is not None else {}
    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            random_state=seed,
            verbosity=0,
            **config,
            **thread_kw,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        val_score = -root_mean_squared_error(y_val, preds)

    elif task == "binary":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=seed,
            verbosity=0,
            **config,
            **thread_kw,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, preds)

    elif task == "multiclass":
        n_classes = len(np.unique(y_train))
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            use_label_encoder=False,
            random_state=seed,
            verbosity=0,
            **config,
            **thread_kw,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)
        val_score = roc_auc_score(
            y_val,
            preds,
            multi_class="ovr",
            average="macro",
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    return model, val_score


def generate_model_population(
    X_train,
    y_train,
    X_val,
    y_val,
    M=200,
    task="regression",
    search_space=None,
    configs=None,
    sampling_strategy="random",
    n_estimators=1000,
    early_stopping_rounds=20,
    n_jobs=-1,
    seed=42,
    verbose=True,
    nthread=None,
):
    """Train M diverse XGBoost models and return (models, val_scores, configs).

    Parameters
    ----------
    configs : list of dict or None
        Pre-generated hyperparameter configurations. If provided, skips
        ``sample_configurations`` and trains these exact configs. Used by
        the colsample ablation experiment to control hyperparameters across
        conditions without RNG confounds.
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    if configs is None:
        configs = sample_configurations(
            search_space,
            M,
            seed=seed,
            strategy=sampling_strategy,
        )

    def _train(i, config):
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
            nthread=nthread,
        )
        return i, model, score

    if verbose:
        print(f"Training {M} models with {n_jobs} parallel jobs...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_train)(i, config) for i, config in enumerate(tqdm(configs, disable=not verbose, desc="Training"))
    )

    models, val_scores = {}, {}
    for i, model, score in results:
        models[i] = model
        val_scores[i] = score

    if verbose:
        scores = list(val_scores.values())
        print(f"Population trained. Best: {max(scores):.4f}, Worst: {min(scores):.4f}, Mean: {np.mean(scores):.4f}")

    return models, val_scores, configs
