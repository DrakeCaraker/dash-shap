"""Stage 4 (NN variant): Attribution computation for neural network models."""

from __future__ import annotations

import numpy as np
import shap
from joblib import Parallel, delayed
from tqdm import tqdm

__all__ = ["compute_nn_attributions"]


def _compute_kernel_shap_for_model(model: object, bg_data: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """Compute KernelSHAP values for a single model."""
    explainer = shap.KernelExplainer(model.predict, bg_data)  # type: ignore[attr-defined]
    sv = explainer.shap_values(X_ref, nsamples="auto", silent=True)
    if isinstance(sv, list):
        sv = np.mean(sv, axis=0)
    return sv


def compute_nn_attributions(
    models: dict | list,
    selected_indices: list[int],
    X_ref: np.ndarray,
    background_size: int = 100,
    method: str = "kernel",
    seed: int | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute consensus attribution matrix for neural network models.

    Uses KernelSHAP by default. Gradient-based methods require torch + captum.

    Parameters
    ----------
    models : list
        Trained models exposing a ``.predict()`` method.
    selected_indices : list[int]
        Indices into *models* for the selected subset.
    X_ref : np.ndarray
        Reference data used for both background sampling and explanation.
    background_size : int
        Number of background samples for the explainer.
    method : str
        Attribution method: ``"kernel"`` (default), ``"gradient"``, or ``"ig"``.
    seed : int or None
        If provided, randomly samples background rows from X_ref.
        If None, uses the first ``background_size`` rows (deterministic).
    n_jobs : int
        Number of parallel jobs. Default 1 (sequential). Set to -1 for all cores.
    verbose : bool
        Print progress information.

    Returns
    -------
    consensus : np.ndarray
        Element-wise mean of attribution matrices, shape ``(N', P)``.
    all_shap : np.ndarray
        Stacked attribution matrices, shape ``(K, N', P)``.
    """
    if method == "kernel":
        compute_fn = _compute_kernel_shap_for_model
    elif method == "gradient":
        raise ImportError("Install torch and captum for gradient-based attribution methods: pip install torch captum")
    elif method == "ig":
        raise ImportError("Install torch and captum for Integrated Gradients: pip install torch captum")
    else:
        raise ValueError(f"Unknown attribution method: {method}")

    K = len(selected_indices)
    N_prime, P = X_ref.shape
    all_shap = np.zeros((K, N_prime, P))
    n_bg = min(background_size, N_prime)

    if seed is not None:
        rng = np.random.RandomState(seed)
        bg_idx = rng.choice(N_prime, size=n_bg, replace=False)
        bg_data = X_ref[bg_idx]
    else:
        bg_data = X_ref[:n_bg]

    if n_jobs == 1:
        iterator = enumerate(selected_indices)
        if verbose:
            iterator = tqdm(list(iterator), desc="Computing NN attributions")
        for k, idx in iterator:
            sv = compute_fn(models[idx], bg_data, X_ref)
            if sv.shape != (N_prime, P):
                raise ValueError(f"Model {idx}: expected shape {(N_prime, P)}, got {sv.shape}")
            all_shap[k] = sv
    else:
        if verbose:
            print(f"Computing NN attributions for {K} models with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs)(delayed(compute_fn)(models[idx], bg_data, X_ref) for idx in selected_indices)
        for k, sv in enumerate(results):
            if sv.shape != (N_prime, P):
                raise ValueError(f"Model at position {k}: expected shape {(N_prime, P)}, got {sv.shape}")
            all_shap[k] = sv

    consensus = np.mean(all_shap, axis=0)

    if verbose:
        global_imp = np.mean(np.abs(consensus), axis=0)
        top_5 = np.argsort(global_imp)[-5:][::-1]
        print(f"Consensus computed from {K} models. Top 5 features: {top_5.tolist()}")

    return consensus, all_shap
