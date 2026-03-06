"""Stage 4: Consensus SHAP Aggregation."""
import numpy as np
import shap
from typing import Dict, List, Tuple
from tqdm import tqdm

__all__ = ["compute_consensus"]


def compute_consensus(
    models,
    selected_indices,
    X_ref,
    background_size=100,
    seed=None,
    verbose=True,
):
    """Compute consensus SHAP matrix via element-wise averaging.

    Uses interventional TreeSHAP. Note: SHAP interaction values are NOT
    preserved under this averaging — they would need separate computation.

    Parameters
    ----------
    seed : int or None
        If provided, randomly samples background rows from X_ref.
        If None, uses the first ``background_size`` rows (deterministic).
    """
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

    iterator = enumerate(selected_indices)
    if verbose:
        iterator = tqdm(list(iterator), desc="Computing SHAP")

    for k, idx in iterator:
        model = models[idx]
        explainer = shap.TreeExplainer(
            model, data=bg_data, feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean(sv, axis=0)
        all_shap[k] = sv

    consensus = np.mean(all_shap, axis=0)

    if verbose:
        global_imp = np.mean(np.abs(consensus), axis=0)
        top_5 = np.argsort(global_imp)[-5:][::-1]
        print(
            f"Consensus computed from {K} models. "
            f"Top 5 features: {top_5.tolist()}"
        )

    return consensus, all_shap
