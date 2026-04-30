"""Stage 4: Consensus SHAP Aggregation."""

import numpy as np
import shap
from joblib import Parallel, delayed
from tqdm import tqdm

__all__ = ["compute_consensus"]


def _compute_shap_for_model(model, bg_data, X_ref):
    """Compute interventional TreeSHAP values for a single model."""
    explainer = shap.TreeExplainer(
        model,
        data=bg_data,
        feature_perturbation="interventional",
    )
    sv = explainer.shap_values(X_ref, check_additivity=False)
    if isinstance(sv, list):
        sv = np.mean(sv, axis=0)
    return sv


def compute_consensus(
    models,
    selected_indices,
    X_ref,
    background_size=100,
    seed=None,
    verbose=True,
    n_jobs=1,
    aggregation="mean",
    groups=None,
):
    """Compute consensus SHAP matrix via element-wise averaging.

    Uses interventional TreeSHAP. Note: SHAP interaction values are NOT
    preserved under this averaging — they would need separate computation.

    Parameters
    ----------
    seed : int or None
        If provided, randomly samples background rows from X_ref.
        If None, uses the first ``background_size`` rows (deterministic).
    n_jobs : int
        Number of parallel jobs for SHAP computation. Default 1 (sequential).
        Set to -1 to use all available cores.
    aggregation : str
        'mean' (default): element-wise mean across models.
        'pca': within each group, project SHAP onto first principal component.
            More robust than sum for opposite-directional features (e.g.,
            MedInc-AveOccup). Requires ``groups`` parameter.
    groups : list of list of int or None
        Feature index groups for PCA aggregation. Required if aggregation='pca'.
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

    if n_jobs == 1:
        iterator = enumerate(selected_indices)
        if verbose:
            iterator = tqdm(list(iterator), desc="Computing SHAP")
        for k, idx in iterator:
            sv = _compute_shap_for_model(models[idx], bg_data, X_ref)
            if sv.shape != (N_prime, P):
                raise ValueError(f"Model {idx}: expected SHAP shape {(N_prime, P)}, got {sv.shape}")
            all_shap[k] = sv
    else:
        if verbose:
            print(f"Computing SHAP for {K} models with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_shap_for_model)(models[idx], bg_data, X_ref) for idx in selected_indices
        )
        for k, sv in enumerate(results):
            if sv.shape != (N_prime, P):
                raise ValueError(f"Model at position {k}: expected SHAP shape {(N_prime, P)}, got {sv.shape}")
            all_shap[k] = sv

    if aggregation == "pca" and groups is not None:
        from sklearn.decomposition import PCA

        consensus = np.mean(all_shap, axis=0)  # start with mean
        for group in groups:
            if len(group) < 2:
                continue
            # For each model, extract group SHAP and project onto PC1
            group_shap = all_shap[:, :, group]  # (K, N', G)
            # Reshape to (K*N', G) for PCA
            flat = group_shap.reshape(-1, len(group))
            pca = PCA(n_components=1)
            pca.fit(flat)
            loadings = pca.components_[0]  # (G,)
            # Ensure loadings point in the direction of positive mean importance
            mean_imp = np.mean(np.abs(consensus[:, group]), axis=0)
            if np.dot(loadings, mean_imp) < 0:
                loadings = -loadings
            # Project each model's group SHAP onto PC1, distribute back
            for k in range(K):
                scores = group_shap[k] @ loadings  # (N',)
                for idx, feat in enumerate(group):
                    all_shap[k, :, feat] = scores * loadings[idx]
        consensus = np.mean(all_shap, axis=0)
    else:
        consensus = np.mean(all_shap, axis=0)

    if verbose:
        global_imp = np.mean(np.abs(consensus), axis=0)
        top_5 = np.argsort(global_imp)[-5:][::-1]
        agg_label = f" ({aggregation})" if aggregation != "mean" else ""
        print(f"Consensus{agg_label} computed from {K} models. Top 5 features: {top_5.tolist()}")

    return consensus, all_shap
