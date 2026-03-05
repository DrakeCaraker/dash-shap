"""Stage 4: Consensus SHAP Aggregation."""
import numpy as np
import shap
from typing import Dict, List, Tuple
from tqdm import tqdm

def compute_consensus(models, selected_indices, X_ref, background_size=100, verbose=True):
    """Compute consensus SHAP matrix via element-wise averaging.
    Uses interventional TreeSHAP. Note: SHAP interaction values are NOT
    preserved under this averaging — they would need separate computation."""
    K = len(selected_indices)
    N_prime, P = X_ref.shape
    all_shap = np.zeros((K, N_prime, P))
    bg_data = X_ref[:min(background_size, N_prime)]
    iterator = enumerate(selected_indices)
    if verbose:
        iterator = tqdm(list(iterator), desc="Computing SHAP")
    for k, idx in iterator:
        model = models[idx]
        explainer = shap.TreeExplainer(model, data=bg_data, feature_perturbation="interventional")
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean(sv, axis=0)
        all_shap[k] = sv
    consensus = np.mean(all_shap, axis=0)
    if verbose:
        global_imp = np.mean(np.abs(consensus), axis=0)
        top_5 = np.argsort(global_imp)[-5:][::-1]
        print(f"Consensus computed from {K} models. Top 5 features: {top_5.tolist()}")
    return consensus, all_shap

def compute_diagnostics(all_shap_matrices, epsilon=1e-8):
    consensus = np.mean(all_shap_matrices, axis=0)
    variance_matrix = np.var(all_shap_matrices, axis=0, ddof=1)
    std_matrix = np.sqrt(variance_matrix)
    global_importance = np.mean(np.abs(consensus), axis=0)
    mean_std = np.mean(std_matrix, axis=0)
    mean_abs_consensus = np.mean(np.abs(consensus), axis=0)
    fsi = mean_std / (mean_abs_consensus + epsilon)
    return consensus, variance_matrix, fsi, global_importance
