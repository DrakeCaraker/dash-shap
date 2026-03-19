"""Stage 3: Diversity-Aware Model Selection."""
import numpy as np
import shap
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from typing import Dict, List

__all__ = [
    "get_preliminary_importance",
    "greedy_maxmin_selection",
    "cluster_coverage_selection",
    "deduplication_selection",
]


def get_preliminary_importance(models, indices, X_ref, method="gain", n_subsample=500, seed=None):
    """Compute preliminary feature importance vectors for each model."""
    P = X_ref.shape[1]
    importance_vectors = {}

    for idx in indices:
        model = models[idx]

        if method == "gain":
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            imp = np.zeros(P)
            for key, val in score.items():
                try:
                    feat_idx = int(key.replace("f", ""))
                    imp[feat_idx] = val
                except (ValueError, IndexError):
                    pass
            importance_vectors[idx] = imp

        elif method == "shap_subsample":
            n_sub = min(n_subsample, len(X_ref))
            if seed is not None:
                rng = np.random.RandomState(seed)
                sub_idx = rng.choice(len(X_ref), n_sub, replace=False)
                X_sub = X_ref[sub_idx]
            else:
                rng = np.random.RandomState(0)
                sub_idx = rng.choice(len(X_ref), n_sub, replace=False)
                X_sub = X_ref[sub_idx]
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sub, check_additivity=False)
            if isinstance(sv, list):
                sv = np.mean([np.abs(s) for s in sv], axis=0)
            importance_vectors[idx] = np.mean(np.abs(sv), axis=0)

    return importance_vectors


def greedy_maxmin_selection(
    importance_vectors,
    performance_scores,
    K=20,
    delta=0.1,
    verbose=True,
):
    """Select up to K models maximizing minimum pairwise diversity."""
    # Pre-compute normalized vectors and pairwise cosine similarity matrix
    indices = list(importance_vectors.keys())
    idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
    normed_matrix = np.array([
        importance_vectors[i] / (np.linalg.norm(importance_vectors[i]) + 1e-10)
        for i in indices
    ])
    sim_matrix = normed_matrix @ normed_matrix.T

    best_idx = max(performance_scores, key=performance_scores.get)
    selected = [best_idx]
    candidates = set(indices) - {best_idx}

    while len(selected) < K and candidates:
        best_candidate, best_min_dist = None, -1.0
        selected_positions = [idx_to_pos[s] for s in selected]
        for c in candidates:
            c_pos = idx_to_pos[c]
            min_dist = min(
                1.0 - sim_matrix[c_pos, s_pos] for s_pos in selected_positions
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c
        if best_min_dist < delta:
            if verbose:
                print(
                    f"  Diversity threshold reached at K={len(selected)} "
                    f"(min_dist={best_min_dist:.4f} < delta={delta})"
                )
            break
        selected.append(best_candidate)
        candidates.discard(best_candidate)

    if verbose:
        print(
            f"MaxMin selection: {len(selected)} models selected "
            f"from {len(importance_vectors)} candidates"
        )
    return selected


def cluster_coverage_selection(
    importance_vectors,
    performance_scores,
    X_train,
    tau=0.3,
    K=20,
    verbose=True,
):
    """Select models to maximize coverage of feature correlation clusters."""
    P = X_train.shape[1]
    corr_matrix = np.abs(np.corrcoef(X_train.T))
    dist_matrix = np.clip(1.0 - corr_matrix, 0, None)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=tau, criterion="distance")

    if verbose:
        print(
            f"  Feature clustering: {len(set(clusters))} clusters "
            f"from {P} features (tau={tau})"
        )

    def get_reps(imp):
        reps = {}
        for cid in set(clusters):
            mask = clusters == cid
            fidx = np.where(mask)[0]
            reps[cid] = fidx[np.argmax(imp[mask])]
        return reps

    model_reps = {i: get_reps(v) for i, v in importance_vectors.items()}
    selected, covered = [], set()
    candidates = set(importance_vectors.keys())

    while len(selected) < K and candidates:
        best_c, best_new = None, -1
        for c in candidates:
            n_new = len(set(model_reps[c].items()) - covered)
            if n_new > best_new or (
                n_new == best_new
                and performance_scores[c]
                > performance_scores.get(best_c, -np.inf)
            ):
                best_new = n_new
                best_c = c

        if best_c is None or best_new == 0:
            for r in sorted(
                candidates,
                key=lambda c: performance_scores[c],
                reverse=True,
            ):
                if len(selected) >= K:
                    break
                selected.append(r)
            break

        selected.append(best_c)
        covered.update(model_reps[best_c].items())
        candidates.discard(best_c)

    if verbose:
        print(f"Cluster coverage selection: {len(selected)} models selected")
    return selected


def deduplication_selection(
    importance_vectors,
    performance_scores,
    rho_threshold=0.95,
    verbose=True,
):
    """Remove near-duplicate models based on importance vector correlation."""
    indices = list(importance_vectors.keys())
    removed = set()

    for i in range(len(indices)):
        if indices[i] in removed:
            continue
        for j in range(i + 1, len(indices)):
            if indices[j] in removed:
                continue
            rho, _ = spearmanr(
                importance_vectors[indices[i]],
                importance_vectors[indices[j]],
            )
            if rho > rho_threshold:
                if performance_scores[indices[i]] >= performance_scores[indices[j]]:
                    removed.add(indices[j])
                else:
                    removed.add(indices[i])
                    break

    selected = [idx for idx in indices if idx not in removed]
    if verbose:
        print(
            f"Deduplication: {len(selected)}/{len(indices)} models retained "
            f"(rho threshold={rho_threshold})"
        )
    return selected
