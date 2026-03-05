#!/bin/bash
# Run from the dash-shap project root:
#   bash setup_dash.sh

set -e
echo "Creating DASH package structure..."

mkdir -p dash/core dash/baselines dash/experiments dash/evaluation dash/utils scripts notebooks

# ─── dash/__init__.py ───
cat > dash/__init__.py << 'EOF'
"""DASH: Diversified Aggregation of SHAP for Stable Feature Importance Under Feature Collinearity."""
__version__ = "0.1.0"

def __getattr__(name):
    if name == "DASHPipeline":
        from dash.core.pipeline import DASHPipeline
        return DASHPipeline
    elif name == "FeatureStabilityIndex":
        from dash.core.diagnostics import FeatureStabilityIndex
        return FeatureStabilityIndex
    elif name == "ImportanceStabilityPlot":
        from dash.core.diagnostics import ImportanceStabilityPlot
        return ImportanceStabilityPlot
    elif name in ("compute_consensus", "compute_diagnostics"):
        from dash.core import consensus
        return getattr(consensus, name)
    raise AttributeError(f"module 'dash' has no attribute {name}")
EOF

# ─── dash/core/__init__.py ───
cat > dash/core/__init__.py << 'EOF'
# Lazy imports
EOF

# ─── dash/core/population.py ───
cat > dash/core/population.py << 'PYEOF'
"""Stage 1: Diversified Model Population Generation."""
import numpy as np
import xgboost as xgb
from itertools import product
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

DEFAULT_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "colsample_bytree": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0, 5.0, 10.0],
    "reg_lambda": [0, 0.01, 0.1, 1.0, 5.0, 10.0],
    "min_child_weight": [1, 3, 5, 10, 20],
}

def _sample_configurations(search_space, M, seed=42, strategy="random"):
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
        config = {k: float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v for k, v in config.items()}
        configs.append(config)
    return configs

def _train_single_model(config, X_train, y_train, X_val, y_val, task="regression", n_estimators=1000, early_stopping_rounds=20, seed=42):
    if task == "regression":
        model = xgb.XGBRegressor(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, eval_metric="rmse", random_state=seed, verbosity=0, **config)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        from sklearn.metrics import mean_squared_error
        preds = model.predict(X_val)
        val_score = -mean_squared_error(y_val, preds, squared=False)
    elif task == "binary":
        model = xgb.XGBClassifier(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, eval_metric="auc", use_label_encoder=False, random_state=seed, verbosity=0, **config)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        from sklearn.metrics import roc_auc_score
        preds = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, preds)
    elif task == "multiclass":
        n_classes = len(np.unique(y_train))
        model = xgb.XGBClassifier(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, eval_metric="mlogloss", objective="multi:softprob", num_class=n_classes, use_label_encoder=False, random_state=seed, verbosity=0, **config)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        from sklearn.metrics import roc_auc_score
        preds = model.predict_proba(X_val)
        val_score = roc_auc_score(y_val, preds, multi_class="ovr", average="macro")
    else:
        raise ValueError(f"Unknown task: {task}")
    return model, val_score

def generate_model_population(X_train, y_train, X_val, y_val, M=200, task="regression", search_space=None, sampling_strategy="random", n_estimators=1000, early_stopping_rounds=20, n_jobs=-1, seed=42, verbose=True):
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    configs = _sample_configurations(search_space, M, seed=seed, strategy=sampling_strategy)
    def _train(i, config):
        model, score = _train_single_model(config, X_train, y_train, X_val, y_val, task=task, n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, seed=seed + i)
        return i, model, score
    if verbose:
        print(f"Training {M} models with {n_jobs} parallel jobs...")
    results = Parallel(n_jobs=n_jobs)(delayed(_train)(i, config) for i, config in enumerate(tqdm(configs, disable=not verbose, desc="Training")))
    models, val_scores = {}, {}
    for i, model, score in results:
        models[i] = model
        val_scores[i] = score
    if verbose:
        scores = list(val_scores.values())
        print(f"Population trained. Best: {max(scores):.4f}, Worst: {min(scores):.4f}, Mean: {np.mean(scores):.4f}")
    return models, val_scores, configs
PYEOF

# ─── dash/core/filtering.py ───
cat > dash/core/filtering.py << 'PYEOF'
"""Stage 2: Performance Filtering."""
import numpy as np
from typing import Dict, List

def performance_filter(val_scores, epsilon=0.02, higher_is_better=True, verbose=True):
    best_score = max(val_scores.values()) if higher_is_better else min(val_scores.values())
    filtered = [i for i, s in val_scores.items() if abs(s - best_score) <= epsilon]
    if verbose:
        print(f"Performance filter: {len(filtered)}/{len(val_scores)} models within ε={epsilon} of best ({best_score:.4f})")
    return filtered
PYEOF

# ─── dash/core/diversity.py ───
cat > dash/core/diversity.py << 'PYEOF'
"""Stage 3: Diversity-Aware Model Selection."""
import numpy as np
import shap
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from typing import Dict, List

def get_preliminary_importance(models, indices, X_ref, method="gain", n_subsample=500):
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
            X_sub = X_ref[:min(n_subsample, len(X_ref))]
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sub)
            if isinstance(sv, list):
                sv = np.mean([np.abs(s) for s in sv], axis=0)
            importance_vectors[idx] = np.mean(np.abs(sv), axis=0)
    return importance_vectors

def greedy_maxmin_selection(importance_vectors, performance_scores, K=20, delta=0.1, verbose=True):
    normed = {i: v / (np.linalg.norm(v) + 1e-10) for i, v in importance_vectors.items()}
    best_idx = max(performance_scores, key=performance_scores.get)
    selected = [best_idx]
    candidates = set(importance_vectors.keys()) - {best_idx}
    while len(selected) < K and candidates:
        best_candidate, best_min_dist = None, -1.0
        for c in candidates:
            min_dist = min(1.0 - np.dot(normed[c], normed[s]) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c
        if best_min_dist < delta:
            if verbose:
                print(f"  Diversity threshold reached at K={len(selected)} (min_dist={best_min_dist:.4f} < δ={delta})")
            break
        selected.append(best_candidate)
        candidates.discard(best_candidate)
    if verbose:
        print(f"MaxMin selection: {len(selected)} models selected from {len(importance_vectors)} candidates")
    return selected

def cluster_coverage_selection(importance_vectors, performance_scores, X_train, tau=0.3, K=20, verbose=True):
    P = X_train.shape[1]
    corr_matrix = np.abs(np.corrcoef(X_train.T))
    dist_matrix = np.clip(1.0 - corr_matrix, 0, None)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=tau, criterion="distance")
    if verbose:
        print(f"  Feature clustering: {len(set(clusters))} clusters from {P} features (τ={tau})")
    def get_reps(imp):
        reps = {}
        for cid in set(clusters):
            mask = clusters == cid
            fidx = np.where(mask)[0]
            reps[cid] = fidx[np.argmax(imp[mask])]
        return reps
    model_reps = {i: get_reps(v) for i, v in importance_vectors.items()}
    selected, covered, candidates = [], set(), set(importance_vectors.keys())
    while len(selected) < K and candidates:
        best_c, best_new = None, -1
        for c in candidates:
            n_new = len(set(model_reps[c].items()) - covered)
            if n_new > best_new or (n_new == best_new and performance_scores[c] > performance_scores.get(best_c, -np.inf)):
                best_new = n_new
                best_c = c
        if best_c is None or best_new == 0:
            for r in sorted(candidates, key=lambda c: performance_scores[c], reverse=True):
                if len(selected) >= K: break
                selected.append(r)
            break
        selected.append(best_c)
        covered.update(model_reps[best_c].items())
        candidates.discard(best_c)
    if verbose:
        print(f"Cluster coverage selection: {len(selected)} models selected")
    return selected

def deduplication_selection(importance_vectors, performance_scores, rho_threshold=0.95, verbose=True):
    indices = list(importance_vectors.keys())
    removed = set()
    for i in range(len(indices)):
        if indices[i] in removed: continue
        for j in range(i + 1, len(indices)):
            if indices[j] in removed: continue
            rho, _ = spearmanr(importance_vectors[indices[i]], importance_vectors[indices[j]])
            if rho > rho_threshold:
                if performance_scores[indices[i]] >= performance_scores[indices[j]]:
                    removed.add(indices[j])
                else:
                    removed.add(indices[i])
                    break
    selected = [idx for idx in indices if idx not in removed]
    if verbose:
        print(f"Deduplication: {len(selected)}/{len(indices)} models retained (ρ threshold={rho_threshold})")
    return selected
PYEOF

# ─── dash/core/consensus.py ───
cat > dash/core/consensus.py << 'PYEOF'
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
PYEOF

# ─── dash/core/diagnostics.py ───
cat > dash/core/diagnostics.py << 'PYEOF'
"""Stage 5: Stability Diagnostics — FSI, IS Plot, disagreement maps."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

class FeatureStabilityIndex:
    def __init__(self, fsi, global_importance, feature_names=None):
        self.fsi = fsi
        self.global_importance = global_importance
        self.P = len(fsi)
        self.feature_names = feature_names or [f"f{i}" for i in range(self.P)]

    def get_quadrant_labels(self, importance_threshold=None, fsi_threshold=None):
        if importance_threshold is None:
            importance_threshold = np.median(self.global_importance)
        if fsi_threshold is None:
            high_imp_mask = self.global_importance >= importance_threshold
            fsi_threshold = np.median(self.fsi[high_imp_mask]) if high_imp_mask.any() else np.median(self.fsi)
        labels = np.empty(self.P, dtype=object)
        for j in range(self.P):
            hi = self.global_importance[j] >= importance_threshold
            hf = self.fsi[j] >= fsi_threshold
            if hi and not hf: labels[j] = "I: Robust Drivers"
            elif hi and hf: labels[j] = "II: Collinear Cluster"
            elif not hi and not hf: labels[j] = "III: Confirmed Unimportant"
            else: labels[j] = "IV: Fragile Interactions"
        return labels

    def summary(self, top_k=10):
        order = np.argsort(self.global_importance)[::-1]
        lines = ["Feature Stability Summary", "=" * 40, f"{'Feature':<20} {'Importance':>12} {'FSI':>8}", "-" * 40]
        for j in order[:top_k]:
            lines.append(f"{self.feature_names[j]:<20} {self.global_importance[j]:>12.4f} {self.fsi[j]:>8.3f}")
        return "\n".join(lines)

class ImportanceStabilityPlot:
    @staticmethod
    def plot(global_importance, fsi, feature_names=None, groups=None, importance_threshold=None, fsi_threshold=None, title="Importance-Stability Plot", figsize=(10, 7), annotate_top_k=5, ax=None):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(fsi))]
        if importance_threshold is None:
            importance_threshold = np.median(global_importance)
        if fsi_threshold is None:
            high_mask = global_importance >= importance_threshold
            fsi_threshold = np.median(fsi[high_mask]) if high_mask.any() else np.median(fsi)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        if groups is not None:
            unique_groups = np.unique(groups)
            cmap = plt.cm.get_cmap("tab10", len(unique_groups))
            for i, g in enumerate(unique_groups):
                mask = groups == g
                ax.scatter(global_importance[mask], fsi[mask], c=[cmap(i)], label=f"Group {g}", s=60, alpha=0.7, edgecolors="k", linewidths=0.5)
        else:
            fsi_obj = FeatureStabilityIndex(fsi, global_importance, feature_names)
            labels = fsi_obj.get_quadrant_labels(importance_threshold, fsi_threshold)
            colors_map = {"I: Robust Drivers": "#2ecc71", "II: Collinear Cluster": "#e74c3c", "III: Confirmed Unimportant": "#95a5a6", "IV: Fragile Interactions": "#f39c12"}
            for label, color in colors_map.items():
                mask = labels == label
                if mask.any():
                    ax.scatter(global_importance[mask], fsi[mask], c=color, label=label, s=60, alpha=0.7, edgecolors="k", linewidths=0.5)
        ax.axvline(importance_threshold, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(fsi_threshold, color="gray", linestyle="--", alpha=0.5)
        top_k_idx = np.argsort(global_importance)[-annotate_top_k:][::-1]
        for j in top_k_idx:
            ax.annotate(feature_names[j], (global_importance[j], fsi[j]), fontsize=8, alpha=0.8, xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Consensus Importance $\\bar{I}_j$", fontsize=12)
        ax.set_ylabel("Feature Stability Index (FSI$_j$)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        return fig

def local_disagreement_map(all_shap_matrices, observation_idx, feature_names=None, top_k=15, figsize=(10, 6)):
    K, N_prime, P = all_shap_matrices.shape
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(P)]
    consensus_i = np.mean(all_shap_matrices[:, observation_idx, :], axis=0)
    std_i = np.std(all_shap_matrices[:, observation_idx, :], axis=0, ddof=1)
    order = np.argsort(np.abs(consensus_i))[::-1][:top_k]
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(order))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in consensus_i[order]]
    ax.barh(y_pos, consensus_i[order], xerr=std_i[order], color=colors, alpha=0.7, edgecolor="k", linewidth=0.5, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[j] for j in order], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("SHAP Value (consensus ± 1 std)", fontsize=11)
    ax.set_title(f"Local Disagreement Map — Observation {observation_idx}", fontsize=13)
    fig.tight_layout()
    return fig
PYEOF

# ─── dash/core/pipeline.py ───
cat > dash/core/pipeline.py << 'PYEOF'
"""DASHPipeline: End-to-end orchestration of all five DASH stages."""
import numpy as np
import time
from typing import Dict, List, Optional
from dash.core.population import generate_model_population, DEFAULT_SEARCH_SPACE
from dash.core.filtering import performance_filter
from dash.core.diversity import get_preliminary_importance, greedy_maxmin_selection, cluster_coverage_selection, deduplication_selection
from dash.core.consensus import compute_consensus, compute_diagnostics
from dash.core.diagnostics import FeatureStabilityIndex, ImportanceStabilityPlot

class DASHPipeline:
    def __init__(self, M=200, K=20, epsilon=0.02, selection_method="maxmin", delta=0.1, tau=0.3, task="regression", search_space=None, preliminary_importance_method="gain", background_size=100, n_jobs=-1, seed=42, verbose=True):
        self.M, self.K, self.epsilon = M, K, epsilon
        self.selection_method, self.delta, self.tau = selection_method, delta, tau
        self.task = task
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.preliminary_importance_method = preliminary_importance_method
        self.background_size = background_size
        self.n_jobs, self.seed, self.verbose = n_jobs, seed, verbose
        self.models_ = self.val_scores_ = self.configs_ = None
        self.filtered_indices_ = self.selected_indices_ = None
        self.consensus_matrix_ = self.all_shap_matrices_ = None
        self.fsi_ = self.global_importance_ = self.variance_matrix_ = None
        self.timing_ = {}

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, feature_names=None):
        if X_ref is None: X_ref = X_val
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        t0 = time.time()
        if self.verbose: print("=" * 60 + "\nDASH Stage 1: Population Generation\n" + "=" * 60)
        self.models_, self.val_scores_, self.configs_ = generate_model_population(X_train, y_train, X_val, y_val, M=self.M, task=self.task, search_space=self.search_space, n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose)
        self.timing_["stage1_training"] = time.time() - t0

        t0 = time.time()
        if self.verbose: print(f"\nDASH Stage 2: Performance Filtering (ε={self.epsilon})")
        self.filtered_indices_ = performance_filter(self.val_scores_, epsilon=self.epsilon, higher_is_better=True, verbose=self.verbose)
        self.timing_["stage2_filtering"] = time.time() - t0
        if len(self.filtered_indices_) < 2:
            raise ValueError(f"Only {len(self.filtered_indices_)} models passed filter. Increase epsilon.")

        t0 = time.time()
        if self.verbose: print(f"\nDASH Stage 3: Diversity Selection ({self.selection_method})")
        imp_vecs = get_preliminary_importance(self.models_, self.filtered_indices_, X_ref, method=self.preliminary_importance_method)
        filt_scores = {i: self.val_scores_[i] for i in self.filtered_indices_}
        if self.selection_method == "maxmin":
            self.selected_indices_ = greedy_maxmin_selection(imp_vecs, filt_scores, K=self.K, delta=self.delta, verbose=self.verbose)
        elif self.selection_method == "cluster":
            self.selected_indices_ = cluster_coverage_selection(imp_vecs, filt_scores, X_train, tau=self.tau, K=self.K, verbose=self.verbose)
        elif self.selection_method == "dedup":
            self.selected_indices_ = deduplication_selection(imp_vecs, filt_scores, verbose=self.verbose)
            if len(self.selected_indices_) > self.K:
                self.selected_indices_ = sorted(self.selected_indices_, key=lambda i: self.val_scores_[i], reverse=True)[:self.K]
        self.timing_["stage3_selection"] = time.time() - t0

        t0 = time.time()
        if self.verbose: print(f"\nDASH Stage 4: Consensus SHAP (K={len(self.selected_indices_)})")
        self.consensus_matrix_, self.all_shap_matrices_ = compute_consensus(self.models_, self.selected_indices_, X_ref, background_size=self.background_size, verbose=self.verbose)
        self.timing_["stage4_shap"] = time.time() - t0

        t0 = time.time()
        if self.verbose: print("\nDASH Stage 5: Stability Diagnostics")
        _, self.variance_matrix_, self.fsi_, self.global_importance_ = compute_diagnostics(self.all_shap_matrices_)
        self.timing_["stage5_diagnostics"] = time.time() - t0
        if self.verbose:
            total = sum(self.timing_.values())
            print(f"\nPipeline complete in {total:.1f}s (Training: {self.timing_['stage1_training']:.1f}s, SHAP: {self.timing_['stage4_shap']:.1f}s)")
        return self

    def get_fsi(self):
        return FeatureStabilityIndex(self.fsi_, self.global_importance_, self.feature_names_)

    def plot_importance_stability(self, groups=None, **kwargs):
        return ImportanceStabilityPlot.plot(self.global_importance_, self.fsi_, feature_names=self.feature_names_, groups=groups, **kwargs)

    def get_importance_ranking(self):
        return np.argsort(self.global_importance_)[::-1]

    def get_consensus_ensemble_predictions(self, X):
        preds = []
        for idx in self.selected_indices_:
            model = self.models_[idx]
            if self.task == "regression": preds.append(model.predict(X))
            else: preds.append(model.predict_proba(X))
        return np.mean(preds, axis=0)
PYEOF

# ─── dash/baselines/__init__.py ───
cat > dash/baselines/__init__.py << 'EOF'
def __getattr__(name):
    if name == "SingleBestBaseline":
        from dash.baselines.single_best import SingleBestBaseline
        return SingleBestBaseline
    elif name == "LargeSingleModelBaseline":
        from dash.baselines.large_single import LargeSingleModelBaseline
        return LargeSingleModelBaseline
    elif name == "NaiveAveragingBaseline":
        from dash.baselines.naive_averaging import NaiveAveragingBaseline
        return NaiveAveragingBaseline
    elif name == "StochasticRetrainBaseline":
        from dash.baselines.stochastic_retrain import StochasticRetrainBaseline
        return StochasticRetrainBaseline
    elif name == "EnsembleSHAPBaseline":
        from dash.baselines.ensemble_shap import EnsembleSHAPBaseline
        return EnsembleSHAPBaseline
    raise AttributeError(f"module 'dash.baselines' has no attribute {name}")
EOF

# ─── dash/baselines/single_best.py ───
cat > dash/baselines/single_best.py << 'PYEOF'
"""Baseline: Single Best Model."""
import numpy as np
import shap
from dash.core.population import DEFAULT_SEARCH_SPACE, _sample_configurations, _train_single_model

class SingleBestBaseline:
    def __init__(self, n_trials=100, task="regression", seed=42):
        self.n_trials, self.task, self.seed = n_trials, task, seed
        self.model_ = self.global_importance_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None: X_ref = X_val
        configs = _sample_configurations(DEFAULT_SEARCH_SPACE, self.n_trials, seed=self.seed)
        best_score, best_model = -np.inf, None
        for i, config in enumerate(configs):
            model, score = _train_single_model(config, X_train, y_train, X_val, y_val, task=self.task, seed=self.seed + i)
            if score > best_score:
                best_score, best_model = score, model
        self.model_ = best_model
        explainer = shap.TreeExplainer(best_model)
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
            self.global_importance_ = np.mean(sv, axis=0)
        else:
            self.global_importance_ = np.mean(np.abs(sv), axis=0)
        return self
PYEOF

# ─── dash/baselines/large_single.py ───
cat > dash/baselines/large_single.py << 'PYEOF'
"""Baseline: Large Single Model — tests sequential residual dependency hypothesis."""
import numpy as np
import xgboost as xgb
import shap

class LargeSingleModelBaseline:
    def __init__(self, K=20, T_per_model=500, colsample_bytree=0.2, task="regression", seed=42):
        self.K, self.T_per_model, self.colsample_bytree = K, T_per_model, colsample_bytree
        self.task, self.seed = task, seed
        self.model_ = self.global_importance_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None: X_ref = X_val
        total_trees = self.K * self.T_per_model
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(n_estimators=total_trees, colsample_bytree=self.colsample_bytree, max_depth=6, learning_rate=0.1, early_stopping_rounds=50, eval_metric="rmse", random_state=self.seed, verbosity=0)
        else:
            self.model_ = xgb.XGBClassifier(n_estimators=total_trees, colsample_bytree=self.colsample_bytree, max_depth=6, learning_rate=0.1, early_stopping_rounds=50, eval_metric="auc", use_label_encoder=False, random_state=self.seed, verbosity=0)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        bg = X_ref[:min(100, len(X_ref))]
        explainer = shap.TreeExplainer(self.model_, data=bg, feature_perturbation="interventional")
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
            self.global_importance_ = np.mean(sv, axis=0)
        else:
            self.global_importance_ = np.mean(np.abs(sv), axis=0)
        return self
PYEOF

# ─── dash/baselines/ensemble_shap.py ───
cat > dash/baselines/ensemble_shap.py << 'PYEOF'
"""Baseline: Ensemble SHAP — single large ensemble with standard colsample."""
import numpy as np
import xgboost as xgb
import shap

class EnsembleSHAPBaseline:
    def __init__(self, n_estimators=2000, task="regression", seed=42):
        self.n_estimators, self.task, self.seed = n_estimators, task, seed
        self.model_ = self.global_importance_ = self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None):
        if X_ref is None: X_ref = X_val
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(n_estimators=self.n_estimators, max_depth=6, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, early_stopping_rounds=50, eval_metric="rmse", random_state=self.seed, verbosity=0)
        else:
            self.model_ = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=6, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, early_stopping_rounds=50, eval_metric="auc", use_label_encoder=False, random_state=self.seed, verbosity=0)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        explainer = shap.TreeExplainer(self.model_)
        sv = explainer.shap_values(X_ref)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
            self.global_importance_ = np.mean(sv, axis=0)
        else:
            self.global_importance_ = np.mean(np.abs(sv), axis=0)
        self.fsi_ = np.zeros_like(self.global_importance_)
        return self
PYEOF

# ─── dash/baselines/naive_averaging.py ───
cat > dash/baselines/naive_averaging.py << 'PYEOF'
"""Baseline: Naive Top-N Averaging (no diversity selection)."""
import numpy as np
from dash.core.consensus import compute_consensus, compute_diagnostics

class NaiveAveragingBaseline:
    def __init__(self, N=20, task="regression"):
        self.N, self.task = N, task
        self.global_importance_ = self.fsi_ = None

    def fit_from_population(self, models, val_scores, X_ref):
        sorted_idx = sorted(val_scores.keys(), key=lambda i: val_scores[i], reverse=True)
        top_n = sorted_idx[:self.N]
        consensus, all_shap = compute_consensus(models, top_n, X_ref, verbose=False)
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self
PYEOF

# ─── dash/baselines/stochastic_retrain.py ───
cat > dash/baselines/stochastic_retrain.py << 'PYEOF'
"""Baseline: Stochastic Retrain Averaging."""
import numpy as np
from joblib import Parallel, delayed
from dash.core.population import _train_single_model, DEFAULT_SEARCH_SPACE, _sample_configurations
from dash.core.consensus import compute_consensus, compute_diagnostics

class StochasticRetrainBaseline:
    def __init__(self, N=20, task="regression", n_jobs=-1, seed=42):
        self.N, self.task, self.n_jobs, self.seed = N, task, n_jobs, seed
        self.global_importance_ = self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None, best_config=None):
        if X_ref is None: X_ref = X_val
        if best_config is None:
            configs = _sample_configurations(DEFAULT_SEARCH_SPACE, 100, seed=self.seed)
            best_score, best_config = -np.inf, configs[0]
            for i, config in enumerate(configs):
                _, score = _train_single_model(config, X_train, y_train, X_val, y_val, task=self.task, seed=self.seed + i)
                if score > best_score:
                    best_score, best_config = score, config
        def _train(i):
            model, score = _train_single_model(best_config, X_train, y_train, X_val, y_val, task=self.task, seed=self.seed + 1000 + i)
            return i, model, score
        results = Parallel(n_jobs=self.n_jobs)(delayed(_train)(i) for i in range(self.N))
        models = {i: model for i, model, _ in results}
        consensus, all_shap = compute_consensus(models, list(models.keys()), X_ref, verbose=False)
        _, _, self.fsi_, self.global_importance_ = compute_diagnostics(all_shap)
        return self
PYEOF

# ─── dash/experiments/__init__.py ───
cat > dash/experiments/__init__.py << 'EOF'
from dash.experiments.synthetic import generate_synthetic_linear, generate_synthetic_nonlinear
EOF

# ─── dash/experiments/synthetic.py ───
cat > dash/experiments/synthetic.py << 'PYEOF'
"""Synthetic Data Generation — Linear and Nonlinear DGPs."""
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

def make_correlation_matrix(P=50, group_size=5, rho=0.9, structure="block"):
    if rho == 0.0: return np.eye(P)
    n_groups = P // group_size
    Sigma = np.eye(P)
    if structure == "block":
        for g in range(n_groups):
            s, e = g * group_size, (g + 1) * group_size
            for i in range(s, e):
                for j in range(s, e):
                    if i != j: Sigma[i, j] = rho
    elif structure == "overlapping":
        overlap = 2
        for g in range(n_groups):
            s = g * group_size
            e = min(s + group_size + overlap, P)
            for i in range(s, e):
                for j in range(s, e):
                    if i != j:
                        core_end = s + group_size
                        if i >= core_end or j >= core_end:
                            Sigma[i, j] = max(Sigma[i, j], rho * 0.7)
                        else:
                            Sigma[i, j] = max(Sigma[i, j], rho)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(Sigma))
        Sigma = Sigma / np.outer(d, d)
    return Sigma

def generate_synthetic_linear(N=5000, P=50, group_size=5, rho=0.9, sigma_noise=0.5, seed=42, test_size=0.15, val_size=0.15, structure="block"):
    rng = np.random.RandomState(seed)
    n_groups = P // group_size
    beta_groups = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0])
    if n_groups != len(beta_groups): beta_groups = np.linspace(2.0, 0.0, n_groups)
    Sigma = make_correlation_matrix(P, group_size, rho, structure=structure)
    X = rng.multivariate_normal(np.zeros(P), Sigma, size=N)
    group_means = np.zeros((N, n_groups))
    groups = np.zeros(P, dtype=int)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        group_means[:, g] = X[:, s:e].mean(axis=1)
        groups[s:e] = g
    y = group_means @ beta_groups + rng.normal(0, sigma_noise, N)
    true_importance = np.zeros(P)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        true_importance[s:e] = np.abs(beta_groups[g]) / group_size
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_size/(1-test_size), random_state=seed)
    meta = {"dgp": "linear", "N": N, "P": P, "group_size": group_size, "n_groups": n_groups, "rho": rho, "sigma_noise": sigma_noise, "beta_groups": beta_groups, "seed": seed, "structure": structure}
    return X_train, y_train, X_val, y_val, X_test, y_test, groups, true_importance, meta

def generate_synthetic_nonlinear(N=5000, P=50, group_size=5, rho=0.9, sigma_noise=0.5, seed=42, test_size=0.15, val_size=0.15, structure="block"):
    rng = np.random.RandomState(seed)
    n_groups = P // group_size
    beta_1, beta_2, beta_3 = 1.0, 0.8, 1.2
    beta_4_to_G = np.random.RandomState(42).uniform(0.3, 1.0, max(n_groups - 3, 0))
    Sigma = make_correlation_matrix(P, group_size, rho, structure=structure)
    X = rng.multivariate_normal(np.zeros(P), Sigma, size=N)
    group_means = np.zeros((N, n_groups))
    groups = np.zeros(P, dtype=int)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        group_means[:, g] = X[:, s:e].mean(axis=1)
        groups[s:e] = g
    z1, z2, z3 = group_means[:, 0], group_means[:, 1], group_means[:, 2]
    y = beta_1 * z1**2 + beta_2 * z1 * z2 + beta_3 * np.sin(np.pi * z3)
    for g_idx, bg in enumerate(beta_4_to_G):
        y += bg * group_means[:, g_idx + 3]
    y += rng.normal(0, sigma_noise, N)
    true_importance = np.zeros(P)
    approx = np.zeros(n_groups)
    approx[0], approx[1], approx[2] = 1.5, 0.8, 1.2
    for g_idx, bg in enumerate(beta_4_to_G): approx[g_idx + 3] = bg
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        true_importance[s:e] = approx[g] / group_size
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_size/(1-test_size), random_state=seed)
    meta = {"dgp": "nonlinear", "N": N, "P": P, "group_size": group_size, "n_groups": n_groups, "rho": rho, "sigma_noise": sigma_noise, "seed": seed, "structure": structure}
    return X_train, y_train, X_val, y_val, X_test, y_test, groups, true_importance, meta
PYEOF

# ─── dash/evaluation/__init__.py ───
cat > dash/evaluation/__init__.py << 'PYEOF'
"""Evaluation metrics."""
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from typing import List, Tuple

def importance_accuracy(estimated, true):
    rho, _ = spearmanr(estimated, true)
    est_norm = estimated / (estimated.sum() + 1e-10)
    true_norm = true / (true.sum() + 1e-10)
    mse = np.mean((est_norm - true_norm) ** 2)
    return rho, mse

def importance_stability(vectors):
    n = len(vectors)
    if n < 2: return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(vectors[i], vectors[j])
            corrs.append(rho)
    return float(np.mean(corrs))

def within_group_equity(importance_vector, groups):
    cvs = []
    for g in np.unique(groups):
        gi = importance_vector[groups == g]
        cvs.append(gi.std() / gi.mean() if gi.mean() > 1e-10 else 0.0)
    return float(np.mean(cvs))

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-10 else 0.0

def compare_methods(a, b):
    if np.allclose(a, b): return 0.0, 1.0
    stat, pval = wilcoxon(a, b)
    return float(stat), float(pval)

def friedman_test(*method_scores):
    from scipy.stats import friedmanchisquare
    stat, pval = friedmanchisquare(*method_scores)
    return float(stat), float(pval)
PYEOF

# ─── dash/utils/__init__.py ───
cat > dash/utils/__init__.py << 'EOF'
# Utility modules
EOF

echo ""
echo "✅ DASH package created successfully."
echo ""
echo "Files created:"
find dash -name "*.py" | sort
echo ""
echo "Now run:"
echo "  pip install -e ."
echo "  python -c 'from dash.core.pipeline import DASHPipeline; print(\"OK\")'"
