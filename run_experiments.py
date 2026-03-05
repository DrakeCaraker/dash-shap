#!/usr/bin/env python3
"""
DASH Experimental Validation — Complete Runner
===============================================
Run with:  python run_experiments.py

Executes:
  1. Synthetic Linear DGP: rho in {0, 0.5, 0.7, 0.9, 0.95} x 5 reps
  2. Synthetic Linear Overlapping: rho=0.9, overlapping structure
  3. Synthetic Nonlinear DGP: rho=0.9 x 5 reps
  4. Real data: California Housing, Breast Cancer
  5. Stability comparison across all methods
  6. Generates all figures and result tables
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# Add package root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash.core.pipeline import DASHPipeline
from dash.core.consensus import compute_consensus
from dash.core.diagnostics import compute_diagnostics
from dash.core.diversity import (
    get_preliminary_importance,
    cluster_coverage_selection,
    deduplication_selection,
)
from dash.core.diagnostics import (
    FeatureStabilityIndex, ImportanceStabilityPlot, local_disagreement_map,
)
from dash.experiments.synthetic import (
    generate_synthetic_linear, generate_synthetic_nonlinear,
)
from dash.baselines import (
    SingleBestBaseline, LargeSingleModelBaseline, NaiveAveragingBaseline,
    StochasticRetrainBaseline, EnsembleSHAPBaseline,
)
from dash.evaluation import (
    importance_accuracy, importance_stability, within_group_equity,
    compare_methods, friedman_test,
)
from dash.utils.io import save_json

SEED = 42
OUT = "results"
os.makedirs(f"{OUT}/figures", exist_ok=True)
os.makedirs(f"{OUT}/tables", exist_ok=True)

# Pipeline params (reduce M for faster runs; paper uses M=200, K=20)
M = 100
K = 15
N_REPS = 5
EPSILON = 0.03
DELTA = 0.05
N_TRIALS_SINGLE = 50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


###############################################################################
# EXPERIMENT HELPERS
###############################################################################

def run_all_methods(
    X_train, y_train, X_val, y_val, X_ref, groups, true_imp,
    task='regression', seed=42, feature_names=None,
):
    """Run all 7 methods on a single dataset split. Returns dict of importance vectors."""
    results = {}

    # --- DASH MaxMin ---
    dm = DASHPipeline(
        M=M, K=K, epsilon=EPSILON, delta=DELTA,
        selection_method='maxmin', task=task, n_jobs=-1, seed=seed, verbose=False,
    )
    dm.fit(X_train, y_train, X_val, y_val, X_ref=X_ref, feature_names=feature_names)
    results['DASH (MaxMin)'] = {
        'importance': dm.global_importance_,
        'fsi': dm.fsi_,
        'pipeline': dm,
    }

    # --- DASH Cluster (reuse population) ---
    imp_vecs = get_preliminary_importance(
        dm.models_, dm.filtered_indices_, X_ref, method='gain',
    )
    filt_scores = {i: dm.val_scores_[i] for i in dm.filtered_indices_}

    sel_cluster = cluster_coverage_selection(
        imp_vecs, filt_scores, X_train, tau=0.3, K=K, verbose=False,
    )
    cons_c, shap_c = compute_consensus(dm.models_, sel_cluster, X_ref, verbose=False)
    _, _, fsi_c, imp_c = compute_diagnostics(shap_c)
    results['DASH (Cluster)'] = {'importance': imp_c, 'fsi': fsi_c}

    # --- DASH Dedup (reuse population) ---
    sel_dedup = deduplication_selection(
        imp_vecs, filt_scores, rho_threshold=0.95, verbose=False,
    )
    if len(sel_dedup) > K:
        sel_dedup = sorted(
            sel_dedup, key=lambda i: dm.val_scores_[i], reverse=True,
        )[:K]
    if len(sel_dedup) >= 2:
        cons_d, shap_d = compute_consensus(dm.models_, sel_dedup, X_ref, verbose=False)
        _, _, fsi_d, imp_d = compute_diagnostics(shap_d)
        results['DASH (Dedup)'] = {'importance': imp_d, 'fsi': fsi_d}
    else:
        results['DASH (Dedup)'] = {'importance': dm.global_importance_, 'fsi': dm.fsi_}

    # --- Naive Top-N (reuse population) ---
    naive = NaiveAveragingBaseline(N=K, task=task)
    naive.fit_from_population(dm.models_, dm.val_scores_, X_ref)
    results['Naive Top-N'] = {'importance': naive.global_importance_, 'fsi': naive.fsi_}

    # --- Single Best ---
    sb = SingleBestBaseline(n_trials=N_TRIALS_SINGLE, task=task, seed=seed)
    sb.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)
    results['Single Best'] = {'importance': sb.global_importance_}

    # --- Large Single Model ---
    lsm = LargeSingleModelBaseline(
        K=K, T_per_model=500, colsample_bytree=0.2, task=task, seed=seed,
    )
    lsm.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)
    results['Large Single Model'] = {'importance': lsm.global_importance_}

    # --- Ensemble SHAP ---
    ens = EnsembleSHAPBaseline(n_estimators=2000, task=task, seed=seed)
    ens.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)
    results['Ensemble SHAP'] = {'importance': ens.global_importance_}

    # --- Stochastic Retrain ---
    sr = StochasticRetrainBaseline(N=K, task=task, n_jobs=-1, seed=seed)
    sr.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)
    results['Stochastic Retrain'] = {'importance': sr.global_importance_, 'fsi': sr.fsi_}

    return results


def evaluate_results(results, true_imp, groups):
    """Compute accuracy, equity for each method."""
    metrics = {}
    for name, res in results.items():
        imp = res['importance']
        rho_acc, mse = importance_accuracy(imp, true_imp)
        eq = within_group_equity(imp, groups)
        metrics[name] = {'accuracy': rho_acc, 'mse': mse, 'equity': eq}
    return metrics


###############################################################################
# EXPERIMENT 1: Synthetic Linear — Correlation Sweep
###############################################################################

def experiment_synthetic_linear():
    log("=" * 70)
    log("EXPERIMENT 1: Synthetic Linear DGP — Correlation Sweep")
    log("=" * 70)

    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    method_names = [
        'Single Best', 'Large Single Model', 'Ensemble SHAP', 'Naive Top-N',
        'Stochastic Retrain', 'DASH (Dedup)', 'DASH (MaxMin)', 'DASH (Cluster)',
    ]
    feature_names = [f'G{g}_f{j}' for g in range(10) for j in range(5)]

    all_results = {}

    for rho in rho_levels:
        log(f"\n--- rho = {rho} ---")
        stability_vectors = {n: [] for n in method_names}
        accuracy_vectors = {n: [] for n in method_names}
        equity_vectors = {n: [] for n in method_names}

        for rep in range(N_REPS):
            rep_seed = SEED + rep
            log(f"  Rep {rep+1}/{N_REPS} (seed={rep_seed})")

            Xtr, ytr, Xv, yv, Xte, yte, grps, true_imp, meta = \
                generate_synthetic_linear(N=5000, rho=rho, seed=rep_seed)

            results = run_all_methods(
                Xtr, ytr, Xv, yv, X_ref=Xv, groups=grps,
                true_imp=true_imp, seed=rep_seed, feature_names=feature_names,
            )
            metrics = evaluate_results(results, true_imp, grps)

            for name in method_names:
                stability_vectors[name].append(results[name]['importance'])
                accuracy_vectors[name].append(metrics[name]['accuracy'])
                equity_vectors[name].append(metrics[name]['equity'])

        rho_results = {}
        for name in method_names:
            stab = importance_stability(stability_vectors[name])
            acc_mean = np.mean(accuracy_vectors[name])
            acc_std = np.std(accuracy_vectors[name])
            eq_mean = np.mean(equity_vectors[name])
            eq_std = np.std(equity_vectors[name])
            rho_results[name] = {
                'stability': stab,
                'accuracy_mean': acc_mean, 'accuracy_std': acc_std,
                'equity_mean': eq_mean, 'equity_std': eq_std,
            }
            log(f"    {name:<20} stab={stab:.4f}  acc={acc_mean:.4f}+/-{acc_std:.4f}  eq={eq_mean:.4f}")

        all_results[rho] = rho_results

    save_json(all_results, f"{OUT}/tables/synthetic_linear_sweep.json")
    log(f"  Saved: {OUT}/tables/synthetic_linear_sweep.json")
    plot_correlation_sweep(all_results, rho_levels, method_names)

    return all_results


###############################################################################
# EXPERIMENT 2: Overlapping Correlation Structure
###############################################################################

def experiment_overlapping():
    log("\n" + "=" * 70)
    log("EXPERIMENT 2: Overlapping Correlation Structure")
    log("=" * 70)

    feature_names = [f'G{g}_f{j}' for g in range(10) for j in range(5)]
    method_names = ['Single Best', 'DASH (MaxMin)', 'DASH (Cluster)']
    stability_vectors = {n: [] for n in method_names}

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xte, yte, grps, true_imp, meta = \
            generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed, structure="overlapping")

        sb = SingleBestBaseline(n_trials=N_TRIALS_SINGLE, seed=rep_seed)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
        stability_vectors['Single Best'].append(sb.global_importance_)

        dm = DASHPipeline(
            M=M, K=K, epsilon=EPSILON, delta=DELTA,
            selection_method='maxmin', n_jobs=-1, seed=rep_seed, verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=feature_names)
        stability_vectors['DASH (MaxMin)'].append(dm.global_importance_)

        imp_vecs = get_preliminary_importance(
            dm.models_, dm.filtered_indices_, Xv, method='gain',
        )
        filt_scores = {i: dm.val_scores_[i] for i in dm.filtered_indices_}
        sel_c = cluster_coverage_selection(
            imp_vecs, filt_scores, Xtr, tau=0.3, K=K, verbose=False,
        )
        cons_c, shap_c = compute_consensus(dm.models_, sel_c, Xv, verbose=False)
        _, _, _, imp_c = compute_diagnostics(shap_c)
        stability_vectors['DASH (Cluster)'].append(imp_c)

    log("\n  Overlapping structure results:")
    for name in method_names:
        stab = importance_stability(stability_vectors[name])
        log(f"    {name:<20} stability={stab:.4f}")


###############################################################################
# EXPERIMENT 3: Synthetic Nonlinear
###############################################################################

def experiment_nonlinear():
    log("\n" + "=" * 70)
    log("EXPERIMENT 3: Synthetic Nonlinear DGP (rho=0.9)")
    log("  (Evaluating stability & equity, not accuracy)")
    log("=" * 70)

    feature_names = [f'G{g}_f{j}' for g in range(10) for j in range(5)]
    method_names = [
        'Single Best', 'Large Single Model', 'Ensemble SHAP',
        'DASH (MaxMin)', 'DASH (Cluster)',
    ]
    stability_vectors = {n: [] for n in method_names}
    equity_vectors = {n: [] for n in method_names}

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{N_REPS}")

        Xtr, ytr, Xv, yv, _, _, grps, true_imp, _ = \
            generate_synthetic_nonlinear(N=5000, rho=0.9, seed=rep_seed)

        results = run_all_methods(
            Xtr, ytr, Xv, yv, X_ref=Xv, groups=grps,
            true_imp=true_imp, seed=rep_seed, feature_names=feature_names,
        )

        for name in method_names:
            stability_vectors[name].append(results[name]['importance'])
            equity_vectors[name].append(
                within_group_equity(results[name]['importance'], grps),
            )

    log("\n  Nonlinear DGP results:")
    for name in method_names:
        stab = importance_stability(stability_vectors[name])
        eq = np.mean(equity_vectors[name])
        log(f"    {name:<20} stability={stab:.4f}  equity={eq:.4f}")


###############################################################################
# EXPERIMENT 4: Real Data
###############################################################################

def experiment_real_data():
    log("\n" + "=" * 70)
    log("EXPERIMENT 4: Real Data — California Housing & Breast Cancer")
    log("=" * 70)

    from sklearn.datasets import fetch_california_housing, load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    # --- California Housing ---
    log("\n  California Housing:")
    cal = fetch_california_housing()
    X, y = cal.data, cal.target
    cal_names = list(cal.feature_names)
    stability_vectors = {
        n: [] for n in ['Single Best', 'Large Single Model', 'Ensemble SHAP', 'DASH (MaxMin)']
    }

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=rep_seed)
        Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=rep_seed)
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xv = scaler.transform(Xtr), scaler.transform(Xv)

        sb = SingleBestBaseline(n_trials=N_TRIALS_SINGLE, seed=rep_seed)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
        stability_vectors['Single Best'].append(sb.global_importance_)

        lsm = LargeSingleModelBaseline(K=K, T_per_model=500, seed=rep_seed)
        lsm.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
        stability_vectors['Large Single Model'].append(lsm.global_importance_)

        ens = EnsembleSHAPBaseline(n_estimators=2000, seed=rep_seed)
        ens.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
        stability_vectors['Ensemble SHAP'].append(ens.global_importance_)

        dm = DASHPipeline(
            M=M, K=K, epsilon=EPSILON, delta=DELTA,
            selection_method='maxmin', n_jobs=-1, seed=rep_seed, verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=cal_names)
        stability_vectors['DASH (MaxMin)'].append(dm.global_importance_)

    for name in stability_vectors:
        stab = importance_stability(stability_vectors[name])
        log(f"    {name:<20} stability={stab:.4f}")

    fig = dm.plot_importance_stability(title='IS Plot — California Housing', annotate_top_k=8)
    fig.savefig(f"{OUT}/figures/is_plot_california.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log("    Saved IS Plot: figures/is_plot_california.png")

    # --- Breast Cancer ---
    log("\n  Breast Cancer:")
    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    bc_names = list(bc.feature_names)

    corr = np.abs(np.corrcoef(X_bc.T))
    n_high = (np.sum(corr > 0.9) - len(bc_names)) // 2
    log(f"    {len(bc_names)} features, {n_high} pairs with |r|>0.9")

    stability_vectors_bc = {n: [] for n in ['Single Best', 'DASH (MaxMin)']}

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        Xtr, Xte, ytr, yte = train_test_split(X_bc, y_bc, test_size=0.2, random_state=rep_seed)
        Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=rep_seed)
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xv = scaler.transform(Xtr), scaler.transform(Xv)

        sb = SingleBestBaseline(n_trials=N_TRIALS_SINGLE, task='binary', seed=rep_seed)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
        stability_vectors_bc['Single Best'].append(sb.global_importance_)

        dm = DASHPipeline(
            M=M, K=K, epsilon=0.02, delta=DELTA,
            selection_method='maxmin', task='binary',
            n_jobs=-1, seed=rep_seed, verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=bc_names)
        stability_vectors_bc['DASH (MaxMin)'].append(dm.global_importance_)

    for name in stability_vectors_bc:
        stab = importance_stability(stability_vectors_bc[name])
        log(f"    {name:<20} stability={stab:.4f}")

    fig = dm.plot_importance_stability(title='IS Plot — Breast Cancer', annotate_top_k=8)
    fig.savefig(f"{OUT}/figures/is_plot_breast_cancer.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    var_obs = np.mean(dm.variance_matrix_, axis=1)
    fig = local_disagreement_map(
        dm.all_shap_matrices_, np.argmax(var_obs),
        feature_names=bc_names, top_k=12,
    )
    fig.savefig(f"{OUT}/figures/disagreement_breast_cancer.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log("    Saved figures: is_plot_breast_cancer.png, disagreement_breast_cancer.png")


###############################################################################
# PLOTTING
###############################################################################

def plot_correlation_sweep(all_results, rho_levels, method_names):
    """Generate main result figures from the correlation sweep."""
    plot_methods = [
        'Single Best', 'Large Single Model', 'Ensemble SHAP', 'Naive Top-N',
        'Stochastic Retrain', 'DASH (MaxMin)', 'DASH (Cluster)',
    ]
    colors = {
        'Single Best': '#95a5a6',
        'Large Single Model': '#e74c3c',
        'Ensemble SHAP': '#9b59b6',
        'Naive Top-N': '#f39c12',
        'Stochastic Retrain': '#e67e22',
        'DASH (Dedup)': '#3498db',
        'DASH (MaxMin)': '#2ecc71',
        'DASH (Cluster)': '#1abc9c',
    }
    markers = {
        'Single Best': 's',
        'Large Single Model': 'X',
        'Ensemble SHAP': 'D',
        'Naive Top-N': '^',
        'Stochastic Retrain': 'v',
        'DASH (MaxMin)': 'o',
        'DASH (Cluster)': 'P',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for name in plot_methods:
        if name not in method_names:
            continue
        c = colors.get(name, '#333')
        m = markers.get(name, 'o')

        vals = [all_results[rho][name]['stability'] for rho in rho_levels]
        axes[0].plot(rho_levels, vals, f'{m}-', color=c, label=name, linewidth=2, markersize=7)

        vals = [all_results[rho][name]['accuracy_mean'] for rho in rho_levels]
        errs = [all_results[rho][name]['accuracy_std'] for rho in rho_levels]
        axes[1].errorbar(
            rho_levels, vals, yerr=errs, fmt=f'{m}-', color=c,
            label=name, linewidth=2, markersize=7, capsize=3,
        )

        vals = [all_results[rho][name]['equity_mean'] for rho in rho_levels]
        errs = [all_results[rho][name]['equity_std'] for rho in rho_levels]
        axes[2].errorbar(
            rho_levels, vals, yerr=errs, fmt=f'{m}-', color=c,
            label=name, linewidth=2, markersize=7, capsize=3,
        )

    axes[0].set_xlabel('Within-Group Correlation rho')
    axes[0].set_ylabel('Importance Stability\n(Mean Pairwise Spearman)')
    axes[0].set_title('Stability vs. Collinearity')
    axes[0].legend(fontsize=7, loc='lower left')

    axes[1].set_xlabel('Within-Group Correlation rho')
    axes[1].set_ylabel('Spearman rho vs Ground Truth')
    axes[1].set_title('Accuracy vs. Collinearity')

    axes[2].set_xlabel('Within-Group Correlation rho')
    axes[2].set_ylabel('Mean Within-Group CV\n(lower = better)')
    axes[2].set_title('Within-Group Equity vs. Collinearity')

    fig.suptitle('DASH vs Baselines — Synthetic Linear DGP', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/correlation_sweep.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/correlation_sweep.png")

    # Bar chart for rho=0.9
    rho_key = 0.9
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bar_methods = list(all_results[rho_key].keys())
    bar_colors = [colors.get(n, '#333') for n in bar_methods]

    stab_vals = [all_results[rho_key][n]['stability'] for n in bar_methods]
    axes[0].bar(range(len(bar_methods)), stab_vals, color=bar_colors, edgecolor='k', linewidth=0.5)
    axes[0].set_xticks(range(len(bar_methods)))
    axes[0].set_xticklabels(bar_methods, rotation=35, ha='right', fontsize=8)
    axes[0].set_ylabel('Stability')
    axes[0].set_title('Importance Stability (rho=0.9)')

    acc_vals = [all_results[rho_key][n]['accuracy_mean'] for n in bar_methods]
    acc_errs = [all_results[rho_key][n]['accuracy_std'] for n in bar_methods]
    axes[1].bar(
        range(len(bar_methods)), acc_vals, yerr=acc_errs,
        color=bar_colors, edgecolor='k', linewidth=0.5, capsize=3,
    )
    axes[1].set_xticks(range(len(bar_methods)))
    axes[1].set_xticklabels(bar_methods, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Accuracy (Spearman rho)')
    axes[1].set_title('Importance Accuracy (rho=0.9)')

    eq_vals = [all_results[rho_key][n]['equity_mean'] for n in bar_methods]
    eq_errs = [all_results[rho_key][n]['equity_std'] for n in bar_methods]
    axes[2].bar(
        range(len(bar_methods)), eq_vals, yerr=eq_errs,
        color=bar_colors, edgecolor='k', linewidth=0.5, capsize=3,
    )
    axes[2].set_xticks(range(len(bar_methods)))
    axes[2].set_xticklabels(bar_methods, rotation=35, ha='right', fontsize=8)
    axes[2].set_ylabel('Within-Group CV')
    axes[2].set_title('Equity (rho=0.9, lower=better)')

    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/bar_chart_rho09.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/bar_chart_rho09.png")


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    t_start = time.time()

    log("DASH Experimental Validation")
    log(f"Config: M={M}, K={K}, epsilon={EPSILON}, delta={DELTA}, reps={N_REPS}")
    log("")

    sweep_results = experiment_synthetic_linear()
    experiment_overlapping()
    experiment_nonlinear()
    experiment_real_data()

    # --- Statistical summary ---
    log("\n" + "=" * 70)
    log("STATISTICAL TESTS")
    log("=" * 70)
    rho09 = sweep_results[0.9]
    method_names = list(rho09.keys())
    log(f"  Stability at rho=0.9:")
    for n in method_names:
        log(f"    {n:<20} {rho09[n]['stability']:.4f}")

    best_dash = max(
        rho09['DASH (MaxMin)']['stability'],
        rho09['DASH (Cluster)']['stability'],
    )
    sb_stab = rho09['Single Best']['stability']
    lsm_stab = rho09.get(
        'Large Single Model', rho09.get('Ensemble SHAP', {}),
    ).get('stability', 0)
    log(f"\n  DASH best stability:       {best_dash:.4f}")
    log(f"  Single Best stability:     {sb_stab:.4f}")
    log(f"  Large Single Model stab:   {lsm_stab:.4f}")
    log(f"  DASH improvement over SB:  +{best_dash - sb_stab:.4f}")
    log(f"  DASH improvement over LSM: +{best_dash - lsm_stab:.4f}")

    # --- Success criteria check ---
    log("\n" + "=" * 70)
    log("SUCCESS CRITERIA CHECK")
    log("=" * 70)

    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    n_wins = sum(
        1 for rho in rho_levels
        if sweep_results[rho]['DASH (MaxMin)']['stability']
        > sweep_results[rho]['Single Best']['stability']
    )
    log(f"  1. Stability wins: {n_wins}/{len(rho_levels)} "
        f"({'PASS' if n_wins >= 4 else 'FAIL'}, need >=80%)")

    acc_09 = sweep_results[0.9]['DASH (MaxMin)']['accuracy_mean']
    log(f"  2. Accuracy at rho=0.9: {acc_09:.4f} "
        f"({'PASS' if acc_09 >= 0.90 else 'check'}, target >=0.90)")

    n_eq_wins = sum(
        1 for rho in rho_levels
        if sweep_results[rho]['DASH (MaxMin)']['equity_mean']
        < sweep_results[rho]['Single Best']['equity_mean']
    )
    log(f"  3. Equity wins: {n_eq_wins}/{len(rho_levels)} "
        f"({'PASS' if n_eq_wins == len(rho_levels) else 'check'})")

    rho0_dash = sweep_results[0.0]['DASH (MaxMin)']['accuracy_mean']
    rho0_sb = sweep_results[0.0]['Single Best']['accuracy_mean']
    log(f"  4. rho=0 control: DASH acc={rho0_dash:.4f}, SB acc={rho0_sb:.4f} "
        f"({'PASS' if abs(rho0_dash - rho0_sb) < 0.1 else 'check'})")

    if 'Large Single Model' in sweep_results[0.9]:
        n_lsm_wins = sum(
            1 for rho in rho_levels
            if 'Large Single Model' in sweep_results[rho]
            and sweep_results[rho]['DASH (MaxMin)']['stability']
            > sweep_results[rho]['Large Single Model']['stability']
        )
        log(f"  8. Independence value: DASH > LSM on {n_lsm_wins}/{len(rho_levels)} "
            f"({'PASS' if n_lsm_wins >= int(0.7 * len(rho_levels)) else 'check'}, need >=70%)")
        gaps = {
            rho: (sweep_results[rho]['DASH (MaxMin)']['stability']
                  - sweep_results[rho].get('Large Single Model', {}).get('stability', 0))
            for rho in rho_levels
            if 'Large Single Model' in sweep_results[rho]
        }
        if gaps:
            max_gap_rho = max(gaps, key=gaps.get)
            log(f"     Largest gap at rho={max_gap_rho} ({gaps[max_gap_rho]:.4f})")

    total_time = time.time() - t_start
    log(f"\nTotal runtime: {total_time/60:.1f} minutes")
    log(f"Results saved to {OUT}/")
