#!/usr/bin/env python3
"""
DASH Experimental Validation — Complete Runner
===============================================
Aligned with audited demo_benchmark_7.ipynb (PAPER_CONFIG).

Run all experiments:
    python run_experiments.py

Run specific experiments:
    python run_experiments.py --experiments linear_sweep nonlinear_sweep
    python run_experiments.py --experiments real_california real_breast_cancer
    python run_experiments.py --experiments table2_baselines ablation

Available experiments:
    linear_sweep           Synthetic Linear DGP correlation sweep (rho ∈ {0,0.5,0.7,0.9,0.95})
    overlapping            Overlapping correlation structure (rho=0.9)
    nonlinear_sweep        Nonlinear DGP correlation sweep
    table2_baselines       Extended baselines at rho=0.9 (Ensemble SHAP, Stochastic Retrain, Dedup)
    real_california        California Housing benchmark
    real_breast_cancer     Breast Cancer benchmark
    real_superconductor    Superconductor UCI benchmark
    epsilon_sensitivity    Epsilon sensitivity analysis
    ablation               Ablation studies (M, K, epsilon, delta)
    variance_decomposition Variance decomposition (data vs model variance)
    first_mover_visualization First-mover bias concentration figure
    first_mover_bias       First-mover bias isolation (concentration vs tree count)
    success_criteria       Run linear_sweep then evaluate pass/fail criteria
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as rmse_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash.core.pipeline import DASHPipeline
from dash.core.consensus import compute_consensus
from dash.core.diagnostics import compute_diagnostics
from dash.core.diversity import (
    get_preliminary_importance,
    greedy_maxmin_selection,
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
    StochasticRetrainBaseline, EnsembleSHAPBaseline, RandomSelectionBaseline,
    RandomForestBaseline, PermutationImportanceBaseline,
)
from dash.evaluation import (
    dgp_agreement, importance_accuracy, group_level_accuracy, group_level_mse,
    importance_stability, stability_bootstrap_ci, within_group_equity,
    compare_methods, cohens_d, friedman_test, holm_bonferroni,
    feature_ablation_score, tost_equivalence,
)
from dash.utils.io import save_json

# ---------------------------------------------------------------------------
# Canonical configuration — matches PAPER_CONFIG from audited notebook
# ---------------------------------------------------------------------------
PAPER_CONFIG = {
    'M': 200,
    'K': 30,
    'N_REPS': 50,
    'EPSILON': 0.08,
    'DELTA': 0.05,
    'N_TRIALS_SB': 30,
    'T_PER_MODEL': 500,
    'N_ESTIMATORS_ESHAP': 2000,
    'TAU_CLUSTER': 0.3,
}

SEED = 42
M = PAPER_CONFIG['M']
K = PAPER_CONFIG['K']
N_REPS = PAPER_CONFIG['N_REPS']
EPSILON = PAPER_CONFIG['EPSILON']
DELTA = PAPER_CONFIG['DELTA']
N_TRIALS_SB = PAPER_CONFIG['N_TRIALS_SB']

# F2 fix: Scale-invariant epsilon for real-world datasets via relative mode.
# Replaces the manually-tuned CAL_EPSILON/BC_EPSILON/SC_EPSILON constants.
REAL_EPSILON = 0.05
REAL_EPSILON_MODE = 'relative'

OUT = "results"


def make_feature_names(n_groups=10, group_size=5):
    """Generate feature names matching the DGP structure (m6 fix)."""
    return [f'G{g}_f{j}' for g in range(n_groups) for j in range(group_size)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dirs():
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    os.makedirs(f"{OUT}/tables", exist_ok=True)


###############################################################################
# PLOTTING
###############################################################################

COLORS = {
    'Single Best': '#95a5a6',
    'Single Best (M=200)': '#7f8c8d',
    'Large Single Model': '#e74c3c',
    'LSM (Tuned)': '#c0392b',
    'Ensemble SHAP': '#9b59b6',
    'Naive Top-N': '#f39c12',
    'Stochastic Retrain': '#e67e22',
    'Random Selection': '#d4ac0d',
    'DASH (Dedup)': '#3498db',
    'DASH (MaxMin)': '#2ecc71',
    'DASH (Cluster)': '#1abc9c',
    'Random Forest': '#16a085',
    'Permutation Importance': '#8e44ad',
}

MARKERS = {
    'Single Best': 's',
    'Single Best (M=200)': 'S',
    'Large Single Model': 'X',
    'LSM (Tuned)': 'x',
    'Ensemble SHAP': 'D',
    'Naive Top-N': '^',
    'Stochastic Retrain': 'v',
    'Random Selection': 'd',
    'DASH (MaxMin)': 'o',
    'DASH (Cluster)': 'P',
    'Random Forest': 'H',
    'Permutation Importance': 'p',
}


def plot_correlation_sweep(all_results, rho_levels, method_names):
    """Generate main result figures from the correlation sweep."""
    _ensure_dirs()
    plot_methods = [n for n in MARKERS if n in method_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for name in plot_methods:
        c = COLORS.get(name, '#333')
        m = MARKERS.get(name, 'o')

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

    axes[0].set_xlabel('Within-Group Correlation ρ')
    axes[0].set_ylabel('Importance Stability\n(Mean Pairwise Spearman)')
    axes[0].set_title('Stability vs. Collinearity')
    axes[0].legend(fontsize=7, loc='lower left')

    axes[1].set_xlabel('Within-Group Correlation ρ')
    axes[1].set_ylabel('Spearman ρ vs Ground Truth')
    axes[1].set_title('Accuracy vs. Collinearity')

    axes[2].set_xlabel('Within-Group Correlation ρ')
    axes[2].set_ylabel('Mean Within-Group CV\n(lower = better)')
    axes[2].set_title('Within-Group Equity vs. Collinearity')

    fig.suptitle('DASH vs Baselines — Synthetic Linear DGP', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/correlation_sweep.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{OUT}/figures/correlation_sweep.pdf", bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/correlation_sweep.png, correlation_sweep.pdf")

    # Bar chart for rho=0.9
    if 0.9 in all_results:
        rho_key = 0.9
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        bar_methods = list(all_results[rho_key].keys())
        bar_colors = [COLORS.get(n, '#333') for n in bar_methods]

        stab_vals = [all_results[rho_key][n]['stability'] for n in bar_methods]
        axes[0].bar(range(len(bar_methods)), stab_vals, color=bar_colors, edgecolor='k', linewidth=0.5)
        axes[0].set_xticks(range(len(bar_methods)))
        axes[0].set_xticklabels(bar_methods, rotation=35, ha='right', fontsize=8)
        axes[0].set_ylabel('Stability')
        axes[0].set_title('Importance Stability (ρ=0.9)')

        acc_vals = [all_results[rho_key][n]['accuracy_mean'] for n in bar_methods]
        acc_errs = [all_results[rho_key][n]['accuracy_std'] for n in bar_methods]
        axes[1].bar(
            range(len(bar_methods)), acc_vals, yerr=acc_errs,
            color=bar_colors, edgecolor='k', linewidth=0.5, capsize=3,
        )
        axes[1].set_xticks(range(len(bar_methods)))
        axes[1].set_xticklabels(bar_methods, rotation=35, ha='right', fontsize=8)
        axes[1].set_ylabel('Accuracy (Spearman ρ)')
        axes[1].set_title('Importance Accuracy (ρ=0.9)')

        eq_vals = [all_results[rho_key][n]['equity_mean'] for n in bar_methods]
        eq_errs = [all_results[rho_key][n]['equity_std'] for n in bar_methods]
        axes[2].bar(
            range(len(bar_methods)), eq_vals, yerr=eq_errs,
            color=bar_colors, edgecolor='k', linewidth=0.5, capsize=3,
        )
        axes[2].set_xticks(range(len(bar_methods)))
        axes[2].set_xticklabels(bar_methods, rotation=35, ha='right', fontsize=8)
        axes[2].set_ylabel('Within-Group CV')
        axes[2].set_title('Equity (ρ=0.9, lower=better)')

        fig.tight_layout()
        fig.savefig(f"{OUT}/figures/bar_chart_rho09.png", dpi=150, bbox_inches='tight')
        fig.savefig(f"{OUT}/figures/bar_chart_rho09.pdf", bbox_inches='tight')
        plt.close(fig)
        log(f"  Saved: figures/bar_chart_rho09.png, bar_chart_rho09.pdf")


###############################################################################
# PLOT: Ablation sensitivity
###############################################################################

def plot_ablation_sensitivity(ablation_results):
    """Generate 2x2 ablation sensitivity figure (M, K, epsilon, delta)."""
    _ensure_dirs()

    param_configs = [
        ('M', [50, 100, 200, 500], 'Population Size $M$'),
        ('K', [5, 10, 20, 30, 50], 'Selected Models $K$'),
        ('epsilon', [0.01, 0.03, 0.05, 0.08, 0.10], 'Filter Threshold $\\varepsilon$'),
        ('delta', [0.01, 0.05, 0.10, 0.20], 'Diversity Threshold $\\delta$'),
    ]
    rho_styles = {
        0.0:  ('--', '#7f8c8d', 'o', '$\\rho=0.0$'),
        0.9:  ('-',  '#2980b9', 's', '$\\rho=0.9$'),
        0.95: ('-',  '#e74c3c', '^', '$\\rho=0.95$'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax_idx, (param_name, param_vals, xlabel) in enumerate(param_configs):
        ax = axes[ax_idx]
        for rho, (ls, color, marker, label) in rho_styles.items():
            rho_key = str(rho)
            if rho_key not in ablation_results:
                continue
            param_data = ablation_results[rho_key].get(param_name, {})
            x_vals, y_vals = [], []
            for v in param_vals:
                v_key = str(v)
                if v_key in param_data:
                    x_vals.append(v)
                    y_vals.append(param_data[v_key].get('stability', 0))
            if x_vals:
                ax.plot(x_vals, y_vals, f'{marker}{ls}', color=color,
                        label=label, linewidth=2, markersize=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Stability')
        ax.set_title(f'{param_name} Sensitivity')
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('DASH Hyperparameter Sensitivity', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/ablation_sensitivity.pdf", bbox_inches='tight')
    fig.savefig(f"{OUT}/figures/ablation_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/ablation_sensitivity.pdf")


###############################################################################
# Helpers
###############################################################################

def _log_pairwise_significance(results, dash_name, method_names, dataset_label):
    """Log Wilcoxon signed-rank tests and Cohen's d for DASH vs each baseline.

    Requires that each method in *results* stores 'rmse_runs' and optionally
    'ablation_runs' as numpy arrays of per-rep values.
    """
    baselines = [n for n in method_names if n != dash_name]
    if not baselines or dash_name not in results:
        return
    dash = results[dash_name]
    log(f"\n  Significance tests ({dataset_label}): {dash_name} vs baselines")
    log(f"  {'Baseline':<22} {'Metric':<12} {'Wilcoxon p':>12} {'Cohen d':>10}")
    log("  " + "-" * 60)
    sig_results = {}
    for bname in baselines:
        bl = results[bname]
        sig_results[bname] = {}
        for metric in ('rmse_runs', 'ablation_runs'):
            if metric not in dash or metric not in bl:
                continue
            label = metric.replace('_runs', '')
            _, pval = compare_methods(dash[metric], bl[metric])
            d = cohens_d(dash[metric], bl[metric])
            log(f"  {bname:<22} {label:<12} {pval:>12.4g} {d:>10.3f}")
            sig_results[bname][label] = {'p': pval, 'cohens_d': d}
        # TOST equivalence test for DASH vs Stochastic Retrain
        if bname == 'Stochastic Retrain':
            for metric in ('rmse_runs', 'ablation_runs'):
                if metric not in dash or metric not in bl:
                    continue
                label = metric.replace('_runs', '')
                _, p1, _, p2, equiv = tost_equivalence(
                    dash[metric], bl[metric],
                )
                log(f"  {bname:<22} {label:<12} TOST equiv={'YES' if equiv else 'no'}  "
                    f"p_max={max(p1, p2):.4g}")
                sig_results[bname][f'{label}_tost'] = {
                    'p1': p1, 'p2': p2, 'equivalent': equiv,
                }
    # Apply Holm-Bonferroni correction to all raw Wilcoxon p-values
    raw_keys = []
    raw_pvals = []
    for bname in sig_results:
        for label in sig_results[bname]:
            if '_tost' not in label and 'p' in sig_results[bname][label]:
                raw_keys.append((bname, label))
                raw_pvals.append(sig_results[bname][label]['p'])
    if raw_pvals:
        adjusted = holm_bonferroni(np.array(raw_pvals))
        for (bname, label), adj_p in zip(raw_keys, adjusted):
            sig_results[bname][label]['p_holm'] = float(adj_p)
        log(f"\n  Holm-Bonferroni corrected p-values:")
        for (bname, label), adj_p in zip(raw_keys, adjusted):
            log(f"    {bname:<22} {label:<12} p_HB={adj_p:.4g}")
    # Store in results for JSON serialisation
    results['_significance'] = sig_results


###############################################################################
# EXPERIMENT: Synthetic Linear — Correlation Sweep
###############################################################################

def experiment_linear_sweep():
    """Canonical correlation sweep: rho ∈ {0.0, 0.5, 0.7, 0.9, 0.95}.

    Matches notebook Section 4.  Uses four-way split: X_explain for SHAP
    (X_ref), X_test exclusively for RMSE (A4 fix).  Reports BCa bootstrap
    CI for stability (A3 fix), within_group_equity with include_zero_groups
    robustness check (A2), and dgp_agreement instead of importance_accuracy
    (A5 reframing).

    Note on bootstrap CIs (A1): CIs reflect run-to-run variability and do
    not include uncertainty from the model selection stage.  Since K_eff is
    typically 10-30 out of ~110 filtered candidates, selection-stage variance
    is expected to be small relative to cross-run variance.

    Includes LSM (Tuned) — a tuned variant of the Large Single Model that
    searches over max_depth and learning_rate, making the comparison fair.
    """
    _ensure_dirs()
    t0 = time.time()
    log("=" * 70)
    log("EXPERIMENT: Synthetic Linear DGP — Correlation Sweep")
    log("=" * 70)

    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    sweep_methods = [
        'Single Best', 'Single Best (M=200)',  # M3: matched budget
        'Large Single Model', 'LSM (Tuned)',
        'Stochastic Retrain', 'Random Selection',  # M2: isolate MaxMin value
        'Random Forest',
        'DASH (MaxMin)',
        'Naive Top-N',  # Ablation: averaging without diversity selection (reuses DASH population)
    ]
    sweep_results = {rho: {} for rho in rho_levels}
    feature_names = make_feature_names()  # m6: dynamic

    for rho in rho_levels:
        log(f"\n--- ρ = {rho} ---")
        dash_cache = {}  # Cache DASH pipelines per rep for Naive Top-N reuse
        for name in sweep_methods:
            t_method = time.time()
            acc_runs, eq_runs, imp_runs, rmse_runs, gacc_runs, gmse_runs, keff_runs = \
                [], [], [], [], [], [], []
            for rep in range(N_REPS):
                rep_seed = SEED + rep
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
                    generate_synthetic_linear(N=5000, rho=rho, seed=rep_seed)

                if name == 'Single Best':
                    m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                    rmse_val = rmse_score(yte, preds)
                elif name == 'Single Best (M=200)':
                    m = SingleBestBaseline(n_trials=M, seed=rep_seed)
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                    rmse_val = rmse_score(yte, preds)
                elif name in ('Large Single Model', 'LSM (Tuned)'):
                    m = LargeSingleModelBaseline(
                        K=K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
                        colsample_bytree=0.2, seed=rep_seed,
                        tune=(name == 'LSM (Tuned)'),
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                    rmse_val = rmse_score(yte, preds)
                elif name == 'Stochastic Retrain':
                    m = StochasticRetrainBaseline(
                        N=K, task='regression', n_jobs=-1, seed=rep_seed,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)
                    rmse_val = rmse_score(yte, preds)
                elif name == 'Random Selection':
                    m = RandomSelectionBaseline(
                        M=M, K=K, epsilon=EPSILON, delta=DELTA,
                        n_jobs=-1, seed=rep_seed, verbose=False,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)
                    rmse_val = rmse_score(yte, preds)
                elif name == 'Random Forest':
                    m = RandomForestBaseline(
                        n_estimators=500, task='regression', seed=rep_seed,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                    rmse_val = rmse_score(yte, preds)
                elif name == 'DASH (MaxMin)':
                    m = DASHPipeline(
                        M=M, K=K, epsilon=EPSILON, delta=DELTA,
                        selection_method='maxmin', n_jobs=-1,
                        seed=rep_seed, verbose=False,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)
                    rmse_val = rmse_score(yte, preds)
                    # Cache for Naive Top-N reuse
                    dash_cache[rep] = m
                elif name == 'Naive Top-N':
                    # Reuse DASH population: top-K by val score, no diversity selection
                    cached = dash_cache.get(rep)
                    if cached is None:
                        raise RuntimeError("Naive Top-N requires DASH (MaxMin) to run first")
                    naive = NaiveAveragingBaseline(N=K, task='regression')
                    naive.fit_from_population(
                        cached.models_, cached.val_scores_, Xexp,
                    )
                    imp = naive.global_importance_
                    rmse_val = np.nan  # No ensemble predictions for this baseline
                    m = naive  # Update m so k_eff tracking below uses correct object

                r, _ = importance_accuracy(imp, true_imp)
                acc_runs.append(r)
                gacc_runs.append(group_level_accuracy(imp, true_imp, grps))
                gmse_runs.append(group_level_mse(imp, true_imp, grps))
                eq_runs.append(within_group_equity(imp, grps))
                imp_runs.append(imp)
                rmse_runs.append(rmse_val)
                # F2: Track effective K for ensemble methods
                if hasattr(m, 'selected_indices_') and m.selected_indices_ is not None:
                    keff_runs.append(len(m.selected_indices_))

            # M8 fix: bootstrap CI for stability
            stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
            n_reps = len(acc_runs)
            elapsed_method = time.time() - t_method
            sweep_results[rho][name] = {
                'stability': stab,
                'stability_se': stab_se,
                'stability_ci_lo': stab_ci_lo,
                'stability_ci_hi': stab_ci_hi,
                'k_eff_mean': float(np.mean(keff_runs)) if keff_runs else None,
                'k_eff_std': float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
                'accuracy_mean': np.mean(acc_runs),
                'accuracy_se': np.std(acc_runs, ddof=1) / np.sqrt(n_reps),
                'accuracy_std': np.std(acc_runs, ddof=1),
                'group_accuracy_mean': np.mean(gacc_runs),
                'group_accuracy_std': np.std(gacc_runs, ddof=1),
                'group_mse_mean': np.mean(gmse_runs),
                'group_mse_std': np.std(gmse_runs, ddof=1),
                'equity_mean': np.mean(eq_runs),
                'equity_se': np.std(eq_runs, ddof=1) / np.sqrt(n_reps),
                'equity_std': np.std(eq_runs, ddof=1),
                'rmse_mean': np.mean(rmse_runs),
                'rmse_std': np.std(rmse_runs, ddof=1),
                'elapsed_s': elapsed_method,
                # Save per-rep arrays for significance tests
                'acc_runs': np.array(acc_runs),
                'eq_runs': np.array(eq_runs),
                'rmse_runs': np.array(rmse_runs),
                'imp_runs': imp_runs,
            }
            log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
                f"acc={np.mean(acc_runs):.4f}  gacc={np.mean(gacc_runs):.4f}  gmse={np.mean(gmse_runs):.6f}  "
                f"eq={np.mean(eq_runs):.4f}  RMSE={np.mean(rmse_runs):.4f}  "
                f"({elapsed_method:.1f}s)")

    save_json(sweep_results, f"{OUT}/tables/synthetic_linear_sweep.json")
    log(f"  Saved: {OUT}/tables/synthetic_linear_sweep.json")
    plot_correlation_sweep(sweep_results, rho_levels, sweep_methods)

    elapsed = time.time() - t0
    log(f"  Linear sweep completed in {elapsed/60:.1f} min")
    format_timing_table(sweep_results, rho=0.9)
    return sweep_results


###############################################################################
# EXPERIMENT: Overlapping Correlation Structure
###############################################################################

def experiment_overlapping():
    """Overlapping correlation structure at rho=0.9.

    M7 fix: now reports accuracy, equity, and RMSE alongside stability.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Overlapping Correlation Structure")
    log("=" * 70)

    method_names = ['Single Best', 'DASH (MaxMin)', 'DASH (Cluster)']
    results = {n: {'imp_runs': [], 'acc_runs': [], 'grp_acc_runs': [], 'gmse_runs': [],
                    'eq_runs': [], 'rmse_runs': []} for n in method_names}
    feature_names = make_feature_names()

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = \
            generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed, structure="overlapping")

        sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        results['Single Best']['imp_runs'].append(sb.global_importance_)
        r, _ = dgp_agreement(sb.global_importance_, true_imp)
        results['Single Best']['acc_runs'].append(r)
        results['Single Best']['grp_acc_runs'].append(group_level_accuracy(sb.global_importance_, true_imp, grps))
        results['Single Best']['gmse_runs'].append(group_level_mse(sb.global_importance_, true_imp, grps))
        results['Single Best']['eq_runs'].append(within_group_equity(sb.global_importance_, grps))
        results['Single Best']['rmse_runs'].append(rmse_score(yte, sb.model_.predict(Xte)))

        dm = DASHPipeline(
            M=M, K=K, epsilon=EPSILON, delta=DELTA,
            selection_method='maxmin', n_jobs=-1, seed=rep_seed, verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        results['DASH (MaxMin)']['imp_runs'].append(dm.global_importance_)
        r, _ = dgp_agreement(dm.global_importance_, true_imp)
        results['DASH (MaxMin)']['acc_runs'].append(r)
        results['DASH (MaxMin)']['grp_acc_runs'].append(group_level_accuracy(dm.global_importance_, true_imp, grps))
        results['DASH (MaxMin)']['gmse_runs'].append(group_level_mse(dm.global_importance_, true_imp, grps))
        results['DASH (MaxMin)']['eq_runs'].append(within_group_equity(dm.global_importance_, grps))
        results['DASH (MaxMin)']['rmse_runs'].append(rmse_score(yte, dm.get_consensus_ensemble_predictions(Xte)))

        imp_vecs = get_preliminary_importance(
            dm.models_, dm.filtered_indices_, Xexp, method='gain',
        )
        filt_scores = {i: dm.val_scores_[i] for i in dm.filtered_indices_}
        sel_c = cluster_coverage_selection(
            imp_vecs, filt_scores, Xtr,
            tau=PAPER_CONFIG['TAU_CLUSTER'], K=K, verbose=False,
        )
        cons_c, shap_c = compute_consensus(dm.models_, sel_c, Xexp, verbose=False)
        _, _, _, imp_c = compute_diagnostics(shap_c)
        results['DASH (Cluster)']['imp_runs'].append(imp_c)
        r, _ = dgp_agreement(imp_c, true_imp)
        results['DASH (Cluster)']['acc_runs'].append(r)
        results['DASH (Cluster)']['grp_acc_runs'].append(group_level_accuracy(imp_c, true_imp, grps))
        results['DASH (Cluster)']['gmse_runs'].append(group_level_mse(imp_c, true_imp, grps))
        results['DASH (Cluster)']['eq_runs'].append(within_group_equity(imp_c, grps))
        # Cluster uses same models as DASH, so use DASH predictions for RMSE
        results['DASH (Cluster)']['rmse_runs'].append(rmse_score(yte, dm.get_consensus_ensemble_predictions(Xte)))

    log(f"\n  {'Method':<20} {'Stability':>10} {'DGP Agree':>10} {'Grp Acc':>10} {'Grp MSE':>10} {'Equity':>10} {'RMSE':>10}")
    log("  " + "=" * 85)
    overlap_results = {}
    for name in method_names:
        stab = importance_stability(results[name]['imp_runs'])
        acc = np.mean(results[name]['acc_runs'])
        grp = np.mean(results[name]['grp_acc_runs'])
        gmse = np.mean(results[name]['gmse_runs'])
        eq = np.mean(results[name]['eq_runs'])
        rmse = np.mean(results[name]['rmse_runs'])
        log(f"  {name:<20} {stab:>10.4f} {acc:>10.4f} {grp:>10.4f} {gmse:>10.6f} {eq:>10.4f} {rmse:>10.4f}")
        overlap_results[name] = {
            'stability': stab,
            'accuracy_mean': acc, 'accuracy_std': float(np.std(results[name]['acc_runs'])),
            'group_accuracy_mean': grp,
            'group_mse_mean': gmse, 'group_mse_std': float(np.std(results[name]['gmse_runs'], ddof=1)),
            'equity_mean': eq, 'equity_std': float(np.std(results[name]['eq_runs'])),
            'rmse_mean': rmse, 'rmse_std': float(np.std(results[name]['rmse_runs'])),
        }

    save_json(overlap_results, f"{OUT}/tables/overlapping.json")
    log(f"  Saved: {OUT}/tables/overlapping.json")

    elapsed = time.time() - t0
    log(f"  Overlapping completed in {elapsed/60:.1f} min")
    return overlap_results


###############################################################################
# EXPERIMENT: Nonlinear DGP Correlation Sweep
###############################################################################

def experiment_nonlinear_sweep():
    """Nonlinear DGP sweep: rho ∈ {0.0, 0.5, 0.7, 0.9, 0.95}.

    Evaluates stability and equity (no ground-truth accuracy for nonlinear DGP).
    NOTE: true_importance for the nonlinear DGP is an approximate ordinal
    ranking, not exact analytic SHAP.  See generate_synthetic_nonlinear().
    Includes LSM (Tuned) for fair comparison.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Nonlinear DGP — Correlation Sweep")
    log("=" * 70)

    nl_rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    nl_methods = [
        'Single Best', 'Large Single Model', 'LSM (Tuned)',
        'Stochastic Retrain', 'Random Forest', 'DASH (MaxMin)',
    ]
    nl_sweep = {rho: {} for rho in nl_rho_levels}
    feature_names = make_feature_names()

    for rho in nl_rho_levels:
        log(f"\n--- Nonlinear DGP, ρ = {rho} ---")
        for name in nl_methods:
            eq_runs, imp_runs, rmse_runs, keff_runs = [], [], [], []
            for rep in range(N_REPS):
                rep_seed = SEED + rep
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, _, _ = \
                    generate_synthetic_nonlinear(N=5000, rho=rho, seed=rep_seed)

                if name == 'Single Best':
                    m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                elif name in ('Large Single Model', 'LSM (Tuned)'):
                    m = LargeSingleModelBaseline(
                        K=K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
                        colsample_bytree=0.2, seed=rep_seed,
                        tune=(name == 'LSM (Tuned)'),
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                elif name == 'Stochastic Retrain':
                    m = StochasticRetrainBaseline(
                        N=K, task='regression', n_jobs=-1, seed=rep_seed,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)
                elif name == 'Random Forest':
                    m = RandomForestBaseline(
                        n_estimators=500, task='regression', seed=rep_seed,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                else:  # DASH MaxMin
                    m = DASHPipeline(
                        M=M, K=K, epsilon=EPSILON, delta=DELTA,
                        selection_method='maxmin', n_jobs=-1,
                        seed=rep_seed, verbose=False,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)

                rmse_val = rmse_score(yte, preds)
                eq_runs.append(within_group_equity(imp, grps))
                imp_runs.append(imp)
                rmse_runs.append(rmse_val)
                if hasattr(m, 'selected_indices_') and m.selected_indices_ is not None:
                    keff_runs.append(len(m.selected_indices_))

            stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
            nl_sweep[rho][name] = {
                'stability': stab,
                'stability_se': stab_se,
                'stability_ci_lo': stab_ci_lo,
                'stability_ci_hi': stab_ci_hi,
                'k_eff_mean': float(np.mean(keff_runs)) if keff_runs else None,
                'k_eff_std': float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
                'equity_mean': np.mean(eq_runs), 'equity_std': np.std(eq_runs, ddof=1),
                'eq_runs': np.array(eq_runs),
                'rmse_mean': float(np.mean(rmse_runs)),
                'rmse_std': float(np.std(rmse_runs, ddof=1)),
                'rmse_runs': np.array(rmse_runs),
            }
            log(f"  {name:<20} stab={stab:.4f}±{stab_se:.4f}  eq={np.mean(eq_runs):.4f}  "
                f"RMSE={np.mean(rmse_runs):.4f}")

    save_json(nl_sweep, f"{OUT}/tables/nonlinear_sweep.json")
    log(f"  Saved: {OUT}/tables/nonlinear_sweep.json")

    elapsed = time.time() - t0
    log(f"  Nonlinear sweep completed in {elapsed/60:.1f} min")
    return nl_sweep


###############################################################################
# EXPERIMENT: Extended Baselines (Table 2) at rho=0.9
###############################################################################

def experiment_table2_baselines():
    """Extended baselines at rho=0.9: Ensemble SHAP, Stochastic Retrain, DASH (Dedup)."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Table 2 — Extended Baselines at ρ=0.9")
    log("=" * 70)

    table2_methods = [
        'Ensemble SHAP', 'Stochastic Retrain', 'Random Forest',
        'Permutation Importance', 'DASH (Dedup)',
    ]
    table2_results = {}
    feature_names = make_feature_names()

    for name in table2_methods:
        imp_runs, acc_runs, eq_runs = [], [], []
        for rep in range(N_REPS):
            rep_seed = SEED + rep
            Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
                generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)

            if name == 'Ensemble SHAP':
                m = EnsembleSHAPBaseline(
                    n_estimators=PAPER_CONFIG['N_ESTIMATORS_ESHAP'],
                    task='regression', seed=rep_seed,
                )
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                imp = m.global_importance_
            elif name == 'Stochastic Retrain':
                m = StochasticRetrainBaseline(N=K, task='regression', n_jobs=-1, seed=rep_seed)
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                imp = m.global_importance_
            elif name == 'Random Forest':
                m = RandomForestBaseline(
                    n_estimators=500, task='regression', seed=rep_seed,
                )
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
                imp = m.global_importance_
            elif name == 'Permutation Importance':
                m = PermutationImportanceBaseline(
                    n_trials=N_TRIALS_SB, task='regression', seed=rep_seed,
                )
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, y_ref=yexp)
                imp = m.global_importance_
            else:  # DASH (Dedup)
                dm = DASHPipeline(
                    M=M, K=K, epsilon=EPSILON, delta=DELTA,
                    selection_method='dedup', n_jobs=-1,
                    seed=rep_seed, verbose=False,
                )
                dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
                imp = dm.global_importance_

            r, _ = dgp_agreement(imp, true_imp)
            acc_runs.append(r)
            eq_runs.append(within_group_equity(imp, grps))
            imp_runs.append(imp)

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        table2_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
            'accuracy_mean': np.mean(acc_runs), 'accuracy_std': np.std(acc_runs, ddof=1),
            'equity_mean': np.mean(eq_runs), 'equity_std': np.std(eq_runs, ddof=1),
            'acc_runs': np.array(acc_runs), 'eq_runs': np.array(eq_runs),
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
            f"acc={np.mean(acc_runs):.4f}  eq={np.mean(eq_runs):.4f}")

    save_json(table2_results, f"{OUT}/tables/table2_baselines.json")
    log(f"  Saved: {OUT}/tables/table2_baselines.json")

    elapsed = time.time() - t0
    log(f"  Table 2 baselines completed in {elapsed/60:.1f} min")
    return table2_results


###############################################################################
# EXPERIMENT: California Housing
###############################################################################

def experiment_real_california():
    """California Housing benchmark with scale-appropriate epsilon."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Real Data — California Housing")
    log("=" * 70)

    from sklearn.datasets import fetch_california_housing

    cal = fetch_california_housing()
    X_cal, y_cal = cal.data, cal.target
    cal_names = list(cal.feature_names)
    log(f"  {X_cal.shape[0]} samples, {X_cal.shape[1]} features")

    corr_cal = np.abs(np.corrcoef(X_cal.T))
    n_high_cal = (np.sum(corr_cal > 0.7) - len(cal_names)) // 2
    log(f"  Feature pairs with |r|>0.7: {n_high_cal}")

    X_cal_pool, X_cal_test, y_cal_pool, y_cal_test = train_test_split(
        X_cal, y_cal, test_size=0.2, random_state=SEED,
    )

    cal_methods = ['Single Best', 'Random Forest', 'DASH (MaxMin)']
    cal_results = {}

    for name in cal_methods:
        imp_runs, rmse_runs, ablation_runs, keff_runs = [], [], [], []
        for rep in range(N_REPS):
            rep_seed = SEED + rep

            # D2: Re-split and re-fit scaler per rep
            # A4: Three-way split from pool — explain set separate from test
            Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(
                X_cal_pool, y_cal_pool, test_size=0.2, random_state=rep_seed,
            )
            Xtr_r, Xexp_r, ytr_r, yexp_r = train_test_split(
                Xtr_r, ytr_r, test_size=0.12, random_state=rep_seed,
            )
            scaler_r = StandardScaler().fit(Xtr_r)
            Xtr_r = scaler_r.transform(Xtr_r)
            Xv_r = scaler_r.transform(Xv_r)
            Xexp_r = scaler_r.transform(Xexp_r)
            Xte_r = scaler_r.transform(X_cal_test)

            if name == 'Single Best':
                m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
                imp = m.global_importance_
                rmse_val = rmse_score(y_cal_test, m.model_.predict(Xte_r))
                abl = feature_ablation_score(m.model_, Xte_r, y_cal_test, imp)
            elif name == 'Random Forest':
                m = RandomForestBaseline(
                    n_estimators=500, task='regression', seed=rep_seed,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
                imp = m.global_importance_
                rmse_val = rmse_score(y_cal_test, m.model_.predict(Xte_r))
                abl = feature_ablation_score(m.model_, Xte_r, y_cal_test, imp)
            else:
                m = DASHPipeline(
                    M=M, K=K, epsilon=REAL_EPSILON, delta=DELTA,
                    epsilon_mode=REAL_EPSILON_MODE,
                    selection_method='maxmin', n_jobs=-1,
                    seed=rep_seed, verbose=False,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=cal_names)
                imp = m.global_importance_
                preds = m.get_consensus_ensemble_predictions(Xte_r)
                rmse_val = rmse_score(y_cal_test, preds)
                proxy_model = m.selected_models_[0]
                abl = feature_ablation_score(proxy_model, Xte_r, y_cal_test, imp)

            imp_runs.append(imp)
            rmse_runs.append(rmse_val)
            ablation_runs.append(abl)
            if hasattr(m, 'selected_indices_') and m.selected_indices_ is not None:
                keff_runs.append(len(m.selected_indices_))

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        cal_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
            'k_eff_mean': float(np.mean(keff_runs)) if keff_runs else None,
            'k_eff_std': float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
            'rmse_mean': np.mean(rmse_runs), 'rmse_std': np.std(rmse_runs, ddof=1),
            'ablation_mean': np.mean(ablation_runs), 'ablation_std': np.std(ablation_runs, ddof=1),
            'rmse_runs': np.array(rmse_runs),
            'ablation_runs': np.array(ablation_runs),
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
            f"RMSE={np.mean(rmse_runs):.4f}±{np.std(rmse_runs, ddof=1):.4f}  "
            f"ablation={np.mean(ablation_runs):.4f}")

    # C7+F1: Wilcoxon signed-rank test and Cohen's d between DASH and baselines
    _log_pairwise_significance(cal_results, 'DASH (MaxMin)', cal_methods, 'California Housing')

    # IS plot from last DASH run
    fig = m.plot_importance_stability(title='IS Plot — California Housing', annotate_top_k=8)
    fig.savefig(f"{OUT}/figures/is_plot_california.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{OUT}/figures/is_plot_california.pdf", bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/is_plot_california.png/pdf")

    save_json(cal_results, f"{OUT}/tables/california_housing.json")
    elapsed = time.time() - t0
    log(f"  California Housing completed in {elapsed/60:.1f} min")
    return cal_results


###############################################################################
# EXPERIMENT: Breast Cancer
###############################################################################

def experiment_real_breast_cancer():
    """Breast Cancer benchmark (binary classification, N_REPS=20)."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Real Data — Breast Cancer")
    log("=" * 70)

    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target
    bc_names = list(bc.feature_names)

    corr = np.abs(np.corrcoef(X_bc.T))
    n_high = (np.sum(corr > 0.9) - len(bc_names)) // 2
    log(f"  {len(bc_names)} features, {n_high} pairs with |r|>0.9")

    # Hold out test set from raw data, re-split and re-scale per rep (D2 fix)
    X_bc_pool, X_bc_test, y_bc_pool, y_bc_test = train_test_split(
        X_bc, y_bc, test_size=0.2, random_state=SEED,
    )

    bc_methods = ['Single Best', 'Random Forest', 'DASH (MaxMin)']
    bc_results = {}

    for name in bc_methods:
        imp_runs, ablation_runs, keff_runs = [], [], []
        for rep in range(N_REPS):
            rep_seed = SEED + rep

            # D2: Re-split and re-fit scaler per rep (avoids scaler leakage)
            Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(
                X_bc_pool, y_bc_pool, test_size=0.2, random_state=rep_seed,
            )
            # A4: Separate explain set from train
            Xtr_r, Xexp_r, ytr_r, yexp_r = train_test_split(
                Xtr_r, ytr_r, test_size=0.12, random_state=rep_seed,
            )
            scaler_r = StandardScaler().fit(Xtr_r)
            Xtr_r = scaler_r.transform(Xtr_r)
            Xv_r = scaler_r.transform(Xv_r)
            Xexp_r = scaler_r.transform(Xexp_r)
            Xte_r = scaler_r.transform(X_bc_test)

            if name == 'Single Best':
                m = SingleBestBaseline(n_trials=N_TRIALS_SB, task='binary', seed=rep_seed)
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
                imp = m.global_importance_
                abl = feature_ablation_score(m.model_, Xte_r, y_bc_test, imp)
            elif name == 'Random Forest':
                m = RandomForestBaseline(
                    n_estimators=500, task='binary', seed=rep_seed,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
                imp = m.global_importance_
                abl = feature_ablation_score(m.model_, Xte_r, y_bc_test, imp)
            else:
                m = DASHPipeline(
                    M=M, K=K, epsilon=REAL_EPSILON, delta=DELTA,
                    epsilon_mode=REAL_EPSILON_MODE,
                    selection_method='maxmin', task='binary',
                    n_jobs=-1, seed=rep_seed, verbose=False,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=bc_names)
                imp = m.global_importance_
                proxy_model = m.selected_models_[0]
                abl = feature_ablation_score(proxy_model, Xte_r, y_bc_test, imp)

            imp_runs.append(imp)
            ablation_runs.append(abl)
            if hasattr(m, 'selected_indices_') and m.selected_indices_ is not None:
                keff_runs.append(len(m.selected_indices_))

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        bc_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
            'k_eff_mean': float(np.mean(keff_runs)) if keff_runs else None,
            'k_eff_std': float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
            'ablation_mean': np.mean(ablation_runs),
            'ablation_std': np.std(ablation_runs, ddof=1),
            'ablation_runs': np.array(ablation_runs),
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
            f"ablation={np.mean(ablation_runs):.4f}")

    # C7+F1: Wilcoxon signed-rank test and Cohen's d
    _log_pairwise_significance(bc_results, 'DASH (MaxMin)', bc_methods, 'Breast Cancer')

    # IS plot and disagreement map from last DASH run
    fig = m.plot_importance_stability(title='IS Plot — Breast Cancer', annotate_top_k=8)
    fig.savefig(f"{OUT}/figures/is_plot_breast_cancer.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{OUT}/figures/is_plot_breast_cancer.pdf", bbox_inches='tight')
    plt.close(fig)

    var_obs = np.mean(m.variance_matrix_, axis=1)
    fig = local_disagreement_map(
        m.all_shap_matrices_, np.argmax(var_obs),
        feature_names=bc_names, top_k=12,
    )
    fig.savefig(f"{OUT}/figures/disagreement_breast_cancer.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{OUT}/figures/disagreement_breast_cancer.pdf", bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: is_plot_breast_cancer.png/pdf, disagreement_breast_cancer.png/pdf")

    save_json(bc_results, f"{OUT}/tables/breast_cancer.json")
    elapsed = time.time() - t0
    log(f"  Breast Cancer completed in {elapsed/60:.1f} min")
    return bc_results


###############################################################################
# EXPERIMENT: Superconductor UCI Benchmark
###############################################################################

def experiment_real_superconductor():
    """Superconductor UCI benchmark with scale-appropriate epsilon."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Real Data — Superconductor UCI")
    log("=" * 70)

    from sklearn.datasets import fetch_openml

    log("  Loading Superconductor dataset...")
    data = fetch_openml(name='superconduct', version=1, as_frame=False, parser='auto')
    X_sc, y_sc = data.data, data.target
    sc_names = [f'f{i}' for i in range(X_sc.shape[1])]
    log(f"  {X_sc.shape[0]} samples, {X_sc.shape[1]} features")

    X_sc_pool, X_sc_test, y_sc_pool, y_sc_test = train_test_split(
        X_sc, y_sc, test_size=0.2, random_state=SEED,
    )

    SC_M = 200  # Lighter compute for real-world dataset
    SC_K = 30
    sc_methods = ['Single Best', 'Large Single Model', 'Random Forest', 'DASH (MaxMin)']
    sc_results = {}

    for name in sc_methods:
        imp_runs, rmse_runs, ablation_runs, keff_runs = [], [], [], []
        for rep in range(N_REPS):
            rep_seed = SEED + rep

            # A4: Separate explain set from test
            Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(
                X_sc_pool, y_sc_pool, test_size=0.2, random_state=rep_seed,
            )
            Xtr_r, Xexp_r, ytr_r, yexp_r = train_test_split(
                Xtr_r, ytr_r, test_size=0.12, random_state=rep_seed,
            )
            scaler_r = StandardScaler().fit(Xtr_r)
            Xtr_r = scaler_r.transform(Xtr_r)
            Xv_r = scaler_r.transform(Xv_r)
            Xexp_r = scaler_r.transform(Xexp_r)
            Xte_r = scaler_r.transform(X_sc_test)

            if name == 'Single Best':
                m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
                imp = m.global_importance_
                rmse_val = rmse_score(y_sc_test, m.model_.predict(Xte_r))
                abl = feature_ablation_score(m.model_, Xte_r, y_sc_test, imp)
            elif name == 'Large Single Model':
                m = LargeSingleModelBaseline(
                    K=SC_K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
                    colsample_bytree=0.2, seed=rep_seed,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
                imp = m.global_importance_
                rmse_val = rmse_score(y_sc_test, m.model_.predict(Xte_r))
                abl = feature_ablation_score(m.model_, Xte_r, y_sc_test, imp)
            elif name == 'Random Forest':
                m = RandomForestBaseline(
                    n_estimators=500, task='regression', seed=rep_seed,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
                imp = m.global_importance_
                rmse_val = rmse_score(y_sc_test, m.model_.predict(Xte_r))
                abl = feature_ablation_score(m.model_, Xte_r, y_sc_test, imp)
            else:
                m = DASHPipeline(
                    M=SC_M, K=SC_K, epsilon=REAL_EPSILON, delta=DELTA,
                    epsilon_mode=REAL_EPSILON_MODE,
                    selection_method='maxmin', n_jobs=-1,
                    seed=rep_seed, verbose=False,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=sc_names)
                imp = m.global_importance_
                preds = m.get_consensus_ensemble_predictions(Xte_r)
                rmse_val = rmse_score(y_sc_test, preds)
                proxy_model = m.selected_models_[0]
                abl = feature_ablation_score(proxy_model, Xte_r, y_sc_test, imp)

            imp_runs.append(imp)
            rmse_runs.append(rmse_val)
            ablation_runs.append(abl)
            if hasattr(m, 'selected_indices_') and m.selected_indices_ is not None:
                keff_runs.append(len(m.selected_indices_))

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        sc_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
            'k_eff_mean': float(np.mean(keff_runs)) if keff_runs else None,
            'k_eff_std': float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
            'rmse_mean': np.mean(rmse_runs), 'rmse_std': np.std(rmse_runs, ddof=1),
            'ablation_mean': np.mean(ablation_runs), 'ablation_std': np.std(ablation_runs, ddof=1),
            'rmse_runs': np.array(rmse_runs),
            'ablation_runs': np.array(ablation_runs),
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
            f"RMSE={np.mean(rmse_runs):.2f}±{np.std(rmse_runs, ddof=1):.2f}  "
            f"ablation={np.mean(ablation_runs):.4f}")

    # C7+F1: Wilcoxon signed-rank test and Cohen's d
    _log_pairwise_significance(sc_results, 'DASH (MaxMin)', sc_methods, 'Superconductor')

    save_json(sc_results, f"{OUT}/tables/superconductor.json")
    elapsed = time.time() - t0
    log(f"  Superconductor completed in {elapsed/60:.1f} min")
    return sc_results


###############################################################################
# EXPERIMENT: Epsilon Sensitivity
###############################################################################

def experiment_epsilon_sensitivity():
    """Epsilon sensitivity sweep: epsilon ∈ {0.03, 0.05, 0.08, 0.10}.

    Trains population ONCE per rep, then varies epsilon on the same models
    to properly isolate the effect of the filtering threshold.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Epsilon Sensitivity")
    log("=" * 70)

    from dash.core.population import generate_model_population
    from dash.core.filtering import performance_filter
    from dash.core.diversity import greedy_maxmin_selection

    EPS_VALUES = [0.03, 0.05, 0.08, 0.10]
    EPS_M = M
    eps_results = {eps: {
        'n_passing': [], 'k_eff': [],
        'acc_runs': [], 'eq_runs': [], 'imp_runs': [],
    } for eps in EPS_VALUES}

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
            generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)

        # Train population ONCE per rep
        models, val_scores, configs = generate_model_population(
            Xtr, ytr, Xv, yv, M=EPS_M, task='regression',
            n_jobs=-1, seed=rep_seed, verbose=False,
        )

        for eps in EPS_VALUES:
            # Stage 2: Filter at this epsilon
            filtered = performance_filter(val_scores, epsilon=eps,
                                          higher_is_better=True, verbose=False)
            eps_results[eps]['n_passing'].append(len(filtered))

            if len(filtered) < 2:
                log(f"  eps={eps}: only {len(filtered)} passed, skipping")
                continue

            # Stage 3: MaxMin diversity selection
            imp_vecs = get_preliminary_importance(models, filtered, Xexp, method='gain')
            filt_scores = {i: val_scores[i] for i in filtered}
            selected = greedy_maxmin_selection(imp_vecs, filt_scores,
                                              K=K, delta=DELTA, verbose=False)
            eps_results[eps]['k_eff'].append(len(selected))

            # Stage 4-5: Consensus SHAP (use Xexp, not Xte, to avoid data leakage)
            cons, all_shap = compute_consensus(models, selected, Xexp, seed=rep_seed, verbose=False)
            _, _, _, imp = compute_diagnostics(all_shap)

            r, _ = dgp_agreement(imp, true_imp)
            eps_results[eps]['acc_runs'].append(r)
            eps_results[eps]['eq_runs'].append(within_group_equity(imp, grps))
            eps_results[eps]['imp_runs'].append(imp)

    log(f"\n  {'ε':>6} {'Models Passing':>16} {'K_eff':>12} {'Stability':>10} {'Accuracy':>10}")
    log("  " + "=" * 60)
    for eps in EPS_VALUES:
        stab = importance_stability(eps_results[eps]['imp_runs'])
        eps_results[eps]['stability'] = stab
        log(f"  {eps:>6.2f} {np.mean(eps_results[eps]['n_passing']):>12.1f}±"
            f"{np.std(eps_results[eps]['n_passing']):<4.1f}"
            f"{np.mean(eps_results[eps]['k_eff']):>8.1f}±"
            f"{np.std(eps_results[eps]['k_eff']):<4.1f}"
            f"{stab:>10.4f}"
            f"{np.mean(eps_results[eps]['acc_runs']):>10.4f}")

    save_json(eps_results, f"{OUT}/tables/epsilon_sensitivity.json")
    elapsed = time.time() - t0
    log(f"  Epsilon sensitivity completed in {elapsed/60:.1f} min")
    return eps_results


###############################################################################
# EXPERIMENT: Ablation Studies
###############################################################################

def experiment_ablation():
    """Ablation studies: one parameter at a time, across multiple rho levels."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Ablation Studies")
    log("=" * 70)

    ABL_N_REPS = N_REPS  # Match main sweep for comparable stability estimates
    ABL_DEFAULTS = {'M': M, 'K': K, 'eps': EPSILON, 'delta': DELTA}
    ABL_RHOS = [0.0, 0.9, 0.95]

    ablations = {
        'M': [50, 100, 200, 500],
        'K': [5, 10, 20, 30, 50],
        'eps': [0.01, 0.03, 0.05, 0.08, 0.10],
        'delta': [0.01, 0.05, 0.10, 0.20],
    }

    abl_results = {rho: {} for rho in ABL_RHOS}

    for abl_rho in ABL_RHOS:
        log(f"\n{'=' * 60}")
        log(f"Ablation at ρ = {abl_rho}")
        log(f"{'=' * 60}")
        for param_name, values in ablations.items():
            log(f"\n--- Ablation: {param_name} ---")
            abl_results[abl_rho][param_name] = {}
            for val in values:
                p = ABL_DEFAULTS.copy()
                p[param_name] = val
                log(f"  {param_name}={val}  ", )

                imp_runs, acc_runs = [], []
                for rep in range(ABL_N_REPS):
                    rep_seed = SEED + rep
                    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
                        generate_synthetic_linear(N=5000, rho=abl_rho, seed=rep_seed)

                    dm = DASHPipeline(
                        M=p['M'], K=p['K'], epsilon=p['eps'],
                        delta=p['delta'], selection_method='maxmin',
                        n_jobs=-1, seed=rep_seed, verbose=False,
                    )
                    try:
                        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=make_feature_names())
                    except ValueError:
                        # Too few models passed filter (e.g. eps too tight)
                        continue
                    imp = dm.global_importance_
                    r, _ = dgp_agreement(imp, true_imp)
                    acc_runs.append(r)
                    imp_runs.append(imp)

                if len(imp_runs) >= 2:
                    stab = importance_stability(imp_runs)
                    abl_results[abl_rho][param_name][val] = {
                        'stability': stab,
                        'accuracy_mean': np.mean(acc_runs),
                        'accuracy_std': np.std(acc_runs, ddof=1),
                        'n_successful': len(imp_runs),
                    }
                    log(f"    stab={stab:.4f}  acc={np.mean(acc_runs):.4f}  ({len(imp_runs)}/{ABL_N_REPS} reps)")
                else:
                    abl_results[abl_rho][param_name][val] = {
                        'stability': float('nan'),
                        'accuracy_mean': float('nan'),
                        'accuracy_std': float('nan'),
                        'n_successful': len(imp_runs),
                    }
                    log(f"    SKIPPED — only {len(imp_runs)}/{ABL_N_REPS} reps passed filter")

    save_json(abl_results, f"{OUT}/tables/ablation.json")
    log(f"  Saved: {OUT}/tables/ablation.json")

    # Generate publication figure
    plot_ablation_sensitivity(abl_results)

    elapsed = time.time() - t0
    log(f"  Ablation completed in {elapsed/60:.1f} min")
    return abl_results


###############################################################################
# EXPERIMENT: Variance Decomposition (F1 fix)
###############################################################################

def experiment_variance_decomposition():
    """Variance decomposition: separates data-sampling vs model-selection variance."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Variance Decomposition")
    log("=" * 70)

    from dash.core.population import generate_model_population
    from dash.core.filtering import performance_filter

    VD_RHO = 0.9
    feature_names = make_feature_names()

    conditions = {
        'data_fixed': 'Fix data seed, vary model seeds → isolates model-selection variance',
        'model_fixed': 'Fix model seed, vary data seeds → isolates data-sampling variance',
        'both_varied': 'Vary both → total variance (reference)',
    }

    methods = ['Single Best', 'DASH (MaxMin)']
    results = {cond: {m: [] for m in methods} for cond in conditions}

    for rep in range(N_REPS):
        log(f"  Rep {rep+1}/{N_REPS}")

        for cond in conditions:
            if cond == 'data_fixed':
                data_seed, model_seed = SEED, SEED + 1000 + rep
            elif cond == 'model_fixed':
                data_seed, model_seed = SEED + rep, SEED
            else:  # both_varied
                data_seed, model_seed = SEED + rep, SEED + rep

            Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
                generate_synthetic_linear(N=5000, rho=VD_RHO, seed=data_seed)

            # Single Best
            sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=model_seed)
            sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=model_seed)
            results[cond]['Single Best'].append(sb.global_importance_)

            # DASH (MaxMin)
            dm = DASHPipeline(
                M=M, K=K, epsilon=EPSILON, delta=DELTA,
                selection_method='maxmin', n_jobs=-1,
                seed=model_seed, verbose=False,
            )
            dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
            results[cond]['DASH (MaxMin)'].append(dm.global_importance_)

    # Compute stability for each condition × method
    log(f"\n  {'Condition':<16} {'Method':<20} {'Stability':>10}")
    log("  " + "=" * 50)
    summary = {}
    for cond in conditions:
        summary[cond] = {}
        for m in methods:
            stab = importance_stability(results[cond][m])
            summary[cond][m] = {'stability': stab}
            log(f"  {cond:<16} {m:<20} {stab:>10.4f}")

    # Variance decomposition ratios
    # CAVEAT: (1 - stability) is a proxy for instability, not a proper
    # variance.  Stability is pairwise Spearman ρ, so the "ratios" below
    # are indicative of relative contribution but do not satisfy an exact
    # additive decomposition (model_var + data_var ≠ total_var in general).
    log(f"\n  Variance Decomposition Ratios (indicative — see caveat in code):")
    for m in methods:
        total_var = 1.0 - summary['both_varied'][m]['stability']
        model_var = 1.0 - summary['data_fixed'][m]['stability']
        data_var = 1.0 - summary['model_fixed'][m]['stability']
        summary_ratios = {}
        if total_var > 0:
            model_frac = model_var / total_var
            data_frac = data_var / total_var
            log(f"    {m}: model-selection={model_frac:.1%}, "
                f"data-sampling={data_frac:.1%} of total instability")
            summary_ratios = {
                'model_selection_frac': model_frac,
                'data_sampling_frac': data_frac,
            }
        else:
            log(f"    {m}: total variance ≈ 0 (perfectly stable)")
        for cond in conditions:
            summary[cond][m]['decomposition'] = summary_ratios

    save_json(summary, f"{OUT}/tables/variance_decomposition.json")
    elapsed = time.time() - t0
    log(f"  Variance decomposition completed in {elapsed/60:.1f} min")
    return summary


###############################################################################
# SUCCESS CRITERIA
###############################################################################

def check_success_criteria(sweep_results, epsilon_results=None,
                           nonlinear_results=None, sig_results=None,
                           sc_results=None, cal_results=None,
                           bc_results=None, vardecomp_results=None):
    """Evaluate pass/fail criteria against experimental results.

    Criteria 1-5 use sweep_results (required). Criteria 6-11 use optional
    results from other experiments; they are skipped if not provided.
    """
    log("\n" + "=" * 70)
    log("SUCCESS CRITERIA CHECK")
    log("=" * 70)

    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    results = []

    # 1. Stability wins
    n_wins = sum(
        1 for rho in rho_levels
        if sweep_results[rho]['DASH (MaxMin)']['stability']
        > sweep_results[rho]['Single Best']['stability']
    )
    passed = n_wins >= 4
    results.append(passed)
    log(f"  1. Stability wins: {n_wins}/{len(rho_levels)} "
        f"({'PASS' if passed else 'FAIL'}, need >=80%)")

    # 2. DGP agreement at rho=0.9 (relative to Single Best baseline)
    acc_09 = sweep_results[0.9]['DASH (MaxMin)']['accuracy_mean']
    sb_acc_09 = sweep_results[0.9]['Single Best']['accuracy_mean']
    passed = acc_09 >= sb_acc_09
    results.append(passed)
    log(f"  2. DGP agreement at ρ=0.9: DASH={acc_09:.4f} vs SB={sb_acc_09:.4f} "
        f"({'PASS' if passed else 'FAIL'}, DASH >= SB)")

    # 3. Equity wins
    n_eq_wins = sum(
        1 for rho in rho_levels
        if sweep_results[rho]['DASH (MaxMin)']['equity_mean']
        < sweep_results[rho]['Single Best']['equity_mean']
    )
    passed = n_eq_wins >= 4
    results.append(passed)
    log(f"  3. Equity wins: {n_eq_wins}/{len(rho_levels)} "
        f"({'PASS' if passed else 'FAIL'}, need >=80%)")

    # 4. Safety control at rho=0
    rho0_dash = sweep_results[0.0]['DASH (MaxMin)']['accuracy_mean']
    rho0_sb = sweep_results[0.0]['Single Best']['accuracy_mean']
    passed = abs(rho0_dash - rho0_sb) < 0.1
    results.append(passed)
    log(f"  4. ρ=0 control: DASH dgp={rho0_dash:.4f}, SB dgp={rho0_sb:.4f} "
        f"({'PASS' if passed else 'FAIL'}, gap < 0.1)")

    # 5. K_eff increases with epsilon
    if epsilon_results is not None:
        eps_vals = sorted(epsilon_results.keys(), key=lambda x: float(x))
        keffs = []
        for e in eps_vals:
            k_data = epsilon_results[e].get('k_eff', [])
            keffs.append(np.mean(k_data) if k_data else 0)
        passed = all(keffs[i] <= keffs[i + 1] for i in range(len(keffs) - 1))
        results.append(passed)
        keffs_str = [f"{k:.1f}" for k in keffs]
        log(f"  5. K_eff monotonic with epsilon: {keffs_str} "
            f"({'PASS' if passed else 'FAIL'})")
    else:
        log("  5. K_eff monotonicity: SKIP (no epsilon_results)")

    # 6. Nonlinear DGP: DASH > SB stability at rho=0.9
    if nonlinear_results is not None and 0.9 in nonlinear_results:
        nl_09 = nonlinear_results[0.9]
        if 'DASH (MaxMin)' in nl_09 and 'Single Best' in nl_09:
            d_stab = nl_09['DASH (MaxMin)']['stability']
            s_stab = nl_09['Single Best']['stability']
            passed = d_stab > s_stab
            results.append(passed)
            log(f"  6. Nonlinear (ρ=0.9): DASH={d_stab:.4f} vs SB={s_stab:.4f} "
                f"({'PASS' if passed else 'FAIL'})")
        else:
            log("  6. Nonlinear DGP: SKIP (missing methods in results)")
    else:
        log("  6. Nonlinear DGP: SKIP (no nonlinear_results)")

    # 7. Significance: enough tests significant
    if sig_results is not None:
        n_sig = sum(1 for t in sig_results if t.get('significant', False))
        n_total = len(sig_results)
        passed = n_total > 0 and n_sig >= n_total * 0.5
        results.append(passed)
        log(f"  7. Significance: {n_sig}/{n_total} tests significant "
            f"({'PASS' if passed else 'FAIL'}, need >=50%)")
    else:
        log("  7. Significance tests: SKIP (no sig_results)")

    # 8. Superconductor: DASH stability > SB
    if sc_results is not None:
        if 'DASH (MaxMin)' in sc_results and 'Single Best' in sc_results:
            d_stab = sc_results['DASH (MaxMin)']['stability']
            s_stab = sc_results['Single Best']['stability']
            passed = d_stab > s_stab
            results.append(passed)
            log(f"  8. Superconductor: DASH={d_stab:.4f} vs SB={s_stab:.4f} "
                f"({'PASS' if passed else 'FAIL'})")
        else:
            log("  8. Superconductor: SKIP (missing methods)")
    else:
        log("  8. Superconductor: SKIP (no sc_results)")

    # 9. California Housing: DASH stability > SB
    if cal_results is not None:
        if 'DASH (MaxMin)' in cal_results and 'Single Best' in cal_results:
            d_stab = cal_results['DASH (MaxMin)']['stability']
            s_stab = cal_results['Single Best']['stability']
            passed = d_stab > s_stab
            results.append(passed)
            log(f"  9. California Housing: DASH={d_stab:.4f} vs SB={s_stab:.4f} "
                f"({'PASS' if passed else 'FAIL'})")
        else:
            log("  9. California Housing: SKIP (missing methods)")
    else:
        log("  9. California Housing: SKIP (no cal_results)")

    # 10. Breast Cancer: DASH stability > SB
    if bc_results is not None:
        if 'DASH (MaxMin)' in bc_results and 'Single Best' in bc_results:
            d_stab = bc_results['DASH (MaxMin)']['stability']
            s_stab = bc_results['Single Best']['stability']
            passed = d_stab > s_stab
            results.append(passed)
            log(f" 10. Breast Cancer: DASH={d_stab:.4f} vs SB={s_stab:.4f} "
                f"({'PASS' if passed else 'FAIL'})")
        else:
            log(" 10. Breast Cancer: SKIP (missing methods)")
    else:
        log(" 10. Breast Cancer: SKIP (no bc_results)")

    # 11. Variance decomposition: DASH model-var < SB model-var
    # Structure: {condition: {method: {'stability': float}}}
    # Model-selection instability = 1 - stability under 'data_fixed' condition
    if vardecomp_results is not None:
        data_fixed = vardecomp_results.get('data_fixed', {})
        dash_df = data_fixed.get('DASH (MaxMin)', {})
        sb_df = data_fixed.get('Single Best', {})
        if 'stability' in dash_df and 'stability' in sb_df:
            d_model = 1.0 - dash_df['stability']
            s_model = 1.0 - sb_df['stability']
            passed = d_model < s_model
            results.append(passed)
            log(f" 11. Variance decomp: DASH model-instab={d_model:.4f} vs SB={s_model:.4f} "
                f"({'PASS' if passed else 'FAIL'})")
        else:
            log(" 11. Variance decomposition: SKIP (missing data_fixed stability)")
    else:
        log(" 11. Variance decomposition: SKIP (no vardecomp_results)")

    # Summary
    n_passed = sum(results)
    n_total = len(results)
    log(f"\n  Overall: {n_passed}/{n_total} criteria passed")

    rho09 = sweep_results[0.9]
    log(f"\n  Stability at ρ=0.9:")
    for n in rho09:
        log(f"    {n:<20} {rho09[n]['stability']:.4f}")

    dash_stab = rho09['DASH (MaxMin)']['stability']
    sb_stab = rho09['Single Best']['stability']
    log(f"\n  DASH improvement over Single Best: +{dash_stab - sb_stab:.4f}")
    if 'Large Single Model' in rho09:
        lsm_stab = rho09['Large Single Model']['stability']
        log(f"  DASH improvement over LSM:         +{dash_stab - lsm_stab:.4f}")

    return results


###############################################################################
# EXPERIMENT: Success Criteria (m7 — registered as runnable experiment)
###############################################################################

def experiment_success_criteria():
    """Run linear_sweep (if needed) then evaluate pass/fail success criteria."""
    sweep_results = experiment_linear_sweep()
    check_success_criteria(sweep_results)
    return sweep_results


###############################################################################
# EXPERIMENT: First-Mover Bias Visualization
###############################################################################

def experiment_first_mover_visualization():
    """Visualize first-mover bias: importance concentration within a correlated group.

    Generates a grouped bar chart showing per-feature importance within
    correlated group 1 (features 0-4, all with true importance beta/5 = 0.4)
    for Single Best, Large Single Model, and DASH (MaxMin).

    This figure directly demonstrates the first-mover bias mechanism:
    SB/LSM concentrate importance on one arbitrary feature, while DASH
    distributes it proportionally.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: First-Mover Bias Visualization")
    log("=" * 70)

    rho = 0.9
    feature_names = make_feature_names()
    group_features = list(range(5))  # Group 1: features 0-4
    group_labels = [feature_names[i] for i in group_features]
    n_vis_reps = 20  # Match N_REPS for publication-quality results

    methods_to_run = ['Single Best', 'Large Single Model', 'DASH (MaxMin)']
    method_importances = {m: [] for m in methods_to_run}

    for rep in range(n_vis_reps):
        rep_seed = SEED + rep
        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
            generate_synthetic_linear(N=5000, rho=rho, seed=rep_seed)

        # Single Best
        sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        method_importances['Single Best'].append(sb.global_importance_[group_features])

        # Large Single Model
        lsm = LargeSingleModelBaseline(
            K=K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
            colsample_bytree=0.2, seed=rep_seed,
        )
        lsm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        method_importances['Large Single Model'].append(lsm.global_importance_[group_features])

        # DASH (MaxMin)
        dash = DASHPipeline(
            M=M, K=K, epsilon=EPSILON, delta=DELTA,
            selection_method='maxmin', n_jobs=-1,
            seed=rep_seed, verbose=False,
        )
        dash.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        method_importances['DASH (MaxMin)'].append(dash.global_importance_[group_features])

    # Average across reps
    avg_imp = {m: np.mean(method_importances[m], axis=0) for m in methods_to_run}
    std_imp = {m: np.std(method_importances[m], axis=0, ddof=1) for m in methods_to_run}

    # True importance for group 1
    true_per_feature = 2.0 / 5  # beta_1 / group_size

    # Log results
    log(f"\n  Per-feature importance within Group 1 (true = {true_per_feature:.2f} each):")
    log(f"  {'Feature':<12} {'SB':>8} {'LSM':>8} {'DASH':>8}")
    log("  " + "-" * 40)
    for i, fname in enumerate(group_labels):
        log(f"  {fname:<12} {avg_imp['Single Best'][i]:>8.4f} "
            f"{avg_imp['Large Single Model'][i]:>8.4f} "
            f"{avg_imp['DASH (MaxMin)'][i]:>8.4f}")

    # Concentration metric: max/sum within group
    for m in methods_to_run:
        conc = np.max(avg_imp[m]) / (np.sum(avg_imp[m]) + 1e-10)
        log(f"  {m}: concentration = {conc:.3f} (1/5 = 0.200 is perfectly equitable)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(group_features))
    width = 0.22
    colors_list = [COLORS.get(m, '#333') for m in methods_to_run]
    for i, m in enumerate(methods_to_run):
        ax.bar(x + i * width, avg_imp[m], width, yerr=std_imp[m],
               label=m, color=colors_list[i], capsize=3, alpha=0.85)
    ax.axhline(y=true_per_feature, color='black', linestyle='--', linewidth=1,
               label=f'True importance ({true_per_feature:.2f})', alpha=0.6)
    ax.set_xlabel('Feature (within correlated group 1)')
    ax.set_ylabel('Global importance (mean |SHAP|)')
    ax.set_title(f'First-Mover Bias: Importance Concentration at ρ={rho}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(group_labels)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/first_mover_concentration.pdf", dpi=150)
    fig.savefig(f"{OUT}/figures/first_mover_concentration.png", dpi=150)
    plt.close(fig)
    log(f"  Saved: {OUT}/figures/first_mover_concentration.pdf")

    elapsed = time.time() - t0
    log(f"  First-mover visualization completed in {elapsed/60:.1f} min")
    return avg_imp


###############################################################################
# EXPERIMENT: First-Mover Bias Isolation (IMPL_PLAN B3)
###############################################################################

def experiment_first_mover_bias():
    """First-mover bias isolation: concentration grows with tree count.

    Trains a single XGBoost with increasing n_estimators and measures how
    importance concentration within a correlated group grows with depth.
    Compares against M independent models averaged at the same total tree
    count.  Produces a line plot: concentration vs n_estimators.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: First-Mover Bias Isolation")
    log("=" * 70)

    import xgboost as xgb
    import shap

    rho = 0.9
    n_estimator_levels = [50, 100, 200, 500, 1000, 2000]
    n_bias_reps = 20  # Match N_REPS for publication-quality results
    feature_names_loc = make_feature_names()
    group_features = list(range(5))  # Group 1: features 0-4

    single_concentrations = {n: [] for n in n_estimator_levels}
    dash_concentrations = {n: [] for n in n_estimator_levels}

    for rep in range(n_bias_reps):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{n_bias_reps}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
            generate_synthetic_linear(N=5000, rho=rho, seed=rep_seed)

        for n_est in n_estimator_levels:
            # --- Single model: concentration grows with depth ---
            model = xgb.XGBRegressor(
                n_estimators=n_est,
                max_depth=6, learning_rate=0.1,
                colsample_bytree=0.3, subsample=0.8,
                random_state=rep_seed,
            )
            model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xexp[:200])
            imp_single = np.mean(np.abs(sv), axis=0)
            grp_imp = imp_single[group_features]
            conc = np.max(grp_imp) / (np.sum(grp_imp) + 1e-10)
            single_concentrations[n_est].append(conc)

            # --- Independent ensemble: M small models, averaged ---
            n_per_model = max(10, n_est // 20)
            m_models = n_est // n_per_model
            imp_accum = np.zeros(len(feature_names_loc))
            for mi in range(m_models):
                m_seed = rep_seed * 10000 + mi
                mdl = xgb.XGBRegressor(
                    n_estimators=n_per_model,
                    max_depth=6, learning_rate=0.1,
                    colsample_bytree=np.random.RandomState(m_seed).uniform(0.1, 0.5),
                    subsample=0.8,
                    random_state=m_seed,
                )
                mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
                sv_m = shap.TreeExplainer(mdl).shap_values(Xexp[:200])
                imp_accum += np.mean(np.abs(sv_m), axis=0)
            imp_avg = imp_accum / m_models
            grp_imp_avg = imp_avg[group_features]
            conc_avg = np.max(grp_imp_avg) / (np.sum(grp_imp_avg) + 1e-10)
            dash_concentrations[n_est].append(conc_avg)

    # Summarize
    summary = {}
    log(f"\n  {'n_estimators':>14} {'Single Conc':>14} {'Indep Conc':>14} {'Ratio':>8}")
    log("  " + "=" * 55)
    for n_est in n_estimator_levels:
        sc = np.mean(single_concentrations[n_est])
        dc = np.mean(dash_concentrations[n_est])
        log(f"  {n_est:>14} {sc:>14.4f} {dc:>14.4f} {sc/dc:>8.2f}")
        summary[str(n_est)] = {
            'single_concentration': sc,
            'single_concentration_std': float(np.std(single_concentrations[n_est], ddof=1)),
            'independent_concentration': dc,
            'independent_concentration_std': float(np.std(dash_concentrations[n_est], ddof=1)),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = n_estimator_levels
    sc_means = [np.mean(single_concentrations[n]) for n in xs]
    sc_stds = [np.std(single_concentrations[n], ddof=1) for n in xs]
    dc_means = [np.mean(dash_concentrations[n]) for n in xs]
    dc_stds = [np.std(dash_concentrations[n], ddof=1) for n in xs]

    ax.errorbar(xs, sc_means, yerr=sc_stds, fmt='s-', color='#e74c3c',
                label='Single Sequential Model', linewidth=2, capsize=4, markersize=7)
    ax.errorbar(xs, dc_means, yerr=dc_stds, fmt='o-', color='#2ecc71',
                label='Independent Ensemble (averaged)', linewidth=2, capsize=4, markersize=7)
    ax.axhline(y=0.2, color='black', linestyle='--', alpha=0.5,
               label='Perfect equity (1/5 = 0.20)')
    ax.set_xlabel('Number of Trees (n_estimators)')
    ax.set_ylabel('Concentration (max/sum within group)')
    ax.set_title('First-Mover Bias Isolation: Concentration Grows with Tree Count')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/first_mover_bias_isolation.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"{OUT}/figures/first_mover_bias_isolation.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/first_mover_bias_isolation.{{png,pdf}}")

    save_json(summary, f"{OUT}/tables/first_mover_bias.json")
    elapsed = time.time() - t0
    log(f"  First-mover bias isolation completed in {elapsed/60:.1f} min")
    return summary


###############################################################################
# HELPERS: Timing summary
###############################################################################

def format_timing_table(sweep_results, rho=0.9):
    """Print a formatted timing comparison table from sweep results."""
    if rho not in sweep_results:
        log(f"  No results for ρ={rho}")
        return
    results = sweep_results[rho]
    log(f"\n  Wall-clock timings at ρ={rho} ({N_REPS} reps):")
    log(f"  {'Method':<24} {'Total (s)':>10} {'Per-rep (s)':>12}")
    log("  " + "-" * 48)
    for name, data in results.items():
        elapsed = data.get('elapsed_s', 0)
        per_rep = elapsed / N_REPS if N_REPS > 0 else 0
        log(f"  {name:<24} {elapsed:>10.1f} {per_rep:>12.1f}")


###############################################################################
# EXPERIMENT REGISTRY
###############################################################################

EXPERIMENTS = {
    'linear_sweep': experiment_linear_sweep,
    'overlapping': experiment_overlapping,
    'nonlinear_sweep': experiment_nonlinear_sweep,
    'table2_baselines': experiment_table2_baselines,
    'real_california': experiment_real_california,
    'real_breast_cancer': experiment_real_breast_cancer,
    'real_superconductor': experiment_real_superconductor,
    'epsilon_sensitivity': experiment_epsilon_sensitivity,
    'ablation': experiment_ablation,
    'variance_decomposition': experiment_variance_decomposition,
    'first_mover_visualization': experiment_first_mover_visualization,
    'first_mover_bias': experiment_first_mover_bias,
    'success_criteria': experiment_success_criteria,
}

# Default run order (all experiments)
DEFAULT_ORDER = [
    'linear_sweep',
    'first_mover_visualization',
    'overlapping',
    'nonlinear_sweep',
    'table2_baselines',
    'real_california',
    'real_breast_cancer',
    'real_superconductor',
    'epsilon_sensitivity',
    'ablation',
    'variance_decomposition',
    'first_mover_bias',
]


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DASH Experimental Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                                  # Run all experiments
  python run_experiments.py --experiments linear_sweep       # Run only the linear sweep
  python run_experiments.py --experiments linear_sweep nonlinear_sweep
  python run_experiments.py --experiments real_california real_breast_cancer
  python run_experiments.py --list                           # List available experiments
        """,
    )
    parser.add_argument(
        '--experiments', nargs='+', choices=list(EXPERIMENTS.keys()),
        help='Specific experiments to run (default: all)',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available experiments and exit',
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, fn in EXPERIMENTS.items():
            doc = (fn.__doc__ or '').strip().split('\n')[0]
            print(f"  {name:<24} {doc}")
        sys.exit(0)

    _ensure_dirs()
    t_start = time.time()

    log("DASH Experimental Validation")
    log(f"Config: M={M}, K={K}, ε={EPSILON}, δ={DELTA}, N_REPS={N_REPS}")
    log(f"  Real-data: ε={REAL_EPSILON} (mode={REAL_EPSILON_MODE})")
    log("")

    to_run = args.experiments or DEFAULT_ORDER
    sweep_results = None

    for name in to_run:
        result = EXPERIMENTS[name]()
        if name == 'linear_sweep':
            sweep_results = result

    # Run success criteria if linear_sweep was included
    if sweep_results is not None:
        check_success_criteria(sweep_results)

    total_time = time.time() - t_start
    log(f"\nTotal runtime: {total_time/60:.1f} minutes")
    log(f"Results saved to {OUT}/")
