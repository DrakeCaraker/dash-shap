#!/usr/bin/env python3
"""
DASH Experimental Validation — Complete Runner
===============================================
Aligned with audited demo_benchmark_4.ipynb (PAPER_CONFIG).

Run all experiments:
    python run_experiments.py

Run specific experiments:
    python run_experiments.py --experiments linear_sweep nonlinear_sweep
    python run_experiments.py --experiments real_california real_breast_cancer
    python run_experiments.py --experiments table2_baselines ablation

Available experiments:
    linear_sweep       Synthetic Linear DGP correlation sweep (rho ∈ {0,0.5,0.7,0.9,0.95})
    overlapping        Overlapping correlation structure (rho=0.9)
    nonlinear_sweep    Nonlinear DGP correlation sweep
    table2_baselines   Extended baselines at rho=0.9 (Ensemble SHAP, Stochastic Retrain, Dedup)
    real_california    California Housing benchmark
    real_breast_cancer Breast Cancer benchmark
    real_superconductor Superconductor UCI benchmark
    epsilon_sensitivity Epsilon sensitivity analysis
    ablation           Ablation studies (M, K, epsilon, delta)
    success_criteria   Run linear_sweep then evaluate pass/fail criteria
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
    dgp_agreement, importance_accuracy, importance_stability,
    stability_bootstrap_ci, within_group_equity, compare_methods,
    friedman_test,
)
from dash.utils.io import save_json

# ---------------------------------------------------------------------------
# Canonical configuration — matches PAPER_CONFIG from audited notebook
# ---------------------------------------------------------------------------
PAPER_CONFIG = {
    'M': 200,
    'K': 30,
    'N_REPS': 20,
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

# Scale-appropriate epsilons for real-world datasets
CAL_EPSILON = 0.05   # California Housing: RMSE ~0.5-0.7
BC_EPSILON = 0.08    # Breast Cancer (classification)
SC_EPSILON = 0.40    # Superconductor: RMSE ~18

OUT = "results"
FEATURE_NAMES = [f'G{g}_f{j}' for g in range(10) for j in range(5)]


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
    'Large Single Model': '#e74c3c',
    'LSM (Tuned)': '#c0392b',
    'Ensemble SHAP': '#9b59b6',
    'Naive Top-N': '#f39c12',
    'Stochastic Retrain': '#e67e22',
    'DASH (Dedup)': '#3498db',
    'DASH (MaxMin)': '#2ecc71',
    'DASH (Cluster)': '#1abc9c',
}

MARKERS = {
    'Single Best': 's',
    'Large Single Model': 'X',
    'LSM (Tuned)': 'x',
    'Ensemble SHAP': 'D',
    'Naive Top-N': '^',
    'Stochastic Retrain': 'v',
    'DASH (MaxMin)': 'o',
    'DASH (Cluster)': 'P',
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
    plt.close(fig)
    log(f"  Saved: figures/correlation_sweep.png")

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
        plt.close(fig)
        log(f"  Saved: figures/bar_chart_rho09.png")


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
        'Single Best', 'Large Single Model', 'LSM (Tuned)',
        'Stochastic Retrain', 'DASH (MaxMin)',
    ]
    sweep_results = {rho: {} for rho in rho_levels}

    for rho in rho_levels:
        log(f"\n--- ρ = {rho} ---")
        for name in sweep_methods:
            acc_runs, eq_runs, eq_zero_runs, imp_runs, rmse_runs = [], [], [], [], []
            for rep in range(N_REPS):
                rep_seed = SEED + rep
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
                    generate_synthetic_linear(N=5000, rho=rho, seed=rep_seed)

                if name == 'Single Best':
                    m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                elif name == 'Large Single Model':
                    m = LargeSingleModelBaseline(
                        K=K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
                        colsample_bytree=0.2, seed=rep_seed, tune=False,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
                    imp = m.global_importance_
                    preds = m.model_.predict(Xte)
                elif name == 'LSM (Tuned)':
                    m = LargeSingleModelBaseline(
                        K=K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
                        colsample_bytree=0.2, seed=rep_seed, tune=True,
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
                else:  # DASH MaxMin
                    m = DASHPipeline(
                        M=M, K=K, epsilon=EPSILON, delta=DELTA,
                        selection_method='maxmin', n_jobs=-1,
                        seed=rep_seed, verbose=False,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=FEATURE_NAMES)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)

                rmse_val = rmse_score(yte, preds) if preds is not None else np.nan
                r, _ = dgp_agreement(imp, true_imp)
                acc_runs.append(r)
                eq_runs.append(within_group_equity(imp, grps))
                eq_zero_runs.append(within_group_equity(imp, grps, include_zero_groups=True))
                imp_runs.append(imp)
                rmse_runs.append(rmse_val)

            stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
            sweep_results[rho][name] = {
                'stability': stab,
                'stability_se': stab_se,
                'stability_ci_lo': stab_ci_lo,
                'stability_ci_hi': stab_ci_hi,
                'dgp_agreement_mean': float(np.mean(acc_runs)),
                'dgp_agreement_se': float(np.std(acc_runs, ddof=1) / np.sqrt(len(acc_runs))),
                'dgp_agreement_std': float(np.std(acc_runs, ddof=1)),
                # Backward-compatible aliases
                'accuracy_mean': float(np.mean(acc_runs)),
                'accuracy_se': float(np.std(acc_runs, ddof=1) / np.sqrt(len(acc_runs))),
                'accuracy_std': float(np.std(acc_runs, ddof=1)),
                'equity_mean': float(np.mean(eq_runs)),
                'equity_se': float(np.std(eq_runs, ddof=1) / np.sqrt(len(eq_runs))),
                'equity_std': float(np.std(eq_runs, ddof=1)),
                'equity_incl_zero_mean': float(np.mean(eq_zero_runs)),
                'equity_incl_zero_std': float(np.std(eq_zero_runs, ddof=1)),
                'rmse_mean': float(np.mean(rmse_runs)),
                'rmse_std': float(np.std(rmse_runs, ddof=1)),
                'acc_runs': np.array(acc_runs),
                'eq_runs': np.array(eq_runs),
                'rmse_runs': np.array(rmse_runs),
                'imp_runs': imp_runs,
            }
            log(f"  {name:<20} stab={stab:.4f}±{stab_se:.4f} [{stab_ci_lo:.4f},{stab_ci_hi:.4f}]  "
                f"dgp={np.mean(acc_runs):.4f}±{np.std(acc_runs, ddof=1)/np.sqrt(len(acc_runs)):.4f}  "
                f"eq={np.mean(eq_runs):.4f}  RMSE={np.nanmean(rmse_runs):.4f}")

    save_json(sweep_results, f"{OUT}/tables/synthetic_linear_sweep.json")
    log(f"  Saved: {OUT}/tables/synthetic_linear_sweep.json")
    plot_correlation_sweep(sweep_results, rho_levels, sweep_methods)

    elapsed = time.time() - t0
    log(f"  Linear sweep completed in {elapsed/60:.1f} min")
    return sweep_results


###############################################################################
# EXPERIMENT: Overlapping Correlation Structure
###############################################################################

def experiment_overlapping():
    """Overlapping correlation structure at rho=0.9."""
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Overlapping Correlation Structure")
    log("=" * 70)

    method_names = ['Single Best', 'DASH (MaxMin)', 'DASH (Cluster)']
    stability_vectors = {n: [] for n in method_names}

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = \
            generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed, structure="overlapping")

        sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
        stability_vectors['Single Best'].append(sb.global_importance_)

        dm = DASHPipeline(
            M=M, K=K, epsilon=EPSILON, delta=DELTA,
            selection_method='maxmin', n_jobs=-1, seed=rep_seed, verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=FEATURE_NAMES)
        stability_vectors['DASH (MaxMin)'].append(dm.global_importance_)

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
        stability_vectors['DASH (Cluster)'].append(imp_c)

    log("\n  Overlapping structure results:")
    for name in method_names:
        stab = importance_stability(stability_vectors[name])
        log(f"    {name:<20} stability={stab:.4f}")

    elapsed = time.time() - t0
    log(f"  Overlapping completed in {elapsed/60:.1f} min")


###############################################################################
# EXPERIMENT: Nonlinear DGP Correlation Sweep
###############################################################################

def experiment_nonlinear_sweep():
    """Nonlinear DGP sweep: rho ∈ {0.0, 0.5, 0.7, 0.9, 0.95}.

    Evaluates stability and equity (no ground-truth accuracy for nonlinear DGP).
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
        'Stochastic Retrain', 'DASH (MaxMin)',
    ]
    nl_sweep = {rho: {} for rho in nl_rho_levels}

    for rho in nl_rho_levels:
        log(f"\n--- Nonlinear DGP, ρ = {rho} ---")
        for name in nl_methods:
            eq_runs, imp_runs, rmse_runs = [], [], []
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
                else:  # DASH MaxMin
                    m = DASHPipeline(
                        M=M, K=K, epsilon=EPSILON, delta=DELTA,
                        selection_method='maxmin', n_jobs=-1,
                        seed=rep_seed, verbose=False,
                    )
                    m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=FEATURE_NAMES)
                    imp = m.global_importance_
                    preds = m.get_consensus_ensemble_predictions(Xte)

                rmse_val = rmse_score(yte, preds)
                eq_runs.append(within_group_equity(imp, grps))
                imp_runs.append(imp)
                rmse_runs.append(rmse_val)

            stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
            nl_sweep[rho][name] = {
                'stability': stab,
                'stability_se': stab_se,
                'stability_ci_lo': stab_ci_lo,
                'stability_ci_hi': stab_ci_hi,
                'equity_mean': float(np.mean(eq_runs)),
                'equity_std': float(np.std(eq_runs, ddof=1)),
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

    table2_methods = ['Ensemble SHAP', 'Stochastic Retrain', 'DASH (Dedup)']
    table2_results = {}

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
            else:  # DASH (Dedup)
                dm = DASHPipeline(
                    M=M, K=K, epsilon=EPSILON, delta=DELTA,
                    selection_method='dedup', n_jobs=-1,
                    seed=rep_seed, verbose=False,
                )
                dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=FEATURE_NAMES)
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
            'accuracy_mean': float(np.mean(acc_runs)),
            'accuracy_se': float(np.std(acc_runs, ddof=1) / np.sqrt(len(acc_runs))),
            'equity_mean': float(np.mean(eq_runs)),
            'equity_se': float(np.std(eq_runs, ddof=1) / np.sqrt(len(eq_runs))),
            'accuracy_std': float(np.std(acc_runs, ddof=1)),
            'equity_std': float(np.std(eq_runs, ddof=1)),
            'acc_runs': np.array(acc_runs),
            'eq_runs': np.array(eq_runs),
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

    cal_methods = ['Single Best', 'DASH (MaxMin)']
    cal_results = {}

    for name in cal_methods:
        imp_runs, rmse_runs = [], []
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
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
                imp = m.global_importance_
                rmse_val = rmse_score(y_cal_test, m.model_.predict(Xte_r))
            else:
                m = DASHPipeline(
                    M=M, K=K, epsilon=CAL_EPSILON, delta=DELTA,
                    selection_method='maxmin', n_jobs=-1,
                    seed=rep_seed, verbose=False,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=cal_names)
                imp = m.global_importance_
                preds = m.get_consensus_ensemble_predictions(Xte_r)
                rmse_val = rmse_score(y_cal_test, preds)

            imp_runs.append(imp)
            rmse_runs.append(rmse_val)

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        cal_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
            'rmse_mean': float(np.mean(rmse_runs)),
            'rmse_std': float(np.std(rmse_runs, ddof=1)),
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
            f"RMSE={np.mean(rmse_runs):.4f}±{np.std(rmse_runs, ddof=1):.4f}")

    # IS plot from last DASH run
    fig = m.plot_importance_stability(title='IS Plot — California Housing', annotate_top_k=8)
    fig.savefig(f"{OUT}/figures/is_plot_california.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: figures/is_plot_california.png")

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

    X_bc_pool, X_bc_test, y_bc_pool, y_bc_test = train_test_split(
        X_bc, y_bc, test_size=0.2, random_state=SEED,
    )

    bc_methods = ['Single Best', 'DASH (MaxMin)']
    bc_results = {}

    for name in bc_methods:
        imp_runs = []
        for rep in range(N_REPS):
            rep_seed = SEED + rep

            # A4: Separate explain set from test
            Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(
                X_bc_pool, y_bc_pool, test_size=0.2, random_state=rep_seed,
            )
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
            else:
                m = DASHPipeline(
                    M=M, K=K, epsilon=BC_EPSILON, delta=DELTA,
                    selection_method='maxmin', task='binary',
                    n_jobs=-1, seed=rep_seed, verbose=False,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=bc_names)

            imp_runs.append(m.global_importance_)

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        bc_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
        }
        log(f"  {name:<22} stability={stab:.4f}±{stab_se:.4f}")

    # IS plot and disagreement map from last DASH run
    fig = m.plot_importance_stability(title='IS Plot — Breast Cancer', annotate_top_k=8)
    fig.savefig(f"{OUT}/figures/is_plot_breast_cancer.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    var_obs = np.mean(m.variance_matrix_, axis=1)
    fig = local_disagreement_map(
        m.all_shap_matrices_, np.argmax(var_obs),
        feature_names=bc_names, top_k=12,
    )
    fig.savefig(f"{OUT}/figures/disagreement_breast_cancer.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: is_plot_breast_cancer.png, disagreement_breast_cancer.png")

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
    sc_methods = ['Single Best', 'Large Single Model', 'DASH (MaxMin)']
    sc_results = {}

    for name in sc_methods:
        imp_runs, rmse_runs = [], []
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
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
                imp = m.global_importance_
                rmse_val = rmse_score(y_sc_test, m.model_.predict(Xte_r))
            elif name == 'Large Single Model':
                m = LargeSingleModelBaseline(
                    K=SC_K, T_per_model=PAPER_CONFIG['T_PER_MODEL'],
                    colsample_bytree=0.2, seed=rep_seed,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
                imp = m.global_importance_
                rmse_val = rmse_score(y_sc_test, m.model_.predict(Xte_r))
            else:
                m = DASHPipeline(
                    M=SC_M, K=SC_K, epsilon=SC_EPSILON, delta=DELTA,
                    selection_method='maxmin', n_jobs=-1,
                    seed=rep_seed, verbose=False,
                )
                m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=sc_names)
                imp = m.global_importance_
                preds = m.get_consensus_ensemble_predictions(Xte_r)
                rmse_val = rmse_score(y_sc_test, preds)

            imp_runs.append(imp)
            rmse_runs.append(rmse_val)

        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        sc_results[name] = {
            'stability': stab,
            'stability_se': stab_se,
            'stability_ci_lo': stab_ci_lo,
            'stability_ci_hi': stab_ci_hi,
            'rmse_mean': float(np.mean(rmse_runs)),
            'rmse_std': float(np.std(rmse_runs, ddof=1)),
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  "
            f"RMSE={np.mean(rmse_runs):.2f}±{np.std(rmse_runs, ddof=1):.2f}")

    save_json(sc_results, f"{OUT}/tables/superconductor.json")
    elapsed = time.time() - t0
    log(f"  Superconductor completed in {elapsed/60:.1f} min")
    return sc_results


###############################################################################
# EXPERIMENT: Epsilon Sensitivity
###############################################################################

def experiment_epsilon_sensitivity():
    """Epsilon sensitivity sweep: epsilon ∈ {0.03, 0.05, 0.08, 0.10}."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Epsilon Sensitivity")
    log("=" * 70)

    from dash.core.population import generate_model_population

    EPS_VALUES = [0.03, 0.05, 0.08, 0.10]
    eps_results = {eps: {
        'n_passing': [], 'k_eff': [],
        'acc_runs': [], 'eq_runs': [], 'imp_runs': [],
    } for eps in EPS_VALUES}

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep+1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = \
            generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)

        for eps in EPS_VALUES:
            dm = DASHPipeline(
                M=M, K=K, epsilon=eps, delta=DELTA,
                selection_method='maxmin', n_jobs=-1,
                seed=rep_seed, verbose=False,
            )
            dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=FEATURE_NAMES)
            imp = dm.global_importance_

            r, _ = dgp_agreement(imp, true_imp)
            eps_results[eps]['n_passing'].append(len(dm.filtered_indices_))
            eps_results[eps]['k_eff'].append(len(dm.selected_indices_))
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

    ABL_N_REPS = 10  # C(10,2)=45 pairwise comparisons
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
                    dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=FEATURE_NAMES)
                    imp = dm.global_importance_
                    r, _ = dgp_agreement(imp, true_imp)
                    acc_runs.append(r)
                    imp_runs.append(imp)

                stab = importance_stability(imp_runs)
                abl_results[abl_rho][param_name][val] = {
                    'stability': stab,
                    'accuracy_mean': float(np.mean(acc_runs)),
                    'accuracy_std': float(np.std(acc_runs, ddof=1)),
                }
                log(f"    stab={stab:.4f}  acc={np.mean(acc_runs):.4f}")

    save_json(abl_results, f"{OUT}/tables/ablation.json")
    log(f"  Saved: {OUT}/tables/ablation.json")
    elapsed = time.time() - t0
    log(f"  Ablation completed in {elapsed/60:.1f} min")
    return abl_results


###############################################################################
# SUCCESS CRITERIA
###############################################################################

def check_success_criteria(sweep_results):
    """Evaluate pass/fail criteria against the linear sweep results."""
    log("\n" + "=" * 70)
    log("SUCCESS CRITERIA CHECK")
    log("=" * 70)

    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]

    # 1. Stability wins
    n_wins = sum(
        1 for rho in rho_levels
        if sweep_results[rho]['DASH (MaxMin)']['stability']
        > sweep_results[rho]['Single Best']['stability']
    )
    log(f"  1. Stability wins: {n_wins}/{len(rho_levels)} "
        f"({'PASS' if n_wins >= 4 else 'FAIL'}, need >=80%)")

    # 2. DGP agreement at rho=0.9 (relative to Single Best baseline)
    acc_09 = sweep_results[0.9]['DASH (MaxMin)']['accuracy_mean']
    sb_acc_09 = sweep_results[0.9]['Single Best']['accuracy_mean']
    log(f"  2. DGP agreement at ρ=0.9: DASH={acc_09:.4f} vs SB={sb_acc_09:.4f} "
        f"({'PASS' if acc_09 >= sb_acc_09 else 'check'}, DASH >= SB)")

    # 3. Equity wins
    n_eq_wins = sum(
        1 for rho in rho_levels
        if sweep_results[rho]['DASH (MaxMin)']['equity_mean']
        < sweep_results[rho]['Single Best']['equity_mean']
    )
    log(f"  3. Equity wins: {n_eq_wins}/{len(rho_levels)} "
        f"({'PASS' if n_eq_wins == len(rho_levels) else 'check'})")

    # 4. Safety control at rho=0
    rho0_dash = sweep_results[0.0]['DASH (MaxMin)']['accuracy_mean']
    rho0_sb = sweep_results[0.0]['Single Best']['accuracy_mean']
    log(f"  4. ρ=0 control: DASH dgp={rho0_dash:.4f}, SB dgp={rho0_sb:.4f} "
        f"({'PASS' if abs(rho0_dash - rho0_sb) < 0.1 else 'check'})")

    # 5. Independence value (DASH vs Large Single Model)
    if 'Large Single Model' in sweep_results[0.9]:
        n_lsm_wins = sum(
            1 for rho in rho_levels
            if 'Large Single Model' in sweep_results[rho]
            and sweep_results[rho]['DASH (MaxMin)']['stability']
            > sweep_results[rho]['Large Single Model']['stability']
        )
        log(f"  5. Independence value: DASH > LSM on {n_lsm_wins}/{len(rho_levels)} "
            f"({'PASS' if n_lsm_wins >= int(0.7 * len(rho_levels)) else 'check'}, need >=70%)")

    # Summary
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
}

# Default run order (all experiments)
DEFAULT_ORDER = [
    'linear_sweep',
    'overlapping',
    'nonlinear_sweep',
    'table2_baselines',
    'real_california',
    'real_breast_cancer',
    'real_superconductor',
    'epsilon_sensitivity',
    'ablation',
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
    log(f"  CAL_EPSILON={CAL_EPSILON}, BC_EPSILON={BC_EPSILON}, SC_EPSILON={SC_EPSILON}")
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
