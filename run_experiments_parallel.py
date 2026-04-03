#!/usr/bin/env python3
"""
DASH Experimental Validation — Parallel Optimized Runner
=========================================================
Performance-optimized fork of run_experiments.py.  Produces identical results
via population sharing, parallel SHAP, and vectorized stability computation.

All experimental design, seeds, hyperparameters, and statistical tests are
preserved exactly — only execution order and redundant computation change.

Run all experiments:
    python run_experiments_parallel.py

Run specific experiments:
    python run_experiments_parallel.py --experiments linear_sweep nonlinear_sweep
    python run_experiments_parallel.py --experiments real_california real_breast_cancer
    python run_experiments_parallel.py --experiments table2_baselines ablation

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
    variance_decomposition_crossed Exact ANOVA decomposition (7×7 crossed design)
    asymmetric_dgp         Asymmetric causal DGP: f0 causal, f1 passive correlate
    first_mover_visualization First-mover bias concentration figure
    first_mover_bias       First-mover bias isolation (concentration vs tree count)
    background_sensitivity Background set sensitivity analysis
    k_sweep_independence   K sweep: stability scaling vs ensemble size
    success_criteria       Run linear_sweep then evaluate pass/fail criteria
"""

import argparse
import os as _os

# Pin BLAS/OpenMP thread counts to 1 before numpy is imported.
# joblib + XGBoost nthread handle all parallelism; leaving these unset
# causes each worker to spawn O(cores) threads, producing severe oversubscription.
# Override with DASH_BLAS_THREADS if you need more (e.g. for pure-numpy workloads).
_blas_threads = _os.environ.get("DASH_BLAS_THREADS", "1")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_var, _blas_threads)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as rmse_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash_shap.core.pipeline import DASHPipeline
from dash_shap.core.consensus import compute_consensus
from dash_shap.core.diagnostics import compute_diagnostics
from dash_shap.core.diversity import (
    get_preliminary_importance,
    greedy_maxmin_selection,
    cluster_coverage_selection,
)
from dash_shap.core.diagnostics import (
    local_disagreement_map,
)
from dash_shap.experiments.synthetic import (
    generate_synthetic_linear,
    generate_synthetic_nonlinear,
    generate_synthetic_asymmetric,
)
from dash_shap.baselines import (
    SingleBestBaseline,
    LargeSingleModelBaseline,
    NaiveAveragingBaseline,
    StochasticRetrainBaseline,
    EnsembleSHAPBaseline,
    RandomSelectionBaseline,
    RandomForestBaseline,
    PermutationImportanceBaseline,
)

try:
    import lightgbm  # noqa: F401
    from dash_shap.baselines import LightGBMSingleBestBaseline

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False
from dash_shap.evaluation import (
    dgp_agreement,
    importance_accuracy,
    group_level_accuracy,
    group_level_mse,
    importance_stability,
    stability_bootstrap_ci,
    within_group_equity,
    topk_stability_bootstrap_ci,
    fsi_collinearity_correlation,
    compare_methods,
    cohens_d,
    holm_bonferroni,
    feature_ablation_score,
    tost_equivalence,
    bootstrap_stability_test,
    bootstrap_topk5_test,
    anova_decomposition,
)
from dash_shap.utils.io import save_json
from dash_shap.utils.provenance import (
    append_provenance_md,
    capture_run_meta,
    validate_result,
    write_environment_snapshot,
)
from dash_shap.utils.thread_budget import (
    compute_thread_budget,
    compute_rep_worker_budget,
    get_available_cores,
)
from dash_shap.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    clear_checkpoint,
    clear_checkpoints_by_prefix,
    _sanitize_ckpt_name,
)

# ---------------------------------------------------------------------------
# Canonical configuration — single source of truth in dash_shap.config
# ---------------------------------------------------------------------------
from dash_shap.config import PAPER_CONFIG, SEED, REAL_EPSILON, REAL_EPSILON_MODE
from dash_shap.experiments.schemas import (
    AsymmetricRhoMethodResult,
    KSweepMethodResult,
    VarianceDecompositionMethodResult,
)

M = PAPER_CONFIG["M"]
K = PAPER_CONFIG["K"]
N_REPS = PAPER_CONFIG["N_REPS"]
EPSILON = PAPER_CONFIG["EPSILON"]
DELTA = PAPER_CONFIG["DELTA"]
N_TRIALS_SB = PAPER_CONFIG["N_TRIALS_SB"]

OUT = "results"
CKPT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), OUT, "checkpoints")

# Checkpoint wrappers — bake in CKPT_DIR and PAPER_CONFIG so all call sites
# can use _save/_load/_has without repeating checkpoint_dir and config args.
_CKPT_CONFIG = PAPER_CONFIG  # canonical; frozen at import time


def _save(name: str, **data) -> None:
    save_checkpoint(name, checkpoint_dir=CKPT_DIR, config=_CKPT_CONFIG, **data)


def _load(name: str):
    return load_checkpoint(name, checkpoint_dir=CKPT_DIR, config=_CKPT_CONFIG)


def _has(name: str) -> bool:
    return has_checkpoint(name, checkpoint_dir=CKPT_DIR)


def _publish_results(data, path, experiment_name, n_reps, t0):
    """Validate, attach provenance, save JSON, and append PROVENANCE.md."""
    elapsed = time.time() - t0
    issues = validate_result(data, experiment_name)
    for issue in issues:
        log(f"  WARNING: {experiment_name}: {issue}")
    meta = capture_run_meta(experiment_name, n_reps, PAPER_CONFIG, elapsed, path)
    if meta.get("code_dirty"):
        log(f"  WARNING: code_dirty=True for {experiment_name} — results from uncommitted changes")
    save_json(data, path, meta=meta, overwrite_protection=True)
    append_provenance_md(meta, OUT)
    log(f"  Saved: {path}")


def make_feature_names(n_groups=10, group_size=5):
    """Generate feature names matching the DGP structure (m6 fix)."""
    return [f"G{g}_f{j}" for g in range(n_groups) for j in range(group_size)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dirs():
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    os.makedirs(f"{OUT}/tables", exist_ok=True)


def _shutdown_loky_workers():
    """Shut down the reusable loky executor to reclaim worker memory.

    After a Parallel() call returns, loky keeps worker processes alive for
    reuse.  In long-running real-world experiments this leaks memory (workers
    hold copies of large datasets).  Calling this between experiments releases
    those resources so subsequent experiments can use the full core budget.
    """
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=False)
    except Exception:
        pass  # non-critical; loky may not be available in all environments


###############################################################################
# PLOTTING
###############################################################################

COLORS = {
    "Single Best": "#95a5a6",
    "Single Best (M=200)": "#7f8c8d",
    "Large Single Model": "#e74c3c",
    "LSM (Tuned)": "#c0392b",
    "Ensemble SHAP": "#9b59b6",
    "Naive Top-N": "#f39c12",
    "Stochastic Retrain": "#e67e22",
    "Random Selection": "#d4ac0d",
    "DASH (Dedup)": "#3498db",
    "DASH (MaxMin)": "#2ecc71",
    "DASH (Cluster)": "#1abc9c",
    "Random Forest": "#16a085",
    "Permutation Importance": "#8e44ad",
}

MARKERS = {
    "Single Best": "s",
    "Single Best (M=200)": "*",
    "Large Single Model": "X",
    "LSM (Tuned)": "x",
    "Ensemble SHAP": "D",
    "Naive Top-N": "^",
    "Stochastic Retrain": "v",
    "Random Selection": "d",
    "DASH (MaxMin)": "o",
    "DASH (Cluster)": "P",
    "Random Forest": "H",
    "Permutation Importance": "p",
}


def plot_correlation_sweep(all_results, rho_levels, method_names):
    """Generate main result figures from the correlation sweep."""
    _ensure_dirs()
    plot_methods = [n for n in MARKERS if n in method_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for name in plot_methods:
        c = COLORS.get(name, "#333")
        m = MARKERS.get(name, "o")

        vals = [all_results[rho][name]["stability"] for rho in rho_levels]
        axes[0].plot(rho_levels, vals, "-", marker=m, color=c, label=name, linewidth=2, markersize=7)

        vals = [all_results[rho][name]["accuracy_mean"] for rho in rho_levels]
        errs = [all_results[rho][name]["accuracy_std"] for rho in rho_levels]
        axes[1].errorbar(
            rho_levels,
            vals,
            yerr=errs,
            fmt="-",
            marker=m,
            color=c,
            label=name,
            linewidth=2,
            markersize=7,
            capsize=3,
        )

        vals = [all_results[rho][name]["equity_mean"] for rho in rho_levels]
        errs = [all_results[rho][name]["equity_std"] for rho in rho_levels]
        axes[2].errorbar(
            rho_levels,
            vals,
            yerr=errs,
            fmt="-",
            marker=m,
            color=c,
            label=name,
            linewidth=2,
            markersize=7,
            capsize=3,
        )

    axes[0].set_xlabel("Within-Group Correlation ρ")
    axes[0].set_ylabel("Importance Stability\n(Mean Pairwise Spearman)")
    axes[0].set_title("Stability vs. Collinearity")
    axes[0].legend(fontsize=7, loc="lower left")

    axes[1].set_xlabel("Within-Group Correlation ρ")
    axes[1].set_ylabel("Spearman ρ vs Ground Truth")
    axes[1].set_title("Accuracy vs. Collinearity")

    axes[2].set_xlabel("Within-Group Correlation ρ")
    axes[2].set_ylabel("Mean Within-Group CV\n(lower = better)")
    axes[2].set_title("Within-Group Equity vs. Collinearity")

    fig.suptitle("DASH vs Baselines — Synthetic Linear DGP", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/correlation_sweep.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/correlation_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    log("  Saved: figures/correlation_sweep.png, correlation_sweep.pdf")

    # Bar chart for rho=0.9
    if 0.9 in all_results:
        rho_key = 0.9
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        bar_methods = list(all_results[rho_key].keys())
        bar_colors = [COLORS.get(n, "#333") for n in bar_methods]

        stab_vals = [all_results[rho_key][n]["stability"] for n in bar_methods]
        axes[0].bar(range(len(bar_methods)), stab_vals, color=bar_colors, edgecolor="k", linewidth=0.5)
        axes[0].set_xticks(range(len(bar_methods)))
        axes[0].set_xticklabels(bar_methods, rotation=35, ha="right", fontsize=8)
        axes[0].set_ylabel("Stability")
        axes[0].set_title("Importance Stability (ρ=0.9)")

        acc_vals = [all_results[rho_key][n]["accuracy_mean"] for n in bar_methods]
        acc_errs = [all_results[rho_key][n]["accuracy_std"] for n in bar_methods]
        axes[1].bar(
            range(len(bar_methods)),
            acc_vals,
            yerr=acc_errs,
            color=bar_colors,
            edgecolor="k",
            linewidth=0.5,
            capsize=3,
        )
        axes[1].set_xticks(range(len(bar_methods)))
        axes[1].set_xticklabels(bar_methods, rotation=35, ha="right", fontsize=8)
        axes[1].set_ylabel("Accuracy (Spearman ρ)")
        axes[1].set_title("Importance Accuracy (ρ=0.9)")

        eq_vals = [all_results[rho_key][n]["equity_mean"] for n in bar_methods]
        eq_errs = [all_results[rho_key][n]["equity_std"] for n in bar_methods]
        axes[2].bar(
            range(len(bar_methods)),
            eq_vals,
            yerr=eq_errs,
            color=bar_colors,
            edgecolor="k",
            linewidth=0.5,
            capsize=3,
        )
        axes[2].set_xticks(range(len(bar_methods)))
        axes[2].set_xticklabels(bar_methods, rotation=35, ha="right", fontsize=8)
        axes[2].set_ylabel("Within-Group CV")
        axes[2].set_title("Equity (ρ=0.9, lower=better)")

        fig.tight_layout()
        fig.savefig(f"{OUT}/figures/bar_chart_rho09.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{OUT}/figures/bar_chart_rho09.pdf", bbox_inches="tight")
        plt.close(fig)
        log("  Saved: figures/bar_chart_rho09.png, bar_chart_rho09.pdf")


###############################################################################
# PLOT: Nonlinear sweep
###############################################################################


def plot_nonlinear_sweep(nl_results, rho_levels, method_names):
    """Line plot: stability vs rho for each method (nonlinear DGP)."""
    _ensure_dirs()
    plot_methods = [n for n in MARKERS if n in method_names]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in plot_methods:
        c = COLORS.get(name, "#333")
        m = MARKERS.get(name, "o")
        vals = []
        errs = []
        xs = []
        for rho in rho_levels:
            entry = nl_results.get(rho, {}).get(name)
            if entry is None:
                continue
            xs.append(rho)
            vals.append(entry["stability"])
            errs.append(entry.get("stability_se", 0.0))
        if xs:
            ax.errorbar(
                xs,
                vals,
                yerr=errs,
                fmt="-",
                marker=m,
                color=c,
                label=name,
                linewidth=2,
                markersize=7,
                capsize=3,
            )

    ax.set_xlabel("Within-Group Correlation \u03c1")
    ax.set_ylabel("Importance Stability\n(Mean Pairwise Spearman)")
    ax.set_title("DASH vs Baselines \u2014 Synthetic Nonlinear DGP")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/nonlinear_sweep.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/nonlinear_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {OUT}/figures/nonlinear_sweep.png, nonlinear_sweep.pdf")


###############################################################################
# PLOT: Ablation sensitivity
###############################################################################


def plot_ablation_sensitivity(ablation_results):
    """Generate 2x2 ablation sensitivity figure (M, K, epsilon, delta)."""
    _ensure_dirs()

    param_configs = [
        ("M", [50, 100, 200, 500], "Population Size $M$"),
        ("K", [5, 10, 20, 30, 50], "Selected Models $K$"),
        ("epsilon", [0.01, 0.03, 0.05, 0.08, 0.10], "Filter Threshold $\\varepsilon$"),
        ("delta", [0.01, 0.05, 0.10, 0.20], "Diversity Threshold $\\delta$"),
    ]
    rho_styles = {
        0.0: ("--", "#7f8c8d", "o", "$\\rho=0.0$"),
        0.9: ("-", "#2980b9", "s", "$\\rho=0.9$"),
        0.95: ("-", "#e74c3c", "^", "$\\rho=0.95$"),
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
                    y_vals.append(param_data[v_key].get("stability", 0))
            if x_vals:
                ax.plot(x_vals, y_vals, f"{marker}{ls}", color=color, label=label, linewidth=2, markersize=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Stability")
        ax.set_title(f"{param_name} Sensitivity")
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("DASH Hyperparameter Sensitivity", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/ablation_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/ablation_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("  Saved: figures/ablation_sensitivity.pdf")


###############################################################################
# PLOT: Epsilon sensitivity
###############################################################################


def plot_epsilon_sensitivity(eps_results):
    """Line plot: stability vs epsilon threshold."""
    _ensure_dirs()
    eps_values = sorted(k for k in eps_results if not str(k).startswith("_"))
    stab_vals = [eps_results[eps].get("stability", float("nan")) for eps in eps_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        eps_values,
        stab_vals,
        "-o",
        color=COLORS.get("DASH (MaxMin)", "#2ecc71"),
        linewidth=2,
        markersize=7,
        label="DASH (MaxMin)",
    )
    ax.set_xlabel("Filter Threshold \u03b5")
    ax.set_ylabel("Importance Stability\n(Mean Pairwise Spearman)")
    ax.set_title("DASH Stability vs. Epsilon Filter Threshold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/epsilon_sensitivity.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/epsilon_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {OUT}/figures/epsilon_sensitivity.png, epsilon_sensitivity.pdf")


###############################################################################
# PLOT: Table 2 baselines bar chart
###############################################################################


def plot_table2_baselines(table2_results):
    """Horizontal bar chart comparing Table 2 methods on stability, accuracy, equity."""
    _ensure_dirs()
    method_names = [k for k in table2_results if not str(k).startswith("_")]

    stab_vals = [table2_results[n].get("stability", float("nan")) for n in method_names]
    stab_errs = [table2_results[n].get("stability_se", 0.0) for n in method_names]
    acc_vals = [table2_results[n].get("accuracy_mean", float("nan")) for n in method_names]
    acc_errs = [table2_results[n].get("accuracy_std", 0.0) for n in method_names]
    eq_vals = [table2_results[n].get("equity_mean", float("nan")) for n in method_names]
    eq_errs = [table2_results[n].get("equity_std", 0.0) for n in method_names]

    bar_colors = [COLORS.get(n, "#333") for n in method_names]
    y_pos = range(len(method_names))

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(method_names) * 0.7)))

    axes[0].barh(list(y_pos), stab_vals, xerr=stab_errs, color=bar_colors, edgecolor="k", linewidth=0.5, capsize=3)
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(method_names, fontsize=8)
    axes[0].set_xlabel("Stability")
    axes[0].set_title("Importance Stability (\u03c1=0.9)")

    axes[1].barh(list(y_pos), acc_vals, xerr=acc_errs, color=bar_colors, edgecolor="k", linewidth=0.5, capsize=3)
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(method_names, fontsize=8)
    axes[1].set_xlabel("Accuracy (Spearman \u03c1)")
    axes[1].set_title("Importance Accuracy (\u03c1=0.9)")

    axes[2].barh(list(y_pos), eq_vals, xerr=eq_errs, color=bar_colors, edgecolor="k", linewidth=0.5, capsize=3)
    axes[2].set_yticks(list(y_pos))
    axes[2].set_yticklabels(method_names, fontsize=8)
    axes[2].set_xlabel("Within-Group CV (lower=better)")
    axes[2].set_title("Within-Group Equity (\u03c1=0.9)")

    fig.suptitle("Table 2: Extended Baselines at \u03c1=0.9", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/table2_baselines.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/table2_baselines.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {OUT}/figures/table2_baselines.png, table2_baselines.pdf")


###############################################################################
# PLOT: Real-world datasets bar chart
###############################################################################


def plot_real_world_bar(results, dataset_name):
    """Horizontal bar chart: methods vs stability for a real-world dataset.

    dataset_name is used in the title and file stem:
        california_housing  → california_housing_results.{png,pdf}
        breast_cancer       → breast_cancer_results.{png,pdf}
        superconductor      → superconductor_results.{png,pdf}
    """
    _ensure_dirs()
    method_names = [k for k in results if not str(k).startswith("_")]
    stab_vals = [results[n].get("stability", float("nan")) for n in method_names]
    stab_errs = [results[n].get("stability_se", 0.0) for n in method_names]
    bar_colors = [COLORS.get(n, "#333") for n in method_names]
    y_pos = range(len(method_names))

    fig, ax = plt.subplots(figsize=(8, max(4, len(method_names) * 0.7)))
    ax.barh(
        list(y_pos),
        stab_vals,
        xerr=stab_errs,
        color=bar_colors,
        edgecolor="k",
        linewidth=0.5,
        capsize=3,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(method_names, fontsize=9)
    ax.set_xlabel("Importance Stability (Mean Pairwise Spearman)")
    title_name = dataset_name.replace("_", " ").title()
    ax.set_title(f"DASH vs Baselines \u2014 {title_name}")
    fig.tight_layout()
    stem = f"{dataset_name}_results"
    fig.savefig(f"{OUT}/figures/{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {OUT}/figures/{stem}.png, {stem}.pdf")


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
        for metric in ("rmse_runs", "ablation_runs"):
            if metric not in dash or metric not in bl:
                continue
            label = metric.replace("_runs", "")
            _, pval = compare_methods(dash[metric], bl[metric])
            d = cohens_d(dash[metric], bl[metric])
            log(f"  {bname:<22} {label:<12} {pval:>12.4g} {d:>10.3f}")
            sig_results[bname][label] = {"p": pval, "cohens_d": d}
        # TOST equivalence test for DASH vs Stochastic Retrain
        if bname == "Stochastic Retrain":
            for metric in ("rmse_runs", "ablation_runs"):
                if metric not in dash or metric not in bl:
                    continue
                label = metric.replace("_runs", "")
                _, p1, _, p2, equiv = tost_equivalence(
                    dash[metric],
                    bl[metric],
                )
                log(f"  {bname:<22} {label:<12} TOST equiv={'YES' if equiv else 'no'}  p_max={max(p1, p2):.4g}")
                sig_results[bname][f"{label}_tost"] = {
                    "p1": p1,
                    "p2": p2,
                    "equivalent": equiv,
                }
    # Apply Holm-Bonferroni correction to all raw Wilcoxon p-values
    raw_keys = []
    raw_pvals = []
    for bname in sig_results:
        for label in sig_results[bname]:
            if "_tost" not in label and "p" in sig_results[bname][label]:
                raw_keys.append((bname, label))
                raw_pvals.append(sig_results[bname][label]["p"])
    if raw_pvals:
        adjusted = holm_bonferroni(np.array(raw_pvals))
        for (bname, label), adj_p in zip(raw_keys, adjusted):
            sig_results[bname][label]["p_holm"] = float(adj_p)
        log("\n  Holm-Bonferroni corrected p-values:")
        for (bname, label), adj_p in zip(raw_keys, adjusted):
            log(f"    {bname:<22} {label:<12} p_HB={adj_p:.4g}")
    # Bootstrap stability hypothesis test (REVIEW_v7 M3)
    if "imp_runs" in dash:
        log(f"\n  Bootstrap stability tests ({dataset_label}):")
        log(f"  {'Baseline':<22} {'Δ Stab':>10} {'p-value':>10} {'95% CI':>22}")
        log("  " + "-" * 70)
        for bname in baselines:
            bl = results[bname]
            if "imp_runs" not in bl:
                continue
            try:
                diff, pval, ci_lo, ci_hi = bootstrap_stability_test(dash["imp_runs"], bl["imp_runs"])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                log(f"  {bname:<22} {diff:+10.4f} {pval:10.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")
                sig_results[bname]["stability"] = {
                    "diff": float(diff),
                    "p": float(pval),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
            except Exception as e:
                log(f"  {bname:<22} SKIP ({e})")
        # Top-k5 bootstrap test
        log(f"\n  Bootstrap top-k5 tests ({dataset_label}):")
        log(f"  {'Baseline':<22} {'Δ TopK5':>10} {'p-value':>10} {'95% CI':>22}")
        log("  " + "-" * 70)
        for bname in baselines:
            bl = results[bname]
            if "imp_runs" not in bl:
                continue
            try:
                diff, pval, ci_lo, ci_hi = bootstrap_topk5_test(dash["imp_runs"], bl["imp_runs"])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                log(f"  {bname:<22} {diff:+10.4f} {pval:10.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")
                sig_results[bname]["topk5"] = {
                    "diff": float(diff),
                    "p": float(pval),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
            except Exception as e:
                log(f"  {bname:<22} SKIP ({e})")

    # Store in results for JSON serialisation
    results["_significance"] = sig_results


###############################################################################
# HELPER: Collect per-rep results (avoids repetition in rep-outer loop)
###############################################################################


def _collect_rep(md, imp, true_imp, grps, rmse_val, model_obj):
    """Accumulate per-rep metrics into method_data dict."""
    r, _ = importance_accuracy(imp, true_imp)
    md["acc_runs"].append(r)
    md["gacc_runs"].append(group_level_accuracy(imp, true_imp, grps))
    md["gmse_runs"].append(group_level_mse(imp, true_imp, grps))
    md["eq_runs"].append(within_group_equity(imp, grps))
    md["imp_runs"].append(imp)
    md["rmse_runs"].append(rmse_val)
    if hasattr(model_obj, "selected_indices_") and model_obj.selected_indices_ is not None:
        md["keff_runs"].append(len(model_obj.selected_indices_))


def _init_method_data(sweep_methods):
    """Return a fresh method_data accumulator dict for the given method names."""
    return {
        name: {
            "acc_runs": [],
            "eq_runs": [],
            "imp_runs": [],
            "rmse_runs": [],
            "gacc_runs": [],
            "gmse_runs": [],
            "keff_runs": [],
            "t_accum": 0.0,
        }
        for name in sweep_methods
    }


def _rep_metrics(imp, true_imp, grps, rmse_val, model_obj, elapsed):
    """Extract per-rep metrics as a plain dict. model_obj is queried but NOT stored."""
    r, _ = importance_accuracy(imp, true_imp)
    keff = None
    if hasattr(model_obj, "selected_indices_") and model_obj.selected_indices_ is not None:
        keff = len(model_obj.selected_indices_)
    return {
        "acc": r,
        "eq": within_group_equity(imp, grps),
        "imp": imp,  # numpy array only
        "rmse": rmse_val,
        "gacc": group_level_accuracy(imp, true_imp, grps),
        "gmse": group_level_mse(imp, true_imp, grps),
        "keff": keff,
        "t": elapsed,
    }


def _merge_rep(method_data, per_method):
    """Merge one rep's result dict into accumulated method_data (in-place)."""
    for name, m in per_method.items():
        if name not in method_data:
            continue  # graceful skip for methods added/removed between runs
        md = method_data[name]
        md["acc_runs"].append(m["acc"])
        md["eq_runs"].append(m["eq"])
        md["imp_runs"].append(m["imp"])
        md["rmse_runs"].append(m["rmse"])
        md["gacc_runs"].append(m["gacc"])
        md["gmse_runs"].append(m["gmse"])
        if m["keff"] is not None:
            md["keff_runs"].append(m["keff"])
        md["t_accum"] += m["t"]


def _aggregate_method_data(method_data, sweep_methods):
    """Compute per-method statistics from accumulated rep lists.

    Extracted from _run_single_rho for reuse in parallel rep mode.
    Returns a rho_results dict with the same schema as _run_single_rho.
    """
    rho_results = {}
    for name in sweep_methods:
        md = method_data[name]
        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(md["imp_runs"])
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(md["imp_runs"], k=5)
        n_reps = len(md["acc_runs"])
        rho_results[name] = {
            "stability": stab,
            "stability_se": stab_se,
            "stability_ci_lo": stab_ci_lo,
            "stability_ci_hi": stab_ci_hi,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "k_eff_mean": float(np.mean(md["keff_runs"])) if md["keff_runs"] else None,
            "k_eff_std": float(np.std(md["keff_runs"], ddof=1)) if len(md["keff_runs"]) > 1 else None,
            "accuracy_mean": np.mean(md["acc_runs"]),
            "accuracy_se": np.std(md["acc_runs"], ddof=1) / np.sqrt(n_reps),
            "accuracy_std": np.std(md["acc_runs"], ddof=1),
            "group_accuracy_mean": np.mean(md["gacc_runs"]),
            "group_accuracy_std": np.std(md["gacc_runs"], ddof=1),
            "group_mse_mean": np.mean(md["gmse_runs"]),
            "group_mse_std": np.std(md["gmse_runs"], ddof=1),
            "equity_mean": np.mean(md["eq_runs"]),
            "equity_se": np.std(md["eq_runs"], ddof=1) / np.sqrt(n_reps),
            "equity_std": np.std(md["eq_runs"], ddof=1),
            "rmse_mean": np.mean(md["rmse_runs"]),
            "rmse_std": np.std(md["rmse_runs"], ddof=1),
            "elapsed_s": md["t_accum"],
            # Save per-rep arrays for significance tests
            "acc_runs": np.array(md["acc_runs"]),
            "eq_runs": np.array(md["eq_runs"]),
            "rmse_runs": np.array(md["rmse_runs"]),
            "imp_runs": md["imp_runs"],
        }
        log(
            f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  "
            f"acc={np.mean(md['acc_runs']):.4f}  gacc={np.mean(md['gacc_runs']):.4f}  gmse={np.mean(md['gmse_runs']):.6f}  "
            f"eq={np.mean(md['eq_runs']):.4f}  RMSE={np.mean(md['rmse_runs']):.4f}  "
            f"({md['t_accum']:.1f}s)"
        )
    return rho_results


###############################################################################
# EXPERIMENT: Synthetic Linear — Correlation Sweep (PARALLEL OPTIMIZED)
###############################################################################


def _run_single_rep(rho, rep, sweep_methods, feature_names, *, nthread=1):
    """Run all 9 methods for one (rho, rep) pair.

    No inner joblib — all trained models explicitly deleted before return so that
    loky worker processes inherit no model state across rep boundaries.

    Returns
    -------
    tuple : (rep, per_method_dict, fsi_or_none, grps)
        per_method_dict keys: acc, eq, imp, rmse, gacc, gmse, keff, t
        All values are scalars or numpy arrays — NO model object references.
    """
    rep_seed = SEED + rep
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
        N=5000, rho=rho, seed=rep_seed
    )

    per_method: dict = {}
    fsi_out = None

    # ── Phase 1: population-dependent methods ─────────────────────────────
    # DASH builds M=200 models; three baselines reuse via fit_from_population.
    # Deletion order is mandatory (see ref-count analysis in plan).

    t0 = time.time()
    dash_pipeline = DASHPipeline(
        M=M,
        K=K,
        epsilon=EPSILON,
        delta=DELTA,
        selection_method="maxmin",
        n_jobs=1,
        nthread=nthread,
        seed=rep_seed,
        verbose=False,
    )
    dash_pipeline.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
    imp = dash_pipeline.global_importance_
    rmse_val = rmse_score(yte, dash_pipeline.get_consensus_ensemble_predictions(Xte))
    per_method["DASH (MaxMin)"] = _rep_metrics(imp, true_imp, grps, rmse_val, dash_pipeline, time.time() - t0)
    if hasattr(dash_pipeline, "fsi_") and dash_pipeline.fsi_ is not None:
        fsi_out = dash_pipeline.fsi_.copy()

    t0 = time.time()
    sb200 = SingleBestBaseline(n_trials=M, seed=rep_seed, n_jobs=1, nthread=nthread)
    sb200.fit_from_population(dash_pipeline.models_, dash_pipeline.val_scores_, Xexp, seed=rep_seed)
    per_method["Single Best (M=200)"] = _rep_metrics(
        sb200.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, sb200.model_.predict(Xte)),
        sb200,
        time.time() - t0,
    )
    del sb200  # releases 1 best-model ref (does not hold population list)

    t0 = time.time()
    rs = RandomSelectionBaseline(
        M=M,
        K=K,
        epsilon=EPSILON,
        delta=DELTA,
        n_jobs=1,
        nthread=nthread,
        seed=rep_seed,
        verbose=False,
    )
    rs.fit_from_population(
        dash_pipeline.models_,
        dash_pipeline.val_scores_,
        Xexp,
        feature_names=feature_names,
    )
    per_method["Random Selection"] = _rep_metrics(
        rs.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, rs.get_consensus_ensemble_predictions(Xte)),
        rs,
        time.time() - t0,
    )
    del rs  # releases self.models_ ref → population ref count: 3→2

    t0 = time.time()
    naive = NaiveAveragingBaseline(N=K, task="regression")
    naive.fit_from_population(dash_pipeline.models_, dash_pipeline.val_scores_, Xexp)
    per_method["Naive Top-N"] = _rep_metrics(
        naive.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, naive.get_consensus_ensemble_predictions(Xte)),
        naive,
        time.time() - t0,
    )
    del naive  # releases self.models_ aliased ref → ref count: 2→1

    del dash_pipeline  # ← ref count hits 0; all 200 XGBoost models freed HERE

    # ── Phase 2: independent methods (population gone) ────────────────────

    t0 = time.time()
    sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=1, nthread=nthread)
    sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
    per_method["Single Best"] = _rep_metrics(
        sb.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, sb.model_.predict(Xte)),
        sb,
        time.time() - t0,
    )
    del sb

    t0 = time.time()
    lsm = LargeSingleModelBaseline(
        K=K,
        T_per_model=PAPER_CONFIG["T_PER_MODEL"],
        colsample_bytree=0.2,
        seed=rep_seed,
        tune=False,
        nthread=nthread,
    )
    lsm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
    per_method["Large Single Model"] = _rep_metrics(
        lsm.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, lsm.model_.predict(Xte)),
        lsm,
        time.time() - t0,
    )
    del lsm

    t0 = time.time()
    lsm_t = LargeSingleModelBaseline(
        K=K,
        T_per_model=PAPER_CONFIG["T_PER_MODEL"],
        colsample_bytree=0.2,
        seed=rep_seed,
        tune=True,
        nthread=nthread,
    )
    lsm_t.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
    per_method["LSM (Tuned)"] = _rep_metrics(
        lsm_t.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, lsm_t.model_.predict(Xte)),
        lsm_t,
        time.time() - t0,
    )
    del lsm_t

    t0 = time.time()
    sr = StochasticRetrainBaseline(N=K, task="regression", n_jobs=1, nthread=nthread, seed=rep_seed)
    sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
    per_method["Stochastic Retrain"] = _rep_metrics(
        sr.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, sr.get_consensus_ensemble_predictions(Xte)),
        sr,
        time.time() - t0,
    )
    del sr

    t0 = time.time()
    rf = RandomForestBaseline(n_estimators=500, task="regression", seed=rep_seed)
    rf.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
    per_method["Random Forest"] = _rep_metrics(
        rf.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, rf.model_.predict(Xte)),
        rf,
        time.time() - t0,
    )
    del rf

    return rep, per_method, fsi_out, grps


def _run_single_rho(rho, sweep_methods, feature_names, n_jobs_inner, *, nthread=1):
    """Run all reps for a single rho level. Returns (rho, rho_results, fsi_list, grps).

    Designed to be called in parallel across rho levels via joblib.
    n_jobs_inner controls parallelism within each method (SHAP, etc.).
    """
    log(f"\n--- ρ = {rho} ---")

    method_data = _init_method_data(sweep_methods)
    fsi_list = []
    grps_last = None

    for rep in range(N_REPS):
        rep_seed = SEED + rep
        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
            N=5000, rho=rho, seed=rep_seed
        )
        grps_last = grps

        # --- DASH (MaxMin): trains the M=200 population ---
        t_m = time.time()
        dash_pipeline = DASHPipeline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            selection_method="maxmin",
            n_jobs=n_jobs_inner,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        dash_pipeline.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        imp = dash_pipeline.global_importance_
        preds = dash_pipeline.get_consensus_ensemble_predictions(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["DASH (MaxMin)"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["DASH (MaxMin)"], imp, true_imp, grps, rmse_val, dash_pipeline)
        if hasattr(dash_pipeline, "fsi_") and dash_pipeline.fsi_ is not None:
            fsi_list.append(dash_pipeline.fsi_.copy())

        # --- SingleBest(M=200): reuse DASH population ---
        t_m = time.time()
        sb200 = SingleBestBaseline(n_trials=M, seed=rep_seed, n_jobs=n_jobs_inner, nthread=nthread)
        sb200.fit_from_population(
            dash_pipeline.models_,
            dash_pipeline.val_scores_,
            Xexp,
            seed=rep_seed,
        )
        imp = sb200.global_importance_
        preds = sb200.model_.predict(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Single Best (M=200)"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Single Best (M=200)"], imp, true_imp, grps, rmse_val, sb200)

        # --- RandomSelection: reuse DASH population ---
        t_m = time.time()
        rs = RandomSelectionBaseline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            n_jobs=n_jobs_inner,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        rs.fit_from_population(
            dash_pipeline.models_,
            dash_pipeline.val_scores_,
            Xexp,
            feature_names=feature_names,
        )
        imp = rs.global_importance_
        preds = rs.get_consensus_ensemble_predictions(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Random Selection"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Random Selection"], imp, true_imp, grps, rmse_val, rs)

        # --- Naive Top-N: reuse DASH population ---
        t_m = time.time()
        naive = NaiveAveragingBaseline(N=K, task="regression")
        naive.fit_from_population(
            dash_pipeline.models_,
            dash_pipeline.val_scores_,
            Xexp,
        )
        imp = naive.global_importance_
        preds = naive.get_consensus_ensemble_predictions(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Naive Top-N"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Naive Top-N"], imp, true_imp, grps, rmse_val, naive)

        # --- Single Best (n_trials=30): independent ---
        t_m = time.time()
        sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=n_jobs_inner, nthread=nthread)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = sb.global_importance_
        preds = sb.model_.predict(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Single Best"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Single Best"], imp, true_imp, grps, rmse_val, sb)

        # --- Large Single Model ---
        t_m = time.time()
        lsm = LargeSingleModelBaseline(
            K=K,
            T_per_model=PAPER_CONFIG["T_PER_MODEL"],
            colsample_bytree=0.2,
            seed=rep_seed,
            tune=False,
            nthread=nthread,
        )
        lsm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = lsm.global_importance_
        preds = lsm.model_.predict(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Large Single Model"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Large Single Model"], imp, true_imp, grps, rmse_val, lsm)

        # --- LSM (Tuned) ---
        t_m = time.time()
        lsm_t = LargeSingleModelBaseline(
            K=K,
            T_per_model=PAPER_CONFIG["T_PER_MODEL"],
            colsample_bytree=0.2,
            seed=rep_seed,
            tune=True,
            nthread=nthread,
        )
        lsm_t.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = lsm_t.global_importance_
        preds = lsm_t.model_.predict(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["LSM (Tuned)"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["LSM (Tuned)"], imp, true_imp, grps, rmse_val, lsm_t)

        # --- Stochastic Retrain ---
        t_m = time.time()
        sr = StochasticRetrainBaseline(
            N=K,
            task="regression",
            n_jobs=n_jobs_inner,
            nthread=nthread,
            seed=rep_seed,
        )
        sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = sr.global_importance_
        preds = sr.get_consensus_ensemble_predictions(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Stochastic Retrain"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Stochastic Retrain"], imp, true_imp, grps, rmse_val, sr)

        # --- Random Forest ---
        t_m = time.time()
        rf = RandomForestBaseline(
            n_estimators=500,
            task="regression",
            seed=rep_seed,
        )
        rf.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = rf.global_importance_
        preds = rf.model_.predict(Xte)
        rmse_val = rmse_score(yte, preds)
        method_data["Random Forest"]["t_accum"] += time.time() - t_m
        _collect_rep(method_data["Random Forest"], imp, true_imp, grps, rmse_val, rf)

    # Aggregate results per method
    rho_results = _aggregate_method_data(method_data, sweep_methods)

    # Checkpoint per-rho results
    save_checkpoint(
        f"linear_sweep_rho_{rho}",
        checkpoint_dir=CKPT_DIR,
        rho_results=rho_results,
        fsi_list=fsi_list,
        grps=grps_last,
    )

    return rho, rho_results, fsi_list, grps_last


def experiment_linear_sweep(resume=False, cleanup=False, sequential=False):
    """Canonical correlation sweep: rho ∈ {0.0, 0.5, 0.7, 0.9, 0.95}.

    Parallel-optimized: rep-outer loop with population sharing between
    DASH, RandomSelection, SingleBest(M=200), and NaiveTop-N.

    When sequential=False (default): flattens all (rho, rep) pairs into a
    single Parallel call — up to N_REPS * len(rho_levels) concurrent workers.
    Uses per-rep checkpointing for crash resilience.

    When sequential=True: runs rho levels in parallel (existing behavior) and
    reps sequentially within each rho. Useful for validation or single-core
    machines.

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
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("=" * 70)
    log("EXPERIMENT: Synthetic Linear DGP — Correlation Sweep (Parallel)")
    log("=" * 70)

    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    sweep_methods = [
        "Single Best",
        "Single Best (M=200)",  # M3: matched budget
        "Large Single Model",
        "LSM (Tuned)",
        "Stochastic Retrain",
        "Random Selection",  # M2: isolate MaxMin value
        "Random Forest",
        "DASH (MaxMin)",
        "Naive Top-N",  # Ablation: averaging without diversity selection (reuses DASH population)
    ]
    sweep_results = {rho: {} for rho in rho_levels}
    feature_names = make_feature_names()  # m6: dynamic
    fsi_by_rho = {}  # FSI validation: collect DASH FSI per rho (reviewer #7)
    grps_by_rho = {}  # Groups array per rho (deterministic, saved once)

    # Separate resumed vs pending rho levels
    pending_rhos = []
    for rho in rho_levels:
        ckpt_name = f"linear_sweep_rho_{rho}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for ρ={rho}")
            ckpt_data = load_checkpoint(ckpt_name, CKPT_DIR)
            sweep_results[rho] = ckpt_data["rho_results"]
            if "fsi_list" in ckpt_data:
                fsi_by_rho[rho] = ckpt_data["fsi_list"]
            if "grps" in ckpt_data:
                grps_by_rho[rho] = ckpt_data["grps"]
        else:
            pending_rhos.append(rho)

    if pending_rhos:
        if sequential:
            # ── Sequential fallback: existing behavior (parallel across rho levels) ──
            n_rho = len(pending_rhos)
            budget = compute_thread_budget(n_outer=n_rho)
            n_jobs_inner = budget.n_inner
            nthread = budget.nthread
            log(
                f"  Running {n_rho} rho levels in parallel ({budget.n_outer * budget.n_inner * budget.nthread} cores, {n_jobs_inner} per level, nthread={nthread})"
            )

            results_list = Parallel(n_jobs=n_rho, backend="loky")(
                delayed(_run_single_rho)(rho, sweep_methods, feature_names, n_jobs_inner, nthread=nthread)
                for rho in pending_rhos
            )

            for rho, rho_results, fsi_list, grps_last in results_list:
                sweep_results[rho] = rho_results
                if fsi_list:
                    fsi_by_rho[rho] = fsi_list
                if grps_last is not None:
                    grps_by_rho[rho] = grps_last
        else:
            # ── Parallel rep mode: single flat Parallel over all (rho × rep) pairs ──
            partial_data = {rho: _init_method_data(sweep_methods) for rho in pending_rhos}
            pending_pairs = []

            for rho in pending_rhos:
                for rep in range(N_REPS):
                    ckpt_key = f"linear_sweep_{rho}_rep_{rep}"
                    if resume and _has(ckpt_key):
                        data = _load(ckpt_key)
                        _merge_rep(partial_data[rho], data["per_method"])
                        if data.get("fsi") is not None:
                            fsi_by_rho.setdefault(rho, []).append(data["fsi"])
                        grps_by_rho[rho] = data["grps"]
                    else:
                        pending_pairs.append((rho, rep))

            n_total = len(pending_rhos) * N_REPS
            n_pending = len(pending_pairs)
            n_resumed = n_total - n_pending
            log(
                f"  {n_pending} (rho, rep) pairs pending"
                + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else "")
            )

            if pending_pairs:
                n_workers = compute_rep_worker_budget(n_work=len(pending_pairs))
                total_cores = get_available_cores()
                nthread = 1  # single-threaded XGBoost: deterministic, maximizes worker throughput
                log(f"  Running {n_workers} workers on {total_cores} cores (nthread={nthread})")

                results_list = Parallel(n_jobs=n_workers, backend="loky")(
                    delayed(_run_single_rep)(rho, rep, sweep_methods, feature_names, nthread=nthread)
                    for rho, rep in pending_pairs
                )

                for (rho, rep), (rep_out, per_method, fsi_out, grps) in zip(pending_pairs, results_list):
                    _merge_rep(partial_data[rho], per_method)
                    if fsi_out is not None:
                        fsi_by_rho.setdefault(rho, []).append(fsi_out)
                    grps_by_rho[rho] = grps
                    _save(
                        f"linear_sweep_{rho}_rep_{rep}",
                        per_method=per_method,
                        fsi=fsi_out,
                        grps=grps,
                    )

            # Aggregate per rho, write final checkpoints, clean up per-rep files
            for rho in pending_rhos:
                rho_results = _aggregate_method_data(partial_data[rho], sweep_methods)
                sweep_results[rho] = rho_results
                save_checkpoint(
                    f"linear_sweep_rho_{rho}",
                    checkpoint_dir=CKPT_DIR,
                    rho_results=rho_results,
                    fsi_list=fsi_by_rho.get(rho, []),
                    grps=grps_by_rho.get(rho),
                )
                for rep in range(N_REPS):
                    clear_checkpoint(f"linear_sweep_{rho}_rep_{rep}", CKPT_DIR)
            _shutdown_loky_workers()

    # FSI collinearity validation (reviewer #7): show FSI rises with rho
    log("\n  FSI Collinearity Validation (DASH):")
    log(
        f"  {'rho':>5} {'Mean FSI (signal)':>18} {'Mean FSI (noise)':>18} {'Ratio':>8}  {'Beta corr':>10}  {'p-value':>10}"
    )
    log("  " + "=" * 80)
    fsi_validation = {}
    for rho in rho_levels:
        fsi_list = fsi_by_rho.get(rho, [])
        grps_arr = grps_by_rho.get(rho)
        if not fsi_list or grps_arr is None:
            continue
        mean_fsi = np.mean(fsi_list, axis=0)
        n_groups = len(np.unique(grps_arr))
        signal_mask = grps_arr < (n_groups - 1)  # last group is noise (beta=0)
        mean_fsi_signal = float(np.mean(mean_fsi[signal_mask]))
        mean_fsi_noise = float(np.mean(mean_fsi[~signal_mask]))
        ratio = mean_fsi_signal / max(mean_fsi_noise, 1e-10)
        # Beta-correlation: within each rho level, correlate FSI with group beta
        # magnitude — features in high-beta groups should show higher FSI because
        # there is more signal to compete over under collinearity.
        beta_groups = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0])
        group_size = len(mean_fsi) // len(beta_groups)
        feature_beta = np.repeat(beta_groups, group_size)
        beta_corr = fsi_collinearity_correlation(mean_fsi, feature_beta, groups=grps_arr)
        fsi_validation[str(rho)] = {
            "mean_fsi_signal": mean_fsi_signal,
            "mean_fsi_noise": mean_fsi_noise,
            "mean_fsi_all": float(np.mean(mean_fsi)),
            "signal_noise_ratio": ratio,
            "beta_spearman": beta_corr["feature_spearman"],
            "beta_pvalue": beta_corr["feature_pvalue"],
            "beta_group_spearman": beta_corr.get("group_spearman"),
            "beta_group_pvalue": beta_corr.get("group_pvalue"),
        }
        log(
            f"  {rho:5.2f} {mean_fsi_signal:18.4f} {mean_fsi_noise:18.4f} {ratio:8.2f}"
            f"  beta_rho={beta_corr['feature_spearman']:.3f}"
            f"  (p={beta_corr['feature_pvalue']:.4g})"
        )
    sweep_results["_fsi_validation"] = fsi_validation

    # Bootstrap stability tests for sweep results (REVIEW_v7 M3)
    log("\n  Bootstrap stability tests (sweep):")
    log(f"  {'rho':>5} {'Comparison':<35} {'d Stab':>10} {'p-value':>10} {'95% CI':>22}  Sig")
    log("  " + "=" * 90)
    stab_test_results = {}
    dash_name = "DASH (MaxMin)"
    for rho in rho_levels:
        if rho not in sweep_results or dash_name not in sweep_results[rho]:
            continue
        dash_imp = sweep_results[rho][dash_name].get("imp_runs")
        if dash_imp is None:
            continue
        stab_test_results[str(rho)] = {}
        for bname in sweep_methods:
            if bname == dash_name:
                continue
            bl_imp = sweep_results[rho].get(bname, {}).get("imp_runs")
            if bl_imp is None:
                continue
            try:
                diff, pval, ci_lo, ci_hi = bootstrap_stability_test(dash_imp, bl_imp)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                label = f"{dash_name} vs {bname}"
                log(f"  {rho:5.2f} {label:<35} {diff:+10.4f} {pval:10.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")
                stab_test_results[str(rho)][bname] = {
                    "diff": float(diff),
                    "p": float(pval),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
            except Exception:
                pass
    sweep_results["_stability_tests"] = stab_test_results

    # Equity significance tests: Wilcoxon on equity (lower is better for DASH)
    log("\n  Equity significance tests (sweep):")
    log(f"  {'rho':>5} {'Comparison':<35} {'Wilcoxon p':>12} {'Cohen d':>10}")
    log("  " + "-" * 65)
    eq_test_results = {}
    for rho in rho_levels:
        if rho not in sweep_results or dash_name not in sweep_results[rho]:
            continue
        dash_eq = sweep_results[rho][dash_name].get("eq_runs")
        if dash_eq is None:
            continue
        eq_test_results[str(rho)] = {}
        for bname in sweep_methods:
            if bname == dash_name:
                continue
            bl_eq = sweep_results[rho].get(bname, {}).get("eq_runs")
            if bl_eq is None:
                continue
            try:
                _, pval = compare_methods(dash_eq, bl_eq)
                d = cohens_d(dash_eq, bl_eq)
                sig = "*" if pval < 0.05 else "n.s."
                label = f"{dash_name} vs {bname}"
                log(f"  {rho:5.2f} {label:<35} {pval:12.4g} {d:10.3f}  {sig}")
                eq_test_results[str(rho)][bname] = {"p": float(pval), "cohens_d": float(d)}
            except Exception:
                pass
    sweep_results["_equity_tests"] = eq_test_results

    # Top-k5 significance tests: bootstrap permutation on Jaccard overlap
    log("\n  Top-k5 significance tests (sweep):")
    log(f"  {'rho':>5} {'Comparison':<35} {'d TopK5':>10} {'p-value':>10} {'95% CI':>22}  Sig")
    log("  " + "=" * 90)
    topk5_test_results = {}
    for rho in rho_levels:
        if rho not in sweep_results or dash_name not in sweep_results[rho]:
            continue
        dash_imp = sweep_results[rho][dash_name].get("imp_runs")
        if dash_imp is None:
            continue
        topk5_test_results[str(rho)] = {}
        for bname in sweep_methods:
            if bname == dash_name:
                continue
            bl_imp = sweep_results[rho].get(bname, {}).get("imp_runs")
            if bl_imp is None:
                continue
            try:
                diff, pval, ci_lo, ci_hi = bootstrap_topk5_test(dash_imp, bl_imp)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                label = f"{dash_name} vs {bname}"
                log(f"  {rho:5.2f} {label:<35} {diff:+10.4f} {pval:10.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")
                topk5_test_results[str(rho)][bname] = {
                    "diff": float(diff),
                    "p": float(pval),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
            except Exception:
                pass
    sweep_results["_topk5_tests"] = topk5_test_results

    _publish_results(sweep_results, f"{OUT}/tables/synthetic_linear_sweep.json", "linear_sweep", N_REPS, t0)
    sweep_results.pop("_equity_tests", None)  # remove string key before return
    sweep_results.pop("_stability_tests", None)  # remove string key before return
    sweep_results.pop("_topk5_tests", None)  # remove string key before return
    sweep_results.pop("_fsi_validation", None)  # remove string key before return
    plot_correlation_sweep(sweep_results, rho_levels, sweep_methods)

    if cleanup:
        clear_checkpoints_by_prefix("linear_sweep_rho_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Linear sweep completed in {elapsed / 60:.1f} min")
    format_timing_table(sweep_results, rho=0.9)
    return sweep_results


###############################################################################
# EXPERIMENT: Overlapping Correlation Structure
###############################################################################


def experiment_overlapping(resume=False, cleanup=False):
    """Overlapping correlation structure at rho=0.9.

    M7 fix: now reports accuracy, equity, and RMSE alongside stability.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Overlapping Correlation Structure")
    log("=" * 70)

    method_names = ["Single Best", "DASH (MaxMin)", "DASH (Cluster)"]
    results = {
        n: {"imp_runs": [], "acc_runs": [], "grp_acc_runs": [], "gmse_runs": [], "eq_runs": [], "rmse_runs": []}
        for n in method_names
    }
    feature_names = make_feature_names()
    seq_budget = compute_thread_budget(n_outer=1)

    start_rep = 0
    if resume:
        for batch_end in range(N_REPS, 0, -10):
            ckpt_name = f"overlapping_batch_{batch_end}"
            if _has(ckpt_name):
                cached = _load(ckpt_name)
                results = cached["results"]
                start_rep = cached["completed_reps"]
                log(f"  Resuming from rep {start_rep}")
                break

    for rep in range(start_rep, N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep + 1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, meta = generate_synthetic_linear(
            N=5000, rho=0.9, seed=rep_seed, structure="overlapping"
        )

        sb = SingleBestBaseline(
            n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=seq_budget.n_inner, nthread=seq_budget.nthread
        )
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        results["Single Best"]["imp_runs"].append(sb.global_importance_)
        r, _ = dgp_agreement(sb.global_importance_, true_imp)
        results["Single Best"]["acc_runs"].append(r)
        results["Single Best"]["grp_acc_runs"].append(group_level_accuracy(sb.global_importance_, true_imp, grps))
        results["Single Best"]["gmse_runs"].append(group_level_mse(sb.global_importance_, true_imp, grps))
        results["Single Best"]["eq_runs"].append(within_group_equity(sb.global_importance_, grps))
        results["Single Best"]["rmse_runs"].append(rmse_score(yte, sb.model_.predict(Xte)))

        dm = DASHPipeline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            selection_method="maxmin",
            n_jobs=seq_budget.n_inner,
            nthread=seq_budget.nthread,
            seed=rep_seed,
            verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        results["DASH (MaxMin)"]["imp_runs"].append(dm.global_importance_)
        r, _ = dgp_agreement(dm.global_importance_, true_imp)
        results["DASH (MaxMin)"]["acc_runs"].append(r)
        results["DASH (MaxMin)"]["grp_acc_runs"].append(group_level_accuracy(dm.global_importance_, true_imp, grps))
        results["DASH (MaxMin)"]["gmse_runs"].append(group_level_mse(dm.global_importance_, true_imp, grps))
        results["DASH (MaxMin)"]["eq_runs"].append(within_group_equity(dm.global_importance_, grps))
        results["DASH (MaxMin)"]["rmse_runs"].append(rmse_score(yte, dm.get_consensus_ensemble_predictions(Xte)))
        dm.all_shap_matrices_ = None
        dm.consensus_matrix_ = None

        imp_vecs = get_preliminary_importance(
            dm.models_,
            dm.filtered_indices_,
            Xexp,
            method="gain",
        )
        filt_scores = {i: dm.val_scores_[i] for i in dm.filtered_indices_}
        sel_c = cluster_coverage_selection(
            imp_vecs,
            filt_scores,
            Xtr,
            tau=PAPER_CONFIG["TAU_CLUSTER"],
            K=K,
            verbose=False,
        )
        cons_c, shap_c = compute_consensus(dm.models_, sel_c, Xexp, verbose=False, n_jobs=seq_budget.n_inner)
        _, _, _, imp_c = compute_diagnostics(shap_c)
        results["DASH (Cluster)"]["imp_runs"].append(imp_c)
        r, _ = dgp_agreement(imp_c, true_imp)
        results["DASH (Cluster)"]["acc_runs"].append(r)
        results["DASH (Cluster)"]["grp_acc_runs"].append(group_level_accuracy(imp_c, true_imp, grps))
        results["DASH (Cluster)"]["gmse_runs"].append(group_level_mse(imp_c, true_imp, grps))
        results["DASH (Cluster)"]["eq_runs"].append(within_group_equity(imp_c, grps))
        # Cluster uses same models as DASH, so use DASH predictions for RMSE
        results["DASH (Cluster)"]["rmse_runs"].append(rmse_score(yte, dm.get_consensus_ensemble_predictions(Xte)))
        del sb, dm

        if (rep + 1) % 10 == 0:
            _save(f"overlapping_batch_{rep + 1}", results=results, completed_reps=rep + 1)

    log(
        f"\n  {'Method':<20} {'Stability':>10} {'Top-5':>8} {'DGP Agree':>10} {'Grp Acc':>10} {'Grp MSE':>10} {'Equity':>10} {'RMSE':>10}"
    )
    log("  " + "=" * 93)
    overlap_results = {}
    for name in method_names:
        stab = importance_stability(results[name]["imp_runs"])
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(results[name]["imp_runs"], k=5)
        acc = np.mean(results[name]["acc_runs"])
        grp = np.mean(results[name]["grp_acc_runs"])
        gmse = np.mean(results[name]["gmse_runs"])
        eq = np.mean(results[name]["eq_runs"])
        rmse = np.mean(results[name]["rmse_runs"])
        log(
            f"  {name:<20} {stab:>10.4f} {topk5:>8.4f} {acc:>10.4f} {grp:>10.4f} {gmse:>10.6f} {eq:>10.4f} {rmse:>10.4f}"
        )
        overlap_results[name] = {
            "stability": stab,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "accuracy_mean": acc,
            "accuracy_std": float(np.std(results[name]["acc_runs"])),
            "group_accuracy_mean": grp,
            "group_mse_mean": gmse,
            "group_mse_std": float(np.std(results[name]["gmse_runs"], ddof=1)),
            "equity_mean": eq,
            "equity_std": float(np.std(results[name]["eq_runs"])),
            "rmse_mean": rmse,
            "rmse_std": float(np.std(results[name]["rmse_runs"])),
            "acc_runs": [float(x) for x in results[name]["acc_runs"]],
            "eq_runs": [float(x) for x in results[name]["eq_runs"]],
            "rmse_runs": [float(x) for x in results[name]["rmse_runs"]],
            "n_reps": len(results[name]["acc_runs"]),
        }

    _publish_results(overlap_results, f"{OUT}/tables/overlapping.json", "overlapping", N_REPS, t0)

    if cleanup:
        clear_checkpoints_by_prefix("overlapping_batch_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Overlapping completed in {elapsed / 60:.1f} min")
    return overlap_results


###############################################################################
# EXPERIMENT: Nonlinear DGP Correlation Sweep
###############################################################################


def _run_single_rep_nonlinear(rho, rep, nl_methods, feature_names, *, nthread=1):
    """Run all nonlinear methods for one (rho, rep) pair.

    Returns
    -------
    tuple : (rho, rep, per_method_dict)
        per_method_dict keys: eq, imp, rmse, keff
        All values are scalars or numpy arrays — NO model object references.
    """
    rep_seed = SEED + rep
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, _, _ = generate_synthetic_nonlinear(N=5000, rho=rho, seed=rep_seed)

    per_method: dict = {}

    for name in nl_methods:
        if name == "Single Best":
            m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=1, nthread=nthread)
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
            imp = m.global_importance_
            preds = m.model_.predict(Xte)
        elif name in ("Large Single Model", "LSM (Tuned)"):
            m = LargeSingleModelBaseline(
                K=K,
                T_per_model=PAPER_CONFIG["T_PER_MODEL"],
                colsample_bytree=0.2,
                seed=rep_seed,
                tune=(name == "LSM (Tuned)"),
                nthread=nthread,
            )
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
            imp = m.global_importance_
            preds = m.model_.predict(Xte)
        elif name == "Stochastic Retrain":
            m = StochasticRetrainBaseline(
                N=K,
                task="regression",
                n_jobs=1,
                nthread=nthread,
                seed=rep_seed,
            )
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
            imp = m.global_importance_
            preds = m.get_consensus_ensemble_predictions(Xte)
        elif name == "Random Forest":
            m = RandomForestBaseline(
                n_estimators=500,
                task="regression",
                seed=rep_seed,
            )
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
            imp = m.global_importance_
            preds = m.model_.predict(Xte)
        elif name == "DASH (MaxMin)":
            m = DASHPipeline(
                M=M,
                K=K,
                epsilon=EPSILON,
                delta=DELTA,
                selection_method="maxmin",
                n_jobs=1,
                nthread=nthread,
                seed=rep_seed,
                verbose=False,
            )
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
            imp = m.global_importance_
            preds = m.get_consensus_ensemble_predictions(Xte)
            # Save population for Random Selection reuse
            _dash_models = m.models_
            _dash_val_scores = m.val_scores_
        elif name == "Random Selection":
            m = RandomSelectionBaseline(
                M=M,
                K=K,
                epsilon=EPSILON,
                delta=DELTA,
                n_jobs=1,
                nthread=nthread,
                seed=rep_seed,
            )
            m.fit_from_population(_dash_models, _dash_val_scores, Xexp, feature_names=feature_names)
            imp = m.global_importance_
            preds = m.get_consensus_ensemble_predictions(Xte)
        else:
            raise ValueError(f"Unknown nonlinear method: {name}")

        rmse_val = rmse_score(yte, preds)
        keff = None
        if hasattr(m, "selected_indices_") and m.selected_indices_ is not None:
            keff = len(m.selected_indices_)
        per_method[name] = {
            "eq": within_group_equity(imp, grps),
            "imp": imp,
            "rmse": rmse_val,
            "keff": keff,
        }
        del m

    return rho, rep, per_method


def experiment_nonlinear_sweep(resume=False, cleanup=False):
    """Nonlinear DGP sweep: rho ∈ {0.0, 0.5, 0.7, 0.9, 0.95}.

    Evaluates stability and equity (no ground-truth accuracy for nonlinear DGP).
    NOTE: true_importance for the nonlinear DGP is an approximate ordinal
    ranking, not exact analytic SHAP.  See generate_synthetic_nonlinear().
    Includes LSM (Tuned) for fair comparison.
    Rep-level parallelism: flattens all (rho × rep) pairs into a single
    Parallel call with per-rep checkpointing for crash resilience.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Nonlinear DGP — Correlation Sweep")
    log("=" * 70)

    nl_rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    nl_methods = [
        "Single Best",
        "Large Single Model",
        "LSM (Tuned)",
        "Stochastic Retrain",
        "Random Forest",
        "DASH (MaxMin)",
        "Random Selection",  # must follow DASH — reuses its population
    ]
    nl_sweep = {rho: {} for rho in nl_rho_levels}
    feature_names = make_feature_names()

    # Separate resumed vs pending rho levels
    pending_rhos = []
    for rho in nl_rho_levels:
        ckpt_name = f"nonlinear_sweep_rho_{rho}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for NL ρ={rho}")
            nl_sweep[rho] = _load(ckpt_name)["rho_results"]
        else:
            pending_rhos.append(rho)

    if pending_rhos:
        # ── Parallel rep mode: single flat Parallel over all (rho × rep) pairs ──
        partial_data = {
            rho: {name: {"eq_runs": [], "imp_runs": [], "rmse_runs": [], "keff_runs": []} for name in nl_methods}
            for rho in pending_rhos
        }
        pending_pairs = []

        for rho in pending_rhos:
            for rep in range(N_REPS):
                ckpt_key = f"nonlinear_sweep_{rho}_rep_{rep}"
                if resume and _has(ckpt_key):
                    data = _load(ckpt_key)
                    for name in nl_methods:
                        if name not in data["per_method"]:
                            continue
                        m = data["per_method"][name]
                        partial_data[rho][name]["eq_runs"].append(m["eq"])
                        partial_data[rho][name]["imp_runs"].append(m["imp"])
                        partial_data[rho][name]["rmse_runs"].append(m["rmse"])
                        if m["keff"] is not None:
                            partial_data[rho][name]["keff_runs"].append(m["keff"])
                else:
                    pending_pairs.append((rho, rep))

        n_total = len(pending_rhos) * N_REPS
        n_pending = len(pending_pairs)
        n_resumed = n_total - n_pending
        log(
            f"  {n_pending} (rho, rep) pairs pending"
            + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else "")
        )

        if pending_pairs:
            n_workers = compute_rep_worker_budget(n_work=len(pending_pairs))
            nthread = 1
            log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

            results_list = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_run_single_rep_nonlinear)(rho, rep, nl_methods, feature_names, nthread=nthread)
                for rho, rep in pending_pairs
            )

            for (rho, rep), (_, _, per_method) in zip(pending_pairs, results_list):
                for name in nl_methods:
                    if name not in per_method:
                        continue
                    m = per_method[name]
                    partial_data[rho][name]["eq_runs"].append(m["eq"])
                    partial_data[rho][name]["imp_runs"].append(m["imp"])
                    partial_data[rho][name]["rmse_runs"].append(m["rmse"])
                    if m["keff"] is not None:
                        partial_data[rho][name]["keff_runs"].append(m["keff"])
                _save(f"nonlinear_sweep_{rho}_rep_{rep}", per_method=per_method)
            _shutdown_loky_workers()

        # Aggregate per rho, write final checkpoints, clean up per-rep files
        for rho in pending_rhos:
            rho_results = {}
            for name in nl_methods:
                d = partial_data[rho][name]
                eq_runs = d["eq_runs"]
                imp_runs = d["imp_runs"]
                rmse_runs = d["rmse_runs"]
                keff_runs = d["keff_runs"]
                stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
                topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
                rho_results[name] = {
                    "stability": stab,
                    "stability_se": stab_se,
                    "stability_ci_lo": stab_ci_lo,
                    "stability_ci_hi": stab_ci_hi,
                    "topk5_stability": topk5,
                    "topk5_se": topk5_se,
                    "topk5_ci_lo": topk5_ci_lo,
                    "topk5_ci_hi": topk5_ci_hi,
                    "k_eff_mean": float(np.mean(keff_runs)) if keff_runs else None,
                    "k_eff_std": float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
                    "equity_mean": np.mean(eq_runs),
                    "equity_std": np.std(eq_runs, ddof=1),
                    "eq_runs": np.array(eq_runs),
                    "rmse_mean": float(np.mean(rmse_runs)),
                    "rmse_std": float(np.std(rmse_runs, ddof=1)),
                    "rmse_runs": np.array(rmse_runs),
                }
                log(
                    f"  {name:<20} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  eq={np.mean(eq_runs):.4f}  "
                    f"RMSE={np.mean(rmse_runs):.4f}"
                )
            nl_sweep[rho] = rho_results
            _save(f"nonlinear_sweep_rho_{rho}", rho_results=rho_results)

        clear_checkpoints_by_prefix("nonlinear_sweep_", CKPT_DIR)

    # Safety desideratum check: flag rho levels where DASH < SB
    log("\n  Safety desideratum check (nonlinear DGP):")
    log(f"  {'rho':>5} {'DASH':>10} {'SB':>10} {'Status':>12}")
    log("  " + "-" * 42)
    for rho in nl_rho_levels:
        dash_s = nl_sweep[rho].get("DASH (MaxMin)", {}).get("stability")
        sb_s = nl_sweep[rho].get("Single Best", {}).get("stability")
        if dash_s is not None and sb_s is not None:
            status = "PASS" if dash_s >= sb_s else "VIOLATION"
            log(f"  {rho:5.2f} {dash_s:10.4f} {sb_s:10.4f} {status:>12}")
    log("  Note: DASH advantage is expected only at rho >= 0.7 under nonlinear DGPs.")

    _publish_results(nl_sweep, f"{OUT}/tables/nonlinear_sweep.json", "nonlinear_sweep", N_REPS, t0)
    plot_nonlinear_sweep(nl_sweep, nl_rho_levels, nl_methods)

    if cleanup:
        clear_checkpoints_by_prefix("nonlinear_sweep_rho_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Nonlinear sweep completed in {elapsed / 60:.1f} min")
    return nl_sweep


###############################################################################
# EXPERIMENT: Extended Baselines (Table 2) at rho=0.9
###############################################################################


def _run_single_table2_rep(name, rep, feature_names, *, nthread=1):
    """Run one rep for a single method in the table2 baselines experiment.

    Returns
    -------
    tuple : (name, rep, per_rep_dict)
        per_rep_dict keys: imp, acc, eq
        All values are scalars or numpy arrays — NO model object references.
    """
    rep_seed = SEED + rep
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
        N=5000, rho=0.9, seed=rep_seed
    )

    if name == "Ensemble SHAP":
        m = EnsembleSHAPBaseline(
            n_estimators=PAPER_CONFIG["N_ESTIMATORS_ESHAP"],
            task="regression",
            nthread=nthread,
            seed=rep_seed,
        )
        m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
        imp = m.global_importance_
    elif name == "Stochastic Retrain":
        m = StochasticRetrainBaseline(N=K, task="regression", n_jobs=1, nthread=nthread, seed=rep_seed)
        m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
        imp = m.global_importance_
    elif name == "Random Forest":
        m = RandomForestBaseline(
            n_estimators=500,
            task="regression",
            seed=rep_seed,
        )
        m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = m.global_importance_
    elif name == "Permutation Importance":
        m = PermutationImportanceBaseline(
            n_trials=N_TRIALS_SB,
            task="regression",
            seed=rep_seed,
        )
        m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, y_ref=yexp)
        imp = m.global_importance_
    elif name == "LightGBM Single Best":
        m = LightGBMSingleBestBaseline(
            n_estimators=500,
            task="regression",
            seed=rep_seed,
        )
        m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        imp = m.global_importance_
    else:  # DASH (Dedup)
        m = DASHPipeline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            selection_method="dedup",
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        imp = m.global_importance_
    del m

    r, _ = dgp_agreement(imp, true_imp)
    return (
        name,
        rep,
        {
            "imp": imp,
            "acc": r,
            "eq": within_group_equity(imp, grps),
        },
    )


def experiment_table2_baselines(resume=False, cleanup=False):
    """Extended baselines at rho=0.9: Ensemble SHAP, Stochastic Retrain, DASH (Dedup).

    Rep-level parallelism: flattens all (method × rep) pairs into a single
    Parallel call with per-rep checkpointing for crash resilience.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Table 2 — Extended Baselines at ρ=0.9")
    log("=" * 70)

    table2_methods = [
        "Ensemble SHAP",
        "Stochastic Retrain",
        "Random Forest",
        "Permutation Importance",
        "DASH (Dedup)",
    ]
    if _HAS_LIGHTGBM:
        table2_methods.append("LightGBM Single Best")
    feature_names = make_feature_names()

    # Check for method-level checkpoints (fully computed methods)
    partial_data = {name: {"imp_runs": [], "acc_runs": [], "eq_runs": []} for name in table2_methods}
    pending_methods_set = set(table2_methods)
    for name in table2_methods:
        ckpt_name = f"table2_{_sanitize_ckpt_name(name)}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for {name}")
            mr = _load(ckpt_name)["method_results"]
            # Store aggregated result directly — skip per-rep processing
            partial_data[name]["_aggregated"] = mr
            pending_methods_set.discard(name)

    pending_names = [n for n in table2_methods if n in pending_methods_set]

    pending_pairs = []
    for name in pending_names:
        for rep in range(N_REPS):
            ckpt_key = f"table2_flat_{_sanitize_ckpt_name(name)}_rep_{rep}"
            if resume and _has(ckpt_key):
                data = _load(ckpt_key)
                m = data["per_rep"]
                partial_data[name]["imp_runs"].append(m["imp"])
                partial_data[name]["acc_runs"].append(m["acc"])
                partial_data[name]["eq_runs"].append(m["eq"])
            else:
                pending_pairs.append((name, rep))

    n_total = len(pending_names) * N_REPS
    n_pending = len(pending_pairs)
    n_resumed = n_total - n_pending
    log(
        f"  {n_pending} (method, rep) pairs pending" + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else "")
    )

    if pending_pairs:
        n_workers = compute_rep_worker_budget(n_work=len(pending_pairs))
        nthread = 1
        log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_table2_rep)(name, rep, feature_names, nthread=nthread) for name, rep in pending_pairs
        )

        for (name, rep), (_, _, per_rep) in zip(pending_pairs, results_list):
            partial_data[name]["imp_runs"].append(per_rep["imp"])
            partial_data[name]["acc_runs"].append(per_rep["acc"])
            partial_data[name]["eq_runs"].append(per_rep["eq"])
            _save(f"table2_flat_{_sanitize_ckpt_name(name)}_rep_{rep}", per_rep=per_rep)
        _shutdown_loky_workers()

    # Aggregate per method, write final checkpoints, clean up per-rep files
    table2_results = {}
    for name in table2_methods:
        if "_aggregated" in partial_data[name]:
            table2_results[name] = partial_data[name]["_aggregated"]
            continue
        imp_runs = partial_data[name]["imp_runs"]
        acc_runs = partial_data[name]["acc_runs"]
        eq_runs = partial_data[name]["eq_runs"]
        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
        method_results = {
            "stability": stab,
            "stability_se": stab_se,
            "stability_ci_lo": stab_ci_lo,
            "stability_ci_hi": stab_ci_hi,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "accuracy_mean": np.mean(acc_runs),
            "accuracy_std": np.std(acc_runs, ddof=1),
            "equity_mean": np.mean(eq_runs),
            "equity_std": np.std(eq_runs, ddof=1),
            "acc_runs": np.array(acc_runs),
            "eq_runs": np.array(eq_runs),
        }
        log(
            f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  "
            f"acc={np.mean(acc_runs):.4f}  eq={np.mean(eq_runs):.4f}"
        )
        table2_results[name] = method_results
        _save(f"table2_{_sanitize_ckpt_name(name)}", method_results=method_results)

    clear_checkpoints_by_prefix("table2_flat_", CKPT_DIR)

    _publish_results(table2_results, f"{OUT}/tables/table2_baselines.json", "table2_baselines", N_REPS, t0)
    plot_table2_baselines(table2_results)

    if cleanup:
        clear_checkpoints_by_prefix("table2_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Table 2 baselines completed in {elapsed / 60:.1f} min")
    return table2_results


###############################################################################
# EXPERIMENT: California Housing
###############################################################################


def _run_single_cal_rep(name, rep, X_pool, X_test, y_pool, y_test, cal_names, *, nthread=1):
    """Run one rep for a single method in the California Housing experiment.

    Returns
    -------
    tuple : (name, rep, per_rep_dict)
        per_rep_dict keys: imp, rmse, abl, keff
        All values are scalars or numpy arrays — NO model object references.
    """
    rep_seed = SEED + rep
    Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(X_pool, y_pool, test_size=0.2, random_state=rep_seed)
    Xtr_r, Xexp_r, ytr_r, yexp_r = train_test_split(Xtr_r, ytr_r, test_size=0.12, random_state=rep_seed)
    scaler_r = StandardScaler().fit(Xtr_r)
    Xtr_r = scaler_r.transform(Xtr_r)
    Xv_r = scaler_r.transform(Xv_r)
    Xexp_r = scaler_r.transform(Xexp_r)
    Xte_r = scaler_r.transform(X_test)

    if name == "Single Best":
        m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=1, nthread=nthread)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Single Best (M=200)":
        m = SingleBestBaseline(n_trials=M, seed=rep_seed, n_jobs=1, nthread=nthread)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Large Single Model":
        m = LargeSingleModelBaseline(
            K=K,
            T_per_model=PAPER_CONFIG["T_PER_MODEL"],
            colsample_bytree=0.2,
            seed=rep_seed,
            nthread=nthread,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Random Forest":
        m = RandomForestBaseline(n_estimators=500, task="regression", seed=rep_seed)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Stochastic Retrain":
        m = StochasticRetrainBaseline(N=K, task="regression", n_jobs=1, nthread=nthread, seed=rep_seed)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(m.models_[0], Xte_r, y_test, imp)
    elif name == "Random Selection":
        m = RandomSelectionBaseline(
            M=M,
            K=K,
            epsilon=REAL_EPSILON,
            delta=DELTA,
            epsilon_mode=REAL_EPSILON_MODE,
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(m.models_[m.selected_indices_[0]], Xte_r, y_test, imp)
    elif name == "Ensemble SHAP":
        m = EnsembleSHAPBaseline(
            n_estimators=PAPER_CONFIG["N_ESTIMATORS_ESHAP"],
            task="regression",
            seed=rep_seed,
            nthread=nthread,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Naive Top-N":
        # Train population (identical to DASH's — same seed, same search space)
        from dash_shap.core.population import generate_model_population

        pop_models, pop_scores, _ = generate_model_population(
            Xtr_r,
            ytr_r,
            Xv_r,
            yv_r,
            M=M,
            task="regression",
            seed=rep_seed,
            n_jobs=1,
            verbose=False,
            nthread=nthread,
        )
        m = NaiveAveragingBaseline(N=K, task="regression", n_jobs=1)
        m.fit_from_population(pop_models, pop_scores, Xexp_r)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(pop_models[m.selected_indices_[0]], Xte_r, y_test, imp)
        del pop_models, pop_scores
    else:  # DASH (MaxMin)
        m = DASHPipeline(
            M=M,
            K=K,
            epsilon=REAL_EPSILON,
            delta=DELTA,
            epsilon_mode=REAL_EPSILON_MODE,
            selection_method="maxmin",
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=cal_names)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(m.selected_models_[0], Xte_r, y_test, imp)

    keff = None
    if hasattr(m, "selected_indices_") and m.selected_indices_ is not None:
        keff = len(m.selected_indices_)
    del m

    return name, rep, {"imp": imp, "rmse": rmse_val, "abl": abl, "keff": keff}


def experiment_real_california(resume=False, cleanup=False):
    """California Housing benchmark with scale-appropriate epsilon.

    Rep-level parallelism: flattens all (method × rep) pairs into a single
    Parallel call with per-rep checkpointing for crash resilience.
    """
    from joblib import Parallel, delayed

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
        X_cal,
        y_cal,
        test_size=0.2,
        random_state=SEED,
    )

    cal_methods = [
        "Single Best",
        "Single Best (M=200)",
        "Large Single Model",
        "Ensemble SHAP",
        "Random Forest",
        "Stochastic Retrain",
        "Random Selection",
        "Naive Top-N",
        "DASH (MaxMin)",
    ]

    # Check for method-level checkpoints (fully computed methods)
    partial_data = {
        name: {"imp_runs": [], "rmse_runs": [], "ablation_runs": [], "keff_runs": []} for name in cal_methods
    }
    pending_methods_set = set(cal_methods)
    for name in cal_methods:
        ckpt_name = f"california_{_sanitize_ckpt_name(name)}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for {name}")
            mr = _load(ckpt_name)["method_results"]
            partial_data[name]["_aggregated"] = mr
            pending_methods_set.discard(name)

    pending_names = [n for n in cal_methods if n in pending_methods_set]

    pending_pairs = []
    for name in pending_names:
        for rep in range(N_REPS):
            ckpt_key = f"cal_flat_{_sanitize_ckpt_name(name)}_rep_{rep}"
            if resume and _has(ckpt_key):
                data = _load(ckpt_key)
                m = data["per_rep"]
                partial_data[name]["imp_runs"].append(m["imp"])
                partial_data[name]["rmse_runs"].append(m["rmse"])
                partial_data[name]["ablation_runs"].append(m["abl"])
                if m["keff"] is not None:
                    partial_data[name]["keff_runs"].append(m["keff"])
            else:
                pending_pairs.append((name, rep))

    n_total = len(pending_names) * N_REPS
    n_pending = len(pending_pairs)
    n_resumed = n_total - n_pending
    log(
        f"  {n_pending} (method, rep) pairs pending" + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else "")
    )

    if pending_pairs:
        n_workers = compute_rep_worker_budget(n_work=len(pending_pairs))
        nthread = 1
        log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_cal_rep)(
                name, rep, X_cal_pool, X_cal_test, y_cal_pool, y_cal_test, cal_names, nthread=nthread
            )
            for name, rep in pending_pairs
        )

        for (name, rep), (_, _, per_rep) in zip(pending_pairs, results_list):
            partial_data[name]["imp_runs"].append(per_rep["imp"])
            partial_data[name]["rmse_runs"].append(per_rep["rmse"])
            partial_data[name]["ablation_runs"].append(per_rep["abl"])
            if per_rep["keff"] is not None:
                partial_data[name]["keff_runs"].append(per_rep["keff"])
            _save(f"cal_flat_{_sanitize_ckpt_name(name)}_rep_{rep}", per_rep=per_rep)

        _shutdown_loky_workers()

    # Aggregate per method, write final checkpoints, clean up per-rep files
    cal_results = {}
    for name in cal_methods:
        if "_aggregated" in partial_data[name]:
            cal_results[name] = partial_data[name]["_aggregated"]
            continue
        imp_runs = partial_data[name]["imp_runs"]
        rmse_runs = partial_data[name]["rmse_runs"]
        ablation_runs = partial_data[name]["ablation_runs"]
        keff_runs = partial_data[name]["keff_runs"]
        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
        method_results = {
            "stability": stab,
            "stability_se": stab_se,
            "stability_ci_lo": stab_ci_lo,
            "stability_ci_hi": stab_ci_hi,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "k_eff_mean": float(np.mean(keff_runs)) if keff_runs else None,
            "k_eff_std": float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
            "rmse_mean": np.mean(rmse_runs),
            "rmse_std": np.std(rmse_runs, ddof=1),
            "ablation_mean": np.mean(ablation_runs),
            "ablation_std": np.std(ablation_runs, ddof=1),
            "rmse_runs": np.array(rmse_runs),
            "ablation_runs": np.array(ablation_runs),
            "imp_runs": imp_runs,
        }
        log(
            f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  "
            f"RMSE={np.mean(rmse_runs):.4f}±{np.std(rmse_runs, ddof=1):.4f}  "
            f"ablation={np.mean(ablation_runs):.4f}"
        )
        cal_results[name] = method_results
        _save(f"california_{_sanitize_ckpt_name(name)}", method_results=method_results)

    clear_checkpoints_by_prefix("cal_flat_", CKPT_DIR)

    # C7+F1: Wilcoxon signed-rank test and Cohen's d between DASH and baselines
    _log_pairwise_significance(cal_results, "DASH (MaxMin)", cal_methods, "California Housing")

    # IS plot skipped when methods run in parallel (model object not available)
    log("  Skipped IS plot (methods ran in parallel)")

    import sklearn as _sklearn

    cal_results["_dataset"] = {
        "sklearn_version": _sklearn.__version__,
        "dataset": "california_housing",
        "source": "sklearn.datasets.fetch_california_housing",
    }
    _publish_results(cal_results, f"{OUT}/tables/california_housing.json", "real_california", N_REPS, t0)
    plot_real_world_bar(cal_results, "california_housing")

    if cleanup:
        clear_checkpoints_by_prefix("california_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  California Housing completed in {elapsed / 60:.1f} min")
    return cal_results


###############################################################################
# EXPERIMENT: Breast Cancer
###############################################################################


def _run_single_bc_rep(name, rep, X_pool, X_test, y_pool, y_test, bc_names, *, nthread=1):
    """Run one rep for a single method in the Breast Cancer experiment.

    Returns
    -------
    tuple : (name, rep, per_rep_dict)
        per_rep_dict keys: imp, abl, keff
        All values are scalars or numpy arrays — NO model object references.
    """
    rep_seed = SEED + rep
    Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(X_pool, y_pool, test_size=0.2, random_state=rep_seed)
    Xtr_r, Xexp_r, ytr_r, yexp_r = train_test_split(Xtr_r, ytr_r, test_size=0.12, random_state=rep_seed)
    scaler_r = StandardScaler().fit(Xtr_r)
    Xtr_r = scaler_r.transform(Xtr_r)
    Xv_r = scaler_r.transform(Xv_r)
    Xexp_r = scaler_r.transform(Xexp_r)
    Xte_r = scaler_r.transform(X_test)

    if name == "Single Best":
        m = SingleBestBaseline(n_trials=N_TRIALS_SB, task="binary", seed=rep_seed, n_jobs=1, nthread=nthread)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
        imp = m.global_importance_
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Single Best (M=200)":
        m = SingleBestBaseline(n_trials=M, task="binary", seed=rep_seed, n_jobs=1, nthread=nthread)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
        imp = m.global_importance_
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Large Single Model":
        m = LargeSingleModelBaseline(
            K=K,
            T_per_model=PAPER_CONFIG["T_PER_MODEL"],
            colsample_bytree=0.2,
            task="binary",
            seed=rep_seed,
            nthread=nthread,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
        imp = m.global_importance_
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Random Forest":
        m = RandomForestBaseline(n_estimators=500, task="binary", seed=rep_seed)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Stochastic Retrain":
        m = StochasticRetrainBaseline(N=K, task="binary", n_jobs=1, nthread=nthread, seed=rep_seed)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        abl = feature_ablation_score(m.models_[0], Xte_r, y_test, imp)
    elif name == "Random Selection":
        m = RandomSelectionBaseline(
            M=M,
            K=K,
            epsilon=REAL_EPSILON,
            delta=DELTA,
            epsilon_mode=REAL_EPSILON_MODE,
            task="binary",
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
        imp = m.global_importance_
        abl = feature_ablation_score(m.models_[m.selected_indices_[0]], Xte_r, y_test, imp)
    elif name == "Ensemble SHAP":
        m = EnsembleSHAPBaseline(
            n_estimators=PAPER_CONFIG["N_ESTIMATORS_ESHAP"],
            task="binary",
            seed=rep_seed,
            nthread=nthread,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Naive Top-N":
        from dash_shap.core.population import generate_model_population

        pop_models, pop_scores, _ = generate_model_population(
            Xtr_r,
            ytr_r,
            Xv_r,
            yv_r,
            M=M,
            task="binary",
            seed=rep_seed,
            n_jobs=1,
            verbose=False,
            nthread=nthread,
        )
        m = NaiveAveragingBaseline(N=K, task="binary", n_jobs=1)
        m.fit_from_population(pop_models, pop_scores, Xexp_r)
        imp = m.global_importance_
        abl = feature_ablation_score(pop_models[m.selected_indices_[0]], Xte_r, y_test, imp)
        del pop_models, pop_scores
    else:  # DASH (MaxMin)
        m = DASHPipeline(
            M=M,
            K=K,
            epsilon=REAL_EPSILON,
            delta=DELTA,
            epsilon_mode=REAL_EPSILON_MODE,
            selection_method="maxmin",
            task="binary",
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=bc_names)
        imp = m.global_importance_
        abl = feature_ablation_score(m.selected_models_[0], Xte_r, y_test, imp)

    keff = None
    if hasattr(m, "selected_indices_") and m.selected_indices_ is not None:
        keff = len(m.selected_indices_)
    del m

    return name, rep, {"imp": imp, "abl": abl, "keff": keff}


def experiment_real_breast_cancer(resume=False, cleanup=False):
    """Breast Cancer benchmark (binary classification).

    Rep-level parallelism: flattens all (method × rep) pairs into a single
    Parallel call with per-rep checkpointing for crash resilience.
    """
    from joblib import Parallel, delayed

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
        X_bc,
        y_bc,
        test_size=0.2,
        random_state=SEED,
    )

    bc_methods = [
        "Single Best",
        "Single Best (M=200)",
        "Large Single Model",
        "Ensemble SHAP",
        "Random Forest",
        "Stochastic Retrain",
        "Random Selection",
        "Naive Top-N",
        "DASH (MaxMin)",
    ]

    # Check for method-level checkpoints (fully computed methods)
    partial_data = {name: {"imp_runs": [], "ablation_runs": [], "keff_runs": []} for name in bc_methods}
    pending_methods_set = set(bc_methods)
    for name in bc_methods:
        ckpt_name = f"breast_cancer_{_sanitize_ckpt_name(name)}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for {name}")
            mr = _load(ckpt_name)["method_results"]
            partial_data[name]["_aggregated"] = mr
            pending_methods_set.discard(name)

    pending_names = [n for n in bc_methods if n in pending_methods_set]

    pending_pairs = []
    for name in pending_names:
        for rep in range(N_REPS):
            ckpt_key = f"bc_flat_{_sanitize_ckpt_name(name)}_rep_{rep}"
            if resume and _has(ckpt_key):
                data = _load(ckpt_key)
                m = data["per_rep"]
                partial_data[name]["imp_runs"].append(m["imp"])
                partial_data[name]["ablation_runs"].append(m["abl"])
                if m["keff"] is not None:
                    partial_data[name]["keff_runs"].append(m["keff"])
            else:
                pending_pairs.append((name, rep))

    n_total = len(pending_names) * N_REPS
    n_pending = len(pending_pairs)
    n_resumed = n_total - n_pending
    log(
        f"  {n_pending} (method, rep) pairs pending" + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else "")
    )

    if pending_pairs:
        n_workers = compute_rep_worker_budget(n_work=len(pending_pairs))
        nthread = 1
        log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_bc_rep)(
                name, rep, X_bc_pool, X_bc_test, y_bc_pool, y_bc_test, bc_names, nthread=nthread
            )
            for name, rep in pending_pairs
        )

        for (name, rep), (_, _, per_rep) in zip(pending_pairs, results_list):
            partial_data[name]["imp_runs"].append(per_rep["imp"])
            partial_data[name]["ablation_runs"].append(per_rep["abl"])
            if per_rep["keff"] is not None:
                partial_data[name]["keff_runs"].append(per_rep["keff"])
            _save(f"bc_flat_{_sanitize_ckpt_name(name)}_rep_{rep}", per_rep=per_rep)

        _shutdown_loky_workers()

    # Aggregate per method, write final checkpoints, clean up per-rep files
    bc_results = {}
    for name in bc_methods:
        if "_aggregated" in partial_data[name]:
            bc_results[name] = partial_data[name]["_aggregated"]
            continue
        imp_runs = partial_data[name]["imp_runs"]
        ablation_runs = partial_data[name]["ablation_runs"]
        keff_runs = partial_data[name]["keff_runs"]
        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
        method_results = {
            "stability": stab,
            "stability_se": stab_se,
            "stability_ci_lo": stab_ci_lo,
            "stability_ci_hi": stab_ci_hi,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "k_eff_mean": float(np.mean(keff_runs)) if keff_runs else None,
            "k_eff_std": float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
            "ablation_mean": np.mean(ablation_runs),
            "ablation_std": np.std(ablation_runs, ddof=1),
            "ablation_runs": np.array(ablation_runs),
            "imp_runs": imp_runs,
        }
        log(f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  ablation={np.mean(ablation_runs):.4f}")
        bc_results[name] = method_results
        _save(f"breast_cancer_{_sanitize_ckpt_name(name)}", method_results=method_results)

    clear_checkpoints_by_prefix("bc_flat_", CKPT_DIR)

    # C7+F1: Wilcoxon signed-rank test and Cohen's d
    _log_pairwise_significance(bc_results, "DASH (MaxMin)", bc_methods, "Breast Cancer")

    # IS plot skipped when methods run in parallel (model object not available)
    log("  Skipped IS plot and disagreement map (methods ran in parallel)")

    import sklearn as _sklearn

    bc_results["_dataset"] = {
        "sklearn_version": _sklearn.__version__,
        "dataset": "breast_cancer",
        "source": "sklearn.datasets.load_breast_cancer",
    }
    _publish_results(bc_results, f"{OUT}/tables/breast_cancer.json", "real_breast_cancer", N_REPS, t0)
    plot_real_world_bar(bc_results, "breast_cancer")

    if cleanup:
        clear_checkpoints_by_prefix("breast_cancer_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Breast Cancer completed in {elapsed / 60:.1f} min")
    return bc_results


###############################################################################
# EXPERIMENT: Superconductor UCI Benchmark
###############################################################################


def _run_single_sc_rep(name, rep, X_pool, X_test, y_pool, y_test, sc_names, sc_m, sc_k, *, nthread=1):
    """Run one rep for a single method in the Superconductor experiment.

    Returns
    -------
    tuple : (name, rep, per_rep_dict)
        per_rep_dict keys: imp, rmse, abl, keff
        All values are scalars or numpy arrays — NO model object references.
    """
    rep_seed = SEED + rep
    Xtr_r, Xv_r, ytr_r, yv_r = train_test_split(X_pool, y_pool, test_size=0.2, random_state=rep_seed)
    Xtr_r, Xexp_r, ytr_r, yexp_r = train_test_split(Xtr_r, ytr_r, test_size=0.12, random_state=rep_seed)
    scaler_r = StandardScaler().fit(Xtr_r)
    Xtr_r = scaler_r.transform(Xtr_r)
    Xv_r = scaler_r.transform(Xv_r)
    Xexp_r = scaler_r.transform(Xexp_r)
    Xte_r = scaler_r.transform(X_test)

    if name == "Single Best":
        m = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=1, nthread=nthread)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Single Best (M=200)":
        m = SingleBestBaseline(n_trials=sc_m, seed=rep_seed, n_jobs=1, nthread=nthread)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Large Single Model":
        m = LargeSingleModelBaseline(
            K=sc_k,
            T_per_model=PAPER_CONFIG["T_PER_MODEL"],
            colsample_bytree=0.2,
            seed=rep_seed,
            nthread=nthread,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Random Forest":
        m = RandomForestBaseline(n_estimators=500, task="regression", seed=rep_seed)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Stochastic Retrain":
        m = StochasticRetrainBaseline(N=sc_k, task="regression", n_jobs=1, nthread=nthread, seed=rep_seed)
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(m.models_[0], Xte_r, y_test, imp)
    elif name == "Random Selection":
        m = RandomSelectionBaseline(
            M=sc_m,
            K=sc_k,
            epsilon=REAL_EPSILON,
            delta=DELTA,
            epsilon_mode=REAL_EPSILON_MODE,
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(m.models_[m.selected_indices_[0]], Xte_r, y_test, imp)
    elif name == "Ensemble SHAP":
        m = EnsembleSHAPBaseline(
            n_estimators=PAPER_CONFIG["N_ESTIMATORS_ESHAP"],
            task="regression",
            seed=rep_seed,
            nthread=nthread,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, seed=rep_seed)
        imp = m.global_importance_
        rmse_val = rmse_score(y_test, m.model_.predict(Xte_r))
        abl = feature_ablation_score(m.model_, Xte_r, y_test, imp)
    elif name == "Naive Top-N":
        from dash_shap.core.population import generate_model_population

        pop_models, pop_scores, _ = generate_model_population(
            Xtr_r,
            ytr_r,
            Xv_r,
            yv_r,
            M=sc_m,
            task="regression",
            seed=rep_seed,
            n_jobs=1,
            verbose=False,
            nthread=nthread,
        )
        m = NaiveAveragingBaseline(N=sc_k, task="regression", n_jobs=1)
        m.fit_from_population(pop_models, pop_scores, Xexp_r)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(pop_models[m.selected_indices_[0]], Xte_r, y_test, imp)
        del pop_models, pop_scores
    else:  # DASH (MaxMin)
        m = DASHPipeline(
            M=sc_m,
            K=sc_k,
            epsilon=REAL_EPSILON,
            delta=DELTA,
            epsilon_mode=REAL_EPSILON_MODE,
            selection_method="maxmin",
            n_jobs=1,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xexp_r, feature_names=sc_names)
        imp = m.global_importance_
        preds = m.get_consensus_ensemble_predictions(Xte_r)
        rmse_val = rmse_score(y_test, preds)
        abl = feature_ablation_score(m.selected_models_[0], Xte_r, y_test, imp)

    keff = None
    if hasattr(m, "selected_indices_") and m.selected_indices_ is not None:
        keff = len(m.selected_indices_)
    del m

    return name, rep, {"imp": imp, "rmse": rmse_val, "abl": abl, "keff": keff}


def experiment_real_superconductor(resume=False, cleanup=False):
    """Superconductor UCI benchmark with scale-appropriate epsilon.

    Rep-level parallelism: flattens all (method × rep) pairs into a single
    Parallel call with per-rep checkpointing for crash resilience.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Real Data — Superconductor UCI")
    log("=" * 70)

    from sklearn.datasets import fetch_openml

    log("  Loading Superconductor dataset...")
    data = fetch_openml(name="superconduct", version=1, as_frame=False, parser="auto")
    X_sc, y_sc = data.data, data.target
    sc_names = [f"f{i}" for i in range(X_sc.shape[1])]
    log(f"  {X_sc.shape[0]} samples, {X_sc.shape[1]} features")

    X_sc_pool, X_sc_test, y_sc_pool, y_sc_test = train_test_split(
        X_sc,
        y_sc,
        test_size=0.2,
        random_state=SEED,
    )

    SC_M = 200
    SC_K = 30
    sc_methods = [
        "Single Best",
        "Single Best (M=200)",
        "Large Single Model",
        "Ensemble SHAP",
        "Random Forest",
        "Stochastic Retrain",
        "Random Selection",
        "Naive Top-N",
        "DASH (MaxMin)",
    ]

    # Check for method-level checkpoints (fully computed methods)
    partial_data = {
        name: {"imp_runs": [], "rmse_runs": [], "ablation_runs": [], "keff_runs": []} for name in sc_methods
    }
    pending_methods_set = set(sc_methods)
    for name in sc_methods:
        ckpt_name = f"superconductor_{_sanitize_ckpt_name(name)}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for {name}")
            mr = _load(ckpt_name)["method_results"]
            partial_data[name]["_aggregated"] = mr
            pending_methods_set.discard(name)

    pending_names = [n for n in sc_methods if n in pending_methods_set]

    pending_pairs = []
    for name in pending_names:
        for rep in range(N_REPS):
            ckpt_key = f"sc_flat_{_sanitize_ckpt_name(name)}_rep_{rep}"
            if resume and _has(ckpt_key):
                data_ckpt = _load(ckpt_key)
                m = data_ckpt["per_rep"]
                partial_data[name]["imp_runs"].append(m["imp"])
                partial_data[name]["rmse_runs"].append(m["rmse"])
                partial_data[name]["ablation_runs"].append(m["abl"])
                if m["keff"] is not None:
                    partial_data[name]["keff_runs"].append(m["keff"])
            else:
                pending_pairs.append((name, rep))

    n_total = len(pending_names) * N_REPS
    n_pending = len(pending_pairs)
    n_resumed = n_total - n_pending
    log(
        f"  {n_pending} (method, rep) pairs pending" + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else "")
    )

    if pending_pairs:
        n_workers = compute_rep_worker_budget(n_work=len(pending_pairs), memory_per_worker_mb=500)
        nthread = 1
        log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_sc_rep)(
                name, rep, X_sc_pool, X_sc_test, y_sc_pool, y_sc_test, sc_names, SC_M, SC_K, nthread=nthread
            )
            for name, rep in pending_pairs
        )

        for (name, rep), (_, _, per_rep) in zip(pending_pairs, results_list):
            partial_data[name]["imp_runs"].append(per_rep["imp"])
            partial_data[name]["rmse_runs"].append(per_rep["rmse"])
            partial_data[name]["ablation_runs"].append(per_rep["abl"])
            if per_rep["keff"] is not None:
                partial_data[name]["keff_runs"].append(per_rep["keff"])
            _save(f"sc_flat_{_sanitize_ckpt_name(name)}_rep_{rep}", per_rep=per_rep)

        _shutdown_loky_workers()

    # Aggregate per method, write final checkpoints, clean up per-rep files
    sc_results = {}
    for name in sc_methods:
        if "_aggregated" in partial_data[name]:
            sc_results[name] = partial_data[name]["_aggregated"]
            continue
        imp_runs = partial_data[name]["imp_runs"]
        rmse_runs = partial_data[name]["rmse_runs"]
        ablation_runs = partial_data[name]["ablation_runs"]
        keff_runs = partial_data[name]["keff_runs"]
        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
        method_results = {
            "stability": stab,
            "stability_se": stab_se,
            "stability_ci_lo": stab_ci_lo,
            "stability_ci_hi": stab_ci_hi,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "k_eff_mean": float(np.mean(keff_runs)) if keff_runs else None,
            "k_eff_std": float(np.std(keff_runs, ddof=1)) if len(keff_runs) > 1 else None,
            "rmse_mean": np.mean(rmse_runs),
            "rmse_std": np.std(rmse_runs, ddof=1),
            "ablation_mean": np.mean(ablation_runs),
            "ablation_std": np.std(ablation_runs, ddof=1),
            "rmse_runs": np.array(rmse_runs),
            "ablation_runs": np.array(ablation_runs),
            "imp_runs": imp_runs,
        }
        log(
            f"  {name:<22} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  "
            f"RMSE={np.mean(rmse_runs):.2f}±{np.std(rmse_runs, ddof=1):.2f}  "
            f"ablation={np.mean(ablation_runs):.4f}"
        )
        sc_results[name] = method_results
        _save(f"superconductor_{_sanitize_ckpt_name(name)}", method_results=method_results)

    clear_checkpoints_by_prefix("sc_flat_", CKPT_DIR)

    # C7+F1: Wilcoxon signed-rank test and Cohen's d
    _log_pairwise_significance(sc_results, "DASH (MaxMin)", sc_methods, "Superconductor")

    import sklearn as _sklearn

    sc_results["_dataset"] = {
        "sklearn_version": _sklearn.__version__,
        "dataset": "superconduct",
        "source": "sklearn.datasets.fetch_openml(name='superconduct', version=1)",
    }
    _publish_results(sc_results, f"{OUT}/tables/superconductor.json", "real_superconductor", N_REPS, t0)
    plot_real_world_bar(sc_results, "superconductor")

    if cleanup:
        clear_checkpoints_by_prefix("superconductor_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Superconductor completed in {elapsed / 60:.1f} min")
    return sc_results


###############################################################################
# EXPERIMENT: Epsilon Sensitivity
###############################################################################


def experiment_epsilon_sensitivity(resume=False, cleanup=False):
    """Epsilon sensitivity sweep: epsilon ∈ {0.03, 0.05, 0.08, 0.10}.

    Trains population ONCE per rep, then varies epsilon on the same models
    to properly isolate the effect of the filtering threshold.
    """
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Epsilon Sensitivity")
    log("=" * 70)

    from dash_shap.core.population import generate_model_population
    from dash_shap.core.filtering import performance_filter

    EPS_VALUES = [0.03, 0.05, 0.08, 0.10]
    EPS_M = M
    eps_results = {
        eps: {
            "n_passing": [],
            "k_eff": [],
            "acc_runs": [],
            "eq_runs": [],
            "imp_runs": [],
        }
        for eps in EPS_VALUES
    }
    seq_budget = compute_thread_budget(n_outer=1)

    # Try to resume from latest batch checkpoint
    start_rep = 0
    if resume:
        for batch_end in range(N_REPS, 0, -10):
            ckpt_name = f"epsilon_sens_batch_{batch_end}"
            if _has(ckpt_name):
                cached = _load(ckpt_name)
                eps_results = cached["eps_results"]
                start_rep = cached["completed_reps"]
                log(f"  Resuming from rep {start_rep}")
                break

    for rep in range(start_rep, N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep + 1}/{N_REPS}")

        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
            N=5000, rho=0.9, seed=rep_seed
        )

        # Train population ONCE per rep
        models, val_scores, configs = generate_model_population(
            Xtr,
            ytr,
            Xv,
            yv,
            M=EPS_M,
            task="regression",
            n_jobs=seq_budget.n_inner,
            nthread=seq_budget.nthread,
            seed=rep_seed,
            verbose=False,
        )

        for eps in EPS_VALUES:
            # Stage 2: Filter at this epsilon
            filtered = performance_filter(val_scores, epsilon=eps, higher_is_better=True, verbose=False)
            eps_results[eps]["n_passing"].append(len(filtered))

            if len(filtered) < 2:
                log(f"  eps={eps}: only {len(filtered)} passed, skipping")
                continue

            # Stage 3: MaxMin diversity selection
            imp_vecs = get_preliminary_importance(models, filtered, Xexp, method="gain")
            filt_scores = {i: val_scores[i] for i in filtered}
            selected = greedy_maxmin_selection(imp_vecs, filt_scores, K=K, delta=DELTA, verbose=False)
            eps_results[eps]["k_eff"].append(len(selected))

            # Stage 4-5: Consensus SHAP (use Xexp, not Xte, to avoid data leakage)
            cons, all_shap = compute_consensus(
                models, selected, Xexp, seed=rep_seed, verbose=False, n_jobs=seq_budget.n_inner
            )
            _, _, _, imp = compute_diagnostics(all_shap)

            r, _ = dgp_agreement(imp, true_imp)
            eps_results[eps]["acc_runs"].append(r)
            eps_results[eps]["eq_runs"].append(within_group_equity(imp, grps))
            eps_results[eps]["imp_runs"].append(imp)

        del models, val_scores, configs

        if (rep + 1) % 10 == 0:
            _save(
                f"epsilon_sens_batch_{rep + 1}",
                eps_results=eps_results,
                completed_reps=rep + 1,
            )

    log(f"\n  {'ε':>6} {'Models Passing':>16} {'K_eff':>12} {'Stability':>10} {'Top-5':>8} {'Accuracy':>10}")
    log("  " + "=" * 70)
    for eps in EPS_VALUES:
        stab = importance_stability(eps_results[eps]["imp_runs"])
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(eps_results[eps]["imp_runs"], k=5)
        eps_results[eps]["stability"] = stab
        eps_results[eps]["topk5_stability"] = topk5
        log(
            f"  {eps:>6.2f} {np.mean(eps_results[eps]['n_passing']):>12.1f}±"
            f"{np.std(eps_results[eps]['n_passing']):<4.1f}"
            f"{np.mean(eps_results[eps]['k_eff']):>8.1f}±"
            f"{np.std(eps_results[eps]['k_eff']):<4.1f}"
            f"{stab:>10.4f}"
            f"{topk5:>8.4f}"
            f"{np.mean(eps_results[eps]['acc_runs']):>10.4f}"
        )

    _publish_results(eps_results, f"{OUT}/tables/epsilon_sensitivity.json", "epsilon_sensitivity", N_REPS, t0)
    plot_epsilon_sensitivity(eps_results)

    if cleanup:
        clear_checkpoints_by_prefix("epsilon_sens_batch_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Epsilon sensitivity completed in {elapsed / 60:.1f} min")
    return eps_results


###############################################################################
# EXPERIMENT: Ablation Studies
###############################################################################


def _run_ablation_rho(abl_rho, ablations, abl_defaults, n_jobs_inner, *, nthread=1):
    """Run all ablation sweeps for a single rho level."""
    ABL_N_REPS = N_REPS
    log(f"\n{'=' * 60}")
    log(f"Ablation at ρ = {abl_rho}")
    log(f"{'=' * 60}")
    rho_results = {}
    for param_name, values in ablations.items():
        log(f"\n--- Ablation: {param_name} ---")
        rho_results[param_name] = {}
        for val in values:
            p = abl_defaults.copy()
            p[param_name] = val
            log(
                f"  {param_name}={val}  ",
            )

            imp_runs, acc_runs = [], []
            for rep in range(ABL_N_REPS):
                rep_seed = SEED + rep
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
                    N=5000, rho=abl_rho, seed=rep_seed
                )

                dm = DASHPipeline(
                    M=p["M"],
                    K=p["K"],
                    epsilon=p["eps"],
                    delta=p["delta"],
                    selection_method="maxmin",
                    n_jobs=n_jobs_inner,
                    nthread=nthread,
                    seed=rep_seed,
                    verbose=False,
                )
                try:
                    dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=make_feature_names())
                except ValueError:
                    continue
                imp = dm.global_importance_
                r, _ = dgp_agreement(imp, true_imp)
                acc_runs.append(r)
                imp_runs.append(imp)
                del dm

            if len(imp_runs) >= 2:
                stab = importance_stability(imp_runs)
                topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
                rho_results[param_name][val] = {
                    "stability": stab,
                    "topk5_stability": topk5,
                    "topk5_se": topk5_se,
                    "topk5_ci_lo": topk5_ci_lo,
                    "topk5_ci_hi": topk5_ci_hi,
                    "accuracy_mean": np.mean(acc_runs),
                    "accuracy_std": np.std(acc_runs, ddof=1),
                    "n_successful": len(imp_runs),
                }
                log(
                    f"    stab={stab:.4f}  topk5={topk5:.4f}  acc={np.mean(acc_runs):.4f}  ({len(imp_runs)}/{ABL_N_REPS} reps)"
                )
            else:
                rho_results[param_name][val] = {
                    "stability": float("nan"),
                    "accuracy_mean": float("nan"),
                    "accuracy_std": float("nan"),
                    "n_successful": len(imp_runs),
                }
                log(f"    SKIPPED — only {len(imp_runs)}/{ABL_N_REPS} reps passed filter")

    _save(f"ablation_rho_{abl_rho}", rho_results=rho_results)
    return abl_rho, rho_results


def experiment_ablation(resume=False, cleanup=False):
    """Ablation studies: one parameter at a time, across multiple rho levels.

    Rho levels run in parallel via joblib.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Ablation Studies")
    log("=" * 70)

    ABL_DEFAULTS = {"M": M, "K": K, "eps": EPSILON, "delta": DELTA}
    ABL_RHOS = [0.0, 0.9, 0.95]

    ablations = {
        "M": [50, 100, 200, 500],
        "K": [5, 10, 20, 30, 50],
        "eps": [0.01, 0.03, 0.05, 0.08, 0.10],
        "delta": [0.01, 0.05, 0.10, 0.20],
    }

    abl_results = {rho: {} for rho in ABL_RHOS}

    pending_rhos = []
    for abl_rho in ABL_RHOS:
        ckpt_name = f"ablation_rho_{abl_rho}"
        if resume and _has(ckpt_name):
            log(f"  Resuming: loaded checkpoint for ρ={abl_rho}")
            abl_results[abl_rho] = _load(ckpt_name)["rho_results"]
        else:
            pending_rhos.append(abl_rho)

    if pending_rhos:
        n_rho = len(pending_rhos)
        budget = compute_thread_budget(n_outer=n_rho)
        n_jobs_inner = budget.n_inner
        nthread = budget.nthread
        log(
            f"  Running {n_rho} rho levels in parallel ({budget.n_outer * budget.n_inner * budget.nthread} cores, {n_jobs_inner} per level, nthread={nthread})"
        )

        results_list = Parallel(n_jobs=n_rho, backend="loky")(
            delayed(_run_ablation_rho)(rho, ablations, ABL_DEFAULTS, n_jobs_inner, nthread=nthread)
            for rho in pending_rhos
        )
        for rho, rho_results in results_list:
            abl_results[rho] = rho_results
        _shutdown_loky_workers()

    _publish_results(abl_results, f"{OUT}/tables/ablation.json", "ablation", N_REPS, t0)

    # Generate publication figure
    plot_ablation_sensitivity(abl_results)

    if cleanup:
        clear_checkpoints_by_prefix("ablation_rho_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Ablation completed in {elapsed / 60:.1f} min")
    return abl_results


###############################################################################
# EXPERIMENT: Variance Decomposition (F1 fix)
###############################################################################


def experiment_variance_decomposition(resume=False, cleanup=False) -> dict:
    """Variance decomposition: separates data-sampling vs model-selection variance."""
    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Variance Decomposition")
    log("=" * 70)

    VD_RHO = 0.9
    feature_names = make_feature_names()

    conditions = {
        "data_fixed": "Fix data seed, vary model seeds → isolates model-selection variance",
        "model_fixed": "Fix model seed, vary data seeds → isolates data-sampling variance",
        "both_varied": "Vary both → total variance (reference)",
    }

    methods = ["Single Best", "DASH (MaxMin)"]
    results = {cond: {m: [] for m in methods} for cond in conditions}
    seq_budget = compute_thread_budget(n_outer=1)

    start_rep = 0
    if resume:
        for batch_end in range(N_REPS, 0, -10):
            ckpt_name = f"variance_decomp_batch_{batch_end}"
            if _has(ckpt_name):
                cached = _load(ckpt_name)
                results = cached["results"]
                start_rep = cached["completed_reps"]
                log(f"  Resuming from rep {start_rep}")
                break

    for rep in range(start_rep, N_REPS):
        log(f"  Rep {rep + 1}/{N_REPS}")

        for cond in conditions:
            if cond == "data_fixed":
                data_seed, model_seed = SEED, SEED + 1000 + rep
            elif cond == "model_fixed":
                data_seed, model_seed = SEED + rep, SEED
            else:  # both_varied
                data_seed, model_seed = SEED + rep, SEED + rep

            Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
                N=5000, rho=VD_RHO, seed=data_seed
            )

            # Single Best
            sb = SingleBestBaseline(
                n_trials=N_TRIALS_SB, seed=model_seed, n_jobs=seq_budget.n_inner, nthread=seq_budget.nthread
            )
            sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=model_seed)
            results[cond]["Single Best"].append(sb.global_importance_)

            # DASH (MaxMin)
            dm = DASHPipeline(
                M=M,
                K=K,
                epsilon=EPSILON,
                delta=DELTA,
                selection_method="maxmin",
                n_jobs=seq_budget.n_inner,
                nthread=seq_budget.nthread,
                seed=model_seed,
                verbose=False,
            )
            dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
            results[cond]["DASH (MaxMin)"].append(dm.global_importance_)
            del sb, dm

        if (rep + 1) % 10 == 0:
            _save(f"variance_decomp_batch_{rep + 1}", results=results, completed_reps=rep + 1)

    # Compute stability for each condition × method
    log(f"\n  {'Condition':<16} {'Method':<20} {'Stability':>10} {'Top-5':>8}")
    log("  " + "=" * 58)
    summary = {}
    for cond in conditions:
        summary[cond] = {}
        for m in methods:
            stab = importance_stability(results[cond][m])
            topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(results[cond][m], k=5)
            summary[cond][m] = {"stability": stab, "topk5_stability": topk5}
            log(f"  {cond:<16} {m:<20} {stab:>10.4f} {topk5:>8.4f}")

    # Variance decomposition ratios
    # CAVEAT: (1 - stability) is a proxy for instability, not a proper
    # variance.  Stability is pairwise Spearman ρ, so the "ratios" below
    # are indicative of relative contribution but do not satisfy an exact
    # additive decomposition (model_var + data_var ≠ total_var in general).
    log("\n  Variance Decomposition Ratios (indicative — see caveat in code):")
    for m in methods:
        total_var = 1.0 - summary["both_varied"][m]["stability"]
        model_var = 1.0 - summary["data_fixed"][m]["stability"]
        data_var = 1.0 - summary["model_fixed"][m]["stability"]
        summary_ratios = {}
        if total_var > 0:
            model_frac = model_var / total_var
            data_frac = data_var / total_var
            log(f"    {m}: model-selection={model_frac:.1%}, data-sampling={data_frac:.1%} of total instability")
            summary_ratios = {
                "model_selection_frac": model_frac,
                "data_sampling_frac": data_frac,
            }
        else:
            log(f"    {m}: total variance ≈ 0 (perfectly stable)")
        for cond in conditions:
            summary[cond][m]["decomposition"] = summary_ratios

    _publish_results(summary, f"{OUT}/tables/variance_decomposition.json", "variance_decomposition", N_REPS, t0)

    if cleanup:
        clear_checkpoints_by_prefix("variance_decomp_batch_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Variance decomposition completed in {elapsed / 60:.1f} min")
    return summary


###############################################################################
# ASYMMETRIC CAUSAL DGP EXPERIMENT
###############################################################################


def _run_asymmetric_rho(rho, method_names, n_asym_reps, n_jobs_inner, do_cleanup, *, nthread=1):
    """Run all reps for a single rho level of the asymmetric DGP experiment."""
    feature_names_asym = ["f0", "f1"]
    log(f"\n  rho={rho}")
    rho_imps = {m: [] for m in method_names}

    for rep in range(n_asym_reps):
        rep_seed = SEED + rep
        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, true_imp, _ = generate_synthetic_asymmetric(
            N=5000, rho=rho, seed=rep_seed
        )

        sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=n_jobs_inner, nthread=nthread)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        rho_imps["Single Best"].append(sb.global_importance_)

        sr = StochasticRetrainBaseline(N=K, seed=rep_seed, n_jobs=n_jobs_inner, nthread=nthread)
        sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
        rho_imps["Stochastic Retrain"].append(sr.global_importance_)

        dm = DASHPipeline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            selection_method="maxmin",
            n_jobs=n_jobs_inner,
            nthread=nthread,
            seed=rep_seed,
            verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names_asym)
        rho_imps["DASH (MaxMin)"].append(dm.global_importance_)

        lsm = LargeSingleModelBaseline(seed=rep_seed, nthread=nthread)
        lsm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
        rho_imps["Large Single Model"].append(lsm.global_importance_)
        del sb, sr, dm, lsm

    ckpt_name = f"asym_dgp_rho{rho}"
    if do_cleanup:
        clear_checkpoints_by_prefix(f"asym_dgp_rho{rho}", CKPT_DIR)
    else:
        _save(ckpt_name, rho_imps=rho_imps, completed_reps=n_asym_reps)

    # Compute metrics
    true_imp_fixed = np.array([1.0, 0.0])
    rho_summary = {}
    log(f"  {'Method':<22} {'Stab':>8} {'Bias(f0)':>10} {'Leak(f1)':>10}")
    log("  " + "=" * 55)
    for m_name, imp_list in rho_imps.items():
        if not imp_list:
            continue
        stab = importance_stability(imp_list)
        imp_arr = np.array(imp_list)
        row_sums = imp_arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        imp_norm = imp_arr / row_sums
        mean_imp = imp_norm.mean(axis=0)
        bias_f0 = float(abs(mean_imp[0] - true_imp_fixed[0]))
        leak_f1 = float(mean_imp[1])
        rho_summary[m_name] = {
            "stability": stab,
            "bias_f0": bias_f0,
            "passive_leak_f1": leak_f1,
            "mean_importance": mean_imp.tolist(),
        }
        log(f"  {m_name:<22} {stab:>8.4f} {bias_f0:>10.4f} {leak_f1:>10.4f}")

    return rho, rho_summary


def experiment_asymmetric_dgp(resume=False, cleanup=False) -> dict:
    """Asymmetric causal DGP: f0 is causal, f1 is a passive correlate.

    Tests whether DASH over-equalizes when one feature has all the signal.
    Sweeps rho in {0.5, 0.7, 0.9, 0.95}; measures stability, attribution
    bias for f0, and passive leak to f1.
    Rho levels run in parallel via joblib.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Asymmetric Causal DGP")
    log("=" * 70)

    RHO_LEVELS = [0.5, 0.7, 0.9, 0.95]
    N_ASYM_REPS = 20
    method_names = ["Single Best", "Stochastic Retrain", "DASH (MaxMin)", "Large Single Model"]

    all_results = {}

    # Separate resumed vs pending rho levels
    pending_rhos = []
    for rho in RHO_LEVELS:
        ckpt_name = f"asym_dgp_rho{rho}"
        if resume and _has(ckpt_name):
            cached = _load(ckpt_name)
            rho_imps = cached["rho_imps"]
            # Compute summary from loaded data
            true_imp_fixed = np.array([1.0, 0.0])
            rho_summary = {}
            for m_name, imp_list in rho_imps.items():
                if not imp_list:
                    continue
                stab = importance_stability(imp_list)
                imp_arr = np.array(imp_list)
                row_sums = imp_arr.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                imp_norm = imp_arr / row_sums
                mean_imp = imp_norm.mean(axis=0)
                bias_f0 = float(abs(mean_imp[0] - true_imp_fixed[0]))
                leak_f1 = float(mean_imp[1])
                rho_summary[m_name] = {
                    "stability": stab,
                    "bias_f0": bias_f0,
                    "passive_leak_f1": leak_f1,
                    "mean_importance": mean_imp.tolist(),
                }
            all_results[rho] = rho_summary
            log(f"  Resuming: loaded checkpoint for ρ={rho}")
        else:
            pending_rhos.append(rho)

    if pending_rhos:
        n_rho = len(pending_rhos)
        budget = compute_thread_budget(n_outer=n_rho)
        n_jobs_inner = budget.n_inner
        nthread = budget.nthread
        log(
            f"  Running {n_rho} rho levels in parallel ({budget.n_outer * budget.n_inner * budget.nthread} cores, {n_jobs_inner} per level, nthread={nthread})"
        )

        results_list = Parallel(n_jobs=n_rho, backend="loky")(
            delayed(_run_asymmetric_rho)(rho, method_names, N_ASYM_REPS, n_jobs_inner, cleanup, nthread=nthread)
            for rho in pending_rhos
        )
        for rho, rho_summary in results_list:
            all_results[rho] = rho_summary
        _shutdown_loky_workers()

    _publish_results(all_results, f"{OUT}/tables/asymmetric_dgp.json", "asymmetric_dgp", N_REPS, t0)

    elapsed = time.time() - t0
    log(f"\n  Asymmetric DGP experiment completed in {elapsed / 60:.1f} min")
    return all_results


###############################################################################
# CROSSED VARIANCE DECOMPOSITION (ANOVA)
###############################################################################


def _run_crossed_data_seed(di, data_seed, model_seeds, feature_names, n_jobs_inner, *, nthread=1):
    """Run all model seeds for a single data seed in the crossed ANOVA design."""
    VD_RHO = 0.9
    log(f"  Data seed {di + 1}/7 (seed={data_seed})")
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
        N=5000, rho=VD_RHO, seed=data_seed
    )
    cell_results = {}
    for mi, model_seed in enumerate(model_seeds):
        sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=model_seed, n_jobs=n_jobs_inner, nthread=nthread)
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=model_seed)
        cell_results[("Single Best", di, mi)] = sb.global_importance_
        del sb

        dm = DASHPipeline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            selection_method="maxmin",
            n_jobs=n_jobs_inner,
            nthread=nthread,
            seed=model_seed,
            verbose=False,
        )
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        cell_results[("DASH (MaxMin)", di, mi)] = dm.global_importance_
        del dm

    return di, cell_results


def experiment_variance_decomposition_crossed(resume=False, cleanup=False) -> dict:
    """Crossed R×R variance decomposition: exact ANOVA replacing 1-stability proxy.

    Uses a fully crossed 7×7 design (7 data seeds × 7 model seeds = 49 cells)
    and two-way ANOVA to decompose importance variance into data-sampling,
    model-selection, and residual components.
    Data seeds run in parallel via joblib.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Crossed Variance Decomposition (ANOVA)")
    log("=" * 70)

    R = 7  # 7×7 = 49 cells
    feature_names = make_feature_names()
    data_seeds = [SEED + i for i in range(R)]
    model_seeds = [SEED + 1000 + i for i in range(R)]

    methods = ["Single Best", "DASH (MaxMin)"]
    importances = {m: {} for m in methods}

    # Resume: find latest completed data-seed checkpoint
    start_di = 0
    if resume:
        for di_check in range(R - 1, -1, -1):
            ckpt_name = f"variance_decomp_crossed_di_{di_check}"
            if has_checkpoint(ckpt_name, CKPT_DIR):
                cached = load_checkpoint(ckpt_name, CKPT_DIR)
                importances = cached["importances"]
                start_di = di_check + 1
                log(f"  Resuming: loaded checkpoint through data seed {di_check + 1}/{R}")
                break

    pending_dis = [(di, ds) for di, ds in enumerate(data_seeds) if di >= start_di]

    if pending_dis:
        n_pending = len(pending_dis)
        budget = compute_thread_budget(n_outer=n_pending)
        n_jobs_inner = budget.n_inner
        nthread = budget.nthread
        log(
            f"  Running {n_pending} data seeds in parallel ({budget.n_outer * budget.n_inner * budget.nthread} cores, {n_jobs_inner} per seed, nthread={nthread})"
        )
        log(f"  Running {R}×{R}={R * R} cells...")

        results_list = Parallel(n_jobs=n_pending, backend="loky")(
            delayed(_run_crossed_data_seed)(di, data_seed, model_seeds, feature_names, n_jobs_inner, nthread=nthread)
            for di, data_seed in pending_dis
        )

        for di, cell_results in results_list:
            for (method, d_i, m_i), imp in cell_results.items():
                importances[method][(d_i, m_i)] = imp
            save_checkpoint(
                f"variance_decomp_crossed_di_{di}",
                checkpoint_dir=CKPT_DIR,
                importances=importances,
            )
        _shutdown_loky_workers()

    # Compute ANOVA decomposition for each method
    log(f"\n  {'Method':<22} {'Data%':>8} {'Model%':>8} {'Residual%':>10}")
    log("  " + "=" * 52)
    summary = {}
    for m in methods:
        result = anova_decomposition(importances[m])
        summary[m] = result
        log(
            f"  {m:<22} {result['data_var_frac']:>8.1%} "
            f"{result['model_var_frac']:>8.1%} "
            f"{result['residual_var_frac']:>10.1%}"
        )

    _publish_results(
        summary, f"{OUT}/tables/variance_decomposition_crossed.json", "variance_decomposition_crossed", N_REPS, t0
    )

    elapsed = time.time() - t0
    log(f"\n  Crossed variance decomposition completed in {elapsed / 60:.1f} min")
    return summary


###############################################################################
# SUCCESS CRITERIA
###############################################################################


def check_success_criteria(
    sweep_results,
    epsilon_results=None,
    nonlinear_results=None,
    sig_results=None,
    sc_results=None,
    cal_results=None,
    bc_results=None,
    vardecomp_results=None,
):
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
        1
        for rho in rho_levels
        if sweep_results[rho]["DASH (MaxMin)"]["stability"] > sweep_results[rho]["Single Best"]["stability"]
    )
    passed = n_wins >= 4
    results.append(passed)
    log(f"  1. Stability wins: {n_wins}/{len(rho_levels)} ({'PASS' if passed else 'FAIL'}, need >=80%)")

    # 2. DGP agreement at rho=0.9 (relative to Single Best baseline)
    acc_09 = sweep_results[0.9]["DASH (MaxMin)"]["accuracy_mean"]
    sb_acc_09 = sweep_results[0.9]["Single Best"]["accuracy_mean"]
    passed = acc_09 >= sb_acc_09
    results.append(passed)
    log(
        f"  2. DGP agreement at ρ=0.9: DASH={acc_09:.4f} vs SB={sb_acc_09:.4f} "
        f"({'PASS' if passed else 'FAIL'}, DASH >= SB)"
    )

    # 3. Equity wins
    n_eq_wins = sum(
        1
        for rho in rho_levels
        if sweep_results[rho]["DASH (MaxMin)"]["equity_mean"] < sweep_results[rho]["Single Best"]["equity_mean"]
    )
    passed = n_eq_wins >= 4
    results.append(passed)
    log(f"  3. Equity wins: {n_eq_wins}/{len(rho_levels)} ({'PASS' if passed else 'FAIL'}, need >=80%)")

    # 4. Safety control at rho=0
    rho0_dash = sweep_results[0.0]["DASH (MaxMin)"]["accuracy_mean"]
    rho0_sb = sweep_results[0.0]["Single Best"]["accuracy_mean"]
    passed = abs(rho0_dash - rho0_sb) < 0.1
    results.append(passed)
    log(f"  4. ρ=0 control: DASH dgp={rho0_dash:.4f}, SB dgp={rho0_sb:.4f} ({'PASS' if passed else 'FAIL'}, gap < 0.1)")

    # 5. K_eff increases with epsilon
    if epsilon_results is not None:
        eps_vals = sorted(epsilon_results.keys(), key=lambda x: float(x))
        keffs = []
        for e in eps_vals:
            k_data = epsilon_results[e].get("k_eff", [])
            keffs.append(np.mean(k_data) if k_data else 0)
        passed = all(keffs[i] <= keffs[i + 1] for i in range(len(keffs) - 1))
        results.append(passed)
        keffs_str = [f"{k:.1f}" for k in keffs]
        log(f"  5. K_eff monotonic with epsilon: {keffs_str} ({'PASS' if passed else 'FAIL'})")
    else:
        log("  5. K_eff monotonicity: SKIP (no epsilon_results)")

    # 6. Nonlinear DGP: DASH > SB stability at rho=0.9
    if nonlinear_results is not None and 0.9 in nonlinear_results:
        nl_09 = nonlinear_results[0.9]
        if "DASH (MaxMin)" in nl_09 and "Single Best" in nl_09:
            d_stab = nl_09["DASH (MaxMin)"]["stability"]
            s_stab = nl_09["Single Best"]["stability"]
            passed = d_stab > s_stab
            results.append(passed)
            log(f"  6. Nonlinear (ρ=0.9): DASH={d_stab:.4f} vs SB={s_stab:.4f} ({'PASS' if passed else 'FAIL'})")
        else:
            log("  6. Nonlinear DGP: SKIP (missing methods in results)")
    else:
        log("  6. Nonlinear DGP: SKIP (no nonlinear_results)")

    # 7. Significance: enough tests significant
    if sig_results is not None:
        n_sig = sum(1 for t in sig_results if t.get("significant", False))
        n_total = len(sig_results)
        passed = n_total > 0 and n_sig >= n_total * 0.5
        results.append(passed)
        log(f"  7. Significance: {n_sig}/{n_total} tests significant ({'PASS' if passed else 'FAIL'}, need >=50%)")
    else:
        log("  7. Significance tests: SKIP (no sig_results)")

    # 8. Superconductor: DASH stability > SB
    if sc_results is not None:
        if "DASH (MaxMin)" in sc_results and "Single Best" in sc_results:
            d_stab = sc_results["DASH (MaxMin)"]["stability"]
            s_stab = sc_results["Single Best"]["stability"]
            passed = d_stab > s_stab
            results.append(passed)
            log(f"  8. Superconductor: DASH={d_stab:.4f} vs SB={s_stab:.4f} ({'PASS' if passed else 'FAIL'})")
        else:
            log("  8. Superconductor: SKIP (missing methods)")
    else:
        log("  8. Superconductor: SKIP (no sc_results)")

    # 9. California Housing: DASH stability > SB
    if cal_results is not None:
        if "DASH (MaxMin)" in cal_results and "Single Best" in cal_results:
            d_stab = cal_results["DASH (MaxMin)"]["stability"]
            s_stab = cal_results["Single Best"]["stability"]
            passed = d_stab > s_stab
            results.append(passed)
            log(f"  9. California Housing: DASH={d_stab:.4f} vs SB={s_stab:.4f} ({'PASS' if passed else 'FAIL'})")
        else:
            log("  9. California Housing: SKIP (missing methods)")
    else:
        log("  9. California Housing: SKIP (no cal_results)")

    # 10. Breast Cancer: DASH stability > SB
    if bc_results is not None:
        if "DASH (MaxMin)" in bc_results and "Single Best" in bc_results:
            d_stab = bc_results["DASH (MaxMin)"]["stability"]
            s_stab = bc_results["Single Best"]["stability"]
            passed = d_stab > s_stab
            results.append(passed)
            log(f" 10. Breast Cancer: DASH={d_stab:.4f} vs SB={s_stab:.4f} ({'PASS' if passed else 'FAIL'})")
        else:
            log(" 10. Breast Cancer: SKIP (missing methods)")
    else:
        log(" 10. Breast Cancer: SKIP (no bc_results)")

    # 11. Variance decomposition: DASH model-var < SB model-var
    # Structure: {condition: {method: {'stability': float}}}
    # Model-selection instability = 1 - stability under 'data_fixed' condition
    if vardecomp_results is not None:
        data_fixed = vardecomp_results.get("data_fixed", {})
        dash_df = data_fixed.get("DASH (MaxMin)", {})
        sb_df = data_fixed.get("Single Best", {})
        if "stability" in dash_df and "stability" in sb_df:
            d_model = 1.0 - dash_df["stability"]
            s_model = 1.0 - sb_df["stability"]
            passed = d_model < s_model
            results.append(passed)
            log(
                f" 11. Variance decomp: DASH model-instab={d_model:.4f} vs SB={s_model:.4f} "
                f"({'PASS' if passed else 'FAIL'})"
            )
        else:
            log(" 11. Variance decomposition: SKIP (missing data_fixed stability)")
    else:
        log(" 11. Variance decomposition: SKIP (no vardecomp_results)")

    # Summary
    n_passed = sum(results)
    n_total = len(results)
    log(f"\n  Overall: {n_passed}/{n_total} criteria passed")

    rho09 = sweep_results[0.9]
    log("\n  Stability at ρ=0.9:")
    for n in rho09:
        log(f"    {n:<20} {rho09[n]['stability']:.4f}")

    dash_stab = rho09["DASH (MaxMin)"]["stability"]
    sb_stab = rho09["Single Best"]["stability"]
    log(f"\n  DASH improvement over Single Best: +{dash_stab - sb_stab:.4f}")
    if "Large Single Model" in rho09:
        lsm_stab = rho09["Large Single Model"]["stability"]
        log(f"  DASH improvement over LSM:         +{dash_stab - lsm_stab:.4f}")

    return results


###############################################################################
# EXPERIMENT: Background Size Sensitivity (REVIEW_v7 B5)
###############################################################################


def _run_single_bg_rep(B, rep, feature_names, *, nthread=1):
    """Run one (B, rep) pair for background sensitivity. Returns scalars only."""
    rep_seed = SEED + rep
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
        N=5000, rho=0.9, seed=rep_seed
    )
    dm = DASHPipeline(
        M=M,
        K=K,
        epsilon=EPSILON,
        delta=DELTA,
        selection_method="maxmin",
        n_jobs=1,
        nthread=nthread,
        background_size=B,
        seed=rep_seed,
        verbose=False,
    )
    dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
    imp = dm.global_importance_
    r, _ = dgp_agreement(imp, true_imp)
    eq = within_group_equity(imp, grps)
    del dm
    return B, rep, {"imp": imp, "acc": float(r), "eq": float(eq)}


def experiment_background_sensitivity(resume=False, cleanup=False):
    """Sweep background dataset size B ∈ {50, 100, 200, 500} at ρ=0.9."""
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Background Size Sensitivity at ρ=0.9")
    log("=" * 70)

    B_VALUES = [50, 100, 200, 500]
    feature_names = make_feature_names()
    results = {}

    # Collect per-B data (resumed or pending)
    partial_data = {B: {"imp_runs": [], "acc_runs": [], "eq_runs": []} for B in B_VALUES}
    pending_pairs = []

    for B in B_VALUES:
        # Check for old-style per-B checkpoint (full B completed)
        old_ckpt = f"background_B_{B}"
        if resume and _has(old_ckpt):
            log(f"  Resuming: loaded aggregate checkpoint for B={B}")
            results[str(B)] = _load(old_ckpt)["b_results"]
            continue

        for rep in range(N_REPS):
            ckpt_key = f"bg_flat_{B}_rep_{rep}"
            if resume and _has(ckpt_key):
                data = _load(ckpt_key)
                partial_data[B]["imp_runs"].append(data["imp"])
                partial_data[B]["acc_runs"].append(data["acc"])
                partial_data[B]["eq_runs"].append(data["eq"])
            else:
                pending_pairs.append((B, rep))

    n_total = len(B_VALUES) * N_REPS
    n_pending = len(pending_pairs)
    n_resumed = n_total - n_pending - sum(1 for B in B_VALUES if str(B) in results) * N_REPS
    log(
        f"  {n_pending} (B, rep) pairs pending"
        + (f" ({n_resumed} resumed from per-rep checkpoints)" if n_resumed else "")
    )

    if pending_pairs:
        n_workers = compute_rep_worker_budget(n_work=len(pending_pairs))
        nthread = 1
        log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_bg_rep)(B, rep, feature_names, nthread=nthread) for B, rep in pending_pairs
        )

        for B_out, rep_out, per_rep in results_list:
            partial_data[B_out]["imp_runs"].append(per_rep["imp"])
            partial_data[B_out]["acc_runs"].append(per_rep["acc"])
            partial_data[B_out]["eq_runs"].append(per_rep["eq"])
            _save(f"bg_flat_{B_out}_rep_{rep_out}", **per_rep)
        _shutdown_loky_workers()

    # Aggregate per B
    for B in B_VALUES:
        if str(B) in results:
            continue  # already loaded from old-style checkpoint
        imp_runs = partial_data[B]["imp_runs"]
        acc_runs = partial_data[B]["acc_runs"]
        eq_runs = partial_data[B]["eq_runs"]
        if len(imp_runs) < 2:
            log(f"  B={B}: only {len(imp_runs)} reps completed, skipping")
            continue
        stab, stab_se, stab_ci_lo, stab_ci_hi = stability_bootstrap_ci(imp_runs)
        topk5, topk5_se, topk5_ci_lo, topk5_ci_hi = topk_stability_bootstrap_ci(imp_runs, k=5)
        results[str(B)] = {
            "stability": stab,
            "stability_se": stab_se,
            "stability_ci_lo": stab_ci_lo,
            "stability_ci_hi": stab_ci_hi,
            "topk5_stability": topk5,
            "topk5_se": topk5_se,
            "topk5_ci_lo": topk5_ci_lo,
            "topk5_ci_hi": topk5_ci_hi,
            "accuracy_mean": float(np.mean(acc_runs)),
            "accuracy_std": float(np.std(acc_runs, ddof=1)),
            "equity_mean": float(np.mean(eq_runs)),
            "equity_std": float(np.std(eq_runs, ddof=1)),
        }
        log(
            f"  B={B:<4} stab={stab:.4f}±{stab_se:.4f}  topk5={topk5:.4f}  "
            f"acc={np.mean(acc_runs):.4f}  eq={np.mean(eq_runs):.4f}"
        )

    _publish_results(results, f"{OUT}/tables/background_sensitivity.json", "background_sensitivity", N_REPS, t0)

    if cleanup:
        clear_checkpoints_by_prefix("background_B_", CKPT_DIR)
        clear_checkpoints_by_prefix("bg_flat_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Background sensitivity completed in {elapsed / 60:.1f} min")
    return results


###############################################################################
# EXPERIMENT: Success Criteria (m7 — registered as runnable experiment)
###############################################################################


def experiment_success_criteria(resume=False, cleanup=False):
    """Run linear_sweep (if needed) then evaluate pass/fail success criteria."""
    sweep_results = experiment_linear_sweep(resume=resume, cleanup=cleanup)
    check_success_criteria(sweep_results)
    return sweep_results


###############################################################################
# EXPERIMENT: First-Mover Bias Visualization
###############################################################################


def experiment_first_mover_visualization(resume=False, cleanup=False):
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

    methods_to_run = ["Single Best", "Large Single Model", "DASH (MaxMin)"]
    method_importances = {m: [] for m in methods_to_run}
    seq_budget = compute_thread_budget(n_outer=1)

    start_rep = 0
    if resume:
        for batch_end in range(n_vis_reps, 0, -10):
            ckpt_name = f"first_mover_vis_batch_{batch_end}"
            if _has(ckpt_name):
                cached = _load(ckpt_name)
                method_importances = cached["results"]
                start_rep = cached["completed_reps"]
                log(f"  Resuming from rep {start_rep}")
                break

    for rep in range(start_rep, n_vis_reps):
        rep_seed = SEED + rep
        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
            N=5000, rho=rho, seed=rep_seed
        )

        # Single Best
        sb = SingleBestBaseline(
            n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=seq_budget.n_inner, nthread=seq_budget.nthread
        )
        sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        method_importances["Single Best"].append(sb.global_importance_[group_features])

        # Large Single Model
        lsm = LargeSingleModelBaseline(
            K=K,
            T_per_model=PAPER_CONFIG["T_PER_MODEL"],
            colsample_bytree=0.2,
            seed=rep_seed,
            nthread=seq_budget.nthread,
        )
        lsm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
        method_importances["Large Single Model"].append(lsm.global_importance_[group_features])

        # DASH (MaxMin)
        dash = DASHPipeline(
            M=M,
            K=K,
            epsilon=EPSILON,
            delta=DELTA,
            selection_method="maxmin",
            n_jobs=seq_budget.n_inner,
            nthread=seq_budget.nthread,
            seed=rep_seed,
            verbose=False,
        )
        dash.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        method_importances["DASH (MaxMin)"].append(dash.global_importance_[group_features])
        del sb, lsm, dash

        if (rep + 1) % 10 == 0:
            _save(
                f"first_mover_vis_batch_{rep + 1}",
                results=method_importances,
                completed_reps=rep + 1,
            )

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
        log(
            f"  {fname:<12} {avg_imp['Single Best'][i]:>8.4f} "
            f"{avg_imp['Large Single Model'][i]:>8.4f} "
            f"{avg_imp['DASH (MaxMin)'][i]:>8.4f}"
        )

    # Concentration metric: max/sum within group
    for m in methods_to_run:
        conc = np.max(avg_imp[m]) / (np.sum(avg_imp[m]) + 1e-10)
        log(f"  {m}: concentration = {conc:.3f} (1/5 = 0.200 is perfectly equitable)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(group_features))
    width = 0.22
    colors_list = [COLORS.get(m, "#333") for m in methods_to_run]
    for i, m in enumerate(methods_to_run):
        ax.bar(x + i * width, avg_imp[m], width, yerr=std_imp[m], label=m, color=colors_list[i], capsize=3, alpha=0.85)
    ax.axhline(
        y=true_per_feature,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"True importance ({true_per_feature:.2f})",
        alpha=0.6,
    )
    ax.set_xlabel("Feature (within correlated group 1)")
    ax.set_ylabel("Global importance (mean |SHAP|)")
    ax.set_title(f"First-Mover Bias: Importance Concentration at ρ={rho}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(group_labels)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/first_mover_concentration.pdf", dpi=150)
    fig.savefig(f"{OUT}/figures/first_mover_concentration.png", dpi=150)
    plt.close(fig)
    log(f"  Saved: {OUT}/figures/first_mover_concentration.pdf")

    # Save JSON for explorer notebook
    fmv_json: dict[str, dict[str, object]] = {}
    for m in methods_to_run:
        conc = float(np.max(avg_imp[m]) / (np.sum(avg_imp[m]) + 1e-10))
        fmv_json[m] = {
            "avg_importance": avg_imp[m],
            "std_importance": std_imp[m],
            "concentration": conc,
        }
    _publish_results(fmv_json, f"{OUT}/tables/first_mover_visualization.json", "first_mover_visualization", N_REPS, t0)

    if cleanup:
        clear_checkpoints_by_prefix("first_mover_vis_batch_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  First-mover visualization completed in {elapsed / 60:.1f} min")
    return avg_imp


###############################################################################
# EXPERIMENT: First-Mover Bias Isolation (IMPL_PLAN B3)
###############################################################################


def _run_single_fmb_rep(rep, n_estimator_levels, feature_names_loc, group_features):
    """Run one rep of first-mover bias isolation. Returns per-n_est concentrations."""
    import xgboost as xgb
    import shap

    rho = 0.9
    rep_seed = SEED + rep
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
        N=5000, rho=rho, seed=rep_seed
    )

    single_conc = {}
    dash_conc = {}
    for n_est in n_estimator_levels:
        # --- Single model: concentration grows with depth ---
        model = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=6,
            learning_rate=0.1,
            colsample_bytree=0.3,
            subsample=0.8,
            random_state=rep_seed,
        )
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xexp[:200], check_additivity=False)
        imp_single = np.mean(np.abs(sv), axis=0)
        grp_imp = imp_single[group_features]
        conc = float(np.max(grp_imp) / (np.sum(grp_imp) + 1e-10))
        single_conc[n_est] = conc
        del model, explainer

        # --- Independent ensemble: M small models, averaged ---
        n_per_model = max(10, n_est // 20)
        m_models = n_est // n_per_model
        imp_accum = np.zeros(len(feature_names_loc))
        for mi in range(m_models):
            m_seed = rep_seed * 10000 + mi
            mdl = xgb.XGBRegressor(
                n_estimators=n_per_model,
                max_depth=6,
                learning_rate=0.1,
                colsample_bytree=np.random.RandomState(m_seed).uniform(0.1, 0.5),
                subsample=0.8,
                random_state=m_seed,
            )
            mdl.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
            sv_m = shap.TreeExplainer(mdl).shap_values(Xexp[:200], check_additivity=False)
            imp_accum += np.mean(np.abs(sv_m), axis=0)
            del mdl
        imp_avg = imp_accum / m_models
        grp_imp_avg = imp_avg[group_features]
        conc_avg = float(np.max(grp_imp_avg) / (np.sum(grp_imp_avg) + 1e-10))
        dash_conc[n_est] = conc_avg

    return rep, single_conc, dash_conc


def experiment_first_mover_bias(resume=False, cleanup=False):
    """First-mover bias isolation: concentration grows with tree count.

    Trains a single XGBoost with increasing n_estimators and measures how
    importance concentration within a correlated group grows with depth.
    Compares against M independent models averaged at the same total tree
    count.  Produces a line plot: concentration vs n_estimators.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: First-Mover Bias Isolation")
    log("=" * 70)

    n_estimator_levels = [50, 100, 200, 500, 1000, 2000]
    n_bias_reps = 20  # Match N_REPS for publication-quality results
    feature_names_loc = make_feature_names()
    group_features = list(range(5))  # Group 1: features 0-4

    single_concentrations = {n: [] for n in n_estimator_levels}
    dash_concentrations = {n: [] for n in n_estimator_levels}

    # Resume: check for old-style batch checkpoints or per-rep flat checkpoints
    pending_reps = []
    start_rep = 0
    if resume:
        # Try old-style batch checkpoint first
        for batch_end in range(n_bias_reps, 0, -10):
            ckpt_name = f"first_mover_bias_batch_{batch_end}"
            if _has(ckpt_name):
                cached = _load(ckpt_name)
                single_concentrations = cached["single_concentrations"]
                dash_concentrations = cached["dash_concentrations"]
                start_rep = cached["completed_reps"]
                log(f"  Resuming from batch checkpoint at rep {start_rep}")
                break

        # Check per-rep flat checkpoints for remaining reps
        for rep in range(start_rep, n_bias_reps):
            ckpt_key = f"fmb_flat_rep_{rep}"
            if _has(ckpt_key):
                data = _load(ckpt_key)
                for n_est in n_estimator_levels:
                    single_concentrations[n_est].append(data["single_conc"][n_est])
                    dash_concentrations[n_est].append(data["dash_conc"][n_est])
            else:
                pending_reps.append(rep)
    else:
        pending_reps = list(range(n_bias_reps))

    n_pending = len(pending_reps)
    n_resumed = n_bias_reps - n_pending
    log(f"  {n_pending} reps pending" + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else ""))

    if pending_reps:
        n_workers = compute_rep_worker_budget(n_work=n_pending)
        log(f"  Running {n_workers} workers on {get_available_cores()} cores")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_fmb_rep)(rep, n_estimator_levels, feature_names_loc, group_features)
            for rep in pending_reps
        )

        for rep_out, single_conc, dash_conc in results_list:
            for n_est in n_estimator_levels:
                single_concentrations[n_est].append(single_conc[n_est])
                dash_concentrations[n_est].append(dash_conc[n_est])
            _save(f"fmb_flat_rep_{rep_out}", single_conc=single_conc, dash_conc=dash_conc)
        _shutdown_loky_workers()

    # Summarize
    summary = {}
    log(f"\n  {'n_estimators':>14} {'Single Conc':>14} {'Indep Conc':>14} {'Ratio':>8}")
    log("  " + "=" * 55)
    for n_est in n_estimator_levels:
        sc = np.mean(single_concentrations[n_est])
        dc = np.mean(dash_concentrations[n_est])
        log(f"  {n_est:>14} {sc:>14.4f} {dc:>14.4f} {sc / dc:>8.2f}")
        summary[str(n_est)] = {
            "single_concentration": sc,
            "single_concentration_std": float(np.std(single_concentrations[n_est], ddof=1)),
            "independent_concentration": dc,
            "independent_concentration_std": float(np.std(dash_concentrations[n_est], ddof=1)),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = n_estimator_levels
    sc_means = [np.mean(single_concentrations[n]) for n in xs]
    sc_stds = [np.std(single_concentrations[n], ddof=1) for n in xs]
    dc_means = [np.mean(dash_concentrations[n]) for n in xs]
    dc_stds = [np.std(dash_concentrations[n], ddof=1) for n in xs]

    ax.errorbar(
        xs,
        sc_means,
        yerr=sc_stds,
        fmt="s-",
        color="#e74c3c",
        label="Single Sequential Model",
        linewidth=2,
        capsize=4,
        markersize=7,
    )
    ax.errorbar(
        xs,
        dc_means,
        yerr=dc_stds,
        fmt="o-",
        color="#2ecc71",
        label="Independent Ensemble (averaged)",
        linewidth=2,
        capsize=4,
        markersize=7,
    )
    ax.axhline(y=0.2, color="black", linestyle="--", alpha=0.5, label="Perfect equity (1/5 = 0.20)")
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Concentration (max/sum within group)")
    ax.set_title("First-Mover Bias Isolation: Concentration Grows with Tree Count")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/first_mover_bias_isolation.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/figures/first_mover_bias_isolation.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("  Saved: figures/first_mover_bias_isolation.{png,pdf}")

    _publish_results(summary, f"{OUT}/tables/first_mover_bias.json", "first_mover_bias", N_REPS, t0)

    if cleanup:
        clear_checkpoints_by_prefix("first_mover_bias_batch_", CKPT_DIR)
        clear_checkpoints_by_prefix("fmb_flat_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  First-mover bias isolation completed in {elapsed / 60:.1f} min")
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
        elapsed = data.get("elapsed_s", 0)
        per_rep = elapsed / N_REPS if N_REPS > 0 else 0
        log(f"  {name:<24} {elapsed:>10.1f} {per_rep:>12.1f}")


###############################################################################
# EXPERIMENT REGISTRY
###############################################################################

###############################################################################
# EXPERIMENT: Extensions Sanity Check (V3)
###############################################################################


def experiment_extensions_sanity_check(resume=False, cleanup=False):
    """Lightweight (~2 min) check that Phase 0+1 extensions hold Paper 2 claims.

    Runs one rep at rho=0.9 (M=50, K=15), then asserts:
      1. Top-2 true features (f0, f5) are certified top-4 by Certification (Ext 9)
      2. Within-group π(f0 > f5) ≈ 0.5 ± 0.25 (features collinear, attribution split)
      3. Between-group π(f0 > noise_feat) > 0.7 (clear winner over noise feature)
      4. Confidence intervals contain point estimates for all features
      5. DASHResult serialization round-trip preserves shapes and dtype

    These checks make the Paper 2 claim "within-group π ≈ 0.5" a running CI test
    rather than a narrative claim. No dataset download required.
    """
    _ensure_dirs()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Extensions Sanity Check")
    log("=" * 70)

    import tempfile
    import pathlib
    from dash_shap.extensions import (
        robust_certification,
        confidence_intervals,
        partial_order,
    )
    from dash_shap.core.result import DASHResult

    rho = 0.9
    rng_seed = SEED
    seq_budget = compute_thread_budget(n_outer=1)

    log(f"  Generating synthetic data: N=2000, P=20, rho={rho}")
    Xtr, ytr, Xv, yv, Xexp, _, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(N=2000, rho=rho, seed=rng_seed)

    log("  Fitting DASHPipeline (M=50, K=15) ...")
    pipe = DASHPipeline(
        M=50,
        K=15,
        epsilon=EPSILON,
        delta=DELTA,
        seed=rng_seed,
        verbose=False,
        n_jobs=seq_budget.n_inner,
        nthread=seq_budget.nthread,
    )
    pipe.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=make_feature_names())
    result = pipe.result_

    log(f"  DASHResult: K={result.K}, n_ref={result.n_ref}, P={result.P}")

    failures = []

    # --- Check 1: Certification ---
    log("\n  [1] Certification check ...")
    cert = robust_certification(result, k_values=[1, 2, 3, 4, 5])
    # Group 0 features are f0..f4 (true importance ~0.4 each); f0 is the "first mover"
    # At least one of the true group features should be certified top-4
    top4_names = set(cert.certified.get(4, []))
    true_group0_names = {make_feature_names()[i] for i in grps[0]}
    overlap = top4_names & true_group0_names
    if not overlap:
        failures.append(f"FAIL: No group-0 features in certified top-4 (got {top4_names})")
    else:
        log(f"    PASS: certified top-4 includes group-0 features: {overlap}")

    # --- Check 2: Partial Order — within-group π ≈ 0.5 ---
    log("\n  [2] Partial order — within-group π check ...")
    po = partial_order(result, alpha=0.1, method="fraction")
    feat_names = list(result.feature_names)
    g0 = grps[0]
    # Take the top-2 features by importance within group 0
    g0_by_imp = sorted(g0, key=lambda i: -result.global_importance[i])
    f_a, f_b = g0_by_imp[0], g0_by_imp[1]
    pi_ab = po.confidence_matrix[f_a, f_b]
    log(f"    Within group-0: π({feat_names[f_a]}>{feat_names[f_b]}) = {pi_ab:.3f}")
    if not (0.25 <= pi_ab <= 0.75):
        failures.append(f"FAIL: within-group π = {pi_ab:.3f}, expected 0.25–0.75 (≈ 0.5)")
    else:
        log(f"    PASS: within-group π ≈ 0.5 (value={pi_ab:.3f})")

    # --- Check 3: Partial Order — between-group π > noise ---
    log("\n  [3] Partial order — between-group dominance check ...")
    # Top group-0 feature vs a low-importance feature from group 1
    g0_top = g0_by_imp[0]
    g3 = grps[3]  # last group has lower true importance
    g3_by_imp = sorted(g3, key=lambda i: result.global_importance[i])
    f_noise = g3_by_imp[0]  # least important in group 3
    pi_top_vs_noise = po.confidence_matrix[g0_top, f_noise]
    log(f"    Between-group: π({feat_names[g0_top]}>{feat_names[f_noise]}) = {pi_top_vs_noise:.3f}")
    if pi_top_vs_noise < 0.7:
        failures.append(f"FAIL: between-group π = {pi_top_vs_noise:.3f}, expected ≥ 0.7")
    else:
        log(f"    PASS: between-group dominance π = {pi_top_vs_noise:.3f} ≥ 0.7")

    # --- Check 4: CI contains point estimates ---
    log("\n  [4] Confidence interval containment check ...")
    ci = confidence_intervals(result, alpha=0.05, n_boot=200, seed=rng_seed)
    lower_ok = np.all(ci.importance_ci[:, 0] <= ci.importance_ci[:, 1] + 1e-9)
    upper_ok = np.all(ci.importance_ci[:, 1] <= ci.importance_ci[:, 2] + 1e-9)
    if not (lower_ok and upper_ok):
        failures.append("FAIL: CI does not contain point estimate for some features")
    else:
        log("    PASS: all importance CIs contain their point estimates")

    # --- Check 5: Serialization round-trip ---
    log("\n  [5] Serialization round-trip check ...")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sanity_result"
        result.save(path)
        loaded = DASHResult.load(path)
    shape_ok = loaded.all_shap_matrices.shape == result.all_shap_matrices.shape
    dtype_ok = loaded.all_shap_matrices.dtype == result.all_shap_matrices.dtype
    values_ok = np.allclose(loaded.all_shap_matrices, result.all_shap_matrices)
    if not (shape_ok and dtype_ok and values_ok):
        failures.append(
            f"FAIL: serialization round-trip mismatch (shape={shape_ok}, dtype={dtype_ok}, values={values_ok})"
        )
    else:
        log(f"    PASS: DASHResult({result.K}, {result.n_ref}, {result.P}) round-trips cleanly")

    # --- Summary ---
    log("\n" + "-" * 70)
    if failures:
        for f in failures:
            log(f"  {f}")
        log(f"\n  EXTENSIONS SANITY CHECK: {len(failures)} FAILURE(S)")
        raise AssertionError("Extensions sanity check failed:\n" + "\n".join(failures))
    else:
        log("  EXTENSIONS SANITY CHECK: ALL CHECKS PASSED")

    return {"status": "passed", "K": result.K, "n_failures": 0}


def plot_k_sweep_independence(k_values, results):
    """Line chart: stability vs K for DASH and SR with error bars."""
    _ensure_dirs()

    plot_methods = [
        ("DASH", "DASH (MaxMin)", "#2ecc71", "o"),
        ("RS", "Random Selection", "#d4ac0d", "s"),
        ("SR", "Stochastic Retrain", "#3498db", "^"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label, color, marker in plot_methods:
        stab = [results[k].get(key, {}).get("stability", float("nan")) for k in k_values]
        se = [results[k].get(key, {}).get("stability_se", 0) for k in k_values]
        ax.errorbar(
            k_values,
            stab,
            yerr=se,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
            capsize=4,
        )
    ax.set_xlabel("K (models selected)")
    ax.set_ylabel("Stability (mean \u00b1 SE)")
    ax.set_title("Stability vs K: DASH vs Random Selection vs Stochastic Retrain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/k_sweep_independence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {OUT}/figures/k_sweep_independence.png")


def _run_single_ksweep_pair(k_val, rep, feature_names, *, nthread=1):
    """Run one (k_val, rep) pair for k_sweep_independence. Returns scalars only."""
    import traceback

    rep_seed = SEED + rep
    Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
        N=5000, rho=0.9, seed=rep_seed
    )

    result = {"k_val": k_val, "rep": rep}

    # DASH (MaxMin)
    dm = DASHPipeline(
        M=M,
        K=k_val,
        epsilon=EPSILON,
        delta=DELTA,
        selection_method="maxmin",
        n_jobs=1,
        nthread=nthread,
        seed=rep_seed,
        verbose=False,
    )
    try:
        dm.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        imp = dm.global_importance_
        r, _ = dgp_agreement(imp, true_imp)
        result["dash_imp"] = imp
        result["dash_acc"] = float(r)
    except (ValueError, Exception):
        result["dash_imp"] = None
        result["dash_acc"] = None
        result["dash_error"] = traceback.format_exc()
    del dm

    # Random Selection baseline (was mislabeled "SR" — now correctly "RS")
    rs = RandomSelectionBaseline(
        M=M,
        K=k_val,
        epsilon=EPSILON,
        delta=DELTA,
        n_jobs=1,
        nthread=nthread,
        seed=rep_seed,
    )
    try:
        rs.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
        imp_rs = rs.global_importance_
        r_rs, _ = dgp_agreement(imp_rs, true_imp)
        result["rs_imp"] = imp_rs
        result["rs_acc"] = float(r_rs)
    except (ValueError, Exception):
        result["rs_imp"] = None
        result["rs_acc"] = None
        result["rs_error"] = traceback.format_exc()
    del rs

    # Stochastic Retrain (actual seed-averaging baseline)
    sr = StochasticRetrainBaseline(
        N=k_val,
        task="regression",
        n_jobs=1,
        nthread=nthread,
        seed=rep_seed,
    )
    try:
        sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
        imp_sr = sr.global_importance_
        r_sr, _ = dgp_agreement(imp_sr, true_imp)
        result["sr_imp"] = imp_sr
        result["sr_acc"] = float(r_sr)
    except (ValueError, Exception):
        result["sr_imp"] = None
        result["sr_acc"] = None
        result["sr_error"] = traceback.format_exc()
    del sr

    return result


def experiment_k_sweep_independence(resume=False, cleanup=False) -> dict:
    """Ablation: independence vs. K — stability scaling with ensemble size.

    Tests whether increasing K (models selected) provides logarithmically
    diminishing returns, consistent with the model-independence hypothesis.
    """
    from joblib import Parallel, delayed

    k_values = [1, 3, 5, 10, 20, 30, 50]
    n_reps = N_REPS

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: K Sweep Independence Boundary")
    log("=" * 70)
    log(f"  k_values={k_values}, n_reps={n_reps}, seed={SEED}")

    feature_names = make_feature_names()

    def _fmt(v: float) -> str:
        """Format float values for logging."""
        return f"{v:.4f}" if not np.isnan(v) else "nan"

    # Collect per-k data (resumed or pending)
    methods_keys = ["dash", "rs", "sr"]  # DASH, Random Selection, Stochastic Retrain
    partial_data: dict = {
        k: {f"{m}_imp": [] for m in methods_keys} | {f"{m}_acc": [] for m in methods_keys} for k in k_values
    }
    pending_pairs = []

    for k_val in k_values:
        for rep in range(n_reps):
            ckpt_key = f"ksweep_flat_{k_val}_{rep}"
            if resume and _has(ckpt_key):
                data = _load(ckpt_key)
                for m in methods_keys:
                    if data.get(f"{m}_imp") is not None:
                        partial_data[k_val][f"{m}_imp"].append(data[f"{m}_imp"])
                        partial_data[k_val][f"{m}_acc"].append(data[f"{m}_acc"])
            else:
                pending_pairs.append((k_val, rep))

    n_total = len(k_values) * n_reps
    n_pending = len(pending_pairs)
    n_resumed = n_total - n_pending
    log(f"  {n_pending} (k, rep) pairs pending" + (f" ({n_resumed} resumed from checkpoints)" if n_resumed else ""))

    if pending_pairs:
        n_workers = compute_rep_worker_budget(n_work=n_pending)
        nthread = 1
        log(f"  Running {n_workers} workers on {get_available_cores()} cores (nthread={nthread})")

        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_single_ksweep_pair)(k_val, rep, feature_names, nthread=nthread) for k_val, rep in pending_pairs
        )

        for per_pair in results_list:
            k_val = per_pair["k_val"]
            rep = per_pair["rep"]
            for m in methods_keys:
                if per_pair.get(f"{m}_imp") is not None:
                    partial_data[k_val][f"{m}_imp"].append(per_pair[f"{m}_imp"])
                    partial_data[k_val][f"{m}_acc"].append(per_pair[f"{m}_acc"])
            _save(f"ksweep_flat_{k_val}_{rep}", **{k: v for k, v in per_pair.items() if k not in ("k_val", "rep")})
        _shutdown_loky_workers()

    # Aggregate per k_val
    method_labels = {"dash": "DASH", "rs": "RS", "sr": "SR"}
    results = {}
    for k_val in k_values:
        results[k_val] = {}
        for m in methods_keys:
            imp_runs = partial_data[k_val][f"{m}_imp"]
            acc_runs = partial_data[k_val][f"{m}_acc"]

            if len(imp_runs) >= 2:
                try:
                    stab, se, _, _ = stability_bootstrap_ci(imp_runs)
                except ValueError:
                    stab = importance_stability(imp_runs)
                    se = float("nan")
            else:
                stab, se = float("nan"), float("nan")

            acc_mean = float(np.mean(acc_runs)) if acc_runs else float("nan")
            acc_std = float(np.std(acc_runs, ddof=1)) if len(acc_runs) >= 2 else float("nan")

            results[k_val][method_labels[m]] = {
                "stability": stab,
                "stability_se": se,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "n_successful": len(imp_runs),
            }

            log(
                f"  K={k_val:>3}  {method_labels[m]:<5} stab={_fmt(stab)}±{_fmt(se)}  acc={_fmt(acc_mean)}"
                f"  ({len(imp_runs)}/{n_reps})"
            )

    _publish_results(results, f"{OUT}/tables/k_sweep_independence.json", "k_sweep_independence", N_REPS, t0)

    plot_k_sweep_independence(k_values, results)

    if cleanup:
        clear_checkpoints_by_prefix("ksweep_flat_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  K sweep completed in {elapsed / 60:.1f} min")
    return results


###############################################################################
# EXPERIMENT: Colsample Ablation
###############################################################################

# colsample_bytree ranges for the mechanism ablation
CS_RANGES = {
    "Low (0.1-0.5)": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "High (0.5-1.0)": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Full (0.1-1.0)": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
}


def experiment_colsample_ablation(resume=False, cleanup=False):
    """Colsample_bytree ablation: tests whether forced low colsample is the mechanism.

    Trains DASH and Random Selection under three colsample_bytree ranges,
    plus SR and SB controls. Uses a confound-free design: base hyperparameters
    (max_depth, learning_rate, subsample, etc.) are identical across ranges;
    only colsample_bytree varies (drawn from a separate RNG).

    Tests three conditions: linear ρ=0.0 (safety), linear ρ=0.9 (main),
    nonlinear ρ=0.9 (mechanism).
    """
    from dash_shap.core.population import generate_model_population, sample_configurations, DEFAULT_SEARCH_SPACE

    _ensure_dirs()
    t0 = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: Colsample Ablation")
    log("=" * 70)

    conditions = ["linear_0.0", "linear_0.9", "nonlinear_0.9"]
    cs_method_names = []
    for label in CS_RANGES:
        cs_method_names.append(f"DASH {label}")
        cs_method_names.append(f"RS {label}")
    cs_method_names.extend(["Stochastic Retrain", "Single Best"])

    # Per-condition, per-method storage
    results = {
        cond: {name: {"imp_runs": [], "eq_runs": [], "rmse_runs": [], "keff_runs": []} for name in cs_method_names}
        for cond in conditions
    }

    feature_names = make_feature_names()
    seq_budget = compute_thread_budget(n_outer=1)

    # Try to resume from batch checkpoint
    start_rep = 0
    if resume:
        for batch_end in range(N_REPS, 0, -10):
            ckpt_name = f"colsample_abl_batch_{batch_end}"
            if _has(ckpt_name):
                cached = _load(ckpt_name)
                results = cached["results"]
                start_rep = cached["completed_reps"]
                log(f"  Resuming from rep {start_rep}")
                break

    for rep in range(start_rep, N_REPS):
        rep_seed = SEED + rep
        log(f"  Rep {rep + 1}/{N_REPS}")

        # Generate base configs ONCE per rep (confound-free: non-colsample params fixed)
        base_configs = sample_configurations(DEFAULT_SEARCH_SPACE, M, seed=rep_seed)

        for cond in conditions:
            # Generate data for this condition
            if cond == "linear_0.0":
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
                    N=5000, rho=0.0, seed=rep_seed
                )
            elif cond == "linear_0.9":
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
                    N=5000, rho=0.9, seed=rep_seed
                )
            else:  # nonlinear_0.9
                Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, _, _ = generate_synthetic_nonlinear(
                    N=5000, rho=0.9, seed=rep_seed
                )

            for label, cs_values in CS_RANGES.items():
                # Replace colsample_bytree using SEPARATE RNG (confound fix)
                rng_cs = np.random.RandomState(rep_seed + 7777)
                configs = []
                for cfg in base_configs:
                    c = dict(cfg)
                    c["colsample_bytree"] = float(rng_cs.choice(cs_values))
                    configs.append(c)

                # DASH with these configs
                dash = DASHPipeline(
                    M=M,
                    K=K,
                    epsilon=EPSILON,
                    delta=DELTA,
                    selection_method="maxmin",
                    initial_configs=configs,
                    n_jobs=seq_budget.n_inner,
                    nthread=seq_budget.nthread,
                    seed=rep_seed,
                    verbose=False,
                )
                dash.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=feature_names)
                dash_imp = dash.global_importance_
                dash_preds = dash.get_consensus_ensemble_predictions(Xte)
                dash_keff = len(dash.selected_indices_) if dash.selected_indices_ is not None else None

                results[cond][f"DASH {label}"]["imp_runs"].append(dash_imp)
                results[cond][f"DASH {label}"]["eq_runs"].append(within_group_equity(dash_imp, grps))
                results[cond][f"DASH {label}"]["rmse_runs"].append(rmse_score(yte, dash_preds))
                if dash_keff is not None:
                    results[cond][f"DASH {label}"]["keff_runs"].append(dash_keff)

                # RS reuses DASH's population
                rs = RandomSelectionBaseline(
                    M=M,
                    K=K,
                    epsilon=EPSILON,
                    delta=DELTA,
                    n_jobs=seq_budget.n_inner,
                    nthread=seq_budget.nthread,
                    seed=rep_seed,
                )
                rs.fit_from_population(dash.models_, dash.val_scores_, Xexp, feature_names=feature_names)
                rs_imp = rs.global_importance_
                rs_preds = rs.get_consensus_ensemble_predictions(Xte)
                rs_keff = len(rs.selected_indices_) if rs.selected_indices_ is not None else None

                results[cond][f"RS {label}"]["imp_runs"].append(rs_imp)
                results[cond][f"RS {label}"]["eq_runs"].append(within_group_equity(rs_imp, grps))
                results[cond][f"RS {label}"]["rmse_runs"].append(rmse_score(yte, rs_preds))
                if rs_keff is not None:
                    results[cond][f"RS {label}"]["keff_runs"].append(rs_keff)

                del dash, rs

            # Controls: SR and SB (run once per condition, not per range)
            sr = StochasticRetrainBaseline(
                N=K,
                task="regression",
                n_jobs=seq_budget.n_inner,
                nthread=seq_budget.nthread,
                seed=rep_seed,
            )
            sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
            sr_imp = sr.global_importance_
            sr_preds = sr.get_consensus_ensemble_predictions(Xte)
            results[cond]["Stochastic Retrain"]["imp_runs"].append(sr_imp)
            results[cond]["Stochastic Retrain"]["eq_runs"].append(within_group_equity(sr_imp, grps))
            results[cond]["Stochastic Retrain"]["rmse_runs"].append(rmse_score(yte, sr_preds))
            del sr

            sb = SingleBestBaseline(
                n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=seq_budget.n_inner, nthread=seq_budget.nthread
            )
            sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
            sb_imp = sb.global_importance_
            sb_preds = sb.model_.predict(Xte)
            results[cond]["Single Best"]["imp_runs"].append(sb_imp)
            results[cond]["Single Best"]["eq_runs"].append(within_group_equity(sb_imp, grps))
            results[cond]["Single Best"]["rmse_runs"].append(rmse_score(yte, sb_preds))
            del sb

        if (rep + 1) % 10 == 0:
            _save(f"colsample_abl_batch_{rep + 1}", results=results, completed_reps=rep + 1)

    # ── Aggregate ──────────────────────────────────────────────────────────
    log("\n  Colsample ablation results:")
    for cond in conditions:
        log(f"\n  === {cond} ===")
        log(f"  {'Method':<25} {'Stability':>10} {'SE':>8} {'TopK5':>8} {'Equity':>8} {'RMSE':>8} {'Keff':>6}")
        log("  " + "-" * 80)
        for name in cs_method_names:
            d = results[cond][name]
            imp_runs = d["imp_runs"]
            if len(imp_runs) < 2:
                continue
            stab, se, ci_lo, ci_hi = stability_bootstrap_ci(imp_runs)
            topk5, _, _, _ = topk_stability_bootstrap_ci(imp_runs, k=5)
            eq_mean = float(np.mean(d["eq_runs"])) if d["eq_runs"] else float("nan")
            rmse_mean = float(np.mean(d["rmse_runs"])) if d["rmse_runs"] else float("nan")
            keff_mean = float(np.mean(d["keff_runs"])) if d["keff_runs"] else float("nan")

            d["stability"] = stab
            d["stability_se"] = se
            d["stability_ci_lo"] = ci_lo
            d["stability_ci_hi"] = ci_hi
            d["topk5_stability"] = topk5
            d["equity_mean"] = eq_mean
            d["rmse_mean"] = rmse_mean
            d["k_eff_mean"] = keff_mean

            log(
                f"  {name:<25} {stab:>10.4f} {se:>8.4f} {topk5:>8.4f} {eq_mean:>8.4f} {rmse_mean:>8.3f}"
                f" {keff_mean:>6.1f}"
            )

    # ── Statistical tests ──────────────────────────────────────────────────
    log("\n  Bootstrap stability tests (colsample ablation):")
    stab_tests = {}
    eq_tests = {}
    for cond in conditions:
        stab_tests[cond] = {}
        eq_tests[cond] = {}
        dash_low_imp = results[cond]["DASH Low (0.1-0.5)"].get("imp_runs", [])
        dash_low_eq = results[cond]["DASH Low (0.1-0.5)"].get("eq_runs", [])
        if len(dash_low_imp) < 2:
            continue
        for name in cs_method_names:
            if name == "DASH Low (0.1-0.5)":
                continue
            bl_imp = results[cond][name].get("imp_runs", [])
            bl_eq = results[cond][name].get("eq_runs", [])
            if len(bl_imp) < 2 or len(bl_imp) != len(dash_low_imp):
                continue
            try:
                diff, pval, ci_lo, ci_hi = bootstrap_stability_test(dash_low_imp, bl_imp)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                log(f"  {cond:15s} DASH-Low vs {name:<25} diff={diff:+.4f} p={pval:.4f} {sig}")
                stab_tests[cond][name] = {
                    "diff": float(diff),
                    "p": float(pval),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
            except Exception:
                pass
            try:
                _, pval_eq = compare_methods(dash_low_eq, bl_eq)
                d_eq = cohens_d(dash_low_eq, bl_eq)
                eq_tests[cond][name] = {"p": float(pval_eq), "cohens_d": float(d_eq)}
            except Exception:
                pass

    results["_stability_tests"] = stab_tests
    results["_equity_tests"] = eq_tests

    _publish_results(results, f"{OUT}/tables/colsample_ablation.json", "colsample_ablation", N_REPS, t0)

    if cleanup:
        clear_checkpoints_by_prefix("colsample_abl_batch_", CKPT_DIR)

    elapsed = time.time() - t0
    log(f"  Colsample ablation completed in {elapsed / 60:.1f} min")
    return results


###############################################################################
# EXPERIMENT: High-Dimensional Scaling
###############################################################################


def _run_single_rep_highdim(n_groups, group_size, rho, rep, methods, *, n_noise=0, nthread=1):
    """Run core methods for one (n_groups, rep) pair at given rho.

    Generates a linear DGP with n_groups correlated groups of size group_size,
    plus n_noise uncorrelated features with beta=0. Reuses the standard
    generate_synthetic_linear DGP when n_noise=0.

    Returns (rep, per_method_dict, grps).
    """
    rep_seed = SEED + rep
    P_signal = n_groups * group_size
    P_total = P_signal + n_noise

    if n_noise == 0:
        # Standard DGP — same as linear sweep
        Xtr, ytr, Xv, yv, Xexp, yexp, Xte, yte, grps, true_imp, _ = generate_synthetic_linear(
            N=5000, P=P_signal, group_size=group_size, rho=rho, seed=rep_seed
        )
    else:
        # Generate signal features normally
        Xtr_s, ytr, Xv_s, yv, Xexp_s, yexp, Xte_s, yte, grps_s, true_imp_s, _ = generate_synthetic_linear(
            N=5000, P=P_signal, group_size=group_size, rho=rho, seed=rep_seed
        )
        # Generate uncorrelated noise features
        rng = np.random.RandomState(rep_seed + 99999)
        N_tr, N_v, N_exp, N_te = len(Xtr_s), len(Xv_s), len(Xexp_s), len(Xte_s)
        Xtr = np.hstack([Xtr_s, rng.randn(N_tr, n_noise)])
        Xv = np.hstack([Xv_s, rng.randn(N_v, n_noise)])
        Xexp = np.hstack([Xexp_s, rng.randn(N_exp, n_noise)])
        Xte = np.hstack([Xte_s, rng.randn(N_te, n_noise)])
        grps = np.concatenate([grps_s, np.arange(n_groups, n_groups + n_noise)])
        true_imp = np.concatenate([true_imp_s, np.zeros(n_noise)])

    nthread_xgb = nthread
    per_method: dict = {}

    def rmse_score(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # ── DASH (population-sharing with RS) ──
    t0 = time.time()
    dash_pipeline = DASHPipeline(
        M=M,
        K=K,
        epsilon=EPSILON,
        delta=DELTA,
        selection_method="maxmin",
        task="regression",
        n_jobs=1,
        nthread=nthread_xgb,
        seed=rep_seed,
        verbose=False,
    )
    dash_pipeline.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, feature_names=[f"f{i}" for i in range(P_total)])
    per_method["DASH (MaxMin)"] = _rep_metrics(
        dash_pipeline.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, dash_pipeline.get_consensus_ensemble_predictions(Xte)),
        dash_pipeline,
        time.time() - t0,
    )

    # ── Single Best ──
    t0 = time.time()
    sb = SingleBestBaseline(n_trials=N_TRIALS_SB, seed=rep_seed, n_jobs=1, nthread=nthread_xgb)
    sb.fit(Xtr, ytr, Xv, yv, X_ref=Xexp, seed=rep_seed)
    per_method["Single Best"] = _rep_metrics(
        sb.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, sb.model_.predict(Xte)),
        sb,
        time.time() - t0,
    )

    # ── Stochastic Retrain ──
    t0 = time.time()
    sr = StochasticRetrainBaseline(N=K, seed=rep_seed, n_jobs=1, nthread=nthread_xgb)
    sr.fit(Xtr, ytr, Xv, yv, X_ref=Xexp)
    per_method["Stochastic Retrain"] = _rep_metrics(
        sr.global_importance_,
        true_imp,
        grps,
        rmse_score(yte, sr.models_[0].predict(Xte)),
        sr,
        time.time() - t0,
    )

    # Clean up models to free memory
    del dash_pipeline, sb, sr

    return rep, per_method, grps


def experiment_high_dimensional_scaling(resume=False, cleanup=False):
    """High-dimensional scaling: does DASH's advantage persist as P grows?

    Three sub-experiments for TMLR paper appendix:
    A) Group scaling: L ∈ {10, 20, 40, 80} → P ∈ {50, 100, 200, 400}
    B) Noise dilution: P=200 with L=10 signal groups + 150 noise features
    C) ρ replication at P=200 (L=40): ρ ∈ {0.0, 0.5, 0.7, 0.9, 0.95}

    All use PAPER_CONFIG (M=200, K=30, N_REPS=50, ε=0.08).
    Methods: DASH (MaxMin), Single Best, Stochastic Retrain.
    """
    from joblib import Parallel, delayed

    _ensure_dirs()
    t0_global = time.time()
    log("\n" + "=" * 70)
    log("EXPERIMENT: High-Dimensional Scaling")
    log("=" * 70)

    group_size = 5
    core_methods = ["DASH (MaxMin)", "Single Best", "Stochastic Retrain"]
    nthread = 1  # sequential within rep, parallel across reps

    results = {}

    # ── Sub-experiment A: Group scaling at ρ=0.9 ──
    log("\n--- Sub-experiment A: Group scaling (ρ=0.9) ---")
    group_counts = [10, 20, 40, 80]
    rho_fixed = 0.9

    for n_groups in group_counts:
        P = n_groups * group_size
        log(f"\n  L={n_groups}, P={P}:")
        t0 = time.time()

        rep_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(_run_single_rep_highdim)(n_groups, group_size, rho_fixed, rep, core_methods, nthread=nthread)
            for rep in range(N_REPS)
        )

        method_data = _init_method_data(core_methods)
        for _, per_method, _ in rep_results:
            _merge_rep(method_data, per_method)

        level_results = _aggregate_method_data(method_data, core_methods)
        results[f"A_L{n_groups}_P{P}"] = level_results
        log(f"  L={n_groups} completed in {(time.time() - t0) / 60:.1f} min")

    # ── Sub-experiment B: Noise dilution at P=200 ──
    log("\n--- Sub-experiment B: Noise dilution (P=200, ρ=0.9) ---")
    # Condition (i): L=40, all signal → reuse A_L40_P200
    # Condition (ii): L=10 signal + 150 noise
    log("  Condition (ii): L=10 signal + 150 noise features")
    t0 = time.time()

    rep_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_run_single_rep_highdim)(10, group_size, rho_fixed, rep, core_methods, n_noise=150, nthread=nthread)
        for rep in range(N_REPS)
    )

    method_data = _init_method_data(core_methods)
    for _, per_method, _ in rep_results:
        _merge_rep(method_data, per_method)

    results["B_L10_noise150_P200"] = _aggregate_method_data(method_data, core_methods)
    log(f"  Noise dilution completed in {(time.time() - t0) / 60:.1f} min")

    # ── Sub-experiment C: ρ sweep at P=200 (L=40) ──
    log("\n--- Sub-experiment C: ρ sweep at P=200 (L=40 groups) ---")
    rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
    n_groups_c = 40

    for rho in rho_levels:
        log(f"\n  ρ={rho}:")
        t0 = time.time()

        rep_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(_run_single_rep_highdim)(n_groups_c, group_size, rho, rep, core_methods, nthread=nthread)
            for rep in range(N_REPS)
        )

        method_data = _init_method_data(core_methods)
        for _, per_method, _ in rep_results:
            _merge_rep(method_data, per_method)

        results[f"C_P200_rho{rho}"] = _aggregate_method_data(method_data, core_methods)
        log(f"  ρ={rho} completed in {(time.time() - t0) / 60:.1f} min")

    # ── Summary ──
    log("\n" + "=" * 70)
    log("High-Dimensional Scaling — Summary")
    log("=" * 70)

    log("\nSub-exp A: Group scaling (ρ=0.9)")
    log(f"  {'P':>5}  {'Method':<22} {'Stability':>10} {'TopK5':>8} {'Equity':>8}")
    for n_groups in group_counts:
        P = n_groups * group_size
        key = f"A_L{n_groups}_P{P}"
        for name in core_methods:
            r = results[key][name]
            log(f"  {P:>5}  {name:<22} {r['stability']:>10.4f} {r['topk5_stability']:>8.4f} {r['equity_mean']:>8.4f}")

    log("\nSub-exp B: Noise dilution (P=200, ρ=0.9)")
    log("  Condition (i): 40 groups, all signal [= A_L40_P200]")
    log("  Condition (ii): 10 signal groups + 150 noise features")
    for name in core_methods:
        r_signal = results["A_L40_P200"][name]
        r_noise = results["B_L10_noise150_P200"][name]
        log(f"  {name:<22} signal_only={r_signal['stability']:.4f}  with_noise={r_noise['stability']:.4f}")

    log("\nSub-exp C: ρ sweep at P=200 (L=40)")
    log(f"  {'ρ':>5}  {'DASH':>10} {'SB':>10} {'SR':>10}")
    for rho in rho_levels:
        r = results[f"C_P200_rho{rho}"]
        log(
            f"  {rho:>5.2f}  {r['DASH (MaxMin)']['stability']:>10.4f} "
            f"{r['Single Best']['stability']:>10.4f} {r['Stochastic Retrain']['stability']:>10.4f}"
        )

    _publish_results(
        results,
        f"{OUT}/tables/high_dimensional_scaling.json",
        "high_dimensional_scaling",
        N_REPS,
        t0_global,
    )

    elapsed = time.time() - t0_global
    log(f"\n  High-dimensional scaling completed in {elapsed / 60:.1f} min")
    return results


EXPERIMENTS = {
    "linear_sweep": experiment_linear_sweep,
    "overlapping": experiment_overlapping,
    "nonlinear_sweep": experiment_nonlinear_sweep,
    "table2_baselines": experiment_table2_baselines,
    "real_california": experiment_real_california,
    "real_breast_cancer": experiment_real_breast_cancer,
    "real_superconductor": experiment_real_superconductor,
    "epsilon_sensitivity": experiment_epsilon_sensitivity,
    "ablation": experiment_ablation,
    "variance_decomposition": experiment_variance_decomposition,
    "variance_decomposition_crossed": experiment_variance_decomposition_crossed,
    "first_mover_visualization": experiment_first_mover_visualization,
    "first_mover_bias": experiment_first_mover_bias,
    "background_sensitivity": experiment_background_sensitivity,
    "asymmetric_dgp": experiment_asymmetric_dgp,
    "k_sweep_independence": experiment_k_sweep_independence,
    "colsample_ablation": experiment_colsample_ablation,
    "success_criteria": experiment_success_criteria,
    "extensions_sanity_check": experiment_extensions_sanity_check,
    "high_dimensional_scaling": experiment_high_dimensional_scaling,
}

# Default run order (all experiments)
DEFAULT_ORDER = [
    "linear_sweep",
    "first_mover_visualization",
    "overlapping",
    "nonlinear_sweep",
    "table2_baselines",
    "real_california",
    "real_breast_cancer",
    "real_superconductor",
    "epsilon_sensitivity",
    "ablation",
    "variance_decomposition",
    "variance_decomposition_crossed",
    "asymmetric_dgp",
    "first_mover_bias",
    "background_sensitivity",
    "k_sweep_independence",
    "colsample_ablation",
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
  python run_experiments.py --smoke --experiments linear_sweep  # Quick validation of full pipeline
        """,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from per-level checkpoints if available",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep per-level checkpoints after experiment completes",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable rep parallelism for linear_sweep (use for validation or single-core machines)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 rep, reduced config. Validates full pipeline including serialization.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, fn in EXPERIMENTS.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:<24} {doc}")
        sys.exit(0)

    if args.smoke:
        # Validate the full serialization pipeline without training any models.
        # Builds a synthetic result dict with float keys (like linear_sweep
        # produces) and pushes it through _publish_results. This catches
        # serialization bugs — like the float-key validation issue that
        # crashed a 250-rep SageMaker run — in seconds, not hours.
        import tempfile

        log("SMOKE TEST: validating serialization pipeline...")
        smoke_data: dict = {}
        for rho in [0.0, 0.5, 0.9]:
            smoke_data[rho] = {}
            for method in ["DASH (MaxMin)", "Single Best"]:
                smoke_data[rho][method] = {
                    "stability": 0.95,
                    "stability_lo": 0.93,
                    "stability_hi": 0.97,
                    "stability_se": 0.01,
                    "accuracy": 0.88,
                    "equity": 0.15,
                    "n_successful": 1,
                    "n_reps": 1,
                    "acc_runs": [0.88],
                }
        smoke_path = os.path.join(tempfile.mkdtemp(), "smoke_test.json")
        _publish_results(smoke_data, smoke_path, "smoke_test", 1, time.time())
        os.unlink(smoke_path)
        log("SMOKE TEST PASSED: serialization pipeline OK")
        sys.exit(0)

    _ensure_dirs()
    t_start = time.time()

    log("DASH Experimental Validation")
    log(f"Config: M={M}, K={K}, ε={EPSILON}, δ={DELTA}, N_REPS={N_REPS}")
    log(f"  Real-data: ε={REAL_EPSILON} (mode={REAL_EPSILON_MODE})")
    write_environment_snapshot(OUT)
    log(f"  Environment snapshot: {OUT}/environment.json")
    log("")

    to_run = args.experiments or DEFAULT_ORDER
    sweep_results = None

    for name in to_run:
        kwargs: dict = {"resume": args.resume, "cleanup": not args.no_cleanup}
        if name == "linear_sweep":
            kwargs["sequential"] = args.sequential
        result = EXPERIMENTS[name](**kwargs)
        if name == "linear_sweep":
            sweep_results = result

    # Run success criteria if linear_sweep was included
    if sweep_results is not None:
        check_success_criteria(sweep_results)

    total_time = time.time() - t_start
    log(f"\nTotal runtime: {total_time / 60:.1f} minutes")
    log(f"Results saved to {OUT}/")
