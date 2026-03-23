"""Numerical validation of the impossibility theorem for DASH-SHAP.

Demonstrates that sequential gradient boosting with correlated features faces
an inherent Stability-vs-Equity tradeoff (first-mover bias), and that DASH
circumvents this by averaging SHAP across M independent models.

Three experiments:
  1. First-mover concentration grows with boosting rounds T (Lemma 1)
  2. Stability vs Equity tradeoff for single models vs DASH (Main theorem)
  3. DASH convergence: equity improves with ensemble size M (Corollary)

Results (JSON + figures) are saved to results/impossibility/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_correlated_data(
    n_samples: int = 2000,
    n_correlated: int = 5,
    rho: float = 0.9,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Generate data with a correlated feature group of equal true importance.

    y = sum(X_correlated) + noise, so each correlated feature has equal importance.
    Features are pairwise correlated at level rho via Cholesky decomposition.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build correlation matrix: constant pairwise correlation rho
    corr = np.full((n_correlated, n_correlated), rho)
    np.fill_diagonal(corr, 1.0)

    # Cholesky decomposition for correlated draws
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n_samples, n_correlated))
    X = Z @ L.T

    # Target: equal weight on all correlated features
    y = X.sum(axis=1) + noise_std * rng.standard_normal(n_samples)

    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def concentration(shap_values: NDArray) -> float:
    """max(|phi|) / sum(|phi|) within the feature group."""
    abs_phi = np.abs(shap_values).mean(axis=0)
    total = abs_phi.sum()
    if total == 0:
        return 1.0 / len(abs_phi)
    return float(abs_phi.max() / total)


def gini_coefficient(values: NDArray) -> float:
    """Gini coefficient of a 1-D array (0 = perfect equality)."""
    values = np.abs(np.asarray(values, dtype=float))
    if values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_v).sum() / (n * sorted_v.sum())) - (n + 1) / n)


def equity(shap_values: NDArray) -> float:
    """1 - Gini of mean |phi| across features. 1 = perfect equity."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return 1.0 - gini_coefficient(mean_abs)


def stability_spearman(rankings_list: list[NDArray]) -> float:
    """Mean pairwise Spearman rank correlation across repetitions."""
    from scipy.stats import spearmanr

    n = len(rankings_list)
    if n < 2:
        return 1.0
    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            corr, _ = spearmanr(rankings_list[i], rankings_list[j])
            correlations.append(corr)
    return float(np.mean(correlations))


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _train_and_shap(
    X_train: NDArray,
    y_train: NDArray,
    X_explain: NDArray,
    n_estimators: int = 200,
    seed: int = 0,
    random_colsample: bool = False,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Train one XGBoost model and return interventional SHAP values."""
    import shap
    import xgboost as xgb

    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": seed,
        "verbosity": 0,
    }
    if random_colsample and rng is not None:
        # DASH-style: low colsample_bytree to break sequential dependency
        params["colsample_bytree"] = float(rng.uniform(0.1, 0.5))
    else:
        params["colsample_bytree"] = 1.0

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model, X_explain, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain)
    return np.asarray(shap_values)


# ---------------------------------------------------------------------------
# Experiment 1: First-mover concentration grows with T
# ---------------------------------------------------------------------------


def experiment_concentration(
    T_values: list[int] | None = None,
    n_reps: int = 20,
    seed: int = 42,
    n_samples: int = 2000,
    n_correlated: int = 5,
    rho: float = 0.9,
) -> dict[str, Any]:
    """Show that first-mover concentration increases with boosting rounds T."""
    if T_values is None:
        T_values = [50, 100, 200, 500, 1000, 2000]

    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = {str(t): [] for t in T_values}

    for rep in range(n_reps):
        rep_seed = int(rng.integers(0, 2**31))
        rep_rng = np.random.default_rng(rep_seed)

        X, y = generate_correlated_data(
            n_samples=n_samples,
            n_correlated=n_correlated,
            rho=rho,
            rng=rep_rng,
        )
        n_train = int(0.7 * n_samples)
        X_train, X_explain = X[:n_train], X[n_train:]
        y_train = y[:n_train]

        for t in T_values:
            shap_vals = _train_and_shap(
                X_train,
                y_train,
                X_explain,
                n_estimators=t,
                seed=rep_seed,
            )
            conc = concentration(shap_vals)
            results[str(t)].append(conc)

    summary = {
        "T_values": T_values,
        "mean_concentration": {str(t): float(np.mean(results[str(t)])) for t in T_values},
        "std_concentration": {str(t): float(np.std(results[str(t)])) for t in T_values},
        "n_reps": n_reps,
        "rho": rho,
        "n_correlated": n_correlated,
        "ideal_concentration": 1.0 / n_correlated,
    }
    return summary


# ---------------------------------------------------------------------------
# Experiment 2: Stability vs Equity tradeoff
# ---------------------------------------------------------------------------


def experiment_tradeoff(
    n_reps: int = 50,
    M: int = 30,
    seed: int = 42,
    n_samples: int = 2000,
    n_correlated: int = 5,
    rho: float = 0.9,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """Compare single model vs DASH on stability and equity."""
    rng = np.random.default_rng(seed)

    single_importances: list[NDArray] = []
    single_equities: list[float] = []
    dash_importances: list[NDArray] = []
    dash_equities: list[float] = []

    for rep in range(n_reps):
        rep_seed = int(rng.integers(0, 2**31))
        rep_rng = np.random.default_rng(rep_seed)

        X, y = generate_correlated_data(
            n_samples=n_samples,
            n_correlated=n_correlated,
            rho=rho,
            rng=rep_rng,
        )
        n_train = int(0.7 * n_samples)
        X_train, X_explain = X[:n_train], X[n_train:]
        y_train = y[:n_train]

        # --- Single model ---
        single_shap = _train_and_shap(
            X_train,
            y_train,
            X_explain,
            n_estimators=n_estimators,
            seed=rep_seed,
        )
        single_imp = np.abs(single_shap).mean(axis=0)
        single_importances.append(single_imp)
        single_equities.append(equity(single_shap))

        # --- DASH: M independent models with randomised colsample ---
        model_rng = np.random.default_rng(rep_seed + 1)
        shap_accum = np.zeros_like(single_shap)
        for m in range(M):
            m_seed = int(model_rng.integers(0, 2**31))
            sv = _train_and_shap(
                X_train,
                y_train,
                X_explain,
                n_estimators=n_estimators,
                seed=m_seed,
                random_colsample=True,
                rng=model_rng,
            )
            shap_accum += sv
        dash_shap = shap_accum / M
        dash_imp = np.abs(dash_shap).mean(axis=0)
        dash_importances.append(dash_imp)
        dash_equities.append(equity(dash_shap))

        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Tradeoff experiment: rep {rep + 1}/{n_reps}")

    single_stability = stability_spearman(single_importances)
    dash_stability = stability_spearman(dash_importances)

    return {
        "single": {
            "stability": single_stability,
            "mean_equity": float(np.mean(single_equities)),
            "std_equity": float(np.std(single_equities)),
        },
        "dash": {
            "stability": dash_stability,
            "mean_equity": float(np.mean(dash_equities)),
            "std_equity": float(np.std(dash_equities)),
            "M": M,
        },
        "n_reps": n_reps,
        "rho": rho,
        "n_correlated": n_correlated,
        "n_estimators": n_estimators,
    }


# ---------------------------------------------------------------------------
# Experiment 3: DASH convergence with M
# ---------------------------------------------------------------------------


def experiment_convergence(
    M_values: list[int] | None = None,
    n_reps: int = 50,
    seed: int = 42,
    n_samples: int = 2000,
    n_correlated: int = 5,
    rho: float = 0.9,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """Show equity improves with M while stability remains high."""
    if M_values is None:
        M_values = [1, 5, 10, 20, 30, 50]

    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, list[float]]] = {str(m): {"equities": [], "importances_flat": []} for m in M_values}

    # We need per-rep data; store all per-rep importances for stability calc
    per_m_importances: dict[str, list[NDArray]] = {str(m): [] for m in M_values}

    for rep in range(n_reps):
        rep_seed = int(rng.integers(0, 2**31))
        rep_rng = np.random.default_rng(rep_seed)

        X, y = generate_correlated_data(
            n_samples=n_samples,
            n_correlated=n_correlated,
            rho=rho,
            rng=rep_rng,
        )
        n_train = int(0.7 * n_samples)
        X_train, X_explain = X[:n_train], X[n_train:]
        y_train = y[:n_train]

        # Train max(M_values) models, then take subsets
        max_M = max(M_values)
        model_rng = np.random.default_rng(rep_seed + 1)
        all_shap: list[NDArray] = []
        for m in range(max_M):
            m_seed = int(model_rng.integers(0, 2**31))
            sv = _train_and_shap(
                X_train,
                y_train,
                X_explain,
                n_estimators=n_estimators,
                seed=m_seed,
                random_colsample=True,
                rng=model_rng,
            )
            all_shap.append(sv)

        for m_val in M_values:
            subset = np.stack(all_shap[:m_val])
            avg_shap = subset.mean(axis=0)
            eq = equity(avg_shap)
            imp = np.abs(avg_shap).mean(axis=0)
            results[str(m_val)]["equities"].append(eq)
            per_m_importances[str(m_val)].append(imp)

        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Convergence experiment: rep {rep + 1}/{n_reps}")

    summary: dict[str, Any] = {"M_values": M_values, "metrics": {}}
    for m_val in M_values:
        stab = stability_spearman(per_m_importances[str(m_val)])
        eqs = results[str(m_val)]["equities"]
        summary["metrics"][str(m_val)] = {
            "stability": stab,
            "mean_equity": float(np.mean(eqs)),
            "std_equity": float(np.std(eqs)),
        }
    summary["n_reps"] = n_reps
    summary["rho"] = rho
    summary["n_correlated"] = n_correlated

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_concentration(data: dict[str, Any], output_dir: Path) -> None:
    """Plot first-mover concentration vs boosting rounds T."""
    T_values = data["T_values"]
    means = [data["mean_concentration"][str(t)] for t in T_values]
    stds = [data["std_concentration"][str(t)] for t in T_values]
    ideal = data["ideal_concentration"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(T_values, means, yerr=stds, marker="o", capsize=4, linewidth=2, label="Single model")
    ax.axhline(ideal, color="green", linestyle="--", linewidth=1.5, label=f"Ideal (1/{data['n_correlated']})")
    ax.set_xlabel("Boosting rounds (T)", fontsize=13)
    ax.set_ylabel("First-mover concentration", fontsize=13)
    ax.set_title(f"Lemma 1: Concentration grows with T (rho={data['rho']})", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "concentration_vs_T.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(data: dict[str, Any], output_dir: Path) -> None:
    """Plot stability vs equity for single model and DASH."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        data["single"]["stability"],
        data["single"]["mean_equity"],
        s=200,
        marker="X",
        color="red",
        zorder=5,
        label="Single model",
    )
    ax.scatter(
        data["dash"]["stability"],
        data["dash"]["mean_equity"],
        s=200,
        marker="*",
        color="blue",
        zorder=5,
        label=f"DASH (M={data['dash']['M']})",
    )

    # Shade the "impossible" upper-right region for single models
    ax.set_xlabel("Stability (mean pairwise Spearman)", fontsize=13)
    ax.set_ylabel("Equity (1 - Gini)", fontsize=13)
    ax.set_title(f"Stability-Equity tradeoff (rho={data['rho']})", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stability_vs_equity.png", dpi=150)
    plt.close(fig)


def plot_convergence(data: dict[str, Any], output_dir: Path) -> None:
    """Plot equity and stability vs ensemble size M."""
    M_values = data["M_values"]
    equities = [data["metrics"][str(m)]["mean_equity"] for m in M_values]
    eq_stds = [data["metrics"][str(m)]["std_equity"] for m in M_values]
    stabilities = [data["metrics"][str(m)]["stability"] for m in M_values]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.errorbar(M_values, equities, yerr=eq_stds, marker="o", capsize=4, color=color1, linewidth=2, label="Equity")
    ax1.set_xlabel("Ensemble size M", fontsize=13)
    ax1.set_ylabel("Equity (1 - Gini)", fontsize=13, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.plot(M_values, stabilities, marker="s", color=color2, linewidth=2, linestyle="--", label="Stability")
    ax2.set_ylabel("Stability (Spearman)", fontsize=13, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right")

    ax1.set_title(f"Corollary: DASH convergence (rho={data['rho']})", fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dash_convergence.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(conc: dict[str, Any], tradeoff: dict[str, Any], conv: dict[str, Any]) -> None:
    """Print a concise summary table to stdout."""
    print("\n" + "=" * 70)
    print("IMPOSSIBILITY THEOREM — NUMERICAL VALIDATION")
    print("=" * 70)

    print("\n--- Experiment 1: First-mover concentration vs T ---")
    print(
        f"  rho={conc['rho']}, {conc['n_correlated']} correlated features, "
        f"ideal concentration = {conc['ideal_concentration']:.3f}"
    )
    print(f"  {'T':>6s}  {'Concentration':>15s}")
    print(f"  {'------':>6s}  {'---------------':>15s}")
    for t in conc["T_values"]:
        m = conc["mean_concentration"][str(t)]
        s = conc["std_concentration"][str(t)]
        print(f"  {t:>6d}  {m:>10.4f} +/- {s:.4f}")

    print("\n--- Experiment 2: Stability vs Equity tradeoff ---")
    print(f"  {'Method':<20s}  {'Stability':>10s}  {'Equity':>10s}")
    print(f"  {'------':<20s}  {'----------':>10s}  {'----------':>10s}")
    for label, key in [("Single model", "single"), ("DASH", "dash")]:
        stab = tradeoff[key]["stability"]
        eq = tradeoff[key]["mean_equity"]
        print(f"  {label:<20s}  {stab:>10.4f}  {eq:>10.4f}")
    print(
        f"  DASH advantage:  stability +{tradeoff['dash']['stability'] - tradeoff['single']['stability']:.4f}, "
        f"equity +{tradeoff['dash']['mean_equity'] - tradeoff['single']['mean_equity']:.4f}"
    )

    print("\n--- Experiment 3: DASH convergence with M ---")
    print(f"  {'M':>5s}  {'Stability':>10s}  {'Equity':>10s}")
    print(f"  {'-----':>5s}  {'----------':>10s}  {'----------':>10s}")
    for m in conv["M_values"]:
        met = conv["metrics"][str(m)]
        print(f"  {m:>5d}  {met['stability']:>10.4f}  {met['mean_equity']:>10.4f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_simulation(
    output_dir: str | Path = "results/impossibility",
    seed: int = 42,
    n_reps: int = 50,
    n_samples: int = 2000,
    T_values: list[int] | None = None,
    M_values: list[int] | None = None,
    M_tradeoff: int = 30,
    n_estimators: int = 200,
    n_correlated: int = 5,
    rho: float = 0.9,
    conc_reps: int | None = None,
) -> dict[str, Any]:
    """Run all three impossibility experiments.

    Parameters
    ----------
    output_dir : path for JSON results and figures
    seed : master random seed
    n_reps : repetitions for experiments 2 and 3
    n_samples : samples per synthetic dataset
    T_values : boosting rounds to test in experiment 1
    M_values : ensemble sizes for experiment 3
    M_tradeoff : DASH ensemble size for experiment 2
    n_estimators : default boosting rounds for experiments 2/3
    n_correlated : number of correlated features
    rho : pairwise correlation
    conc_reps : repetitions for experiment 1 (defaults to n_reps)

    Returns
    -------
    dict with keys "concentration", "tradeoff", "convergence"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if conc_reps is None:
        conc_reps = n_reps

    print("Experiment 1/3: First-mover concentration vs T ...")
    conc = experiment_concentration(
        T_values=T_values,
        n_reps=conc_reps,
        seed=seed,
        n_samples=n_samples,
        n_correlated=n_correlated,
        rho=rho,
    )
    plot_concentration(conc, output_dir)
    print("  -> Done.")

    print("Experiment 2/3: Stability vs Equity tradeoff ...")
    tradeoff = experiment_tradeoff(
        n_reps=n_reps,
        M=M_tradeoff,
        seed=seed,
        n_samples=n_samples,
        n_correlated=n_correlated,
        rho=rho,
        n_estimators=n_estimators,
    )
    plot_tradeoff(tradeoff, output_dir)
    print("  -> Done.")

    print("Experiment 3/3: DASH convergence with M ...")
    conv = experiment_convergence(
        M_values=M_values,
        n_reps=n_reps,
        seed=seed,
        n_samples=n_samples,
        n_correlated=n_correlated,
        rho=rho,
        n_estimators=n_estimators,
    )
    plot_convergence(conv, output_dir)
    print("  -> Done.")

    all_results = {
        "concentration": conc,
        "tradeoff": tradeoff,
        "convergence": conv,
        "config": {
            "seed": seed,
            "n_reps": n_reps,
            "n_samples": n_samples,
            "n_correlated": n_correlated,
            "rho": rho,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(conc, tradeoff, conv)
    print(f"\nResults saved to {output_dir}/")
    print(f"  results.json")
    print(f"  concentration_vs_T.png")
    print(f"  stability_vs_equity.png")
    print(f"  dash_convergence.png")

    return all_results


if __name__ == "__main__":
    # Default: full simulation with paper-grade parameters
    run_simulation(
        output_dir="results/impossibility",
        seed=42,
        n_reps=50,
    )
