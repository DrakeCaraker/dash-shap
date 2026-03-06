"""Add all new sections to demo_benchmark_3.ipynb."""
import json
import uuid

def make_cell(cell_type, source):
    """Create a notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    # Convert to newline-terminated lines (standard ipynb format)
    lines = source.split("\n") if isinstance(source, str) else source
    cell["source"] = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            cell["source"].append(line + "\n")
        else:
            cell["source"].append(line)
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def main():
    with open("notebooks/demo_benchmark_3.ipynb") as f:
        nb = json.load(f)

    # Keep original 25 cells, append new sections
    assert len(nb["cells"]) == 25, f"Expected 25 cells, got {len(nb['cells'])}"

    new_cells = []

    # ================================================================
    # SECTION 7: Priority 0A — Epsilon Sensitivity Analysis
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 7. Priority 0A: Epsilon Sensitivity and Effective Ensemble Size

**Critical investigation before submission.** The M=500 run above used ε=0.03, passing only 11/500 models. MaxMin then selected K_eff=3 (hitting the δ=0.05 diversity threshold). A reviewer who notices "11/500 passed, K_eff=3" will question parameter calibration — is the M=200→M=500 improvement coming from deeper diversity selection, or are we just wasting compute?

We test ε ∈ {0.03, 0.05, 0.08, 0.10} at ρ=0.9 with M=500, N_REPS=10. For each epsilon we report:
- **Models passing filter** — how many survive the performance gate
- **K_eff** — how many MaxMin actually selects before hitting the diversity floor
- **Stability, Accuracy, Equity** — the three core metrics

**Key optimization**: We train the M=500 population once per repetition and re-filter at each ε value, isolating the effect of the filter threshold from stochastic population variation."""))

    new_cells.append(make_cell("code", """# Epsilon sensitivity sweep
from dash.core.population import generate_model_population
from sklearn.metrics import root_mean_squared_error

EPS_VALUES = [0.03, 0.05, 0.08, 0.10]
EPS_N_REPS = 10

eps_results = {eps: {
    'n_passing': [], 'k_eff': [],
    'acc_runs': [], 'eq_runs': [], 'imp_runs': []
} for eps in EPS_VALUES}

for rep in range(EPS_N_REPS):
    rep_seed = SEED + rep
    print(f'\\nRepetition {rep + 1}/{EPS_N_REPS} (seed={rep_seed})')

    Xtr, ytr, Xv, yv, _, _, grps, true_imp, _ = \\
        generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)

    # Train population ONCE per rep
    models, val_scores, configs = generate_model_population(
        Xtr, ytr, Xv, yv, M=M, task='regression',
        n_jobs=-1, seed=rep_seed, verbose=False
    )

    for eps in EPS_VALUES:
        # Stage 2: Filter at this epsilon
        filtered = performance_filter(val_scores, epsilon=eps,
                                      higher_is_better=True, verbose=False)
        eps_results[eps]['n_passing'].append(len(filtered))

        if len(filtered) < 2:
            print(f'  eps={eps}: only {len(filtered)} passed, skipping')
            continue

        # Stage 3: MaxMin diversity selection
        imp_vecs = get_preliminary_importance(models, filtered, Xv, method='gain')
        filt_scores = {i: val_scores[i] for i in filtered}
        selected = greedy_maxmin_selection(imp_vecs, filt_scores,
                                          K=K, delta=DELTA, verbose=False)
        eps_results[eps]['k_eff'].append(len(selected))

        # Stage 4-5: Consensus SHAP
        cons, all_shap = compute_consensus(models, selected, Xv, verbose=False)
        _, _, _, imp = compute_diagnostics(all_shap)

        r, _ = importance_accuracy(imp, true_imp)
        eps_results[eps]['acc_runs'].append(r)
        eps_results[eps]['eq_runs'].append(within_group_equity(imp, grps))
        eps_results[eps]['imp_runs'].append(imp)

print('\\nEpsilon sweep complete.')"""))

    new_cells.append(make_cell("code", """# Epsilon sensitivity results table and plots
print(f'{\"ε\":>6} {\"Models Passing\":>16} {\"K_eff\":>12} {\"Stability\":>10} {\"Accuracy\":>10} {\"Equity\":>10}')
print('=' * 70)
for eps in EPS_VALUES:
    n_pass = np.mean(eps_results[eps]['n_passing'])
    k_eff_mean = np.mean(eps_results[eps]['k_eff'])
    k_eff_std = np.std(eps_results[eps]['k_eff'])
    stab = importance_stability(eps_results[eps]['imp_runs']) if len(eps_results[eps]['imp_runs']) >= 2 else float('nan')
    acc = np.mean(eps_results[eps]['acc_runs']) if eps_results[eps]['acc_runs'] else float('nan')
    eq = np.mean(eps_results[eps]['eq_runs']) if eps_results[eps]['eq_runs'] else float('nan')
    print(f'{eps:>6.2f} {n_pass:>16.1f} {k_eff_mean:>8.1f}±{k_eff_std:<4.1f} {stab:>10.4f} {acc:>10.4f} {eq:>10.4f}')

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: n_passing and K_eff vs epsilon
ax1 = axes[0]
ax1_twin = ax1.twinx()
n_passing_means = [np.mean(eps_results[eps]['n_passing']) for eps in EPS_VALUES]
k_eff_means = [np.mean(eps_results[eps]['k_eff']) for eps in EPS_VALUES]
k_eff_stds = [np.std(eps_results[eps]['k_eff']) for eps in EPS_VALUES]

ax1.bar(np.arange(len(EPS_VALUES)) - 0.15, n_passing_means, 0.3,
        color='#3498db', alpha=0.7, label='Models Passing')
ax1_twin.bar(np.arange(len(EPS_VALUES)) + 0.15, k_eff_means, 0.3,
             yerr=k_eff_stds, color='#e74c3c', alpha=0.7, label='K_eff', capsize=3)
ax1.set_xticks(range(len(EPS_VALUES)))
ax1.set_xticklabels([f'{e:.2f}' for e in EPS_VALUES])
ax1.set_xlabel('ε (performance filter threshold)')
ax1.set_ylabel('Models Passing Filter', color='#3498db')
ax1_twin.set_ylabel('K_eff (selected models)', color='#e74c3c')
ax1.set_title('Filter Pass Rate and Effective Ensemble Size')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Panel 2: Stability, Accuracy, Equity vs epsilon
stab_vals = [importance_stability(eps_results[eps]['imp_runs'])
             if len(eps_results[eps]['imp_runs']) >= 2 else float('nan')
             for eps in EPS_VALUES]
acc_vals = [np.mean(eps_results[eps]['acc_runs']) for eps in EPS_VALUES]
eq_vals = [np.mean(eps_results[eps]['eq_runs']) for eps in EPS_VALUES]

axes[1].plot(EPS_VALUES, stab_vals, 'o-', color='#2ecc71', label='Stability', linewidth=2)
axes[1].plot(EPS_VALUES, acc_vals, 's-', color='#3498db', label='Accuracy', linewidth=2)
axes[1].plot(EPS_VALUES, eq_vals, '^-', color='#e74c3c', label='Equity (CV, lower=better)', linewidth=2)
axes[1].set_xlabel('ε (performance filter threshold)')
axes[1].set_ylabel('Metric Value')
axes[1].set_title('DASH Performance vs. ε')
axes[1].legend()

fig.suptitle('Epsilon Sensitivity Analysis — Synthetic Linear (ρ=0.9, M=500)', fontsize=13, y=1.02)
fig.tight_layout()
plt.show()"""))

    # ================================================================
    # SECTION 8: Priority 0B — Predictive Performance (RMSE)
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 8. Priority 0B: Predictive Performance Numbers

The LSM argument rests on "predicts well but explains poorly." Without validation RMSE numbers, a reviewer can dismiss the entire finding. We extract validation RMSE for Single Best, Large Single Model, and DASH consensus at each ρ level.

This enhanced sweep also saves per-repetition accuracy and equity arrays for the statistical significance tests in Section 10.

Note: For regression, `val_scores` are stored as `-RMSE`, so `RMSE = -val_score` for models using the standard pipeline. For DASH consensus predictions, we compute RMSE explicitly from the ensemble average."""))

    new_cells.append(make_cell("code", """# Enhanced correlation sweep with RMSE extraction and per-rep arrays
from sklearn.metrics import root_mean_squared_error as rmse_score

rho_levels_ext = [0.0, 0.5, 0.7, 0.9, 0.95]
sweep_methods_ext = ['Single Best', 'Large Single Model', 'DASH (MaxMin)']
EXT_N_REPS = 10

# Store everything: per-rep arrays + RMSE
ext_sweep = {rho: {} for rho in rho_levels_ext}

for rho in rho_levels_ext:
    print(f'\\n--- ρ = {rho} ---')
    for name in sweep_methods_ext:
        acc_runs, eq_runs, imp_runs, rmse_runs = [], [], [], []
        for rep in range(EXT_N_REPS):
            rep_seed = SEED + rep
            Xtr, ytr, Xv, yv, _, _, grps, true_imp, _ = \\
                generate_synthetic_linear(N=5000, rho=rho, seed=rep_seed)

            if name == 'Single Best':
                m = SingleBestBaseline(n_trials=30, seed=rep_seed)
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
                imp = m.global_importance_
                preds = m.model_.predict(Xv)
                rmse_val = rmse_score(yv, preds)

            elif name == 'Large Single Model':
                m = LargeSingleModelBaseline(K=K, T_per_model=500,
                                             colsample_bytree=0.2, seed=rep_seed)
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
                imp = m.global_importance_
                preds = m.model_.predict(Xv)
                rmse_val = rmse_score(yv, preds)

            else:  # DASH MaxMin
                m = DASHPipeline(M=M, K=K, epsilon=EPSILON, delta=DELTA,
                                selection_method='maxmin', n_jobs=-1,
                                seed=rep_seed, verbose=False)
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=feature_names)
                imp = m.global_importance_
                preds = m.get_consensus_ensemble_predictions(Xv)
                rmse_val = rmse_score(yv, preds)

            r, _ = importance_accuracy(imp, true_imp)
            acc_runs.append(r)
            eq_runs.append(within_group_equity(imp, grps))
            imp_runs.append(imp)
            rmse_runs.append(rmse_val)

        stab = importance_stability(imp_runs)
        ext_sweep[rho][name] = {
            'stability': stab,
            'accuracy_mean': np.mean(acc_runs), 'accuracy_std': np.std(acc_runs),
            'equity_mean': np.mean(eq_runs), 'equity_std': np.std(eq_runs),
            'rmse_mean': np.mean(rmse_runs), 'rmse_std': np.std(rmse_runs),
            # Save per-rep arrays for significance tests
            'acc_runs': np.array(acc_runs),
            'eq_runs': np.array(eq_runs),
            'rmse_runs': np.array(rmse_runs),
        }
        print(f'  {name:<20} stab={stab:.4f}  acc={np.mean(acc_runs):.4f}  '
              f'eq={np.mean(eq_runs):.4f}  RMSE={np.mean(rmse_runs):.4f}')

print('\\nEnhanced sweep complete.')"""))

    new_cells.append(make_cell("code", """# RMSE table and plot
print(f'{\"ρ\":>5} {\"Method\":<22} {\"Val RMSE\":>12} {\"Accuracy\":>10} {\"Stability\":>10}')
print('=' * 65)
for rho in rho_levels_ext:
    for name in sweep_methods_ext:
        r = ext_sweep[rho][name]
        print(f'{rho:>5.2f} {name:<22} {r[\"rmse_mean\"]:>8.4f}±{r[\"rmse_std\"]:<5.4f}'
              f' {r[\"accuracy_mean\"]:>10.4f} {r[\"stability\"]:>10.4f}')
    print()

# RMSE plot
fig, ax = plt.subplots(figsize=(10, 5))
for name in sweep_methods_ext:
    c = colors_sweep[name]
    m = markers_sweep[name]
    vals = [ext_sweep[rho][name]['rmse_mean'] for rho in rho_levels_ext]
    errs = [ext_sweep[rho][name]['rmse_std'] for rho in rho_levels_ext]
    ax.errorbar(rho_levels_ext, vals, yerr=errs, fmt=f'{m}-', color=c,
                label=name, linewidth=2, markersize=8, capsize=3)
ax.set_xlabel('Within-Group Correlation ρ')
ax.set_ylabel('Validation RMSE')
ax.set_title('Predictive Performance vs. Collinearity')
ax.legend()
fig.tight_layout()
plt.show()

# Key finding: Does LSM predict well but explain poorly?
print('\\n--- Key Finding: LSM Predictive vs Explanatory Quality ---')
for rho in [0.9, 0.95]:
    lsm_rmse = ext_sweep[rho]['Large Single Model']['rmse_mean']
    sb_rmse = ext_sweep[rho]['Single Best']['rmse_mean']
    dash_rmse = ext_sweep[rho]['DASH (MaxMin)']['rmse_mean']
    lsm_stab = ext_sweep[rho]['Large Single Model']['stability']
    dash_stab = ext_sweep[rho]['DASH (MaxMin)']['stability']
    print(f'  ρ={rho}: LSM RMSE={lsm_rmse:.4f} vs SB={sb_rmse:.4f} vs DASH={dash_rmse:.4f}')
    print(f'         LSM stability={lsm_stab:.4f} vs DASH={dash_stab:.4f}')"""))

    # ================================================================
    # SECTION 9: Priority 1A — Nonlinear DGP
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 9. Priority 1A: Nonlinear DGP Correlation Sweep

The linear DGP is the easiest case — the target is an exact linear combination of group means. Here we test whether DASH works with a nonlinear DGP containing:
- **Quadratic term**: β₁ · z₁²
- **Interaction**: β₂ · z₁ · z₂
- **Trigonometric**: β₃ · sin(π · z₃)
- **Linear tail**: remaining group means with random coefficients

Since the ground truth is approximate (Sobol indices ≠ SHAP values for nonlinear functions), we report **stability and equity only**, not accuracy. The key question: does DASH's stability advantage persist when the underlying relationship includes feature interactions?

We use M=200 (lighter compute) with N_REPS=10."""))

    new_cells.append(make_cell("code", """# Nonlinear DGP correlation sweep
NL_M = 200
NL_N_REPS = 10
nl_rho_levels = [0.0, 0.5, 0.7, 0.9, 0.95]
nl_methods = ['Single Best', 'DASH (MaxMin)']
nl_sweep = {rho: {} for rho in nl_rho_levels}

for rho in nl_rho_levels:
    print(f'\\n--- Nonlinear DGP, ρ = {rho} ---')
    for name in nl_methods:
        eq_runs, imp_runs = [], []
        for rep in range(NL_N_REPS):
            rep_seed = SEED + rep
            Xtr, ytr, Xv, yv, _, _, grps, _, _ = \\
                generate_synthetic_nonlinear(N=5000, rho=rho, seed=rep_seed)

            if name == 'Single Best':
                m = SingleBestBaseline(n_trials=30, seed=rep_seed)
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
                imp = m.global_importance_
            else:
                m = DASHPipeline(M=NL_M, K=K, epsilon=EPSILON, delta=DELTA,
                                selection_method='maxmin', n_jobs=-1,
                                seed=rep_seed, verbose=False)
                m.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=feature_names)
                imp = m.global_importance_

            eq_runs.append(within_group_equity(imp, grps))
            imp_runs.append(imp)

        stab = importance_stability(imp_runs)
        nl_sweep[rho][name] = {
            'stability': stab,
            'equity_mean': np.mean(eq_runs), 'equity_std': np.std(eq_runs),
            'eq_runs': np.array(eq_runs),
        }
        print(f'  {name:<20} stab={stab:.4f}  eq={np.mean(eq_runs):.4f}')

print('\\nNonlinear sweep complete.')"""))

    new_cells.append(make_cell("code", """# Nonlinear DGP results
print(f'{\"ρ\":>5} {\"Method\":<22} {\"Stability\":>10} {\"Equity (CV)\":>12}')
print('=' * 55)
for rho in nl_rho_levels:
    for name in nl_methods:
        r = nl_sweep[rho][name]
        print(f'{rho:>5.2f} {name:<22} {r[\"stability\"]:>10.4f} {r[\"equity_mean\"]:>12.4f}')
    print()

# 2-panel figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
nl_colors = {'Single Best': '#95a5a6', 'DASH (MaxMin)': '#2ecc71'}
nl_markers = {'Single Best': 's', 'DASH (MaxMin)': 'o'}

for name in nl_methods:
    c, mk = nl_colors[name], nl_markers[name]
    stab_vals = [nl_sweep[rho][name]['stability'] for rho in nl_rho_levels]
    axes[0].plot(nl_rho_levels, stab_vals, f'{mk}-', color=c, label=name,
                 linewidth=2, markersize=8)
    eq_vals = [nl_sweep[rho][name]['equity_mean'] for rho in nl_rho_levels]
    eq_errs = [nl_sweep[rho][name]['equity_std'] for rho in nl_rho_levels]
    axes[1].errorbar(nl_rho_levels, eq_vals, yerr=eq_errs, fmt=f'{mk}-',
                     color=c, label=name, linewidth=2, markersize=8, capsize=3)

axes[0].set_xlabel('Within-Group Correlation ρ')
axes[0].set_ylabel('Stability')
axes[0].set_title('Nonlinear DGP: Stability vs. Collinearity')
axes[0].legend()
axes[1].set_xlabel('Within-Group Correlation ρ')
axes[1].set_ylabel('Within-Group CV')
axes[1].set_title('Nonlinear DGP: Equity vs. Collinearity (lower=better)')
axes[1].legend()

fig.suptitle('Nonlinear DGP — DASH vs Single Best', fontsize=13, y=1.02)
fig.tight_layout()
plt.show()"""))

    # ================================================================
    # SECTION 10: Priority 1B — Statistical Significance Tests
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 10. Priority 1B: Statistical Significance Tests

We apply **Wilcoxon signed-rank tests** to per-repetition accuracy and equity values from the enhanced sweep (Section 8). At each ρ level, we test:
- DASH vs Single Best (accuracy and equity)
- DASH vs Large Single Model (accuracy and equity)

**Bonferroni correction**: With 5 ρ levels × 2 comparisons × 2 metrics = 20 tests, we multiply each p-value by 20.

**Cohen's d** effect sizes provide a standardized measure of the magnitude of the difference.

**Methodological note**: Stability is computed as a single number across all repetitions (mean pairwise Spearman ρ), not per-repetition. Wilcoxon requires paired per-rep values, so we do NOT test stability — only accuracy and equity which have N_REPS=10 paired observations."""))

    new_cells.append(make_cell("code", """# Statistical significance tests using per-rep arrays from Section 8
from dash.evaluation import compare_methods, cohens_d

N_TESTS = len(rho_levels_ext) * 2 * 2  # 5 rho × 2 comparisons × 2 metrics = 20

print(f'{\"ρ\":>5} {\"Metric\":<10} {\"Comparison\":<25} {\"W stat\":>8} {\"p-value\":>10} '
      f'{\"Bonf. p\":>10} {\"Cohen d\":>10} {\"Sig?\":>6}')
print('=' * 90)

sig_results = []
for rho in rho_levels_ext:
    dash = ext_sweep[rho]['DASH (MaxMin)']
    sb = ext_sweep[rho]['Single Best']
    lsm = ext_sweep[rho]['Large Single Model']

    for metric, metric_key in [('Accuracy', 'acc_runs'), ('Equity', 'eq_runs')]:
        for comp_name, comp_data in [('DASH vs SB', sb), ('DASH vs LSM', lsm)]:
            d_vals = dash[metric_key]
            c_vals = comp_data[metric_key]

            if len(d_vals) < 6:
                print(f'{rho:>5.2f} {metric:<10} {comp_name:<25} {"N/A (too few reps)":>30}')
                continue

            w_stat, p_val = compare_methods(d_vals, c_vals)
            bonf_p = min(p_val * N_TESTS, 1.0)
            d = cohens_d(d_vals, c_vals)
            sig = '*' if bonf_p < 0.05 else ''

            print(f'{rho:>5.2f} {metric:<10} {comp_name:<25} {w_stat:>8.1f} {p_val:>10.6f} '
                  f'{bonf_p:>10.6f} {d:>10.3f} {sig:>6}')
            sig_results.append({
                'rho': rho, 'metric': metric, 'comparison': comp_name,
                'w_stat': w_stat, 'p_val': p_val, 'bonf_p': bonf_p, 'd': d,
            })
    print()

n_sig = sum(1 for r in sig_results if r['bonf_p'] < 0.05)
print(f'\\nSignificant results (Bonferroni α=0.05): {n_sig}/{len(sig_results)}')"""))

    # ================================================================
    # SECTION 11: Priority 1C — Extended Baseline Stability (Table 2)
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 11. Priority 1C: Extended Baseline Stability (Table 2)

The correlation sweep (Sections 4 and 8) only includes Single Best, Large Single Model, and DASH (MaxMin). Table 2 in the paper requires stability values for all 8 methods. Here we run Ensemble SHAP, Stochastic Retrain, and DASH (Dedup) across N_REPS=10 at ρ=0.9 to complete the table."""))

    new_cells.append(make_cell("code", """# Extended baselines at rho=0.9
TABLE2_N_REPS = 10
table2_methods = ['Ensemble SHAP', 'Stochastic Retrain', 'DASH (Dedup)']
table2_results = {}

for name in table2_methods:
    imp_runs, acc_runs, eq_runs = [], [], []
    for rep in range(TABLE2_N_REPS):
        rep_seed = SEED + rep
        print(f'  {name} rep {rep+1}/{TABLE2_N_REPS}', end='\\r')
        Xtr, ytr, Xv, yv, _, _, grps, true_imp, _ = \\
            generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)

        if name == 'Ensemble SHAP':
            m = EnsembleSHAPBaseline(n_estimators=2000, task='regression', seed=rep_seed)
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
            imp = m.global_importance_
        elif name == 'Stochastic Retrain':
            m = StochasticRetrainBaseline(N=15, task='regression', n_jobs=-1, seed=rep_seed)
            m.fit(Xtr, ytr, Xv, yv, X_ref=Xv)
            imp = m.global_importance_
        else:  # DASH (Dedup)
            dm = DASHPipeline(M=M, K=K, epsilon=EPSILON, delta=DELTA,
                             selection_method='dedup', n_jobs=-1,
                             seed=rep_seed, verbose=False)
            dm.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=feature_names)
            imp = dm.global_importance_

        r, _ = importance_accuracy(imp, true_imp)
        acc_runs.append(r)
        eq_runs.append(within_group_equity(imp, grps))
        imp_runs.append(imp)

    table2_results[name] = {
        'stability': importance_stability(imp_runs),
        'accuracy_mean': np.mean(acc_runs), 'accuracy_std': np.std(acc_runs),
        'equity_mean': np.mean(eq_runs), 'equity_std': np.std(eq_runs),
    }
    print(f'  {name:<22} stab={table2_results[name][\"stability\"]:.4f}  '
          f'acc={np.mean(acc_runs):.4f}  eq={np.mean(eq_runs):.4f}')

print('\\nTable 2 baselines complete.')"""))

    new_cells.append(make_cell("code", """# Combined Table 2: All methods at rho=0.9
print('\\nTable 2: All Methods at ρ=0.9')
print(f'{\"Method\":<22} {\"Stability\":>10} {\"Accuracy\":>10} {\"Equity (CV)\":>12}')
print('=' * 58)

# Methods from Section 8 (ext_sweep)
for name in sweep_methods_ext:
    r = ext_sweep[0.9][name]
    print(f'{name:<22} {r[\"stability\"]:>10.4f} {r[\"accuracy_mean\"]:>10.4f} {r[\"equity_mean\"]:>12.4f}')

# Methods from this section
for name in table2_methods:
    r = table2_results[name]
    print(f'{name:<22} {r[\"stability\"]:>10.4f} {r[\"accuracy_mean\"]:>10.4f} {r[\"equity_mean\"]:>12.4f}')"""))

    # ================================================================
    # SECTION 12: Priority 2A — Superconductor UCI Benchmark
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 12. Priority 2A: Superconductor UCI Benchmark

Real-world validation beyond synthetic data. The Superconductor dataset from UCI (N=21,263, P=81) has naturally correlated material properties. No ground truth for feature importance exists, so we compare stability across methods and report prediction RMSE.

We use M=200, K=20 with N_REPS=10 to keep compute manageable."""))

    new_cells.append(make_cell("code", """# Load Superconductor dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS

print('Loading Superconductor dataset...')
data = fetch_openml(name='superconduct', version=1, as_frame=False, parser='auto')
X_sc, y_sc = data.data, data.target
sc_names = [f'f{i}' for i in range(X_sc.shape[1])]
print(f'Dataset: {X_sc.shape[0]} samples, {X_sc.shape[1]} features')

# Check collinearity
corr_sc = np.abs(np.corrcoef(X_sc.T))
n_high_sc = (np.sum(corr_sc > 0.8) - X_sc.shape[1]) // 2
print(f'Feature pairs with |r|>0.8: {n_high_sc}')

# Split and standardize
Xtr_sc, Xte_sc, ytr_sc, yte_sc = tts(X_sc, y_sc, test_size=0.2, random_state=SEED)
Xtr_sc, Xv_sc, ytr_sc, yv_sc = tts(Xtr_sc, ytr_sc, test_size=0.2, random_state=SEED)
scaler_sc = SS().fit(Xtr_sc)
Xtr_sc = scaler_sc.transform(Xtr_sc)
Xv_sc = scaler_sc.transform(Xv_sc)
Xte_sc = scaler_sc.transform(Xte_sc)
print(f'Train: {Xtr_sc.shape}, Val: {Xv_sc.shape}, Test: {Xte_sc.shape}')"""))

    new_cells.append(make_cell("code", """# Superconductor benchmark
SC_M = 200
SC_K = 20
SC_N_REPS = 10
sc_methods = ['Single Best', 'Large Single Model', 'DASH (MaxMin)']
sc_results = {}

for name in sc_methods:
    imp_runs, rmse_runs = [], []
    for rep in range(SC_N_REPS):
        rep_seed = SEED + rep
        print(f'  {name} rep {rep+1}/{SC_N_REPS}', end='\\r')

        # Use different train/val splits per rep for stability measurement
        Xtr_r, Xv_r, ytr_r, yv_r = tts(
            np.vstack([Xtr_sc, Xv_sc]), np.concatenate([ytr_sc, yv_sc]),
            test_size=0.2, random_state=rep_seed
        )

        if name == 'Single Best':
            m = SingleBestBaseline(n_trials=30, seed=rep_seed)
            m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xv_r)
            imp = m.global_importance_
            rmse_val = rmse_score(yv_r, m.model_.predict(Xv_r))
        elif name == 'Large Single Model':
            m = LargeSingleModelBaseline(K=SC_K, T_per_model=500,
                                         colsample_bytree=0.2, seed=rep_seed)
            m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xv_r)
            imp = m.global_importance_
            rmse_val = rmse_score(yv_r, m.model_.predict(Xv_r))
        else:
            m = DASHPipeline(M=SC_M, K=SC_K, epsilon=0.05, delta=DELTA,
                            selection_method='maxmin', n_jobs=-1,
                            seed=rep_seed, verbose=False)
            m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xv_r,
                  feature_names=sc_names)
            imp = m.global_importance_
            preds = m.get_consensus_ensemble_predictions(Xv_r)
            rmse_val = rmse_score(yv_r, preds)

        imp_runs.append(imp)
        rmse_runs.append(rmse_val)

    sc_results[name] = {
        'stability': importance_stability(imp_runs),
        'rmse_mean': np.mean(rmse_runs), 'rmse_std': np.std(rmse_runs),
    }
    print(f'  {name:<22} stab={sc_results[name][\"stability\"]:.4f}  '
          f'RMSE={np.mean(rmse_runs):.2f}±{np.std(rmse_runs):.2f}')

print('\\nSuperconductor benchmark complete.')"""))

    new_cells.append(make_cell("code", """# Superconductor results
print(f'{\"Method\":<22} {\"Stability\":>10} {\"Val RMSE\":>12}')
print('=' * 48)
for name in sc_methods:
    r = sc_results[name]
    print(f'{name:<22} {r[\"stability\"]:>10.4f} {r[\"rmse_mean\"]:>8.2f}±{r[\"rmse_std\"]:<5.2f}')

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc_colors = ['#95a5a6', '#e74c3c', '#2ecc71']

stabs = [sc_results[n]['stability'] for n in sc_methods]
axes[0].bar(range(len(sc_methods)), stabs, color=sc_colors, edgecolor='k', linewidth=0.5)
axes[0].set_xticks(range(len(sc_methods)))
axes[0].set_xticklabels(sc_methods, rotation=20, ha='right', fontsize=9)
axes[0].set_ylabel('Stability')
axes[0].set_title('Superconductor: Importance Stability')

rmses = [sc_results[n]['rmse_mean'] for n in sc_methods]
rmse_errs = [sc_results[n]['rmse_std'] for n in sc_methods]
axes[1].bar(range(len(sc_methods)), rmses, yerr=rmse_errs, color=sc_colors,
            edgecolor='k', linewidth=0.5, capsize=3)
axes[1].set_xticks(range(len(sc_methods)))
axes[1].set_xticklabels(sc_methods, rotation=20, ha='right', fontsize=9)
axes[1].set_ylabel('Validation RMSE')
axes[1].set_title('Superconductor: Prediction Quality')

fig.suptitle('UCI Superconductor Benchmark', fontsize=13, y=1.02)
fig.tight_layout()
plt.show()"""))

    # ================================================================
    # SECTION 13: Priority 2B — Formal Ablation Studies
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 13. Priority 2B: Formal Ablation Studies

One-at-a-time parameter variation at ρ=0.9 to characterize DASH's sensitivity to each hyperparameter. We vary:
- **M** (population size) ∈ {50, 100, 200, 500}
- **K** (max selected) ∈ {5, 10, 20, 30, 50}
- **ε** (performance filter) ∈ {0.01, 0.03, 0.05, 0.08, 0.10}
- **δ** (diversity floor) ∈ {0.01, 0.05, 0.10, 0.20}

Default baseline: M=200, K=20, ε=0.03, δ=0.05. N_REPS=5 per setting."""))

    new_cells.append(make_cell("code", """# Ablation studies — one parameter at a time
ABL_N_REPS = 5
ABL_DEFAULTS = {'M': 200, 'K': 20, 'eps': 0.03, 'delta': 0.05}

ablations = {
    'M': [50, 100, 200, 500],
    'K': [5, 10, 20, 30, 50],
    'eps': [0.01, 0.03, 0.05, 0.08, 0.10],
    'delta': [0.01, 0.05, 0.10, 0.20],
}

abl_results = {}
for param_name, values in ablations.items():
    print(f'\\n--- Ablation: {param_name} ---')
    abl_results[param_name] = {}
    for val in values:
        # Set parameters: default + override
        p = ABL_DEFAULTS.copy()
        p[param_name] = val
        print(f'  {param_name}={val}', end='... ')

        imp_runs, acc_runs = [], []
        for rep in range(ABL_N_REPS):
            rep_seed = SEED + rep
            Xtr, ytr, Xv, yv, _, _, grps, true_imp, _ = \\
                generate_synthetic_linear(N=5000, rho=0.9, seed=rep_seed)

            dm = DASHPipeline(M=p['M'], K=p['K'], epsilon=p['eps'],
                             delta=p['delta'], selection_method='maxmin',
                             n_jobs=-1, seed=rep_seed, verbose=False)
            dm.fit(Xtr, ytr, Xv, yv, X_ref=Xv, feature_names=feature_names)
            imp = dm.global_importance_
            r, _ = importance_accuracy(imp, true_imp)
            acc_runs.append(r)
            imp_runs.append(imp)

        stab = importance_stability(imp_runs)
        abl_results[param_name][val] = {
            'stability': stab,
            'accuracy_mean': np.mean(acc_runs),
            'accuracy_std': np.std(acc_runs),
        }
        print(f'stab={stab:.4f}  acc={np.mean(acc_runs):.4f}')

print('\\nAblation studies complete.')"""))

    new_cells.append(make_cell("code", """# Ablation plots — 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
param_labels = {'M': 'Population Size (M)', 'K': 'Max Selected (K)',
                'eps': 'ε (Performance Filter)', 'delta': 'δ (Diversity Floor)'}
param_defaults = {'M': 200, 'K': 20, 'eps': 0.03, 'delta': 0.05}

for ax, (param_name, values) in zip(axes.flat, ablations.items()):
    stab_vals = [abl_results[param_name][v]['stability'] for v in values]
    acc_vals = [abl_results[param_name][v]['accuracy_mean'] for v in values]
    acc_errs = [abl_results[param_name][v]['accuracy_std'] for v in values]

    ax.plot(range(len(values)), stab_vals, 'o-', color='#2ecc71',
            label='Stability', linewidth=2, markersize=8)
    ax.errorbar(range(len(values)), acc_vals, yerr=acc_errs, fmt='s-',
                color='#3498db', label='Accuracy', linewidth=2, markersize=8, capsize=3)

    # Highlight default
    default_idx = values.index(param_defaults[param_name])
    ax.axvline(default_idx, color='gray', linestyle='--', alpha=0.5, label='Default')

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(param_labels[param_name])
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Ablation: {param_labels[param_name]}')
    ax.legend(fontsize=8)

fig.suptitle('DASH Ablation Studies — Synthetic Linear (ρ=0.9)', fontsize=13, y=1.02)
fig.tight_layout()
plt.show()"""))

    # ================================================================
    # SECTION 14: Priority 2C — Publication-Quality Figures
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 14. Priority 2C: Publication-Quality Figures

Re-render the central correlation sweep figure with higher DPI, LaTeX-style labels, consistent colors, and proper legends. Saved to `results/figures/` for direct inclusion in the paper."""))

    new_cells.append(make_cell("code", """# Publication-quality correlation sweep figure
import os
os.makedirs('results/figures', exist_ok=True)

# Use publication settings
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
pub_colors = {'Single Best': '#7f8c8d', 'Large Single Model': '#c0392b', 'DASH (MaxMin)': '#27ae60'}
pub_markers = {'Single Best': 's', 'Large Single Model': 'X', 'DASH (MaxMin)': 'o'}

# Use ext_sweep data from Section 8
for name in sweep_methods_ext:
    c, mk = pub_colors[name], pub_markers[name]

    stab = [ext_sweep[rho][name]['stability'] for rho in rho_levels_ext]
    axes[0].plot(rho_levels_ext, stab, f'{mk}-', color=c, label=name,
                 linewidth=2.5, markersize=9)

    acc = [ext_sweep[rho][name]['accuracy_mean'] for rho in rho_levels_ext]
    acc_err = [ext_sweep[rho][name]['accuracy_std'] for rho in rho_levels_ext]
    axes[1].errorbar(rho_levels_ext, acc, yerr=acc_err, fmt=f'{mk}-', color=c,
                     label=name, linewidth=2.5, markersize=9, capsize=4)

    eq = [ext_sweep[rho][name]['equity_mean'] for rho in rho_levels_ext]
    eq_err = [ext_sweep[rho][name]['equity_std'] for rho in rho_levels_ext]
    axes[2].errorbar(rho_levels_ext, eq, yerr=eq_err, fmt=f'{mk}-', color=c,
                     label=name, linewidth=2.5, markersize=9, capsize=4)

axes[0].set_xlabel(r'Within-Group Correlation $\\rho$')
axes[0].set_ylabel(r'Stability (Mean Pairwise Spearman $\\rho$)')
axes[0].set_title('(a) Importance Stability')
axes[0].legend(loc='lower left')

axes[1].set_xlabel(r'Within-Group Correlation $\\rho$')
axes[1].set_ylabel(r'Spearman $\\rho$ vs. Ground Truth')
axes[1].set_title('(b) Importance Accuracy')

axes[2].set_xlabel(r'Within-Group Correlation $\\rho$')
axes[2].set_ylabel('Within-Group CV')
axes[2].set_title('(c) Within-Group Equity (lower = better)')

fig.tight_layout()
fig.savefig('results/figures/correlation_sweep_pub.png', dpi=300, bbox_inches='tight')
fig.savefig('results/figures/correlation_sweep_pub.pdf', bbox_inches='tight')
plt.show()
print('Saved to results/figures/correlation_sweep_pub.{png,pdf}')

# Reset to default DPI
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'font.family': 'sans-serif'})"""))

    # ================================================================
    # SECTION 15: Priority 3 — Additional Analyses
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 15. Priority 3: Additional Analyses

### 15.1 California Housing (Priority 3A)
Real-world regression with moderate collinearity. `sklearn.datasets.fetch_california_housing()`.

### 15.2 Cohen's d Effect Sizes (Priority 3D)
Formatted summary of effect sizes from the significance tests.

### 15.3 Bootstrap CIs on Stability (Priority 3E)
Bootstrap confidence intervals for stability estimates.

### 15.4 Deferred: UCR Time Series, Communities and Crime, Paillard comparison (3B, 3C, 3F)
Deferred to follow-up work due to compute requirements."""))

    new_cells.append(make_cell("code", """# 15.1 California Housing
from sklearn.datasets import fetch_california_housing

cal = fetch_california_housing()
X_cal, y_cal = cal.data, cal.target
cal_names = list(cal.feature_names)
print(f'California Housing: {X_cal.shape[0]} samples, {X_cal.shape[1]} features')

# Check collinearity
corr_cal = np.abs(np.corrcoef(X_cal.T))
n_high_cal = (np.sum(corr_cal > 0.7) - len(cal_names)) // 2
print(f'Feature pairs with |r|>0.7: {n_high_cal}')

Xtr_cal, Xte_cal, ytr_cal, yte_cal = tts(X_cal, y_cal, test_size=0.2, random_state=SEED)
Xtr_cal, Xv_cal, ytr_cal, yv_cal = tts(Xtr_cal, ytr_cal, test_size=0.2, random_state=SEED)
scaler_cal = SS().fit(Xtr_cal)
Xtr_cal, Xv_cal = scaler_cal.transform(Xtr_cal), scaler_cal.transform(Xv_cal)

CAL_N_REPS = 10
cal_methods = ['Single Best', 'DASH (MaxMin)']
cal_results = {}

for name in cal_methods:
    imp_runs, rmse_runs = [], []
    for rep in range(CAL_N_REPS):
        rep_seed = SEED + rep
        print(f'  {name} rep {rep+1}/{CAL_N_REPS}', end='\\r')

        Xtr_r, Xv_r, ytr_r, yv_r = tts(
            np.vstack([Xtr_cal, Xv_cal]), np.concatenate([ytr_cal, yv_cal]),
            test_size=0.2, random_state=rep_seed
        )

        if name == 'Single Best':
            m = SingleBestBaseline(n_trials=30, seed=rep_seed)
            m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xv_r)
            imp = m.global_importance_
            rmse_val = rmse_score(yv_r, m.model_.predict(Xv_r))
        else:
            m = DASHPipeline(M=200, K=20, epsilon=0.05, delta=DELTA,
                            selection_method='maxmin', n_jobs=-1,
                            seed=rep_seed, verbose=False)
            m.fit(Xtr_r, ytr_r, Xv_r, yv_r, X_ref=Xv_r, feature_names=cal_names)
            imp = m.global_importance_
            preds = m.get_consensus_ensemble_predictions(Xv_r)
            rmse_val = rmse_score(yv_r, preds)

        imp_runs.append(imp)
        rmse_runs.append(rmse_val)

    cal_results[name] = {
        'stability': importance_stability(imp_runs),
        'rmse_mean': np.mean(rmse_runs), 'rmse_std': np.std(rmse_runs),
    }
    print(f'  {name:<22} stab={cal_results[name][\"stability\"]:.4f}  '
          f'RMSE={np.mean(rmse_runs):.4f}±{np.std(rmse_runs):.4f}')

print('\\nCalifornia Housing complete.')
print(f'\\n{\"Method\":<22} {\"Stability\":>10} {\"Val RMSE\":>12}')
print('=' * 48)
for name in cal_methods:
    r = cal_results[name]
    print(f'{name:<22} {r[\"stability\"]:>10.4f} {r[\"rmse_mean\"]:>8.4f}±{r[\"rmse_std\"]:<6.4f}')"""))

    new_cells.append(make_cell("code", """# 15.2 Cohen's d effect sizes — formatted summary
print("Cohen's d Effect Sizes (from Section 10 significance tests)")
print(f'{\"ρ\":>5} {\"Metric\":<10} {\"DASH vs SB\":>12} {\"DASH vs LSM\":>14} {\"Interpretation\":>20}')
print('=' * 65)

def interpret_d(d):
    d_abs = abs(d)
    if d_abs < 0.2: return 'negligible'
    elif d_abs < 0.5: return 'small'
    elif d_abs < 0.8: return 'medium'
    else: return 'large'

for rho in rho_levels_ext:
    for metric in ['Accuracy', 'Equity']:
        d_sb = [r for r in sig_results if r['rho'] == rho
                and r['metric'] == metric and r['comparison'] == 'DASH vs SB']
        d_lsm = [r for r in sig_results if r['rho'] == rho
                 and r['metric'] == metric and r['comparison'] == 'DASH vs LSM']
        d_sb_val = d_sb[0]['d'] if d_sb else float('nan')
        d_lsm_val = d_lsm[0]['d'] if d_lsm else float('nan')
        interp = interpret_d(max(abs(d_sb_val), abs(d_lsm_val)))
        print(f'{rho:>5.2f} {metric:<10} {d_sb_val:>12.3f} {d_lsm_val:>14.3f} {interp:>20}')"""))

    new_cells.append(make_cell("code", """# 15.3 Bootstrap CIs on stability
from numpy.random import RandomState

def bootstrap_stability_ci(imp_vectors, n_bootstrap=1000, alpha=0.05, seed=42):
    \"\"\"Bootstrap confidence interval for importance stability.\"\"\"
    rng = RandomState(seed)
    n = len(imp_vectors)
    boot_stabs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_vecs = [imp_vectors[i] for i in idx]
        boot_stabs.append(importance_stability(boot_vecs))
    lo = np.percentile(boot_stabs, 100 * alpha / 2)
    hi = np.percentile(boot_stabs, 100 * (1 - alpha / 2))
    return np.mean(boot_stabs), lo, hi

# Use per-rep importance vectors from Section 8 at rho=0.9
# We need to re-collect them since ext_sweep only stored aggregated stats
# For now, use the stability_vectors from Section 3 (N_REPS=20 at rho=0.9)
print('Bootstrap 95% CIs on Stability (ρ=0.9, from Section 3)')
print(f'{\"Method\":<22} {\"Stability\":>10} {\"95% CI\":>20}')
print('=' * 55)
for name in method_names:
    if name in stability_vectors and len(stability_vectors[name]) >= 5:
        mean_s, lo, hi = bootstrap_stability_ci(stability_vectors[name])
        print(f'{name:<22} {mean_s:>10.4f} [{lo:.4f}, {hi:.4f}]')"""))

    # ================================================================
    # SECTION 16: Updated Success Criteria
    # ================================================================
    new_cells.append(make_cell("markdown", """---
## 16. Updated Success Criteria

Re-check all success criteria including the new analyses."""))

    new_cells.append(make_cell("code", """print('EXTENDED SUCCESS CRITERIA')
print('=' * 70)

# Original criteria (from Section 6)
n_wins = sum(1 for rho in rho_levels_ext
             if ext_sweep[rho]['DASH (MaxMin)']['stability']
             > ext_sweep[rho]['Single Best']['stability'])
print(f'1. Stability wins (linear): {n_wins}/{len(rho_levels_ext)} '
      f'({\"PASS\" if n_wins >= 4 else \"FAIL\"})')

acc_09 = ext_sweep[0.9]['DASH (MaxMin)']['accuracy_mean']
print(f'2. Accuracy at ρ=0.9: {acc_09:.4f} '
      f'({\"PASS\" if acc_09 >= 0.90 else \"FAIL\"})')

n_eq = sum(1 for rho in rho_levels_ext
           if ext_sweep[rho]['DASH (MaxMin)']['equity_mean']
           < ext_sweep[rho]['Single Best']['equity_mean'])
print(f'3. Equity wins (linear): {n_eq}/{len(rho_levels_ext)} '
      f'({\"PASS\" if n_eq >= 4 else \"FAIL\"})')

d0 = ext_sweep[0.0]['DASH (MaxMin)']['accuracy_mean']
s0 = ext_sweep[0.0]['Single Best']['accuracy_mean']
print(f'4. ρ=0 control: gap={abs(d0-s0):.4f} '
      f'({\"PASS\" if abs(d0-s0) < 0.1 else \"FAIL\"})')

# New criteria
print()
print('--- New Criteria ---')

# 5. Epsilon sensitivity: K_eff increases with ε
k_effs = [np.mean(eps_results[eps]['k_eff']) for eps in EPS_VALUES]
print(f'5. K_eff increases with ε: {k_effs} '
      f'({\"PASS\" if k_effs[-1] > k_effs[0] else \"CHECK\"})')

# 6. Nonlinear DGP: DASH > SB stability at rho=0.9
nl_dash_09 = nl_sweep[0.9]['DASH (MaxMin)']['stability']
nl_sb_09 = nl_sweep[0.9]['Single Best']['stability']
print(f'6. Nonlinear DGP stability (ρ=0.9): DASH={nl_dash_09:.4f} vs SB={nl_sb_09:.4f} '
      f'({\"PASS\" if nl_dash_09 > nl_sb_09 else \"CHECK\"})')

# 7. Statistical significance count
n_sig_total = sum(1 for r in sig_results if r['bonf_p'] < 0.05)
print(f'7. Significant results (Bonferroni): {n_sig_total}/{len(sig_results)}')

# 8. Superconductor: DASH stability > SB
sc_dash = sc_results['DASH (MaxMin)']['stability']
sc_sb = sc_results['Single Best']['stability']
print(f'8. Superconductor: DASH stability={sc_dash:.4f} vs SB={sc_sb:.4f} '
      f'({\"PASS\" if sc_dash > sc_sb else \"CHECK\"})')

# 9. California Housing: DASH stability > SB
cal_dash = cal_results['DASH (MaxMin)']['stability']
cal_sb = cal_results['Single Best']['stability']
print(f'9. California Housing: DASH stability={cal_dash:.4f} vs SB={cal_sb:.4f} '
      f'({\"PASS\" if cal_dash > cal_sb else \"CHECK\"})')

print()
print('All criteria evaluated.')"""))

    # Append all new cells
    nb["cells"].extend(new_cells)
    print(f"Added {len(new_cells)} new cells. Total: {len(nb['cells'])}")

    with open("notebooks/demo_benchmark_3.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook saved.")

if __name__ == "__main__":
    main()
