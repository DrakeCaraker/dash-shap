# Experiment Summary

Inventory all experiment results and figures, format into paper-ready tables with full provenance.

## Step 1: Determine canonical source

- Ask the user (or infer from context) whether they are working on **ArXiv** or **TMLR** content
- **ArXiv**: Use `notebooks/demo_benchmark_6.ipynb` as the canonical notebook
- **TMLR**: Use `notebooks/demo_benchmark_7.ipynb` as the canonical notebook (when available)

## Step 2: Inventory all result artifacts

Scan all four output locations and report what exists:

### JSON tables (`results/tables/`)
List all files with modification dates. Known experiment tables:
- `synthetic_linear_sweep.json` — linear DGP correlation sweep
- `nonlinear_sweep.json` — nonlinear DGP correlation sweep
- `overlapping.json` — overlapping correlation structure
- `table2_baselines.json` — full baseline comparison
- `california_housing.json` — California Housing dataset
- `breast_cancer.json` — Breast Cancer dataset
- `superconductor.json` — Superconductor dataset
- `epsilon_sensitivity.json` — epsilon threshold sweep
- `ablation.json` — pipeline ablation study
- `variance_decomposition.json` — between/within variance analysis
- `first_mover_bias.json` — first-mover concentration data

### Figures from `run_experiments.py` (`results/figures/`)
List all `.png` and `.pdf` files with sizes. Known figures:
- `correlation_sweep.{png,pdf}` — main 3-panel synthetic sweep
- `correlation_sweep_4panel.{png,pdf}` — 4-panel version
- `nonlinear_sweep.{png,pdf}` — nonlinear DGP results
- `is_plot_synthetic.{png,pdf}` — Importance-Stability plot (synthetic)
- `is_plot_breast_cancer.{png,pdf}` — Importance-Stability plot (Breast Cancer)
- `disagreement_synthetic.{png,pdf}` — local disagreement map (synthetic)
- `disagreement_breast_cancer.{png,pdf}` — local disagreement map (Breast Cancer)
- `epsilon_sensitivity.{png,pdf}` — epsilon threshold sensitivity
- `ablation_sensitivity.{png,pdf}` — 2×2 ablation grid
- `first_mover_concentration.{png,pdf}` — first-mover visualization
- `superconductor_results.{png,pdf}` — Superconductor dataset results
- `bar_chart_rho09.{png,pdf}` — bar chart at ρ=0.9

### Figures from notebooks (`notebooks/results/figures/`)
List all `.png` and `.pdf` files. These are publication-quality versions saved from the canonical notebook (e.g., `correlation_sweep_pub.pdf`).

### Checkpoints (`checkpoints/`)
List all `.pkl` files if the directory exists. Notebook 7 checkpoints follow the naming pattern `ckpt_nb7_*.pkl`:
- `ckpt_nb7_mechanism_sweep.pkl`, `ckpt_nb7_diagnostics_demo.pkl`
- `ckpt_nb7_california.pkl`, `ckpt_nb7_breast_cancer.pkl`, `ckpt_nb7_superconductor.pkl`
- `ckpt_nb7_epsilon.pkl`, `ckpt_nb7_ablation.pkl`, `ckpt_nb7_nonlinear_v2.pkl`
- `ckpt_nb7_overlapping.pkl`, `ckpt_nb7_vardecomp.pkl`
- `ckpt_nb7_first_mover_viz.pkl`, `ckpt_nb7_first_mover_bias.pkl`
- `ckpt_nb7_bc_diagnostics.pkl`, `ckpt_v7_sec1_poc.pkl`

Note which checkpoints exist vs which are missing (indicating that experiment hasn't been run in the notebook yet).

## Step 3: Read and format results

For each JSON table that exists, read it and extract per-method metrics:

**Synthetic experiments** — key columns:
| Method | Stability | Stability CI | Accuracy | Equity (CV) | Group MSE | RMSE | K_eff | Time (s) |

**Real-world experiments** — key columns:
| Method | Stability | Stability CI | Ablation | RMSE | Time (s) |

Also extract `_significance` blocks (pairwise Wilcoxon p-values, Holm-Bonferroni corrections, Cohen's d) and present as a separate significance table when present.

Produce two formats for each table:
1. **Markdown table** — for quick review
2. **LaTeX `\begin{tabular}`** — ready to paste into `paper/` source files

## Step 4: Figure summary

Group figures by experiment and note their source:

| Experiment | Script Figure | Notebook Figure | JSON Data |
|------------|--------------|-----------------|-----------|
| Linear Sweep | `results/figures/correlation_sweep.png` | `notebooks/results/figures/correlation_sweep_pub.pdf` | `results/tables/synthetic_linear_sweep.json` |
| ... | ... | ... | ... |

Flag any experiment that has JSON data but no figure (or vice versa) — this indicates an incomplete run.

## Step 5: Regression check

Compare key metrics from the JSON tables against the canonical numbers from `demo_benchmark_6.ipynb` (also in `CLAUDE.md` "Key Results"):
- rho=0.9: DASH stability=0.977, Single Best=0.958, LSM=0.938
- rho=0.95: DASH stability=0.977, Single Best=0.951, LSM=0.925
- Breast Cancer: DASH stability=0.930, Single Best (M=200)=0.317
- Superconductor: DASH stability=0.962, Single Best=0.830, LSM=0.689
- California Housing: DASH stability=0.982, Single Best=0.967

Flag any metric that decreased by >0.005 from these reference values.

## Step 6: Provenance

Label every table and figure listing with:
- **Source**: which file the data came from (JSON path, notebook name, or checkpoint)
- **Date**: file modification time (`stat -c '%y'` or `ls -l`)
- **Config**: PAPER_CONFIG values used (read from the JSON or notebook)
- **Canonical for**: ArXiv, TMLR, or both
