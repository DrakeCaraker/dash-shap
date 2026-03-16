# Experiment Summary

Format experiment results into paper-ready tables.

## Step 1: Determine canonical source

- Ask the user (or infer from context) whether they are working on **ArXiv** or **TMLR** content
- **ArXiv**: Use `notebooks/demo_benchmark_6.ipynb` results and associated checkpoints
- **TMLR**: Use `notebooks/demo_benchmark_7.ipynb` results (once available)
- Also check `results/` directory for JSON output from `run_experiments.py`

## Step 2: Gather results

1. Read available result files: `ls results/*.json 2>/dev/null`
2. Check for checkpoint data: `ls checkpoints/ 2>/dev/null`
3. Read the canonical notebook to extract inline results if no JSON files exist
4. Read `docs/BENCHMARK_RESULTS.md` for the current documented results

## Step 3: Format output

For each experiment with results, produce:

1. **Markdown table** — methods as rows, metrics (stability, DGP agreement, equity) as columns
2. **LaTeX table fragment** — ready to paste into `paper/` source files, using `\begin{tabular}` format
3. **Regression check** — compare against headline numbers in `CLAUDE.md` "Key Results" section; flag any metric that decreased by >0.005

## Step 4: Provenance

Clearly label every table with:
- Source: which notebook or JSON file the numbers came from
- Date: when the results were generated (file modification time)
- Config: which PAPER_CONFIG values were used
