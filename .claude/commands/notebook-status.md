# Notebook Status

Summarize the state of all experiment notebooks:

1. List all notebooks with sizes and dates: `ls -lh notebooks/*.ipynb`
2. For each notebook, check if it has cell outputs by reading the first few cells
3. Check for associated checkpoint files: `ls checkpoints/ 2>/dev/null`
4. Present a table with columns: Notebook | Size | Has Outputs | Status

Mark `demo_benchmark_6.ipynb` as **AUTHORITATIVE (ArXiv)** — canonical results for the ArXiv pre-print.
Mark `demo_benchmark_7.ipynb` as **IN DEVELOPMENT (TMLR)** — being developed as the TMLR publication source; not yet run/finalized.
Notebooks 0–5 are historical iterations and should not be modified.

Flag any notebook >1MB as having large embedded outputs that may need clearing before commit.
