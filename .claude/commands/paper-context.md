# Paper Context (TMLR)

Load TMLR research context for writing tasks. Read and internalize these files:

1. `EXPERIMENT_GUIDE.md` — methodology, 10 methods, 8 experiments, fix tags
2. `docs/BENCHMARK_RESULTS.md` — read the **v7 (TMLR — in progress)** section; v6 (ArXiv) numbers are also in the file but labeled frozen
3. `ROADMAP.md` — 5-paper research program, timeline, decision gates
4. `CLAUDE.md` — pipeline summary, canonical config

> ⚠️ **TMLR numbers are PENDING** — `demo_benchmark_7_parallel.ipynb` has not yet completed its full run.
> Do NOT use v6/ArXiv numbers when writing TMLR sections. Check the v7 section of BENCHMARK_RESULTS.md for current status.
> Use `/paper-context-arxiv` if working on the ArXiv/Zenodo version or responding to ArXiv comments.

Key facts (stable across versions):
- **Venue**: TMLR (Transactions on Machine Learning Research)
- **Authors**: Caraker, Arnold, Rhoads (2026)
- **Core mechanism**: sequential residual dependency in iterative optimization causes path-dependent feature attributions; DASH breaks this via model independence
- **TMLR PAPER_CONFIG**: M=200, K=30, N_REPS=50, EPSILON=0.08, DELTA=0.05, SEED=42
- **Caveat (from v6)**: Stochastic Retrain achieves within 0.001 of DASH at rho=0.9; DASH's edge is diagnostics + equity + high-collinearity regime

After loading all context, ask the user what they need help writing.
