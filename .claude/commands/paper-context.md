# Paper Context

Load full DASH research context for writing tasks. Read and internalize these files:

1. `EXPERIMENT_GUIDE.md` — methodology, 10 methods, 8 experiments, fix tags
2. `docs/BENCHMARK_RESULTS.md` — results tables, key numbers, success criteria
3. `ROADMAP.md` — 5-paper research program, timeline, decision gates
4. `CLAUDE.md` — pipeline summary, canonical config

Key facts to keep loaded:
- **Venue**: TMLR (Transactions on Machine Learning Research)
- **Authors**: Caraker, Arnold, Rhoads (2026)
- **Core mechanism**: sequential residual dependency in iterative optimization causes path-dependent feature attributions; DASH breaks this via model independence
- **PAPER_CONFIG**: M=200, K=30, N_REPS=20 (ArXiv), EPSILON=0.08, DELTA=0.05
- **Headline results** (rho=0.95, from `demo_benchmark_6.ipynb`): DASH 0.977 stability vs Single Best 0.951 vs LSM 0.925
- **Real-world**: Breast Cancer DASH 0.930 vs Single Best (M=200) 0.317; Superconductor DASH 0.962 vs Single Best 0.830
- **Caveat**: Stochastic Retrain achieves within 0.001 of DASH at rho=0.9; DASH's edge is diagnostics + equity + high-collinearity regime

After loading all context, ask the user what they need help writing.
