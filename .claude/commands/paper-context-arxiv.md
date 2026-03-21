# Paper Context (ArXiv/Zenodo — v6 FROZEN)

Load the frozen ArXiv/Zenodo research context. Use this when responding to ArXiv comments or referencing published results — not for TMLR writing.

Read and internalize these files:

1. `EXPERIMENT_GUIDE.md` — methodology, 10 methods, 8 experiments, fix tags
2. `docs/BENCHMARK_RESULTS.md` — read the **v6 (ArXiv/Zenodo — FROZEN)** section only; ignore the v7 section
3. `ROADMAP.md` — 5-paper research program, timeline, decision gates
4. `CLAUDE.md` — pipeline summary, canonical config

> ✅ **These are published, frozen numbers. Do not mix with TMLR (v7) numbers.**
> For TMLR writing, use `/paper-context` instead.

Key facts:
- **Published at**: ArXiv / Zenodo
- **Authors**: Caraker, Arnold, Rhoads (2026)
- **Core mechanism**: sequential residual dependency in iterative optimization causes path-dependent feature attributions; DASH breaks this via model independence
- **ArXiv PAPER_CONFIG**: M=200, K=30, N_REPS=20, EPSILON=0.08, DELTA=0.05, SEED=42
- **Headline results** (rho=0.95, `demo_benchmark_6.ipynb`): DASH 0.977 vs Single Best 0.951 vs LSM 0.925
- **Headline results** (rho=0.9): DASH 0.977 vs SB 0.958 vs LSM 0.938 vs SR 0.977
- **Real-world**: Breast Cancer DASH 0.930 vs Single Best (M=200) 0.317 (+0.614); Superconductor DASH 0.962 vs SB 0.830; California Housing DASH 0.982 vs SB 0.967
- **Caveat**: Stochastic Retrain achieves within 0.001 of DASH at rho=0.9; DASH's edge is diagnostics + equity + high-collinearity regime

After loading all context, ask the user what they need help writing.
