# Changelog

## v0.2.0 (2026-04-15)

### New Extensions (5)

All 12 extensions are now implemented. New in this release:

- **`causal_flags`** — Label each feature as robust, collinear, fragile, or unimportant based on FSI + feature correlation. Requires `X_ref`.
- **`audit_report`** — Structured explanation audit with sections and warnings for stakeholder communication, regulatory review, or model documentation. Accepts optional enrichments from other extensions.
- **`DriftMonitor`** — Detect explanation drift between model versions via cosine distance on importance vectors. Tracks a timeline of checks with `.plot_timeline()`.
- **`ParetoSelector`** — Find optimal DASH configurations on the RMSE vs. stability Pareto frontier. Evaluate multiple configs, then call `.frontier()` to identify non-dominated solutions.
- **`federated_consensus`** — Combine `DASHResult` objects from multiple sites without sharing raw data. The combined result works with all other extensions.

### New Extension: Theory Bridge

Bridges formulas from the Attribution Impossibility theorem (Lean 4 verified) into practical diagnostics:

- `compute_snr()` — Per-pair signal-to-noise ratio
- `predict_flip_rate()` — Predicted flip probability via Phi(-SNR)
- `recommend_M()` — Minimum ensemble size for target stability
- `divergence_ratio()` — 1/(1-rho^2) attribution divergence bound

Integrated into `check()` — reports now include predicted flip rates and M recommendations.

### Model-Agnostic Stability Workflow

- `validate_from_attributions(matrix)` — Z-tests and flip rates from any pre-computed attribution matrix (LIME, Integrated Gradients, attention maps, etc.)
- `consensus_from_attributions(matrix)` — Averaged importance from any source
- Complements the existing model-based `validate(models, X_test)` path

### Documentation

- New practitioner-first README (install -> examples -> extensions)
- New `docs/EXTENSIONS_GUIDE.md` — all 12 extensions with worked examples, organized by use case
- Complete API reference for all extensions and stability workflow functions
- Research content moved to `docs/RESEARCH.md`

### Other Changes

- Removed paper-internal F5/F1 diagnostic labels — replaced with descriptive names
- Generalized framing: impossibility theorem proved for all iterative optimizers, not just XGBoost
- `CheckResult` now includes `recommended_M` property and `worst_snr` in DataFrames
- Backward-compatible: `f1_correlation` key retained as alias for `z_flip_correlation`

## v0.1.0 (2026-04-03)

Initial release. Core pipeline (5 stages), `check()` API, 7 extensions, 9 baselines, evaluation metrics, synthetic data generators.
