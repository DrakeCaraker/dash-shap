# DASH Extensions Framework

> **Status**: Design specification (not yet implemented).
> **Scope**: 12 extensions that exploit DASH's full K×N'×P SHAP tensor beyond the current consensus mean and FSI.

---

## 1. Foundation — `DASHResult`

**File**: `dash_shap/core/result.py` (~100 lines)

The `DASHResult` dataclass decouples downstream analysis from `DASHPipeline`.
All extensions accept a `DASHResult` as input — never the pipeline object itself.

### Dataclass Definition

```python
@dataclass
class DASHResult:
    all_shap_matrices: np.ndarray   # (K, n_ref, P) — the core tensor
    feature_names: list[str]
    val_scores: np.ndarray | None = None  # (K,) optional

    # Computed in __post_init__, not passed to __init__
    consensus: np.ndarray = field(init=False, repr=False)
    variance: np.ndarray = field(init=False, repr=False)
    global_importance: np.ndarray = field(init=False, repr=False)
    fsi: np.ndarray = field(init=False, repr=False)
```

### Behavior

- **`__post_init__`**: computes `consensus`, `variance`, `global_importance`, and `fsi`;
  locks all arrays via `.flags.writeable = False`.
- **FSI formula** (matches `compute_diagnostics()` in `dash_shap/core/diagnostics.py:14-35`):
  ```
  consensus     = mean(shap, axis=0)                    # (n_ref, P)
  variance      = var(shap, axis=0, ddof=1)             # (n_ref, P)
  global_importance = mean(|consensus|, axis=0)          # (P,)
  mean_std      = mean(sqrt(variance), axis=0)           # (P,)
  fsi           = mean_std / (global_importance + eps)    # (P,)
  ```
- **Properties**: `K`, `n_ref`, `P`, `memory_bytes` (sum of all stored arrays' `nbytes`).

### Construction & Serialization

| Method | Description |
|---|---|
| `DASHResult(matrices, names, scores)` | Direct construction; validation in `__post_init__` |
| `DASHResult.from_shap_matrices(matrices, feature_names=None, val_scores=None)` | Classmethod; validates 3D shape, K≥2, name count |
| `result.save(path)` | Two-file serialization: `.npz` (arrays) + `.json` sidecar (metadata). No pickle, no new deps. |
| `DASHResult.load(path)` | Reconstruct from `.npz` + `.json` |

### Pipeline Integration

Modify `dash_shap/core/pipeline.py:305-308` — after Stage 5, add one line:

```python
self.result_ = DASHResult.from_shap_matrices(
    self.all_shap_matrices_, self.feature_names_,
    val_scores=[self.val_scores_[i] for i in self.selected_indices_])
```

All existing attributes (`consensus_matrix_`, `fsi_`, etc.) preserved unchanged.
`result_` is purely additive.

### `fit_from_attributions()` — Neural / External Attribution Support

New method on `DASHPipeline` (~30 lines):

```python
def fit_from_attributions(self, attribution_matrices, val_scores,
                          feature_names=None):
    """Run stages 2-5 on pre-computed (M, n_ref, P) attribution matrices.

    Works with neural nets, linear models, LIME, external SHAP — any source.
    """
```

Follows the established `fit_from_population()` pattern already used by
`RandomSelectionBaseline`, `SingleBestBaseline`, and `NaiveAveragingBaseline`
(all in `dash_shap/baselines/`). This absorbs what would have been "Extension 11"
(Neural) — no separate module needed.

---

## 2. Shared Utilities — `_base.py`

**File**: `dash_shap/extensions/_base.py` (~40 lines, exactly 3 functions)

```python
def per_model_importance(result: DASHResult) -> np.ndarray:    # (K, P)
    """Mean absolute SHAP value per model per feature."""

def per_model_rankings(result: DASHResult) -> np.ndarray:      # (K, P)
    """Rank features within each model (1 = most important)."""

def bootstrap_over_models(result, stat_fn, n_boot=1000, seed=42) -> np.ndarray:
    """Resample K model indices and apply stat_fn to each bootstrap sample."""
```

Used by 5+ extensions. Resists growth — single-extension utilities stay in their
own module.

---

## 3. Extensions — 11 Modules in `dash_shap/extensions/`

All follow the pattern: `result_obj = extension_fn(dash_result, **kwargs)`.
Every result type has `.summary() -> str` and `.plot() -> Figure`.

### Extension 1: Confidence Intervals (`confidence.py`)

```python
confidence_intervals(result, alpha=0.05, n_boot=1000, seed=42) -> ConfidenceResult
```

**`ConfidenceResult`** fields:
- `importance_ci`: `(P, 3)` — lower, point, upper
- `fsi_ci`: `(P, 3)`
- `ranking_ci`: `(P, 3)`

**Implementation notes**:
- Memory-safe: resamples K indices, computes stats incrementally (no second tensor copy).
- Reuses BCa logic from `evaluation.stability_bootstrap_ci()` (line 115).

### Extension 2: Partial Orders (`partial_order.py`) — Paper 2 Core

```python
partial_order(result, alpha=0.05, method="fraction") -> PartialOrderResult
```

**`PartialOrderResult`** fields:
- `adjacency`: `(P, P)` bool — `True` if feature i is confidently more important than j
- `confidence_matrix`: `(P, P)` float — π(i>j), fraction of models ranking i above j
- `n_determined`, `n_undetermined`: int

**Methods**:
- `method="fraction"`: raw fraction of K models (fast).
- `method="bootstrap"`: bootstrap test on π.

**Design**:
- Standalone — does NOT depend on CI extension.
- Uses `per_model_importance()` from `_base.py`.
- Paper 2 decision gate: raw `confidence_matrix` values directly answer
  "within-group π ≈ 0.5?"

### Extension 3: Audit Report (`audit.py`)

```python
audit_report(result, X_ref=None, *, confidence=None, partial_order=None,
             groups=None) -> AuditResult
```

**`AuditResult`** fields:
- `sections`: dict — structured report sections
- `warnings`: list — flagged issues

**Design**:
- Basic report needs only `DASHResult`.
- `X_ref` adds collinearity analysis.
- Optional enrichments (`confidence`, `partial_order`, `groups`) are duck-typed
  with `TYPE_CHECKING` imports — zero runtime coupling.

### Extension 4: Feature Groups (`groups.py`)

```python
feature_groups(result, threshold=0.8, X_ref=None,
               method="shap_substitutability") -> GroupResult
```

**Substitutability metric**: for each pair (i, j),
`sub[i,j] = mean over observations of corr_across_K_models(shap[:,n,i], shap[:,n,j])`.
Negative values = substitutable. Cluster on `-sub` via
`scipy.cluster.hierarchy`.

**Methods**:
- `method="shap_substitutability"`: uses the K×N'×P tensor directly.
- `method="correlation"` (requires `X_ref`): hierarchical clustering on |corr|.

Reuses `evaluation.within_group_equity()` (line 171) for per-group CV.

### Extension 5: Stable Feature Selection (`selection.py`)

```python
stable_feature_selection(result, k=10, importance_weight=0.7,
                         stability_weight=0.3) -> SelectionResult
```

Composite score = weighted rank combination. QI features (high importance, low FSI)
naturally dominate.

Standalone — does not require Groups extension (optional enrichment only).

### Extension 6: Drift Monitor (`drift.py`) — Class

```python
class DriftMonitor:
    def __init__(self, baseline: DASHResult, threshold: float = 0.1): ...
    def check(self, current: DASHResult, label: str = None) -> DriftAlert: ...
    def plot_timeline(self) -> Figure: ...
```

Depends on `DASHResult` serialization for persisting baselines.

### Extension 7: Pareto Model Selection (`model_selection.py`) — Class

```python
class ParetoSelector:
    def evaluate(self, config, result, X_test, y_test) -> None: ...
    def frontier(self) -> ParetoFrontier: ...
```

Docstring warns about test-set reuse / selection bias.
Aligns with Paper 5 (months 12-18).

### Extension 8: Local Uncertainty (`local.py`)

```python
local_uncertainty(result, obs_idx, top_k=15) -> LocalResult
```

Extracts only `result.all_shap_matrices[:, obs_idx, :]` — a `(K, P)` slice,
never copies the full tensor. Extends existing `local_disagreement_map()` in
`diagnostics.py:170-204` with quantitative output (per-feature mean, std,
sign-flip rate).

### Extension 9: Robust Certification (`certification.py`)

```python
robust_certification(result, k_values=None) -> CertificationResult
```

Feature j is **certified top-k** if `max_rank(j) < k` across all K models.

**Properties**:
- Monotone: certified top-3 implies certified top-4.
- ~5 lines of core logic using `per_model_rankings()` from `_base.py`.

Standalone — does NOT depend on Partial Orders.

### Extension 10: Federated Consensus (`federated.py`)

```python
federated_consensus(results: list[DASHResult], weights=None) -> FederatedResult
```

**`FederatedResult`** fields:
- `combined`: `DASHResult` — the weighted consensus, usable by all extensions
- `per_site_importance`: `np.ndarray` — `(n_sites, P)`
- `cross_site_agreement`: `float`

**Design**: `FederatedResult` is NOT a `DASHResult` subclass (composition, not
inheritance). The K axis in the `combined` result means "sites" not "models" —
inheritance would give wrong semantics for bootstrap-based extensions.

### Extension 12: Causal Flags (`causal.py`)

```python
causal_flags(result, X_ref, groups=None, alpha=0.05) -> CausalResult
```

Combines FSI + correlation structure to produce per-feature flags:
**"robust"** / **"collinear"** / **"fragile"**.

- Works standalone (computes correlation from `X_ref`).
- Richer with pre-computed `GroupResult`.
- Requires `X_ref` explicitly passed (never stored in `DASHResult`).

### Neural Extension (Ext 11): Absorbed

Absorbed into `DASHPipeline.fit_from_attributions()`. No separate module — see
Section 1.

---

## 4. Dependency Graph

```
DASHResult (foundation)
  └── _base.py (3 utilities)
       ├── confidence (1)        ← standalone
       ├── partial_order (2)     ← standalone
       ├── feature_groups (4)    ← standalone
       ├── local_uncertainty (8) ← standalone
       ├── certification (9)     ← standalone
       ├── selection (5)         ← standalone
       ├── causal_flags (12)     ← takes optional GroupResult
       ├── audit_report (3)      ← takes optional any results
       ├── DriftMonitor (6)      ← needs DASHResult.save/load
       ├── federated (10)        ← needs DASHResult.save/load
       └── ParetoSelector (7)    ← outer loop
```

**Zero hard dependencies between extensions.** Optional enrichments flow via
typed result objects passed as keyword arguments. Inter-extension imports are
`TYPE_CHECKING`-only.

---

## 5. Phasing

| Phase | Deliverables | Rationale |
|-------|-------------|-----------|
| **0** | `DASHResult`, `_base.py`, `fit_from_attributions()`, tests, conftest fixture | Foundation; gate for Phase 1 |
| **1** | Partial Orders (2), CI (1), Certification (9) | Paper 2 core. All standalone, share `_base.py` |
| **2** | Groups (4), Local (8), Selection (5) | High practitioner value |
| **3** | Causal (12), Audit (3) | Benefit from Phase 1-2 results as optional enrichments |
| **4** | Drift (6), Federated (10) | Need stable serialization layer |
| **5** | ParetoSelector (7) | Paper 5 timeline (months 12-18) |

**Gate**: Phase 0 must pass serialization round-trip test before Phase 1 begins.

---

## 6. Testing Strategy

### Shared Fixture

New fixture in `tests/conftest.py` (alongside existing `synthetic_linear`,
`synthetic_small`, `trained_population`):

```python
@pytest.fixture(scope="session")
def dash_result():
    """DASHResult with 4 features spanning all 4 IS-plot quadrants.

    f0: high importance, low FSI   (QI:   Robust Driver)
    f1: high importance, high FSI  (QII:  Collinear Cluster)
    f2: low importance,  low FSI   (QIII: Unimportant)
    f3: low importance,  high FSI  (QIV:  Fragile)
    """
```

### Test Files

- `tests/test_result.py` — construction, validation, computed fields, serialization round-trip
- `tests/test_extensions/test_confidence.py`
- `tests/test_extensions/test_partial_order.py`
- `tests/test_extensions/test_certification.py`
- (one test file per extension)

### Key Property Tests

| Extension | Property |
|-----------|----------|
| CI (1) | Interval contains point estimate; wider at lower alpha |
| Partial Orders (2) | Transitivity; known feature confidently > noise feature |
| Certification (9) | Monotone in k |
| Federated (10) | Single-site result ≈ original |
| Drift (6) | Identical result → no alert |

---

## 7. Backward Compatibility

| Concern | Guarantee |
|---------|-----------|
| Existing pipeline attributes | `pipe.all_shap_matrices_`, `pipe.consensus_matrix_`, `pipe.fsi_`, etc. — **unchanged** |
| `pipe.result_` | Additive — new attribute, never replaces existing ones |
| `fit_from_attributions()` | Additive — new method, does not touch `fit()` |
| `dash_shap/extensions/` | New subpackage — **zero changes** to `core/`, `baselines/`, `evaluation/` |
| Import direction | `extensions → core`, `extensions → evaluation` (clean, downward only) |
