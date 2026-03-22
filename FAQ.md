# DASH-SHAP FAQ

## Is DASH Right for My Problem?

**Q: My features aren't correlated — will DASH help?**

Probably not. DASH's stability advantage is statistically significant at ρ ≥ 0.7 and grows with correlation severity. At low correlations (ρ < 0.5), single-model SHAP is already stable and DASH adds overhead without benefit. Check whether your features are correlated before using DASH.

**Q: My dataset is small (< 500 rows). Will DASH work?**

Yes, but reduce the parameters to avoid overfitting the population stage:
```python
pipe = DASHPipeline(M=50, K=10, background_size=50, epsilon=0.10, seed=42)
```
With fewer rows, each model trains faster and the background SHAP set can be smaller.

**Q: Does DASH work for classification?**

Yes. Set `task="binary"` for binary classification or `task="multiclass"` for multi-class. XGBoost's `binary:logistic` and `multi:softprob` objectives are used, respectively. SHAP values are computed the same way.

```python
pipe = DASHPipeline(M=100, K=20, task="binary", seed=42)
pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)
```

---

## Installation & Setup

**Q: How do I verify my installation?**

```bash
python -c "from dash_shap import DASHPipeline; print('DASH installation OK')"
```

Or run the quickstart script to end-to-end verify:
```bash
python examples/quickstart.py
```

**Q: What Python versions are supported?**

Python 3.9–3.12. Tested in CI on Python 3.9 and 3.11.

**Q: What's the difference between `requirements.txt` and `pyproject.toml`?**

`pyproject.toml` is the canonical dependency specification (minimum versions). `requirements.txt` pins exact versions used in CI for reproducibility. Install from `pyproject.toml` for normal development (`pip install -e .`); use `requirements.txt` only if you hit version conflicts.

**Q: How do I install LightGBM support?**

```bash
pip install -e ".[lightgbm]"
```

This enables the `LightGBMSingleBestBaseline`. It is optional — all other baselines and the core pipeline use XGBoost only.

---

## Using the Pipeline

**Q: What is `X_ref` / `X_explain` and why must it be separate from `X_test`?**

DASH uses a **four-way data split**:
- `X_train` — model fitting
- `X_val` — model selection (epsilon filter)
- `X_ref` / `X_explain` — SHAP background computation (TreeExplainer reference)
- `X_test` — final RMSE evaluation only

`X_ref` must be separate from `X_test` to avoid **data leakage**: if the SHAP background overlaps with your evaluation set, importance values are computed on data that also scores the pipeline's final performance. Keep them separate so that SHAP attributions and RMSE are independent.

Use `generate_synthetic_linear()` for a pre-built 4-way split, or create your own:
```python
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_temp, X_explain, y_temp, _ = train_test_split(X_temp, y_temp, test_size=0.12, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
```

**Q: How do I pick M and K?**

- **M=50, K=10** — quick exploration; trains fast, often sufficient for real data
- **M=200, K=30** — paper-quality results; slower but more stable and diverse
- Rule of thumb: K should be ≤ 15–20% of M after filtering. If fewer than K models pass the epsilon filter, increase epsilon or switch to `epsilon_mode="quantile"`.

**Q: What does `epsilon` control?**

`epsilon` is the performance filter threshold. Models whose validation score is more than `epsilon` below the best model's score are discarded.

- `epsilon_mode="absolute"` (default): `epsilon` is in raw score units (e.g., 0.08 R²)
- `epsilon_mode="relative"`: `epsilon` is a fraction of the best score (e.g., 8% worse)
- `epsilon_mode="quantile"`: keep the top `(1 - epsilon)` fraction of models by score

For real-world datasets where score scale varies, use `epsilon_mode="relative"` with `epsilon=0.05`.

**Q: How do I use DASH on a real dataset (not synthetic)?**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from dash_shap import DASHPipeline

X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_temp, X_explain, y_temp, _ = train_test_split(X_temp, y_temp, test_size=0.12, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

pipe = DASHPipeline(
    M=100, K=20, epsilon=0.05, epsilon_mode="relative",
    task="binary", seed=42, verbose=True,
)
pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)
print(pipe.global_importance_)
```

**Q: What is `colsample_bytree` and why is it forced low?**

`colsample_bytree` controls what fraction of features each tree in XGBoost is allowed to use. DASH forces it to 0.1–0.5 (the default search space) to ensure that each model in the population is forced to explore different features. This is the **diversity mechanism**: at high `colsample_bytree`, all models converge on the same arbitrary first-mover feature, defeating the aggregation. Low `colsample_bytree` ensures model independence.

Do not override the search space to allow `colsample_bytree > 0.5` — it breaks DASH's core property.

---

## Interpreting Results

**Q: What is FSI (Feature Stability Index)?**

FSI measures how much a feature's SHAP attribution varies across the K selected models. Concretely, FSI[j] = std(SHAP[j]) / mean(|SHAP[j]|) — the coefficient of variation of SHAP values across models for feature j.

- **Low FSI**: the feature's importance is consistent across models (stable attribution)
- **High FSI**: attribution varies greatly across models (likely collinear with another feature)

Access via `pipe.get_fsi().summary(top_k=10)`.

**Q: What are the four IS plot quadrants?**

The Importance-Stability (IS) plot places features on axes of global importance (x) and FSI (y):

| Quadrant | Importance | FSI | Interpretation |
|---|---|---|---|
| I: Robust Drivers | High | **Low** | Trustworthy — important and stable; report individually |
| II: Collinear Cluster | High | **High** | Interpret as a group — important but contested by correlated features |
| III: Confirmed Unimportant | Low | Low | Reliably irrelevant — safe to omit |
| IV: Fragile Interactions | Low | High | Unstable and weak — investigate or de-emphasize |

**Q: My top feature has high FSI — what does that mean?**

It's in a collinear group. The model is arbitrarily concentrating importance on this feature in some runs and a correlated feature in others. Report the **group's** total importance rather than any individual feature. DASH's `groups` parameter (from `generate_synthetic_linear`) or your own domain knowledge can identify the group.

---

## Performance

**Q: How do I speed up fitting?**

1. Set `n_jobs=-1` (default) — uses all CPU cores for population training and SHAP
2. Reduce `M` (e.g., M=50 instead of M=200) for exploration
3. Reduce `background_size` (e.g., 50 instead of 100) for faster SHAP computation
4. Use `run_experiments_parallel.py` for batch experiments — it shares model populations across methods

**Q: Can I run DASH on a GPU?**

No. TreeSHAP is CPU-based. XGBoost's GPU training is not exercised in DASH's population generation. All computation is CPU-bound and parallelized via joblib.

**Q: My SHAP computation hangs — what should I do?**

1. Reduce `background_size` to 50: `DASHPipeline(background_size=50, ...)`
2. Set `n_jobs=1` to disable parallelism and isolate the hang
3. Reduce K to confirm it's the SHAP stage (not population or filtering)
4. Check memory: K × background_size × n_features SHAP matrices are held in RAM

---

## Troubleshooting

**Q: Fewer than K models passed the epsilon filter. What should I do?**

If you see a warning like "Only N models passed the performance filter (K=30)", there are three fixes:

1. **Increase epsilon** — Widen the performance band. If using `epsilon_mode="absolute"`, try doubling epsilon. If using `epsilon_mode="relative"`, increase from 0.05 to 0.10.

2. **Switch to quantile mode** — `epsilon_mode="quantile"` always passes a fixed fraction of models regardless of score scale:
   ```python
   pipe = DASHPipeline(M=100, K=20, epsilon=0.5, epsilon_mode="quantile", ...)
   # Always passes top 50% of models — guarantees at least 50 candidates with M=100
   ```

3. **Increase M** — A larger population gives more candidates. With M=200 and K=30, the filter typically passes 60–100 models, leaving plenty of diversity candidates.

The rule of thumb: after filtering, you want at least 2×K models remaining. Check with:
```python
print(f"Filtered: {len(pipe.filtered_indices_)} models passed (K={pipe.K})")
```

**Q: How should I preprocess features before using DASH?**

Tree-based models (XGBoost) are scale-invariant — standardization (`StandardScaler`) is not required and will not change model predictions or SHAP values. DASH's diversity mechanism uses gain-importance vectors for the MaxMin selection in Stage 3; these are information-based (not scale-based) and will not be affected by feature scaling.

SHAP values are in output space: same units as `y` for regression, log-odds for binary classification. If you standardize `y` for regression, SHAP values will be in standardized units.

Recommendation: preprocessing is optional but harmless. You do not need to standardize before using DASH.

**Q: My SHAP computation runs out of memory. What should I do?**

In addition to the tips above (reduce `background_size`, set `n_jobs=1`):

5. **Reduce K** — each selected model holds an N' × P SHAP matrix in RAM. With K=30, N'=100, P=30, this is 30 × 100 × 30 × 8 bytes ≈ 7 MB — small. But with K=30, N'=5000, P=500, it's 30 × 5000 × 500 × 8 bytes ≈ 600 MB. Reduce K to 10 for large datasets.

6. **Reduce X_ref to 200 rows** — The SHAP background and reference set can both be trimmed:
   ```python
   pipe = DASHPipeline(M=100, K=20, background_size=50, ...)
   pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain[:200])
   ```
   200 rows is sufficient for stable global importance estimates on most datasets.
