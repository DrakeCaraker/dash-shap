# We Trained a Bigger Model to Fix Unstable SHAP Explanations. It Made Them Worse.

You train an XGBoost model on your data. You compute SHAP values. You report: "income is the most important predictor." Your colleague retrains the same model with a different random seed — same data, same hyperparameters — and gets debt-to-income ratio as the top predictor. Both models predict equally well.

If your features are correlated, this isn't a bug. It's a mathematical property of how gradient boosting works. And the intuitive fix — training a bigger, better model — makes it *worse*.

## The Mechanism: First-Mover Bias

Gradient boosting builds trees sequentially. Each tree fits the residuals of all previous trees. When two features carry overlapping signal (income and DTI, radius and perimeter, education and occupation), the model has to pick one at each split. Whichever it picks first gets a compounding advantage: it partially removes the shared signal, making the other feature look less useful in subsequent trees.

Over hundreds of trees, this creates a **first-mover bias** — whichever feature was selected first accumulates most of the SHAP credit. Change the random seed, and a different feature goes first. The predictions barely change. The explanations change completely.

## Bigger Models Make It Worse

We tested 9 methods across 50 repetitions. A Large Single Model with 15,000 trees — matching our method's total compute — produced the **worst explanation reproducibility of any method tested**. Worse than a standard 75-tree model. More sequential trees means more compounding of the arbitrary initial choice.

![Stability across correlation levels — independent methods are flat, dependent methods degrade](../results/figures/correlation_sweep.png)
*Independent methods (DASH, Stochastic Retrain) are immune to correlation. Single-model methods (Single Best, Large Single Model) degrade monotonically.*

## The Fix Is Simple

Train 25 models with different random seeds. Average their SHAP values. The arbitrary first-mover choices cancel out.

```python
from dash_shap import check
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
result = check(data.data, data.target, task="binary",
               feature_names=list(data.feature_names))
print(result.report())
```

That's it. The `check()` function trains 25 models, computes SHAP for each, identifies which feature rankings are stable vs. unstable, and gives you consensus rankings. Under 60 seconds on a laptop.

On the Breast Cancer dataset — where 21 feature pairs have correlation above 0.9 — single-model stability is 0.376 (essentially random rankings). DASH consensus stability is 0.925. The radius/perimeter/area features, which flip arbitrarily across retrains, are correctly identified as an unstable group.

## What This Means for You

**If you use XGBoost + SHAP on correlated features:**
- Your feature rankings may be partially arbitrary
- Training a bigger model won't fix it — it'll make it worse
- Average over 25 independent models to get stable rankings
- Use the FSI diagnostic to identify which features are affected

**If you're in a regulated setting (banking, healthcare):**
- SHAP-based adverse action notices may change based solely on the training seed
- This is a "known and foreseeable circumstance" under the EU AI Act
- The `check()` function provides auditable evidence of explanation reliability

## The Science

The mechanism (first-mover bias), the fix (model independence), and the diagnostics (FSI, IS Plot) are described in our paper:

> Caraker, Arnold, Rhoads (2026). *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* [arXiv:2603.22346](https://arxiv.org/abs/2603.22346)

A companion paper proves this instability is mathematically inevitable — no single-model feature ranking can be simultaneously faithful, stable, and complete when features are collinear:

> Caraker, Arnold, Rhoads (2026). *The Attribution Impossibility.* [arXiv preprint]

Code and reproducible benchmarks: [github.com/DrakeCaraker/dash-shap](https://github.com/DrakeCaraker/dash-shap)

`pip install dash-shap`
