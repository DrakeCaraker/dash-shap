# Reddit r/MachineLearning Post

## Title
[R] SHAP explanations change when you retrain XGBoost. We found the mechanism (first-mover bias) and a simple fix

## Body

**The problem:** When features are correlated, SHAP feature importance rankings from XGBoost change every time you retrain — even with the same data and hyperparameters. The "most important feature" depends on the random seed, not the data.

**The mechanism:** We identified *first-mover bias*: gradient boosting picks features sequentially, and whichever correlated feature is selected first at early splits accumulates a compounding advantage through the residuals. More trees = more compounding. A Large Single Model with 15,000 trees produces **worse** explanations than a standard 75-tree model.

**The fix:** Train 25 models with different seeds. Average their SHAP values. The arbitrary choices cancel. We tested 9 methods across 50 repetitions on 3 real-world datasets — even simple seed averaging achieves stability of 0.977 vs 0.958 for the single-best workflow. On Breast Cancer (21 feature pairs with |r| > 0.9), stability goes from 0.376 to 0.925.

**3-line version:**
```python
from dash_shap import check
result = check(X, y, task="binary", feature_names=feature_names)
print(result.report())
```

**Paper:** [arXiv:2603.22346](https://arxiv.org/abs/2603.22346) — includes crossed ANOVA variance decomposition, TOST equivalence testing, FSI diagnostic validation (Spearman ρ = −0.995 against ground truth), and 7 appendices.

**Code:** [github.com/DrakeCaraker/dash-shap](https://github.com/DrakeCaraker/dash-shap) — `pip install dash-shap`

Happy to answer questions about the mechanism or the fix.
