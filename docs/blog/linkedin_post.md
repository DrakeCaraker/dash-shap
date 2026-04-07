# LinkedIn Post

If your organization uses SHAP-based feature importance for model validation or regulatory reporting, you should know: the feature rankings may change substantially when the model is retrained with a different random seed.

This isn't a software bug. It's a mathematical property of how gradient boosting handles correlated features. We call it "first-mover bias" — whichever feature is selected first at early tree splits accumulates a compounding advantage, concentrating SHAP importance on an arbitrary feature rather than distributing it across the correlated group.

The counterintuitive part: training a bigger model to fix this makes it worse. More sequential trees means more compounding of the arbitrary initial choice.

The fix is straightforward: train 25 models independently (different random seeds) and average their SHAP values. The arbitrary choices cancel out. We validated this across 9 methods, 50 repetitions, and 3 real-world datasets including the Breast Cancer benchmark, where stability improves from 0.376 to 0.925.

We've released a diagnostic tool that detects which features are affected:

```python
from dash_shap import check
result = check(X, y, feature_names=feature_names)
print(result.report())
```

For model risk teams: this constitutes a "known and foreseeable circumstance" under EU AI Act Art. 13 and is relevant to SR 11-7 model validation requirements. The diagnostic output provides auditable evidence of explanation reliability.

Paper: https://arxiv.org/abs/2603.22346
Tool: https://github.com/DrakeCaraker/dash-shap
