# Twitter/X Thread — SHAP Instability

## Tweet 1 (hook)
Your SHAP feature rankings from XGBoost are partially random when features are correlated.

Retrain with a different seed → different "most important feature." Same data, same accuracy.

We found out exactly why — and the fix is 3 lines of Python.

🧵

## Tweet 2 (mechanism)
The mechanism: gradient boosting picks features sequentially. Whichever correlated feature is selected first gets a compounding advantage through the residuals.

We call it "first-mover bias" — the same arbitrary advantage, just in feature attribution instead of markets.

## Tweet 3 (counterintuitive + figure)
The intuitive fix — a bigger model — produces the WORST explanations of any method we tested.

15,000 trees = more sequential compounding = more arbitrary concentration.

Independent methods (average 25 small models) are immune.

[ATTACH: correlation_sweep.png]

## Tweet 4 (the fix)
The fix:

```python
from dash_shap import check
result = check(X, y, task="binary")
print(result.report())
```

Trains 25 models, computes SHAP, identifies unstable pairs, gives stable consensus rankings. Under 60 seconds.

## Tweet 5 (evidence)
9 methods, 50 reps, 3 real-world datasets, crossed ANOVA.

Key finding: even simple seed averaging works as well as our full pipeline. Because independence — not pipeline design — is what resolves first-mover bias.

Breast Cancer: stability from 0.376 → 0.925.

## Tweet 6 (links)
Paper: arxiv.org/abs/2603.22346
Code: github.com/DrakeCaraker/dash-shap
pip install dash-shap

Companion paper proves this is mathematically inevitable (impossibility theorem, Lean 4 verified).
