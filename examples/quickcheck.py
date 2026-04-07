"""Quick stability check for SHAP explanations — 5 lines.

Detects first-mover bias on the Breast Cancer dataset.
"""
from sklearn.datasets import load_breast_cancer
from dash_shap import check

data = load_breast_cancer()
result = check(data.data, data.target, task="binary", feature_names=list(data.feature_names))
print(result.report())
