"""Quick stability check — regression example (California Housing)."""

from sklearn.datasets import fetch_california_housing
from dash_shap import check

data = fetch_california_housing()
result = check(data.data, data.target, feature_names=list(data.feature_names))
print(result.report())
