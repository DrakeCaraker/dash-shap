"""TypedDicts describing experiment output schemas.

Return type annotations on experiment functions allow mypy to catch key-name
bugs at development time. No runtime overhead.

The canonical consumer of these schemas is explore_experiment_results.ipynb.
"""
from __future__ import annotations

from typing import TypedDict


class MethodStabilityMetrics(TypedDict):
    stability: float
    dgp_agreement: float
    equity: float
    rmse: float
    n_successful: int


class AsymmetricRhoMethodResult(TypedDict):
    stability: float
    bias_f0: float
    passive_leak_f1: float
    mean_importance: list


# Full type: dict[str, dict[str, AsymmetricRhoMethodResult]]
# outer keys: rho as string ("0.5", "0.7", "0.9", "0.95")
# inner keys: method names


class VarianceDecompositionMethodResult(TypedDict):
    data_var_frac: float
    model_var_frac: float
    residual_var_frac: float
    ss_data: float
    ss_model: float
    ss_residual: float


class KSweepMethodResult(TypedDict):
    stability: float
    stability_se: float
    accuracy_mean: float
    accuracy_std: float
    n_successful: int


# Full type: dict[int, dict[str, KSweepMethodResult]]
# outer keys: K values (int); inner keys: method names


class LinearSweepRhoResult(TypedDict):
    stability: float
    dgp_agreement: float
    equity: float
    rmse: float
    n_successful: int


class BackgroundSensitivityResult(TypedDict):
    stability: float
    n_successful: int


class FirstMoverBiasResult(TypedDict):
    concentration: float
    n_successful: int
