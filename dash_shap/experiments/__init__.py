"""Experiment modules for DASH validation."""

from dash_shap.experiments.synthetic import (
    generate_synthetic_linear,
    generate_synthetic_nonlinear,
    generate_synthetic_asymmetric,
)
from dash_shap.experiments.schemas import (
    MethodStabilityMetrics,
    AsymmetricRhoMethodResult,
    VarianceDecompositionMethodResult,
    KSweepMethodResult,
    LinearSweepRhoResult,
    BackgroundSensitivityResult,
    FirstMoverBiasResult,
)

__all__ = [
    "generate_synthetic_linear",
    "generate_synthetic_nonlinear",
    "generate_synthetic_asymmetric",
    "MethodStabilityMetrics",
    "AsymmetricRhoMethodResult",
    "VarianceDecompositionMethodResult",
    "KSweepMethodResult",
    "LinearSweepRhoResult",
    "BackgroundSensitivityResult",
    "FirstMoverBiasResult",
]
