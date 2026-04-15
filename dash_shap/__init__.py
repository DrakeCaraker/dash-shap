"""DASH: Diversified Aggregation of SHAP for Stable Feature Importance Under Feature Collinearity."""

try:
    from importlib.metadata import version as _version

    __version__ = _version("dash-shap")
except Exception:
    __version__ = "0.1.0"  # fallback for editable installs or uninstalled usage

__all__ = [
    # Quick API (3-line stability check)
    "check",
    "CheckResult",
    # Full pipeline
    "DASHPipeline",
    "FeatureStabilityIndex",
    "ImportanceStabilityPlot",
    "compute_consensus",
    "compute_diagnostics",
    # Convenience exports
    "generate_synthetic_linear",
    "generate_synthetic_nonlinear",
    # Stability workflow: screen → validate → resolve
    "screen",
    "validate",
    "consensus",
    "report",
    "validate_from_attributions",
    "consensus_from_attributions",
]

# Quick API
from dash_shap.quick import check, CheckResult

# Lazy imports for the stability workflow
from dash_shap.stability import (
    screen,
    validate,
    consensus,
    report,
    validate_from_attributions,
    consensus_from_attributions,
)


def __getattr__(name):
    if name == "DASHPipeline":
        from dash_shap.core.pipeline import DASHPipeline

        return DASHPipeline
    elif name == "FeatureStabilityIndex":
        from dash_shap.core.diagnostics import FeatureStabilityIndex

        return FeatureStabilityIndex
    elif name == "ImportanceStabilityPlot":
        from dash_shap.core.diagnostics import ImportanceStabilityPlot

        return ImportanceStabilityPlot
    elif name == "compute_consensus":
        from dash_shap.core.consensus import compute_consensus

        return compute_consensus
    elif name == "compute_diagnostics":
        from dash_shap.core.diagnostics import compute_diagnostics

        return compute_diagnostics
    elif name == "generate_synthetic_linear":
        from dash_shap.experiments.synthetic import generate_synthetic_linear

        return generate_synthetic_linear
    elif name == "generate_synthetic_nonlinear":
        from dash_shap.experiments.synthetic import generate_synthetic_nonlinear

        return generate_synthetic_nonlinear
    raise AttributeError(f"module 'dash_shap' has no attribute {name}")
