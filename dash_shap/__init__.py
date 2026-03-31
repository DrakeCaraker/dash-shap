"""DASH: Diversified Aggregation of SHAP for Stable Feature Importance Under Feature Collinearity."""

try:
    from importlib.metadata import version as _version

    __version__ = _version("dash-shap")
except Exception:
    __version__ = "0.1.0"  # fallback for editable installs or uninstalled usage

__all__ = [
    "DASHPipeline",
    "FeatureStabilityIndex",
    "ImportanceStabilityPlot",
    "compute_consensus",
    "compute_diagnostics",
    # Convenience exports
    "generate_synthetic_linear",
    "generate_synthetic_nonlinear",
]


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
