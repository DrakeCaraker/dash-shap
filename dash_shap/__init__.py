"""DASH: Diversified Aggregation of SHAP for Stable Feature Importance Under Feature Collinearity."""
__version__ = "0.1.0"

__all__ = [
    "DASHPipeline",
    "FeatureStabilityIndex",
    "ImportanceStabilityPlot",
    "compute_consensus",
    "compute_diagnostics",
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
    raise AttributeError(f"module 'dash_shap' has no attribute {name}")
