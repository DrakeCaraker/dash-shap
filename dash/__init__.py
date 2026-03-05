"""DASH: Diversified Aggregation of SHAP for Stable Feature Importance Under Feature Collinearity."""
__version__ = "0.1.0"

def __getattr__(name):
    if name == "DASHPipeline":
        from dash.core.pipeline import DASHPipeline
        return DASHPipeline
    elif name == "FeatureStabilityIndex":
        from dash.core.diagnostics import FeatureStabilityIndex
        return FeatureStabilityIndex
    elif name == "ImportanceStabilityPlot":
        from dash.core.diagnostics import ImportanceStabilityPlot
        return ImportanceStabilityPlot
    elif name in ("compute_consensus", "compute_diagnostics"):
        from dash.core import consensus
        return getattr(consensus, name)
    raise AttributeError(f"module 'dash' has no attribute {name}")
