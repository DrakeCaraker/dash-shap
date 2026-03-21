"""DASH extensions — downstream analyses exploiting the K×N'×P SHAP tensor."""

__all__ = [
    "per_model_importance",
    "per_model_rankings",
    "bootstrap_over_models",
    "confidence_intervals",
    "partial_order",
    "robust_certification",
]


def __getattr__(name):
    if name in ("per_model_importance", "per_model_rankings", "bootstrap_over_models"):
        import dash_shap.extensions._base as _base

        return getattr(_base, name)
    elif name == "confidence_intervals":
        from dash_shap.extensions.confidence import confidence_intervals

        return confidence_intervals
    elif name == "partial_order":
        from dash_shap.extensions.partial_order import partial_order

        return partial_order
    elif name == "robust_certification":
        from dash_shap.extensions.certification import robust_certification

        return robust_certification
    raise AttributeError(f"module 'dash_shap.extensions' has no attribute {name}")
