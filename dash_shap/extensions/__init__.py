"""DASH Extensions — analysis beyond the consensus mean.

All extensions accept a DASHResult and return a result dataclass with
.summary() -> str and .plot() -> Figure methods.

Flat import (recommended):
    from dash_shap.extensions import confidence_intervals, robust_certification

Sub-module import (for result types):
    from dash_shap.extensions.confidence import confidence_intervals, ConfidenceResult
"""

__all__ = [
    "confidence_intervals",
    "partial_order",
    "feature_groups",
    "stable_feature_selection",
    "local_uncertainty",
    "robust_certification",
]

_EXTENSION_MAP = {
    "confidence_intervals": ("dash_shap.extensions.confidence", "confidence_intervals"),
    "partial_order": ("dash_shap.extensions.partial_order", "partial_order"),
    "feature_groups": ("dash_shap.extensions.groups", "feature_groups"),
    "stable_feature_selection": ("dash_shap.extensions.selection", "stable_feature_selection"),
    "local_uncertainty": ("dash_shap.extensions.local", "local_uncertainty"),
    "robust_certification": ("dash_shap.extensions.certification", "robust_certification"),
}

_PLANNED = {"audit_report", "causal_flags", "DriftMonitor", "ParetoSelector", "federated_consensus"}


def __getattr__(name):
    if name in _PLANNED:
        raise NotImplementedError(
            f"'dash_shap.extensions.{name}' is planned for a future release. See docs/EXTENSIONS.md for the roadmap."
        )
    if name in _EXTENSION_MAP:
        module_path, attr = _EXTENSION_MAP[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'dash_shap.extensions' has no attribute {name!r}")
