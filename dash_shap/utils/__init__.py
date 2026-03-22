"""Utility modules for DASH."""

from dash_shap.utils.shap_helpers import compute_global_importance
from dash_shap.utils.io import save_json
from dash_shap.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    has_checkpoint,
    clear_checkpoint,
    clear_checkpoints_by_prefix,
)
from dash_shap.utils.thread_budget import (
    get_available_cores,
    compute_thread_budget,
    ThreadBudget,
)

__all__ = [
    "compute_global_importance",
    "save_json",
    "save_checkpoint",
    "load_checkpoint",
    "has_checkpoint",
    "clear_checkpoint",
    "clear_checkpoints_by_prefix",
    "get_available_cores",
    "compute_thread_budget",
    "ThreadBudget",
]
