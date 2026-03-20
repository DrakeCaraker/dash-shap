"""SHAP value processing utilities."""

import numpy as np


def compute_global_importance(shap_values):
    """Compute mean |SHAP| importance from raw SHAP values.

    Handles both single-output (2D array) and multi-output (list of 2D arrays)
    SHAP value formats.

    Parameters
    ----------
    shap_values : np.ndarray or list of np.ndarray
        Raw SHAP values from an explainer.

    Returns
    -------
    np.ndarray
        1D array of mean absolute SHAP importance per feature.
    """
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)
    return np.mean(np.abs(shap_values), axis=0)
