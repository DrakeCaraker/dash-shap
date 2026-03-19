"""Baseline methods for comparison with DASH."""

__all__ = [
    "SingleBestBaseline",
    "LargeSingleModelBaseline",
    "NaiveAveragingBaseline",
    "StochasticRetrainBaseline",
    "EnsembleSHAPBaseline",
    "RandomSelectionBaseline",
    "RandomForestBaseline",
    "PermutationImportanceBaseline",
    "LightGBMSingleBestBaseline",
]


def __getattr__(name):
    if name == "SingleBestBaseline":
        from dash_shap.baselines.single_best import SingleBestBaseline
        return SingleBestBaseline
    elif name == "LargeSingleModelBaseline":
        from dash_shap.baselines.large_single import LargeSingleModelBaseline
        return LargeSingleModelBaseline
    elif name == "NaiveAveragingBaseline":
        from dash_shap.baselines.naive_averaging import NaiveAveragingBaseline
        return NaiveAveragingBaseline
    elif name == "StochasticRetrainBaseline":
        from dash_shap.baselines.stochastic_retrain import StochasticRetrainBaseline
        return StochasticRetrainBaseline
    elif name == "EnsembleSHAPBaseline":
        from dash_shap.baselines.ensemble_shap import EnsembleSHAPBaseline
        return EnsembleSHAPBaseline
    elif name == "RandomSelectionBaseline":
        from dash_shap.baselines.random_selection import RandomSelectionBaseline
        return RandomSelectionBaseline
    elif name == "RandomForestBaseline":
        from dash_shap.baselines.random_forest import RandomForestBaseline
        return RandomForestBaseline
    elif name == "PermutationImportanceBaseline":
        from dash_shap.baselines.permutation_importance import PermutationImportanceBaseline
        return PermutationImportanceBaseline
    elif name == "LightGBMSingleBestBaseline":
        from dash_shap.baselines.lightgbm_single import LightGBMSingleBestBaseline
        return LightGBMSingleBestBaseline
    raise AttributeError(f"module 'dash_shap.baselines' has no attribute {name}")
