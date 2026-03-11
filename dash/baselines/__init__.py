"""Baseline methods for comparison with DASH."""

__all__ = [
    "SingleBestBaseline",
    "LargeSingleModelBaseline",
    "NaiveAveragingBaseline",
    "StochasticRetrainBaseline",
    "EnsembleSHAPBaseline",
    "RandomSelectionBaseline",
]


def __getattr__(name):
    if name == "SingleBestBaseline":
        from dash.baselines.single_best import SingleBestBaseline
        return SingleBestBaseline
    elif name == "LargeSingleModelBaseline":
        from dash.baselines.large_single import LargeSingleModelBaseline
        return LargeSingleModelBaseline
    elif name == "NaiveAveragingBaseline":
        from dash.baselines.naive_averaging import NaiveAveragingBaseline
        return NaiveAveragingBaseline
    elif name == "StochasticRetrainBaseline":
        from dash.baselines.stochastic_retrain import StochasticRetrainBaseline
        return StochasticRetrainBaseline
    elif name == "EnsembleSHAPBaseline":
        from dash.baselines.ensemble_shap import EnsembleSHAPBaseline
        return EnsembleSHAPBaseline
    elif name == "RandomSelectionBaseline":
        from dash.baselines.random_selection import RandomSelectionBaseline
        return RandomSelectionBaseline
    raise AttributeError(f"module 'dash.baselines' has no attribute {name}")
