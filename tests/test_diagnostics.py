"""Smoke tests for diagnostics visualization and summary methods."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dash_shap.core.diagnostics import (
    FeatureStabilityIndex,
    ImportanceStabilityPlot,
    compute_diagnostics,
    local_disagreement_map,
)


def _make_shap_matrices(K=6, n_ref=30, P=4, seed=0):
    """Small SHAP tensor for fast smoke tests."""
    rng = np.random.default_rng(seed)
    matrices = np.zeros((K, n_ref, P), dtype=np.float64)
    matrices[:, :, 0] = 1.0 + rng.normal(0, 0.03, size=(K, n_ref))
    matrices[:, :, 1] = 1.0 + rng.normal(0, 0.8, size=(K, 1)) + rng.normal(0, 0.02, size=(K, n_ref))
    matrices[:, :, 2] = 0.1 + rng.normal(0, 0.01, size=(K, n_ref))
    matrices[:, :, 3] = 0.1 + rng.normal(0, 0.4, size=(K, 1)) + rng.normal(0, 0.01, size=(K, n_ref))
    return matrices


class TestFeatureStabilityIndex:
    def test_get_quadrant_labels(self):
        fsi = np.array([0.03, 0.80, 0.10, 4.00])
        imp = np.array([1.0, 1.0, 0.1, 0.1])
        obj = FeatureStabilityIndex(fsi, imp, ["f0", "f1", "f2", "f3"])
        labels = obj.get_quadrant_labels()
        assert labels[0] == "I: Robust Drivers"
        assert labels[1] == "II: Collinear Cluster"
        assert labels[2] == "III: Confirmed Unimportant"
        assert labels[3] == "IV: Fragile Interactions"

    def test_get_quadrant_labels_custom_thresholds(self):
        fsi = np.array([0.1, 0.5, 0.1, 0.5])
        imp = np.array([1.0, 1.0, 0.1, 0.1])
        obj = FeatureStabilityIndex(fsi, imp)
        labels = obj.get_quadrant_labels(importance_threshold=0.5, fsi_threshold=0.3)
        assert labels[0] == "I: Robust Drivers"
        assert labels[3] == "IV: Fragile Interactions"

    def test_summary_returns_string(self):
        fsi = np.array([0.03, 0.80, 0.10, 4.00])
        imp = np.array([1.0, 1.0, 0.1, 0.1])
        obj = FeatureStabilityIndex(fsi, imp, ["f0", "f1", "f2", "f3"])
        text = obj.summary(top_k=2)
        assert "Feature Stability Summary" in text
        assert "f0" in text or "f1" in text
        assert isinstance(text, str)

    def test_default_feature_names(self):
        fsi = np.array([0.1, 0.2])
        imp = np.array([1.0, 0.5])
        obj = FeatureStabilityIndex(fsi, imp)
        assert obj.feature_names == ["f0", "f1"]


class TestImportanceStabilityPlot:
    def test_plot_without_groups(self):
        matrices = _make_shap_matrices()
        _, _, fsi, imp = compute_diagnostics(matrices)
        fig = ImportanceStabilityPlot.plot(imp, fsi, feature_names=["a", "b", "c", "d"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_groups(self):
        matrices = _make_shap_matrices()
        _, _, fsi, imp = compute_diagnostics(matrices)
        groups = np.array([0, 0, 1, 1])
        fig = ImportanceStabilityPlot.plot(imp, fsi, groups=groups)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_existing_ax(self):
        matrices = _make_shap_matrices()
        _, _, fsi, imp = compute_diagnostics(matrices)
        fig, ax = plt.subplots()
        result = ImportanceStabilityPlot.plot(imp, fsi, ax=ax)
        assert result is fig
        plt.close(fig)


class TestLocalDisagreementMap:
    def test_smoke(self):
        matrices = _make_shap_matrices()
        fig = local_disagreement_map(matrices, observation_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_params(self):
        matrices = _make_shap_matrices()
        fig = local_disagreement_map(
            matrices,
            observation_idx=5,
            feature_names=["a", "b", "c", "d"],
            top_k=3,
            title="Test Map",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
