"""Unit tests for diversity selection and performance filtering modules."""
import numpy as np
import pytest
from dash_shap.core.diversity import (
    greedy_maxmin_selection,
    cluster_coverage_selection,
    deduplication_selection,
)
from dash_shap.core.filtering import performance_filter


# ---------------------------------------------------------------------------
# greedy_maxmin_selection
# ---------------------------------------------------------------------------

class TestGreedyMaxMinSelection:
    """Tests for greedy_maxmin_selection."""

    def _make_vectors(self):
        """Create a set of importance vectors with known structure."""
        return {
            0: np.array([1.0, 0.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 0.0, 1.0]),
            4: np.array([1.0, 0.01, 0.0, 0.0]),  # near-duplicate of 0
        }

    def _make_scores(self):
        return {0: 0.95, 1: 0.93, 2: 0.92, 3: 0.91, 4: 0.90}

    def test_selects_up_to_k(self):
        vecs = self._make_vectors()
        scores = self._make_scores()
        selected = greedy_maxmin_selection(vecs, scores, K=3, delta=0.0, verbose=False)
        assert len(selected) == 3

    def test_best_model_always_first(self):
        vecs = self._make_vectors()
        scores = self._make_scores()
        selected = greedy_maxmin_selection(vecs, scores, K=3, delta=0.0, verbose=False)
        assert selected[0] == 0  # highest score

    def test_prefers_diverse_over_duplicate(self):
        vecs = self._make_vectors()
        scores = self._make_scores()
        selected = greedy_maxmin_selection(vecs, scores, K=4, delta=0.0, verbose=False)
        # Orthogonal vectors 1,2,3 should be selected over near-duplicate 4
        assert 4 not in selected
        assert set(selected) == {0, 1, 2, 3}

    def test_delta_stops_early(self):
        vecs = self._make_vectors()
        scores = self._make_scores()
        # Very high delta should stop after first model
        selected = greedy_maxmin_selection(vecs, scores, K=5, delta=2.0, verbose=False)
        assert len(selected) == 1

    def test_single_model(self):
        vecs = {0: np.array([1.0, 2.0])}
        scores = {0: 0.9}
        selected = greedy_maxmin_selection(vecs, scores, K=5, delta=0.0, verbose=False)
        assert selected == [0]


# ---------------------------------------------------------------------------
# cluster_coverage_selection
# ---------------------------------------------------------------------------

class TestClusterCoverageSelection:
    """Tests for cluster_coverage_selection."""

    def test_returns_list_of_indices(self):
        rng = np.random.RandomState(42)
        P = 10
        X_train = rng.randn(100, P)
        vecs = {i: rng.rand(P) for i in range(5)}
        scores = {i: 0.9 - 0.01 * i for i in range(5)}
        selected = cluster_coverage_selection(
            vecs, scores, X_train, tau=0.5, K=3, verbose=False,
        )
        assert isinstance(selected, list)
        assert len(selected) <= 3
        assert all(s in vecs for s in selected)

    def test_respects_k_limit(self):
        rng = np.random.RandomState(42)
        P = 6
        X_train = rng.randn(50, P)
        vecs = {i: rng.rand(P) for i in range(10)}
        scores = {i: 0.9 - 0.01 * i for i in range(10)}
        selected = cluster_coverage_selection(
            vecs, scores, X_train, tau=0.3, K=4, verbose=False,
        )
        assert len(selected) <= 4


# ---------------------------------------------------------------------------
# deduplication_selection
# ---------------------------------------------------------------------------

class TestDeduplicationSelection:
    """Tests for deduplication_selection."""

    def test_removes_near_duplicates(self):
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vecs = {
            0: base,
            1: base + 0.001 * np.array([1, -1, 1, -1, 1]),  # near-duplicate
            2: np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # reversed, different
        }
        scores = {0: 0.95, 1: 0.90, 2: 0.92}
        selected = deduplication_selection(vecs, scores, rho_threshold=0.95, verbose=False)
        # Should keep 0 (better score) and 2 (different), drop 1
        assert 0 in selected
        assert 2 in selected
        assert len(selected) == 2

    def test_keeps_all_when_diverse(self):
        vecs = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
        }
        scores = {0: 0.9, 1: 0.9, 2: 0.9}
        selected = deduplication_selection(vecs, scores, rho_threshold=0.95, verbose=False)
        assert len(selected) == 3

    def test_keeps_better_performer(self):
        base = np.array([1.0, 2.0, 3.0])
        vecs = {
            0: base,
            1: base * 1.001,  # near-identical
        }
        scores = {0: 0.80, 1: 0.95}
        selected = deduplication_selection(vecs, scores, rho_threshold=0.9, verbose=False)
        assert 1 in selected  # better score kept
        assert len(selected) == 1


# ---------------------------------------------------------------------------
# performance_filter modes
# ---------------------------------------------------------------------------

class TestPerformanceFilterModes:
    """Tests for relative and quantile filter modes."""

    def test_absolute_mode(self):
        scores = {0: 0.90, 1: 0.88, 2: 0.80, 3: 0.85}
        filtered = performance_filter(scores, epsilon=0.03, mode='absolute', verbose=False)
        assert 0 in filtered
        assert 1 in filtered
        assert 2 not in filtered
        assert 3 not in filtered

    def test_relative_mode(self):
        scores = {0: 1.00, 1: 0.95, 2: 0.80}
        # relative: threshold = |1.0| * 0.1 = 0.1
        filtered = performance_filter(scores, epsilon=0.1, mode='relative', verbose=False)
        assert 0 in filtered
        assert 1 in filtered
        assert 2 not in filtered  # 0.20 > 0.10

    def test_relative_mode_scale_invariant(self):
        scores_small = {0: 0.10, 1: 0.095, 2: 0.05}
        scores_large = {0: 100.0, 1: 95.0, 2: 50.0}
        eps = 0.1
        filtered_small = performance_filter(scores_small, epsilon=eps, mode='relative', verbose=False)
        filtered_large = performance_filter(scores_large, epsilon=eps, mode='relative', verbose=False)
        # Both should filter out index 2 (50% away from best)
        assert 2 not in filtered_small
        assert 2 not in filtered_large

    def test_quantile_mode(self):
        scores = {i: 0.5 + 0.05 * i for i in range(10)}
        # epsilon=0.3 → keep top 30% = 3 models (indices 7, 8, 9)
        filtered = performance_filter(scores, epsilon=0.3, mode='quantile', verbose=False)
        assert 9 in filtered
        assert 8 in filtered
        assert len(filtered) >= 2  # at least 2 (min cutoff)

    def test_quantile_mode_keeps_at_least_two(self):
        scores = {0: 0.9, 1: 0.8}
        # Even with very small epsilon, quantile keeps at least 2
        filtered = performance_filter(scores, epsilon=0.01, mode='quantile', verbose=False)
        assert len(filtered) >= 2

    def test_lower_is_better(self):
        # RMSE-like scores where lower is better
        scores = {0: 0.10, 1: 0.12, 2: 0.50}
        filtered = performance_filter(
            scores, epsilon=0.05, higher_is_better=False,
            mode='absolute', verbose=False,
        )
        assert 0 in filtered
        assert 1 in filtered
        assert 2 not in filtered
