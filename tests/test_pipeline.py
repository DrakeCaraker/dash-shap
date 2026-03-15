"""Smoke tests for DASHPipeline and core modules."""
import numpy as np
import pytest
import warnings
from dash.core.population import sample_configurations, DEFAULT_SEARCH_SPACE
from dash.core.filtering import performance_filter
from dash.core.diagnostics import compute_diagnostics
from dash.core.diversity import greedy_maxmin_selection
from dash.core.pipeline import DASHPipeline
from dash.experiments.synthetic import generate_synthetic_linear
from dash.utils.shap_helpers import compute_global_importance


def test_sample_configurations_count():
    configs = sample_configurations(DEFAULT_SEARCH_SPACE, M=10, seed=42)
    assert len(configs) == 10
    assert all(isinstance(c, dict) for c in configs)


def test_sample_configurations_grid():
    small_space = {"a": [1, 2], "b": [3, 4]}
    configs = sample_configurations(small_space, M=10, seed=42, strategy="grid")
    assert len(configs) == 4  # 2x2 = 4 total combos


def test_performance_filter_basic():
    scores = {0: 0.90, 1: 0.89, 2: 0.80, 3: 0.85}
    filtered = performance_filter(scores, epsilon=0.02, verbose=False)
    assert 0 in filtered
    assert 1 in filtered
    assert 2 not in filtered


def test_compute_diagnostics_shapes():
    K, N, P = 3, 10, 5
    all_shap = np.random.randn(K, N, P)
    consensus, variance, fsi, importance = compute_diagnostics(all_shap)
    assert consensus.shape == (N, P)
    assert variance.shape == (N, P)
    assert fsi.shape == (P,)
    assert importance.shape == (P,)


def test_compute_global_importance_2d():
    sv = np.array([[1.0, -2.0], [3.0, -1.0]])
    imp = compute_global_importance(sv)
    np.testing.assert_array_almost_equal(imp, [2.0, 1.5])


def test_compute_global_importance_list():
    sv = [
        np.array([[1.0, -2.0], [3.0, -1.0]]),
        np.array([[2.0, -1.0], [1.0, -3.0]]),
    ]
    imp = compute_global_importance(sv)
    assert imp.shape == (2,)
    assert all(imp > 0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_dash_pipeline_end_to_end():
    """Full pipeline produces expected shapes and non-degenerate output."""
    data = generate_synthetic_linear(N=500, P=20, group_size=5, rho=0.9, seed=42)
    X_train, y_train, X_val, y_val, X_explain = data[0], data[1], data[2], data[3], data[4]

    pipe = DASHPipeline(
        M=10, K=5, epsilon=0.15, delta=0.01,
        seed=42, verbose=False, n_jobs=1,
    )
    pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    assert pipe.consensus_matrix_.shape[1] == 20  # P features
    assert pipe.global_importance_.shape == (20,)
    assert pipe.fsi_.shape == (20,)
    assert len(pipe.selected_indices_) >= 2
    assert np.all(pipe.global_importance_ >= 0)
    assert np.sum(pipe.global_importance_) > 0  # non-degenerate


def test_dash_reproducibility():
    """Same seed produces identical output."""
    data = generate_synthetic_linear(N=500, P=20, group_size=5, rho=0.9, seed=42)
    X_train, y_train, X_val, y_val, X_explain = data[0], data[1], data[2], data[3], data[4]

    kwargs = dict(M=10, K=5, epsilon=0.15, delta=0.01, seed=42, verbose=False, n_jobs=1)

    pipe1 = DASHPipeline(**kwargs)
    pipe1.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    pipe2 = DASHPipeline(**kwargs)
    pipe2.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    np.testing.assert_array_almost_equal(
        pipe1.global_importance_, pipe2.global_importance_,
    )


def test_maxmin_selects_diverse_vectors():
    """MaxMin prefers orthogonal vectors over duplicates."""
    # 3 near-identical + 2 orthogonal vectors
    v_base = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    v_orth1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    v_orth2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    importance_vectors = {
        0: v_base,
        1: v_base + 0.01 * np.random.RandomState(1).randn(5),
        2: v_base + 0.01 * np.random.RandomState(2).randn(5),
        3: v_orth1,
        4: v_orth2,
    }
    # All models have same performance; model 0 is best
    performance_scores = {0: 0.95, 1: 0.94, 2: 0.93, 3: 0.92, 4: 0.91}

    selected = greedy_maxmin_selection(
        importance_vectors, performance_scores,
        K=3, delta=0.0, verbose=False,
    )
    # Should pick model 0 (best score), then prefer 3 and 4 (orthogonal)
    assert 0 in selected
    assert 3 in selected
    assert 4 in selected


def test_pipeline_warns_when_xref_missing():
    """Pipeline warns when X_ref defaults to X_val."""
    data = generate_synthetic_linear(N=300, P=10, group_size=5, rho=0.5, seed=42)
    X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

    pipe = DASHPipeline(M=5, K=3, epsilon=0.2, delta=0.01, seed=42, verbose=False, n_jobs=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(X_train, y_train, X_val, y_val)  # No X_ref
        xref_warnings = [x for x in w if "X_ref not provided" in str(x.message)]
        assert len(xref_warnings) == 1
