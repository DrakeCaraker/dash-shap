"""Smoke tests for DASHPipeline and core modules."""

import numpy as np
import warnings
from dash_shap.core.population import sample_configurations, DEFAULT_SEARCH_SPACE
from dash_shap.core.filtering import performance_filter
from dash_shap.core.diagnostics import compute_diagnostics
from dash_shap.core.diversity import greedy_maxmin_selection
from dash_shap.core.pipeline import DASHPipeline
from dash_shap.experiments.synthetic import generate_synthetic_linear
from dash_shap.utils.shap_helpers import compute_global_importance


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


def test_performance_filter_quantile_lower_is_better():
    # Covers line 38: quantile mode with higher_is_better=False
    scores = {0: 0.10, 1: 0.20, 2: 0.50, 3: 0.80}
    filtered = performance_filter(scores, epsilon=0.5, mode="quantile", higher_is_better=False, verbose=False)
    assert 0 in filtered  # lowest error, should be kept


def test_performance_filter_verbose(capsys):
    # Covers line 43: verbose print path
    scores = {0: 0.90, 1: 0.85}
    performance_filter(scores, epsilon=0.1, verbose=True)
    captured = capsys.readouterr()
    assert "Performance filter" in captured.out


def test_performance_filter_high_pass_rate_warning():
    # 95% of models pass with absolute mode → should warn about possible misconfiguration
    scores = {i: 0.90 + i * 0.001 for i in range(20)}
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        filtered = performance_filter(scores, epsilon=0.5, mode="absolute", verbose=False)
        assert len(filtered) / len(scores) > 0.90
        assert len(w) == 1
        assert "mode='relative'" in str(w[0].message)


def test_performance_filter_no_warning_relative_mode():
    # Relative mode should never trigger the absolute-mode warning even with high pass rate
    scores = {i: 0.90 + i * 0.001 for i in range(20)}
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        performance_filter(scores, epsilon=0.5, mode="relative", verbose=False)
        absolute_warnings = [x for x in w if "mode='relative'" in str(x.message)]
        assert len(absolute_warnings) == 0


def test_config_values():
    # Covers dash_shap/config.py (was 0% coverage)
    from dash_shap.config import PAPER_CONFIG, SEED, REAL_EPSILON, REAL_EPSILON_MODE

    assert PAPER_CONFIG["M"] == 200
    assert PAPER_CONFIG["K"] == 30
    assert SEED == 42
    assert REAL_EPSILON_MODE == "relative"


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
    data = generate_synthetic_linear(N=200, P=6, group_size=3, rho=0.9, seed=42)
    X_train, y_train, X_val, y_val, X_explain = data[0], data[1], data[2], data[3], data[4]

    pipe = DASHPipeline(
        M=5,
        K=3,
        epsilon=0.15,
        delta=0.01,
        seed=42,
        verbose=False,
        n_jobs=1,
    )
    pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    assert pipe.consensus_matrix_.shape[1] == 6  # P features
    assert pipe.global_importance_.shape == (6,)
    assert pipe.fsi_.shape == (6,)
    assert len(pipe.selected_indices_) >= 2
    assert np.all(pipe.global_importance_ >= 0)
    assert np.sum(pipe.global_importance_) > 0  # non-degenerate


def test_dash_reproducibility():
    """Same seed produces identical output."""
    data = generate_synthetic_linear(N=200, P=6, group_size=3, rho=0.9, seed=42)
    X_train, y_train, X_val, y_val, X_explain = data[0], data[1], data[2], data[3], data[4]

    kwargs = dict(M=5, K=3, epsilon=0.15, delta=0.01, seed=42, verbose=False, n_jobs=1)

    pipe1 = DASHPipeline(**kwargs)
    pipe1.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    pipe2 = DASHPipeline(**kwargs)
    pipe2.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    np.testing.assert_array_almost_equal(
        pipe1.global_importance_,
        pipe2.global_importance_,
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
        importance_vectors,
        performance_scores,
        K=3,
        delta=0.0,
        verbose=False,
    )
    # Should pick model 0 (best score), then prefer 3 and 4 (orthogonal)
    assert 0 in selected
    assert 3 in selected
    assert 4 in selected


def test_pipeline_warns_when_xref_missing():
    """Pipeline warns when X_ref defaults to X_val."""
    data = generate_synthetic_linear(N=300, P=10, group_size=5, rho=0.5, seed=42)
    X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]

    pipe = DASHPipeline(M=3, K=2, epsilon=0.2, delta=0.01, seed=42, verbose=False, n_jobs=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pipe.fit(X_train, y_train, X_val, y_val)  # No X_ref
        xref_warnings = [x for x in w if "X_ref not provided" in str(x.message)]
        assert len(xref_warnings) == 1
