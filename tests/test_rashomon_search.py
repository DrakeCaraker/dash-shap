"""Tests for surrogate-assisted Rashomon set search."""

import numpy as np
import pytest

from dash_shap.core.population import DEFAULT_SEARCH_SPACE
from dash_shap.core.rashomon_search import (
    RashomonSurrogate,
    decode_config,
    diverse_rashomon_acquisition,
    encode_config,
    encode_configs,
    level_set_boundary_acquisition,
    rashomon_probability_acquisition,
    rashomon_search,
)
from dash_shap.experiments.synthetic import generate_synthetic_linear


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data():
    """Small synthetic dataset for integration tests."""
    result = generate_synthetic_linear(N=1000, P=10, group_size=5, rho=0.7, seed=42)
    X_train, y_train, X_val, y_val, X_explain = result[0], result[1], result[2], result[3], result[4]
    return X_train, y_train, X_val, y_val, X_explain


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    """Encoding then decoding should recover the original config."""
    config = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "colsample_bytree": 0.3,
        "subsample": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 0.1,
        "min_child_weight": 5,
    }
    encoded = encode_config(config)
    decoded = decode_config(encoded)
    for key in config:
        assert decoded[key] == config[key], f"Mismatch on {key}: {decoded[key]} != {config[key]}"


def test_encode_decode_boundary_values():
    """First and last values in each parameter should map to 0 and 1."""
    for key, values in DEFAULT_SEARCH_SPACE.items():
        sorted_vals = sorted(values)
        # First value → 0.0
        config_lo = {k: sorted(DEFAULT_SEARCH_SPACE[k])[0] for k in DEFAULT_SEARCH_SPACE}
        encoded_lo = encode_config(config_lo)
        assert np.allclose(encoded_lo, 0.0), f"First values should encode to 0.0"
        # Last value → 1.0
        config_hi = {k: sorted(DEFAULT_SEARCH_SPACE[k])[-1] for k in DEFAULT_SEARCH_SPACE}
        encoded_hi = encode_config(config_hi)
        assert np.allclose(encoded_hi, 1.0), f"Last values should encode to 1.0"
        break  # Just check the boundary property once


def test_encode_configs_batch():
    """encode_configs should return (N, d) array."""
    configs = [{k: np.random.choice(v) for k, v in DEFAULT_SEARCH_SPACE.items()} for _ in range(5)]
    X = encode_configs(configs)
    assert X.shape == (5, len(DEFAULT_SEARCH_SPACE))
    assert np.all(X >= 0) and np.all(X <= 1)


def test_encode_preserves_type():
    """Decoded config should have correct Python types (int for max_depth, etc.)."""
    encoded = np.array([0.5] * len(DEFAULT_SEARCH_SPACE))
    decoded = decode_config(encoded)
    assert isinstance(decoded["max_depth"], int)
    assert isinstance(decoded["learning_rate"], float)


# ---------------------------------------------------------------------------
# Surrogate tests
# ---------------------------------------------------------------------------


def test_surrogate_fit_predict():
    """GP surrogate should fit synthetic data and return reasonable predictions."""
    rng = np.random.RandomState(42)
    X = rng.rand(20, 3)
    y = np.sin(X[:, 0] * 3) + 0.1 * rng.randn(20)

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X, y)

    mu, sigma = surrogate.predict(X[:5])
    assert mu.shape == (5,)
    assert sigma.shape == (5,)
    assert np.all(sigma >= 0)
    # Predictions at training points should be close to observations
    assert np.corrcoef(mu, y[:5])[0, 1] > 0.5


def test_rashomon_probability():
    """High-score region should get high Rashomon probability."""
    rng = np.random.RandomState(42)
    X = rng.rand(30, 2)
    # Score is high when x[0] > 0.5
    y = X[:, 0] + 0.05 * rng.randn(30)

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X, y)

    threshold = 0.5
    X_high = np.array([[0.9, 0.5]])
    X_low = np.array([[0.1, 0.5]])

    p_high = surrogate.rashomon_probability(X_high, threshold)
    p_low = surrogate.rashomon_probability(X_low, threshold)
    assert p_high[0] > p_low[0], "High-score region should have higher Rashomon probability"


def test_level_set_entropy():
    """Entropy should be highest near the level set boundary."""
    rng = np.random.RandomState(42)
    X = rng.rand(30, 2)
    y = X[:, 0] + 0.05 * rng.randn(30)

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X, y)

    threshold = 0.5
    X_test = np.array([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]])  # low, boundary, high
    entropy = surrogate.level_set_entropy(X_test, threshold)
    assert entropy.shape == (3,)
    # Boundary point should have higher entropy than extremes
    assert entropy[1] > entropy[0] or entropy[1] > entropy[2]


# ---------------------------------------------------------------------------
# Acquisition function tests
# ---------------------------------------------------------------------------


def test_rashomon_probability_acquisition_shape():
    """Acquisition function should return scores for all candidates."""
    rng = np.random.RandomState(42)
    X_obs = rng.rand(20, 3)
    y_obs = rng.rand(20)

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X_obs, y_obs)

    X_cand = rng.rand(50, 3)
    scores = rashomon_probability_acquisition(surrogate, X_cand, threshold=0.5)
    assert scores.shape == (50,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_diverse_acquisition_spreads():
    """Diverse acquisition should prefer points far from already-found set."""
    rng = np.random.RandomState(42)
    X_obs = rng.rand(30, 2)
    y_obs = np.ones(30)  # All perfect scores → all Rashomon members

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X_obs, y_obs)

    # Found set clustered at (0.1, 0.1)
    X_found = np.array([[0.1, 0.1], [0.12, 0.11], [0.09, 0.13]])
    # Candidates: one near found, one far
    X_cand = np.array([[0.11, 0.12], [0.9, 0.9]])

    scores = diverse_rashomon_acquisition(surrogate, X_cand, threshold=0.5, X_found=X_found)
    assert scores[1] > scores[0], "Far-away candidate should score higher with diverse acquisition"


def test_diverse_acquisition_empty_found():
    """With no found models, diverse acquisition should equal probability acquisition."""
    rng = np.random.RandomState(42)
    X_obs = rng.rand(20, 2)
    y_obs = rng.rand(20)

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X_obs, y_obs)

    X_cand = rng.rand(10, 2)
    X_found = np.empty((0, 2))

    scores_div = diverse_rashomon_acquisition(surrogate, X_cand, threshold=0.3, X_found=X_found)
    scores_prob = rashomon_probability_acquisition(surrogate, X_cand, threshold=0.3)
    np.testing.assert_array_almost_equal(scores_div, scores_prob)


def test_level_set_boundary_acquisition_shape():
    """Boundary acquisition should return valid scores."""
    rng = np.random.RandomState(42)
    X_obs = rng.rand(20, 3)
    y_obs = rng.rand(20)

    surrogate = RashomonSurrogate(n_restarts=2)
    surrogate.fit(X_obs, y_obs)

    X_cand = rng.rand(50, 3)
    scores = level_set_boundary_acquisition(surrogate, X_cand, threshold=0.5)
    assert scores.shape == (50,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rashomon_search_finds_diverse_models(synthetic_data):
    """End-to-end: surrogate search should find diverse Rashomon members."""
    X_train, y_train, X_val, y_val, X_explain = synthetic_data

    models, val_scores, configs, selected, info = rashomon_search(
        X_train,
        y_train,
        X_val,
        y_val,
        X_ref=X_explain,
        budget=40,
        batch_size=5,
        n_initial=15,
        K=5,
        epsilon=0.15,
        seed=42,
        n_jobs=1,
        verbose=False,
    )

    assert len(models) == 40
    assert len(val_scores) == 40
    assert len(configs) == 40
    assert len(selected) >= 2
    assert len(selected) <= 5
    assert info["n_trained"] == 40
    assert info["hit_rate"] > 0


@pytest.mark.slow
def test_pipeline_surrogate_method(synthetic_data):
    """DASHPipeline with selection_method='surrogate' runs end-to-end."""
    from dash_shap.core.pipeline import DASHPipeline

    X_train, y_train, X_val, y_val, X_explain = synthetic_data

    pipe = DASHPipeline(
        M=40,
        K=5,
        epsilon=0.15,
        selection_method="surrogate",
        surrogate_batch_size=5,
        surrogate_n_initial=15,
        seed=42,
        n_jobs=1,
        verbose=False,
    )
    pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_explain)

    assert pipe.consensus_matrix_.shape[1] == 10  # P=10 features
    assert pipe.global_importance_.shape == (10,)
    assert pipe.fsi_.shape == (10,)
    assert len(pipe.selected_indices_) >= 2
    assert pipe.surrogate_info_ is not None
    assert pipe.surrogate_info_["n_trained"] == 40


def test_rashomon_search_invalid_acquisition():
    """Invalid acquisition function should raise ValueError."""
    rng = np.random.RandomState(42)
    X = rng.rand(20, 3)
    y = rng.rand(20)

    with pytest.raises(ValueError, match="Unknown acquisition"):
        rashomon_search(
            X,
            y,
            X,
            y,
            budget=10,
            n_initial=10,
            K=3,
            acquisition="nonexistent",
            seed=42,
            n_jobs=1,
            verbose=False,
        )
