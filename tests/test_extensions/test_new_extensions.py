"""Tests for the five newly implemented extensions."""

import numpy as np
import pytest

from dash_shap.core.result import DASHResult


def _make_result(K=12, n_ref=50, P=6, seed=0):
    """Create a DASHResult with controllable properties."""
    rng = np.random.default_rng(seed)
    # f0-f1: high importance, stable (QI)
    # f2-f3: high importance, unstable (QII)
    # f4: low importance, stable (QIII)
    # f5: low importance, unstable (QIV)
    m = np.zeros((K, n_ref, P))
    m[:, :, 0] = 2.0 + rng.normal(0, 0.05, (K, n_ref))
    m[:, :, 1] = 1.8 + rng.normal(0, 0.05, (K, n_ref))
    m[:, :, 2] = 1.5 + rng.normal(0, 0.5, (K, 1)) + rng.normal(0, 0.02, (K, n_ref))
    m[:, :, 3] = 1.5 + rng.normal(0, 0.5, (K, 1)) + rng.normal(0, 0.02, (K, n_ref))
    m[:, :, 4] = 0.1 + rng.normal(0, 0.01, (K, n_ref))
    m[:, :, 5] = 0.1 + rng.normal(0, 0.3, (K, 1)) + rng.normal(0, 0.01, (K, n_ref))
    return DASHResult.from_shap_matrices(m, feature_names=[f"f{i}" for i in range(P)])


def _make_X_ref(n=100, P=6, seed=42):
    """Create reference data with correlated pairs (f0/f1 and f2/f3)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3))
    X = np.column_stack(
        [
            base[:, 0],  # f0
            base[:, 0] + rng.normal(0, 0.1, n),  # f1 correlated with f0
            base[:, 1],  # f2
            base[:, 1] + rng.normal(0, 0.1, n),  # f3 correlated with f2
            base[:, 2],  # f4 independent
            rng.standard_normal(n),  # f5 independent
        ]
    )
    return X


# ── Causal Flags ─────────────────────────────────────────────────────────


class TestCausalFlags:
    def test_returns_result(self):
        from dash_shap.extensions.causal import causal_flags, CausalResult

        result = _make_result()
        X_ref = _make_X_ref()
        cf = causal_flags(result, X_ref)
        assert isinstance(cf, CausalResult)

    def test_correct_flag_count(self):
        from dash_shap.extensions.causal import causal_flags

        result = _make_result()
        cf = causal_flags(result, _make_X_ref())
        assert len(cf.flags) == result.P

    def test_flags_are_valid_labels(self):
        from dash_shap.extensions.causal import causal_flags

        result = _make_result()
        cf = causal_flags(result, _make_X_ref())
        valid = {"robust", "collinear", "fragile", "unimportant"}
        assert all(f in valid for f in cf.flags)

    def test_detects_correlated_pairs(self):
        from dash_shap.extensions.causal import causal_flags

        result = _make_result()
        cf = causal_flags(result, _make_X_ref(), correlation_threshold=0.5)
        # f0/f1 and f2/f3 should be correlated
        pair_features = set()
        for i, j in cf.correlated_pairs:
            pair_features.add(i)
            pair_features.add(j)
        assert 0 in pair_features and 1 in pair_features

    def test_summary_is_string(self):
        from dash_shap.extensions.causal import causal_flags

        cf = causal_flags(_make_result(), _make_X_ref())
        assert isinstance(cf.summary(), str)


# ── Audit Report ─────────────────────────────────────────────────────────


class TestAuditReport:
    def test_returns_result(self):
        from dash_shap.extensions.audit import audit_report, AuditResult

        result = _make_result()
        ar = audit_report(result)
        assert isinstance(ar, AuditResult)

    def test_basic_sections_present(self):
        from dash_shap.extensions.audit import audit_report

        ar = audit_report(_make_result())
        assert "Overview" in ar.sections
        assert "Top Features" in ar.sections

    def test_with_X_ref_adds_collinearity(self):
        from dash_shap.extensions.audit import audit_report

        ar = audit_report(_make_result(), X_ref=_make_X_ref())
        assert "Collinearity" in ar.sections

    def test_warnings_list(self):
        from dash_shap.extensions.audit import audit_report

        ar = audit_report(_make_result())
        assert isinstance(ar.warnings, list)

    def test_summary_is_string(self):
        from dash_shap.extensions.audit import audit_report

        ar = audit_report(_make_result(), X_ref=_make_X_ref())
        s = ar.summary()
        assert isinstance(s, str)
        assert "Overview" in s

    def test_with_enrichments(self):
        from dash_shap.extensions.audit import audit_report
        from dash_shap.extensions.causal import causal_flags

        result = _make_result()
        cf = causal_flags(result, _make_X_ref())
        ar = audit_report(result, X_ref=_make_X_ref(), causal=cf)
        assert "Causal Flags" in ar.sections


# ── Drift Monitor ────────────────────────────────────────────────────────


class TestDriftMonitor:
    def test_no_drift_same_result(self):
        from dash_shap.extensions.drift import DriftMonitor

        result = _make_result()
        monitor = DriftMonitor(result, threshold=0.1)
        alert = monitor.check(result)
        assert not alert.drifted
        assert alert.distance < 0.01

    def test_detects_drift(self):
        from dash_shap.extensions.drift import DriftMonitor

        baseline = _make_result(seed=0)
        different = _make_result(seed=999)
        monitor = DriftMonitor(baseline, threshold=0.01)
        alert = monitor.check(different)
        # Different seeds should produce different importance → drift
        assert alert.distance > 0

    def test_timeline_accumulates(self):
        from dash_shap.extensions.drift import DriftMonitor

        result = _make_result()
        monitor = DriftMonitor(result)
        monitor.check(result, label="v1")
        monitor.check(result, label="v2")
        assert len(monitor._timeline) == 2

    def test_feature_count_mismatch_raises(self):
        from dash_shap.extensions.drift import DriftMonitor

        r1 = _make_result(P=6)
        rng = np.random.default_rng(99)
        r2 = DASHResult.from_shap_matrices(
            rng.standard_normal((12, 50, 4)),
            feature_names=[f"g{i}" for i in range(4)],
        )
        monitor = DriftMonitor(r1)
        with pytest.raises(ValueError, match="mismatch"):
            monitor.check(r2)

    def test_summary_is_string(self):
        from dash_shap.extensions.drift import DriftMonitor

        result = _make_result()
        alert = DriftMonitor(result).check(result, label="test")
        assert isinstance(alert.summary(), str)


# ── Pareto Selector ──────────────────────────────────────────────────────


class TestParetoSelector:
    def test_basic_usage(self):
        from dash_shap.extensions.model_selection import ParetoSelector, ParetoFrontier

        selector = ParetoSelector()
        result = _make_result()
        X_test = np.random.default_rng(0).standard_normal((20, 6))
        y_test = np.random.default_rng(0).standard_normal(20)

        selector.evaluate({"M": 100}, result, X_test, y_test, predict_fn=lambda x: np.zeros(len(x)))
        selector.evaluate({"M": 200}, result, X_test, y_test, predict_fn=lambda x: np.ones(len(x)))

        frontier = selector.frontier()
        assert isinstance(frontier, ParetoFrontier)
        assert len(frontier.all_configs) == 2

    def test_frontier_nonempty(self):
        from dash_shap.extensions.model_selection import ParetoSelector

        selector = ParetoSelector()
        result = _make_result()
        X_test = np.random.default_rng(0).standard_normal((20, 6))
        y_test = np.random.default_rng(0).standard_normal(20)

        selector.evaluate({"M": 50}, result, X_test, y_test, predict_fn=lambda x: np.zeros(len(x)))
        frontier = selector.frontier()
        assert len(frontier.configs) >= 1

    def test_empty_raises(self):
        from dash_shap.extensions.model_selection import ParetoSelector

        with pytest.raises(ValueError, match="No configurations"):
            ParetoSelector().frontier()

    def test_summary_is_string(self):
        from dash_shap.extensions.model_selection import ParetoSelector

        selector = ParetoSelector()
        result = _make_result()
        X_test = np.random.default_rng(0).standard_normal((20, 6))
        y_test = np.random.default_rng(0).standard_normal(20)
        selector.evaluate({"M": 100}, result, X_test, y_test, predict_fn=lambda x: np.zeros(len(x)))
        assert isinstance(selector.frontier().summary(), str)


# ── Federated Consensus ─────────────────────────────────────────────────


class TestFederatedConsensus:
    def test_returns_result(self):
        from dash_shap.extensions.federated import federated_consensus, FederatedResult

        r1 = _make_result(seed=0)
        r2 = _make_result(seed=1)
        fed = federated_consensus([r1, r2])
        assert isinstance(fed, FederatedResult)

    def test_combined_is_dash_result(self):
        from dash_shap.extensions.federated import federated_consensus

        r1 = _make_result(seed=0)
        r2 = _make_result(seed=1)
        fed = federated_consensus([r1, r2])
        assert isinstance(fed.combined, DASHResult)

    def test_per_site_importance_shape(self):
        from dash_shap.extensions.federated import federated_consensus

        r1 = _make_result(seed=0)
        r2 = _make_result(seed=1)
        r3 = _make_result(seed=2)
        fed = federated_consensus([r1, r2, r3])
        assert fed.per_site_importance.shape == (3, 6)

    def test_cross_site_agreement_range(self):
        from dash_shap.extensions.federated import federated_consensus

        r1 = _make_result(seed=0)
        r2 = _make_result(seed=0)  # same seed = same result
        fed = federated_consensus([r1, r2])
        assert -1.0 <= fed.cross_site_agreement <= 1.0

    def test_rejects_single_site(self):
        from dash_shap.extensions.federated import federated_consensus

        with pytest.raises(ValueError, match="at least 2"):
            federated_consensus([_make_result()])

    def test_rejects_mismatched_features(self):
        from dash_shap.extensions.federated import federated_consensus

        r1 = _make_result(P=6)
        rng = np.random.default_rng(99)
        r2 = DASHResult.from_shap_matrices(
            rng.standard_normal((12, 50, 4)),
            feature_names=[f"g{i}" for i in range(4)],
        )
        with pytest.raises(ValueError, match="mismatch"):
            federated_consensus([r1, r2])

    def test_weighted_consensus(self):
        from dash_shap.extensions.federated import federated_consensus

        r1 = _make_result(seed=0)
        r2 = _make_result(seed=1)
        fed = federated_consensus([r1, r2], weights=[0.8, 0.2])
        assert fed.n_sites == 2

    def test_summary_is_string(self):
        from dash_shap.extensions.federated import federated_consensus

        fed = federated_consensus([_make_result(seed=0), _make_result(seed=1)])
        assert isinstance(fed.summary(), str)
