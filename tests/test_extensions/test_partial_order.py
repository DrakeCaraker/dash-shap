"""Tests for Extension 2: Partial Orders."""

import numpy as np
import pytest

from dash_shap.core.result import DASHResult
from dash_shap.extensions.partial_order import partial_order, PartialOrderResult


def _make_dominant_result(seed=0):
    """f0 is overwhelmingly more important than f1, f2, f3."""
    rng = np.random.default_rng(seed)
    K, n_ref, P = 15, 30, 4
    m = rng.standard_normal((K, n_ref, P)) * 0.1
    m[:, :, 0] += 10.0  # f0 dominates
    return DASHResult.from_shap_matrices(m, feature_names=["dominant", "noise1", "noise2", "noise3"])


class TestPartialOrder:
    def test_returns_partial_order_result(self, dash_result):
        result = partial_order(dash_result)
        assert isinstance(result, PartialOrderResult)

    def test_shapes(self, dash_result):
        P = dash_result.P
        po = partial_order(dash_result)
        assert po.adjacency.shape == (P, P)
        assert po.confidence_matrix.shape == (P, P)

    def test_confidence_matrix_diagonal_is_zero(self, dash_result):
        po = partial_order(dash_result)
        np.testing.assert_allclose(np.diag(po.confidence_matrix), 0.0)

    def test_confidence_matrix_antisymmetric(self, dash_result):
        """π(i>j) + π(j>i) ≈ 1 (ties are rare with float importance)."""
        po = partial_order(dash_result)
        P = dash_result.P
        for i in range(P):
            for j in range(P):
                if i != j:
                    total = po.confidence_matrix[i, j] + po.confidence_matrix[j, i]
                    assert abs(total - 1.0) < 0.05, f"π({i}>{j}) + π({j}>{i}) = {total:.3f}, expected ≈ 1"

    def test_dominant_feature_ranked_first(self):
        """Known dominant feature should have π > 0.9 over all others."""
        result = _make_dominant_result()
        po = partial_order(result)
        # f0 (index 0) should beat f1, f2, f3 with near certainty
        assert np.all(po.confidence_matrix[0, 1:] > 0.9), f"dominant feature π values: {po.confidence_matrix[0, 1:]}"

    def test_transitivity(self):
        """If A > B and B > C, then A > C (check fraction method)."""
        result = _make_dominant_result()
        po = partial_order(result, alpha=0.05, method="fraction")
        adj = po.adjacency
        P = len(result.feature_names)
        for i in range(P):
            for j in range(P):
                for k in range(P):
                    if i != j != k and i != k:
                        if adj[i, j] and adj[j, k]:
                            # Transitivity may not hold perfectly with K models
                            # but should hold for very confident orderings
                            assert adj[i, k], f"Transitivity violated: {i}>{j}>{k} but not {i}>{k}"

    def test_n_determined_plus_undetermined(self, dash_result):
        po = partial_order(dash_result)
        P = dash_result.P
        n_pairs = P * (P - 1) // 2
        assert po.n_undetermined <= n_pairs

    def test_warns_when_k_small_with_bootstrap(self):
        rng = np.random.default_rng(0)
        small_result = DASHResult.from_shap_matrices(rng.standard_normal((5, 20, 3)))
        with pytest.warns(UserWarning, match="K=5 < 10"):
            partial_order(small_result, method="bootstrap")

    def test_feature_names_preserved(self, dash_result):
        po = partial_order(dash_result)
        assert po.feature_names == list(dash_result.feature_names)

    def test_summary_returns_string(self, dash_result):
        po = partial_order(dash_result)
        s = po.summary()
        assert isinstance(s, str)

    def test_confidence_matrix_range(self, dash_result):
        po = partial_order(dash_result)
        assert np.all(po.confidence_matrix >= 0.0)
        assert np.all(po.confidence_matrix <= 1.0)

    def test_bootstrap_differs_from_fraction(self):
        """Bootstrap CI must differ from raw fraction on borderline pairs.

        Construct π(f0>f1) = 0.80 (above 0.5, below 1-alpha=0.95).
        - fraction: 0.80 < 0.95 → NOT adjacent
        - bootstrap: lower 5th-percentile CI ≈ 0.59 > 0.5 → IS adjacent

        The two methods must produce different adjacency matrices.
        """
        rng = np.random.default_rng(0)
        K, n_ref, P = 10, 40, 3

        # Build SHAP tensor with controlled per-model importance ordering
        m = np.zeros((K, n_ref, P))

        # Models 0-7: f0 (imp≈2.0) > f1 (imp≈1.0) > f2 (imp≈0.5)
        for k in range(8):
            m[k, :, 0] = 2.0 + rng.standard_normal(n_ref) * 0.001
            m[k, :, 1] = 1.0 + rng.standard_normal(n_ref) * 0.001
            m[k, :, 2] = 0.5 + rng.standard_normal(n_ref) * 0.001

        # Models 8-9: f1 (imp≈2.0) > f0 (imp≈1.0) > f2 (imp≈0.5)
        for k in range(8, 10):
            m[k, :, 0] = 1.0 + rng.standard_normal(n_ref) * 0.001
            m[k, :, 1] = 2.0 + rng.standard_normal(n_ref) * 0.001
            m[k, :, 2] = 0.5 + rng.standard_normal(n_ref) * 0.001

        # π(f0>f1) = 8/10 = 0.80: above 0.5 but below 0.95
        result = DASHResult.from_shap_matrices(m, feature_names=["f0", "f1", "f2"])

        po_frac = partial_order(result, alpha=0.05, method="fraction")
        po_boot = partial_order(result, alpha=0.05, method="bootstrap", n_boot=1000, seed=42)

        # Sanity: verify π(f0>f1) ≈ 0.80
        assert abs(po_frac.confidence_matrix[0, 1] - 0.80) < 0.05

        # fraction: 0.80 < 0.95 → not adjacent
        assert not po_frac.adjacency[0, 1], "fraction should NOT include π=0.80 at alpha=0.05"

        # bootstrap lower CI ≈ 0.59 > 0.5 → adjacent
        assert po_boot.adjacency[0, 1], "bootstrap lower CI for π=0.80 should be > 0.5, making (f0,f1) adjacent"

        # The matrices must differ
        assert not np.array_equal(po_boot.adjacency, po_frac.adjacency), (
            "bootstrap and fraction produced identical adjacency — bootstrap CI is not being applied correctly"
        )

    def test_transitivity_enforced_flag_stored(self, dash_result):
        """transitivity_enforced field should reflect the parameter passed."""
        po_on = partial_order(dash_result, enforce_transitivity=True)
        po_off = partial_order(dash_result, enforce_transitivity=False)
        assert po_on.transitivity_enforced is True
        assert po_off.transitivity_enforced is False


    def test_transitivity_closure_boolean_matrix(self):
        """Floyd-Warshall closes A→B, B→C chain into A→C."""
        rng = np.random.default_rng(0)
        K, n_ref, P = 15, 30, 4
        # Use _make_dominant_result style with 4 features; closure is tested on boolean logic
        m = rng.standard_normal((K, n_ref, P)) * 0.1
        m[:, :, 0] += 10.0  # f0 dominates f1, f2, f3
        result = DASHResult.from_shap_matrices(m, feature_names=["A", "B", "C", "D"])

        # With enforce_transitivity=True all reachable edges must be closed
        po = partial_order(result, alpha=0.05, method="fraction", enforce_transitivity=True)
        adj = po.adjacency
        P_actual = adj.shape[0]

        # Verify closure: no pair (i, k) should exist where adj[i,j] and adj[j,k] but not adj[i,k]
        for i in range(P_actual):
            for j in range(P_actual):
                for k in range(P_actual):
                    if adj[i, j] and adj[j, k]:
                        assert adj[i, k], (
                            f"Transitivity violated: adj[{i},{j}]=True, adj[{j},{k}]=True, "
                            f"but adj[{i},{k}]=False after enforce_transitivity=True"
                        )

    def test_transitivity_closure_with_condorcet_setup(self):
        """Construct a case where π(A>C) < π(A>B) and π(B>C) using mixed orderings.

        Use alpha=0.10 (threshold=0.90) and K=20:
          Group 1 (19 models): A > B > C  → each contributes to A>B, B>C, A>C
          Group 2 (1 model):   C > A > B  → contributes to A>B only (rank_A=2<rank_B=3), NOT B>C, NOT A>C

        π(A>B) = 20/20 = 1.00 > 0.90 ✓
        π(B>C) = 19/20 = 0.95 > 0.90 ✓
        π(A>C) = 19/20 = 0.95 > 0.90 ✓  ← still True, closure irrelevant

        Need π(A>C) ≤ 0.90. Must reduce A>C agreements further.

        Use K=10, alpha=0.15 (threshold=0.85):
          Group 1 (9 models): A > B > C  → A>B ✓, B>C ✓, A>C ✓ per model
          Group 2 (1 model):  B > C > A  → A>B ✗, B>C ✓, A>C ✗ per model
                                            (rank_B=1, rank_C=2, rank_A=3 → rank_A>rank_C so NOT A>C)

        π(A>B) = 9/10 = 0.90 > 0.85 ✓
        π(B>C) = 10/10 = 1.00 > 0.85 ✓
        π(A>C) = 9/10 = 0.90 > 0.85 ✓  ← still True!

        The problem: if A>B>C in group 1, then A>C always holds in those models too.
        To get π(A>C) below threshold while π(A>B) and π(B>C) stay above threshold,
        we need models where A>B but C>A, AND models where B>C but C>A.
        Only ordering satisfying A>B AND C>A is: C>A>B (rank_C=1, rank_A=2, rank_B=3).
        In C>A>B: B>C? rank_B=3 > rank_C=1 → NOT B>C.
        So any model contributing to "not A>C" also hurts B>C.

        This means: with 3 features and total orderings, π(A>C) ≥ min(π(A>B), π(B>C)).
        Intransitivity at aggregate level with 3 features requires cycles WITHIN the ordering
        (Condorcet paradox) which requires at least 3 groups.

        With K=3*m and groups: A>B>C, B>C>A, C>A>B (m each):
          π(A>B) = 2m/(3m) = 2/3
          π(B>C) = 2m/(3m) = 2/3
          π(A>C) = 2m/(3m) = 2/3  ← all equal, Condorcet is symmetric

        With asymmetric groups: a models A>B>C, b models B>C>A, c models C>A>B:
          π(A>B) = (a + c) / K    [A>B in group1; C>A>B so rank_A=2<rank_B=3 ✓ in group3]
          π(B>C) = (a + b) / K    [B>C in groups 1,2]
          π(A>C) = (a + c) / K    [A>C in group1; C>A>B rank_A=2<rank_C... wait]

        Let me be careful with C>A>B:
          rank_C=1, rank_A=2, rank_B=3
          A>B: rank_A(2) < rank_B(3) ✓
          B>C: rank_B(3) < rank_C(1)? NO, 3>1 ✗
          A>C: rank_A(2) < rank_C(1)? NO, 2>1 ✗

        So C>A>B: A>B ✓, B>C ✗, A>C ✗

        With groups A>B>C (a), B>C>A (b), C>A>B (c), K=a+b+c:
          π(A>B) = (a + c) / K
          π(B>C) = (a + b) / K
          π(A>C) = a / K

        For transitivity violation: π(A>B)>T and π(B>C)>T and π(A>C)≤T:
          a/K ≤ T < (a+b)/K and a/K ≤ T < (a+c)/K
          → a ≤ T*K < a+b and a ≤ T*K < a+c

        Use K=20, T=0.85, a=16, b=3, c=2:
          π(A>B) = (16+2)/20 = 18/20 = 0.90 > 0.85 ✓
          π(B>C) = (16+3)/20 = 19/20 = 0.95 > 0.85 ✓
          π(A>C) = 16/20 = 0.80 ≤ 0.85 ✓ (False)
        """
        # Construction: K=20 with 3 groups (Condorcet-style asymmetric)
        # Group 1 (16 models): A>B>C  → A>B ✓, B>C ✓, A>C ✓
        # Group 2 (2 models):  B>C>A  → A>B ✗, B>C ✓, A>C ✗
        # Group 3 (2 models):  C>A>B  → A>B ✓, B>C ✗, A>C ✗
        # (C>A>B: rank_C=1, rank_A=2, rank_B=3; A>B: rank_A(2)<rank_B(3) ✓)
        #
        # π(A>B) = (16+0+2)/20 = 18/20 = 0.90 > 0.85 ✓
        # π(B>C) = (16+2+0)/20 = 18/20 = 0.90 > 0.85 ✓
        # π(A>C) = (16+0+0)/20 = 16/20 = 0.80 ≤ 0.85 ✓ (False)
        K = 20
        n_ref = 1
        P = 3
        m = np.zeros((K, n_ref, P))

        # Group 1 (16 models): A(3.0) > B(2.0) > C(1.0)
        for k in range(16):
            m[k, 0, 0] = 3.0  # A
            m[k, 0, 1] = 2.0  # B
            m[k, 0, 2] = 1.0  # C

        # Group 2 (2 models): B(3.0) > C(2.0) > A(1.0)
        for k in range(16, 18):
            m[k, 0, 0] = 1.0  # A last
            m[k, 0, 1] = 3.0  # B first
            m[k, 0, 2] = 2.0  # C second

        # Group 3 (2 models): C(3.0) > A(2.0) > B(1.0)
        for k in range(18, 20):
            m[k, 0, 0] = 2.0  # A second
            m[k, 0, 1] = 1.0  # B last
            m[k, 0, 2] = 3.0  # C first

        result = DASHResult.from_shap_matrices(m, feature_names=["A", "B", "C"])

        # Verify the confidence values match our construction (use alpha=0.5 to get all edges)
        po_check = partial_order(result, alpha=0.5, method="fraction", enforce_transitivity=False)
        assert abs(po_check.confidence_matrix[0, 1] - 18 / 20) < 0.01, f"π(A>B)={po_check.confidence_matrix[0,1]}"
        assert abs(po_check.confidence_matrix[1, 2] - 18 / 20) < 0.01, f"π(B>C)={po_check.confidence_matrix[1,2]}"
        assert abs(po_check.confidence_matrix[0, 2] - 16 / 20) < 0.01, f"π(A>C)={po_check.confidence_matrix[0,2]}"

        # Use alpha=0.15 → threshold = 1 - 0.15 = 0.85
        po_no_tc = partial_order(result, alpha=0.15, method="fraction", enforce_transitivity=False)
        po_tc = partial_order(result, alpha=0.15, method="fraction", enforce_transitivity=True)

        # Confirm base adjacency: A>B True, B>C True, A>C False
        assert po_no_tc.adjacency[0, 1], f"A>B should be True: π={po_no_tc.confidence_matrix[0,1]:.3f} > 0.85"
        assert po_no_tc.adjacency[1, 2], f"B>C should be True: π={po_no_tc.confidence_matrix[1,2]:.3f} > 0.85"
        assert not po_no_tc.adjacency[0, 2], f"A>C should be False: π={po_no_tc.confidence_matrix[0,2]:.3f} ≤ 0.85"

        # Confirm closure promotes A>C
        assert po_tc.adjacency[0, 2], "Transitivity closure must set A>C=True when A>B and B>C are True"

        # n_determined should be strictly higher with closure (one more edge: A>C)
        assert po_tc.n_determined > po_no_tc.n_determined

        # transitivity_enforced flag
        assert po_tc.transitivity_enforced is True
        assert po_no_tc.transitivity_enforced is False
