"""Tests for dash.evaluation metrics."""
import numpy as np
from dash.evaluation import (
    importance_accuracy,
    importance_stability,
    within_group_equity,
    cohens_d,
    compare_methods,
)


def test_importance_accuracy_perfect():
    true = np.array([1.0, 0.5, 0.25, 0.1])
    rho, mse = importance_accuracy(true, true)
    assert rho > 0.99
    assert mse < 1e-10


def test_importance_accuracy_inverted():
    true = np.array([1.0, 0.5, 0.25, 0.1])
    inverted = true[::-1]
    rho, _ = importance_accuracy(inverted, true)
    assert rho < 0


def test_importance_stability_identical():
    vectors = [np.array([1.0, 0.5, 0.3])] * 3
    stab = importance_stability(vectors)
    assert stab > 0.99


def test_importance_stability_single():
    vectors = [np.array([1.0, 0.5, 0.3])]
    assert importance_stability(vectors) == 1.0


def test_within_group_equity_uniform():
    importance = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
    groups = np.array([0, 0, 0, 1, 1, 1])
    eq = within_group_equity(importance, groups)
    assert eq < 1e-10


def test_within_group_equity_nonuniform():
    importance = np.array([1.0, 0.1, 0.5, 0.3, 0.3, 0.3])
    groups = np.array([0, 0, 0, 1, 1, 1])
    eq = within_group_equity(importance, groups)
    assert eq > 0


def test_cohens_d_identical():
    g = np.array([1.0, 2.0, 3.0])
    assert cohens_d(g, g) == 0.0


def test_cohens_d_different():
    g1 = np.array([10.0, 10.5, 11.0])
    g2 = np.array([1.0, 1.5, 2.0])
    d = cohens_d(g1, g2)
    assert d > 0


def test_compare_methods_identical():
    a = np.array([1.0, 2.0, 3.0])
    stat, pval = compare_methods(a, a)
    assert pval == 1.0
