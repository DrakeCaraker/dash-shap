"""Synthetic Data Generation — Linear and Nonlinear DGPs."""
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

__all__ = [
    "make_correlation_matrix",
    "generate_synthetic_linear",
    "generate_synthetic_nonlinear",
]


def make_correlation_matrix(P=50, group_size=5, rho=0.9, structure="block"):
    """Build a block or overlapping correlation matrix."""
    if rho == 0.0:
        return np.eye(P)

    n_groups = P // group_size
    Sigma = np.eye(P)

    if structure == "block":
        for g in range(n_groups):
            s, e = g * group_size, (g + 1) * group_size
            for i in range(s, e):
                for j in range(s, e):
                    if i != j:
                        Sigma[i, j] = rho

    elif structure == "overlapping":
        overlap = 2
        for g in range(n_groups):
            s = g * group_size
            e = min(s + group_size + overlap, P)
            for i in range(s, e):
                for j in range(s, e):
                    if i != j:
                        core_end = s + group_size
                        if i >= core_end or j >= core_end:
                            Sigma[i, j] = max(Sigma[i, j], rho * 0.7)
                        else:
                            Sigma[i, j] = max(Sigma[i, j], rho)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(Sigma))
        Sigma = Sigma / np.outer(d, d)

    return Sigma


def generate_synthetic_linear(
    N=5000,
    P=50,
    group_size=5,
    rho=0.9,
    sigma_noise=0.5,
    seed=42,
    test_size=0.15,
    val_size=0.15,
    structure="block",
):
    """Generate synthetic linear DGP with correlated feature groups."""
    rng = np.random.RandomState(seed)
    n_groups = P // group_size

    beta_groups = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0])
    if n_groups != len(beta_groups):
        beta_groups = np.linspace(2.0, 0.0, n_groups)

    Sigma = make_correlation_matrix(P, group_size, rho, structure=structure)
    X = rng.multivariate_normal(np.zeros(P), Sigma, size=N)

    group_means = np.zeros((N, n_groups))
    groups = np.zeros(P, dtype=int)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        group_means[:, g] = X[:, s:e].mean(axis=1)
        groups[s:e] = g

    y = group_means @ beta_groups + rng.normal(0, sigma_noise, N)

    true_importance = np.zeros(P)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        true_importance[s:e] = np.abs(beta_groups[g]) / group_size

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size / (1 - test_size), random_state=seed,
    )

    meta = {
        "dgp": "linear", "N": N, "P": P, "group_size": group_size,
        "n_groups": n_groups, "rho": rho, "sigma_noise": sigma_noise,
        "beta_groups": beta_groups, "seed": seed, "structure": structure,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, groups, true_importance, meta


def generate_synthetic_nonlinear(
    N=5000,
    P=50,
    group_size=5,
    rho=0.9,
    sigma_noise=0.5,
    seed=42,
    test_size=0.15,
    val_size=0.15,
    structure="block",
):
    """Generate synthetic nonlinear DGP with interactions and correlated features."""
    rng = np.random.RandomState(seed)
    n_groups = P // group_size

    beta_1, beta_2, beta_3 = 1.0, 0.8, 1.2
    # Hardcoded seed=42 ensures identical ground-truth coefficients across
    # reps.  This is intentional: accuracy is NOT reported for nonlinear DGP,
    # so identical ground truth avoids confounding the stability/equity analysis.
    beta_4_to_G = np.random.RandomState(42).uniform(0.3, 1.0, max(n_groups - 3, 0))

    Sigma = make_correlation_matrix(P, group_size, rho, structure=structure)
    X = rng.multivariate_normal(np.zeros(P), Sigma, size=N)

    group_means = np.zeros((N, n_groups))
    groups = np.zeros(P, dtype=int)
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        group_means[:, g] = X[:, s:e].mean(axis=1)
        groups[s:e] = g

    z1, z2, z3 = group_means[:, 0], group_means[:, 1], group_means[:, 2]
    y = beta_1 * z1**2 + beta_2 * z1 * z2 + beta_3 * np.sin(np.pi * z3)
    for g_idx, bg in enumerate(beta_4_to_G):
        y += bg * group_means[:, g_idx + 3]
    y += rng.normal(0, sigma_noise, N)

    true_importance = np.zeros(P)
    approx = np.zeros(n_groups)
    approx[0], approx[1], approx[2] = 1.5, 0.8, 1.2
    for g_idx, bg in enumerate(beta_4_to_G):
        approx[g_idx + 3] = bg
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        true_importance[s:e] = approx[g] / group_size

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size / (1 - test_size), random_state=seed,
    )

    meta = {
        "dgp": "nonlinear", "N": N, "P": P, "group_size": group_size,
        "n_groups": n_groups, "rho": rho, "sigma_noise": sigma_noise,
        "seed": seed, "structure": structure,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, groups, true_importance, meta
