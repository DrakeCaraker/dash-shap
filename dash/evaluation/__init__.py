"""Evaluation metrics."""
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from typing import List, Tuple

def importance_accuracy(estimated, true):
    rho, _ = spearmanr(estimated, true)
    est_norm = estimated / (estimated.sum() + 1e-10)
    true_norm = true / (true.sum() + 1e-10)
    mse = np.mean((est_norm - true_norm) ** 2)
    return rho, mse

def importance_stability(vectors):
    n = len(vectors)
    if n < 2: return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(vectors[i], vectors[j])
            corrs.append(rho)
    return float(np.mean(corrs))

def within_group_equity(importance_vector, groups):
    cvs = []
    for g in np.unique(groups):
        gi = importance_vector[groups == g]
        cvs.append(gi.std() / gi.mean() if gi.mean() > 1e-10 else 0.0)
    return float(np.mean(cvs))

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-10 else 0.0

def compare_methods(a, b):
    if np.allclose(a, b): return 0.0, 1.0
    stat, pval = wilcoxon(a, b)
    return float(stat), float(pval)

def friedman_test(*method_scores):
    from scipy.stats import friedmanchisquare
    stat, pval = friedmanchisquare(*method_scores)
    return float(stat), float(pval)
