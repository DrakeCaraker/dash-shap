"""Stage 2: Performance Filtering."""
import numpy as np
from typing import Dict, List

__all__ = ["performance_filter"]


def performance_filter(val_scores, epsilon=0.08, higher_is_better=True,
                       mode='absolute', verbose=True):
    """Filter models within epsilon of the best validation score.

    Parameters
    ----------
    val_scores : dict
        Mapping of model index to validation score.
    epsilon : float
        Tolerance for filtering.  Interpretation depends on ``mode``.
    higher_is_better : bool
        Whether higher validation scores are better.
    mode : str
        ``'absolute'`` — keep models within ``epsilon`` of the best score
        (original behaviour, backward-compatible).
        ``'relative'`` — keep models within ``epsilon * |best_score|`` of the
        best score.  Scale-invariant across datasets.
        ``'quantile'`` — keep the top ``epsilon`` fraction of models (e.g.
        epsilon=0.5 keeps the top 50%).
    verbose : bool
    """
    best_score = (
        max(val_scores.values()) if higher_is_better
        else min(val_scores.values())
    )

    if mode == 'relative':
        threshold = abs(best_score) * epsilon
        filtered = [
            i for i, s in val_scores.items()
            if abs(s - best_score) <= threshold
        ]
    elif mode == 'quantile':
        scores_sorted = sorted(val_scores.values(), reverse=higher_is_better)
        cutoff_idx = max(2, int(len(scores_sorted) * epsilon))
        cutoff_score = scores_sorted[min(cutoff_idx - 1, len(scores_sorted) - 1)]
        if higher_is_better:
            filtered = [i for i, s in val_scores.items() if s >= cutoff_score]
        else:
            filtered = [i for i, s in val_scores.items() if s <= cutoff_score]
    else:  # absolute (default)
        filtered = [
            i for i, s in val_scores.items()
            if abs(s - best_score) <= epsilon
        ]

    if verbose:
        print(
            f"Performance filter ({mode}): {len(filtered)}/{len(val_scores)} "
            f"models within epsilon={epsilon} of best ({best_score:.4f})"
        )
    return filtered
