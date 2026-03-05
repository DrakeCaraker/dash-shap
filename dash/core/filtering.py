"""Stage 2: Performance Filtering."""
import numpy as np
from typing import Dict, List

def performance_filter(val_scores, epsilon=0.02, higher_is_better=True, verbose=True):
    best_score = max(val_scores.values()) if higher_is_better else min(val_scores.values())
    filtered = [i for i, s in val_scores.items() if abs(s - best_score) <= epsilon]
    if verbose:
        print(f"Performance filter: {len(filtered)}/{len(val_scores)} models within ε={epsilon} of best ({best_score:.4f})")
    return filtered
