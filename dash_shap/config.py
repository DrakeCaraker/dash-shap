"""Canonical experiment configuration for the DASH paper.

Import this module in experiment runners and notebooks instead of redefining
PAPER_CONFIG locally. This is the single source of truth.

Usage:
    from dash_shap.config import PAPER_CONFIG, SEED, REAL_EPSILON, REAL_EPSILON_MODE
"""

PAPER_CONFIG: dict = {
    "M": 200,
    "K": 30,
    "N_REPS": 50,
    "EPSILON": 0.08,
    "DELTA": 0.05,
    "N_TRIALS_SB": 30,
    "T_PER_MODEL": 500,
    "N_ESTIMATORS_ESHAP": 2000,
    "TAU_CLUSTER": 0.3,
}

SEED: int = 42
REAL_EPSILON: float = 0.05
REAL_EPSILON_MODE: str = "relative"
