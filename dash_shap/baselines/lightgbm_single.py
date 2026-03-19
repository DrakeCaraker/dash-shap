"""Baseline: LightGBM Single Best — tests whether leaf-wise GBM exhibits first-mover bias."""
import numpy as np

from dash_shap.utils.shap_helpers import compute_global_importance

__all__ = ["LightGBMSingleBestBaseline"]


class LightGBMSingleBestBaseline:
    """Train a single LightGBM model and compute interventional TreeSHAP importance.

    LightGBM uses leaf-wise (best-first) tree growth rather than XGBoost's
    level-wise approach.  If first-mover bias is a general property of
    sequential gradient boosting (not specific to the splitting strategy),
    LightGBM should exhibit similar instability.
    """

    def __init__(self, n_estimators=500, task="regression", seed=42):
        self.n_estimators = n_estimators
        self.task = task
        self.seed = seed
        self.model_ = None
        self.global_importance_ = None
        self.fsi_ = None

    def fit(self, X_train, y_train, X_val, y_val, X_ref=None,
            background_size=100, seed=None):
        """Fit a LightGBM model and compute interventional TreeSHAP importance.

        Parameters
        ----------
        seed : int or None
            If provided, randomly samples SHAP background rows from X_ref.
        """
        import lightgbm as lgb
        import shap

        if X_ref is None:
            X_ref = X_val

        if self.task == "regression":
            self.model_ = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                verbosity=-1,
                n_jobs=-1,
            )
        else:
            self.model_ = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                verbosity=-1,
                n_jobs=-1,
            )

        self.model_.fit(X_train, y_train)

        n_bg = min(background_size, len(X_ref))
        if seed is not None:
            rng = np.random.RandomState(seed)
            bg_idx = rng.choice(len(X_ref), size=n_bg, replace=False)
            bg = X_ref[bg_idx]
        else:
            bg = X_ref[:n_bg]

        explainer = shap.TreeExplainer(
            self.model_, data=bg, feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_ref, check_additivity=False)
        self.global_importance_ = compute_global_importance(sv)
        # FSI is undefined for single-model baselines (no inter-model variation).
        self.fsi_ = np.full_like(self.global_importance_, np.nan)
        return self
