"""Extension: Pareto Model Selection — find optimal DASH configurations.

Tracks (prediction quality, explanation stability) pairs across different
DASH configurations and identifies the Pareto frontier — configurations
where you cannot improve one metric without sacrificing the other.

Usage:
    selector = ParetoSelector()
    for config in configs:
        pipe = DASHPipeline(**config)
        pipe.fit(X_train, y_train, X_val, y_val, X_ref=X_ref)
        selector.evaluate(config, pipe.result_, X_test, y_test)
    frontier = selector.frontier()
    print(frontier.summary())

Warning: Using the same X_test for both RMSE evaluation and configuration
selection can introduce selection bias. Use a held-out set or cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dash_shap.core.result import DASHResult

__all__ = ["ParetoSelector", "ParetoFrontier"]


@dataclass
class _EvalPoint:
    config: dict
    rmse: float
    stability: float
    K: int


@dataclass
class ParetoFrontier:
    """Result of ParetoSelector.frontier().

    Attributes
    ----------
    configs : list of dict
        Configurations on the Pareto frontier.
    rmse : list of float
        RMSE for each frontier configuration.
    stability : list of float
        Mean pairwise Spearman stability for each frontier configuration.
    all_configs : list of dict
        All evaluated configurations (including dominated ones).
    all_rmse : list of float
    all_stability : list of float
    """

    configs: list
    rmse: list
    stability: list
    all_configs: list
    all_rmse: list
    all_stability: list

    def summary(self) -> str:
        lines = [
            f"ParetoFrontier: {len(self.configs)} optimal configs (out of {len(self.all_configs)} evaluated)",
            "",
            f"{'Config':<40} {'RMSE':>8} {'Stability':>10}",
            "-" * 60,
        ]
        for cfg, rmse, stab in zip(self.configs, self.rmse, self.stability):
            cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
            if len(cfg_str) > 37:
                cfg_str = cfg_str[:34] + "..."
            lines.append(f"{cfg_str:<40} {rmse:>8.4f} {stab:>10.4f}")
        return "\n".join(lines)

    def plot(self):
        """Scatter plot of all configs with Pareto frontier highlighted."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # All points
        ax.scatter(self.all_rmse, self.all_stability, c="tab:gray", alpha=0.4, s=40, label="Dominated", zorder=1)

        # Frontier points
        ax.scatter(
            self.rmse,
            self.stability,
            c="tab:red",
            s=80,
            label="Pareto frontier",
            zorder=2,
            edgecolors="black",
            linewidths=0.5,
        )

        # Connect frontier points
        if len(self.rmse) > 1:
            order = np.argsort(self.rmse)
            ax.plot(
                [self.rmse[i] for i in order],
                [self.stability[i] for i in order],
                "r--",
                alpha=0.5,
                zorder=1,
            )

        ax.set_xlabel("RMSE (lower is better)")
        ax.set_ylabel("Stability (higher is better)")
        ax.set_title("Pareto Frontier: Prediction vs Explanation Quality")
        ax.legend()
        fig.tight_layout()
        return fig


class ParetoSelector:
    """Track DASH configurations and find the Pareto-optimal set.

    Warning: Using the same test set for both RMSE evaluation and config
    selection introduces selection bias. Use a held-out validation set
    or cross-validation for rigorous model selection.
    """

    def __init__(self):
        self._points: list[_EvalPoint] = []

    def evaluate(
        self,
        config: dict,
        result: "DASHResult",
        X_test: np.ndarray,
        y_test: np.ndarray,
        predict_fn=None,
    ) -> None:
        """Record a configuration's performance.

        Parameters
        ----------
        config : dict
            Pipeline configuration (e.g., {"M": 100, "K": 20, "epsilon": 0.05}).
        result : DASHResult
        X_test : ndarray
            Test features for RMSE computation.
        y_test : ndarray
            Test targets.
        predict_fn : callable or None
            Function that takes X_test and returns predictions. If None,
            uses mean of result.consensus as a proxy score (stability-only mode).
        """
        from dash_shap.extensions._base import per_model_importance

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        # RMSE
        if predict_fn is not None:
            preds = predict_fn(X_test)
            rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        else:
            rmse = float("nan")

        # Stability: mean pairwise Spearman of per-model importance vectors
        importance = per_model_importance(result)  # (K, P)
        from scipy.stats import spearmanr

        K = importance.shape[0]
        correlations = []
        for i in range(K):
            for j in range(i + 1, K):
                r, _ = spearmanr(importance[i], importance[j])
                correlations.append(r)
        stability = float(np.mean(correlations)) if correlations else 1.0

        self._points.append(
            _EvalPoint(
                config=dict(config),
                rmse=rmse,
                stability=stability,
                K=result.K,
            )
        )

    def frontier(self) -> ParetoFrontier:
        """Compute the Pareto frontier.

        A configuration is Pareto-optimal if no other configuration has
        both lower RMSE and higher stability.

        Returns
        -------
        ParetoFrontier
        """
        if not self._points:
            raise ValueError("No configurations evaluated yet. Call .evaluate() first.")

        all_configs = [p.config for p in self._points]
        all_rmse = [p.rmse for p in self._points]
        all_stability = [p.stability for p in self._points]

        # Filter out NaN RMSE for Pareto computation
        valid = [(i, r, s) for i, (r, s) in enumerate(zip(all_rmse, all_stability)) if np.isfinite(r)]

        if not valid:
            # Stability-only mode: all configs are "Pareto optimal"
            return ParetoFrontier(
                configs=all_configs,
                rmse=all_rmse,
                stability=all_stability,
                all_configs=all_configs,
                all_rmse=all_rmse,
                all_stability=all_stability,
            )

        frontier_idx = []
        for i, ri, si in valid:
            dominated = False
            for j, rj, sj in valid:
                if i != j and rj <= ri and sj >= si and (rj < ri or sj > si):
                    dominated = True
                    break
            if not dominated:
                frontier_idx.append(i)

        return ParetoFrontier(
            configs=[all_configs[i] for i in frontier_idx],
            rmse=[all_rmse[i] for i in frontier_idx],
            stability=[all_stability[i] for i in frontier_idx],
            all_configs=all_configs,
            all_rmse=all_rmse,
            all_stability=all_stability,
        )
