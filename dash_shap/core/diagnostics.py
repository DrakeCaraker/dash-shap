"""Stage 5: Stability Diagnostics — FSI, IS Plot, disagreement maps."""

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "compute_diagnostics",
    "FeatureStabilityIndex",
    "ImportanceStabilityPlot",
    "local_disagreement_map",
]


def compute_diagnostics(all_shap_matrices, epsilon=1e-8):
    """Compute consensus, variance, FSI, and global importance from SHAP matrices.

    Parameters
    ----------
    all_shap_matrices : np.ndarray
        Shape (K, N', P) — SHAP values from K models.
    epsilon : float
        Small constant to prevent division by zero in FSI.

    Returns
    -------
    tuple of (consensus, variance_matrix, fsi, global_importance)
    """
    consensus = np.mean(all_shap_matrices, axis=0)
    variance_matrix = np.var(all_shap_matrices, axis=0, ddof=1)
    std_matrix = np.sqrt(variance_matrix)
    global_importance = np.mean(np.abs(consensus), axis=0)
    mean_std = np.mean(std_matrix, axis=0)
    mean_abs_consensus = np.mean(np.abs(consensus), axis=0)
    fsi = mean_std / (mean_abs_consensus + epsilon)
    return consensus, variance_matrix, fsi, global_importance


class FeatureStabilityIndex:
    def __init__(self, fsi, global_importance, feature_names=None):
        self.fsi = fsi
        self.global_importance = global_importance
        self.P = len(fsi)
        self.feature_names = feature_names or [f"f{i}" for i in range(self.P)]

    def get_quadrant_labels(self, importance_threshold=None, fsi_threshold=None):
        if importance_threshold is None:
            importance_threshold = np.median(self.global_importance)
        if fsi_threshold is None:
            high_imp_mask = self.global_importance >= importance_threshold
            fsi_threshold = np.median(self.fsi[high_imp_mask]) if high_imp_mask.any() else np.median(self.fsi)

        labels = np.empty(self.P, dtype=object)
        for j in range(self.P):
            hi = self.global_importance[j] >= importance_threshold
            hf = self.fsi[j] >= fsi_threshold
            if hi and not hf:
                labels[j] = "I: Robust Drivers"
            elif hi and hf:
                labels[j] = "II: Collinear Cluster"
            elif not hi and not hf:
                labels[j] = "III: Confirmed Unimportant"
            else:
                labels[j] = "IV: Fragile Interactions"
        return labels

    def summary(self, top_k=10):
        order = np.argsort(self.global_importance)[::-1]
        lines = [
            "Feature Stability Summary",
            "=" * 40,
            f"{'Feature':<20} {'Importance':>12} {'FSI':>8}",
            "-" * 40,
        ]
        for j in order[:top_k]:
            lines.append(f"{self.feature_names[j]:<20} {self.global_importance[j]:>12.4f} {self.fsi[j]:>8.3f}")
        return "\n".join(lines)


class ImportanceStabilityPlot:
    @staticmethod
    def plot(
        global_importance,
        fsi,
        feature_names=None,
        groups=None,
        importance_threshold=None,
        fsi_threshold=None,
        title="Importance-Stability Plot",
        figsize=(10, 7),
        annotate_top_k=5,
        ax=None,
    ):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(fsi))]
        if importance_threshold is None:
            importance_threshold = np.median(global_importance)
        if fsi_threshold is None:
            high_mask = global_importance >= importance_threshold
            fsi_threshold = np.median(fsi[high_mask]) if high_mask.any() else np.median(fsi)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if groups is not None:
            unique_groups = np.unique(groups)
            cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_groups))
            for i, g in enumerate(unique_groups):
                mask = groups == g
                ax.scatter(
                    global_importance[mask],
                    fsi[mask],
                    c=[cmap(i)],
                    label=f"Group {g}",
                    s=60,
                    alpha=0.7,
                    edgecolors="k",
                    linewidths=0.5,
                )
        else:
            fsi_obj = FeatureStabilityIndex(
                fsi,
                global_importance,
                feature_names,
            )
            labels = fsi_obj.get_quadrant_labels(
                importance_threshold,
                fsi_threshold,
            )
            colors_map = {
                "I: Robust Drivers": "#2ecc71",
                "II: Collinear Cluster": "#e74c3c",
                "III: Confirmed Unimportant": "#95a5a6",
                "IV: Fragile Interactions": "#f39c12",
            }
            for label, color in colors_map.items():
                mask = labels == label
                if mask.any():
                    ax.scatter(
                        global_importance[mask],
                        fsi[mask],
                        c=color,
                        label=label,
                        s=60,
                        alpha=0.7,
                        edgecolors="k",
                        linewidths=0.5,
                    )

        ax.axvline(importance_threshold, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(fsi_threshold, color="gray", linestyle="--", alpha=0.5)

        # Quadrant background shading and corner labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_betweenx(
            [ylim[0], fsi_threshold],
            xlim[0],
            importance_threshold,
            alpha=0.04,
            color="#95a5a6",
            zorder=0,
        )  # Q-III gray
        ax.fill_betweenx(
            [fsi_threshold, ylim[1]],
            xlim[0],
            importance_threshold,
            alpha=0.04,
            color="#f39c12",
            zorder=0,
        )  # Q-IV orange
        ax.fill_betweenx(
            [ylim[0], fsi_threshold],
            importance_threshold,
            xlim[1],
            alpha=0.04,
            color="#2ecc71",
            zorder=0,
        )  # Q-I green
        ax.fill_betweenx(
            [fsi_threshold, ylim[1]],
            importance_threshold,
            xlim[1],
            alpha=0.04,
            color="#e74c3c",
            zorder=0,
        )  # Q-II red
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xp = 0.015 * (xlim[1] - xlim[0])
        yp = 0.025 * (ylim[1] - ylim[0])
        ax.text(
            importance_threshold + xp,
            ylim[0] + yp,
            "Q-I: Robust",
            fontsize=7,
            color="#1a8a47",
            va="bottom",
            alpha=0.8,
        )
        ax.text(
            importance_threshold + xp,
            fsi_threshold + yp,
            "Q-II: Collinear",
            fontsize=7,
            color="#a93226",
            va="bottom",
            alpha=0.8,
        )
        ax.text(
            xlim[0] + xp,
            ylim[0] + yp,
            "Q-III: Unimportant",
            fontsize=7,
            color="#5d6d7e",
            va="bottom",
            alpha=0.8,
        )
        ax.text(
            xlim[0] + xp,
            fsi_threshold + yp,
            "Q-IV: Fragile",
            fontsize=7,
            color="#b7770d",
            va="bottom",
            alpha=0.8,
        )

        top_k_idx = np.argsort(global_importance)[-annotate_top_k:][::-1]
        for j in top_k_idx:
            ax.annotate(
                feature_names[j],
                (global_importance[j], fsi[j]),
                fontsize=8,
                alpha=0.85,
                xytext=(8, 8),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.7),
            )

        ax.set_xlabel("Consensus Importance (mean |SHAP|)", fontsize=12)
        ax.set_ylabel("Feature Stability Index (FSI)", fontsize=12)
        ax.set_title(title.replace("\u2014", "-").replace("\u2013", "-"), fontsize=14)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        return fig


def local_disagreement_map(
    all_shap_matrices,
    observation_idx,
    feature_names=None,
    top_k=15,
    figsize=(10, 6),
    title=None,
):
    """Plot a local disagreement map for a single observation."""
    K, N_prime, P = all_shap_matrices.shape
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(P)]

    consensus_i = np.mean(all_shap_matrices[:, observation_idx, :], axis=0)
    std_i = np.std(all_shap_matrices[:, observation_idx, :], axis=0, ddof=1)
    order = np.argsort(np.abs(consensus_i))[::-1][:top_k]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(order))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in consensus_i[order]]

    ax.barh(
        y_pos,
        consensus_i[order],
        xerr=std_i[order],
        color=colors,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[j] for j in order], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("SHAP Value (consensus +/- 1 std)", fontsize=11)
    if title is None:
        title = f"Local Disagreement Map - Observation {observation_idx}"
    ax.set_title(title.replace("\u2014", "-").replace("\u2013", "-"), fontsize=13)
    fig.tight_layout()
    return fig
