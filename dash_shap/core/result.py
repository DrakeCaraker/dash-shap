"""DASHResult — immutable container for DASH pipeline outputs.

Decouples downstream analysis from DASHPipeline. All extensions accept
a DASHResult as input, never the pipeline object itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DASHResult:
    """Immutable container for the K×N'×P SHAP tensor and derived quantities.

    Parameters
    ----------
    all_shap_matrices : np.ndarray
        Shape ``(K, n_ref, P)`` — SHAP values from K selected models.
    feature_names : list[str]
        Length-P list of feature labels.
    val_scores : np.ndarray | None
        Shape ``(K,)`` validation scores for the K selected models.

    Attributes (computed in ``__post_init__``)
    -------------------------------------------
    consensus : np.ndarray — ``(n_ref, P)``
    variance : np.ndarray — ``(n_ref, P)``
    global_importance : np.ndarray — ``(P,)``
    fsi : np.ndarray — ``(P,)``
    """

    all_shap_matrices: np.ndarray
    feature_names: list[str]
    val_scores: np.ndarray | None = None

    # Computed fields — not passed to __init__
    consensus: np.ndarray = field(init=False, repr=False)
    variance: np.ndarray = field(init=False, repr=False)
    global_importance: np.ndarray = field(init=False, repr=False)
    fsi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        m = self.all_shap_matrices
        if m.ndim != 3:
            raise ValueError(
                f"all_shap_matrices must be 3-D (K, n_ref, P), got shape {m.shape}"
            )
        if m.shape[0] < 2:
            raise ValueError(f"Need K >= 2 models, got {m.shape[0]}")
        if len(self.feature_names) != m.shape[2]:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != P={m.shape[2]}"
            )
        if self.val_scores is not None:
            self.val_scores = np.asarray(self.val_scores)
            if self.val_scores.shape != (m.shape[0],):
                raise ValueError(
                    f"val_scores shape {self.val_scores.shape} != (K={m.shape[0]},)"
                )

        eps = 1e-8
        self.consensus = np.mean(m, axis=0)  # (n_ref, P)
        self.variance = np.var(m, axis=0, ddof=1)  # (n_ref, P)
        self.global_importance = np.mean(np.abs(self.consensus), axis=0)  # (P,)
        mean_std = np.mean(np.sqrt(self.variance), axis=0)  # (P,)
        self.fsi = mean_std / (self.global_importance + eps)  # (P,)

        # Lock all arrays
        for arr_name in ("all_shap_matrices", "consensus", "variance",
                         "global_importance", "fsi"):
            getattr(self, arr_name).flags.writeable = False
        if self.val_scores is not None:
            self.val_scores.flags.writeable = False

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def K(self) -> int:
        return self.all_shap_matrices.shape[0]

    @property
    def n_ref(self) -> int:
        return self.all_shap_matrices.shape[1]

    @property
    def P(self) -> int:
        return self.all_shap_matrices.shape[2]

    @property
    def memory_bytes(self) -> int:
        total = self.all_shap_matrices.nbytes + self.consensus.nbytes
        total += self.variance.nbytes + self.global_importance.nbytes
        total += self.fsi.nbytes
        if self.val_scores is not None:
            total += self.val_scores.nbytes
        return total

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_shap_matrices(
        cls,
        matrices: np.ndarray,
        feature_names: list[str] | None = None,
        val_scores: np.ndarray | None = None,
    ) -> DASHResult:
        """Create a DASHResult with input validation.

        Parameters
        ----------
        matrices : np.ndarray
            Shape ``(K, n_ref, P)``.
        feature_names : list[str] | None
            If ``None``, auto-generates ``["f0", "f1", ...]``.
        val_scores : np.ndarray | None
            Shape ``(K,)``.
        """
        matrices = np.asarray(matrices, dtype=np.float64)
        if matrices.ndim != 3:
            raise ValueError(
                f"matrices must be 3-D (K, n_ref, P), got shape {matrices.shape}"
            )
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(matrices.shape[2])]
        if val_scores is not None:
            val_scores = np.asarray(val_scores, dtype=np.float64)
        return cls(
            all_shap_matrices=matrices,
            feature_names=feature_names,
            val_scores=val_scores,
        )

    # ── Serialization ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save to ``<path>.npz`` (arrays) + ``<path>.json`` (metadata).

        No pickle — safe for untrusted environments.
        """
        path = Path(path)
        arrays = {"all_shap_matrices": self.all_shap_matrices}
        if self.val_scores is not None:
            arrays["val_scores"] = self.val_scores
        np.savez_compressed(path.with_suffix(".npz"), **arrays)

        meta = {"feature_names": self.feature_names}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> DASHResult:
        """Reconstruct from ``<path>.npz`` + ``<path>.json``."""
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        val_scores = data["val_scores"] if "val_scores" in data else None
        return cls(
            all_shap_matrices=data["all_shap_matrices"],
            feature_names=meta["feature_names"],
            val_scores=val_scores,
        )
