"""DASHResult — decoupled container for DASH outputs.

All extensions accept a DASHResult; never the pipeline object itself.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["DASHResult", "VersionError"]

_FORMAT_VERSION = 1


class VersionError(ValueError):
    """Raised when a serialized DASHResult uses an unsupported format version."""


@dataclass
class DASHResult:
    """Container for the K×N'×P SHAP tensor produced by DASHPipeline.

    Parameters
    ----------
    all_shap_matrices : ndarray of shape (K, n_ref, P)
        The core tensor — SHAP values from each of the K selected models.
    feature_names : list[str]
        Length P. Auto-generated as ["f0", "f1", ...] if omitted.
    val_scores : list | ndarray | None
        Validation scores for the K selected models (length K). Accepts list;
        __post_init__ converts to ndarray.

    Computed Attributes (read-only after construction)
    ---------------------------------------------------
    consensus : ndarray (n_ref, P)
    variance  : ndarray (n_ref, P)
    global_importance : ndarray (P,)
    fsi       : ndarray (P,)
    """

    all_shap_matrices: np.ndarray
    feature_names: list
    val_scores: Optional[np.ndarray] = None

    # Computed in __post_init__; not passed to __init__
    consensus: np.ndarray = field(init=False, repr=False)
    variance: np.ndarray = field(init=False, repr=False)
    global_importance: np.ndarray = field(init=False, repr=False)
    fsi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        m = self.all_shap_matrices
        if m.ndim != 3:
            raise ValueError(f"all_shap_matrices must be 3D (K, n_ref, P), got {m.ndim}D")
        K, n_ref, P = m.shape
        if K < 2:
            raise ValueError(f"K must be >= 2, got K={K}")
        if len(self.feature_names) != P:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != P={P}"
            )

        # Convert list val_scores to ndarray
        if self.val_scores is not None:
            vs = np.asarray(self.val_scores, dtype=float)
            if vs.shape != (K,):
                raise ValueError(f"val_scores must have length K={K}, got {vs.shape}")
            object.__setattr__(self, "val_scores", vs)

        # Compute derived arrays (epsilon = 1e-8, matching diagnostics.py)
        eps = 1e-8
        consensus = np.mean(m, axis=0)                    # (n_ref, P)
        variance = np.var(m, axis=0, ddof=1)              # (n_ref, P)
        global_importance = np.mean(np.abs(consensus), axis=0)  # (P,)
        mean_std = np.mean(np.sqrt(variance), axis=0)     # (P,)
        fsi = mean_std / (global_importance + eps)        # (P,)

        object.__setattr__(self, "consensus", consensus)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "global_importance", global_importance)
        object.__setattr__(self, "fsi", fsi)

        # Lock all arrays read-only
        for arr in (m, consensus, variance, global_importance, fsi):
            arr.flags.writeable = False
        if self.val_scores is not None:
            self.val_scores.flags.writeable = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        total = self.all_shap_matrices.nbytes
        for arr in (self.consensus, self.variance, self.global_importance, self.fsi):
            total += arr.nbytes
        if self.val_scores is not None:
            total += self.val_scores.nbytes
        return total

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_shap_matrices(
        cls,
        matrices: np.ndarray,
        feature_names: Optional[list] = None,
        val_scores=None,
    ) -> "DASHResult":
        """Construct a DASHResult from a raw (K, n_ref, P) array.

        Parameters
        ----------
        matrices : ndarray (K, n_ref, P)
        feature_names : list[str] or None
            Auto-generated as ["f0", "f1", ...] if None.
        val_scores : list | ndarray | None
        """
        matrices = np.asarray(matrices, dtype=float)
        if matrices.ndim != 3:
            raise ValueError(f"matrices must be 3D, got {matrices.ndim}D")
        P = matrices.shape[2]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(P)]
        return cls(matrices, list(feature_names), val_scores)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Serialize to two files: <path>.npz (arrays) + <path>.json (metadata).

        The JSON sidecar always includes "format_version": 1. If DASHResult gains
        fields in later versions, DASHResult.load() raises VersionError for
        format_version > current, preventing silent misreads.
        """
        path = pathlib.Path(path)
        arrays = {"all_shap_matrices": self.all_shap_matrices}
        if self.val_scores is not None:
            arrays["val_scores"] = self.val_scores
        np.savez_compressed(str(path) + ".npz", **arrays)

        meta = {
            "format_version": _FORMAT_VERSION,
            "feature_names": list(self.feature_names),
            "has_val_scores": self.val_scores is not None,
            "shape": list(self.all_shap_matrices.shape),
        }
        with open(str(path) + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "DASHResult":
        """Load from <path>.npz + <path>.json.

        Raises
        ------
        VersionError
            If the sidecar's format_version exceeds the current version.
        FileNotFoundError
            If either file is missing.
        """
        path = pathlib.Path(path)
        with open(str(path) + ".json") as f:
            meta = json.load(f)

        version = meta.get("format_version", 0)
        if version > _FORMAT_VERSION:
            raise VersionError(
                f"Cannot load DASHResult saved with format_version={version}; "
                f"current version is {_FORMAT_VERSION}. "
                f"Upgrade dash_shap to load this file."
            )

        data = np.load(str(path) + ".npz")
        matrices = data["all_shap_matrices"]
        val_scores = data["val_scores"] if meta.get("has_val_scores") else None
        return cls.from_shap_matrices(
            matrices,
            feature_names=meta["feature_names"],
            val_scores=val_scores,
        )
