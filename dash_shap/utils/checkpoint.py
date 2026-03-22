"""Checkpoint utilities for resumable experiment execution.

Provides save/load/has/clear checkpoint operations using pickle serialization.
Checkpoints are stored in a configurable directory (default: checkpoints/ at repo root).
"""

import hashlib
import json as _json
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

DEFAULT_CHECKPOINT_DIR = "checkpoints"


def _sanitize_ckpt_name(s: str) -> str:
    """Sanitize a string for use in checkpoint filenames."""
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()


def _checkpoint_path(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Return the full path for a named checkpoint."""
    return Path(checkpoint_dir) / f"ckpt_{name}.pkl"


def _config_fingerprint(config: Optional[dict]) -> Optional[str]:
    """Return a stable 64-char SHA-256 hex digest of a config dict, or None."""
    if config is None:
        return None
    canonical = _json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_checkpoint(
    name: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    config: Optional[dict] = None,
    **data,
):
    """Save keyword arguments as a named checkpoint.

    Parameters
    ----------
    name : str
        Checkpoint name (e.g. 'linear_sweep_rho_0.9').
    checkpoint_dir : str
        Directory for checkpoint files.
    **data
        Arbitrary keyword data to serialize.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _checkpoint_path(name, checkpoint_dir)
    payload = dict(data)
    fingerprint = _config_fingerprint(config)
    if fingerprint is not None:
        payload["__meta__"] = {"config_hash": fingerprint}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [CHECKPOINT] Saved {name} ({size_mb:.1f} MB)")


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler that remaps old 'dash.*' modules to 'dash_shap.*'.

    Checkpoints saved before the package rename (dash → dash_shap) embed
    the old module paths. This remaps them transparently on load.
    """

    def find_class(self, module: str, name: str):
        if module == "dash" or module.startswith("dash."):
            module = "dash_shap" + module[4:]
        return super().find_class(module, name)


def load_checkpoint(
    name: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    config: Optional[dict] = None,
) -> Optional[dict]:
    """Load a named checkpoint. Returns dict or None if not found."""
    path = _checkpoint_path(name, checkpoint_dir)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = _LegacyUnpickler(f).load()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [CHECKPOINT] Loaded {name} ({size_mb:.1f} MB)")
    if config is not None:
        expected = _config_fingerprint(config)
        assert expected is not None  # config is not None, so fingerprint is never None
        stored_hash = data.get("__meta__", {}).get("config_hash")
        if stored_hash is not None and stored_hash != expected:
            warnings.warn(
                f"[CHECKPOINT] '{name}' was saved with a different config "
                f"(stored={stored_hash[:8]}…, current={expected[:8]}…). "
                "Results may be stale — run clear_checkpoint() to recompute.",
                UserWarning,
                stacklevel=2,
            )
    return data


def has_checkpoint(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> bool:
    """Check if a named checkpoint exists."""
    return _checkpoint_path(name, checkpoint_dir).exists()


def clear_checkpoint(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Delete a named checkpoint if it exists."""
    path = _checkpoint_path(name, checkpoint_dir)
    if path.exists():
        path.unlink()
        print(f"  [CHECKPOINT] Cleared {name}")


def clear_checkpoints_by_prefix(prefix: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Delete all checkpoints whose name starts with prefix.

    Useful for cleaning up per-level checkpoints after an experiment completes.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return
    for path in ckpt_dir.glob(f"ckpt_{prefix}*.pkl"):
        path.unlink()
    print(f"  [CHECKPOINT] Cleared all {prefix}* checkpoints")
