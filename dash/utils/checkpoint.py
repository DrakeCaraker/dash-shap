"""Checkpoint utilities for resumable experiment execution.

Provides save/load/has/clear checkpoint operations using pickle serialization.
Checkpoints are stored in a configurable directory (default: checkpoints/ at repo root).
"""
import os
import pickle
from pathlib import Path

DEFAULT_CHECKPOINT_DIR = "checkpoints"


def _sanitize_ckpt_name(s: str) -> str:
    """Sanitize a string for use in checkpoint filenames."""
    return s.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').lower()


def _checkpoint_path(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Return the full path for a named checkpoint."""
    return Path(checkpoint_dir) / f"ckpt_{name}.pkl"


def save_checkpoint(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR, **data):
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
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f'  [CHECKPOINT] Saved {name} ({size_mb:.1f} MB)')


def load_checkpoint(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Load a named checkpoint. Returns dict or None if not found."""
    path = _checkpoint_path(name, checkpoint_dir)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f'  [CHECKPOINT] Loaded {name} ({size_mb:.1f} MB)')
    return data


def has_checkpoint(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> bool:
    """Check if a named checkpoint exists."""
    return _checkpoint_path(name, checkpoint_dir).exists()


def clear_checkpoint(name: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Delete a named checkpoint if it exists."""
    path = _checkpoint_path(name, checkpoint_dir)
    if path.exists():
        path.unlink()
        print(f'  [CHECKPOINT] Cleared {name}')


def clear_checkpoints_by_prefix(prefix: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Delete all checkpoints whose name starts with prefix.

    Useful for cleaning up per-level checkpoints after an experiment completes.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return
    for path in ckpt_dir.glob(f"ckpt_{prefix}*.pkl"):
        path.unlink()
    print(f'  [CHECKPOINT] Cleared all {prefix}* checkpoints')
