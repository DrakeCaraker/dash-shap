"""I/O utilities for saving experiment results."""

from __future__ import annotations

import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def _convert(obj):
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def save_json(
    data: dict,
    path: str,
    meta: Optional[dict] = None,
    overwrite_protection: bool = True,
) -> None:
    """Save results dict to JSON, converting numpy types to native Python.

    Parameters
    ----------
    data:
        The result dict to serialize.
    path:
        Destination file path.
    meta:
        Optional provenance dict. When provided, embedded as ``"_meta"`` as
        the first key in the output JSON.
    overwrite_protection:
        When True (default), if a valid JSON file already exists at *path*
        it is backed up to ``<stem>.<YYYYMMDD_HHMMSS>.bak.json`` before
        overwriting.  Silently skips backup if the existing file is empty or
        corrupt.
    """
    p = Path(path)

    if overwrite_protection and p.exists():
        try:
            existing_text = p.read_text()
            json.loads(existing_text)  # validate — only backup valid JSON
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup = p.parent / f"{p.stem}.{ts}.bak.json"
            shutil.copy2(p, backup)
            warnings.warn(
                f"save_json: existing file backed up to {backup}",
                UserWarning,
                stacklevel=2,
            )
        except (json.JSONDecodeError, Exception):
            pass  # empty or corrupt — skip backup, overwrite silently

    if meta is not None:
        output = {"_meta": _convert(meta)}
        output.update(_convert(data))
    else:
        output = _convert(data)

    with open(p, "w") as f:
        json.dump(output, f, indent=2)
