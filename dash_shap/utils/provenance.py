"""Provenance utilities for DASH-SHAP experiment reproducibility."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def git_sha() -> str:
    """Return HEAD commit SHA, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def git_dirty() -> bool:
    """Return True if there are uncommitted changes in the working tree."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def config_sha(config: dict) -> str:
    """Return SHA256 hex digest of the canonically-serialized config dict."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hardware_info() -> dict:
    """Return a dict of hardware/environment info."""
    info: dict = {
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    # RAM via psutil (optional)
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    # SageMaker instance type (env var set by SM runtime)
    instance_type = os.environ.get("SM_CURRENT_INSTANCE_TYPE")
    if instance_type:
        info["instance_type"] = instance_type
    return info


def pip_freeze() -> list[str]:
    """Return sorted list of installed packages as 'name==version' strings."""
    try:
        import importlib.metadata as meta

        pkgs = []
        for dist in meta.distributions():
            name = dist.metadata.get("Name", "")
            version = dist.metadata.get("Version", "")
            if name and version:
                pkgs.append(f"{name}=={version}")
        return sorted(pkgs)
    except Exception:
        return []


def write_environment_snapshot(results_dir: str) -> None:
    """Write results_dir/environment.json with pip packages + hardware."""
    from dash_shap.utils.io import save_json

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(),
        "packages": pip_freeze(),
    }
    path = Path(results_dir) / "environment.json"
    save_json(snapshot, str(path))


def capture_run_meta(
    experiment_name: str,
    n_reps: int,
    config: dict,
    elapsed_s: float,
    output_path: str,
) -> dict:
    """Build full _meta dict for a completed experiment run."""
    sha = git_sha()
    dirty = git_dirty()
    return {
        "experiment": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_sha": sha,
        "code_dirty": dirty,
        "config_sha": config_sha(config),
        "n_reps": n_reps,
        "paper_config": config,
        "elapsed_s": round(elapsed_s, 2),
        "output": output_path,
        "hardware": hardware_info(),
    }


def append_provenance_md(meta: dict, results_dir: str) -> None:
    """Append one entry to results_dir/PROVENANCE.md; create with header if new."""
    prov_path = Path(results_dir) / "PROVENANCE.md"
    header = (
        "# Experiment Provenance\n\n"
        "Auto-generated log of experiment runs.\n"
        "Each entry records the code SHA, config fingerprint, hardware, and elapsed time.\n\n"
        "---\n\n"
    )

    lines = []
    dirty = meta.get("code_dirty", False)
    if dirty:
        lines.append("**⚠ DIRTY** — results from uncommitted changes\n")
    lines.append(f"## {meta['experiment']} — {meta['timestamp']}\n")
    lines.append(f"- **code_sha**: `{meta.get('code_sha', 'unknown')}`\n")
    lines.append(f"- **code_dirty**: {meta.get('code_dirty', False)}\n")
    lines.append(f"- **config_sha**: `{meta.get('config_sha', 'unknown')}`\n")
    lines.append(f"- **n_reps**: {meta.get('n_reps')}\n")
    lines.append(f"- **elapsed_s**: {meta.get('elapsed_s')}\n")
    hw = meta.get("hardware", {})
    lines.append(
        f"- **hardware**: {hw.get('cpu_count')} CPUs, "
        f"{hw.get('ram_gb', 'N/A')} GB RAM, "
        f"instance={hw.get('instance_type', 'local')}\n"
    )
    lines.append(f"- **output**: `{meta.get('output')}`\n")
    lines.append("\n")

    entry = "".join(lines)

    if prov_path.exists():
        with open(prov_path, "a") as f:
            f.write(entry)
    else:
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prov_path, "w") as f:
            f.write(header)
            f.write(entry)


def validate_result(data: dict, experiment_name: str) -> list[str]:
    """Return a list of warning strings for any data quality issues found.

    Skips keys starting with '_' (treated as metadata).
    Inspects per-rho/per-method entries that contain a 'runs' list or
    scalar stability/accuracy/equity fields.
    """
    warnings_out: list[str] = []

    def _is_nan(v) -> bool:
        try:
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return False

    def _check_entry(key: str, entry: dict) -> None:
        if not isinstance(entry, dict):
            return

        # NaN in scalar fields
        for field in ("stability", "accuracy", "equity", "dgp_agreement", "rmse"):
            if field in entry and _is_nan(entry[field]):
                warnings_out.append(f"{key}: NaN in field '{field}'")

        # Zero successful reps
        if entry.get("n_successful") == 0:
            warnings_out.append(f"{key}: n_successful == 0")

        # Completeness: acc_runs length vs n_reps
        n_reps = entry.get("n_reps")
        acc_runs = entry.get("acc_runs")
        if acc_runs is not None and n_reps is not None:
            if len(acc_runs) != n_reps:
                warnings_out.append(f"{key}: len(acc_runs)={len(acc_runs)} != n_reps={n_reps}")

        # CI sanity: lo <= point <= hi
        if all(k in entry for k in ("stability_lo", "stability", "stability_hi")):
            lo = entry["stability_lo"]
            pt = entry["stability"]
            hi = entry["stability_hi"]
            try:
                if float(lo) > float(pt) or float(pt) > float(hi):
                    warnings_out.append(f"{key}: CI sanity fail — lo={lo} pt={pt} hi={hi}")
            except (TypeError, ValueError):
                pass

        # Degenerate bootstrap
        se = entry.get("stability_se")
        if se is not None:
            try:
                if float(se) <= 0:
                    warnings_out.append(f"{key}: stability_se <= 0 ({se})")
            except (TypeError, ValueError):
                pass

    for outer_key, outer_val in data.items():
        if isinstance(outer_key, str) and outer_key.startswith("_"):
            continue
        if isinstance(outer_val, dict):
            for inner_key, inner_val in outer_val.items():
                if isinstance(inner_key, str) and inner_key.startswith("_"):
                    continue
                if isinstance(inner_val, dict):
                    _check_entry(f"{outer_key}/{inner_key}", inner_val)
                else:
                    # outer_val itself might be a method-level result
                    pass
            # Also check outer_val as a direct entry (e.g. real-world results)
            _check_entry(outer_key, outer_val)

    return warnings_out
