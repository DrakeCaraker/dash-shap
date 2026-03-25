"""Backfill _meta blocks into result JSONs that predate provenance infrastructure.

Run this on the SageMaker instance after the run completes:

    python scripts/backfill_meta.py

It is safe to re-run: files that already have a '_meta' block are skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results/tables")

# filename stem → experiment name (matches _publish_results calls in runner)
FILENAME_TO_EXPERIMENT: dict[str, str] = {
    "synthetic_linear_sweep": "linear_sweep",
    "nonlinear_sweep": "nonlinear_sweep",
    "overlapping": "overlapping",
    "table2_baselines": "table2_baselines",
    "california_housing": "real_california",
    "breast_cancer": "real_breast_cancer",
    "superconductor": "real_superconductor",
    "epsilon_sensitivity": "epsilon_sensitivity",
    "ablation": "ablation",
    "variance_decomposition": "variance_decomposition",
    "asymmetric_dgp": "asymmetric_dgp",
    "variance_decomposition_crossed": "variance_decomposition_crossed",
    "background_sensitivity": "background_sensitivity",
    "first_mover_visualization": "first_mover_visualization",
    "first_mover_bias": "first_mover_bias",
    "k_sweep_independence": "k_sweep_independence",
}

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


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty() -> bool:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return bool(r.stdout.strip())
    except Exception:
        pass
    return False


def _config_sha(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _hardware_info() -> dict:
    info: dict = {
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    instance_type = os.environ.get("SM_CURRENT_INSTANCE_TYPE")
    if instance_type:
        info["instance_type"] = instance_type
    return info


def build_meta(experiment_name: str, output_path: str, code_sha: str, code_dirty: bool) -> dict:
    return {
        "experiment": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_sha": code_sha,
        "code_dirty": code_dirty,
        "config_sha": _config_sha(PAPER_CONFIG),
        "n_reps": PAPER_CONFIG["N_REPS"],
        "paper_config": PAPER_CONFIG,
        "elapsed_s": None,  # not recorded — run predated provenance infrastructure
        "output": output_path,
        "hardware": _hardware_info(),
        "backfilled": True,  # honest marker: _meta was added post-hoc
    }


def backfill_file(path: Path, experiment_name: str, code_sha: str, code_dirty: bool) -> str:
    data = json.loads(path.read_text())

    if "_meta" in data:
        return "skip"

    meta = build_meta(experiment_name, str(path), code_sha, code_dirty)

    # Write _meta as first key
    out = {"_meta": meta, **data}
    path.write_text(json.dumps(out, indent=2, default=str))
    return "backfilled"


def main() -> None:
    code_sha = _git_sha()
    code_dirty = _git_dirty()

    print(f"code_sha : {code_sha}")
    print(f"code_dirty: {code_dirty}")
    print(f"config_sha: {_config_sha(PAPER_CONFIG)}")
    print()

    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return

    for path in json_files:
        stem = path.stem
        if stem not in FILENAME_TO_EXPERIMENT:
            print(f"  UNKNOWN  {path.name} — not in filename map, skipping")
            continue
        experiment_name = FILENAME_TO_EXPERIMENT[stem]
        status = backfill_file(path, experiment_name, code_sha, code_dirty)
        print(f"  {status:10s}  {path.name}  ({experiment_name})")

    print("\nDone. Re-run to verify (already-backfilled files will show 'skip').")


if __name__ == "__main__":
    main()
