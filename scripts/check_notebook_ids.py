#!/usr/bin/env python3
"""Report code cells with empty IDs in tracked notebooks.

Run before any notebook editing session to identify cells that cannot be
targeted by cell_id in NotebookEdit operations.

Usage:
    python scripts/check_notebook_ids.py
"""
import json
import sys
from pathlib import Path

NOTEBOOKS = [
    "notebooks/demo_benchmark_7_parallel.ipynb",
    "notebooks/explore_experiment_results.ipynb",
]

issues = []
for nb_path in NOTEBOOKS:
    p = Path(nb_path)
    if not p.exists():
        continue
    nb = json.load(p.open())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and not cell.get("id", "").strip():
            preview = "".join(cell["source"])[:60].replace("\n", " ")
            issues.append(f"{nb_path}:cell[{i}] — unnamed: {preview!r}")

if issues:
    print("Unnamed code cells found (assign IDs before editing):")
    for issue in issues:
        print(f"  {issue}")
    sys.exit(1)

print(f"All code cells have IDs ({len(NOTEBOOKS)} notebooks checked).")
