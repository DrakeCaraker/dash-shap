#!/usr/bin/env bash
# Run make ci when dash_shap/ or tests/ Python files have been modified.
set -euo pipefail

REPO=/Users/drake.caraker/ds_projects/dash-shap

changed=$(git -C "$REPO" status --porcelain 2>/dev/null \
  | awk '{print $2}' \
  | grep -E '^(dash_shap|tests)/.*\.py$' || true)

if [ -n "$changed" ]; then
  echo "Source files modified — running make ci..."
  make -C "$REPO" ci
else
  echo "No dash_shap/ or tests/ .py changes — skipping make ci."
fi
