#!/usr/bin/env bash
# Run make ci when dash_shap/ or tests/ Python files have been modified.
set -euo pipefail

REPO=/Users/drake.caraker/ds_projects/dash-shap

# Ensure user-installed binaries (ruff, mypy, etc.) are on PATH
export PATH="/Users/drake.caraker/Library/Python/3.9/bin:$PATH"

changed=$(git -C "$REPO" status --porcelain 2>/dev/null \
  | awk '{print $2}' \
  | grep -E '^(dash_shap|tests)/.*\.py$' || true)

if [ -n "$changed" ]; then
  echo "Source files modified — running lint + typecheck..."
  make -C "$REPO" lint typecheck

  # Run tests only if xgboost is available (required by pipeline tests)
  if python3 -c "import xgboost" 2>/dev/null; then
    make -C "$REPO" test coverage
  else
    echo "xgboost not installed — skipping test/coverage (run: pip install xgboost)"
    pytest --ignore=tests/test_baselines.py \
           --ignore=tests/test_baselines_extended.py \
           --ignore=tests/test_consensus.py \
           --ignore=tests/test_diversity_filtering.py \
           --ignore=tests/test_pipeline.py \
           --ignore=tests/test_utils.py \
           --ignore=tests/test_cli_smoke.py \
           -q --tb=short
  fi
else
  echo "No dash_shap/ or tests/ .py changes — skipping make ci."
fi
