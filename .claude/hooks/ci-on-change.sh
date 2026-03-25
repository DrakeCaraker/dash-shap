#!/usr/bin/env bash
# Run make ci when dash_shap/ or tests/ Python files have been modified.
set -euo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# Ensure user-installed binaries (ruff, mypy, etc.) are on PATH
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$PATH"

changed=$(git -C "$REPO" status --porcelain 2>/dev/null \
  | awk '{print $2}' \
  | grep -E '^(dash_shap|tests)/.*\.py$' || true)

if [ -n "$changed" ]; then
  echo "Source files modified — running lint + typecheck..."
  if ! make -C "$REPO" lint typecheck 2>&1; then
    echo '{"decision": "block", "reason": "Lint or typecheck failed. Run /ci-fix to auto-repair, or fix manually with: make lint && make typecheck"}'
    exit 0
  fi

  # Run tests only if xgboost is available (required by pipeline tests)
  if python3 -c "import xgboost" 2>/dev/null; then
    if ! make -C "$REPO" test coverage 2>&1; then
      echo '{"decision": "block", "reason": "Tests or coverage failed. Run /ci-fix to auto-repair, or fix manually with: make test-fast"}'
      exit 0
    fi
  else
    echo "xgboost not installed — skipping test/coverage (run: pip install xgboost)"
    if ! pytest --ignore=tests/test_baselines.py \
           --ignore=tests/test_baselines_extended.py \
           --ignore=tests/test_consensus.py \
           --ignore=tests/test_diversity_filtering.py \
           --ignore=tests/test_pipeline.py \
           --ignore=tests/test_utils.py \
           --ignore=tests/test_cli_smoke.py \
           -q --tb=short 2>&1; then
      echo '{"decision": "block", "reason": "Tests failed. Run /ci-fix to auto-repair."}'
      exit 0
    fi
  fi
else
  echo "No dash_shap/ or tests/ .py changes — skipping make ci."
fi
