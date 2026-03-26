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
  CI_FAILED=""

  if ! make -C "$REPO" lint typecheck 2>&1; then
    CI_FAILED="lint_or_typecheck"
  fi

  # Run tests only if lint/typecheck passed and xgboost is available
  if [ -z "$CI_FAILED" ]; then
    if python3 -c "import xgboost" 2>/dev/null; then
      if ! make -C "$REPO" test coverage 2>&1; then
        CI_FAILED="tests_or_coverage"
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
        CI_FAILED="tests"
      fi
    fi
  fi

  if [ -n "$CI_FAILED" ]; then
    echo '{"systemMessage": "CI checks failed ('"$CI_FAILED"'). Before ending this session, offer to run /ci-fix to auto-repair. If the user declines, remind them to run it next session. Do not push code with failing checks."}'
    exit 0
  fi
else
  echo "No dash_shap/ or tests/ .py changes — skipping make ci."
fi
