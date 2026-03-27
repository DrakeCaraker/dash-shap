#!/usr/bin/env bash
# Run make ci when dash_shap/ or tests/ Python files have been modified.
set -euo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# Ensure user-installed binaries (ruff, mypy, etc.) are on PATH
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$PATH"

# Validate hook scripts have no syntax errors (catches stale variables, typos)
for hook in "$REPO"/.claude/hooks/*.sh; do
  if ! bash -n "$hook" 2>/dev/null; then
    echo "WARNING: syntax error in $hook"
  fi
done

changed=$(git -C "$REPO" status --porcelain 2>/dev/null \
  | awk '{print $2}' \
  | grep -E '^(dash_shap|tests)/.*\.py$' || true)

if [ -n "$changed" ]; then
  # Fast local checks only — formatting and typecheck are handled by CI.
  echo "Source files modified — running lint + fast tests..."
  if ! make -C "$REPO" lint 2>&1; then
    echo '{"decision": "block", "reason": "Lint failed. Run /ci-fix to auto-repair, or fix manually with: make lint"}'
    exit 0
  fi

  # Run fast tests only if xgboost is available (required by pipeline tests)
  if python3 -c "import xgboost" 2>/dev/null; then
    if ! make -C "$REPO" test-fast 2>&1; then
      echo '{"decision": "block", "reason": "Tests failed. Run /ci-fix to auto-repair, or fix manually with: make test-fast"}'
      exit 0
    fi
  else
    echo "xgboost not installed — skipping tests (run: pip install xgboost)"
  fi
else
  echo "No dash_shap/ or tests/ .py changes — skipping local CI."
fi
