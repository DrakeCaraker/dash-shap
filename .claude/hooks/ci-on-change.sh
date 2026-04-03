#!/usr/bin/env bash
# Run lint + fast tests when dash_shap/ or tests/ Python files have been modified.
# Reads commands from alfred.yaml via alfred-config.sh (falls back to defaults).
set -euo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# Ensure user-installed binaries (ruff, mypy, etc.) are on PATH
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:$PATH"

# Read config from alfred.yaml (with defaults)
source "$(dirname "$0")/alfred-config.sh"

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
  echo "Source files modified — running lint + fast tests..."
  # Run from repo root so make targets resolve correctly
  if ! (cd "$REPO" && $ALFRED_LINT) 2>&1; then
    echo "Alfred: lint failed — run /ci-fix to auto-repair."
    exit 0
  fi

  if python3 -c "import xgboost" 2>/dev/null; then
    if ! (cd "$REPO" && $ALFRED_TEST_FAST) 2>&1; then
      echo "Alfred: tests failed — run /ci-fix to auto-repair."
      exit 0
    fi
  else
    echo "xgboost not installed — skipping tests (run: pip install xgboost)"
  fi
else
  echo "No dash_shap/ or tests/ .py changes — skipping local CI."
fi
