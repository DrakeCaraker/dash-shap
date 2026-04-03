#!/usr/bin/env bash
# Reads values from alfred.yaml. Source this from other hooks:
#   source "$(dirname "$0")/alfred-config.sh"
#
# Provides: ALFRED_LINT, ALFRED_TEST_FAST, ALFRED_FORMAT_TOOL, ALFRED_SOURCE_PATHS
# Falls back to sensible defaults if alfred.yaml is missing or python/yaml unavailable.

_REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
_ALFRED="$_REPO/.claude/alfred.yaml"

_alfred_read() {
  # Usage: _alfred_read "ci.lint_command" "make lint"
  local key="$1" default="$2"
  if [ ! -f "$_ALFRED" ]; then
    echo "$default"
    return
  fi
  # Use python to parse YAML (available everywhere, no extra deps)
  val=$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$_ALFRED'))
    keys = '$key'.split('.')
    v = cfg
    for k in keys:
        v = v[k]
    print(v)
except Exception:
    sys.exit(1)
" 2>/dev/null) && echo "$val" || echo "$default"
}

ALFRED_LINT=$(_alfred_read "ci.lint_command" "make lint")
ALFRED_TEST_FAST=$(_alfred_read "testing.fast_command" "make test-fast")
ALFRED_FORMAT_TOOL=$(_alfred_read "formatting.tool" "ruff")
ALFRED_SOURCE_PATHS=$(_alfred_read "ci.source_paths" '["dash_shap/", "tests/"]')
