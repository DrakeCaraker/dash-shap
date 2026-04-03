#!/usr/bin/env bash
# Auto-format Python files on write. Reads formatter from alfred.yaml.
source "$(dirname "$0")/alfred-config.sh" 2>/dev/null || ALFRED_FORMAT_TOOL="ruff"
f=$(jq -r '.tool_input.file_path // ""')
echo "$f" | grep -qE '\.py$' || exit 0
echo "$f" | grep -qE '/notebooks/' && exit 0
$ALFRED_FORMAT_TOOL format "$f" 2>/dev/null || true
