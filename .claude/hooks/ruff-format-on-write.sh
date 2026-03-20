#!/usr/bin/env bash
f=$(jq -r '.tool_input.file_path // ""')
echo "$f" | grep -qE '\.py$' || exit 0
echo "$f" | grep -qE '/notebooks/' && exit 0
ruff format "$f" 2>/dev/null || true
