#!/usr/bin/env bash
f=$(jq -r '.tool_input.file_path // ""')
echo "$f" | grep -qE '\.tex$' || exit 0
echo "$f" | grep -q '/paper/' || exit 0
command -v chktex >/dev/null 2>&1 || exit 0
echo "LaTeX lint: $f"
chktex -q "$f" 2>&1 | head -20 || true
