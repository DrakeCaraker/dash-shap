#!/bin/bash
# PreToolUse hook: warn if a notebook file is already large before editing
# Reads tool input JSON from stdin

input=$(cat)
tool_name=$(echo "$input" | jq -r '.tool_name // empty')

# Only check Write and NotebookEdit tools
if [[ "$tool_name" != "Write" && "$tool_name" != "NotebookEdit" ]]; then
  exit 0
fi

# Extract file path from tool input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')

# Only check notebook files
if [[ ! "$file_path" =~ \.ipynb$ ]]; then
  exit 0
fi

# Check if file exists and its size
if [[ -f "$file_path" ]]; then
  size=$(stat -c%s "$file_path" 2>/dev/null)
  if [[ "$size" -gt 2097152 ]]; then  # 2MB
    size_mb=$(echo "scale=1; $size / 1048576" | bc)
    echo "WARNING: $file_path is ${size_mb}MB. Consider clearing outputs before committing (notebooks >10MB are blocked by pre-push hook)." >&2
  fi
fi

exit 0
