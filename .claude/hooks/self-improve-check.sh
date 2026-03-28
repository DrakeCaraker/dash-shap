#!/bin/bash
# Stop hook: lightweight self-improve classification
# Checks if feedback memories have accumulated and surfaces proposals.
# Does NOT modify any files — only reads and reports.

MEMORY_DIR="$HOME/.claude/projects/-Users-drake-caraker-ds-projects-dash-shap/memory"
FEEDBACK_COUNT=$(ls "$MEMORY_DIR"/feedback_*.md 2>/dev/null | wc -l | tr -d ' ')

if [ "$FEEDBACK_COUNT" -ge 3 ]; then
    echo "Alfred: $FEEDBACK_COUNT feedback memories accumulated — consider running /self-improve."
fi
