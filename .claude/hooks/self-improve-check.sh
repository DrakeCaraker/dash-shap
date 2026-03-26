#!/bin/bash
# Stop hook: lightweight self-improve classification
# Checks if feedback memories have accumulated and surfaces proposals.
# Does NOT modify any files — only reads and reports.

MEMORY_DIR="$HOME/.claude/projects/-Users-drake-caraker-ds-projects-dash-shap/memory"
FEEDBACK_COUNT=$(ls "$MEMORY_DIR"/feedback_*.md 2>/dev/null | wc -l | tr -d ' ')

if [ "$FEEDBACK_COUNT" -ge 3 ]; then
    echo "{\"systemMessage\": \"Self-improve check: $FEEDBACK_COUNT feedback memories have accumulated. Before ending, run the /self-improve classification: read each feedback memory, check if it duplicates a CLAUDE.md rule (delete if so), and if any pattern has appeared 2+ times, propose promoting it to a rule. Show the user what you found and ask before making changes.\"}"
fi
