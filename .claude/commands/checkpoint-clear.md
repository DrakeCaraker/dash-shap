# Checkpoint Management

List and optionally clear checkpoint files:

1. Find all checkpoint/pkl files:
   ```
   find . -name '*.pkl' -not -path './.git/*' -not -path './.venv/*' 2>/dev/null
   ls -lh checkpoints/ 2>/dev/null
   ```
2. Show total size: `du -sh checkpoints/ 2>/dev/null`
3. Present a numbered list of files with sizes
4. Ask the user which to delete: all, specific numbers, or none
5. **Only delete after explicit confirmation** — never auto-delete

If no checkpoint files exist, say so and exit.
