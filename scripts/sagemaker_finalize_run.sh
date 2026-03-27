#!/usr/bin/env bash
# =============================================================================
# Finalize a SageMaker experiment run: backfill metadata, commit, tag, push.
# Run this from the repo root after all experiments have completed.
# =============================================================================
set -euo pipefail

RUN_DATE=$(date +%Y%m%d)
TAG_END="run-tmlr-${RUN_DATE}-end"

# --- Verify we're on a results branch ---
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != results/* ]]; then
    echo "ERROR: Not on a results branch (on: ${BRANCH})."
    echo "Switch first: git checkout results/sagemaker-run-YYYYMMDD"
    exit 1
fi

# --- Verify experiments aren't still running ---
if pgrep -f run_experiments_parallel > /dev/null 2>&1; then
    echo "WARNING: run_experiments_parallel is still running."
    read -rp "Continue anyway? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

# --- Show results ---
echo "Result files:"
ls -lht results/tables/*.json 2>/dev/null || echo "  (none found)"
echo ""

# --- Backfill provenance ---
echo "Backfilling provenance metadata..."
python scripts/backfill_meta.py

# --- Commit ---
echo ""
git add results/
git status --short results/

INSTANCE="${SM_CURRENT_INSTANCE_TYPE:-unknown}"
read -rp "Commit these results? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || exit 1

git commit -m "data: SageMaker run ${RUN_DATE} — experiment results (N_REPS=50)

Instance: ${INSTANCE}
Code SHA: $(git rev-parse --short HEAD)
Branch: ${BRANCH}"

# --- Tag ---
if git rev-parse "$TAG_END" &>/dev/null; then
    echo "Tag ${TAG_END} already exists, skipping."
else
    git tag "$TAG_END" "$(git rev-parse HEAD)"
fi

# --- Push ---
git push origin "$BRANCH" --tags

echo ""
echo "Done."
echo "  Branch: ${BRANCH}"
echo "  Tag:    ${TAG_END} -> $(git rev-parse --short HEAD)"
echo ""
echo "Next: open a data-only PR to main from your workstation."
