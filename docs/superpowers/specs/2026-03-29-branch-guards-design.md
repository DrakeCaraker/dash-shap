# Branch Guards: Preventing Accidental Results-to-Main Merges

**Date:** 2026-03-29
**Status:** Proposed
**Motivation:** PR #239 accidentally merged an in-progress results branch into main, requiring a revert. PR #198 had the same pattern earlier. The existing CI pipeline has no guard against this.

## Problem

Two failure modes exist with `results/*` experiment branches:

1. **Results merged to main prematurely.** A PR from `results/sagemaker-run-*` to `main` is opened (via GitHub UI, `gh pr create`, or a Claude session) and merged before the run is complete. This pollutes main with interim data and violates the SageMaker Run Protocol.

2. **Code committed directly to results branches.** A Claude session on SageMaker pushes code changes (`.py`, `.tex`, config) directly to the results branch instead of routing through main. This breaks the provenance invariant: "every line of code on the results branch must also exist on main."

## Design

### New file: `.github/workflows/branch-guards.yml`

A dedicated workflow with two jobs. Triggers on all `pull_request` events (no branch filter) with types `opened`, `synchronize`, `labeled`, `unlabeled`.

**No branch filter is intentional.** If the workflow used `branches: [main]`, it would not trigger for PRs targeting other branches. Jobs that never trigger report as "Pending" in required checks, blocking unrelated PRs. By triggering on all PRs and using job-level `if` conditions, non-matching jobs are skipped, which GitHub treats as passing for required checks.

#### Job 1: `block-results-to-main`

**When it fires:** PR head branch starts with `results/` AND base branch is `main`.

**Logic:**
- Check if the PR has a `run-complete` label.
- If present: pass. Print confirmation message.
- If absent: fail. Print actionable error listing what must be verified before adding the label.

**Required status check:** Yes. Add `block-results-to-main` to the required checks on `main`. For normal PRs (not from results branches), the job is skipped, which satisfies the required check.

**Label re-evaluation:** The `labeled` and `unlabeled` event types ensure the check re-runs when the label is added or removed. GitHub will not allow merge until the latest check run passes.

**Error message should instruct the user to verify:**
1. All experiments are complete
2. `python scripts/backfill_meta.py` has been run
3. `code_dirty: false` in all result JSONs
4. The results branch is tagged (`results-YYYYMMDD-final`)
5. Only data files (JSON, figures) differ from main

#### Job 2: `block-code-to-results`

**When it fires:** PR base branch starts with `results/`.

**Logic:**
- Use the GitHub API to list all files changed in the PR.
- If every changed file path starts with `results/`: pass.
- If any file path is outside `results/`: fail. List the violating files.

**Required status check:** No. This is informational. Making it required on `results/*` branches would force all data pushes through PRs, breaking the direct-push workflow from SageMaker.

**No checkout needed.** Uses `gh api repos/{owner}/{repo}/pulls/{number}/files` with pagination.

### CLAUDE.md updates

1. Add `block-results-to-main` to the recommended required status checks list (alongside `freshness`, `lint`, `test`, `typecheck`).
2. Add a step to the "After run completes" protocol: "Add the `run-complete` label to the data PR after verifying provenance."
3. Document the `branch-guards` workflow in the Hooks/CI section.

## Documented limitation

**Direct pushes to results branches are not CI-guarded.** The `block-code-to-results` job only fires on PRs targeting results branches. If a Claude session on SageMaker pushes code commits directly (as happened in the 20260326 run), this guard does not fire. Enforcing this in CI would require branch protection on `results/*` branches, which would block the direct-push data workflow that SageMaker relies on.

The primary guard against code-on-results-branch remains the SageMaker Run Protocol in CLAUDE.md and Non-Negotiable Rule #8 ("Scripts before remote work").

## Implementation plan

1. Create `.github/workflows/branch-guards.yml` with both jobs.
2. Update CLAUDE.md: required checks list, "After run completes" section, hooks documentation.
3. Create the `run-complete` label in the GitHub repo.
4. Add `block-results-to-main` to the required status checks in GitHub repo settings.
5. Test by opening a dummy PR from a `results/test-*` branch.

## Files changed

| File | Change |
|------|--------|
| `.github/workflows/branch-guards.yml` | New file |
| `CLAUDE.md` | Update required checks, run protocol, hooks docs |
