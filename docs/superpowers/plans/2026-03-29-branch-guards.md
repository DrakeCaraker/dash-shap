# Branch Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent accidental merges of in-progress results branches to main via a CI guard with label-based bypass, and block code changes in PRs targeting results branches.

**Architecture:** A dedicated GitHub Actions workflow (`branch-guards.yml`) with two jobs. Job 1 blocks PRs from `results/*` to `main` unless a `run-complete` label is present. Job 2 blocks PRs targeting `results/*` branches that modify files outside `results/`. No branch filter on the workflow trigger — job-level `if` conditions handle routing so skipped jobs satisfy required checks.

**Tech Stack:** GitHub Actions, `gh` CLI, GitHub API

**Spec:** `docs/superpowers/specs/2026-03-29-branch-guards-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `.github/workflows/branch-guards.yml` | New. Two CI jobs: `block-results-to-main` and `block-code-to-results` |
| `CLAUDE.md` | Modify. Update required checks list, "After run completes" protocol, hooks docs |

---

### Task 1: Create the workflow file with both jobs

**Files:**
- Create: `.github/workflows/branch-guards.yml`

- [ ] **Step 1: Create `.github/workflows/branch-guards.yml`**

```yaml
name: Branch Guards

on:
  pull_request:
    types: [opened, synchronize, labeled, unlabeled]

jobs:
  block-results-to-main:
    # Fires when a results/* branch targets main.
    # Skipped for all other PRs (satisfies required checks).
    if: >-
      startsWith(github.head_ref, 'results/') &&
      github.base_ref == 'main'
    runs-on: ubuntu-latest
    steps:
      - name: Check for run-complete label
        run: |
          if echo '${{ toJson(github.event.pull_request.labels.*.name) }}' | grep -q 'run-complete'; then
            echo "run-complete label found. Merge allowed."
            echo ""
            echo "Reminder: verify before merging:"
            echo "  1. All experiments are complete"
            echo "  2. backfill_meta.py has been run"
            echo "  3. code_dirty: false in all result JSONs"
            echo "  4. Results branch is tagged (results-YYYYMMDD-final)"
            echo "  5. Only data files (JSON, figures) differ from main"
          else
            echo "::error::Results branches cannot be merged to main without the 'run-complete' label."
            echo ""
            echo "Before adding the label, verify:"
            echo "  1. All experiments are complete"
            echo "  2. Run: python scripts/backfill_meta.py"
            echo "  3. Confirm code_dirty: false in all result JSONs"
            echo "  4. Tag the branch: git tag results-YYYYMMDD-final"
            echo "  5. Confirm only data files (JSON, figures) differ from main"
            echo ""
            echo "Then add the 'run-complete' label to this PR to unblock."
            exit 1
          fi

  block-code-to-results:
    # Fires when any PR targets a results/* branch.
    # Fails if the PR modifies files outside results/.
    if: startsWith(github.base_ref, 'results/')
    runs-on: ubuntu-latest
    steps:
      - name: Check for non-data files
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          echo "Checking PR #${{ github.event.pull_request.number }} for code files..."
          files=$(gh api "repos/${{ github.repository }}/pulls/${{ github.event.pull_request.number }}/files" \
            --paginate --jq '.[].filename')

          violations=""
          while IFS= read -r f; do
            [ -z "$f" ] && continue
            if [[ "$f" != results/* ]]; then
              violations="$violations  $f"$'\n'
            fi
          done <<< "$files"

          if [ -n "$violations" ]; then
            echo "::error::PRs to results branches may only modify files under results/."
            echo ""
            echo "The following files are outside results/:"
            echo "$violations"
            echo ""
            echo "Code changes must go through main first (feature branch -> PR -> main),"
            echo "then cherry-pick to the results branch. See CLAUDE.md: SageMaker Run Protocol."
            exit 1
          fi

          echo "All changed files are under results/. OK."
```

- [ ] **Step 2: Verify the YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/branch-guards.yml')); print('YAML OK')"`
Expected: `YAML OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/branch-guards.yml
git commit -m "ci: add branch guards to block premature results merges

Job 1 (block-results-to-main): blocks PRs from results/* to main
unless the 'run-complete' label is present. Add as required check.

Job 2 (block-code-to-results): blocks PRs targeting results/* branches
that modify files outside results/. Informational (not required)."
```

---

### Task 2: Create the `run-complete` label in GitHub

**Files:** None (GitHub API only)

- [ ] **Step 1: Create the label**

```bash
gh label create "run-complete" \
  --description "All experiments finished, provenance verified. Unblocks results->main merge." \
  --color "0E8A16"
```

Expected: label created (green color, matches "success" convention).

- [ ] **Step 2: Verify the label exists**

```bash
gh label list --search "run-complete"
```

Expected: one row showing `run-complete` with the description.

---

### Task 3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:151` (required status checks)
- Modify: `CLAUDE.md:205-210` ("After run completes" section)
- Modify: `CLAUDE.md:230-237` (hooks section)

- [ ] **Step 1: Add `block-results-to-main` to required status checks**

At `CLAUDE.md:151`, change:

```markdown
1. **Require status checks to pass before merging**: `freshness`, `lint`, `test`, `typecheck`
```

to:

```markdown
1. **Require status checks to pass before merging**: `freshness`, `lint`, `test`, `typecheck`, `block-results-to-main`
```

- [ ] **Step 2: Update "After run completes" protocol with label step**

At `CLAUDE.md:205-210`, change:

```markdown
### After run completes
1. Run `python scripts/backfill_meta.py` on SageMaker — ensures all JSONs have `_meta` hardware blocks
2. Create a completion commit: `chore: mark run-YYYYMMDD complete — all N experiments finished`
3. Tag the results branch: `git tag results-YYYYMMDD-final`
4. Open a **data-only PR to main** — only JSON/figure additions; code changes already landed via their own PRs
5. Freeze the branch — no further commits after the PR merges
```

to:

```markdown
### After run completes
1. Run `python scripts/backfill_meta.py` on SageMaker — ensures all JSONs have `_meta` hardware blocks
2. Create a completion commit: `chore: mark run-YYYYMMDD complete — all N experiments finished`
3. Tag the results branch: `git tag results-YYYYMMDD-final`
4. Open a **data-only PR to main** — only JSON/figure additions; code changes already landed via their own PRs
5. Verify provenance, then add the `run-complete` label to the PR — CI blocks merge until this label is present
6. Freeze the branch — no further commits after the PR merges
```

- [ ] **Step 3: Add branch-guards to hooks documentation**

At `CLAUDE.md:231` (after the Pre-push hook line), add:

```markdown
- **CI: Branch guards** — blocks PRs from `results/*` → `main` unless `run-complete` label is present; blocks code files in PRs targeting `results/*` branches
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document branch guards in CLAUDE.md

Add block-results-to-main to required status checks. Add run-complete
label step to the post-run protocol. Document branch-guards workflow
in the hooks section."
```

---

### Task 4: Test with a dummy PR

**Files:** None (verification only)

- [ ] **Step 1: Create a dummy results branch**

```bash
git checkout main
git checkout -b results/test-guard
git commit --allow-empty -m "test: dummy commit to verify branch guard"
git push -u origin results/test-guard
```

- [ ] **Step 2: Open a PR from the dummy branch to main**

```bash
gh pr create --base main --head results/test-guard \
  --title "test: verify branch guard blocks results merge" \
  --body "This PR tests that the branch-guards workflow blocks results->main merges. Should fail CI."
```

- [ ] **Step 3: Verify the `block-results-to-main` check fails**

```bash
sleep 30  # wait for CI to trigger
gh pr checks <PR_NUMBER>
```

Expected: `block-results-to-main` shows `fail`. Error message mentions the `run-complete` label.

- [ ] **Step 4: Add the label and verify it passes**

```bash
gh pr edit <PR_NUMBER> --add-label "run-complete"
sleep 30
gh pr checks <PR_NUMBER>
```

Expected: `block-results-to-main` shows `pass`.

- [ ] **Step 5: Clean up**

```bash
gh pr close <PR_NUMBER> --delete-branch
```

- [ ] **Step 6: Commit the spec and plan docs**

```bash
git checkout main
git add docs/superpowers/specs/2026-03-29-branch-guards-design.md \
        docs/superpowers/plans/2026-03-29-branch-guards.md
git commit -m "docs: add branch guards spec and implementation plan"
```

---

### Task 5: Add required status check in GitHub settings

**Files:** None (GitHub UI)

- [ ] **Step 1: Navigate to repo settings**

Go to: GitHub > Settings > Branches > Branch protection rules > `main` > Edit

- [ ] **Step 2: Add `block-results-to-main` to required checks**

Under "Require status checks to pass before merging", search for and add `block-results-to-main`.

Note: The check must have run at least once before it appears in the search. Task 4's test PR triggers it.

- [ ] **Step 3: Save changes**
