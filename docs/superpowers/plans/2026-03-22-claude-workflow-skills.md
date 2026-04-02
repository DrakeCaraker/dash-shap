# Claude Workflow Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three slash commands that eliminate the top three workflow bottlenecks: CI fix loops, multi-dimension repo audits, and safe refactoring with automatic rollback.

**Architecture:** Each feature is a markdown skill file in `.claude/commands/`. Skills are prompt instructions — they tell Claude *how* to execute a workflow when invoked. No library code changes. The three skills are fully independent and can be implemented in any order.

**Tech Stack:** Bash (`make ci`, `ruff`, `mypy`, `pytest`, `git`), Claude Code Agent tool (parallel audit subagents), Python (characterization tests).

---

## Existing Infrastructure to Know

Before implementing, understand what already exists:

- **`.claude/hooks/ruff-format-on-write.sh`** — PostToolUse hook that auto-runs `ruff format` on every `.py` file after Write/Edit. This means format failures are pre-empted in real time during sessions; `/ci-fix` rarely needs to fix format manually.
- **`.claude/hooks/ci-on-change.sh`** — Stop hook that runs `make lint typecheck test` when `dash_shap/` or `tests/` files are modified. This is a *reporting* hook — it does not fix anything.
- **`.githooks/pre-push`** — Blocks push on ruff lint/format failures. Most format issues are caught here before CI ever sees them.
- **`make ci`** — Runs `lint fmt-check typecheck test coverage` in sequence. Use this as the ground truth for "CI green."
- **`.claude/commands/`** — Where all slash command skill files live. Each `.md` file becomes a `/filename` command.

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `.claude/commands/ci-fix.md` | Autonomous CI fix loop skill |
| Create | `.claude/commands/audit.md` | Parallel repo audit dispatcher skill |
| Create | `.claude/commands/safe-refactor.md` | Test-gated refactoring skill |
| Modify | `CLAUDE.md` | Document all three new slash commands |

---

## Task 1: `/ci-fix` — Autonomous CI Fix Loop

**What it does:** Runs the CI suite step-by-step, fixes each failure type with the appropriate strategy, and loops until all checks pass or it gets stuck.

**Design decisions baked in:**
- Runs steps *incrementally* (`lint → fmt → typecheck → test-fast → coverage`) rather than `make ci` all at once, so each failure is isolated
- Ruff format failures are almost always pre-empted by the PostToolUse hook, but the skill handles them anyway
- `ruff check --fix .` handles ~80% of lint errors automatically; remaining errors need targeted manual fixes
- Mypy fixes go to source, never to `# type: ignore` (unless the error is in third-party stubs)
- Pytest fixes go to *implementation*, never to tests — modifying a test requires stopping and asking the user
- Stuck detection: if the same error message appears in consecutive iterations, stop and explain
- Hard cap: 5 iterations maximum

**Files:**
- Create: `.claude/commands/ci-fix.md`

- [ ] **Step 1: Write the skill file**

```markdown
# CI Fix Loop

Run the CI suite incrementally, fix all failures autonomously, and loop until green.

## Rules (read before starting)
- Fix one failure type at a time, then re-run the full suite
- NEVER modify test files to make tests pass — fix the implementation
- NEVER add `# type: ignore` to local code — fix the actual type issue
- NEVER use `ruff check --no-fix` or `--unsafe-fixes` without reading the diff first
- STOP and report to the user if:
  - The same error message appears in two consecutive iterations unchanged
  - You have reached iteration 5 without going green
  - Fixing a failure requires modifying a test file

## Process

### Iteration start
Run each CI step in order and record failures:

1. `ruff check .` — capture any lint errors
2. `ruff format --check .` — capture any format errors
3. `mypy dash_shap/ --ignore-missing-imports --no-error-summary` — capture type errors
4. `pytest -v -m "not slow"` — capture test failures
5. `pytest --cov=dash_shap --cov-report=term-missing --cov-fail-under=70` — capture coverage failures

### Fix strategies by failure type

**ruff lint errors:**
- Run `ruff check --fix .` first — this auto-fixes most issues
- Re-run `ruff check .` to see what remains
- For remaining errors: read the flagged file at the flagged line, apply the minimal manual fix
- Common remaining errors in this repo: line-too-long (wrap the long expression), E501 (split string)

**ruff format errors:**
- Run `ruff format .` — this is always safe and complete
- Note: the PostToolUse hook on Edit/Write already handles this in real time, so this should be rare

**mypy errors:**
- For each error: read the file at the flagged line number
- Add the minimal type annotation that resolves the error
- Prefer `-> None`, `-> dict[str, Any]`, `-> list[float]` etc. over `-> Any`
- If the error is in a third-party module stub, then `# type: ignore[import-untyped]` is acceptable
- Re-run mypy after each fix to verify it resolved

**pytest failures:**
- For each failing test: read the test AND the implementation it tests
- Understand what the test expects vs. what the implementation produces
- Fix the *implementation* to match the test's expectation
- If the test itself is clearly wrong (tests a removed API, references a deleted function), STOP and report — do not modify tests silently
- Re-run the failing test alone first: `pytest tests/test_<file>.py::test_<name> -v`
- Then re-run the full fast suite to check for regressions

**coverage failures (below 70%):**
- Read the coverage report — find which lines in `dash_shap/` are uncovered
- Check if there's already a test that *should* cover that path but doesn't reach it
- Add a minimal test to the relevant test file covering the uncovered branch
- Do NOT write test files that don't already exist — only extend existing test files

### After fixing all failures in an iteration
Re-run all 5 steps above from scratch. Do not assume a fix from earlier in the iteration is still valid.

### When CI is green
Report:
```
CI green ✓ (N iterations)
All checks passed: ruff lint, ruff format, mypy, pytest (fast), coverage
```

Then run `/commit` to commit the fixes.
```

- [ ] **Step 2: Verify the skill file is parseable**

```bash
wc -l .claude/commands/ci-fix.md
```
Expected: ~65-70 lines, no errors.

- [ ] **Step 3: Test the skill on a controlled failure**

Introduce a deliberate mypy error in a non-critical file:
```bash
# Add a type error to a utility function temporarily
echo "def _test_type_error() -> int: return 'not an int'" >> dash_shap/utils/__init__.py
```

Run `/ci-fix` and verify it:
1. Detects the mypy error
2. Reads the file at the flagged line
3. Fixes the type error (or removes the temporary line)
4. Re-runs mypy and confirms green
5. Does NOT modify any test files

Clean up if the skill didn't catch it:
```bash
# Remove the test line
git checkout -- dash_shap/utils/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/ci-fix.md
git commit -m "feat: add /ci-fix skill — autonomous CI fix loop"
```

---

## Task 2: `/audit` — Parallel Repo Audit

**What it does:** Dispatches four parallel subagents, each auditing one dimension of the codebase. Each writes structured findings with severity ratings. After all complete, merges into a single prioritized report.

**Design decisions baked in:**
- Four dimensions are truly independent — perfect for parallel agents
- Each agent gets a precise, scoped task prompt (not "look at the whole repo")
- Severity levels: `CRITICAL` (blocks release), `HIGH` (significant issue), `MEDIUM` (should fix), `LOW` (nice to have)
- Output goes to `docs/audit/YYYY-MM-DD-<dimension>.md` — date-stamped so audit history accumulates
- Agents are instructed to be conservative on sensitive data (flag any doubt as HIGH)
- The merge step uses a fifth sequential Agent call to produce the consolidated report

**Files:**
- Create: `.claude/commands/audit.md`

- [ ] **Step 1: Write the skill file**

```markdown
# Parallel Repo Audit

Dispatch four parallel subagents to audit different dimensions of the codebase. Each writes structured findings. Merge into a single prioritized report.

## Usage
Run `/audit` with no arguments for a full four-dimension audit.
Optionally: `/audit notebooks` or `/audit preprint` or `/audit sensitive` or `/audit release` for a single dimension.

## Process

### Get today's date
```bash
date +%Y-%m-%d
```
Use this as AUDIT_DATE in all output filenames.

### Spawn four agents IN PARALLEL using the Agent tool

Use a single message with four Agent tool calls to launch all four simultaneously.

---

**Agent 1: Notebook Health**

Prompt:
```
You are auditing the notebooks in this DASH-SHAP research repository. Your job is notebook health only — do not audit anything else.

Audit every .ipynb in notebooks/ (excluding notebooks/archive/):
1. Parse each notebook with Python's json module — flag any that fail to parse
2. For every code cell: check that it has a non-empty `id` field. Flag any without IDs as MEDIUM.
3. Check execution order: execution_count values should be sequential (1, 2, 3...) or all null. Out-of-order execution_count is a HIGH issue.
4. Check for figure references in markdown cells (![...](path)) — verify the referenced file exists. Missing figures are HIGH.
5. Check notebook file sizes — flag any > 2MB as HIGH (large outputs risk git bloat).
6. Check that checkpoint references in code cells (e.g., `load_checkpoint(`) resolve to files that exist in checkpoints/ or that the cell creates them.

Write your findings to: docs/audit/AUDIT_DATE-notebooks.md

Format each finding as:
- [SEVERITY] Notebook: notebooks/name.ipynb — description of issue

End the file with a summary line: "N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)"
```

---

**Agent 2: Preprint-Code Parity**

Prompt:
```
You are auditing the DASH-SHAP research preprint for claim-code parity. Your job is to verify that numeric claims and method descriptions in the LaTeX draft match the actual code and results.

1. Read paper/draft_v7_preprint.tex — extract every numeric claim (stability values, dataset names, parameter values like M=200, K=30, N_REPS)
2. For each numeric claim: check if it matches docs/BENCHMARK_RESULTS.md or results/tables/*.json
   - If the claim matches v6 results (N_REPS=20) but v7 results (N_REPS=50) are not yet available, flag as LOW (expected — v7 is in progress)
   - If the claim matches neither v6 nor v7, flag as HIGH
   - If no results file exists for the experiment, flag as MEDIUM
3. For each table in the paper: verify every row has a corresponding entry in results/tables/ or BENCHMARK_RESULTS.md
4. Read CLAUDE.md — verify the entry points, commands, and key results sections match the actual codebase state. Discrepancies are MEDIUM.

Write your findings to: docs/audit/AUDIT_DATE-preprint.md

Format each finding as:
- [SEVERITY] Claim: "quoted claim" — issue description (source: file:line)

End with: "N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)"
```

---

**Agent 3: Sensitive Data Scan**

Prompt:
```
You are performing a sensitive data scan of this research repository. Be conservative — flag anything that looks like it could be sensitive. False positives are acceptable; false negatives are not.

Scan all tracked files (git ls-files) for:
1. API keys, tokens, passwords, secrets: patterns like `sk-`, `Bearer `, `api_key =`, `password =`, `token =`, `secret =` (case insensitive). Flag as CRITICAL.
2. Absolute paths containing usernames or home directories in committed files (e.g., `/home/username/`, `/Users/username/`). Flag as MEDIUM unless in a .gitignore or example comment.
3. Run `git ls-files` and cross-check against .gitignore — flag any files that appear tracked but match .gitignore patterns. Flag as HIGH.
4. Scan for competitive intelligence: company names, unreleased product names, or reviewer names in committed docs. Check docs/ and paper/ carefully. Flag as HIGH.
5. Check for files that should never be committed: *.pkl in git history (`git log --all --name-only | grep .pkl`), *.env, credentials.json. Flag as CRITICAL if found in history.

Write your findings to: docs/audit/AUDIT_DATE-sensitive.md

Format each finding as:
- [SEVERITY] File: path/to/file:line — description

End with: "N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)"
```

---

**Agent 4: Release Readiness**

Prompt:
```
You are auditing this DASH-SHAP research repository for release readiness. Focus on packaging, documentation completeness, and public API consistency.

1. Packaging: Check that pyproject.toml or setup.py exists and has name, version, dependencies, and entry points defined. Missing version or dependencies are HIGH.
2. Public API docstrings: For every function and class in dash_shap/ that does NOT start with `_`, check that it has a docstring. Undocumented public API is MEDIUM.
3. CLAUDE.md accuracy: Verify that every command listed under "Running" actually works (the script/target exists). Verify the directory map matches the actual structure. Discrepancies are MEDIUM.
4. README.md (if it exists): Check that all links (`[text](url)` or `[text](path)`) resolve. Dead links are LOW.
5. Test coverage flag: Run `pytest --cov=dash_shap --cov-report=term-missing --cov-fail-under=70 -q 2>&1 | tail -5` — if coverage is below 70%, flag as HIGH.
6. Check that `python -c "import dash_shap; print(dash_shap.__version__)"` works. If no __version__ defined, flag as LOW.

Write your findings to: docs/audit/AUDIT_DATE-release.md

Format each finding as:
- [SEVERITY] Area: description (file:line if applicable)

End with: "N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)"
```

---

### After all four agents complete

Create docs/audit/ directory if it doesn't exist:
```bash
mkdir -p docs/audit
```

Dispatch a fifth Agent call (sequential, after the four complete) to merge findings:

Prompt:
```
Read these four audit result files:
- docs/audit/AUDIT_DATE-notebooks.md
- docs/audit/AUDIT_DATE-preprint.md
- docs/audit/AUDIT_DATE-sensitive.md
- docs/audit/AUDIT_DATE-release.md

Write a merged report to docs/audit/AUDIT_DATE-REPORT.md with this structure:

# Audit Report — AUDIT_DATE

## Summary Table
| Dimension | CRITICAL | HIGH | MEDIUM | LOW | Total |
|-----------|----------|------|--------|-----|-------|
| Notebooks | ... |
| Preprint | ... |
| Sensitive Data | ... |
| Release Readiness | ... |
| **Total** | ... |

## Prioritized Fix List
Group all findings by severity. Within each severity, order by dimension.

### CRITICAL (fix before any push)
[list all CRITICAL findings]

### HIGH (fix before release)
[list all HIGH findings]

### MEDIUM (fix before TMLR submission)
[list all MEDIUM findings]

### LOW (nice to have)
[list all LOW findings]
```

Report completion to the user with: "Audit complete. Report at docs/audit/AUDIT_DATE-REPORT.md — N total issues (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)"
```

- [ ] **Step 2: Create the docs/audit/ directory**

```bash
mkdir -p docs/audit
echo "# Audit outputs — date-stamped files written here by /audit" > docs/audit/README.md
```

- [ ] **Step 3: Verify the skill is invocable**

Run `/audit` and confirm:
1. Four Agent calls fire simultaneously (check that all four output files start appearing)
2. Each output file has the expected structure (heading, findings, summary line)
3. The merged report appears after all four complete
4. Total runtime is faster than running four sequential agents

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/audit.md docs/audit/README.md
git commit -m "feat: add /audit skill — parallel four-dimension repo audit"
```

---

## Task 3: `/safe-refactor` — Test-Gated Refactoring

**What it does:** Three-phase workflow for safe refactoring. Phase 1 writes characterization tests capturing current behavior. Phase 2 makes one change at a time, rolling back with `git checkout -- <file>` on test failure. Phase 3 deletes the characterization tests (they captured what IS, not what SHOULD BE — they're temporary scaffolding).

**Design decisions baked in:**
- Works on a new branch `refactor/<target>` so the current branch is never polluted
- "Characterization tests" capture *current behavior* — they pass even if the current behavior is wrong. Their job is to catch regressions, not validate correctness.
- Rollback uses `git checkout -- <file>` (revert one file) not `git stash` (revert everything) — this allows surgical rollback of a single bad change while keeping other already-committed changes
- If the same change fails twice, stop — this indicates a design conflict requiring human judgment
- Characterization tests are committed to the refactoring branch but deleted at cleanup — they never merge to main
- The skill explicitly avoids `git reset --hard` or any destructive operation

**Files:**
- Create: `.claude/commands/safe-refactor.md`

- [ ] **Step 1: Write the skill file**

```markdown
# Safe Refactor

Test-gated refactoring: characterize current behavior → refactor one change at a time → rollback automatically on failure.

## Usage
`/safe-refactor <target>` where target is a module path, file path, or description.
Example: `/safe-refactor dash_shap/utils/thread_budget.py`
Example: `/safe-refactor checkpoint system`

## Hard Rules
- NEVER make two changes in one step — one change, run tests, commit or rollback
- NEVER modify characterization tests to make them pass — if a test fails after your change, rollback the change
- NEVER use `git reset --hard` — use `git checkout -- <file>` to revert a specific file
- STOP and explain to the user if:
  - The same change causes the same test failure twice
  - Characterization tests fail on the baseline (before any refactoring)
  - You need to change a public API signature (this requires user approval)

## Phase 1: Characterize (read-only — no source changes yet)

### 1A. Create a new branch
```bash
git checkout -b refactor/<target-name-slugified>
```

### 1B. Read the target thoroughly
- Read every file in scope
- List all public functions/classes (anything not starting with `_`)
- Note the inputs, outputs, and side effects of each

### 1C. Write characterization tests
Create `tests/test_<target>_characterization.py`. These tests:
- Import from the target module
- Call every public function with representative inputs
- Assert on the *current* outputs, even if behavior seems questionable
- Cover happy path + at least one edge case per function
- Are NOT meant to be good tests — they're a regression tripwire

Example pattern:
```python
"""Characterization tests for <target> — temporary, deleted after refactoring."""
import pytest

def test_<function>_current_behavior():
    from <module> import <function>
    result = <function>(<representative_input>)
    assert result == <actual_current_output>  # captured from running the function

def test_<function>_edge_case():
    from <module> import <function>
    result = <function>(<edge_case_input>)
    assert result == <actual_current_output>
```

To get actual current outputs, run:
```bash
python3 -c "from <module> import <function>; print(<function>(<input>))"
```

### 1D. Run characterization tests and confirm they all pass
```bash
pytest tests/test_<target>_characterization.py -v
```
**All must pass.** If any fail, fix the test (your expected value was wrong), not the code.

### 1E. Commit the characterization tests
```bash
git add tests/test_<target>_characterization.py
git commit -m "test: add characterization tests for <target> refactor"
```

## Phase 2: Refactor (one change at a time)

For each planned change:

### 2A. Describe the change (one sentence, out loud in your response)
Example: "Rename `_rp_mod` variable to `_runner_module` for clarity."

### 2B. Apply the change to ONE file only

### 2C. Run characterization tests
```bash
pytest tests/test_<target>_characterization.py -v
```

### 2D-pass. If tests pass → commit
```bash
git add <modified_file>
git commit -m "refactor: <one-sentence description>"
```

### 2D-fail. If tests fail → rollback immediately
```bash
git checkout -- <modified_file>
```
Then explain what broke: "The change '<description>' broke <N> characterization tests. Root cause: <explanation>. Trying alternative approach: <alternative>."

Apply the alternative, re-run tests. If it fails again with the same error, STOP:
"This change conflicts with existing behavior in a way I cannot resolve without changing the public interface. Human decision needed: [describe the conflict]."

### Repeat 2A-2D for every planned change.

## Phase 3: Cleanup

### 3A. Run the full fast test suite to confirm nothing regressed
```bash
make test-fast
```

### 3B. Delete the characterization tests
```bash
git rm tests/test_<target>_characterization.py
git commit -m "chore: remove characterization tests for <target> (refactor complete)"
```

### 3C. Report completion
"Safe refactor complete on branch `refactor/<target>`. N changes applied, M rollbacks. Ready to push and open a PR."

Then run `/commit` if there are any uncommitted changes, followed by offering to push and open a PR.
```

- [ ] **Step 2: Test the skill on a small target**

Use `dash_shap/utils/thread_budget.py` as a test target — it's small (one file, clear API, well-tested already):

```bash
# Verify the target has existing tests
pytest tests/test_thread_budget.py -v
```
Expected: all tests pass.

Run `/safe-refactor dash_shap/utils/thread_budget.py` and verify Phase 1 completes correctly:
1. A new branch `refactor/thread-budget` is created
2. `tests/test_thread_budget_characterization.py` is written and committed
3. All characterization tests pass on the baseline

Then verify Phase 2 with one trivial change (rename a variable, add a comment), confirm test pass, commit, and rollback is not needed.

Run Phase 3 cleanup and confirm the characterization test file is deleted.

```bash
# Clean up the test branch after verification
git checkout fix/pin-blas-threads-before-import  # or main
git branch -d refactor/thread-budget
```

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/safe-refactor.md
git commit -m "feat: add /safe-refactor skill — test-gated refactoring with auto-rollback"
```

---

## Task 4: Update CLAUDE.md

Document all three new skills under the existing `## Slash Commands` section.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add entries to the Slash Commands section**

Find the existing slash commands table in CLAUDE.md and add three rows:

```markdown
- `/ci-fix` — autonomous CI fix loop: runs ruff/mypy/pytest, fixes failures by type, loops until green (max 5 iterations)
- `/audit` — parallel four-dimension repo audit: notebooks, preprint parity, sensitive data, release readiness → merged report in `docs/audit/`
- `/safe-refactor <target>` — test-gated refactoring: writes characterization tests, refactors one change at a time, auto-rollbacks on failure
```

- [ ] **Step 2: Verify CLAUDE.md still parses correctly**

```bash
python3 -c "
with open('CLAUDE.md') as f:
    content = f.read()
assert '/ci-fix' in content
assert '/audit' in content
assert '/safe-refactor' in content
print('OK: all three skills documented in CLAUDE.md')
"
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document /ci-fix, /audit, /safe-refactor in CLAUDE.md"
```

---

## End-to-End Verification

After all four tasks:

```bash
# All three skill files exist
ls -la .claude/commands/ci-fix.md .claude/commands/audit.md .claude/commands/safe-refactor.md

# All documented in CLAUDE.md
grep -E "ci-fix|/audit|safe-refactor" CLAUDE.md

# docs/audit/ exists
ls docs/audit/

# No unintended changes to library code
git diff main -- dash_shap/ tests/
# Expected: empty (no library changes)
```

---

## What Is NOT Changing

- No changes to `dash_shap/` library code
- No changes to `tests/` (permanent test files) — only the temporary characterization test file during a safe-refactor session
- No changes to CI configuration (`.github/workflows/ci.yml`)
- No changes to existing hooks or settings

---

## Branch Strategy

These are all small, independent changes. Options:
1. **One PR** — `feat/claude-workflow-skills` — all three skills in one branch (recommended, since they're all meta-tooling)
2. **Three PRs** — one per skill — cleaner history but unnecessary overhead

Recommended: implement all three on one branch `feat/claude-workflow-skills`, open one PR.
