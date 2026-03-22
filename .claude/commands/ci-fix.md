# CI Fix Loop

Autonomously fix CI failures in a loop until all checks pass or a stop condition is reached.

## Loop Behavior

Run the following 4 steps in sequence, then repeat the loop. Maximum 5 iterations:

1. **Ruff lint check**: `make lint`
2. **Ruff format check**: `make fmt-check`
3. **Mypy type check**: `make typecheck`
4. **Fast pytest suite**: `make test-fast`

After each complete iteration, immediately re-run all 4 steps from scratch.

Once all 4 steps pass in a single iteration (loop goes green), run the final coverage check **one time only**:

5. **Coverage check**: `make coverage` (one-time final step after loop is green)

## Rules (Do NOT violate)

- **Never modify test files to fix a failing test** — always fix the implementation instead. (Adding new tests to an existing test file to cover uncovered lines for a coverage failure is permitted.)
- **Never add `# type: ignore`** to local code — fix the actual type issue (only acceptable for third-party stubs)
- **Isolate failures** — run each step individually so you can pinpoint which check is failing
- **Use targeted fixes** — apply the minimal change needed to address each error

## Stop Conditions (MUST STOP immediately)

1. **Repeated error**: Same error appears in two consecutive iterations unchanged
2. **Max iterations reached**: Iteration 5 completes without going green
3. **Test file modification needed**: Fixing a failure requires modifying a test file — STOP and report to user

## Fix Strategies by Failure Type

### Ruff Lint Failures

1. Run `ruff check --fix .` first (handles ~80% automatically)
2. If failures remain, read each flagged file at the line numbers shown
3. Apply minimal manual fixes for remaining issues (typically unused imports, undefined names, or style violations)
4. Re-run `make lint` to verify

### Ruff Format Failures

1. Run `ruff format .` (safe and complete)
2. Note: The PostToolUse `ruff-format-on-write.sh` hook auto-formats `.py` files after every Edit/Write, so format failures are rare in practice.
3. Re-run `make fmt-check` to verify

### Mypy Type Failures

For each mypy error:

1. Read the file at the flagged line number
2. Identify the missing or incorrect type annotation
3. Add the minimal type annotation needed — prefer specific types like:
   - `-> None` (for functions with no return)
   - `-> dict[str, Any]` (for dicts)
   - `-> list[float]` (for lists)
   - `-> tuple[int, str]` (for tuples)
   - over generic `-> Any`
4. Do not add `# type: ignore` comments
5. Re-run `make typecheck` to verify

### Pytest Failures

1. Run the failing test alone first: `pytest tests/test_X.py::test_name -v`
2. Read both the test file AND the implementation it tests
3. Identify the root cause in the implementation (not the test)
4. Fix the implementation with minimal changes
5. Re-run just that test to verify the fix
6. Re-run the full `make test-fast` suite to check for regressions

### Coverage Below 70%

1. Run `make coverage` and capture the coverage report
2. Find uncovered lines in `dash_shap/` (implementation only, not tests)
3. Add a minimal test to an EXISTING test file that exercises the uncovered line
4. Do NOT create new test files
5. Re-run `make coverage` to verify

## When Green

Once all 4 loop checks pass in a single iteration (ruff lint, ruff format, mypy, pytest fast), then coverage passes in the final one-time check:

1. Report: "CI green ✓ (N iterations)"
2. Offer to run `/commit` if the user wants to commit the fixes
3. Do not push; wait for user direction

## Iteration Summary

After each iteration, show:

- Iteration number (1-5)
- Which checks passed ✓ and which failed ✗
- Brief summary of fixes applied in this iteration
- Estimated progress toward green

If no progress is made in an iteration, flag as a potential stop condition.
