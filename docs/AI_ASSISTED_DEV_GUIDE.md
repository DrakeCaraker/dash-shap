# AI-Assisted Development Field Guide

A field guide for AI-assisted development, distilled from 200+ PRs on a real ML research project. Zero dependencies beyond git and a terminal. Tool-agnostic and language-agnostic in principles. Each section points to the next — start at the Quick Start and add complexity only when you feel friction.

---

## PART 1: QUICK START

**Prerequisites:** git, a terminal, a text editor. That's it.

Run these three blocks from your project root to get baseline safety.

### Block 1: .gitignore

```bash
cat << 'EOF' > .gitignore
# -- Universal --
.DS_Store
*.log
*.tmp

# -- Python --
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
*.pkl
checkpoints/

# -- JavaScript/TypeScript --
# node_modules/
# .next/
# .nuxt/

# -- Go --
# vendor/  (if not vendoring)

# -- Rust --
# target/

# -- Data/models (all ecosystems) --
*.h5
*.pt
*.onnx
*.parquet
EOF
```

### Block 2: Pre-push hook

```bash
mkdir -p .githooks && cat << 'HOOK' > .githooks/pre-push && chmod +x .githooks/pre-push
#!/usr/bin/env bash
set -euo pipefail

BLOCKED_EXTENSIONS="pkl|h5|pt|onnx|parquet"
MAX_BYTES=$((10 * 1024 * 1024))  # 10 MB
ERRORS=0

# Block pushes directly to main/master
current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
  echo "ERROR: Direct push to $current_branch is blocked. Use a feature branch."
  exit 1
fi

# Parse stdin: each line is "local_ref local_sha remote_ref remote_sha"
while read -r _ local_sha _ remote_sha; do
  if [[ "$local_sha" == "0000000000000000000000000000000000000000" ]]; then
    continue  # branch deletion
  fi
  if [[ "$remote_sha" == "0000000000000000000000000000000000000000" ]]; then
    range="$local_sha"
  else
    range="$remote_sha..$local_sha"
  fi

  # Check each file being pushed
  for file in $(git diff --name-only --diff-filter=ACM "$range" 2>/dev/null); do
    # Blocked extensions
    if echo "$file" | grep -qE "\\.(${BLOCKED_EXTENSIONS})$"; then
      echo "BLOCKED: $file — binary/data file not allowed in repo"
      ERRORS=1
    fi
    # Size check using the blob in the commit (not the working tree)
    blob_sha=$(git ls-tree -r "$local_sha" -- "$file" 2>/dev/null | awk '{print $3}')
    if [[ -n "$blob_sha" ]]; then
      size=$(git cat-file -s "$blob_sha")
      if (( size > MAX_BYTES )); then
        echo "BLOCKED: $file is $(( size / 1024 / 1024 ))MB (limit: $(( MAX_BYTES / 1024 / 1024 ))MB)"
        ERRORS=1
      fi
    fi
  done
done

# Drift warning (non-blocking)
if git rev-parse --verify origin/main &>/dev/null; then
  behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
  if (( behind > 0 )); then
    echo "WARNING: Branch is $behind commit(s) behind origin/main. Consider rebasing."
  fi
fi

if (( ERRORS )); then
  echo "Push blocked. Fix the issues above and retry."
  exit 1
fi
HOOK
git config core.hooksPath .githooks
echo "Pre-push hook installed."
```

### Block 3: Project instructions file skeleton

```bash
cat << 'EOF' > CLAUDE.md
# Project Name

TODO: One-sentence description of the project.

## Non-Negotiable Rules

1. Never push directly to main. Always use a feature branch and open a PR.
2. Keep commits atomic — one concern per commit.
3. Read before planning — verify by reading code, not guessing from filenames.

## Directory Map

TODO: List your top-level directories and what they contain.

## Key Entry Points

TODO: List the 3-5 files someone needs to understand the project.

## Running

```bash
TODO: List your test, lint, format, and build commands here.
```

## Do NOT

TODO: List things that are easy to get wrong in this codebase.
EOF
echo "Project instructions file created. Fill in the TODOs."
```

Fill in the TODOs. You're at Level 1.

**Next -->** Level 2 when you want automation.

---

## PART 2: LEVELS

### Level 1: Safety (protect the repo)

**Principle:** Make dangerous states structurally impossible, not just discouraged.

**Pattern:** Three layers of protection — a project instructions file that tells humans and AI tools what not to do, git hooks that enforce it mechanically, and a `.gitignore` that prevents accidents silently.

**Recipe: Manual**

1. Create a project instructions file at the repo root. The filename depends on your tool:

   | Tool | File |
   |------|------|
   | Claude Code | `CLAUDE.md` |
   | Cursor | `.cursorrules` |
   | GitHub Copilot | `.github/copilot-instructions.md` |
   | Windsurf | `.windsurfrules` |
   | No AI tool | `CONTRIBUTING.md` |

   Content is the same regardless of filename: non-negotiable rules, directory map, entry points, commands, things to avoid. See Block 3 above.

2. Install the pre-push hook from Block 2. It blocks binary files, direct pushes to main, and files over 10MB.

3. Write an onboarding command that installs deps, activates hooks, and verifies tools:
   ```bash
   # In your Makefile:
   setup:
   	pip install -e ".[dev]"
   	git config core.hooksPath .githooks
   	@echo "Ready."
   ```

**Recipe: With AI tools** — The AI tool reads your project instructions file at session start. The rules you write there become the AI's constraints. Spend time on the "Do NOT" section — that's where most value comes from.

**Escape hatch:** Solo personal projects with no collaborators may skip branching if the overhead exceeds the value. Keep the hooks and `.gitignore` regardless.

**Next -->** Level 2.

---

### Level 2: Automation (eliminate repetitive friction)

**Principle:** If you've done it manually 3 times, automate it.

**Pattern:** Four automations cover 80% of session friction: format on save, status on start, standardized commands, and safe commits.

**Recipe: Manual**

1. **Auto-format on save.** Add a pre-commit hook that runs your formatter:
   ```bash
   cat << 'HOOK' > .githooks/pre-commit && chmod +x .githooks/pre-commit
   #!/usr/bin/env bash
   # Format only staged files
   staged=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
   if [[ -n "$staged" ]]; then
     echo "$staged" | xargs ruff format --quiet
     echo "$staged" | xargs git add
   fi
   HOOK
   ```

2. **Status on start.** Add a shell alias that shows branch, drift, and uncommitted changes:
   ```bash
   alias ws='echo "Branch: $(git branch --show-current)" && git status -s && echo "Behind main: $(git rev-list --count HEAD..origin/main 2>/dev/null || echo "?") commits"'
   ```

3. **Standardized command runner.** Pick one: Makefile, npm scripts, `just`, `taskfile`. Put every command anyone needs to run behind a short target:
   ```makefile
   lint:    ruff check .
   fmt:     ruff format .
   test:    pytest
   test-fast: pytest -m "not slow"
   typecheck: mypy .
   ```

4. **Safe commit workflow.** Before committing, always: check `git status`, review `git diff --staged`, stage files by name (not `git add -A`).

**Recipe: With AI tools** — Claude Code: `PostToolUse` hook runs formatter after file writes. `SessionStart` hook shows workspace status. Slash commands (`.claude/commands/commit.md`) wrap the safe commit workflow. Other tools: configure format-on-save in editor settings.

**Next -->** Level 3.

---

### Level 3: Domain-Specific (adapt to your work type)

**Principle:** Generic tools need project-specific guardrails.

**Pattern:** Six practices that prevent domain-specific classes of errors.

**Recipe: Manual**

1. **Protect canonical outputs.** Some files are empirical records, not source code — notebooks with published results, frozen benchmark tables, signed reports. Document which files are canonical in your project instructions file. Before clearing or overwriting, check the list.

2. **Centralize configuration.** Keep parameters that appear in multiple files in one authoritative location. In Python, a config dict at the top of your main module or a dedicated `config.py`. In other languages, a config file or constants module. The point: when a parameter changes, you change it in one place.

3. **Tag outputs with provenance.** Every generated result file should record what produced it:
   ```python
   import datetime, subprocess
   meta = {
       "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "git_commit": subprocess.check_output(
           ["git", "rev-parse", "HEAD"]
       ).decode().strip(),
       "config": {"M": 200, "K": 30, "epsilon": 0.08},
   }
   ```
   Embed this in your output format (JSON `_meta` key, CSV header comment, HDF5 attribute).

4. **Lock dependencies.** Pin your environment so results are reproducible: `pip freeze > requirements.txt`, `conda env export`, `npm shrinkwrap`, `cargo.lock`. Commit the lockfile.

5. **Separate fast and slow tests.** Mark slow tests (integration, GPU, large data) so you can run fast feedback loops locally:
   ```python
   import pytest
   @pytest.mark.slow
   def test_full_pipeline(): ...
   ```
   ```makefile
   test-fast: pytest -m "not slow"
   ```

6. **Separate code from data.** For long-running experiments, use data-only branches. The protocol: create a results branch from main, commit only data files (JSON, figures, environment info) to it, never commit code changes there. Code fixes go through PR to main, then cherry-pick to the results branch. After the run, merge data back to main via PR.

**Recipe: With AI tools** — Claude Code: `PreToolUse` hook warns when editing files over a size threshold (catches accidental notebook output commits). Slash commands can automate provenance insertion and config sync checks.

**Escape hatch:** EDA-only and notebook-only projects may skip config centralization, test tiers, and the data branch protocol. Keep provenance tagging and dependency locking regardless.

**Next -->** Level 4.

---

### Level 4: Resilience (recover from failures gracefully)

**Principle:** Assume things will break. Build recovery into the workflow.

**Pattern:** Five practices that turn failures into bounded, recoverable events.

**Recipe: Manual**

1. **Automated CI fix loop.** When CI fails, run: check, diagnose, fix, recheck. Repeat up to 5 times. Hard rules:
   - Never modify tests to make them pass.
   - Never suppress type errors with `# type: ignore`.
   - Never mask regressions — if a test passed before your changes and now fails, that is your bug.
   - Stop if the same error repeats twice.

   ```bash
   for i in 1 2 3 4 5; do
     echo "--- Attempt $i ---"
     make lint && make typecheck && make test-fast && echo "All green." && break
     echo "Failures found. Fixing..."
     # Fix the specific failure, then loop.
   done
   ```

2. **Multi-dimension audit.** Periodically run parallel checks across dimensions: output health (are generated files current?), doc-code parity (do docs match implementation?), sensitive data scan (any credentials or PII?), release readiness (version bumped? changelog updated?). One script, four reports.

3. **Test-gated refactoring.** Before refactoring, write characterization tests that capture current behavior — including known bugs. Apply one change at a time. Run tests after each change. Auto-rollback on failure (`git checkout -- <file>`). Characterization tests assert what the code does now, not what it should do. This prevents refactoring from silently changing behavior.

4. **Session-end CI gate.** Before ending a work session, run `make lint && make typecheck && make test-fast`. Do not leave broken code on a branch overnight.

5. **Context preservation on compression.** During long sessions, AI tools may compress their working context. Before that happens, save your in-progress state: the current goal, files being modified, and any decisions made. A text file or commit message works.

**Recipe: With AI tools** — Claude Code: `/ci-fix` slash command runs the automated fix loop. `/audit` runs the multi-dimension audit. `/safe-refactor` runs test-gated refactoring. `Stop` hook runs the session-end gate. `PreCompact` hook reminds you to save context.

**Next -->** Level 5.

---

### Level 5: Institutional Knowledge (cross-session memory)

**Principle:** The best corrections are the ones you never have to give twice.

**Pattern:** Persist the right things, in the right format, and prune actively.

**What to persist:**
- User role and expertise level
- Hardware constraints (GPU availability, memory limits)
- Workflow preferences (branching conventions, commit style)
- External resource locations (cloud endpoints, data sources)
- Corrections with reasoning ("don't do X because Y")

**What NOT to persist:**
- Code patterns (read the code — it changes)
- Git history (use `git log`)
- File paths (use search — they move)

**Feedback memories are most valuable.** When you correct a mistake, save why so it does not recur:

```markdown
---
type: feedback
date: 2026-03-15
---
# Never reuse branches across unrelated work

When context changes mid-session, create a new descriptively-named branch.
Stacking unrelated work on one branch makes PRs unreviewable and reverts dangerous.
```

**Periodic health assessment.** Once a month (or every 20 sessions), review your project instructions file and memories. Delete stale entries. Check if recurring friction points have been addressed structurally.

**Friction detection.** When the same correction happens 3+ times across sessions, it indicates a structural gap. The fix is not another memory — it is a hook, a lint rule, or a project instructions file update that prevents the friction mechanically.

**Recipe: With AI tools** — Claude Code: memory files in `.claude/` persist across sessions. Other tools: use a `CONVENTIONS.md` or similar file that the tool reads at startup.

---

## PART 3: ANTI-PATTERNS

1. **Don't let AI tools stage all files.**
   `git add -A` and `git add .` can commit secrets, binaries, and generated files. Stage by name.
   *Escape hatch:* Acceptable in a fresh repo with a comprehensive `.gitignore` and no sensitive files.

2. **Don't stack unrelated work on one branch.**
   Mixed branches produce unreviewable PRs and make selective reverts impossible.
   *Escape hatch:* Trivial one-line fixes (typos, comment updates) can piggyback if the PR author notes them.

3. **Don't skip pre-push checks.**
   The five minutes you save will cost you thirty in CI debugging.
   *Escape hatch:* Draft PRs for WIP sharing where CI failure is expected and communicated.

4. **Don't clear outputs without knowing if they're canonical.**
   Notebook outputs and result files may be empirical records that took hours to generate.
   *Escape hatch:* Scratch and development notebooks can be cleared freely — just know which are which.

5. **Don't run unlimited background processes.**
   Parallel test runs and build processes compete for resources and produce interleaved, unreadable output.
   *Escape hatch:* CI systems with isolated runners can parallelize freely.

6. **Don't trust file names — read the code.**
   `utils.py` might contain core business logic. `config.py` might be dead code. Verify before planning.
   *Escape hatch:* Well-maintained projects with strict naming conventions and code review can rely on names for navigation.

7. **Don't modify tests to fix failures.**
   If a test fails after your change, your change broke something. Fix the implementation, not the assertion.
   *Escape hatch:* Tests that assert on non-deterministic output (timestamps, random values) may need updating when the contract genuinely changes.

8. **Don't suppress type errors with ignore comments.**
   `# type: ignore` hides bugs. Fix the type signature or add a proper overload.
   *Escape hatch:* Third-party libraries with missing or incorrect stubs sometimes require targeted ignores with specific error codes.

---

## PART 4: APPENDICES

### Appendix A: Project Instructions File by Tool

| Tool | File | Notes |
|------|------|-------|
| Claude Code | `CLAUDE.md` | Read at session start; supports markdown |
| Cursor | `.cursorrules` or `.cursor/rules/*.mdc` | `.mdc` files for context-specific rules |
| GitHub Copilot | `.github/copilot-instructions.md` | Scoped to GitHub Copilot features |
| Windsurf | `.windsurfrules` | Similar format to `.cursorrules` |
| Aider | `CONVENTIONS.md` | Referenced in `.aider.conf.yml` |
| No AI tool | `CONTRIBUTING.md` or `README.md` | For human contributors |

Content is the same regardless of filename: rules, directory map, entry points, commands, things to avoid. Some teams maintain one canonical file and symlink it to the other paths.

### Appendix B: Recipes by Language

**Python**
- Format: `ruff format .`
- Lint: `ruff check .`
- Typecheck: `mypy .`
- Test: `pytest`
- Lock: `pip freeze > requirements.txt` or `uv pip compile`

**JavaScript / TypeScript**
- Format: `prettier --write .`
- Lint: `eslint .`
- Typecheck: `tsc --noEmit`
- Test: `npm test` (jest, vitest, mocha)
- Lock: `package-lock.json` (npm) or `yarn.lock`

**Go**
- Format: `gofmt -w .`
- Lint: `go vet ./...`
- Typecheck: (compiler does this)
- Test: `go test ./...`
- Lock: `go.sum`

**Rust**
- Format: `rustfmt`
- Lint: `cargo clippy`
- Typecheck: (compiler does this)
- Test: `cargo test`
- Lock: `Cargo.lock`

**R**
- Format: `styler::style_dir(".")`
- Lint: `lintr::lint_dir(".")`
- Typecheck: (not standard in R)
- Test: `testthat::test_dir("tests")`
- Lock: `renv.lock`

### Appendix C: AI Tool Automation Features

**Claude Code**
- `PostToolUse` hooks: run commands after file writes (e.g., auto-format)
- `PreToolUse` hooks: warn before risky operations (e.g., editing large files)
- `SessionStart` / `Stop` hooks: status checks on open, CI gate on close
- `PreCompact` hooks: save context before compression
- Slash commands: `.claude/commands/*.md` define reusable workflows
- Memory: `.claude/` directory persists user-specific notes across sessions
- Settings: `.claude/settings.json` for project-level configuration

**Cursor**
- `.cursorrules`: project-level instructions read by the AI
- `.cursor/rules/*.mdc`: context-specific rule files
- Format-on-save: configure in editor settings

**GitHub Copilot**
- `.github/copilot-instructions.md`: project-level instructions
- VS Code `settings.json`: format-on-save, linter integration

### Appendix D: Full Pre-Push Hook (Production Version)

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
BLOCKED_EXTENSIONS="pkl|h5|pt|onnx|parquet|bin|weights"
MAX_BYTES=$((10 * 1024 * 1024))
PROTECTED_BRANCHES="main master"
ERRORS=0

# --- Block protected branches ---
current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
for protected in $PROTECTED_BRANCHES; do
  if [[ "$current_branch" == "$protected" ]]; then
    echo "ERROR: Direct push to '$protected' is blocked."
    echo "  Fix: git checkout -b feat/your-topic && git push -u origin feat/your-topic"
    exit 1
  fi
done

# --- Parse pushed refs from stdin ---
while read -r _ local_sha _ remote_sha; do
  # Skip branch deletions
  if [[ "$local_sha" == "0000000000000000000000000000000000000000" ]]; then
    continue
  fi

  # Determine diff range
  if [[ "$remote_sha" == "0000000000000000000000000000000000000000" ]]; then
    range="$local_sha"
  else
    range="$remote_sha..$local_sha"
  fi

  # Check each added or modified file
  for file in $(git diff --name-only --diff-filter=ACM "$range" 2>/dev/null); do
    # Blocked extensions
    if echo "$file" | grep -qiE "\\.(${BLOCKED_EXTENSIONS})$"; then
      echo "BLOCKED: $file — binary/data file not allowed in repo."
      echo "  Fix: add the file to .gitignore and remove with git rm --cached $file"
      ERRORS=1
    fi

    # Size check on the actual blob in the commit
    blob_sha=$(git ls-tree -r "$local_sha" -- "$file" 2>/dev/null | awk '{print $3}')
    if [[ -n "$blob_sha" ]]; then
      size=$(git cat-file -s "$blob_sha")
      if (( size > MAX_BYTES )); then
        size_mb=$(( size / 1024 / 1024 ))
        limit_mb=$(( MAX_BYTES / 1024 / 1024 ))
        echo "BLOCKED: $file is ${size_mb}MB (limit: ${limit_mb}MB)."
        echo "  Fix: use Git LFS or move the file out of the repo."
        ERRORS=1
      fi
    fi
  done
done

# --- Lint check (non-blocking warning) ---
if command -v ruff &>/dev/null; then
  if ! ruff check --quiet . 2>/dev/null; then
    echo "WARNING: ruff found lint issues. Run 'ruff check .' to review."
  fi
elif command -v eslint &>/dev/null; then
  if ! eslint --quiet . 2>/dev/null; then
    echo "WARNING: eslint found lint issues."
  fi
fi

# --- Type check (non-blocking warning) ---
if command -v mypy &>/dev/null; then
  if ! mypy --no-error-summary . 2>/dev/null; then
    echo "WARNING: mypy found type errors. Run 'mypy .' to review."
  fi
elif command -v tsc &>/dev/null; then
  if ! tsc --noEmit 2>/dev/null; then
    echo "WARNING: tsc found type errors."
  fi
fi

# --- Drift warning (non-blocking) ---
if git rev-parse --verify origin/main &>/dev/null; then
  behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
  if (( behind > 0 )); then
    echo "WARNING: Branch is $behind commit(s) behind origin/main."
    echo "  Fix: git fetch origin && git rebase origin/main && git push --force-with-lease"
  fi
fi

# --- Final verdict ---
if (( ERRORS )); then
  echo ""
  echo "Push blocked. Fix the issues above and retry."
  exit 1
fi
```

---

## CLOSING

The best configurations are grown, not designed. Start with the Quick Start. Add levels as you discover friction. The guide will be here when you need it.

*Distilled from the DASH-SHAP project -- 200+ PRs, 13 slash commands, 6 hooks, 11 CI jobs, across 10 weeks of AI-assisted development.*
