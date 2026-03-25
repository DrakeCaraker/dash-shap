# Project Bootstrap

Set up AI-assisted development infrastructure for any project. Detects what exists, asks targeted questions, creates what's missing.

## Step 1: Detect current state

Check which infrastructure exists:
- `ls CLAUDE.md .cursorrules .github/copilot-instructions.md .windsurfrules 2>/dev/null` — project instructions file
- `ls .githooks/pre-push .git/hooks/pre-push 2>/dev/null` — pre-push hook
- `ls Makefile justfile taskfile.yml package.json 2>/dev/null` — command runner
- `ls .claude/settings.json 2>/dev/null` — Claude Code hooks
- `ls .gitignore 2>/dev/null` — gitignore
- `git rev-parse --is-inside-work-tree 2>/dev/null` — is this a git repo?
- Scan for project type: `ls *.py pyproject.toml setup.py 2>/dev/null` (Python), `ls package.json 2>/dev/null` (JS/TS), `ls Cargo.toml 2>/dev/null` (Rust), `ls go.mod 2>/dev/null` (Go), `ls *.R DESCRIPTION 2>/dev/null` (R)

Report what level the project is at:
- Level 1: Has instructions file + git hooks + .gitignore
- Level 2: Has auto-formatting + command runner + session hooks
- Level 3+: Domain-specific (check for notebooks, experiment configs, CI workflow)

## Step 2: Determine what to create next

If Level 0 (nothing exists), proceed to Step 3 (full setup).
If Level 1 exists but not Level 2, offer to add: format-on-write hook, session-start hook, Makefile, /commit command.
If Level 2 exists, run a quick health check and recommend Level 3+ items that apply.

Always tell the user what level they're at and what the next level adds.

## Step 3: Ask questions (Level 1 setup)

Ask these questions EXACTLY as formatted. Show all examples. Wait for each answer.

### Question 1: Project description
```
What does this project do? (One sentence for the project instructions file)

Examples:
  o "XGBoost ensemble for stable feature importance under collinearity"
  o "FastAPI backend for the customer analytics dashboard"
  o "ETL pipeline ingesting Salesforce data into BigQuery nightly"
  o "PyTorch image classifier for manufacturing defect detection"
  o "Shared utility library for the data science team's feature engineering"

Your answer:
```

### Question 2: Blocked file types
```
What binary or generated files should NEVER be committed?
(These get added to .gitignore and blocked by the pre-push hook)

Common patterns by project type:
  ML (sklearn):    .pkl, .joblib, checkpoints/
  ML (PyTorch):    .pt, .pth, .ckpt, .safetensors, checkpoints/
  ML (TensorFlow): .h5, saved_model/, .tflite
  Data pipeline:   large .parquet, .db, .sqlite, output/
  Web app:         node_modules/, .env, dist/, build/
  R:               .rds, .rda, .RData
  All projects:    .DS_Store, *.log, .env, __pycache__/

[I'll list what's already in your .gitignore if it exists]

What to add (comma-separated, e.g. ".pkl, .pt, checkpoints/"):
```

### Question 3: Test command
```
How do you run tests?

Common patterns:
  pytest                           # Python + pytest
  pytest -v -m "not slow"         # pytest, skipping slow tests
  python -m unittest discover     # Python unittest
  make test                       # Makefile-wrapped
  npm test                        # JavaScript / TypeScript
  cargo test                      # Rust
  go test ./...                   # Go
  (none yet)                      # No tests set up

Your answer (e.g. "pytest -v"):
```

### Question 4: Protected files
```
Which files contain results or outputs that should NEVER be cleared?
(Notebooks with published figures, frozen benchmark results, generated reports)

Examples:
  o "notebooks/final_experiment.ipynb"    -- has figures used in the paper
  o "results/benchmark_v2.json"           -- frozen reference numbers
  o "reports/quarterly_analysis.html"     -- shared with stakeholders
  o "none"                                -- everything can be regenerated

Your answer (comma-separated paths, or "none"):
```

### Question 5: Entry points
```
What are the 2-3 main entry points someone needs to know about?

Examples:
  o "src/pipeline.py -- DASHPipeline class, main API"
  o "app/main.py -- FastAPI application"
  o "scripts/train.py -- training CLI"
  o "notebooks/analysis.ipynb -- primary analysis notebook"

List yours (one per line, format: "path -- description"):
```

## Step 4: Create files

From the answers, create:

### 4a. .gitignore additions
Append the user's blocked file types to .gitignore (create if missing). Always include `__pycache__/`, `*.pyc`, `.env`, `.DS_Store` as baseline.

### 4b. Pre-push hook
Create `.githooks/pre-push` with:
- Main branch block (hard stop)
- Blocked extensions from Question 2 (case statement)
- File size block >10MB (using git cat-file -s on pushed commits)
- Clean error messages

CRITICAL: The hook must parse stdin for refs. Use `while read local_ref local_sha remote_ref remote_sha` to get the commit range. Use `git diff --name-only --diff-filter=ACM "$range"` for pushed files. Use `git cat-file -s "$local_sha:$file"` for size. Track errors in a variable, exit after the loop.

Activate: `git config core.hooksPath .githooks`

### 4c. Project instructions file
Create CLAUDE.md (since we're in Claude Code) with:
- Project description from Question 1
- Directory map (infer from `ls` and project structure)
- Entry points from Question 5
- Running section with test command from Question 3
- Do NOT section with blocked files from Question 2
- Protected files from Question 4 (if any)
- Non-Negotiable Rules:
  1. Never push directly to main
  2. One concern per commit, one concern per branch
  3. Read code before planning changes

### 4d. Claude Code hooks (if not already present)
Create `.claude/settings.json` with format-on-write PostToolUse hook and session-start hook.
Detect the formatter: check pyproject.toml for ruff/black config, package.json for prettier.

## Step 5: Report and next steps

Show what was created and the project's current level:
```
Level 1 setup complete:
  .gitignore          -- blocks [list]
  .githooks/pre-push  -- blocks binaries + main pushes + >10MB
  CLAUDE.md           -- project instructions (review and edit)
  .claude/settings.json -- format-on-write + session-start hooks

Run /bootstrap again when you want Level 2, or /health-check to see recommendations.
```

## Rules
- Never create files without showing what will be created and getting confirmation
- Never overwrite existing files — merge additions into existing content
- If a project instructions file already exists for a different tool (.cursorrules, etc.), ask if the user wants CLAUDE.md too or wants to keep the existing one
- If git is not initialized, initialize it first (`git init`)
- If the project has no files at all, ask "Is this a new project?" and adjust expectations
