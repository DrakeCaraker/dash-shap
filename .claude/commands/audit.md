# Audit

Run a parallel four-dimension repo audit using subagents.

This skill performs a comprehensive audit across four dimensions: Notebook Health, Preprint-Code Parity, Sensitive Data, and Release Readiness. All four agents run in parallel (or a single dimension if specified via `/audit notebooks|preprint|sensitive|release`), then results are merged into a consolidated report.

## Optional argument
- `/audit` — run all four agents in parallel
- `/audit notebooks` — notebook health only
- `/audit preprint` — preprint-code parity only
- `/audit sensitive` — sensitive data scan only
- `/audit release` — release readiness only

## Execution

### Step 1: Get today's date
Get the current date to use as AUDIT_DATE in all output filenames:
```bash
date +%Y-%m-%d
```

Capture the output into a variable AUDIT_DATE (e.g., `2026-03-22`). Replace every occurrence of the literal placeholder `AUDIT_DATE` in filenames and file headings with this actual date string.

### Step 2: Create output directory
```bash
mkdir -p docs/audit
```

### Step 3: Spawn Agent calls

**Single-dimension routing:** If the user specifies `/audit notebooks`, `/audit preprint`, `/audit sensitive`, or `/audit release`, spawn only the corresponding Agent below with the same output format and severity levels. Otherwise, proceed to all-four dispatch below.

**If all four dimensions are requested OR no dimension specified:**

Each Agent tool call receives the agent's full task description (everything under its Task section) as the `prompt` parameter. Make all four calls in one response to dispatch them simultaneously.

In a single message, spawn FOUR Agent tool calls in parallel. Each agent should follow the specification below and write findings to the designated output file. Use the exact severity format for every finding:
```
- [SEVERITY] <location> — <description>
```
Where SEVERITY is one of: CRITICAL, HIGH, MEDIUM, LOW

Each output file must end with:
```
N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```

---

#### Agent 1: Notebook Health
**Output file:** `docs/audit/AUDIT_DATE-notebooks.md`

**Task:**
- Parse every .ipynb file in `notebooks/` (excluding `notebooks/archive/`) using Python's json module
  - Flag parse failures as CRITICAL with the notebook path
- For each notebook, examine all code cells:
  - Check that every code cell has a non-empty `id` field — missing IDs are MEDIUM
  - Check that `execution_count` values are either: sequential (1, 2, 3, ...) or all null
    - Out-of-order execution_count is HIGH (indicates manual edits or interrupted runs)
- For markdown cells, check figure references `![...](path)`:
  - Verify that referenced paths exist in the notebook's directory or absolute path
  - Missing figures are HIGH
- Check notebook file size: flag notebooks > 2MB as HIGH (git bloat risk)
- For each code cell containing `load_checkpoint(` calls:
  - Extract the checkpoint filename argument
  - Verify the file exists in `checkpoints/` directory
  - Missing checkpoint files are MEDIUM (experiment not yet run or checkpoint cleared)

**Output format:**
```markdown
# Notebook Health Audit — AUDIT_DATE

Scanned X notebooks, found Y issues:

- [CRITICAL] notebooks/notebook_name.ipynb — JSON parse error: <error message>
- [MEDIUM] notebooks/notebook_name.ipynb — code cell 5 missing id field
- [HIGH] notebooks/notebook_name.ipynb — execution_count out of order: [1, 2, 4, 3, 5]
- [HIGH] notebooks/notebook_name.ipynb — missing figure: ![alt](figures/missing.png)
- [HIGH] notebooks/notebook_name.ipynb — file size 2.3MB exceeds 2MB threshold
- [MEDIUM] notebooks/notebook_name.ipynb — load_checkpoint('ckpt_name') but checkpoints/ckpt_name.pkl not found

N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```

---

#### Agent 2: Preprint-Code Parity
**Output file:** `docs/audit/AUDIT_DATE-preprint.md`

**Task:**
- First, find the highest-versioned preprint: `ls paper/draft_v*.tex 2>/dev/null | sort -V | tail -1`. Use that file for all claims extraction below. If no such file exists, report MEDIUM: 'No preprint draft found in paper/'.
- Read that preprint file and extract every numeric claim:
  - Stability values (e.g., "0.977")
  - Model counts (M=, K=)
  - Experiment parameters (N_REPS=)
  - Dataset results (method names with associated numbers)
  - Any numeric comparisons ("X vs Y", "better by 0.05")
- For each extracted claim, determine its source experiment:
  - Check `docs/BENCHMARK_RESULTS.md` for a matching entry
  - Check all `results/tables/*.json` files for data
  - Match by experiment name and method
- Severity assessment:
  - If claim matches v6 data (found in ArXiv results) but v7 not yet available → LOW (expected, v7 in progress)
  - If claim matches neither v6 nor v7 data → HIGH (data missing or incorrect claim)
  - If no results file exists for the referenced experiment → MEDIUM (incomplete results)
- Check every table in the LaTeX source:
  - Each table row should have corresponding entry in results or BENCHMARK_RESULTS.md
  - Missing entries are MEDIUM

**Output format:**
```markdown
# Preprint-Code Parity Audit — AUDIT_DATE

Checked draft_v7_preprint.tex and cross-referenced against results:

- [HIGH] paper/draft_v7_preprint.tex line 42 — stability claim 0.956 not found in results tables
- [MEDIUM] paper/draft_v7_preprint.tex line 85 — experiment "asymmetric_dgp" has no results file in results/tables/
- [LOW] paper/draft_v7_preprint.tex line 120 — stability value 0.977 matches v6 (v7 data not yet available, expected)

N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```

---

#### Agent 3: Sensitive Data Scan
**Output file:** `docs/audit/AUDIT_DATE-sensitive.md`

**Task:** Be conservative — flag any doubt
- Scan all tracked files (via `git ls-files`) for sensitive patterns (case-insensitive):
  - API keys/tokens: `sk-`, `Bearer `, `api_key =`, `password =`, `token =`, `secret =`, `credentials`
  - Flag as CRITICAL with file path and line number
- Absolute paths with usernames or home directories in committed files:
  - Patterns: `/home/`, `/Users/`, `/root/`
  - Exception: paths in comments or documentation strings are MEDIUM (less severe but still worth noting)
  - Paths in code strings or config values are MEDIUM
- Run `git ls-files` and cross-check against `.gitignore`:
  - Files that are tracked but match `.gitignore` patterns are HIGH (should be ignored)
  - This indicates a git configuration issue
- Scan `docs/` and `paper/` directories for sensitive information:
  - In docs/ and paper/: flag any non-author institution names mentioned outside affiliation sections, any named reviewers, any references to unpublished competitor work. Flag as HIGH (could identify proprietary information)
- Check git history for `.pkl` files:
  - Run `git log --all --diff-filter=A --name-only --pretty=format: | grep -E '\.pkl$'`
  - Any .pkl files in history are CRITICAL (binary data accidentally committed)

**Output format:**
```markdown
# Sensitive Data Scan — AUDIT_DATE

Scanned tracked files for API keys, credentials, absolute paths, and git history:

- [CRITICAL] dash_shap/utils/config.py line 15 — API key pattern found: "sk-..."
- [MEDIUM] notebooks/demo_benchmark_6.ipynb — absolute path in comment: "/Users/alice/data"
- [HIGH] results/tables/config.json — tracked but matches .gitignore pattern "results/*"
- [HIGH] paper/draft_v7_preprint.tex — company name "CompanyX" found in section 2.1 (may be under review)
- [CRITICAL] git history — .pkl file found in commits: checkpoints/old_run.pkl

N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```

---

#### Agent 4: Release Readiness
**Output file:** `docs/audit/AUDIT_DATE-release.md`

**Task:**
- **Packaging:**
  - Check that `pyproject.toml` or `setup.py` exists in repo root
  - If found, verify it includes: `name` field, `version` field, `dependencies` field
  - Missing version is HIGH, missing deps is HIGH
- **Public API Docstrings:**
  - Scan `dash_shap/` directory for all modules, classes, and functions not starting with `_`
  - For each public API item, check if it has a docstring (first string literal after definition)
  - Missing docstrings are MEDIUM
- **CLAUDE.md Accuracy:**
  - Every command under "Running" section must have corresponding script or target
  - Verify scripts exist: `run_experiments.py`, `run_experiments_parallel.py`, etc.
  - Verify test targets work: `pytest`, `make lint`, `make fmt`, etc.
  - Missing or broken commands are MEDIUM
- **Test Coverage:**
  - Run: `pytest --cov=dash_shap --cov-report=term-missing --cov-fail-under=70 -q 2>&1 | tail -20`
  - Parse output for coverage percentage
  - If coverage < 70%, flag as HIGH with actual percentage
  - Note any modules with coverage < 50% as separate MEDIUM findings
  - Note: This may take 2-3 minutes. Use `-m 'not slow'` if time is limited — this still catches most coverage gaps.
- **Version Export:**
  - Run: `python -c "import dash_shap; print(dash_shap.__version__)"`
  - If it fails or prints None, missing __version__ is LOW
  - If it succeeds, note the version string

**Output format:**
```markdown
# Release Readiness Audit — AUDIT_DATE

Checked packaging, API docs, CLAUDE.md commands, test coverage, and version export:

- [HIGH] pyproject.toml — missing `version` field
- [MEDIUM] dash_shap/core/pipeline.py line 42 — function DASHPipeline.__init__() missing docstring
- [MEDIUM] CLAUDE.md "Running" section — command `make rebase` references but no Makefile found
- [HIGH] pytest coverage — current coverage 62%, below 70% threshold
- [MEDIUM] dash_shap/evaluation/metrics.py — module coverage 45%, below 50% threshold
- [LOW] dash_shap.__version__ — missing or not exported

N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```

---

**If a single dimension is requested:**

Spawn only the corresponding Agent with the same output format and severity levels.

### Step 4: Merge results (sequential, after all agents complete)

**If all four dimensions were run:**

Spawn one additional Agent to merge results. Pass the resolved date string explicitly in the merge agent's task prompt so it can construct the correct file paths — e.g., 'Read these files: docs/audit/2026-03-22-notebooks.md, docs/audit/2026-03-22-preprint.md, docs/audit/2026-03-22-sensitive.md, docs/audit/2026-03-22-release.md' with the actual date substituted.

**Task:**
- Read all four output files from `docs/audit/AUDIT_DATE-*.md` (excluding the merge output itself)
- Create consolidated report: `docs/audit/AUDIT_DATE-REPORT.md`
- Include a summary table with columns: Dimension | CRITICAL | HIGH | MEDIUM | LOW | Total
- Create a prioritized fix list:
  - Group findings by severity (CRITICAL → HIGH → MEDIUM → LOW)
  - Within each severity, list findings ordered by dimension (Notebooks, Preprint, Sensitive, Release)
  - Include file path and description for each
- Highlight any CRITICAL findings first in bold

**Output format:**
```markdown
# Audit Report — AUDIT_DATE

## Summary

| Dimension | CRITICAL | HIGH | MEDIUM | LOW | Total |
|-----------|----------|------|--------|-----|-------|
| Notebooks | 1 | 2 | 3 | 0 | 6 |
| Preprint | 0 | 1 | 2 | 1 | 4 |
| Sensitive | 2 | 1 | 1 | 0 | 4 |
| Release | 0 | 1 | 2 | 1 | 4 |
| **TOTAL** | **3** | **5** | **8** | **2** | **18** |

## Prioritized Fixes

### CRITICAL (3 issues)
1. **Notebooks** — notebooks/notebook_name.ipynb — JSON parse error: <error>
2. **Sensitive** — dash_shap/utils/config.py line 15 — API key pattern found
3. **Sensitive** — git history contains .pkl file: checkpoints/old_run.pkl

### HIGH (5 issues)
1. **Notebooks** — notebooks/notebook_name.ipynb — file size 2.3MB exceeds threshold
2. **Preprint** — paper/draft_v7_preprint.tex line 42 — stability claim 0.956 not found
3. **Sensitive** — results/tables/config.json — tracked but should be ignored
4. **Release** — pyproject.toml — missing version field
5. **Release** — pytest coverage 62%, below 70% threshold

### MEDIUM (8 issues)
... [continue with MEDIUM and LOW grouped similarly]

## Dimension Reports
- See docs/audit/AUDIT_DATE-notebooks.md for notebook details
- See docs/audit/AUDIT_DATE-preprint.md for preprint parity details
- See docs/audit/AUDIT_DATE-sensitive.md for sensitive data details
- See docs/audit/AUDIT_DATE-release.md for release readiness details
```

**If a single dimension was run:**

Skip the merge step. Report to user: "Single-dimension audit complete. Report at `docs/audit/AUDIT_DATE-<dimension>.md` — N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)"

### Step 5: Report to user

If all four dimensions:
```
Audit complete. Report at docs/audit/AUDIT_DATE-REPORT.md — N total issues (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```

If single dimension:
```
Single-dimension audit complete. Report at docs/audit/AUDIT_DATE-<dimension>.md — N issues found (X CRITICAL, Y HIGH, Z MEDIUM, W LOW)
```
