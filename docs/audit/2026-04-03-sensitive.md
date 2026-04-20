# Sensitive Data Scan — 2026-04-03

Scanned all tracked files (`git ls-files`, 202 files excluding `.claude/` and `.pilot/`) for API keys, credentials, absolute paths, tracked-but-ignored files, sensitive docs/paper content, and `.pkl` files in git history.

## Findings

### API Keys / Tokens / Credentials

No matches found. Patterns scanned (case-insensitive): `sk-[20+ chars]`, `Bearer [token]`, `api_key =`, `password =`, `token =`, `secret =`, `credentials`. The only hits were in documentation/audit files describing scan procedures or policy reminders — no actual secrets.

### Absolute Paths

- [MEDIUM] `notebooks/demo_benchmark_6.ipynb` — 189 occurrences of `/Users/drake.caraker/ds_projects/dash-shap/...` in cell outputs (checkpoint paths, warning tracebacks from `.venv/lib/python3.13/site-packages/`). Exposes local username and directory structure. CLAUDE.md Rule #4 prohibits clearing outputs on this canonical notebook, so remediation requires judgment.
  Evidence: `grep -c '/Users/' notebooks/demo_benchmark_6.ipynb` → 189. Sample lines 63, 71, 777-880, 929, 978, 996, 1045, 1061, 1079.

- [MEDIUM] `notebooks/tutorial_02_dash_walkthrough.ipynb` lines 811-812 — 2 occurrences of `/home/sagemaker-user/shared/.temp_sagemaker_unified_studio_debugging_info/...` in notebook metadata. Exposes SageMaker internal debugging paths and a session hash (`d9477c84`).
  Evidence: `grep -n '/home/' notebooks/tutorial_02_dash_walkthrough.ipynb`
  ```
  811:       "debugging_info_folder": "/home/sagemaker-user/shared/.temp_sagemaker_unified_studio_debugging_info/d9477c84",
  812:       "instruction_file": "/home/sagemaker-user/shared/.temp_sagemaker_unified_studio_debugging_info/ipython_debugging_sop.txt",
  ```

- [LOW] `requirements.lock` line 69 — contains `UNKNOWN @ file:///Users/drake.caraker/ds_projects/dash-shap`, exposing local path in pip freeze output.
  Evidence: `grep -n '/Users/' requirements.lock`
  ```
  69:UNKNOWN @ file:///Users/drake.caraker/ds_projects/dash-shap
  ```

### Tracked Files Matching .gitignore Patterns

No violations found. No `.pkl`, `.env`, `.bak.json`, `checkpoints/`, or LaTeX build artifact files are tracked. The `.gitignore` patterns are properly excluding these file types.

### Git History — .pkl Files

No `.pkl` files found in git history.
  Evidence: `git log --all --diff-filter=A --name-only --pretty=format: | grep -E '\.pkl$'` → no output.

### Docs/Paper — Sensitive Content

- [LOW] `docs/private/` (16 files) — contains review documents, strategy memos, roadmap, and cover letter drafts with references to anticipated reviewer questions, competitive positioning (Bilodeau, Hwang, D'Amour), and publication strategy. These files are properly encrypted via git-crypt (`docs/private/** filter=git-crypt diff=git-crypt` in `.gitattributes`). No named reviewers or action editors identified — all references use generic framing ("a TMLR reviewer will ask...").
  Evidence: `.gitattributes` confirms `docs/private/** filter=git-crypt diff=git-crypt`.

- [LOW] `paper/draft_v7_preprint.tex` — cites Bilodeau et al. (2024), Hwang et al. (2026), and D'Amour et al. (2020) as published/public academic references. These are standard citations, not unpublished competitor work or confidential information.
  Evidence: `grep -n 'bilodeau\|hwang\|damour' paper/draft_v7_preprint.tex` → lines 316, 343, 353, 1821-1822, 1840-1841, 1994-1995 (all bibliography entries and citations).

- No named reviewers, action editor names, or non-author institution affiliations found in `docs/` or `paper/` (outside git-crypt-encrypted `docs/private/`).

## Summary

3 issues found (0 CRITICAL, 0 HIGH, 2 MEDIUM, 1 LOW)

| Severity | File | Description |
|----------|------|-------------|
| MEDIUM | `notebooks/demo_benchmark_6.ipynb` | 189 absolute paths with local username (`/Users/drake.caraker/...`) in cell outputs |
| MEDIUM | `notebooks/tutorial_02_dash_walkthrough.ipynb` | 2 SageMaker internal paths (`/home/sagemaker-user/...`) in notebook metadata |
| LOW | `requirements.lock` | 1 absolute local path in pip freeze output |

### Remediation Notes

1. **demo_benchmark_6.ipynb** (MEDIUM): CLAUDE.md Rule #4 prohibits clearing outputs on this canonical ArXiv notebook. The paths are in warning tracebacks and checkpoint messages — cosmetic but leak the author's local environment. Consider whether future re-runs could use relative paths for checkpoint messages.
2. **tutorial_02_dash_walkthrough.ipynb** (MEDIUM): The SageMaker metadata can be safely removed from the notebook's kernel metadata block without affecting outputs.
3. **requirements.lock** (LOW): Regenerate with `pip freeze` on a clean environment or replace the `file:///` entry with the package name and version.
