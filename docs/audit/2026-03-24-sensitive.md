# Sensitive Data Scan — 2026-03-24

Scanned all tracked files (`git ls-files`, 147 files) for API keys, credentials, absolute paths, tracked-but-ignored files, sensitive docs/paper content, and `.pkl` files in git history.

## Findings

### API Keys / Tokens / Credentials

No matches found. Patterns scanned (case-insensitive): `sk-`, `Bearer `, `api_key =`, `password =`, `token =`, `secret =`, `credentials`. The only hits were in the audit template (`.claude/commands/audit.md`) and a documentation reminder in `CONTRIBUTING.md` — both are instructional text, not actual secrets.

### Absolute Paths with Usernames

- [MEDIUM] `notebooks/demo_benchmark_6.ipynb` — 189 occurrences of `/Users/drake.caraker/ds_projects/dash-shap/...` in cell outputs (checkpoint paths, warning tracebacks from `.venv/lib/python3.13/site-packages/`). These are embedded notebook outputs exposing the author's local username and directory structure. Note: CLAUDE.md Rule #4 prohibits clearing outputs on this canonical notebook, so remediation requires judgment.

- [MEDIUM] `notebooks/tutorial_02_dash_walkthrough.ipynb` lines 811-812 — 2 occurrences of `/home/sagemaker-user/shared/.temp_sagemaker_unified_studio_debugging_info/...` in notebook metadata. Exposes SageMaker internal debugging paths and a session hash (`d9477c84`).

### Tracked Files Matching `.gitignore` Patterns

No tracked files match current `.gitignore` patterns. Git configuration is clean.

### `.pkl` Files in Git History

No `.pkl` files found in any branch history (scanned all branches via `git log --all --diff-filter=A`). The pre-push hook appears to be working as intended.

### docs/private/ — Encrypted Review and Strategy Documents

All 10 files in `docs/private/` are properly encrypted via git-crypt (confirmed via `git crypt status`). The `.gitattributes` filter rule `docs/private/** filter=git-crypt diff=git-crypt` is correctly applied. Files are only readable on machines with the unlocked GPG key.

Content review (on this unlocked machine) found no named reviewers, no reviewer email addresses, and no named institutions beyond standard venue references (TMLR, ArXiv). The review documents use generic framing ("a TMLR reviewer will ask...") rather than identifying specific individuals. The `comms_strategy.md` discusses publication sequencing and scoop risk — sensitive but appropriately encrypted.

### GPG Key ID Exposure

- [LOW] `.git-crypt/keys/default/0/D1CC844DC3B104C8345F43E2FF952C282E1D3AF1.gpg` — GPG key ID is visible in the tracked file path. This is standard git-crypt infrastructure and not avoidable, but the key fingerprint is exposed in the repository file listing. No references to this key ID exist outside `.git-crypt/`.

### Paper and Documentation

- No sensitive information found in `paper/draft_v7_preprint.tex` (no reviewer names, no action editor references).
- `CITATION.cff` contains author names and GitHub URL — expected and intentional for a public research project.
- ArXiv tar.gz archives (`paper/dash-shap-arxiv.tar.gz`, `paper/dash-shap-arxiv-v6.tar.gz`) contain only figures and `main.tex` — no sensitive metadata.
- 6 PDF files tracked in `paper/` (v2 through v6, zenodo). Total ~7.7MB. These are compiled preprints — no unexpected content based on file sizes.

### GitHub Username in CITATION.cff

- [LOW] `CITATION.cff` line 13 — `url: https://github.com/DrakeCaraker/dash-shap`. GitHub username is public by design for a published open-source project. Flagged for completeness only.

## Summary

**5 issues found (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW, 1 informational note)**

| Severity | File | Description |
|----------|------|-------------|
| MEDIUM | `notebooks/demo_benchmark_6.ipynb` | 189 absolute paths with local username (`/Users/drake.caraker/...`) in cell outputs |
| MEDIUM | `notebooks/tutorial_02_dash_walkthrough.ipynb` | SageMaker debugging paths in notebook metadata (lines 811-812) |
| LOW | `.git-crypt/keys/default/0/D1CC...3AF1.gpg` | GPG key fingerprint visible in tracked file path (standard git-crypt) |
| LOW | `CITATION.cff` | GitHub username in URL (intentional for public project) |
| INFO | `docs/private/` (10 files) | Encrypted review/strategy docs — properly protected via git-crypt |

### Recommended Actions

1. **`notebooks/tutorial_02_dash_walkthrough.ipynb`** — Remove the SageMaker debugging metadata keys (`debugging_info_folder`, `instruction_file`) from the notebook metadata. These are not needed for execution and leak infrastructure details.
2. **`notebooks/demo_benchmark_6.ipynb`** — The 189 path leaks are in cell outputs protected by Rule #4 (do not clear canonical notebook outputs). Consider whether a future re-run on a clean environment could produce outputs without local paths. No immediate action required, but flag for next canonical re-execution.
