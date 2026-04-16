# Sensitive Data Scan — 2026-04-16

Scanned all tracked files (`git ls-files`, 295 files) for API keys, credentials, absolute paths, tracked-but-ignored files, sensitive docs/paper content, and `.pkl` files in git history.

---

## API Keys / Tokens / Credentials

No actual secrets found. All matches for `sk-`, `Bearer `, `api_key =`, `password =`, `token =`, `secret =`, `credentials` were in audit templates, documentation reminders, or prior audit reports — instructional text only, not real credentials.

---

## Absolute Paths with Usernames

- [MEDIUM] notebooks/demo_benchmark_6.ipynb lines 63–929 — 189 occurrences of `/Users/drake.caraker/ds_projects/dash-shap/...` in cell outputs (checkpoint paths, warning tracebacks from `.venv/lib/python3.13/site-packages/`). Exposes author's local username and directory structure. Note: CLAUDE.md Rule #4 prohibits clearing outputs on this canonical notebook.
  Evidence: `notebooks/demo_benchmark_6.ipynb:63:      "Checkpoint directory: /Users/drake.caraker/ds_projects/dash-shap/checkpoints\n",`

- [MEDIUM] notebooks/tutorial_02_dash_walkthrough.ipynb lines 811–812 — 2 occurrences of SageMaker internal debugging paths in notebook metadata. Exposes SageMaker session hash.
  Evidence: `notebooks/tutorial_02_dash_walkthrough.ipynb:811:       "debugging_info_folder": "/home/sagemaker-user/shared/.temp_sagemaker_unified_studio_debugging_info/d9477c84",`

- [MEDIUM] requirements.lock line 69 — absolute path to local project directory embedded in lockfile entry.
  Evidence: `requirements.lock:69:UNKNOWN @ file:///Users/drake.caraker/ds_projects/dash-shap`

---

## Tracked Files Matching .gitignore Patterns

No tracked files match any `.gitignore` patterns. Cross-check of all 295 tracked files against `.gitignore` via `git check-ignore` returned zero matches.

---

## Sensitive Content in docs/ and paper/

- [MEDIUM] docs/private/ — 17 tracked files containing review strategy, communications plans, cover letter drafts, and reviewer response notes. These are encrypted via git-crypt (`docs/private/** filter=git-crypt diff=git-crypt` in `.gitattributes`), so they are not readable without the GPG key. No remediation needed as long as git-crypt remains active.
  Evidence: `git check-attr filter docs/private/comms_strategy.md` → `docs/private/comms_strategy.md: filter: git-crypt`

- [MEDIUM] docs/private/outreach/ — 3 untracked files (`lundberg_email.md`, `newsletter_pitch.md`, `researcher_emails.md`) in staging area. These are NOT yet committed and therefore NOT yet encrypted by git-crypt. If committed, they would be encrypted. The `researcher_emails.md` file references named researchers (Blair Bilodeau, Hyeonggeun Hwang) in an outreach context. Risk is low while untracked but would need git-crypt protection if committed.
  Evidence: `docs/private/outreach/researcher_emails.md:37:### Blair Bilodeau (Impossibility theorems, PNAS 2024)`

- [LOW] paper/tmlr_main.tex lines 22–32 — Contains TMLR template placeholder author names (Kyunghyun Cho, Raia Hadsell, Hugo Larochelle) with email addresses and institutions. These are from the official TMLR LaTeX template, not the actual submission. No real author info exposed.
  Evidence: `paper/tmlr_main.tex:22:\author{\name Kyunghyun Cho \email kyunghyun.cho@nyu.edu \\`

- [LOW] paper/ bibliography entries cite Bilodeau (PNAS 2024) and Hwang (2026) as published/public works. These are standard academic citations, not references to unpublished or embargoed work.
  Evidence: `paper/draft_v7_preprint.tex:333:\citet{bilodeau2024impossibility} prove impossibility results for`

---

## .pkl Files in Git History

No `.pkl` files found in git history. Scanned all commits across all branches via `git log --all --diff-filter=A --name-only`.

---

## Summary

6 issues found (0 CRITICAL, 0 HIGH, 5 MEDIUM, 1 LOW)

| Severity | Location | Description |
|----------|----------|-------------|
| MEDIUM | `notebooks/demo_benchmark_6.ipynb` | 189 absolute paths with local username (`/Users/drake.caraker/...`) in cell outputs |
| MEDIUM | `notebooks/tutorial_02_dash_walkthrough.ipynb` | 2 SageMaker debugging paths with session hash in metadata |
| MEDIUM | `requirements.lock` line 69 | Absolute path to local project in lockfile |
| MEDIUM | `docs/private/` (17 files) | Strategy/review docs — protected by git-crypt |
| MEDIUM | `docs/private/outreach/` (3 untracked files) | Researcher contact/outreach drafts — not yet encrypted |
| LOW | `paper/tmlr_main.tex` | TMLR template placeholder authors (not real submission data) |

### Changes Since Last Audit (2026-04-03)

- `requirements.lock` absolute path is a **new finding** (not in prior audit).
- `docs/private/outreach/` untracked files are **new** (3 files appeared since last audit).
- Notebook absolute path counts unchanged (189 in nb6, 2 in tutorial_02).
- No `.pkl` files introduced. No credentials or API keys found.
- git-crypt protection on `docs/private/` remains active and correctly configured.
