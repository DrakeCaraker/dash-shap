#!/bin/bash
# SessionStart hook: full warm-up for DASH-SHAP development sessions
# Prints status info to stderr so it appears as hook output

echo "=== DASH-SHAP Session Warm-Up ===" >&2

# 1. Git status — uncommitted changes
dirty=$(git status --porcelain 2>/dev/null | head -20)
if [ -n "$dirty" ]; then
    count=$(echo "$dirty" | wc -l)
    echo "" >&2
    echo "Git: $count uncommitted change(s):" >&2
    echo "$dirty" >&2
else
    echo "" >&2
    echo "Git: working tree clean" >&2
fi

# 2. Branch drift check
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$branch" != "main" ] && [ "$branch" != "HEAD" ]; then
    git fetch origin main --quiet 2>/dev/null
    if git rev-parse origin/main >/dev/null 2>&1; then
        behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
        if [ "$behind" -gt 0 ]; then
            echo "" >&2
            echo "Drift: branch '$branch' is $behind commit(s) behind origin/main" >&2
            echo "  Rebase before pushing: git rebase origin/main" >&2
        else
            echo "" >&2
            echo "Drift: up to date with origin/main" >&2
        fi
    fi
fi

# 3. Stale checkpoint/pkl files
pkl_files=$(find checkpoints/ -name "*.pkl" 2>/dev/null)
if [ -n "$pkl_files" ]; then
    pkl_count=$(echo "$pkl_files" | wc -l)
    echo "" >&2
    echo "Checkpoints: $pkl_count .pkl file(s) in checkpoints/" >&2
    echo "$pkl_files" | head -5 >&2
    if [ "$pkl_count" -gt 5 ]; then
        echo "  ... and $((pkl_count - 5)) more" >&2
    fi
else
    echo "" >&2
    echo "Checkpoints: none" >&2
fi

# 4. Verify git hooks are active
hooks_path=$(git config core.hooksPath 2>/dev/null)
if [ "$hooks_path" = ".githooks" ]; then
    echo "" >&2
    echo "Git hooks: active (.githooks)" >&2
else
    echo "" >&2
    echo "Git hooks: NOT ACTIVE — run: git config core.hooksPath .githooks" >&2
fi

# 5. Canonical notebook sizes
echo "" >&2
echo "Notebooks:" >&2
for nb in notebooks/demo_benchmark_6.ipynb notebooks/demo_benchmark_7.ipynb; do
    if [ -f "$nb" ]; then
        size=$(stat -c%s "$nb" 2>/dev/null)
        size_kb=$((size / 1024))
        label=""
        if [[ "$nb" == *"_6"* ]]; then label=" (ArXiv canonical)"; fi
        if [[ "$nb" == *"_7"* ]]; then label=" (TMLR in-dev)"; fi
        warn=""
        if [ "$size" -gt 2097152 ]; then warn=" [WARNING: >2MB]"; fi
        echo "  $nb: ${size_kb}KB${label}${warn}" >&2
    fi
done

echo "" >&2
echo "=================================" >&2

exit 0
