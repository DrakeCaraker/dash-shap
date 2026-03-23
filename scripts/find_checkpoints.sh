#!/usr/bin/env bash
# find_checkpoints.sh — Scan the repo for checkpoint and result artifacts.
# Usage: bash scripts/find_checkpoints.sh [ROOT_DIR]

set -euo pipefail

ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    BOLD='\033[1m' DIM='\033[2m' GREEN='\033[32m'
    YELLOW='\033[33m' RED='\033[31m' CYAN='\033[36m' RESET='\033[0m'
else
    BOLD='' DIM='' GREEN='' YELLOW='' RED='' CYAN='' RESET=''
fi

header() { printf "\n${BOLD}${CYAN}=== %s ===${RESET}\n" "$1"; }
warn()   { printf "  ${YELLOW}⚠  %s${RESET}\n" "$1"; }
ok()     { printf "  ${GREEN}✓  %s${RESET}\n" "$1"; }
item()   { printf "  ${DIM}%-50s${RESET} %s\n" "$1" "$2"; }

total_size=0
total_files=0

add_size() {
    local bytes=$1
    total_size=$((total_size + bytes))
    total_files=$((total_files + 1))
}

human_size() {
    local bytes=$1
    if [ "$bytes" -ge 1048576 ]; then
        printf "%.1f MB" "$(echo "$bytes / 1048576" | bc -l)"
    elif [ "$bytes" -ge 1024 ]; then
        printf "%.1f KB" "$(echo "$bytes / 1024" | bc -l)"
    else
        printf "%d B" "$bytes"
    fi
}

# ── 1. Known checkpoint directories ──────────────────────────────────────────
header "Checkpoint Directories"

for dir in "checkpoints" "results/checkpoints" "notebooks/checkpoints"; do
    full="$ROOT/$dir"
    if [ -d "$full" ]; then
        count=$(find "$full" -maxdepth 1 -name '*.pkl' -type f 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            warn "$dir/ — $count pkl file(s):"
            find "$full" -maxdepth 1 -name '*.pkl' -type f -printf '    %f\t' -exec stat --printf='%s' {} \; 2>/dev/null | while read -r line; do
                fname=$(echo "$line" | cut -f1)
                bytes=$(echo "$line" | cut -f2)
                add_size "${bytes:-0}"
                printf "    %-45s %s\n" "$fname" "$(human_size "${bytes:-0}")"
            done
            # Also tally via subshell since pipe loses vars
            dir_bytes=$(find "$full" -maxdepth 1 -name '*.pkl' -type f -exec stat --printf='%s\n' {} + 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
            total_size=$((total_size + dir_bytes))
            total_files=$((total_files + count))
        else
            ok "$dir/ — empty (no pkl files)"
        fi
    else
        ok "$dir/ — does not exist"
    fi
done

# ── 2. Stray .pkl files anywhere in repo ─────────────────────────────────────
header "Stray .pkl Files (outside checkpoint dirs)"

stray_count=0
while IFS= read -r -d '' pkl; do
    rel="${pkl#"$ROOT/"}"
    # Skip known checkpoint dirs
    case "$rel" in
        checkpoints/*|results/checkpoints/*|notebooks/checkpoints/*) continue ;;
    esac
    bytes=$(stat --printf='%s' "$pkl" 2>/dev/null || echo 0)
    total_size=$((total_size + bytes))
    total_files=$((total_files + 1))
    stray_count=$((stray_count + 1))
    item "$rel" "$(human_size "$bytes")"
done < <(find "$ROOT" -name '*.pkl' -type f -not -path '*/.git/*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*' -print0 2>/dev/null)

if [ "$stray_count" -eq 0 ]; then
    ok "No stray .pkl files found"
fi

# ── 3. Result JSON files ─────────────────────────────────────────────────────
header "Result JSON Files (results/tables/)"

tables_dir="$ROOT/results/tables"
if [ -d "$tables_dir" ]; then
    json_count=0
    while IFS= read -r -d '' jf; do
        fname=$(basename "$jf")
        bytes=$(stat --printf='%s' "$jf" 2>/dev/null || echo 0)
        item "$fname" "$(human_size "$bytes")"
        json_count=$((json_count + 1))
    done < <(find "$tables_dir" -maxdepth 1 -name '*.json' -type f -print0 2>/dev/null | sort -z)

    if [ "$json_count" -eq 0 ]; then
        warn "Directory exists but contains no JSON files"
    fi
else
    warn "results/tables/ does not exist"
fi

# ── 4. Expected experiments vs. persisted results ────────────────────────────
header "Experiment Coverage"

declare -A EXPECTED=(
    ["linear_sweep"]="synthetic_linear_sweep.json"
    ["overlapping"]="overlapping.json"
    ["nonlinear_sweep"]="nonlinear_sweep.json"
    ["table2_baselines"]="table2_baselines.json"
    ["real_california"]="california_housing.json"
    ["real_breast_cancer"]="breast_cancer.json"
    ["real_superconductor"]="superconductor.json"
    ["epsilon_sensitivity"]="epsilon_sensitivity.json"
    ["ablation"]="ablation.json"
    ["variance_decomposition"]="variance_decomposition.json"
    ["variance_decomposition_crossed"]="variance_decomposition_crossed.json"
    ["first_mover_visualization"]="first_mover_visualization.json"
    ["first_mover_bias"]="first_mover_bias.json"
    ["background_sensitivity"]="background_sensitivity.json"
    ["asymmetric_dgp"]="asymmetric_dgp.json"
    ["k_sweep_independence"]="k_sweep_independence.json"
)

present=0
missing=0
for exp in $(echo "${!EXPECTED[@]}" | tr ' ' '\n' | sort); do
    outfile="${EXPECTED[$exp]}"
    path="$tables_dir/$outfile"
    if [ -f "$path" ]; then
        printf "  ${GREEN}✓${RESET}  %-38s → %s\n" "$exp" "$outfile"
        present=$((present + 1))
    else
        printf "  ${RED}✗${RESET}  %-38s → %s ${DIM}(missing)${RESET}\n" "$exp" "$outfile"
        missing=$((missing + 1))
    fi
done

# ── 5. Large files that might be checkpoint-like ─────────────────────────────
header "Large Files (>1 MB, non-git)"

large_count=0
while IFS= read -r -d '' lf; do
    rel="${lf#"$ROOT/"}"
    bytes=$(stat --printf='%s' "$lf" 2>/dev/null || echo 0)
    if [ "$bytes" -ge 1048576 ]; then
        item "$rel" "$(human_size "$bytes")"
        large_count=$((large_count + 1))
    fi
done < <(find "$ROOT" -type f -size +1M -not -path '*/.git/*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*' -not -path '*/.venv/*' -not -path '*/venv/*' -print0 2>/dev/null)

if [ "$large_count" -eq 0 ]; then
    ok "No large files found"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
header "Summary"
printf "  Checkpoint/pkl files found:  %d (%s total)\n" "$total_files" "$(human_size "$total_size")"
printf "  Experiments persisted:       %d / %d\n" "$present" "$(( present + missing ))"
printf "  Experiments missing:         %d\n" "$missing"

if [ "$total_files" -gt 0 ]; then
    echo ""
    warn "pkl files are blocked from git by pre-push hook."
    warn "Use S3 or 'results/tables/*.json' for persistent storage."
fi
