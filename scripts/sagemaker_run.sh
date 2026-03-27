#!/usr/bin/env bash
# =============================================================================
# DASH-SHAP SageMaker Experiment Runner
# =============================================================================
#
# Full workflow for running DASH experiments on a fresh SageMaker instance.
# Each phase is idempotent — safe to re-run if interrupted.
#
# Workflow:
#   1. Clone repo into SageMaker instance
#   2. bash scripts/sagemaker_run.sh setup
#   3. bash scripts/sagemaker_run.sh smoke
#   4. bash scripts/sagemaker_run.sh branch
#   5. bash scripts/sagemaker_run.sh run
#   6. bash scripts/sagemaker_run.sh status   (anytime)
#   7. bash scripts/sagemaker_run.sh finish
#
# See REPRODUCE.md "SageMaker Workflow" for full documentation.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/DrakeCaraker/dash-shap.git"
RUN_DATE=$(date +%Y%m%d)
BRANCH_NAME="results/sagemaker-run-${RUN_DATE}"
TAG_START="run-tmlr-${RUN_DATE}-start"
TAG_END="run-tmlr-${RUN_DATE}-end"
SCREEN_NAME="dash-${RUN_DATE}"
PROGRESS_FILE="results/.progress"

# All experiments in recommended run order
EXPERIMENTS=(
    linear_sweep
    nonlinear_sweep
    table2_baselines
    real_california
    real_breast_cancer
    real_superconductor
    epsilon_sensitivity
    ablation
    variance_decomposition
    variance_decomposition_crossed
    first_mover_bias
    background_sensitivity
    asymmetric_dgp
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[DASH]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

check_repo() {
    if [[ ! -f "run_experiments_parallel.py" ]]; then
        die "Not in the dash-shap repo root. cd into the repo first."
    fi
}

detect_instance_type() {
    if [[ -n "${SM_CURRENT_INSTANCE_TYPE:-}" ]]; then
        log "Instance type: $SM_CURRENT_INSTANCE_TYPE"
        return
    fi

    # Try EC2 metadata API (IMDSv1)
    local itype
    itype=$(curl -s --max-time 2 \
        http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "")
    if [[ -n "$itype" ]]; then
        export SM_CURRENT_INSTANCE_TYPE="ml.${itype}"
        log "Detected instance type: $SM_CURRENT_INSTANCE_TYPE"
        return
    fi

    # Try SageMaker resource metadata
    if [[ -f /opt/ml/metadata/resource-metadata.json ]]; then
        itype=$(python3 -c "import json; print(json.load(open('/opt/ml/metadata/resource-metadata.json')).get('ResourceConfig',{}).get('InstanceType',''))" 2>/dev/null || echo "")
        if [[ -n "$itype" ]]; then
            export SM_CURRENT_INSTANCE_TYPE="$itype"
            log "Detected instance type: $SM_CURRENT_INSTANCE_TYPE"
            return
        fi
    fi

    warn "Could not detect instance type."
    warn "Set it manually: export SM_CURRENT_INSTANCE_TYPE='ml.g5.16xlarge'"
}

# ---------------------------------------------------------------------------
# Phase 1: setup — install deps, verify environment
# ---------------------------------------------------------------------------
do_setup() {
    log "=== Phase 1: Environment Setup ==="

    # --- Detect OS ---
    if command -v yum &>/dev/null; then
        PKG_MGR="yum"
    elif command -v apt-get &>/dev/null; then
        PKG_MGR="apt-get"
    else
        die "Unknown package manager. Expected yum (Amazon Linux) or apt-get (Ubuntu)."
    fi
    log "Package manager: $PKG_MGR"

    # --- Install Node.js (for Claude Code, optional) ---
    if command -v node &>/dev/null; then
        log "Node.js already installed: $(node --version)"
    else
        log "Installing Node.js 22.x..."
        if [[ "$PKG_MGR" == "yum" ]]; then
            curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
            sudo yum install -y nodejs
        else
            curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
            sudo apt-get install -y nodejs
        fi
    fi

    # --- Install Claude Code (optional) ---
    if command -v claude &>/dev/null; then
        log "Claude Code already installed."
    else
        log "Installing Claude Code..."
        npm install -g @anthropic-ai/claude-code
    fi

    # --- Check we're in the repo ---
    if [[ -f "run_experiments_parallel.py" ]]; then
        log "Already in dash-shap repo."
    elif [[ -d "dash-shap" ]]; then
        log "dash-shap directory exists, entering..."
        cd dash-shap
    else
        log "Cloning repo..."
        git clone "$REPO_URL"
        cd dash-shap
    fi
    check_repo

    # --- Ensure on latest main ---
    log "Updating main branch..."
    git checkout main
    git pull origin main

    # --- Activate git hooks ---
    git config core.hooksPath .githooks

    # --- Install screen if missing ---
    if ! command -v screen &>/dev/null; then
        log "Installing screen..."
        sudo "$PKG_MGR" install -y screen 2>/dev/null || warn "Could not install screen. Use tmux instead."
    fi

    # --- Python environment ---
    log "Installing Python dependencies (pinned versions)..."
    pip install -r requirements.lock
    pip install -e .
    pip install psutil  # for RAM detection in provenance

    # --- Detect instance type ---
    detect_instance_type

    # --- Verify ---
    log "Verifying installation..."
    python -c "from dash_shap import DASHPipeline; print('DASHPipeline import OK')"
    python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB')"
    python -c "import os; print(f'CPUs: {os.cpu_count()}')"

    # --- Quick tests ---
    log "Running fast test suite..."
    pytest -m "not slow" --timeout=60 -q || {
        warn "Some tests failed. Review output above."
    }

    log ""
    log "=== Setup complete ==="
    log ""
    log "Next: bash scripts/sagemaker_run.sh smoke"
}

# ---------------------------------------------------------------------------
# Phase 2: smoke — validate serialization before committing to a long run
# ---------------------------------------------------------------------------
do_smoke() {
    log "=== Phase 2: Smoke Test ==="
    check_repo

    log "Validating full serialization pipeline..."
    if python run_experiments_parallel.py --smoke --experiments linear_sweep; then
        log "Smoke test PASSED."
        rm -f results/tables/synthetic_linear_sweep.json
        rm -f results/checkpoints/linear_sweep_*
        rm -f results/PROVENANCE.md
    else
        die "Smoke test FAILED. Fix the error above before starting a long run."
    fi

    log ""
    log "Next: bash scripts/sagemaker_run.sh branch"
}

# ---------------------------------------------------------------------------
# Phase 3: branch — create results branch from clean main
# ---------------------------------------------------------------------------
do_branch() {
    log "=== Phase 3: Create Results Branch ==="
    check_repo

    git checkout main
    git pull origin main

    # Verify code is clean
    if [[ -n "$(git status --porcelain dash_shap/ tests/ run_experiments_parallel.py 2>/dev/null)" ]]; then
        die "Uncommitted code changes detected. Commit or stash them first."
    fi

    # Create or switch to branch
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}" 2>/dev/null; then
        warn "Branch ${BRANCH_NAME} already exists locally."
        git checkout "$BRANCH_NAME"
    else
        log "Creating branch: ${BRANCH_NAME}"
        git checkout -b "$BRANCH_NAME"
    fi

    # Clean old results so only this run's output is in the directory
    log "Cleaning old result artifacts..."
    rm -f results/tables/*.json results/figures/* results/environment.json results/PROVENANCE.md 2>/dev/null || true
    rm -rf results/checkpoints/* 2>/dev/null || true

    # Tag start
    if git rev-parse "$TAG_START" &>/dev/null; then
        warn "Tag ${TAG_START} already exists. Skipping."
    else
        log "Tagging start: ${TAG_START}"
        git tag "$TAG_START" "$(git rev-parse HEAD)"
        git push origin "$TAG_START"
    fi

    # Push branch
    git push -u origin "$BRANCH_NAME" 2>/dev/null || true

    log ""
    log "=== Branch ready ==="
    log "  Branch: ${BRANCH_NAME}"
    log "  Tag:    ${TAG_START} -> $(git rev-parse --short HEAD)"
    log "  Code:   $(git rev-parse --short HEAD) (clean)"
    log ""
    log "Next: bash scripts/sagemaker_run.sh run"
}

# ---------------------------------------------------------------------------
# Phase 4: run — start experiments in a screen session
# ---------------------------------------------------------------------------
do_run() {
    log "=== Phase 4: Run Experiments ==="
    check_repo

    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != results/* ]]; then
        die "Not on a results branch (on: ${CURRENT_BRANCH}). Run 'bash scripts/sagemaker_run.sh branch' first."
    fi

    # --- Verify instance type is set ---
    detect_instance_type
    if [[ -z "${SM_CURRENT_INSTANCE_TYPE:-}" ]]; then
        warn "SM_CURRENT_INSTANCE_TYPE not set — provenance will be incomplete."
        read -rp "Continue anyway? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || die "Aborted."
    fi

    # --- Verify finalize script exists ---
    if [[ ! -f "scripts/sagemaker_finalize_run.sh" ]]; then
        die "scripts/sagemaker_finalize_run.sh not found. Cannot finalize without it."
    fi

    # --- Kill stale screen sessions ---
    local stale
    stale=$(screen -ls 2>/dev/null | grep -c "dash-" || true)
    if [[ "$stale" -gt 0 ]]; then
        warn "Found $stale existing dash-* screen session(s). Killing..."
        screen -ls | grep "dash-" | awk -F. '{print $1}' | xargs -I{} screen -X -S {} quit 2>/dev/null || true
    fi

    # --- Print hardware summary ---
    python -c "
import os, platform
try:
    import psutil
    ram = f'{psutil.virtual_memory().total / 1024**3:.1f} GB'
except ImportError:
    ram = 'unknown'
print(f'  CPUs:     {os.cpu_count()}')
print(f'  RAM:      {ram}')
print(f'  Platform: {platform.platform()}')
print(f'  Python:   {platform.python_version()}')
print(f'  Instance: {os.environ.get(\"SM_CURRENT_INSTANCE_TYPE\", \"NOT SET\")}')
"

    log ""
    log "Branch: ${CURRENT_BRANCH}"
    log "Experiments: ${EXPERIMENTS[*]}"
    log ""
    log "Starting in screen session '${SCREEN_NAME}'..."
    log "Detach: Ctrl-A, D  |  Reattach: screen -d -r ${SCREEN_NAME}"
    log "Monitor: bash scripts/sagemaker_run.sh status"
    log ""

    # --- Launch in screen ---
    screen -dmS "$SCREEN_NAME" bash -c "
        export SM_CURRENT_INSTANCE_TYPE='${SM_CURRENT_INSTANCE_TYPE:-}'
        export PYTHONWARNINGS='ignore::FutureWarning'
        cd $(pwd)
        python run_experiments_parallel.py \
            --resume \
            --no-cleanup \
            --experiments ${EXPERIMENTS[*]}
        echo ''
        echo '=== ALL EXPERIMENTS COMPLETE ==='
        echo 'Run: bash scripts/sagemaker_run.sh finish'
        echo 'Press Enter to close this screen session.'
        read
    "

    log "Experiments launched in screen '${SCREEN_NAME}'."
    log ""
    log "  Reattach:  screen -d -r ${SCREEN_NAME}"
    log "  Status:    bash scripts/sagemaker_run.sh status"
    log "  Finalize:  bash scripts/sagemaker_run.sh finish  (after completion)"
}

# ---------------------------------------------------------------------------
# Phase 5: finish — verify metadata, commit, tag, push
# ---------------------------------------------------------------------------
do_finish() {
    log "=== Phase 5: Finalize Results ==="
    check_repo

    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != results/* ]]; then
        die "Not on a results branch (on: ${CURRENT_BRANCH})."
    fi

    # --- Verify experiments aren't still running ---
    if pgrep -f run_experiments_parallel > /dev/null 2>&1; then
        warn "run_experiments_parallel is still running!"
        read -rp "Continue anyway? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || die "Wait for experiments to finish."
    fi

    # --- Show results ---
    log "Result files:"
    ls -lht results/tables/*.json 2>/dev/null || echo "  (none found)"
    echo ""

    # --- Check metadata completeness ---
    log "Checking _meta blocks..."
    local missing_meta=0
    for f in results/tables/*.json; do
        [[ -f "$f" ]] || continue
        has_meta=$(python3 -c "import json; d=json.load(open('$f')); print('yes' if '_meta' in d else 'no')" 2>/dev/null || echo "error")
        if [[ "$has_meta" != "yes" ]]; then
            warn "Missing _meta in $f"
            missing_meta=1
        fi
    done

    if [[ "$missing_meta" -eq 1 ]]; then
        log "Running backfill_meta.py for files missing _meta..."
        python scripts/backfill_meta.py
    else
        log "All result files have _meta blocks. Skipping backfill."
    fi

    # --- Stage and show ---
    git add results/
    echo ""
    git status --short results/

    # --- Count results ---
    local n_results
    n_results=$(find results/tables -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    log "Result files: ${n_results}"

    # --- Confirm ---
    read -rp "Commit these results? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || die "Aborted."

    # --- Commit ---
    INSTANCE="${SM_CURRENT_INSTANCE_TYPE:-unknown}"
    CODE_SHA=$(git rev-parse --short HEAD)
    EXPERIMENT_LIST=$(printf '%s, ' "${EXPERIMENTS[@]}")
    EXPERIMENT_LIST="${EXPERIMENT_LIST%, }"

    git commit -m "data: SageMaker run ${RUN_DATE} — ${#EXPERIMENTS[@]} experiments (N_REPS=50)

Instance: ${INSTANCE}
Code SHA: ${CODE_SHA}
Experiments: ${EXPERIMENT_LIST}"

    # --- Tag end ---
    if git rev-parse "$TAG_END" &>/dev/null; then
        warn "Tag ${TAG_END} already exists. Skipping."
    else
        log "Tagging end: ${TAG_END}"
        git tag "$TAG_END" "$(git rev-parse HEAD)"
    fi

    # --- Push ---
    log "Pushing branch and tags..."
    git push origin "$CURRENT_BRANCH" --tags

    log ""
    log "=== Run finalized ==="
    log "  Branch: ${CURRENT_BRANCH}"
    log "  Start:  ${TAG_START}"
    log "  End:    ${TAG_END} -> $(git rev-parse --short HEAD)"
    log ""
    log "Next: on your workstation, open a data-only PR:"
    log "  gh pr create --base main --head ${CURRENT_BRANCH} \\"
    log "    --title 'data: SageMaker run ${RUN_DATE}' \\"
    log "    --body 'N_REPS=50, ${#EXPERIMENTS[@]} experiments, ${INSTANCE}'"
}

# ---------------------------------------------------------------------------
# Status check (safe to run anytime, no side effects)
# ---------------------------------------------------------------------------
do_status() {
    log "=== Run Status ==="

    if [[ ! -f "run_experiments_parallel.py" ]]; then
        warn "Not in dash-shap repo."
        return
    fi

    echo ""
    echo "Branch: $(git branch --show-current)"
    echo "Commit: $(git rev-parse --short HEAD)"
    echo ""

    echo "Completed experiments:"
    if [[ -d results/tables ]]; then
        ls -lht results/tables/*.json 2>/dev/null || echo "  (none)"
    else
        echo "  (results/tables/ not found)"
    fi

    echo ""
    echo "Runner process:"
    pgrep -af run_experiments_parallel || echo "  (not running)"

    echo ""
    echo "Screen sessions:"
    screen -ls 2>/dev/null | grep "dash-" || echo "  (none)"

    echo ""
    echo "Active python workers:"
    local workers
    workers=$(pgrep -c -f python 2>/dev/null || echo "0")
    echo "  $workers python processes"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${1:-help}" in
    setup)   do_setup   ;;
    smoke)   do_smoke   ;;
    branch)  do_branch  ;;
    run)     do_run     ;;
    finish)  do_finish  ;;
    status)  do_status  ;;
    help|*)
        echo "DASH-SHAP SageMaker Experiment Runner"
        echo ""
        echo "Usage: bash scripts/sagemaker_run.sh <phase>"
        echo ""
        echo "Phases (run in order):"
        echo "  setup    Install deps, verify environment"
        echo "  smoke    Validate serialization pipeline (~1 second)"
        echo "  branch   Clean results, create results branch + start tag"
        echo "  run      Launch experiments in a named screen session"
        echo "  finish   Verify metadata, commit, tag, push"
        echo "  status   Check progress (safe anytime)"
        echo ""
        echo "Quick start on a fresh SageMaker instance:"
        echo "  cd ~/SageMaker"
        echo "  git clone https://github.com/DrakeCaraker/dash-shap.git"
        echo "  cd dash-shap"
        echo "  export SM_CURRENT_INSTANCE_TYPE='ml.g5.16xlarge'"
        echo "  bash scripts/sagemaker_run.sh setup"
        echo "  bash scripts/sagemaker_run.sh smoke"
        echo "  bash scripts/sagemaker_run.sh branch"
        echo "  bash scripts/sagemaker_run.sh run"
        echo "  bash scripts/sagemaker_run.sh status"
        echo "  bash scripts/sagemaker_run.sh finish"
        ;;
esac
