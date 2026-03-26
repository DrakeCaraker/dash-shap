#!/usr/bin/env bash
# =============================================================================
# DASH-SHAP SageMaker Experiment Runner
# =============================================================================
#
# Usage:
#   1. Open a terminal on your SageMaker notebook instance
#   2. Run:  bash sagemaker_run.sh setup
#   3. Run:  bash sagemaker_run.sh run
#   4. After completion:  bash sagemaker_run.sh finish
#
# Each phase is idempotent — safe to re-run if interrupted.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/DrakeCaraker/dash-shap.git"
RUN_DATE=$(date +%Y%m%d)
BRANCH_NAME="results/sagemaker-run-${RUN_DATE}"
TAG_START="run-tmlr-${RUN_DATE}-start"
TAG_END="run-tmlr-${RUN_DATE}-end"

# Experiments to run (prior run completed: linear_sweep, first_mover_visualization,
# overlapping, k_sweep_independence — but we re-run linear_sweep for clean provenance
# and because success_criteria auto-evaluates when it's in the list)
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

# ---------------------------------------------------------------------------
# Phase 1: setup
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

    # --- Install Node.js (for Claude Code) ---
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

    # --- Install Claude Code ---
    if command -v claude &>/dev/null; then
        log "Claude Code already installed: $(claude --version 2>/dev/null || echo 'installed')"
    else
        log "Installing Claude Code..."
        npm install -g @anthropic-ai/claude-code
    fi

    # --- Check API key ---
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        warn "ANTHROPIC_API_KEY not set. Claude Code needs it for headless auth."
        warn "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
        warn "Add it to ~/.bashrc to persist across sessions."
    else
        log "ANTHROPIC_API_KEY is set."
    fi

    # --- Clone repo if needed ---
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

    # --- Python environment ---
    log "Installing Python dependencies (pinned versions)..."
    pip install -r requirements.lock
    pip install -e .
    pip install psutil  # for RAM detection in provenance

    # --- Verify ---
    log "Verifying installation..."
    python -c "from dash_shap import DASHPipeline; print('DASHPipeline import OK')"
    python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB')"
    python -c "import os; print(f'CPUs: {os.cpu_count()}')"

    # --- Set instance type env var ---
    # SM_CURRENT_INSTANCE_TYPE is NOT auto-set on notebook instances
    # (only on Training Jobs). Detect or prompt.
    if [[ -z "${SM_CURRENT_INSTANCE_TYPE:-}" ]]; then
        # Try to detect from instance metadata
        INSTANCE_TYPE=$(curl -s --max-time 2 \
            http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "")
        if [[ -n "$INSTANCE_TYPE" ]]; then
            # SageMaker prefixes with ml.
            export SM_CURRENT_INSTANCE_TYPE="ml.${INSTANCE_TYPE}"
            log "Detected instance type: $SM_CURRENT_INSTANCE_TYPE"
        else
            warn "Could not detect instance type."
            warn "Set it manually: export SM_CURRENT_INSTANCE_TYPE='ml.m5.16xlarge'"
        fi
    else
        log "Instance type: $SM_CURRENT_INSTANCE_TYPE"
    fi

    # --- Quick tests ---
    log "Running fast test suite..."
    pytest -m "not slow" --timeout=60 -q || {
        warn "Some tests failed. Review output above. Experiments may still run."
    }

    log ""
    log "=== Setup complete ==="
    log ""
    log "Next steps:"
    log "  1. Verify SM_CURRENT_INSTANCE_TYPE is set (for provenance)"
    log "  2. Run: bash scripts/sagemaker_run.sh branch"
    log "  3. Run: bash scripts/sagemaker_run.sh run"
}

# ---------------------------------------------------------------------------
# Phase 2: create branch and tag
# ---------------------------------------------------------------------------
do_branch() {
    log "=== Phase 2: Create Results Branch ==="
    check_repo

    git checkout main
    git pull origin main

    # Check if branch already exists
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}" 2>/dev/null; then
        warn "Branch ${BRANCH_NAME} already exists locally."
        warn "If this is a re-run on the same day, that's fine."
        git checkout "$BRANCH_NAME"
    else
        log "Creating branch: ${BRANCH_NAME}"
        git checkout -b "$BRANCH_NAME"
    fi

    # Tag start (skip if tag exists)
    if git rev-parse "$TAG_START" &>/dev/null; then
        warn "Tag ${TAG_START} already exists. Skipping."
    else
        log "Tagging start: ${TAG_START}"
        git tag "$TAG_START" "$(git rev-parse HEAD)"
        git push origin "$TAG_START"
    fi

    log ""
    log "=== Branch ready ==="
    log "  Branch: ${BRANCH_NAME}"
    log "  Tag:    ${TAG_START} -> $(git rev-parse --short HEAD)"
    log "  Code:   $(git rev-parse --short HEAD)"
    log ""
    log "Next: bash scripts/sagemaker_run.sh run"
}

# ---------------------------------------------------------------------------
# Phase 3: run experiments
# ---------------------------------------------------------------------------
do_run() {
    log "=== Phase 3: Run Experiments ==="
    check_repo

    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != results/* ]]; then
        die "Not on a results branch (on: ${CURRENT_BRANCH}). Run 'bash scripts/sagemaker_run.sh branch' first."
    fi

    log "Branch: ${CURRENT_BRANCH}"
    log "Experiments: ${EXPERIMENTS[*]}"
    log "Resume mode: enabled (--resume)"
    log ""

    # Print hardware summary
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

    if [[ -z "${SM_CURRENT_INSTANCE_TYPE:-}" ]]; then
        warn "SM_CURRENT_INSTANCE_TYPE not set — provenance will be incomplete."
        warn "Set it now: export SM_CURRENT_INSTANCE_TYPE='ml.m5.16xlarge'"
        read -rp "Continue anyway? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || die "Aborted. Set the env var and re-run."
    fi

    log ""
    log "Starting experiments. Use Ctrl-A,D to detach if in screen/tmux."
    log "Re-run this command with --resume to pick up from checkpoints."
    log ""

    python run_experiments_parallel.py \
        --resume \
        --no-cleanup \
        --experiments "${EXPERIMENTS[@]}"

    log ""
    log "=== All experiments complete ==="
    log "Next: bash scripts/sagemaker_run.sh finish"
}

# ---------------------------------------------------------------------------
# Phase 4: finalize — backfill, commit, tag, push
# ---------------------------------------------------------------------------
do_finish() {
    log "=== Phase 4: Finalize Results ==="
    check_repo

    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != results/* ]]; then
        die "Not on a results branch (on: ${CURRENT_BRANCH})."
    fi

    # --- Backfill provenance metadata ---
    log "Backfilling provenance metadata into result JSONs..."
    python scripts/backfill_meta.py

    # --- Show what we're committing ---
    log "Results to commit:"
    git status --short results/

    # --- Count result files ---
    N_RESULTS=$(find results/tables -name '*.json' -newer results/environment.json 2>/dev/null | wc -l | tr -d ' ')
    log "New/updated result files: ${N_RESULTS}"

    # --- Commit ---
    INSTANCE="${SM_CURRENT_INSTANCE_TYPE:-unknown}"
    CODE_SHA=$(git rev-parse --short HEAD)
    EXPERIMENT_LIST=$(printf '%s, ' "${EXPERIMENTS[@]}")
    EXPERIMENT_LIST="${EXPERIMENT_LIST%, }"

    git add results/
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
    git push origin "$CURRENT_BRANCH"
    git push origin "$TAG_END" 2>/dev/null || true

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
# Phase 5: status check (safe to run anytime)
# ---------------------------------------------------------------------------
do_status() {
    log "=== Run Status ==="

    if [[ -f "run_experiments_parallel.py" ]]; then
        echo ""
        echo "Branch: $(git branch --show-current)"
        echo "Commit: $(git rev-parse --short HEAD)"
        echo ""

        echo "Result files:"
        if [[ -d results/tables ]]; then
            ls -lht results/tables/*.json 2>/dev/null || echo "  (none)"
        else
            echo "  (results/tables/ not found)"
        fi

        echo ""
        echo "Running processes:"
        pgrep -af run_experiments_parallel || echo "  (none)"

        echo ""
        echo "Checkpoints:"
        if [[ -d results/checkpoints ]]; then
            ls results/checkpoints/ 2>/dev/null | head -20 || echo "  (none)"
        else
            echo "  (no checkpoint directory)"
        fi
    else
        warn "Not in dash-shap repo."
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${1:-help}" in
    setup)   do_setup   ;;
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
        echo "  setup    Install deps, clone repo, verify environment"
        echo "  branch   Create results branch and start tag"
        echo "  run      Run experiments (use inside screen/tmux)"
        echo "  finish   Backfill metadata, commit, tag, push"
        echo "  status   Check progress (safe anytime)"
        echo ""
        echo "Quick start:"
        echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
        echo "  export SM_CURRENT_INSTANCE_TYPE='ml.m5.16xlarge'"
        echo "  bash scripts/sagemaker_run.sh setup"
        echo "  bash scripts/sagemaker_run.sh branch"
        echo "  screen -S dash-run"
        echo "  bash scripts/sagemaker_run.sh run"
        echo "  # Ctrl-A, D to detach"
        echo "  bash scripts/sagemaker_run.sh status   # check progress"
        echo "  bash scripts/sagemaker_run.sh finish    # after completion"
        ;;
esac
