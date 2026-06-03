#!/usr/bin/env bash
: "${GPU_HOST:=gpu-node}"; : "${GPU_USER:=researcher}"  # override via env for your infra
# ============================================================================
# run_headline_campaign.sh
# ----------------------------------------------------------------------------
# Sequential dispatch of the four headline runs across Papers 1 and 3, with
# preflight checks designed to preempt the failure modes seen in prior
# campaigns. Designed to be run under tmux/screen so it survives terminal
# disconnects, laptop sleep, or network drops.
#
# The four jobs (in order):
#   1. p1_ld_uuf                       (256² LD baseline,        ~1 hr)
#   2. p3_rod_plus_outer_p7li_uuf      (256² hybrid headline,    ~1 hr)
#   3. p1_ld_uuf_512                   (512² LD convergence,     ~4.5 hr)
#   4. p3_rod_plus_outer_p7li_512      (512² hybrid convergence, ~4.5 hr)
#
# Total expected wall clock: ~11 hours (one job at a time; no parallelism).
#
# USAGE:
#   cd ~/LaserFusionResearch/research/laser-plasma-research/
#   tmux new -s headline
#   ./run_headline_campaign.sh
#   # Ctrl+B then D to detach. Reattach with: tmux attach -t headline
#
# IDEMPOTENT: re-running after partial success only runs the remaining jobs.
# Existing successful jobs are detected via local artifact presence and
# skipped (override with --force).
# ============================================================================

set -u   # unset variable = error
# Note: we deliberately do NOT use `set -e` because individual job failures
# should not abort the campaign — we want job 2 to run even if job 1 failed.
# Failures are tracked explicitly via $RESULTS array and reported at the end.

# ─── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR="${HOME}/LaserFusionResearch/research/laser-plasma-research"
LOG_DIR_BASE="${PROJECT_DIR}/campaign_logs"
STATUS_FILE="${HOME}/.headline_campaign_status"

# Expected md5 hashes of the deployed orchestrators + visualize_all.py.
# If actual md5s differ, the user has not deployed today's patches and the
# campaign refuses to start (because the cleanup-policy change and the
# histogram extraction are both required for the data we're trying to collect).
EXPECTED_MD5_PAPER01="c92eb23ff8d28aff24c8e8a9d796abcf"
EXPECTED_MD5_PAPER3="70deb1c1eebc7c75c2429c0a06a39f3f"
EXPECTED_MD5_VISUALIZE="3f73f9e8145039d35a08c92dcf843597"

# Per-job wall-clock ceiling (seconds). Acts as a hard cap on top of the
# orchestrator's 30 min no-progress watchdog. If a job exceeds this, the
# campaign kills it and proceeds to the next job.
TIMEOUT_256_SEC=7200       # 2 hr for 256² jobs (typical: 50-90 min)
TIMEOUT_512_SEC=28800      # 8 hr for 512² jobs (typical: 4-5 hr)

# Cloud-disk thresholds (bytes; df returns 1K blocks so divide by 1).
# Approximate: 1.3 TB per 512² run + headroom = 2 TB min before 512² launch
MIN_FREE_KB_256="200000000"      # 200 GB
MIN_FREE_KB_512="2000000000"     # 2 TB

# Bash colors for terminal output (gracefully no-op'd if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
    BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; RESET=''
fi

# ─── Argument parsing ───────────────────────────────────────────────────────
FORCE=0
SKIP_PREFLIGHT=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift ;;
        --skip-preflight)
            SKIP_PREFLIGHT=1
            shift ;;
        --dry-run)
            DRY_RUN=1
            shift ;;
        -h|--help)
            cat <<EOF
run_headline_campaign.sh — sequential dispatch of the 4 headline runs.

Options:
  --force            Re-run jobs even if local artifacts indicate success.
  --skip-preflight   Skip the cloud/disk/md5 preflight (NOT RECOMMENDED).
  --dry-run          Print what would be done, but don't execute.
  -h, --help         This help.

Status file: ${STATUS_FILE}
Logs:        ${LOG_DIR_BASE}/<timestamp>/

Run under tmux so it survives disconnects:
  tmux new -s headline
  ./run_headline_campaign.sh
  # detach: Ctrl+B then D
  # reattach: tmux attach -t headline
EOF
            exit 0 ;;
        *)
            echo "Unknown arg: $1 (use -h for help)"
            exit 2 ;;
    esac
done

# ─── Helpers ────────────────────────────────────────────────────────────────
ts()      { date +'%Y-%m-%d %H:%M:%S'; }
hms()     { local s=$1; printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60)); }
banner()  { echo ""; echo -e "${BOLD}${BLUE}============================================================${RESET}"; echo -e "${BOLD}${BLUE} $1${RESET}"; echo -e "${BOLD}${BLUE}============================================================${RESET}"; }
ok()      { echo -e "${GREEN}[OK]${RESET} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $1"; }
err()     { echo -e "${RED}[FAIL]${RESET} $1"; }
info()    { echo -e "${BLUE}[INFO]${RESET} $1"; }

# Portable md5 (Linux has md5sum, macOS has md5 with different output format).
# Returns just the 32-char hex digest.
md5_of() {
    local f=$1
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$f" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        # macOS: 'md5 file' produces 'MD5 (file) = <hash>'
        # 'md5 -q file' produces just '<hash>'
        md5 -q "$f"
    else
        echo "ERROR_NO_MD5_TOOL"
        return 1
    fi
}

update_status() {
    # update the status file (atomic write)
    local tmp="${STATUS_FILE}.tmp"
    {
        echo "campaign:    $1"
        echo "phase:       $2"
        echo "current:     $3"
        echo "last_update: $(ts)"
        echo "pid:         $$"
        echo ""
        echo "--- per-job results ---"
        # Guard against empty array under `set -u`: "${arr[@]:-}" expands to
        # a single empty element when arr is empty, so filter empties.
        for entry in "${RESULTS[@]:-}"; do
            [[ -n "$entry" ]] && echo "  $entry"
        done
    } > "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

cleanup_on_exit() {
    # shellcheck disable=SC2317  # invoked via 'trap', not directly
    local rc=$?
    # shellcheck disable=SC2317
    update_status "headline_campaign" "EXITED" "rc=${rc}"
    # shellcheck disable=SC2317
    if [[ $rc -ne 0 ]]; then
        # shellcheck disable=SC2317
        err "Campaign exited abnormally with rc=$rc"
    fi
}
trap cleanup_on_exit EXIT
trap 'err "Interrupted by user"; exit 130' INT TERM

# ─── Preflight checks ───────────────────────────────────────────────────────
preflight() {
    banner "Preflight checks"

    # 1. We're in the right directory
    if [[ ! -d "$PROJECT_DIR" ]]; then
        err "Project dir not found: $PROJECT_DIR"
        exit 1
    fi
    cd "$PROJECT_DIR" || exit 1
    ok "Project dir: $PROJECT_DIR"

    # 2. Required local files exist
    #    Orchestrators sit at project root; visualize_all.py lives in
    #    analysis_scripts/ where it's invoked by run_paper{1,3}_chain.sh
    local viz_path="analysis_scripts/visualize_all.py"
    for f in stage_a_paper01.py stage_a_paper3.py "$viz_path"; do
        if [[ ! -f "$f" ]]; then
            err "Missing required file: $f"
            exit 1
        fi
    done
    ok "Required files present locally"

    # 3. Local file md5s match expected (else patches not deployed)
    local local_md5_p1 local_md5_p3 local_md5_viz
    local_md5_p1=$(md5_of stage_a_paper01.py)
    local_md5_p3=$(md5_of stage_a_paper3.py)
    local_md5_viz=$(md5_of "$viz_path")

    if [[ "$local_md5_p1" == "ERROR_NO_MD5_TOOL" ]]; then
        err "No md5 tool available (neither md5sum nor md5 found)."
        err "  Install one: 'brew install coreutils' on macOS gives you md5sum."
        exit 1
    fi

    local md5_ok=1
    if [[ "$local_md5_p1" != "$EXPECTED_MD5_PAPER01" ]]; then
        warn "stage_a_paper01.py md5 mismatch:"
        warn "    expected: $EXPECTED_MD5_PAPER01"
        warn "    actual:   $local_md5_p1"
        md5_ok=0
    fi
    if [[ "$local_md5_p3" != "$EXPECTED_MD5_PAPER3" ]]; then
        warn "stage_a_paper3.py md5 mismatch:"
        warn "    expected: $EXPECTED_MD5_PAPER3"
        warn "    actual:   $local_md5_p3"
        md5_ok=0
    fi
    if [[ "$local_md5_viz" != "$EXPECTED_MD5_VISUALIZE" ]]; then
        warn "$viz_path md5 mismatch:"
        warn "    expected: $EXPECTED_MD5_VISUALIZE"
        warn "    actual:   $local_md5_viz"
        md5_ok=0
    fi

    if [[ $md5_ok -eq 1 ]]; then
        ok "All deployed patches match expected md5"
    else
        if [[ $SKIP_PREFLIGHT -eq 1 ]]; then
            warn "md5 mismatches present but --skip-preflight set; continuing"
        else
            err "Deploy today's patches before running this campaign."
            err "  cp ~/Downloads/stage_a_paper01.py ."
            err "  cp ~/Downloads/stage_a_paper3.py ."
            err "  cp ~/Downloads/visualize_all.py analysis_scripts/"
            err "Or override with --skip-preflight (NOT recommended)."
            exit 1
        fi
    fi

    # 4. Cloud reachable
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes $GPU_HOST 'echo ping' >/dev/null 2>&1; then
        err "Cannot SSH to $GPU_HOST"
        exit 1
    fi
    ok "Cloud SSH reachable"

    # 4b. CRITICAL: visualize_all.py on CLOUD matches expected md5.
    # The analysis chain runs on the cloud and invokes the cloud's copy of
    # visualize_all.py. If user updated locally but forgot to rsync, the
    # campaign will appear to succeed but won't produce proton_energy_histogram.csv
    # because the cloud will run the OLD visualize_all.py.
    local cloud_md5_viz
    cloud_md5_viz=$(ssh $GPU_HOST \
        "md5sum ~/laser-plasma-research/analysis_scripts/visualize_all.py 2>/dev/null | awk '{print \$1}'" \
        || echo "MISSING")
    if [[ "$cloud_md5_viz" == "MISSING" ]] || [[ -z "$cloud_md5_viz" ]]; then
        err "Cloud's analysis_scripts/visualize_all.py not found or unreadable."
        err "  Push it:"
        err "    rsync -avh analysis_scripts/visualize_all.py $GPU_HOST:~/laser-plasma-research/analysis_scripts/"
        exit 1
    fi
    if [[ "$cloud_md5_viz" != "$EXPECTED_MD5_VISUALIZE" ]]; then
        err "Cloud's visualize_all.py is OUT OF DATE."
        err "  Expected md5: $EXPECTED_MD5_VISUALIZE"
        err "  Cloud md5:    $cloud_md5_viz"
        err ""
        err "  Push the local copy:"
        err "    rsync -avh analysis_scripts/visualize_all.py $GPU_HOST:~/laser-plasma-research/analysis_scripts/"
        err ""
        err "  Without this, the campaign will run but the headline figures"
        err "  and proton_energy_histogram.csv will NOT be produced."
        if [[ $SKIP_PREFLIGHT -eq 1 ]]; then
            warn "Cloud md5 mismatch but --skip-preflight set; continuing"
        else
            exit 1
        fi
    else
        ok "Cloud visualize_all.py md5 matches expected ($cloud_md5_viz)"
    fi

    # 4c. Also verify cloud orchestrator copies (used when you re-dispatch
    # via the cloud's run_paper{1,3}_chain.sh directly — less critical, but
    # detects drift).
    local cloud_md5_p1 cloud_md5_p3
    cloud_md5_p1=$(ssh $GPU_HOST \
        "md5sum ~/laser-plasma-research/stage_a_paper01.py 2>/dev/null | awk '{print \$1}'" \
        || echo "MISSING")
    cloud_md5_p3=$(ssh $GPU_HOST \
        "md5sum ~/laser-plasma-research/stage_a_paper3.py 2>/dev/null | awk '{print \$1}'" \
        || echo "MISSING")
    # These orchestrators run from the LAPTOP not the cloud, so cloud copies
    # are advisory rather than blocking. Just warn if drift detected.
    if [[ "$cloud_md5_p1" != "$EXPECTED_MD5_PAPER01" ]] && [[ "$cloud_md5_p1" != "MISSING" ]]; then
        warn "Cloud's stage_a_paper01.py differs from local (advisory only — "
        warn "  orchestrator runs from laptop, but if you ever dispatch from cloud,"
        warn "  rsync it: rsync -avh stage_a_paper01.py $GPU_HOST:~/laser-plasma-research/)"
    fi
    if [[ "$cloud_md5_p3" != "$EXPECTED_MD5_PAPER3" ]] && [[ "$cloud_md5_p3" != "MISSING" ]]; then
        warn "Cloud's stage_a_paper3.py differs from local (advisory only)"
    fi

    # 5. Cloud has the LATE_PERIOD_OVERRIDE patch in place (production branch)
    local cloud_late
    cloud_late=$(ssh $GPU_HOST \
        "grep -E '^\s*LATE_PERIOD_OVERRIDE\s*=\s*' \
         ~/laser-plasma-research/simulation/pb11_ring_reconnection_v15_pulsed.py \
         2>/dev/null | head -3" || echo "")
    if [[ -z "$cloud_late" ]]; then
        warn "Could not grep LATE_PERIOD_OVERRIDE from cloud sim script"
        warn "  (this is the patch that prevents step-5000 OOM at production DT)"
        if [[ $SKIP_PREFLIGHT -ne 1 ]]; then
            err "Aborting. Use --skip-preflight to override (NOT recommended)."
            exit 1
        fi
    else
        # We need to see 99999 (or > 7700) on the production branch
        if echo "$cloud_late" | grep -qE 'LATE_PERIOD_OVERRIDE\s*=\s*99999'; then
            ok "Cloud LATE_PERIOD_OVERRIDE = 99999 (patched)"
        elif echo "$cloud_late" | grep -qE 'LATE_PERIOD_OVERRIDE\s*=\s*5000'; then
            err "Cloud LATE_PERIOD_OVERRIDE = 5000 — this WILL cause step-5000 OOM!"
            err "Patch it on cloud first:"
            err "  ssh $GPU_HOST \"sed -i.bak 's/LATE_PERIOD_OVERRIDE   = 5000/LATE_PERIOD_OVERRIDE   = 99999/' ~/laser-plasma-research/simulation/pb11_ring_reconnection_v15_pulsed.py\""
            exit 1
        else
            warn "LATE_PERIOD_OVERRIDE found but unexpected value:"
            # shellcheck disable=SC2001  # sed is more readable for multi-line indent
            echo "$cloud_late" | sed 's/^/    /'
            if [[ $SKIP_PREFLIGHT -ne 1 ]]; then
                err "Verify the value is acceptable, then use --skip-preflight"
                exit 1
            fi
        fi
    fi

    # 6. Cloud disk has enough free space for the biggest job (512²)
    local cloud_free_kb
    cloud_free_kb=$(ssh $GPU_HOST \
        "df ~/laser-plasma-research | tail -1 | awk '{print \$4}'" 2>/dev/null || echo "0")
    if [[ "$cloud_free_kb" =~ ^[0-9]+$ ]] && [[ $cloud_free_kb -gt $MIN_FREE_KB_512 ]]; then
        local cloud_free_tb
        cloud_free_tb=$(awk "BEGIN{printf \"%.2f\", $cloud_free_kb / 1024 / 1024 / 1024}")
        ok "Cloud disk free: ${cloud_free_tb} TB (>2 TB required for 512²)"
    else
        warn "Cloud disk free: ${cloud_free_kb} KB (less than 2 TB!)"
        warn "  Per-job preflight will re-check before each launch."
    fi

    # 7. GPU is idle (or close to it)
    local gpu_mem_mb
    gpu_mem_mb=$(ssh $GPU_HOST \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1" \
        || echo "?")
    if [[ "$gpu_mem_mb" =~ ^[0-9]+$ ]]; then
        if [[ $gpu_mem_mb -lt 2000 ]]; then
            ok "GPU memory used: ${gpu_mem_mb} MB (effectively idle)"
        elif [[ $gpu_mem_mb -lt 20000 ]]; then
            warn "GPU memory used: ${gpu_mem_mb} MB (something is using GPU)"
            warn "  Check: ssh $GPU_HOST 'nvidia-smi'"
            warn "  If a stuck process, kill it before proceeding."
            if [[ $SKIP_PREFLIGHT -ne 1 ]]; then
                err "Aborting. Use --skip-preflight to override."
                exit 1
            fi
        else
            err "GPU memory used: ${gpu_mem_mb} MB (heavily occupied)"
            err "  A previous job appears still running. Check:"
            err "    ssh $GPU_HOST 'nvidia-smi; ps aux | grep -E pb11|warpx | grep -v grep'"
            exit 1
        fi
    fi

    # 8. No orphan orchestrator processes on cloud
    local cloud_procs
    cloud_procs=$(ssh $GPU_HOST \
        "ps aux | grep -E 'pb11_ring_reconnection|run_paper|run_full_analysis' | grep -v grep | wc -l" \
        2>/dev/null || echo "0")
    if [[ "$cloud_procs" =~ ^[0-9]+$ ]] && [[ $cloud_procs -gt 0 ]]; then
        err "Found $cloud_procs orchestrator/sim/analysis process(es) on cloud:"
        ssh $GPU_HOST "ps aux | grep -E 'pb11_ring_reconnection|run_paper|run_full_analysis' | grep -v grep" | sed 's/^/    /'
        err "Kill them before proceeding, or wait for them to finish."
        exit 1
    fi
    ok "No stale processes on cloud"

    # 9. Confirm tmux (warn only — not enforced)
    if [[ -z "${TMUX:-}" ]]; then
        warn "Not running inside tmux. If your terminal dies, the campaign dies."
        warn "  Recommended: exit, run 'tmux new -s headline', then re-run this script."
        warn "  Pressing Ctrl+C now is safe. Sleeping 10 sec to give you the option..."
        sleep 10
    else
        ok "Running inside tmux session: ${TMUX#*,}"
    fi

    ok "All preflight checks passed"
}

# ─── Per-job artifact-presence check (for idempotent skip) ──────────────────
job_already_succeeded() {
    # Returns 0 (yes, skip) if the local run dir exists AND contains the key
    # output files (proton_energy_histogram.csv from today's patched
    # visualize_all.py, plus reconnection_summary.txt from Stage A).
    local sub_tag=$1
    local paper=$2   # 'paper01' or 'paper3'
    local run_dir="${PROJECT_DIR}/runs/${paper}/${sub_tag}"

    if [[ ! -d "$run_dir" ]]; then
        return 1
    fi
    # Critical output 1: the new histogram CSV (proves visualize_all.py ran with the patch)
    if [[ ! -f "${run_dir}/proton_energy_histogram.csv" ]]; then
        return 1
    fi
    # Critical output 2: Stage A's reconnection summary (proves sim+analysis completed)
    if [[ ! -f "${run_dir}/reconnection_summary.txt" ]]; then
        return 1
    fi
    # Critical output 3: phase analysis (proves Stage D ran)
    if [[ ! -f "${run_dir}/phase_analysis_report.txt" ]]; then
        return 1
    fi
    return 0
}

# ─── Per-job cloud disk check ───────────────────────────────────────────────
check_cloud_disk_for_job() {
    local grid=$1   # '256' or '512'
    local min_kb
    if [[ "$grid" == "512" ]]; then
        min_kb=$MIN_FREE_KB_512
    else
        min_kb=$MIN_FREE_KB_256
    fi

    local free_kb
    free_kb=$(ssh $GPU_HOST \
        "df ~/laser-plasma-research | tail -1 | awk '{print \$4}'" \
        2>/dev/null || echo "0")
    if [[ ! "$free_kb" =~ ^[0-9]+$ ]] || [[ $free_kb -lt $min_kb ]]; then
        err "Cloud disk too low for ${grid}² job: ${free_kb} KB free (need ${min_kb} KB)"
        return 1
    fi
    local free_tb
    free_tb=$(awk "BEGIN{printf \"%.2f\", $free_kb / 1024 / 1024 / 1024}")
    info "Cloud disk free: ${free_tb} TB (sufficient for ${grid}²)"
    return 0
}

# ─── Run one job with timeout, capturing per-job log ────────────────────────
run_job() {
    local idx=$1               # 1, 2, 3, 4
    local total=$2             # 4
    local sub_tag=$3
    local paper=$4             # 'paper01' or 'paper3'
    local grid=$5              # '256' or '512'
    local timeout_sec=$6

    local script
    if [[ "$paper" == "paper01" ]]; then
        script="stage_a_paper01.py"
    else
        script="stage_a_paper3.py"
    fi

    local job_log="${LOG_DIR}/job_${idx}_${sub_tag}.log"
    local job_start
    job_start=$(date +%s)

    banner "Job [${idx}/${total}]: ${sub_tag}"
    echo "Started:  $(ts)"
    echo "Script:   ${script}"
    echo "Grid:     ${grid}²"
    echo "Timeout:  $(hms "$timeout_sec") (campaign-level ceiling)"
    echo "Log:      ${job_log}"

    update_status "headline_campaign" "RUNNING" "${sub_tag} (job ${idx}/${total})"

    # Idempotent skip check
    if [[ $FORCE -eq 0 ]] && job_already_succeeded "$sub_tag" "$paper"; then
        ok "Already succeeded (local artifacts present); SKIPPING. Use --force to re-run."
        RESULTS+=("${sub_tag}: SKIPPED (already complete)")
        return 0
    fi

    # Per-job cloud disk preflight
    if ! check_cloud_disk_for_job "$grid"; then
        err "Cloud disk preflight failed for $sub_tag"
        RESULTS+=("${sub_tag}: FAILED (cloud disk)")
        return 1
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        info "DRY RUN: would launch: python $script --jobs $sub_tag"
        RESULTS+=("${sub_tag}: DRY_RUN")
        return 0
    fi

    # Actual launch under `timeout` so we have a wall-clock ceiling.
    # The orchestrator already has its own NO_PROGRESS_TIMEOUT_SEC=1800
    # internally; this is a backstop covering the case where the
    # orchestrator itself hangs.
    set +e
    timeout --signal=TERM --kill-after=120 "$timeout_sec" \
        python "$script" --jobs "$sub_tag" \
        > "$job_log" 2>&1
    local rc=$?
    set -e

    local job_end
    job_end=$(date +%s)
    local elapsed=$((job_end - job_start))
    local elapsed_hms
    elapsed_hms=$(hms $elapsed)

    # Interpret the exit code
    if [[ $rc -eq 0 ]]; then
        ok "Completed in ${elapsed_hms} (rc=0)"
        # Confirm artifacts actually landed
        if job_already_succeeded "$sub_tag" "$paper"; then
            ok "Artifacts confirmed (proton_energy_histogram.csv + reports present)"
            RESULTS+=("${sub_tag}: SUCCESS (${elapsed_hms})")
            return 0
        else
            warn "rc=0 but expected artifacts not found locally — check log"
            RESULTS+=("${sub_tag}: COMPLETED_NO_ARTIFACTS (${elapsed_hms})")
            return 1
        fi
    elif [[ $rc -eq 124 ]]; then
        err "TIMEOUT after ${elapsed_hms} (rc=124 from timeout(1))"
        err "  The orchestrator was killed by campaign-level ceiling."
        err "  Check ${job_log} for what stage it was in."
        RESULTS+=("${sub_tag}: TIMEOUT (${elapsed_hms})")
        return 1
    elif [[ $rc -eq 137 ]]; then
        err "TIMEOUT-KILL after ${elapsed_hms} (rc=137; SIGTERM didn't work, used SIGKILL)"
        RESULTS+=("${sub_tag}: KILLED (${elapsed_hms})")
        return 1
    elif [[ $rc -eq 130 ]]; then
        err "INTERRUPTED by user (Ctrl+C) after ${elapsed_hms}"
        RESULTS+=("${sub_tag}: INTERRUPTED (${elapsed_hms})")
        exit 130
    else
        err "FAILED after ${elapsed_hms} (rc=$rc)"
        err "  Last 30 lines of log:"
        tail -30 "$job_log" | sed 's/^/    /'
        RESULTS+=("${sub_tag}: FAILED rc=$rc (${elapsed_hms})")
        return 1
    fi
}

# ─── Main ───────────────────────────────────────────────────────────────────
RESULTS=()
CAMPAIGN_START=$(date +%s)

mkdir -p "$LOG_DIR_BASE"
LOG_DIR="${LOG_DIR_BASE}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

banner "Headline campaign starting at $(ts)"
echo "Logs dir:    $LOG_DIR"
echo "Status file: $STATUS_FILE"
echo "PID:         $$"

update_status "headline_campaign" "PREFLIGHT" "starting"

if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
    preflight
else
    warn "Preflight checks SKIPPED via --skip-preflight"
fi

# ─── The four headline jobs (defined here, in dispatch order) ───────────────
# Format: (idx, sub_tag, paper_dir, grid, timeout_sec)
# We don't use bash arrays-of-arrays (clunky); call run_job inline.

CAMPAIGN_OK=1

run_job 1 4 "p1_ld_uuf"                       "paper01" "256" "$TIMEOUT_256_SEC" || CAMPAIGN_OK=0
run_job 2 4 "p3_rod_plus_outer_p7li_uuf"      "paper3"  "256" "$TIMEOUT_256_SEC" || CAMPAIGN_OK=0
run_job 3 4 "p1_ld_uuf_512"                   "paper01" "512" "$TIMEOUT_512_SEC" || CAMPAIGN_OK=0
run_job 4 4 "p3_rod_plus_outer_p7li_512"      "paper3"  "512" "$TIMEOUT_512_SEC" || CAMPAIGN_OK=0

# ─── Summary ────────────────────────────────────────────────────────────────
CAMPAIGN_END=$(date +%s)
TOTAL_ELAPSED=$((CAMPAIGN_END - CAMPAIGN_START))

banner "Campaign complete at $(ts)"
echo "Total wall clock: $(hms $TOTAL_ELAPSED)"
echo ""
echo "Per-job results:"
for entry in "${RESULTS[@]}"; do
    if [[ "$entry" == *"SUCCESS"* ]] || [[ "$entry" == *"SKIPPED"* ]]; then
        echo -e "  ${GREEN}✓${RESET} $entry"
    elif [[ "$entry" == *"DRY_RUN"* ]]; then
        echo -e "  ${BLUE}·${RESET} $entry"
    else
        echo -e "  ${RED}✗${RESET} $entry"
    fi
done

echo ""
if [[ $CAMPAIGN_OK -eq 1 ]]; then
    ok "ALL HEADLINE JOBS SUCCEEDED"
    update_status "headline_campaign" "COMPLETE" "success"
else
    warn "Some jobs failed — review per-job logs in $LOG_DIR"
    update_status "headline_campaign" "COMPLETE" "partial_failure"
fi

echo ""
echo "Headline composites should be available locally at:"
echo "  runs/paper01/p1_ld_uuf/figures/composite_headline.png"
echo "  runs/paper01/p1_ld_uuf_512/figures/composite_headline.png"
echo "  runs/paper3/p3_rod_plus_outer_p7li_uuf/figures/composite_headline.png"
echo "  runs/paper3/p3_rod_plus_outer_p7li_512/figures/composite_headline.png"
echo ""
echo "Histogram CSVs (for ad-hoc gain analysis):"
echo "  runs/paper01/p1_ld_uuf/proton_energy_histogram.csv"
echo "  runs/paper01/p1_ld_uuf_512/proton_energy_histogram.csv"
echo "  runs/paper3/p3_rod_plus_outer_p7li_uuf/proton_energy_histogram.csv"
echo "  runs/paper3/p3_rod_plus_outer_p7li_512/proton_energy_histogram.csv"
echo ""

if [[ $CAMPAIGN_OK -eq 1 ]]; then
    exit 0
else
    exit 2
fi
