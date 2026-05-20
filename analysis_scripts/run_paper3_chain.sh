#!/bin/bash
# run_paper3_chain.sh - Paper 3 analysis chain (v2: skip first-transit for 512² runs)
#
# Runs the four-stage Paper 3 analysis pipeline:
#   A. Standard analysis (run_full_analysis.sh)
#   B. First-transit fusion accounting for p-11B reaction
#   C. First-transit fusion accounting for p-7Li reaction (optional)
#   D. Paper 3 six-zone breakdown
#
# Usage:
#   run_paper3_chain.sh <run_dir> <has_p7li>
#
# Args:
#   run_dir   - relative path to run directory (e.g. runs/p3_preflight)
#   has_p7li  - "yes" or "no" (controls whether Stage C runs)
#
# Called by stage_a_paper3.py via:
#   setsid nohup bash analysis_scripts/run_paper3_chain.sh <run_dir> <has_p7li> \
#       < /dev/null > <sub_tag>_analysis.log 2>&1 & disown
#
# NEW in v2: Stages B and C are SKIPPED automatically for 512² runs (nx=512
# in run_meta.txt).  Reason: at 512² each particle dump is ~8.8 GB and the
# vectorized first-transit script hits memory/I/O thrashing — runs took
# 60+ minutes of CPU and 7+ GB RAM before hanging.  For Paper 3, the 512²
# convergence demonstration only needs the zone-based analysis (Stage D)
# to validate that E95 profiles and core/spot ratios are grid-converged.
# First-transit accounting is reported from 256² Tier 1 / Tier 2 production
# runs where the script handles the smaller dumps cleanly.
#
# This script intentionally does NOT use 'set -e' so that one stage failing
# doesn't prevent the others from running. We track per-stage rc and report
# them at the end.

set +e

# Activate the plasma conda environment ourselves so the script works
# regardless of caller environment (orchestrator, manual SSH, etc.). The
# orchestrator already does this in its SSH wrapper, but doing it here too
# is harmless and means manual invocations work without further setup.
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate plasma
fi

RUN_DIR="$1"
HAS_P7LI="$2"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: run_dir argument missing"
    echo "Usage: run_paper3_chain.sh <run_dir> <has_p7li>"
    exit 2
fi

if [ -z "$HAS_P7LI" ]; then
    HAS_P7LI="no"
fi

# Detect 512² convergence run from run_meta.txt.  We skip first-transit
# (Stages B and C) for these because the script's memory/I/O cost is
# prohibitive on the ~8.8 GB per dump file size at 512².  The convergence
# question is answered by zone analysis (Stage D), and first-transit
# accounting is reported from 256² runs.
IS_512=false
META_FILE="$RUN_DIR/run_meta.txt"
if [ -f "$META_FILE" ]; then
    NX_VALUE=$(grep -E "^nx=" "$META_FILE" | head -1 | cut -d= -f2 | tr -d ' ')
    if [ "$NX_VALUE" = "512" ]; then
        IS_512=true
    fi
fi

START_TS=$(date +%s)
echo "===== PAPER 3 ANALYSIS CHAIN START ====="
echo "  run_dir  = $RUN_DIR"
echo "  has_p7li = $HAS_P7LI"
echo "  nx       = ${NX_VALUE:-unknown}"
echo "  is_512   = $IS_512"
echo "  started  = $(date -Iseconds)"
echo ""

# ----- STAGE A: standard analysis pipeline ---------------------------------
echo "===== STAGE A: standard analysis pipeline ====="
bash analysis_scripts/run_full_analysis.sh "$RUN_DIR"
STAGE_A_RAW_RC=$?
# run_full_analysis.sh returns rc=2 specifically when its Stage 9 sanity
# check finds soft issues (e.g. "fusion CSV has no data rows" on a too-short
# preflight). The pipeline itself ran to completion in that case. Treat
# rc=2 as a warning, not a hard failure, so the orchestrator does not retry
# a deterministic warning. Hard failures (rc=1, rc=127, etc.) propagate.
if [ "$STAGE_A_RAW_RC" = "2" ]; then
    echo "  NOTE: Stage A raw rc=2 indicates a Stage 9 sanity warning,"
    echo "        not a hard pipeline failure. Treating as soft-OK."
    STAGE_A_RC=0
    STAGE_A_NOTE=" (raw rc=2 Stage 9 warning treated as OK)"
else
    STAGE_A_RC=$STAGE_A_RAW_RC
    STAGE_A_NOTE=""
fi
echo "  Stage A finished with rc=$STAGE_A_RC$STAGE_A_NOTE"
echo ""

# ----- STAGE B: first-transit p11b -----------------------------------------
if [ "$IS_512" = "true" ]; then
    echo "===== STAGE B: first-transit p11b SKIPPED (nx=512 convergence run) ====="
    echo "  Reason: at 512² each particle dump is ~8.8 GB; first-transit script"
    echo "  hits memory/I/O ceiling. 512² convergence is demonstrated by Stage D"
    echo "  zone analysis. First-transit reported from 256² Tier 1 / Tier 2 runs."
    STAGE_B_RC=0
    STAGE_B_NOTE=" (skipped for 512² convergence run)"
else
    echo "===== STAGE B: first-transit p11b accounting ====="
    python -u analysis_scripts/pb11_first_transit_fusion.py \
        --run-dir "$RUN_DIR" --reaction p11b
    STAGE_B_RC=$?
    STAGE_B_NOTE=""
fi
echo "  Stage B finished with rc=$STAGE_B_RC$STAGE_B_NOTE"
echo ""

# ----- STAGE C: first-transit p7li (skip when no p-7Li fuel anywhere) ------
if [ "$IS_512" = "true" ]; then
    echo "===== STAGE C: first-transit p7li SKIPPED (nx=512 convergence run) ====="
    STAGE_C_RC=0
    STAGE_C_NOTE=" (skipped for 512² convergence run)"
elif [ "$HAS_P7LI" = "yes" ]; then
    echo "===== STAGE C: first-transit p7li accounting ====="
    python -u analysis_scripts/pb11_first_transit_fusion.py \
        --run-dir "$RUN_DIR" --reaction p7li
    STAGE_C_RC=$?
    STAGE_C_NOTE=""
else
    echo "===== STAGE C: first-transit p7li SKIPPED (has_p7li=no) ====="
    STAGE_C_RC=0
    STAGE_C_NOTE=" (has_p7li=no)"
fi
echo "  Stage C finished with rc=$STAGE_C_RC$STAGE_C_NOTE"
echo ""

# ----- STAGE D: Paper 3 six-zone analysis ---------------------------------
echo "===== STAGE D: Paper 3 six-zone analysis ====="
python -u analysis_scripts/pb11_zone_analysis_paper3.py --run-dir "$RUN_DIR"
STAGE_D_RC=$?
echo "  Stage D finished with rc=$STAGE_D_RC"
echo ""

# ----- Summary -------------------------------------------------------------
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "===== ALL P3 ANALYSIS STAGES COMPLETE ====="
echo "  Stage A standard pipeline      rc=$STAGE_A_RC$STAGE_A_NOTE"
echo "  Stage B first-transit p11b     rc=$STAGE_B_RC$STAGE_B_NOTE"
echo "  Stage C first-transit p7li     rc=$STAGE_C_RC$STAGE_C_NOTE"
echo "  Stage D zone analysis          rc=$STAGE_D_RC"
echo "  Total elapsed: ${ELAPSED}s"

# Exit non-zero only if any stage failed.
if [ "$STAGE_A_RC" -ne 0 ] || [ "$STAGE_B_RC" -ne 0 ] \
   || [ "$STAGE_C_RC" -ne 0 ] || [ "$STAGE_D_RC" -ne 0 ]; then
    exit 1
fi
exit 0
