#!/bin/bash
# run_paper1_chain.sh - Paper 1 analysis chain (ring-only reconnection)
#
# Forked from run_paper3_chain.sh on 2026-05-13. Differences from Paper 3:
#
#   * Stage C dropped entirely (Paper 1 is p-11B only; no p-7Li)
#   * Stage B uses --mode global (Paper 1 first-transit methodology)
#     - terminates per-particle accumulation at min(wrap, B-collapse>50%)
#     - reads t_cut from reconnection_rate_offline.csv automatically
#     - produces *_first_transit_summary_global_p11b.txt with G_FT headline
#   * Stage D uses --paper-tag paper1
#     - produces *_zones_paper1.txt and *_zone_fusion_rates_paper1.csv
#   * Log marker: "ALL P1 ANALYSIS STAGES COMPLETE" (Paper 1 orchestrator
#     watches for this string instead of the P3 equivalent)
#
# This script is intentionally a sibling of run_paper3_chain.sh rather than
# an extension of it. Two reasons:
#   (1) Paper 3's chain is a frozen reference for the Paper 3 manuscript;
#       any change there risks invalidating already-collected data.
#   (2) Paper 1 and Paper 3 are diverging in their post-analysis needs
#       (Paper 1 wants global-mode first-transit, Paper 3 wants zone-mode
#       + p-7Li). A shared script with mode flags would grow opaque fast.
#
# Stages:
#   A. Standard analysis (run_full_analysis.sh) — produces reconnection
#      summary, fusion diagnostics, phase analysis, post-analysis report.
#      Identical to Paper 3 Stage A.
#   B. First-transit fusion (GLOBAL MODE) for p-11B reaction.
#      Reads reconnection_rate_offline.csv to derive t_cut, then accumulates
#      per-particle σ·v·n·dt over the reconnection-event window. Output:
#      *_first_transit_summary_global_p11b.txt with G_FT, G_pretcut, G_legacy.
#   D. Six-zone radial breakdown with --paper-tag paper1. Output:
#      *_zones_paper1.txt and *_zone_fusion_rates_paper1.csv.
#
# Usage:
#   run_paper1_chain.sh <run_dir>
#
# Args:
#   run_dir - relative path to run directory (e.g. runs/p1_ld_uuf)
#
# Called by stage_a_paper01.py via:
#   setsid nohup bash analysis_scripts/run_paper1_chain.sh <run_dir> \
#       < /dev/null > <sub_tag>_analysis.log 2>&1 & disown
#
# 512² handling: Stage B is skipped automatically for nx=512 runs because
# the global-mode pass shares the same memory/I/O profile as Paper 3 zone-
# mode first-transit. Convergence is demonstrated by Stage D's E95 profiles.
# (Paper 1 v0.7 design: 512² is a convergence check, not a headline-data
# run; the headline G_FT is from p1_ld_uuf at 256².)
#
# This script intentionally does NOT use 'set -e' so that one stage failing
# doesn't prevent the others from running. Per-stage rc is tracked and
# summarised at the end.

set +e

# Activate the plasma conda environment so this script works regardless of
# caller environment. The orchestrator already does this in its SSH wrapper,
# but doing it here too is harmless and lets manual invocations work without
# further setup.
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate plasma
fi

RUN_DIR="$1"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: run_dir argument missing"
    echo "Usage: run_paper1_chain.sh <run_dir>"
    exit 2
fi

# Detect 512² convergence run from run_meta.txt. We skip first-transit
# (Stage B) for these — the global-mode pass has the same memory/I/O cost
# as Paper 3 zone-mode at 512², and the convergence question is answered
# by zone analysis (Stage D).
IS_512=false
META_FILE="$RUN_DIR/run_meta.txt"
if [ -f "$META_FILE" ]; then
    NX_VALUE=$(grep -E "^nx=" "$META_FILE" | head -1 | cut -d= -f2 | tr -d ' ')
    if [ "$NX_VALUE" = "512" ]; then
        IS_512=true
    fi
fi

START_TS=$(date +%s)
echo "===== PAPER 1 ANALYSIS CHAIN START ====="
echo "  run_dir  = $RUN_DIR"
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
# preflight). Treat rc=2 as a warning, not a hard failure.
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

# ----- STAGE B: first-transit p11b (GLOBAL MODE) --------------------------
if [ "$IS_512" = "true" ]; then
    echo "===== STAGE B: first-transit p11b (global mode) SKIPPED (nx=512) ====="
    echo "  Reason: at 512² each particle dump is large; global-mode first-"
    echo "  transit hits the same memory/I/O ceiling as Paper 3 zone-mode."
    echo "  512² convergence is demonstrated by Stage D zone analysis."
    echo "  Headline G_FT is reported from p1_ld_uuf (256²)."
    STAGE_B_RC=0
    STAGE_B_NOTE=" (skipped for 512² convergence run)"
else
    echo "===== STAGE B: first-transit p11b (global mode) ====="
    python -u analysis_scripts/pb11_first_transit_fusion.py \
        --run-dir "$RUN_DIR" --reaction p11b --mode global
    STAGE_B_RC=$?
    STAGE_B_NOTE=""
fi
echo "  Stage B finished with rc=$STAGE_B_RC$STAGE_B_NOTE"
echo ""

# ----- STAGE D: six-zone analysis with paper1 tag --------------------------
# (No Stage C in Paper 1 — there is no p-7Li.)
echo "===== STAGE D: six-zone analysis (paper-tag=paper1) ====="
python -u analysis_scripts/pb11_zone_analysis_paper3.py \
    --run-dir "$RUN_DIR" --paper-tag paper1
STAGE_D_RC=$?
echo "  Stage D finished with rc=$STAGE_D_RC"
echo ""

# ----- STAGE M: medical-track data products --------------------------------
# Option C: pointer-only mode. We run ONLY the robust σ-weighted summary
# (medical_track_pointer.py), which depends solely on the canonical analysis
# outputs (first_transit_summary, fusion_rate_power_by_iter.csv,
# reconnection_summary.txt). This gives a publication-defensible medical-
# track deliverable per run without relying on the proton-spectrum-derived
# yields that are currently affected by hybrid-PIC numerical noise.
#
# Sibling scripts medical_proton_yield.py and medical_alpha_kinematic.py are
# deployed but NOT called from the chain. They emit DRAFT-tagged outputs and
# can be invoked manually for diagnostic comparisons when the underlying
# simulation issues are resolved.
#
# Stage M is non-blocking — Paper 1 results are not gated on its success.
echo "===== STAGE M: medical-track pointer (robust σ-weighted only) ====="
python -u analysis_scripts/medical_track_pointer.py --run-dir "$RUN_DIR"
STAGE_M_RC=$?
if [ "$STAGE_M_RC" -ne 0 ]; then
    STAGE_M_NOTE=" (rc=$STAGE_M_RC — non-blocking)"
else
    STAGE_M_NOTE=""
fi
echo "  Stage M finished with rc=$STAGE_M_RC$STAGE_M_NOTE"
echo ""

# ----- Summary -------------------------------------------------------------
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "===== ALL P1 ANALYSIS STAGES COMPLETE ====="
echo "  Stage A standard pipeline             rc=$STAGE_A_RC$STAGE_A_NOTE"
echo "  Stage B first-transit p11b (global)   rc=$STAGE_B_RC$STAGE_B_NOTE"
echo "  Stage D zone analysis (paper1 tag)    rc=$STAGE_D_RC"
echo "  Stage M medical-track products        rc=$STAGE_M_RC$STAGE_M_NOTE"
echo "  Total elapsed: ${ELAPSED}s"

# Exit non-zero only if any stage failed (Stage M is non-blocking).
if [ "$STAGE_A_RC" -ne 0 ] || [ "$STAGE_B_RC" -ne 0 ] || [ "$STAGE_D_RC" -ne 0 ]; then
    exit 1
fi
exit 0
