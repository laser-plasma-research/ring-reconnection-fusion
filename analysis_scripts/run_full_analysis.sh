#!/bin/bash
# run_full_analysis.sh — 8-stage analysis pipeline for ring-reconnection runs
# Usage: bash analysis_scripts/run_full_analysis.sh runs/<run_name>
#
# Runs all 8 stages sequentially. Stops on first failure.
# Each stage logs its progress with timestamps.
#
# Stage 1: visualize_all           — figures, B-field evolution, writes cache npz files
# Stage 2: pb11_consolidate_reports — post_analysis_report.txt + zone_report.txt (from cache)
# Stage 3: pb11_phase_analysis     — phase_analysis_report.txt
# Stage 4: pb11_reconnection_rate_offline — reconnection rates + summary
# Stage 5: pb11_fusion_diagnostics — fusion yields
# Stage 6: pb11_particle_animations — animations
# Stage 7: pb11_particle_trajectories — particle paths
# Stage 8: Final inventory

set -u

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
    echo "Usage: $0 <run_dir>"
    echo "Example: $0 runs/p1_ld_256_200k_long"
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: directory '$RUN_DIR' does not exist"
    exit 1
fi

if [ ! -d "$RUN_DIR/particles" ]; then
    echo "ERROR: '$RUN_DIR/particles' not found - is this a valid run dir?"
    exit 1
fi

cd ~/laser-plasma-research

ANALYSIS_DIR="analysis_scripts"
STAGES_TOTAL=8
STAGE=0
START_TIME=$(date +%s)

stage_start() {
    STAGE=$((STAGE + 1))
    STAGE_START=$(date +%s)
    echo
    echo "============================================================"
    echo "  STAGE $STAGE/$STAGES_TOTAL: $1"
    echo "  Time: $(date)"
    echo "============================================================"
}

stage_end() {
    STAGE_DUR=$(( $(date +%s) - STAGE_START ))
    STAGE_MIN=$((STAGE_DUR / 60))
    STAGE_SEC=$((STAGE_DUR % 60))
    echo
    echo "  STAGE $STAGE COMPLETE in ${STAGE_MIN}m ${STAGE_SEC}s"
}

# ============================================================
# STAGE 1 — visualize_all (basic figures, B-field evolution)
#   Also writes cache/zone_timeseries.npz and cache/field_timeseries.npz
#   which Stage 2 depends on.
# ============================================================
stage_start "visualize_all (figures, B-field evolution)"
python -u $ANALYSIS_DIR/visualize_all.py --dir "$RUN_DIR" --animate || { echo "STAGE 1 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 2 — pb11_consolidate_reports (post_analysis + zone reports)
#   Reads cached npz files from Stage 1. Fast (~1 second).
#   Generates: post_analysis_report.txt, zone_report.txt
# ============================================================
stage_start "pb11_consolidate_reports (post_analysis_report.txt + zone_report.txt)"
python -u $ANALYSIS_DIR/pb11_consolidate_reports.py --dir "$RUN_DIR" || { echo "STAGE 2 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 3 — pb11_phase_analysis (phase-resolved analysis)
#   Generates: phase_analysis_report.txt
# ============================================================
stage_start "pb11_phase_analysis (phase-resolved analysis)"
python -u $ANALYSIS_DIR/pb11_phase_analysis.py --dir "$RUN_DIR" || { echo "STAGE 3 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 4 — reconnection_rate_offline (X-line tracking + rate analysis)
# ============================================================
stage_start "pb11_reconnection_rate_offline (X-line tracking + reconnection rates)"
python -u $ANALYSIS_DIR/pb11_reconnection_rate_offline.py --dir "$RUN_DIR" || { echo "STAGE 4 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 5 — fusion_diagnostics (fusion yield analysis)
# ============================================================
stage_start "pb11_fusion_diagnostics (fusion yields, alpha cascading)"
python -u $ANALYSIS_DIR/pb11_fusion_diagnostics.py --dir "$RUN_DIR" || { echo "STAGE 5 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 6 — particle_animations (particle density + spectrum animations)
# ============================================================
stage_start "pb11_particle_animations (density + spectrum animations)"
python -u $ANALYSIS_DIR/pb11_particle_animations.py --dir "$RUN_DIR" || { echo "STAGE 6 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 7 — particle_trajectories (individual particle tracking)
# ============================================================
stage_start "pb11_particle_trajectories (individual particle paths)"
python -u $ANALYSIS_DIR/pb11_particle_trajectories.py --dir "$RUN_DIR" || { echo "STAGE 7 FAILED"; exit 1; }
stage_end

# ============================================================
# STAGE 8 — final inventory (list all outputs and confirm)
# ============================================================
stage_start "Final inventory (list outputs)"
echo "  Output files in $RUN_DIR/:"
ls -la "$RUN_DIR" | grep -E "\.txt$|\.csv$" | awk '{printf "    %s\n", $9}'
echo
echo "  Output files in $RUN_DIR/figures/:"
ls "$RUN_DIR/figures" 2>/dev/null | head -20 | awk '{printf "    %s\n", $0}'
echo
echo "  Total size:"
du -sh "$RUN_DIR/figures" "$RUN_DIR"/*.txt "$RUN_DIR"/*.csv 2>/dev/null | tail -5
stage_end

# ============================================================
# Pipeline complete
# ============================================================
TOTAL_DUR=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$((TOTAL_DUR / 60))
TOTAL_SEC=$((TOTAL_DUR % 60))

echo
echo "============================================================"
echo "  ALL 8 STAGES COMPLETE for $RUN_DIR"
echo "  Total runtime: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Time: $(date)"
echo "============================================================"
