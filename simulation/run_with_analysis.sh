#!/bin/bash
# run_with_analysis.sh — Run a simulation and automatically post-analyze when done.
#
# Updated v2 (2026-04-29):
#   All output (run.log, post-analysis report) now goes inside runs/<name>/
#   instead of cluttering the project root.
#
# Usage:
#   ./run_with_analysis.sh <run_name> [extra simulation args...]
#
# Example:
#   ./run_with_analysis.sh v1213_1_chbn_b300_1000 \
#       --test --diag-profile test --base-fuel ch_bn --base-density 5e25 \
#       --b-seed 300 --max-steps 1000
#
# This will:
#   1. Create runs/<run_name>/ directory
#   2. Run the simulation, output to that directory
#   3. Save log to runs/<run_name>/run.log
#   4. Run post-analysis, save report to runs/<run_name>/post_analysis_report.txt
#   5. Print the report tail at the end so you see the key results

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name> [simulation args...]"
    echo ""
    echo "Example:"
    echo "  $0 v1213_1_chbn_b300_1000 --test --diag-profile test --base-fuel ch_bn \\"
    echo "      --base-density 5e25 --b-seed 300 --max-steps 1000"
    exit 1
fi

RUN_NAME="$1"
shift  # remove run_name from args; rest are simulation args

OUTDIR="runs/${RUN_NAME}"
LOGFILE="${OUTDIR}/run.log"
REPORTFILE="${OUTDIR}/post_analysis_report.txt"

SIM_SCRIPT="simulation/pb11_ring_reconnection_v12_fuel_center_outer.py"
ANALYSIS_SCRIPT="simulation/pb11_post_analysis.py"

# Make sure the run dir exists before tee tries to write into it
mkdir -p "${OUTDIR}"

# Always run analysis on exit, even if simulation crashed
cleanup() {
    echo ""
    echo "============================================"
    echo "  Running post-analysis on ${OUTDIR}"
    echo "============================================"
    if [ -f "${OUTDIR}/run_meta.txt" ]; then
        python "${ANALYSIS_SCRIPT}" --dir "${OUTDIR}" --report-file "${REPORTFILE}"
        echo ""
        echo "============================================"
        echo "  Report saved to ${REPORTFILE}"
        echo "  Full log saved to ${LOGFILE}"
        echo "============================================"
    else
        echo "  Run output not found — simulation may have failed before writing meta."
        echo "  Check ${LOGFILE} for errors."
    fi
}
trap cleanup EXIT

export OMP_NUM_THREADS=1

echo "============================================"
echo "  Starting simulation: ${RUN_NAME}"
echo "  Output dir: ${OUTDIR}"
echo "  Log file:   ${LOGFILE}"
echo "  Args:       $@"
echo "  Started:    $(date)"
echo "============================================"

mpirun -n 8 -x OMP_NUM_THREADS=1 python "${SIM_SCRIPT}" \
    "$@" \
    --outdir "${OUTDIR}" \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "============================================"
echo "  Simulation finished: $(date)"
echo "============================================"
