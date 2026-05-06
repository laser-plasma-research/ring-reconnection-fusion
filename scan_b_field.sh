#!/bin/bash
# B-field validity scan
export OMP_NUM_THREADS=1

for B in 30 50 75 100 150 200 300; do
    OUTDIR="runs/scan_chbn_b${B}"
    LOGFILE="scan_chbn_b${B}.log"

    echo "============================================"
    echo "  B=${B}T preflight starting at $(date)"
    echo "============================================"

    mpirun -n 8 -x OMP_NUM_THREADS=1 python simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \
        --test --diag-profile test \
        --base-fuel ch_bn --base-density 5e25 \
        --b-seed ${B} \
        --max-steps 50 \
        --outdir ${OUTDIR} \
        2>&1 | tee ${LOGFILE} || echo "  >>> B=${B}T CRASHED <<<"
done

echo ""
echo "============================================"
echo "  Scan summary"
echo "============================================"
for B in 30 50 75 100 150 200 300; do
    LOG="scan_chbn_b${B}.log"
    CSV="runs/scan_chbn_b${B}/fusion_rate_power_by_iter.csv"

    if grep -q "SIMULATION COMPLETE" "${LOG}" 2>/dev/null; then
        STATUS="OK     "
        FF50=$(awk -F, 'NR>1 && $2==50 {print $9}' "${CSV}" 2>/dev/null | head -1)
        FF50=${FF50:-N/A}
    else
        STATUS="CRASHED"
        FF50="N/A"
    fi
    AVG=$(grep "Avg. per step" "${LOG}" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    printf "  B=%4dT: %s  step_avg=%-6ss  fast_frac@50=%s\n" "${B}" "${STATUS}" "${AVG:-N/A}" "${FF50}"
done
