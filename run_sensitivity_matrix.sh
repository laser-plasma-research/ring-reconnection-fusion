#!/usr/bin/env bash
# run_sensitivity_matrix.sh — sequential runs of the 12-spec sensitivity matrix
# minus the 3 already done (s01_baseline, s_drop4_hemisphere with v2 patches done).
# Re-runs s_drop1_4plus3 to apply the density patch too.
#
# For each run:
#   1. Clear any prior run directory
#   2. Sim (~5-10 min wall clock at 256² with --test --diag-profile production)
#   3. Full analysis chain (~42 min)
#   4. Stage out reports to /mnt/vdc/test_runs/_results/<run>/
#   5. Delete particle+field dumps (keep reports)
#
# Total compute: ~10 runs × ~50 min = ~8-10 hours wall clock
# Peak disk: ~150 GB per active run (sequential, so only one at a time)
#
# Usage (on cloud):
#   bash run_sensitivity_matrix.sh > sensitivity_matrix.log 2>&1 &
#   tail -f sensitivity_matrix.log

set -u  # error on unset vars; intentionally NOT -e because we want to
        # keep going if one run fails

RUNS=(
  "s_drop1_4plus3"           # re-run with B+density patches
  "s00_baseline_specform"
  "s_energy_05pct"
  "s_energy_10pct"
  "s_energy_20pct"
  "s_jitter_050um"
  "s_jitter_150um"
  "s_jitter_300um"
  "s_drop2_adjacent"
  "s_drop2_opposite"
  "s_combined_realistic"
)

RESULTS_DIR="/mnt/vdc/test_runs/_results"
mkdir -p "$RESULTS_DIR"

# Common flags — matching baseline_v2 and hemisphere_v2 settings
COMMON_FLAGS=(
  --test
  --base-fuel ch_bn
  --base-density 5e24
  --ring-radius-um 2400
  --spot-radius-um 300
  --n-spots 8
  --b-seed 85
  --diag-profile production
  --max-steps 1500
  --dump-period 20
)

DOCKER_BASE=(docker run --rm --gpus all
  -v /mnt/vdc/test_runs:/work/runs
  -v /mnt/vdc/laser-plasma-research/simulation:/work/simulation
  -v /mnt/vdc/laser-plasma-research/specs:/work/specs
  -v /mnt/vdc/laser-plasma-research/analysis_scripts:/work/analysis_scripts
  ring-reconnection-fusion:v1.2.0)

DOCKER_NOGPU=(docker run --rm
  -v /mnt/vdc/test_runs:/work/runs
  -v /mnt/vdc/laser-plasma-research/simulation:/work/simulation
  -v /mnt/vdc/laser-plasma-research/specs:/work/specs
  -v /mnt/vdc/laser-plasma-research/analysis_scripts:/work/analysis_scripts
  ring-reconnection-fusion:v1.2.0)

echo "============================================================"
echo "Sensitivity matrix sweep started: $(date)"
echo "Runs: ${#RUNS[@]}"
echo "Results stage: $RESULTS_DIR"
echo "============================================================"

for run in "${RUNS[@]}"; do
  echo ""
  echo "============================================================"
  echo "[$(date)] STARTING $run"
  echo "============================================================"

  # Resume-skip: if this run already finished (staged report exists), skip it.
  # Lets you re-launch the script after interruption without redoing work.
  staged_ft="$RESULTS_DIR/$run/${run}_first_transit_summary_global_p11b.txt"
  if [[ -f "$staged_ft" ]]; then
    gft=$(grep "G_FT (first-transit" "$staged_ft" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ')
    echo "  [skip] $run already staged with G_FT = $gft"
    continue
  fi

  spec_file="specs/${run}.json"

  # 1. Clear any prior state (root-owned -> use a container to rm)
  echo "  [clean] removing /work/runs/$run if it exists..."
  "${DOCKER_NOGPU[@]}" rm -rf "/work/runs/$run"

  # 2. Run sim
  echo "  [sim] launching..."
  "${DOCKER_BASE[@]}" bash -c "
    cd /work && \
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm && \
    mpirun --allow-run-as-root -n 1 python simulation/pb11_ring_reconnection_v15_pulsed.py \
      ${COMMON_FLAGS[*]} \
      --spot-spec $spec_file \
      --outdir /work/runs/$run \
      > /work/runs/${run}_sim.log 2>&1
  "

  sim_rc=$?
  if [[ $sim_rc -ne 0 ]]; then
    echo "  [FAIL] sim returned $sim_rc — skipping analysis, moving to next run"
    continue
  fi

  # Confirm 76 dumps landed
  dumps=$(ls /mnt/vdc/test_runs/$run/particles/*.h5 2>/dev/null | wc -l)
  echo "  [sim] complete, $dumps/76 dumps"
  if [[ $dumps -lt 70 ]]; then
    echo "  [FAIL] only $dumps dumps, expected ~76 — skipping analysis"
    continue
  fi

  # 3. Run analysis chain
  echo "  [chain] launching..."
  "${DOCKER_NOGPU[@]}" bash -c "
    cd /work && \
    bash analysis_scripts/run_paper1_chain.sh runs/$run \
      > /work/runs/${run}_chain.log 2>&1
  "
  chain_rc=$?
  if [[ $chain_rc -ne 0 ]]; then
    echo "  [WARN] chain returned $chain_rc — staging what we have"
  fi

  # 4. Stage out reports + logs (everything except particles/fields)
  echo "  [stage] copying reports to $RESULTS_DIR/$run/..."
  mkdir -p "$RESULTS_DIR/$run"
  rsync -a \
    --exclude='particles/' --exclude='fields/' \
    --exclude='*.bp' --exclude='*.h5' \
    "/mnt/vdc/test_runs/$run/" "$RESULTS_DIR/$run/"
  cp "/mnt/vdc/test_runs/${run}_sim.log" "$RESULTS_DIR/$run/" 2>/dev/null
  cp "/mnt/vdc/test_runs/${run}_chain.log" "$RESULTS_DIR/$run/" 2>/dev/null

  # 5. Delete heavy dumps to free disk
  echo "  [clean] deleting particle+field dumps..."
  "${DOCKER_NOGPU[@]}" rm -rf "/work/runs/$run/particles" "/work/runs/$run/fields"

  # Quick G_FT readout to log
  ft_file="/mnt/vdc/test_runs/$run/${run}_first_transit_summary_global_p11b.txt"
  if [[ -f "$ft_file" ]]; then
    gft=$(grep "G_FT (first-transit" "$ft_file" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ')
    echo "  [result] $run: G_FT = $gft"
  fi

  df_free=$(df -h /mnt/vdc | tail -1 | awk '{print $4}')
  echo "  [disk] free: $df_free"
done

echo ""
echo "============================================================"
echo "Sensitivity matrix sweep complete: $(date)"
echo "Results staged at: $RESULTS_DIR"
echo ""
echo "Summary of G_FT across runs:"
for run in "${RUNS[@]}"; do
  ft_file="$RESULTS_DIR/$run/${run}_first_transit_summary_global_p11b.txt"
  if [[ -f "$ft_file" ]]; then
    gft=$(grep "G_FT (first-transit" "$ft_file" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ')
    echo "  $run: G_FT = $gft"
  else
    echo "  $run: NO RESULT"
  fi
done
echo "============================================================"
