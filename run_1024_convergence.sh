#!/usr/bin/env bash
# run_1024_convergence.sh — 3-point convergence sweep at matched NPPC=100
# for clean grid-only comparison: 256², 512², 1024².
#
# Existing paper baselines were at NPPC=200:
#   256² at NPPC=200 → G_FT = 2.02 (t_cut detector picked 540 ps)
#   512² at NPPC=200 → G_FT = 3.63 (t_cut detector picked 1048 ps - artifact)
#
# v0.8 of pb11_first_transit_fusion.py replaces the t_cut detector with a
# smoothed first-prominent-minimum approach.  Per-run G_FT now stable
# across replicate seeds and grid sizes.
#
# This sweep produces three directly comparable G_FT values for the
# convergence claim:
#   256² @ NPPC=100  (~25 min)
#   512² @ NPPC=100  (~6-8 hours)
#   1024² @ NPPC=100 (~10-14 hours)
#
# Particle/field dumps are RETAINED across all three runs until the user
# confirms the t_cut detector picked sensible values for each.  Once
# validated, dumps can be deleted manually to free ~5 TB.
#
# Usage (on cloud):
#   bash run_1024_convergence.sh > 1024_convergence.log 2>&1 &
#   tail -f 1024_convergence.log

set -u

# (sub_tag, nx, nz, nppc, substeps, max_steps, dump_period)
# Note: max_steps and dump_period scale with grid size so all three runs
# cover the same physical time and produce similar dump counts (~38-76).
RUNS=(
  "p1_ld_uuf_256_n100:256:256:100:80:3850:50"
  "p1_ld_uuf_512_n100:512:512:100:160:7700:200"
  "p1_ld_uuf_1024_n100:1024:1024:100:320:15400:400"
)

RESULTS_DIR="/mnt/vdc/test_runs/_results"
mkdir -p "$RESULTS_DIR"

# Common geometry / fuel flags — match existing baselines
COMMON_GEOMETRY="--base-fuel ch_bn --base-density 5e24 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --b-seed 85"
COMMON_DIAG="--diag-profile production"

DOCKER_BASE=(docker run --rm --gpus all
  -v /mnt/vdc/test_runs:/work/runs
  -v /mnt/vdc/laser-plasma-research/simulation:/work/simulation
  -v /mnt/vdc/laser-plasma-research/specs:/work/specs
  -v /mnt/vdc/laser-plasma-research/analysis_scripts:/work/analysis_scripts
  ring-reconnection-fusion:v1.2.0)

DOCKER_NOGPU=(docker run --rm
  -v /mnt/vdc/test_runs:/work/runs
  -v /mnt/vdc/laser-plasma-research/simulation:/work/simulation
  -v /mnt/vdc/laser-plasma-research/analysis_scripts:/work/analysis_scripts
  ring-reconnection-fusion:v1.2.0)

echo "============================================================"
echo "1024² convergence sweep started: $(date)"
echo "Disk free before: $(df -h /mnt/vdc | tail -1 | awk '{print $4}')"
echo "GPU free before: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)"
echo "============================================================"

for spec in "${RUNS[@]}"; do
  IFS=':' read -r run nx nz nppc substeps max_steps dump_period <<< "$spec"
  echo ""
  echo "============================================================"
  echo "[$(date)] STARTING $run"
  echo "  nx=$nx nz=$nz nppc=$nppc substeps=$substeps max_steps=$max_steps dump_period=$dump_period"
  echo "============================================================"

  # Resume-skip: if this run already finished (staged report exists), skip.
  staged_ft="$RESULTS_DIR/$run/${run}_first_transit_summary_global_p11b.txt"
  if [[ -f "$staged_ft" ]]; then
    gft=$(grep "G_FT (first-transit" "$staged_ft" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ')
    echo "  [skip] $run already staged with G_FT = $gft"
    continue
  fi

  # Clean any prior state for this run
  echo "  [clean] removing /work/runs/$run if it exists..."
  "${DOCKER_NOGPU[@]}" rm -rf "/work/runs/$run"

  # Disk check before sim
  free_kb=$(df /mnt/vdc | tail -1 | awk '{print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  echo "  [disk] $free_gb GB free before sim"
  if [[ $nx == "1024" && $free_gb -lt 3000 ]]; then
    echo "  [FAIL] less than 3 TB free, 1024² needs ~2.7 TB — aborting"
    continue
  fi

  # Launch sim
  echo "  [sim] launching..."
  sim_start=$(date +%s)
  "${DOCKER_BASE[@]}" bash -c "
    cd /work && \
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm && \
    mpirun --allow-run-as-root -n 1 python simulation/pb11_ring_reconnection_v15_pulsed.py \
      $COMMON_GEOMETRY $COMMON_DIAG \
      --nx $nx --nz $nz --nppc $nppc --substeps $substeps \
      --max-steps $max_steps --dump-period $dump_period \
      --outdir /work/runs/$run \
      > /work/runs/${run}_sim.log 2>&1
  "
  sim_rc=$?
  sim_end=$(date +%s)
  sim_hours=$(echo "scale=2; ($sim_end - $sim_start) / 3600" | bc)
  echo "  [sim] done in $sim_hours hours, rc=$sim_rc"

  if [[ $sim_rc -ne 0 ]]; then
    echo "  [FAIL] sim returned $sim_rc — check /mnt/vdc/test_runs/${run}_sim.log"
    echo "  [FAIL] last 20 lines of sim log:"
    tail -20 /mnt/vdc/test_runs/${run}_sim.log 2>/dev/null
    continue
  fi

  # Dump count check
  expected_dumps=$((max_steps / dump_period))
  actual_dumps=$(ls /mnt/vdc/test_runs/$run/particles/*.h5 2>/dev/null | wc -l)
  echo "  [sim] $actual_dumps dumps produced (expected ~$expected_dumps)"
  if [[ $actual_dumps -lt $((expected_dumps - 5)) ]]; then
    echo "  [WARN] dump count low — may indicate early termination"
  fi

  # Run analysis chain
  echo "  [chain] launching..."
  chain_start=$(date +%s)
  "${DOCKER_NOGPU[@]}" bash -c "
    cd /work && \
    bash analysis_scripts/run_paper1_chain.sh runs/$run \
      > /work/runs/${run}_chain.log 2>&1
  "
  chain_rc=$?
  chain_end=$(date +%s)
  chain_hours=$(echo "scale=2; ($chain_end - $chain_start) / 3600" | bc)
  echo "  [chain] done in $chain_hours hours, rc=$chain_rc"

  if [[ $chain_rc -ne 0 ]]; then
    echo "  [WARN] chain returned $chain_rc — staging what we have"
  fi

  # Stage out reports + logs (exclude heavy dumps)
  echo "  [stage] copying reports to $RESULTS_DIR/$run/..."
  mkdir -p "$RESULTS_DIR/$run"
  rsync -a \
    --exclude='particles/' --exclude='fields/' --exclude='diags/' --exclude='openpmd/' \
    --exclude='*.bp' --exclude='*.h5' \
    "/mnt/vdc/test_runs/$run/" "$RESULTS_DIR/$run/"
  cp "/mnt/vdc/test_runs/${run}_sim.log" "$RESULTS_DIR/$run/" 2>/dev/null
  cp "/mnt/vdc/test_runs/${run}_chain.log" "$RESULTS_DIR/$run/" 2>/dev/null

  # Quick G_FT readout
  ft_file="/mnt/vdc/test_runs/$run/${run}_first_transit_summary_global_p11b.txt"
  if [[ -f "$ft_file" ]]; then
    gft=$(grep "G_FT (first-transit" "$ft_file" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ')
    echo "  [result] $run: G_FT = $gft"
  fi

  # IMPORTANT: retain particle+field dumps for ALL THREE runs until the
  # user validates that the v0.8 t_cut detector landed on a sensible
  # t_cut for each.  Without retained dumps we cannot re-run the
  # analysis chain with corrected parameters if the detector fires
  # unexpectedly.  Total disk: ~5 TB for the three runs combined.
  # Once validated, the user can manually delete dumps with:
  #   rm -rf /mnt/vdc/test_runs/p1_ld_uuf_{256,512,1024}_n100/{particles,fields}
  echo "  [keep] retaining heavy dumps for $run pending t_cut validation"
done

echo ""
echo "============================================================"
echo "Convergence sweep complete: $(date)"
echo "Results staged at: $RESULTS_DIR"
echo ""
echo "Summary of G_FT and t_cut across runs:"
echo "  (v0.8 detector: smoothed argmin with first-prominent-min selection)"
echo "  (sanity check: t_cut should land near 540 ps for all three runs)"
echo ""
printf "  %-32s %12s %12s\n" "run" "t_cut (ps)" "G_FT"
printf "  %-32s %12s %12s\n" "--------------------------------" "------------" "------------"
for spec in "${RUNS[@]}"; do
  IFS=':' read -r run rest <<< "$spec"
  ft_file="$RESULTS_DIR/$run/${run}_first_transit_summary_global_p11b.txt"
  if [[ -f "$ft_file" ]]; then
    gft=$(grep "G_FT (first-transit" "$ft_file" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ')
    tcut=$(grep "t_cut_ps" "$ft_file" 2>/dev/null | head -1 | awk -F'=' '{print $2}' | awk '{print $1}')
    printf "  %-32s %12s %12s\n" "$run" "$tcut" "$gft"
  else
    printf "  %-32s %12s %12s\n" "$run" "—" "NO RESULT"
  fi
done
echo ""
echo "Validation: if all three t_cut values are within ~100 ps of each"
echo "other AND in the 400-700 ps range, the v0.8 detector worked correctly"
echo "and the G_FT comparison is the convergence claim."
echo ""
echo "Particle+field dumps retained at /mnt/vdc/test_runs/p1_ld_uuf_*_n100/"
echo "Delete manually once validated to free ~5 TB."
echo "============================================================"
