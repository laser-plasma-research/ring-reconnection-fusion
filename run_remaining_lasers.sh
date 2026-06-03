#!/bin/bash
# run_remaining_lasers.sh - fire the 4 remaining laser-scan sims sequentially.
# ml_baseline already ran. Each sim gets the GPU exclusively; we wait for each
# container to exit before launching the next. Runs as UID 1001 (service account),
# detached daemon containers (survive SSH close), named per job.

set -u
IMG=ring-reconnection-fusion:v1.2.1
RUNS=/mnt/vdc/test_runs
SRC=/mnt/vdc/laser-plasma-research
MPI="OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm"
GEOM="--base-fuel ch_bn --base-density 5e24 --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8"

# job: name | b_seed | mode | max_steps | dump_period
JOBS=(
  "ml_fiber_mopa 51  test       1500 20"
  "ml_ndyag_1j   45  test       1500 20"
  "ml_hybrid_10j 120 production 1132 25"
  "ml_ps_fast    361 production 3405 75"
)

run_one() {
  local name=$1 bseed=$2 mode=$3 steps=$4 dump=$5
  echo "============================================================"
  echo "  LAUNCHING $name  (b_seed=$bseed, $mode, steps=$steps)"
  echo "  $(date)"
  echo "============================================================"

  # mode-specific flags: test auto-sets grid/nppc; production pins them explicitly
  local mode_flags
  if [ "$mode" = "test" ]; then
    mode_flags="--test"
  else
    mode_flags="--nx 256 --nz 256 --nppc 200"
  fi

  # clear any stale container of this name + stale run dir
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -rf "$RUNS/$name" 2>/dev/null || true

  docker run -d --name "$name" --gpus all \
    --user 1001:1001 \
    -e CUPY_CACHE_DIR=/work/runs/.cupy_cache -e HOME=/work/runs \
    -v "$RUNS":/work/runs \
    -v "$SRC/simulation":/work/simulation \
    -v "$SRC/analysis_scripts":/work/analysis_scripts \
    -v "$SRC/specs":/work/specs \
    "$IMG" \
    bash -c "cd /work && export $MPI && \
      mpirun --allow-run-as-root -n 1 \
      python -u /work/simulation/pb11_ring_reconnection_v15_pulsed.py \
      $mode_flags $GEOM --b-seed $bseed --diag-profile production \
      --dump-period $dump --max-steps $steps --outdir /work/runs/$name"

  # wait for it to exit
  echo "  waiting for $name to finish..."
  while docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q "$name"; do
    sleep 30
  done

  local rc
  rc=$(docker inspect "$name" --format '{{.State.ExitCode}}' 2>/dev/null)
  echo "  $name EXITED rc=$rc  $(date)"
  docker logs "$name" 2>&1 | tail -3
  echo
}

for spec in "${JOBS[@]}"; do
  run_one $spec
done

echo "============================================================"
echo "  ALL 4 REMAINING SIMS COMPLETE  $(date)"
echo "  run dirs in $RUNS/: ml_fiber_mopa ml_ndyag_1j ml_hybrid_10j ml_ps_fast"
echo "============================================================"
