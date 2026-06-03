# Paper 3 Runbook — Rod + Outer Catcher Multi-Zone Target

This runbook covers the end-to-end procedure for executing Paper 3 production runs and analysis. It assumes you have working SSH access to `gpu-node` with the existing `~/laser-plasma-research` checkout and the `plasma` conda environment, and that the cloud is operating in the same configuration that produced the Paper 1 and Paper 2 data.

## Files in this delivery

```
v15_paper3_patch.py              Minimal v15 patch — adds geometry fields to run_meta.txt
pb11_first_transit_fusion.py     Option B post-analysis (wrap-corrected fusion accounting)
pb11_zone_analysis_paper3.py     Six-zone breakdown with per-zone fusion rates
paper3_jobs.py                   JOBS tuple block for stage_a_simple.py (8 runs)
PAPER3_RUNBOOK.md                This file
```

## What changed from v15 baseline

The simulation script (`pb11_ring_reconnection_v15_pulsed.py`) gets one minimal change: the `[geometry]` block in `run_meta.txt` is extended to record `rod_radius_m`, `outer_radius_m`, `outer_thickness_m`, `lx_m`, `lz_m`, `nx`, and `nz`. All Paper 3 fuel placement, geometry, and zone control is achieved through existing CLI flags — no new flags are introduced. Particle IDs are already saved in WarpX-openPMD dumps (verified against `pb11_particle_trajectories.py` line 217 reading `'id'` directly), so no diagnostic patch is needed.

The Paper 3 four-cell production matrix uses the existing `--fuel-rod` flag for the central rod and the existing `--fuel-ring-full` flag (with `--outer-radius-um 3500 --outer-thickness-um 300`) repurposed for the outer overshoot catcher. Reusing the existing flag namespace keeps Paper 1 and Paper 2 historical configs valid and avoids any risk of breaking older run scripts.

## Step 1 — Apply the v15 patch

Copy the patch script to cloud and run it once:

```bash
scp v15_paper3_patch.py gpu-node:~/laser-plasma-research/

ssh gpu-node '
  cd ~/laser-plasma-research && \
  python v15_paper3_patch.py \
    --target simulation/pb11_ring_reconnection_v15_pulsed.py
'
```

You should see `Patched: simulation/pb11_ring_reconnection_v15_pulsed.py` and a backup file `pb11_ring_reconnection_v15_pulsed.py.bak`. Re-running the script is safe — it detects already-applied patches and exits cleanly. Use `--dry-run` first if you want to preview the change.

To verify by hand:

```bash
ssh gpu-node 'grep "rod_radius_m\|outer_radius_m\|lx_m\|outer_thickness_m\|^.*nx=" \
  ~/laser-plasma-research/simulation/pb11_ring_reconnection_v15_pulsed.py | head -10'
```

You should see the new `fh.write(f'rod_radius_m=...')` and similar lines.

## Step 2 — Stage the analysis scripts

```bash
scp pb11_first_transit_fusion.py pb11_zone_analysis_paper3.py \
    gpu-node:~/laser-plasma-research/analysis_scripts/
```

The scripts use only `numpy` and `openpmd_api`, both already in the `plasma` env (`openpmd_viewer` is also imported but only as a fallback alias).

## Step 3 — Add Paper 3 jobs to the orchestrator

Edit `stage_a_simple.py` on cloud (or locally and push) to append the jobs from `paper3_jobs.py` to the existing `JOBS` list. Open `paper3_jobs.py`, copy the `PAPER3_JOBS` list contents, and paste them inside the `JOBS = [...]` block in `stage_a_simple.py` after the existing entries.

Verify they parse:

```bash
ssh gpu-node '
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && \
  cd ~/laser-plasma-research && \
  python3 -c "
import stage_a_simple
p3_jobs = [j for j in stage_a_simple.JOBS if j[0].startswith(\"p3_\")]
print(f\"Found {len(p3_jobs)} Paper 3 jobs:\")
for j in p3_jobs:
    print(f\"  {j[0]}\")
"'
```

Should list 8 jobs: `p3_preflight`, `p3_ring_only`, `p3_rod_only`, `p3_outer_only`, `p3_rod_plus_outer`, `p3_ring_only_512`, `p3_rod_plus_outer_512`, `p3_rod_plus_outer_ti30`.

## Step 4 — Run the preflight

```bash
ssh gpu-node '
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && \
  cd ~/laser-plasma-research && \
  python3 stage_a_simple.py --jobs p3_preflight
'
```

Expected wall time: ~5 minutes for the simulation, ~5 minutes for analysis. The orchestrator launches the sim, polls until completion, runs the standard analysis pipeline, and rsyncs results back.

After it completes, verify the new geometry fields are in `run_meta.txt`:

```bash
ssh gpu-node 'cat ~/laser-plasma-research/runs/p3_preflight/run_meta.txt | head -25'
```

Required fields under `[geometry]`:
- `rod_radius_m=2.500e-04`
- `outer_radius_m=3.500e-03`
- `outer_thickness_m=3.000e-04`
- `lx_m=9.600e-03`
- `lz_m=9.600e-03`
- `nx=256`
- `nz=256`

If any of these are missing, the patch did not apply correctly — re-run Step 1.

Also verify the particle dumps include IDs by running the Option B analysis on the preflight output:

```bash
ssh gpu-node '
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && \
  cd ~/laser-plasma-research && \
  python3 analysis_scripts/pb11_first_transit_fusion.py \
    --run-dir runs/p3_preflight --max-dumps 3
'
```

Expected output: a few dump-by-dump progress lines, then "Wrote: ...summary.txt". If the script fails with `KeyError: 'id'` or similar, IDs are not being saved and we need to investigate the diagnostic config (this would be unexpected given the existing trajectories script reads them).

## Step 5 — Launch Tier 1 production

After preflight passes:

```bash
ssh gpu-node '
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && \
  cd ~/laser-plasma-research && \
  nohup python3 -u stage_a_simple.py \
    --jobs p3_ring_only,p3_rod_only,p3_outer_only,p3_rod_plus_outer \
    > paper3_tier1.log 2>&1 &'
```

Total wall time: ~6 hours for all four runs (serial). Monitor progress:

```bash
ssh gpu-node 'tail -f ~/laser-plasma-research/paper3_tier1.log'
```

While they run, the orchestrator handles analysis automatically per the existing pipeline. Each run produces the standard `fusion_diagnostics.csv`, `reconnection_rate_*.csv`, etc. The Paper 3-specific analysis is run separately in Step 7.

## Step 6 — Launch Tier 2 convergence (after Tier 1 succeeds)

```bash
ssh gpu-node '
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && \
  cd ~/laser-plasma-research && \
  nohup python3 -u stage_a_simple.py \
    --jobs p3_ring_only_512,p3_rod_plus_outer_512 \
    > paper3_tier2.log 2>&1 &'
```

Total wall time: ~14 hours for both 512² runs. Storage peak: ~50 GB combined.

## Step 7 — Run Paper 3-specific analysis

After all production runs complete, run the two Paper 3-specific analysis scripts on each run. From cloud:

```bash
ssh gpu-node '
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && \
  cd ~/laser-plasma-research && \
  for run in runs/p3_ring_only runs/p3_rod_only runs/p3_outer_only \
             runs/p3_rod_plus_outer runs/p3_ring_only_512 \
             runs/p3_rod_plus_outer_512; do
    echo "=== $run ==="
    python3 analysis_scripts/pb11_zone_analysis_paper3.py --run-dir $run
    python3 analysis_scripts/pb11_first_transit_fusion.py --run-dir $run
  done
'
```

Each run produces:
- `<run>_zones_paper3.txt` — six-zone E95 and population time series
- `<run>_zone_fusion_rates_paper3.csv` — per-zone fusion rates (instantaneous)
- `<run>_fusion_rate_first_transit.csv` — wrap-corrected fusion rates (Option B)
- `<run>_wrap_artifact_metrics.csv` — diagnostic of wrap-effect magnitude
- `<run>_first_transit_summary.txt` — comparison of first-transit vs legacy

## Step 8 — Pull results locally and analyze

```bash
mkdir -p ~/LaserFusionResearch/research/laser-plasma-research/runs/paper03

rsync -av --include='runs/p3_*/run_meta.txt' \
          --include='runs/p3_*/*.csv' \
          --include='runs/p3_*/*.txt' \
          --include='runs/p3_*/' \
          --exclude='runs/p3_*/particles' \
          --exclude='runs/p3_*/diags' \
          --exclude='*' \
          gpu-node:~/laser-plasma-research/ \
          ~/LaserFusionResearch/research/laser-plasma-research/
```

(Adjust paths as needed; the `--exclude='runs/p3_*/particles'` line keeps the heavy particle dumps on cloud.)

The four key comparisons for the paper are:

```
Δ_rod         = E_fusion(p3_rod_only)          − E_fusion(p3_ring_only)
Δ_outer       = E_fusion(p3_outer_only)        − E_fusion(p3_ring_only)
Δ_both        = E_fusion(p3_rod_plus_outer)    − E_fusion(p3_ring_only)
Δ_outer_added = E_fusion(p3_rod_plus_outer)    − E_fusion(p3_rod_only)
```

Use the **first-transit** numbers (from `*_first_transit_summary.txt`) for the headline; the legacy numbers are kept for wrap-artifact disclosure in supplementary material.

The convergence comparison is `(E_full / E_baseline)_512 vs (E_full / E_baseline)_256`. If the gain ratio differs by less than 10% between resolutions, the result is grid-converged.

## Disk-space management

If you're getting tight on cloud disk during Tier 2:

```bash
# After analysis completes, archive particles dirs to free space
ssh gpu-node '
  cd ~/laser-plasma-research/runs && \
  for run in p3_ring_only p3_rod_only p3_outer_only p3_rod_plus_outer \
             p3_ring_only_512 p3_rod_plus_outer_512; do
    if [ -d "$run/particles" ]; then
      tar czf "$run/particles.tar.gz" "$run/particles" && rm -rf "$run/particles"
    fi
  done
'
```

This keeps everything for re-analysis if needed but reclaims most of the per-run footprint.

## Troubleshooting

**Patch script reports "could not find anchor block":** Someone has already modified the run_meta.txt writing block. Open the script around line 3216 and apply the change manually — add the `rod_radius_m`, `outer_radius_m`, `outer_thickness_m`, `lx_m`, `lz_m`, `nx`, `nz` lines after the existing `spot_radius_m` line.

**Preflight fails with simulation error during particle initialization:** Most likely cause is the rod density gradient at higher `--rod-density` values being numerically harsh. Reduce to 2.5e25 for a re-test, then ramp back if it works.

**`pb11_first_transit_fusion.py` reports zero outer-zone particles:** Either the run is too short to push protons past R=2900 µm, or the box is smaller than expected (check `lx_m` in `run_meta.txt`). For an 18000-step LD run at default boxsize, you should see fast protons reaching the outer zone by ~150 ps.

**Massive wrap fraction (>50%) reported:** Indicates many particles have hit the periodic boundary. Either the run is unusually long or the outer protons are moving very fast. Sanity check by examining `wrap_artifact_metrics.csv` — `frac_wrapped` should grow monotonically and reach a plateau by end of run.

## What this delivery does NOT include

- **A new fuel preset for `b18h22 + H2 mix at H:B≈2:1`** — the patent's preferred catcher composition. Current `b18h22` preset (H:B=22/18≈1.22:1) is the closest available; if the rod-only fusion contribution underperforms relative to expectation, consider adding a custom 2:1 preset.
- **`--T-ion-ev` CLI flag** — the Ti=30 eV sensitivity run requires either editing `T_ION_EV` in the v15 script before launching, or adding the flag (one-line argparse addition). The Tier 3 entry in `paper3_jobs.py` is staged but commented; activate when the flag exists.
- **Stopping-range correction for outer-catcher fusion accounting** — addressed via methods-section disclosure rather than code (the simulation is collisionless, so a per-particle stopping-range filter is a post-hoc model layered on top of the wrap-corrected first-transit rate).
- **Retroactive Paper 2 analysis** — explicitly excluded per the latest scope decision; Paper 3 stands on its own data.

## Total cloud budget recap

```
Tier 0 preflight            ~5 min          (1 run @ 256², 5K steps)
Tier 1 production           ~6 hr           (4 runs @ 256², 18K steps)
Tier 2 convergence          ~14 hr          (2 runs @ 512², 18K steps)
Tier 3 sensitivity           ~1.5 hr        (1 run @ 256², 18K steps; optional)
Analysis                     ~10 min total   (Steps 7 across all runs)
                             ─────────────
Total                       ~22 hr           cloud GPU time
                            ~85 GB           peak storage on cloud
                            ~5 GB            local storage after analysis sync
```

Comfortable inside one weekend on `gpu-node`.
