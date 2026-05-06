# 3D Pilot Workflow — pb11 ring reconnection

This directory contains the 3D-pilot extension of the static 8-spot ring
reconnection simulation. The pilot is a targeted test designed to fit on a
single GPU within a wall-time budget you can tolerate, producing the data
needed to (a) answer reviewer questions about 3D effects on the 2D simulation
results that drive Paper 1, and (b) generate publication-quality 3D
visualizations.

## What top-tier acceptance requires

The current paper (v0.3) is submission-ready for **Tier 3** (Physics of
Plasmas, Physical Review Research) with the work already done. To reach
**Tier 1-2** (PRL, Nature Physics, Nature Energy, PRX, Nature Communications)
the gap analysis is:

| Item                       | Current state                  | Top-tier requirement                |
|----------------------------|--------------------------------|-------------------------------------|
| xz convergence             | 256² vs 512² ultrafine, ~3%    | + 1024² ultrafine + Richardson      |
| 3D                         | none                           | pilot at minimum, fuller for Nat-fam |
| 30 ns at 512²              | in progress                    | completed                           |
| Cross-code validation      | WarpX only                     | comparison with PSC/Pegasus++       |
| Lab/observational comparison | cited                        | quantitative figure                 |
| Experimental letter        | none                           | helpful for Nat Energy              |

The 3D pilot in this directory addresses item 2. The other items are pursued
separately. Realistically:
- **PRL**: needs items 1 + 2 + 3. Compute-feasible on your single GPU.
- **PRX / Comms Phys**: needs items 1 + 2 + 3 ideally; Tier 2 acceptance
  possible with item 2 alone.
- **Nature Physics / Energy**: needs all six items. Most require collaborators.

## Files

| File | Purpose |
|------|---------|
| `pb11_ring_reconnection_v12_fuel_center_outer.py` | 2D production deck — UNTOUCHED |
| `make_3d_pilot.py`              | Patcher → 3D pilot deck |
| `pb11_ring_3d_pilot_v1.py`      | 3D pilot deck (generated; do not edit) |
| `pb11_3d_kink_growth.py`        | Diagnostic: kink-mode growth at X-lines |
| `pb11_3d_outflow_tilt.py`       | Diagnostic: out-of-plane outflow tilt |
| `pb11_3d_central_convergence.py`| Diagnostic: central convergence vs 2D |
| `pb11_3d_anim_xz_slice.py`      | Animation: B-field on central xz slice |
| `pb11_3d_anim_yz_xline.py`      | Animation: yz-slice through one X-line |
| `pb11_3d_anim_isosurface.py`    | Animation: 3D proton density isosurfaces |
| `pb11_3d_run_all_diagnostics.py`| Wrapper: runs all 6 diagnostics + animations |

## The three animations

These are essential — top-tier journals expect 3D visualization for any 3D
simulation paper. Each captures something different:

1. **`xz_slice`** — central y-slice of |B|, the direct 3D analog of Paper 1
   Figure 2. Shows the simulation reproduces the 2D phenomenology on the
   symmetry plane. **Cheap**, only needs field dumps.

2. **`yz_xline`** — yz-slice through one X-line. Visual evidence of the
   kink-growth diagnostic: shows whether the X-line stays straight along y
   or develops a sinusoidal bend. **Cheap**, only needs field dumps.

3. **`isosurface`** — true 3D volumetric rendering of proton density,
   colored by mean kinetic energy. The headline 3D figure: shows the
   actual three-dimensional shape of the X-line outflow jets and the
   central convergence cloud. **Expensive** — needs particle dumps at
   every field-dump iteration AND PyVista (~200 MB install).

## Setup (one-time)

```bash
# Copy patcher and analysis scripts to the parent deck's directory
cp make_3d_pilot.py pb11_3d_*.py 3d_pilot_README.md ~/laser-plasma-research/

# Generate the 3D pilot deck
cd ~/laser-plasma-research
python make_3d_pilot.py
# → produces pb11_ring_3d_pilot_v1.py

# Install PyVista on the diagnostics box (only needed for isosurface anim)
pip install "pyvista[all]"   # ~200 MB
```

## Sanity check (~2 hr wall, ~3 GB disk)

```bash
ssh substrate-gpu 'cd ~/laser-plasma-research && \
  python pb11_ring_3d_pilot_v1.py \
    --nx 64 --nz 64 --ny 16 \
    --max-steps 333 --dump-period 50 \
    --diag-profile custom \
    --early-diag-period 20 --early-diag-steps 100'
```

Confirms WarpX builds the 3D grid and writes openPMD output. If this
finishes cleanly, the production pilot will run.

## Production pilot (~40-50 hr wall, ~50-70 GB disk)

**Disk budget revised upward** because we want particle dumps at every
field-dump iteration to support the isosurface animation:

```bash
ssh substrate-gpu 'cd ~/laser-plasma-research && \
  nohup python pb11_ring_3d_pilot_v1.py \
    --max-steps 1340 --dump-period 50 \
    --diag-profile custom \
    --early-diag-period 25 --early-diag-steps 1340 \
    > 3d_pilot_$(date +%Y%m%d).log 2>&1 &'
```

This produces ~50 dumps × (~200 MB field + ~1 GB particle) ≈ ~60 GB peak.
If disk is tight, drop `--early-diag-period` to 50 (cuts dump count in half,
disk to ~30 GB, but the animations get coarser).

Defaults: 256 × 32 × 256 grid, 1340 steps × 149.7 fs ≈ 200 ps. Output goes
to `runs/pb11_3d_pilot_*`.

Monitor:
```bash
ssh substrate-gpu 'cd ~/laser-plasma-research && tail -f 3d_pilot_*.log | grep "Step "'
```

## Run all diagnostics + animations (~15-30 minutes)

```bash
ssh substrate-gpu 'cd ~/laser-plasma-research && \
  python pb11_3d_run_all_diagnostics.py \
    --run-dir runs/pb11_3d_pilot_<timestamp> \
    --compare-2d-zone-report runs/p1_ld_512_4500_ultrafine/zone_report.txt'
```

Outputs land in `runs/pb11_3d_pilot_<ts>/3d_diag/`:

- `3d_kink_growth.{csv,png,txt}`
- `3d_outflow_tilt.{csv,png,txt}`
- `3d_central_convergence.{csv,png,txt}`
- `3d_anim_xz_slice.mp4`
- `3d_anim_yz_xline0.mp4`
- `3d_anim_isosurface.mp4`
- `3d_summary.txt` — combined verdicts

To skip animations (faster, ~5 min):
```bash
python pb11_3d_run_all_diagnostics.py --run-dir <run> --skip-animations
```

## Interpreting the verdicts

Each diagnostic outputs one categorical verdict. Combined:

| Kink     | Tilt        | Convergence  | Paper §3.10 / §5.4 reads                                 |
|----------|-------------|--------------|----------------------------------------------------------|
| STABLE   | ROBUST      | PRESERVED    | "3D pilot confirms the 2D picture across all three diagnostics" |
| MARGINAL | PARTIAL     | one outlier  | "3D pilot identifies effect X as a quantified caveat"   |
| UNSTABLE | STRONG TILT | MODIFIED     | "Substantial 3D modification — paper restructured"      |

## Updating the 2D parent

If you change physics in `pb11_ring_reconnection_v12_fuel_center_outer.py`,
re-run the patcher to regenerate the 3D pilot:

```bash
python make_3d_pilot.py
```

The patcher applies the same eight edits to the new parent. If the parent
changes in a way that breaks the patcher's text matches, the patcher
asserts loudly identifying which edit failed.

## Falling back

If the 3D pilot dies or produces nonsense, the 2D parent deck is untouched
and continues to work exactly as before. The patcher creates a separate
`pb11_ring_3d_pilot_v1.py` only — no shared state, no monkey-patching.

## What you give up by running on a single GPU

The pilot scope is intentionally minimal:
- 256 × 32 × 256 (compared with 512 × 64 × 512 for full 3D). Captures
  enough y-extent (~15 ion skin depths) to resolve kink wavelengths
  but does not resolve sub-spot detail in the y direction.
- 200 ps duration (compared with 30 ns for full long-time physics).
  Captures the active reconnection phase and ~50 ps into equilibration.
- 16 macro-particles per cell (compared with 32 in 2D production).
  Slightly noisier statistics but adequate for the geometry-survivability
  question this pilot is designed to answer.

A reviewer who wants more — and they will, for Tier 1 — will identify
specific extensions:
- Higher resolution in y (256 × 64 × 256 or higher).
- Longer duration (cover full active phase + early equilibration).
- More particles per cell.

These are all single-GPU achievable on extended wall time. None requires
HPC. The pilot establishes the methodology; production scales it up.
