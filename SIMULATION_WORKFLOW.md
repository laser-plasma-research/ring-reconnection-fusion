# Laser-Plasma Ring Reconnection Simulation — Workflow Guide

A complete reference for running, analyzing, transferring, and managing WarpX hybrid-PIC simulations of the 8-spot laser-driven ring reconnection geometry. Covers the simulation pipeline from launch through paper-ready figures.

**Target:** WarpX hybrid-PIC on cloud H100 GPU (`<GPU_HOST>`), local M2 Max for analysis.

**Version:** 1.0 — Initial workflow capture

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Resolution Criteria — Critical First Check](#resolution-criteria)
3. [Simulation Launch — Full Reference](#simulation-launch)
4. [Run Monitoring](#run-monitoring)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Data Transfer (Cloud → Mac)](#data-transfer)
7. [Cleanup & Disk Management](#cleanup--disk-management)
8. [Process Recovery (When Things Go Wrong)](#process-recovery)
9. [Common Workflow Recipes](#common-workflow-recipes)
10. [CLI Argument Reference](#cli-argument-reference)
11. [Troubleshooting](#troubleshooting)
12. [Hardware & Compute Estimates](#hardware--compute-estimates)

---

## Overview & Architecture

### Compute environment

| Component | Location | Use |
|---|---|---|
| Cloud H100 GPU | `ssh <GPU_HOST>` | All WarpX simulation runs + analysis |
| Mac M2 Max (96 GB) | local | Final figures, paper writing, occasional small analyses |

**Why cloud-only for simulations:** WarpX does not support Apple Metal GPU acceleration. Mac CPU runs are 5-10× slower than H100 GPU. We use the Mac only for storing/viewing analysis outputs.

### Directory structure

**Cloud (`~/laser-plasma-research/`):**
```
laser-plasma-research/
├── simulation/                                  # Main simulation script + analysis variants
│   ├── pb11_ring_reconnection_v12_fuel_center_outer.py
│   ├── pb11_post_analysis.py                    # Generates post_analysis_report.txt
│   └── pb11_zone_analysis.py                    # Generates zones report
├── analysis_scripts/                            # Analysis pipeline scripts
│   ├── visualize_all.py                         # Composite plots + b_evolution.mp4
│   ├── pb11_phase_analysis.py
│   ├── pb11_reconnection_rate_offline.py
│   ├── pb11_fusion_diagnostics.py               # V2 with spectrum heatmap
│   └── pb11_particle_animations.py              # V2 density heatmaps + tracer
├── runs/                                        # All simulation output directories
│   ├── p1_hd_512_1500_ultrafine/                # Example: HD ultrafine run
│   │   ├── particles/                           # H5 particle dumps (~16 GB each)
│   │   ├── fields/                              # H5 field dumps
│   │   ├── figures/                             # Generated PNG/MP4/GIF
│   │   ├── run_meta.txt                         # Parameters used
│   │   ├── run.log                              # WarpX output log
│   │   ├── post_analysis_report.txt             # Time-resolved analysis
│   │   ├── phase_analysis_report.txt            # Phase identification
│   │   ├── zone_report.txt                      # Per-zone E95 over time
│   │   ├── reconnection_summary.txt
│   │   ├── reconnection_rate_offline.csv        # Full reconnection rate timeseries
│   │   ├── fusion_diagnostics.csv               # V2 fusion-grade fractions
│   │   └── ...
│   └── p1_<other_run>/
└── post_analysis_reports/                       # Aggregated zone reports (older convention)
```

**Mac (`~/LaserFusionResearch/research/laser-plasma-research/`):** Same structure, but **no `particles/` or `fields/` directories** — those are huge (hundreds of GB) and stay on cloud only.

### Conda environment

**Always activate before running anything:**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma
```

The `plasma` env contains: WarpX, openpmd-viewer, h5py, numpy, scipy, matplotlib, ffmpeg.

---

## Resolution Criteria

**THE most important check before any HD-density run.**

### The rule

WarpX hybrid-PIC requires `cells per d_i ≥ 4` for converged physics, where `d_i` is the ion inertial length:

```
d_i = c / omega_pi
omega_pi = sqrt(n * e^2 / (epsilon_0 * m_p))
```

For our typical configurations:
- **HD density (5×10²⁵ m⁻³):** d_i = **32.2 µm**
- **LD density (5×10²⁴ m⁻³):** d_i = **102 µm** (3.16× larger because d_i ∝ 1/√n)

### Cell size calculation

```
cell_size = domain_size / nx
```

The simulation script auto-scales the domain to `120 × d_i` if `--lx-min-um` is omitted. Otherwise uses `max(120 × d_i, args.lx_min_um × 1e-6)`.

### Resolution table for 9.6 mm domain (default `--lx-min-um 9600`)

| Grid | Cell size | cells/d_i (LD) | cells/d_i (HD) | LD status | HD status |
|---|---|---|---|---|---|
| 256² | 37.5 µm | **2.72** | 0.86 | Marginal | ❌ Disastrous |
| **512²** | 18.75 µm | **5.44** | 1.72 | ✅ Well-resolved | ❌ Under-resolved |
| 1024² | 9.4 µm | 10.86 | 3.43 | ✅ | ⚠️ Marginal |
| 2048² | 4.7 µm | 21.7 | 6.85 | ✅ | ✅ |

### Resolution table for scaled geometry (R=1mm, σ=125 µm, auto-domain ≈ 3.86 mm)

| Grid | Cell size | cells/d_i (HD) | Status |
|---|---|---|---|
| 256² | 15.1 µm | 2.13 | ❌ Under-resolved (convergence baseline only) |
| **512²** | 7.55 µm | **4.27** | ✅ Well-resolved |

### Quick verification of any completed run

```bash
ssh <GPU_HOST> 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && python3 -c "
import openpmd_viewer as ov
ts = ov.OpenPMDTimeSeries(\"/home/<USER>/laser-plasma-research/runs/<RUN_NAME>/particles\")
it = list(ts.iterations)[0]
x, z = ts.get_particle([\"x\", \"z\"], species=\"proton\", iteration=it)
print(f\"Domain: {(x.max()-x.min())*1e3:.3f} mm\")
print(f\"Cell size: {(x.max()-x.min())*1e6/512:.2f} um\")
"'
```

**Replace `<RUN_NAME>` and `512` with the actual run name and grid.**

---

## Simulation Launch

### Standard launch template

```bash
ssh <GPU_HOST> 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && cd ~/laser-plasma-research && nohup python -u simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \
  --base-fuel ch_bn \
  --base-density <DENSITY> \
  --b-seed 85 \
  --n-spots 8 \
  --lx-min-um 9600 \
  --nx <GRID> --nz <GRID> \
  --max-steps <STEPS> \
  --dump-period <DUMP_PERIOD> \
  --diag-profile custom \
  --outdir runs/<RUN_NAME> \
  > <RUN_NAME>.log 2>&1 & echo "PID: $!"'
```

**Note the PID** — useful for tracking and killing the process if needed.

### Specific run examples

**LD 512² ultrafine (early dynamics, ~3 hr, ~1.46 TB disk):**
```bash
--base-density 5e24 --nx 512 --nz 512 --max-steps 4500 --dump-period 50
--outdir runs/p1_ld_512_4500_ultrafine
```

**LD 256² ultrafine (convergence baseline, ~30 min, ~340 GB):**
```bash
--base-density 5e24 --nx 256 --nz 256 --max-steps 4500 --dump-period 50
--outdir runs/p1_ld_256_4500_ultrafine
```

**LD 512² long (30 ns equilibration, ~4 days, ~1.6 TB):**
```bash
--base-density 5e24 --nx 512 --nz 512 --max-steps 200000 --dump-period 2000
--outdir runs/p1_ld_512_200k_long
```

**HD resolved 512² scaled (well-resolved at HD, ~1.5 hr, ~770 GB):**
```bash
--base-density 5e25 --nx 512 --nz 512 --ring-radius-um 1000 --spot-radius-um 125
--max-steps 1500 --dump-period 33
--outdir runs/p1_hd_512_resolved
# Note: omit --lx-min-um to let domain auto-scale to 120*d_i = 3.86 mm
```

### Pre-flight check (recommended for new configurations)

Before committing to a long run, do a 50-step preflight to verify geometry and parameters:

```bash
ssh <GPU_HOST> 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && cd ~/laser-plasma-research && nohup python -u simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \
  <YOUR_PARAMS> \
  --max-steps 50 \
  --dump-period 25 \
  --outdir runs/<RUN_NAME>_preflight \
  > preflight.log 2>&1 & echo "PID: $!"'
```

Wait ~5 minutes, then verify:

```bash
ssh <GPU_HOST> 'grep -E "Geometry|d_i=|B-seed|Grid:|particle access probe|SIMULATION COMPLETE|ERROR|Traceback" ~/laser-plasma-research/preflight.log | head -30'
```

Look for:
- ✅ `Geometry: 8-spot ring at R=<X> um` — matches your `--ring-radius-um`
- ✅ `B-seed: 85.0 T  |  v_A=<X> m/s  |  d_i=<X> um` — matches expected density
- ✅ `Grid: <X>x<X> NPPC=400`
- ✅ `[particle access probe] OK -- <X> protons reachable`
- ✅ `SIMULATION COMPLETE`

If preflight passes, delete it and launch the full run:
```bash
ssh <GPU_HOST> 'rm -rf ~/laser-plasma-research/runs/<RUN_NAME>_preflight && rm -f ~/laser-plasma-research/preflight.log'
```

---

## Run Monitoring

### Active simulation status

```bash
ssh <GPU_HOST> 'pgrep -af "ring_reconnection" | grep -v grep | head -3; echo ""; tail -20 ~/laser-plasma-research/<RUN_NAME>.log'
```

**Look for:**
- Process listed (PID active)
- `STEP <N> ends. TIME = ... DT = ...` advancing
- `Avg. per step = <X>` — performance metric
- `--- INFO    : Writing openPMD file ...` — periodic dumps

### Quick "is it done?" check

```bash
ssh <GPU_HOST> 'pgrep -af "ring_reconnection" | grep -v grep || echo "(SIMULATION FINISHED)"; echo ""; tail -10 ~/laser-plasma-research/<RUN_NAME>.log'
```

### Estimating remaining time

Look at the latest line for `Avg. per step = X.XX s`. Remaining time:
```
remaining_steps × avg_per_step = remaining_seconds
```

Typical performance on H100:
- 256² × 100M particles: ~0.4 sec/step
- 512² × 100M particles: ~1.5-3 sec/step

### Disk usage during run

```bash
ssh <GPU_HOST> 'du -sh ~/laser-plasma-research/runs/<RUN_NAME> 2>/dev/null && df -h ~/laser-plasma-research | head -2'
```

---

## Analysis Pipeline

After a simulation completes, run the **full 7-stage analysis pipeline**. All stages can be chained in a single nohup background command.

### Full pipeline launch

```bash
ssh <GPU_HOST> 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma && cd ~/laser-plasma-research && nohup bash -c "
echo === STAGE 1/7: VISUALIZE_ALL === && \
python -u analysis_scripts/visualize_all.py --dir runs/<RUN_NAME> --animate 2>&1 && \
echo === STAGE 2/7: POST_ANALYSIS === && \
python -u simulation/pb11_post_analysis.py --dir runs/<RUN_NAME> 2>&1 && \
echo === STAGE 3/7: ZONE_ANALYSIS === && \
python -u simulation/pb11_zone_analysis.py --dir runs/<RUN_NAME> 2>&1 && \
echo === STAGE 4/7: PHASE_ANALYSIS === && \
python -u analysis_scripts/pb11_phase_analysis.py --dir runs/<RUN_NAME> 2>&1 && \
echo === STAGE 5/7: RECONNECTION_RATE_OFFLINE === && \
python -u analysis_scripts/pb11_reconnection_rate_offline.py --dir runs/<RUN_NAME> 2>&1 && \
echo === STAGE 6/7: FUSION_DIAGNOSTICS_V2 === && \
python -u analysis_scripts/pb11_fusion_diagnostics.py --dir runs/<RUN_NAME> 2>&1 && \
echo === STAGE 7/7: PARTICLE_ANIMATIONS_V2 === && \
python -u analysis_scripts/pb11_particle_animations.py --dir runs/<RUN_NAME> 2>&1 && \
echo === ALL ANALYSIS COMPLETE ===" > <RUN_NAME>_analysis.log 2>&1 & echo "Analysis PID: $!"'
```

### What each stage produces

| Stage | Script | Outputs |
|---|---|---|
| 1 | `visualize_all.py` | `b_evolution.mp4`, `composite_evolution.png`, `composite_snapshots.png`, `figures/individual/*` cache |
| 2 | `pb11_post_analysis.py` | `post_analysis_report.txt` |
| 3 | `pb11_zone_analysis.py` | `zone_report.txt` (per-zone E95 timeseries) |
| 4 | `pb11_phase_analysis.py` | `phase_analysis_report.txt` (acceleration phases, frequency analysis) |
| 5 | `pb11_reconnection_rate_offline.py` | `reconnection_summary.txt`, `reconnection_rate_offline.csv` |
| 6 | `pb11_fusion_diagnostics.py` | `fusion_diagnostics.csv`, `fusion_grade_fractions_vs_time.png`, `spectrum_evolution_heatmap.png`, `peak_snapshot_fusion_grade.png`, `energy_distribution_by_zone_at_peak.png` |
| 7 | `pb11_particle_animations.py` | `particle_density_by_energy.gif`, `acceleration_tracer.gif`, `energy_spectrum_evolution.gif` (and matching `.mp4` if dimensions allow) |

### Monitor pipeline progress

```bash
ssh <GPU_HOST> 'pgrep -af "pb11\|visualize_all" | grep -v grep | head -5; echo ""; tail -25 ~/laser-plasma-research/<RUN_NAME>_analysis.log'
```

**Look for:**
- `=== STAGE X/7: ... ===` markers showing current stage
- `=== ALL ANALYSIS COMPLETE ===` at the end
- No ERROR or Traceback messages

### Estimated pipeline times (H100)

| Run size | Total pipeline time |
|---|---|
| Small (256² × 91 dumps) | ~25-35 min |
| Medium (512² × 47 dumps) | ~35-50 min |
| Large (512² × 91 dumps) | ~50-70 min |

Stage 1 (visualize_all with --animate) and Stage 7 (particle_animations) are the slowest — they each read all particle dumps in passes.

### Common issue: 0-byte MP4 files

ffmpeg's h264 encoder requires even pixel dimensions. Stages 6-7 produce some matplotlib figures with odd dimensions (e.g., 924×563), causing MP4 encoding to fail with `Error sending frames to consumers`. The script automatically falls back to GIF — those work fine.

**Always remove 0-byte MP4s after the pipeline completes:**

```bash
ssh <GPU_HOST> 'cd ~/laser-plasma-research/runs/<RUN_NAME>/figures && for f in *.mp4; do [ -s "$f" ] || (echo "Removing 0-byte: $f" && rm -f "$f"); done && echo "Done"'
```

### Inventory check after pipeline

```bash
ssh <GPU_HOST> 'echo "=== FIGURES ===" && ls -la ~/laser-plasma-research/runs/<RUN_NAME>/figures/ | grep -E "mp4|gif|png" | awk "{print \$5, \$9}" | sort -k 2; echo ""; echo "=== RUN DIR ===" && ls -la ~/laser-plasma-research/runs/<RUN_NAME>/ | grep -E "csv|txt|log" | awk "{print \$5, \$9}" | sort -k 2'
```

Expect ~10 figures + ~12 reports/CSVs/logs.

---

## Data Transfer

### Pull all artifacts to Mac

After analysis pipeline completes and 0-byte MP4s are removed:

```bash
# Create local directory
mkdir -p ~/LaserFusionResearch/research/laser-plasma-research/runs/<RUN_NAME>/figures && \
\
# Pull figures (only PNG/GIF/MP4)
rsync -avh --include='*.mp4' --include='*.gif' --include='*.png' --exclude='*' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/figures/' \
  ~/LaserFusionResearch/research/laser-plasma-research/runs/<RUN_NAME>/figures/ && \
\
# Pull reports + CSVs + logs (everything text)
rsync -avh \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/post_analysis_report.txt' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/phase_analysis_report.txt' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/reconnection_summary.txt' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/reconnection_rate_offline.csv' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/fusion_diagnostics.csv' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/zone_report.txt' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/run_meta.txt' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/run.log' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/fusion_accounting_notes.txt' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/fusion_rate_power_by_iter.csv' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/reconnection_rate_by_iter.csv' \
  <GPU_HOST>:'~/laser-plasma-research/runs/<RUN_NAME>/step_time_index.csv' \
  ~/LaserFusionResearch/research/laser-plasma-research/runs/<RUN_NAME>/
```

### Verify Mac state matches cloud

```bash
echo "=== MAC FIGURES ===" && ls -la ~/LaserFusionResearch/research/laser-plasma-research/runs/<RUN_NAME>/figures/ | grep -E "mp4|gif|png" | awk '{print $5, $9}' | sort -k 2; echo; echo "=== MAC RUN DIR ===" && ls -la ~/LaserFusionResearch/research/laser-plasma-research/runs/<RUN_NAME>/ | grep -E "csv|txt|log" | awk '{print $5, $9}' | sort -k 2
```

File sizes should match what you see on cloud.

### Transfer single file (quick check)

For pulling just one file (e.g., to view a specific figure):

```bash
scp <GPU_HOST>:~/laser-plasma-research/runs/<RUN_NAME>/figures/<FILE>.png ~/Downloads/
```

### Transfer simulation script changes (Mac → Cloud)

When you've edited the script locally:

```bash
rsync -avhc --progress \
  /Users/brenworth2/LaserFusionResearch/research/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \
  <GPU_HOST>:~/laser-plasma-research/simulation/
```

The `-c` flag uses checksums (not timestamps) — only transfers if files actually differ.

---

## Cleanup & Disk Management

### Disk space check

```bash
ssh <GPU_HOST> 'df -h ~/laser-plasma-research | head -2'
```

Cloud has 3.5 TB total. We aim to keep at least ~1 TB free for the next run.

### Clean particles + fields after analysis (frees most disk)

After analysis pipeline completes AND figures/reports are pulled to Mac:

```bash
ssh <GPU_HOST> 'echo "=== Before cleanup ===" && du -sh ~/laser-plasma-research/runs/<RUN_NAME>/particles ~/laser-plasma-research/runs/<RUN_NAME>/fields 2>/dev/null && df -h ~/laser-plasma-research | head -2 && echo "" && rm -rf ~/laser-plasma-research/runs/<RUN_NAME>/particles ~/laser-plasma-research/runs/<RUN_NAME>/fields ~/laser-plasma-research/runs/<RUN_NAME>/figures/cache && echo "" && echo "=== After cleanup ===" && df -h ~/laser-plasma-research | head -2'
```

This deletes:
- `particles/` (largest — typically 300 GB to 1.5 TB)
- `fields/` (typically 1-50 GB)
- `figures/cache/` (visualize_all.py cache, ~50-200 MB)

Keeps:
- All figures, reports, CSVs, run.log, run_meta.txt

### Full run deletion (only if you want to start fresh)

```bash
ssh <GPU_HOST> 'rm -rf ~/laser-plasma-research/runs/<RUN_NAME> && rm -f ~/laser-plasma-research/<RUN_NAME>.log ~/laser-plasma-research/<RUN_NAME>_analysis.log && echo "Removed"'
```

### Inventory current cloud state

```bash
ssh <GPU_HOST> 'echo "=== Disk ===" && df -h ~/laser-plasma-research | head -2; echo ""; echo "=== Runs ===" && du -sh ~/laser-plasma-research/runs/*/ 2>/dev/null | sort -h'
```

---

## Process Recovery

### When a simulation has hung or you need to kill it

#### Step 1: Identify what's running

```bash
ssh <GPU_HOST> 'ps -u <USER> -o pid,etime,cmd | grep -E "python|warpx|amrex" | grep -v grep'
```

#### Step 2: Force-kill ALL Python processes

```bash
ssh <GPU_HOST> 'pkill -9 -u <USER> python 2>/dev/null; pkill -9 -u <USER> -f "warpx\|pb11\|simulation\|analysis_scripts" 2>/dev/null; sleep 5; echo "After kill:" && ps -u <USER> -o pid,etime,cmd | grep -E "python|warpx|amrex" | grep -v grep || echo "(all clear)"'
```

#### Step 3: Clean up partial run files

If a run was killed mid-execution, the run directory may have partial dumps. Remove it:

```bash
ssh <GPU_HOST> 'rm -rf ~/laser-plasma-research/runs/<RUN_NAME> && rm -f ~/laser-plasma-research/<RUN_NAME>.log && echo "Removed partial run"'
```

#### Step 4: Verify clean state

```bash
ssh <GPU_HOST> 'echo "=== Processes ===" && ps -u <USER> -o pid,etime,cmd | grep -E "python|warpx|amrex" | grep -v grep || echo "(no python/warpx running)"; echo ""; echo "=== Disk ===" && df -h ~/laser-plasma-research | head -2; echo ""; echo "=== Runs ===" && ls ~/laser-plasma-research/runs/'
```

### When `rm -rf` says "Directory not empty"

This means a process is still actively writing to the directory. Do NOT just retry — find the process first:

```bash
ssh <GPU_HOST> 'lsof +D ~/laser-plasma-research/runs/<RUN_NAME> 2>/dev/null | head -5'
```

This shows which PID has files open. Kill it directly:

```bash
ssh <GPU_HOST> 'kill -9 <PID>; sleep 5; rm -rf ~/laser-plasma-research/runs/<RUN_NAME>'
```

### When cloud disk is unexpectedly full

```bash
ssh <GPU_HOST> 'du -sh ~/laser-plasma-research/runs/*/ 2>/dev/null | sort -h | tail -10'
```

Identifies the largest run directories. Delete `particles/` and `fields/` from completed/analyzed ones.

---

## Common Workflow Recipes

### Recipe A: Full new simulation start to finish

```
[1] Verify cloud has space → df check
[2] Decide config (density, grid, steps, dump period)
[3] Run preflight (~5 min) → verify geometry
[4] Delete preflight, launch full run
[5] Monitor every 30-60 min until SIMULATION COMPLETE
[6] Run 7-stage analysis pipeline
[7] Monitor analysis until ALL ANALYSIS COMPLETE
[8] Remove 0-byte MP4s
[9] Inventory check
[10] Pull to Mac (figures + reports)
[11] Verify Mac state
[12] Cleanup cloud (delete particles/fields)
```

### Recipe B: Convergence study (paired runs)

For demonstrating convergence at two grid resolutions, run identical configs at 256² and 512²:

```
[1] Run 256² (faster, smaller — convergence baseline)
[2] Analyze 256², pull to Mac
[3] Cleanup 256² particles
[4] Run 512² (main result)
[5] Analyze 512², pull to Mac
[6] Cleanup 512² particles
[7] Compare zone_report.txt and fusion_diagnostics.csv side-by-side
```

If 256² and 512² give similar physics, 512² is converged. If they differ significantly, may need 1024² for full convergence.

### Recipe C: Quick parameter exploration

For testing a new configuration before committing to a long run:

```bash
# Preflight (50 steps, ~5 min)
ssh <GPU_HOST> '... --max-steps 50 --dump-period 25 --outdir runs/_test_<TAG>'

# Verify
ssh <GPU_HOST> 'tail -20 ~/laser-plasma-research/_test_<TAG>.log | grep -E "Geometry|d_i|Grid|SIMULATION COMPLETE"'

# Cleanup
ssh <GPU_HOST> 'rm -rf ~/laser-plasma-research/runs/_test_<TAG> && rm -f ~/laser-plasma-research/_test_<TAG>.log'
```

### Recipe D: Edit simulation script

When you need to modify the WarpX simulation script:

```bash
# 1. Edit local copy (use your preferred editor)
vim /Users/brenworth2/LaserFusionResearch/research/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py

# 2. Sanity check syntax
python3 -c "import ast; ast.parse(open('/Users/brenworth2/LaserFusionResearch/research/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py').read())" && echo "Syntax OK"

# 3. Push to cloud
rsync -avhc --progress \
  /Users/brenworth2/LaserFusionResearch/research/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \
  <GPU_HOST>:~/laser-plasma-research/simulation/

# 4. Verify on cloud
ssh <GPU_HOST> 'grep -n "your-edit-marker" ~/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py | head -5'

# 5. Run preflight to verify behavior
[See Recipe C]
```

---

## CLI Argument Reference

Full reference for `pb11_ring_reconnection_v12_fuel_center_outer.py`:

### Grid & Domain
| Argument | Default | Description |
|---|---|---|
| `--nx`, `--nz` | 0 (auto) | Grid resolution. Set explicitly: 256, 512, 1024, etc. |
| `--lx-min-um` | 0.0 (auto) | Override domain size in microns. Default lets domain auto-scale to 120 × d_i. Use 9600 for original 9.6 mm domain. |

### Geometry (NEW v12.15: ring/spot now CLI-configurable)
| Argument | Default | Description |
|---|---|---|
| `--n-spots` | 8 | Number of laser spots. Use 1 for TNSA-equivalent baseline (forces ring radius to 0). |
| `--ring-radius-um` | 2400.0 | Laser spot ring radius in microns. Use 1000 for HD scaled geometry to fit auto-domain at HD density. |
| `--spot-radius-um` | 300.0 | Spot Gaussian sigma in microns. Scale proportionally with ring radius. Min: σ/d_i ≥ 2.95. |
| `--rod-radius-um` | 75.0 | Central rod fuel radius (Paper 3 hybrid fuel). |
| `--outer-radius-um` | 1050.0 | Outer catcher ring inner radius (Paper 3). |
| `--outer-thickness-um` | 100.0 | Outer catcher ring thickness. |
| `--face-edge-um` | 15.0 | Face edge thickness. |
| `--sigma-scale` | 1.15 | Multiplier on SPOT_RADIUS_M to get the actual Gaussian sigma. |

### Plasma & Fields
| Argument | Default | Description |
|---|---|---|
| `--b-seed` | 300.0 | Seed B-field magnitude in T. We use 85.0 T for our papers. |
| `--field-mode` | `applied` | Field configuration mode. |
| `--j-scale` | 0.15 | Current scaling factor. |
| `--no-bfield` | (flag) | Run without B-field (control). |
| `--eta-scale` | 1.0 | Resistivity scale factor. |
| `--te-ev` | 2200.0 | Electron temperature in eV. |

### Fuel Configuration
| Argument | Default | Description |
|---|---|---|
| `--base-fuel` | `p11b` | Base plasma composition. We use `ch_bn` (CH-BN, PERLA-style target). |
| `--base-density` | 5e24 | Base plasma density in m⁻³. **5e24 = LD (well-resolved at 512²×9.6mm), 5e25 = HD (needs scaled geometry or 2048²)**. |
| `--rod-fuel` | `ammonia_borane` | Central rod fuel composition. |
| `--rod-density` | 5e24 | Rod density. |
| `--ring-fuel` | `lib_equal` | Outer ring catcher fuel. |
| `--ring-density` | 5e24 | Ring density. |
| `--fuel-rod` | (flag) | Enable central rod fuel region (Paper 3). |
| `--fuel-ring-full` | (flag) | Enable outer ring fuel (Paper 3). |

### Run Control
| Argument | Default | Description |
|---|---|---|
| `--max-steps` | 0 (auto) | Total integration steps. dt is set by cyclotron period (~0.15 ps at B=85T). Sim time = max_steps × dt. |
| `--dump-period` | 2500 | Steps between H5 particle/field dumps. **Use 33 for 5 ps cadence, 50 for 7.5 ps, 2000 for 300 ps**. |
| `--diag-profile` | `custom` | Diagnostic profile. Use `custom` to honor explicit dump-period. |
| `--ramp-steps` | 100 | Steps to ramp up the applied field. |
| `--rotate` | (flag) | Enable rotating Jy current driver (Paper 2). |
| `--freq` | 5e8 | Rotation frequency in Hz (only with --rotate). |
| `--outdir` | `./pb11_diags` | Output directory. Use `runs/<RUN_NAME>` convention. |

---

## Troubleshooting

### Symptom: Pipeline says "Directory not empty" on rm -rf

A process is still writing files. See [Process Recovery](#process-recovery).

### Symptom: cells/d_i looks wrong after launch

Check `run_meta.txt` and the WarpX log:

```bash
ssh <GPU_HOST> 'grep "d_i=" ~/laser-plasma-research/runs/<RUN_NAME>/run.log | head -1; echo ""; cat ~/laser-plasma-research/runs/<RUN_NAME>/run_meta.txt | grep -E "ring_radius|spot_radius|base_density"'
```

Verify:
- `d_i` matches expected value for your density (32.2 µm @ HD, 102 µm @ LD)
- Ring/spot radii match your CLI args

### Symptom: 0-byte MP4 files

Expected. ffmpeg's h264 fails on odd pixel dimensions. GIFs are auto-generated as fallback. Just remove the broken MP4s:
```bash
ssh <GPU_HOST> 'cd ~/laser-plasma-research/runs/<RUN_NAME>/figures && for f in *.mp4; do [ -s "$f" ] || rm -f "$f"; done'
```

### Symptom: Energies look unbelievably high (100+ MeV mean)

Check resolution. cells/d_i = 1.7 (HD at 512² × 9.6 mm) produces severe numerical heating. Solutions:
- Use scaled geometry (R=1000 µm, σ=125 µm) at 512² for HD physics
- Switch to LD density at original geometry (well-resolved by default)
- Use 2048² for HD with original geometry (very expensive)

### Symptom: Pipeline stage 7 (animations) takes forever

Normal — Pass 1 reads all particle dumps (slow), Pass 2 renders animations (slow). For 91 dumps × 100M particles, expect 12-18 min.

### Symptom: Want to verify the simulation is actually doing physics

Check the inline fusion CSV:
```bash
ssh <GPU_HOST> 'head -5 ~/laser-plasma-research/runs/<RUN_NAME>/fusion_rate_power_by_iter.csv; echo "..."; tail -5 ~/laser-plasma-research/runs/<RUN_NAME>/fusion_rate_power_by_iter.csv'
```

You should see fusion rates increasing through the run (not stuck at 0).

### Symptom: SSH session dropped during long run

That's OK — `nohup` keeps the simulation running on cloud regardless of SSH session. Just reconnect and check status.

### Symptom: Need to make CLI scripts/edits but worried about losing them

Always backup before editing:
```bash
cp /Users/brenworth2/LaserFusionResearch/research/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py /Users/brenworth2/LaserFusionResearch/research/laser-plasma-research/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py.backup_$(date +%Y%m%d_%H%M)
```

---

## Hardware & Compute Estimates

### Cloud H100 GPU performance (single GPU, NPPC=400)

| Grid | Particles | Sec/step (avg) | Notes |
|---|---|---|---|
| 256² | 26 M | ~0.4 | Linear in particle count |
| 512² × 9.6 mm domain | 100 M | ~1.5-1.8 | Plain geometry |
| 512² × scaled domain | 100 M | ~3.0 | Scaled geometry has more particles per cell, more sorting |
| 1024² | 419 M | ~6-8 | (estimated, not yet measured) |
| 2048² | 1680 M | ~25-35 | (estimated) |

### Disk requirements

Per particle dump:
- 256² × NPPC=400: ~3.7 GB/dump
- 512² × NPPC=400: ~16-17 GB/dump
- 1024² × NPPC=400: ~64 GB/dump (estimated)

Per field dump:
- ~10× smaller than particle dumps

### Wall time estimates (cloud H100)

| Run | Steps | Sec/step | Wall time |
|---|---|---|---|
| LD 256² ultrafine | 4500 | 0.4 | ~30 min |
| LD 512² ultrafine | 4500 | 1.5 | ~2-3 hr |
| LD 256² long (30 ns) | 200000 | 0.4 | ~22 hr |
| LD 512² long (30 ns) | 200000 | 1.5 | ~83 hr (~3.5 days) |
| HD 512² scaled (1500 steps) | 1500 | 3.0 | ~75 min |

### Mac M2 Max for analysis

Analysis pipeline can be re-run on Mac if needed (uses openpmd-viewer):
- Reading H5 dumps over rsync: very slow (don't recommend)
- Better to do all analysis on cloud, only pull figures/reports to Mac

---

## Appendix: Run naming conventions

We use a consistent naming scheme:

```
p<PAPER>_<DENSITY>_<GRID>_<STEPS>_<TYPE>
```

Examples:
- `p1_hd_512_1500_ultrafine` — Paper 1, HD, 512², 1500 steps, ultrafine cadence
- `p1_ld_512_4500_ultrafine` — Paper 1, LD, 512², 4500 steps, ultrafine
- `p1_ld_512_200k_long` — Paper 1, LD, 512², 200k steps, long-time
- `p1_hd_512_resolved` — Paper 1, HD, 512², scaled geometry (well-resolved)
- `p2_freq_500MHz` — Paper 2, frequency modulation at 500 MHz
- `p3_rod_p11b` — Paper 3, central rod with p-11B fuel

This keeps the runs directory organized and self-documenting.

---

## Maintenance

When the simulation script changes (new arguments, new physics), update this guide:
- Add/modify entries in [CLI Argument Reference](#cli-argument-reference)
- Update relevant launch examples
- Document new run-naming conventions if applicable
- Bump version number at top

---

*Last updated: 2026-05-04*
