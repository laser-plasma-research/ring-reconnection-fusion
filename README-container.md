# Ring Reconnection Fusion — Runtime Container

**Version:** 1.2.2
**Author:** James B. Worth (ORCID: 0009-0005-5000-9497)
**License:** Apache-2.0
**Platform:** Linux x86_64 with NVIDIA H100 (CUDA 12.9, compute capability 9.0)

## What this is

A reproducible Docker container image with a complete, GPU-enabled scientific
Python environment for running 2D hybrid-PIC simulations of laser-driven
p-11B aneutronic fusion via 8-spot ring magnetic reconnection. This is the
captured production environment used to generate results for Worth (2026),
*Reconnection-driven non-thermal proton acceleration to p-11B fusion energies at joule-class laser energy: a ring-geometry hybrid-PIC study*
(in preparation).

The container packages:

- **WarpX 26.4** with CUDA support (built from BLAST-WarpX source on the
  development branch, sm_90 / H100 Hopper target)
- **pywarpx 26.4** Python bindings, CUDA-linked (libcufft 11.4.1, libcurand
  10.3.10)
- **AMReX 26.4** hybrid-PIC framework
- **Python 3.11.15** with numpy 2.4.3, scipy 1.16.3, matplotlib 3.10.9,
  mpi4py 4.1.1, h5py 3.16.0, openPMD-viewer, yt readers, and analysis stack
- **CUDA Toolkit 12.9.1** runtime libraries (cudart, cufft, curand, cublas,
  cusparse, cusolver, nvjitlink)
- **OpenMPI 5.0.10** with UCX
- Source code for the simulation, analysis, and orchestration pipeline

The container was built via `conda-pack` of the live production environment,
guaranteeing byte-level reproducibility of the env that produced the
published data.

## Is the container required?

The container is **required** for full GPU reproduction of the published
results. Without it, you would need to build WarpX from source with CUDA
support on your GPU node — a multi-hour exercise with many version pitfalls,
which is precisely why this container exists.

The container is **not required** for local CPU-only experimentation with
small test cases, code reading, or analysis-pipeline development. For those
use cases, you can install pywarpx natively via pip (`pip install pywarpx`)
on any Linux/macOS machine; it will run sub-grid 2D smoke tests in a few
minutes without GPU. Note that local CPU runs cannot match the resolution
or particle count of the published GPU campaign — they exist only to verify
that the orchestrator and analysis scripts work in your environment.

| Use case                            | GPU required | Container required |
|-------------------------------------|:------------:|:------------------:|
| Read code, understand methodology   | No           | No                 |
| Run analysis on shipped data only   | No           | No                 |
| Local CPU smoke tests (2D, low NPPC) | No           | No (use `pip install pywarpx`) |
| Per-figure 256² runs (Tier 1 UUF)   | Yes          | Yes                |
| Convergence 512² runs (Tier 3)      | Yes          | Yes                |
| Full Paper 1 / Paper 3 campaign     | Yes          | Yes                |

## File integrity

- **Image archive:** `ring-reconnection-fusion-v1.2.2.tar.gz`
- **Size:** 7.6 GB
- **SHA-256:** `d3074eb2f12bb0218f6de05ace504f33c0e5755dbad67873681075f6f221af2e`

Verify the archive before loading:

```bash
sha256sum -c SHA256SUMS
```

## Prerequisites

For GPU reproduction:

- Linux GPU host with Docker 24+
- NVIDIA Container Toolkit (`nvidia-container-toolkit`) — required for
  `--gpus all` to work
- An NVIDIA GPU with compute capability 9.0 (H100) for the published
  configuration; lower compute capabilities may work for some 2D cases but
  are not validated against the published results
- ~30 GB free disk for the loaded image
- Per-run additional disk depending on resolution (256² ~ 200 MB,
  512² ~ 2 TB peak with dump-period=33)

For the orchestrator (local workstation):

- macOS or Linux
- Python 3.10+ (for the orchestrator scripts)
- `rsync`, `ssh`
- Passwordless SSH access to the GPU host (`ssh gpu-node` or your
  equivalent alias resolves and authenticates without password prompt)

## Load the image

```bash
gunzip -c ring-reconnection-fusion-v1.2.2.tar.gz | docker load
```

Verify:

```bash
docker images ring-reconnection-fusion:v1.2.2
```

Smoke-test the load:

```bash
docker run --rm --gpus all ring-reconnection-fusion:v1.2.2 python -c "
import pywarpx
from pywarpx import picmi
print('pywarpx imports:', picmi.__file__)
"
```

Expected output:

```
pywarpx imports: /opt/conda/envs/plasma/lib/python3.11/site-packages/pywarpx/picmi.py
```

## Workflow architecture

The production workflow is split between two machines:

```
   ┌──────────────────────┐                ┌─────────────────────────┐
   │  Local workstation   │     ssh / scp  │  Remote GPU host        │
   │  (Mac or Linux)      │ ──────────────▶│  (single H100 GPU node) │
   │                      │                │                         │
   │  stage_a_paperX.py   │                │  ring-reconnection-fusion:v1.2.2
   │    │                 │                │    (loaded container)   │
   │    ├─ launch_sim ────┼──ssh──────────▶│    docker run --gpus all│
   │    ├─ poll_sim       │   (per job)    │      ↓                  │
   │    ├─ launch_analysis│                │    WarpX + pywarpx      │
   │    ├─ poll_analysis  │                │      ↓                  │
   │    ├─ rsync_down ◀───┼──results back──│    runs/<job>/diags/    │
   │    └─ cleanup_cloud  │                │      ↓                  │
   │                      │                │    runs/<job>/*.csv     │
   │  runs/paper0X/...    │                │    (rsync'd to local)   │
   └──────────────────────┘                └─────────────────────────┘
```

The **orchestrator** (`stage_a_paper01.py` or `stage_a_paper3.py`) runs on
the local workstation. For each job in the campaign it:

1. Cleans local + cloud state (idempotent restarts)
2. ssh's a `setsid nohup mpirun -n 8 python -u pb11_ring_reconnection_v15_pulsed.py ...`
   command to the GPU host
3. Polls the simulation's log file via repeated `ssh stat -c %s` calls
4. When the sim finishes, ssh's a `run_paperX_chain.sh` analysis launch
5. Polls the analysis log
6. `rsync` pulls the post-analysis output (CSVs, summaries, figures) back
   to local under `runs/paperXX/<sub_tag>/`
7. Moves on to the next job, with up to 3 retries on transient failures

The orchestrator is what drives the campaign; the GPU host just executes
individual WarpX + analysis invocations on demand.

### Where the container fits in

The orchestrator's ssh-dispatched commands can either:

- Run the simulation **natively** against a matching conda env installed on
  the GPU host, or
- Run the simulation **inside this container** via
  `docker run --gpus all -v <workdir>:/work ring-reconnection-fusion:v1.2.2 mpirun ...`

The container path is the recommended one for reproduction because the
native install requires building WarpX from source with CUDA and matching
the exact dependency versions; the container has that work already done
and frozen.

## How to use — single-shot mode

For one-off testing or running an isolated simulation case, invoke the
container directly. The example below reproduces the **Paper 1 low-density
256² baseline** (`p1_ld_uuf`, manuscript headline G_FT = 2.02): an 8-spot
ring at R = 2.4 mm, σ = 300 µm, B = 85 T, with ch_bn fuel at 5×10²⁴ m⁻³:

```bash
docker run --rm --gpus all \
    -v $PWD/runs:/work/runs \
    ring-reconnection-fusion:v1.2.2 \
    mpirun -n 8 python -u /work/simulation/pb11_ring_reconnection_v15_pulsed.py \
        --test \
        --base-fuel ch_bn --base-density 5e24 \
        --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --b-seed 85 \
        --diag-profile production \
        --max-steps 1500 --dump-period 20 \
        --outdir /work/runs/p1_ld_uuf
```

A successful run produces particle dumps in `runs/p1_ld_uuf/diags/` and
diagnostic CSVs at the run root. Wall-clock time on an H100: ~5 minutes for
1500 steps at 256².

For the **512² convergence partner** (`p1_ld_uuf_512`, manuscript headline
G_FT = 3.63), drop `--test` (CFL stability at the finer grid requires the
smaller production time-step) and add the resolution / particle / substep
overrides:

```bash
docker run --rm --gpus all \
    -v $PWD/runs:/work/runs \
    ring-reconnection-fusion:v1.2.2 \
    mpirun -n 8 python -u /work/simulation/pb11_ring_reconnection_v15_pulsed.py \
        --base-fuel ch_bn --base-density 5e24 \
        --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --b-seed 85 \
        --nx 512 --nz 512 --nppc 200 --substeps 160 \
        --diag-profile production \
        --max-steps 7700 --dump-period 100 \
        --outdir /work/runs/p1_ld_uuf_512
```

Wall-clock time on an H100: ~64 minutes for 7700 steps at 512². The 512²
case uses `--nppc 200` to match the 256² baseline apples-to-apples — without
that override, production mode defaults to NPPC=400, which OOMs an 80 GB
H100 at 512².

## How to use — local CPU mode (no container, no GPU)

For code reading, methodology checks, or verifying the orchestrator and
analysis scripts work in your environment, you can run the simulation script
directly on any Linux/macOS machine without this container and without a GPU.
Install pywarpx natively:

```bash
pip install pywarpx==26.4 numpy scipy matplotlib h5py mpi4py
```

Then run a sub-grid 2D smoke test of the Paper 1 baseline geometry. Use a
much smaller grid and step count so it completes on CPU in a few minutes:

```bash
python -u simulation/pb11_ring_reconnection_v15_pulsed.py \
    --test \
    --base-fuel ch_bn --base-density 5e24 \
    --ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --b-seed 85 \
    --nx 128 --nz 128 --nppc 50 \
    --diag-profile production \
    --max-steps 200 --dump-period 20 \
    --outdir runs/p1_ld_smoke_cpu
```

This is **not** a scientific reproduction — the reduced grid (128² vs 256²),
particle count (NPPC=50 vs 200), and step count (200 vs 1500) mean the
reconnection event is under-resolved and the gain figures will not match the
manuscript. It exists only to confirm the simulation pipeline runs end-to-end
in your environment. For results matching the paper, use the GPU container.

## How to use — production campaign mode (orchestrator)

To reproduce the full Paper 1 figure set as a managed campaign, use the
`stage_a_paper01.py` orchestrator from the source repository (also archived
on Zenodo; see "Related identifiers"). It runs on your **local workstation**
and dispatches each job to the GPU host, polling progress and pulling
results back automatically.

### One-time setup

1. Edit the config constants at the top of `stage_a_paper01.py` to match
   your environment:

   ```python
   CLOUD_HOST = "gpu-node"            # your SSH alias / hostname for the GPU host
   CLOUD_ROOT = "~/laser-plasma-research"
   CONDA_INIT = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma"
   ```

   `CONDA_INIT` assumes the GPU host has a native conda env named `plasma`.
   If you'd rather run the simulation inside this container on the GPU host,
   set `CONDA_INIT` to a harmless no-op (e.g. `CONDA_INIT = "true"`) and
   change the simulation launch line inside `launch_sim()` from

   ```python
   f"setsid nohup mpirun -n 8 python -u {SIM_SCRIPT} {flags} ..."
   ```

   to a container invocation:

   ```python
   f"setsid nohup docker run --rm --gpus all "
   f"-v {CLOUD_ROOT}:/work ring-reconnection-fusion:v1.2.2 "
   f"mpirun -n 8 python -u /work/{SIM_SCRIPT} {flags} ..."
   ```

   Both paths produce identical results; the container path frees you from
   building WarpX from source on the GPU host.

2. Push the simulation and analysis source trees to the GPU host, into
   `~/laser-plasma-research/` (included in the source repository archived
   alongside this container).

3. If using the container, verify it loaded on the GPU host:

   ```bash
   ssh gpu-node 'docker images ring-reconnection-fusion:v1.2.2'
   ```

### Run the Paper 1 campaign

From the local workstation:

```bash
python3 -u stage_a_paper01.py --tier 1
```

This runs both Paper 1 jobs in sequence — the 256² baseline (`p1_ld_uuf`)
and the 512² convergence partner (`p1_ld_uuf_512`) — using the exact
parameters embedded in the orchestrator's `JOBS` list. For each job the
orchestrator launches the sim, polls until completion, runs the analysis
chain, and rsyncs results to `runs/paper01/<sub_tag>/` on your local
workstation.

To detach so it survives terminal closure:

```bash
nohup python3 -u stage_a_paper01.py --tier 1 \
    > stage_a_paper01_tier1.log 2>&1 &
```

Total wall-clock time for both Paper 1 jobs on a single H100: ~5–5.5 hours
(the 512² convergence run dominates at ~4–4.5 hours including analysis).

When the orchestrator finishes, the post-analysis output for each job is in
`runs/paper01/<sub_tag>/`. Figures are generated by analysis scripts in
`analysis_scripts/` against this local data; see the source repository's
`README.md` for figure-by-figure generation.

## Reproduction notes

To reproduce the full results published in Worth (2026):

1. **Code** (Zenodo DOI, see "Related identifiers" on this record):
   simulation, analysis, and orchestrator source code
2. **Container** (this Zenodo record): execution environment
3. **Data** (Zenodo DOI, see "Related identifiers"): the post-analysis CSV
   and summary outputs from the published campaign, for cross-validation
   against your reproduction run

For Paper 1 specifically, the two headline jobs are `p1_ld_uuf` (256²,
G_FT = 2.02) and `p1_ld_uuf_512` (512², G_FT = 3.63). Run them either via
the single-shot `docker run` commands above or via
`stage_a_paper01.py --tier 1`. Compare your local output under
`runs/paper01/<sub_tag>/` against the published data deposit — in particular
`fusion_rate_power_by_iter.csv` (gain accounting) and the reconnection-rate
and zone-analysis CSVs.

Load this container on a GPU host, push the source tree, run the
orchestrator from your local workstation, and compare your local output
against the published data deposit.

## Build provenance

The container was built on a cloud H100 80 GB node, May 2026,
using the following pipeline:

1. Native production conda env (`plasma`) constructed from `environment.yml`
   with cuda-toolkit 12.9.1, CMake 4.2, gcc 12.4, Ninja
2. WarpX cloned from `https://github.com/BLAST-WarpX/warpx.git` (development
   branch, May 2026), built with `WarpX_DIMS=2`, `WarpX_COMPUTE=CUDA`,
   `AMReX_CUDA_ARCH=90`, `WarpX_PYTHON=ON`
3. `conda-pack` captures the live env including the CUDA-built pywarpx
4. Container layer applies the captured env to ubuntu:22.04, runs
   `conda-unpack` to fix paths, copies the simulation source tree

The aggressive use of `conda-pack` avoids the conda solve / source-rebuild
chain that would otherwise be required at container build time, guaranteeing
that the env in the container is bit-for-bit identical to the one that
produced the published data.

For the Dockerfile, build script, and environment YAML used to construct
this image, see the source repository linked in the Zenodo metadata.

## Citing this container

If you use this container in published work, please cite both the container
DOI (this Zenodo deposit) and the manuscript:

> Worth, J. B. (2026). *Reconnection-driven non-thermal proton acceleration to p-11B fusion energies at joule-class laser energy: a ring-geometry hybrid-PIC study*. (In preparation.)

A `CITATION.cff` file is included in the source repository for automated
citation tooling.

## Contact

James B. Worth — brenworth@gmail.com — ORCID 0009-0005-5000-9497
