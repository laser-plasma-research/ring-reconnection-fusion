#!/usr/bin/env python3
"""
stage_a_paper01_container.py - Paper 1 campaign runner (CONTAINER variant).

Same as stage_a_paper01.py but runs the simulation and analysis chain inside
the ring-reconnection-fusion:v1.2.0 docker container on the GPU host, rather
than against a native conda env.  Use this when the GPU instance does not
have a pre-built `plasma` conda env (e.g., on a freshly provisioned cloud
instance where the container is the only execution environment).

Differences from stage_a_paper01.py:
  - CLOUD_HOST defaults to "gpu-node" (matches current cloud setup)
  - CLOUD_ROOT defaults to "/mnt/vdc/laser-plasma-research" (volume-mounted)
  - CONDA_INIT is "true" (no-op; env is baked into container)
  - launch_sim() wraps mpirun in `docker run --gpus all` with bind mounts
    for /work/simulation, /work/specs, /work/analysis_scripts, and /work/runs
    so that on-disk source overrides the in-image baked copies.  This lets us
    use the reverted-v0.7 detector at /mnt/vdc/laser-plasma-research/
    analysis_scripts/ instead of whatever was baked into v1.2.0 at build time
  - launch_analysis() wraps the analysis chain in `docker run` (no --gpus
    flag; analysis is CPU-only)
  - Sim writes to /mnt/vdc/test_runs/<sub_tag>/ (matches the proven shell
    wrapper convention).  rsync_down pulls from there to the local Mac.
  - MPI invocation uses single-rank with explicit transport selection
    (OMPI_MCA_pml=ob1, BTL=self,vader; UCX_TLS=self,sm) to avoid the
    network-init failures that multi-rank or default-transport MPI hits
    inside the container

Container image expected at:
    ring-reconnection-fusion:v1.2.0  (loaded on GPU host)

Container source path mounts (host -> container):
    /mnt/vdc/laser-plasma-research/simulation       -> /work/simulation
    /mnt/vdc/laser-plasma-research/analysis_scripts -> /work/analysis_scripts
    /mnt/vdc/laser-plasma-research/specs            -> /work/specs (if exists)
    /mnt/vdc/test_runs                              -> /work/runs

JOBS LAYOUT
-----------
Same five entries as stage_a_paper01.py (two NPPC=200 baselines + three
NPPC=100 convergence runs).  Existing local baselines remain untouched
because the orchestrator only re-runs jobs explicitly named via --jobs.

A focused per-paper orchestrator modeled on stage_a_paper3_uuf.py.

    launch sim -> poll log for completion ->
       run_paper1_chain.sh ->
         run_full_analysis.sh ->
         pb11_first_transit_fusion.py --reaction p11b ->
         pb11_zone_analysis_paper3.py ->
    rsync down -> cleanup -> next

PAPER 1 OBJECTIVES (manuscript v0.8.3, May 2026)
-------------------------------------------------
Demonstrate three contributions at one operating point (LD, beta=13.7,
85 T seed, 5e24 m^-3 base density, 5 J coupled energy):

  (1) Geometry of reconnection + physics confirmation
      - 8 X-line locations on the ring of radius R cos(pi/8) = 2.217 mm
      - B-field collapse (58% post-Biermann-ramp at 512^2)
      - rate_Phi in the canonical fast-reconnection band (0.01-0.50)
      - Hall-dominated regime classification (rate_E ~10^4)

  (2) Fusion yield/gain via sigma-weighted first-transit accounting
      - G_FT = 3.63 at 512^2 (headline)
      - G_FT = 2.02 at 256^2 (matched-physics convergence partner)
      - Maxwellian thermal-only baseline G = 0.82, invariant in resolution
      - 4.4x non-thermal enhancement factor at 512^2
      - Thermal-floor bound: 12-46 eV/ion vs 327 keV equilibrium

  (3) Comparison with current approaches
      - Per-joule alpha yield vs published TNSA / PW p-11B campaigns
      - Wall-plug efficiency + repetition rate + footprint argument
      - Mostly literature comparison, no additional simulation needed

JOBS LAYOUT (2 ultraultrafine runs, ~5-5.5 hr unattended)
----------------------------------------------------------
  Run 1   p1_ld_uuf       LD baseline, 256^2, 1500 steps  (~50 min)
  Run 2   p1_ld_uuf_512   LD convergence partner, 512^2, 7700 steps  (~4-4.5 hr)

Run 2 is intentionally LAST so a network blip or compute interruption only
costs the larger run, not the smaller one.

WHAT IS NOT HERE (and why)
--------------------------
  - HD scaling (10x density, R=1mm rescaled geometry): NOT in Paper 1
    v0.8.3 scope. Density and B-seed parameter sweeps are listed as
    Section 5.6 (vi) future work. The previous HD config also crashed
    at step ~555 with an E-field NaN from the tighter Biermann gradients;
    fixing it (substeps/DT) is unrelated to the Paper 1 headline.
  - 3D pilot (256x32x256, ~200 ps): NOT in Paper 1 v0.8.3 scope. Listed
    as Section 5.6 (iv) future work and as a concrete invitation for
    collaboration. Closes Section 5.4 limitation (i) (2D-XZ reduction)
    in a follow-up.
  - 1024^2 convergence checkpoint: NOT in Paper 1 v0.8.3 scope. Listed
    as Section 5.6 (i) future work. The +80% G_FT shift from 256^2 to
    512^2 is flagged in the manuscript; a 1024^2 run is the single
    highest-impact follow-up if compute becomes available before
    submission.
  - no-BC control run: NOT in Paper 1 v0.8.3 scope. Listed as Section
    5.6 highest-priority follow-up. Bounds the 7.5% J_ext/J_th
    boundary-current contribution to G_FT directly.

RETRY POLICY
------------
On failure: clean nuke of remote runs/<sub_tag>/ and local <sub_tag>/, then
retry from scratch. No "resume from where we left off" partial-state
recovery - each attempt starts with a guaranteed-clean data space.

USAGE
-----
    python3 stage_a_paper01.py
    python3 stage_a_paper01.py --jobs p1_ld_uuf                # subset
    python3 stage_a_paper01.py --abort-on-failure              # opt-in
    python3 stage_a_paper01.py --max-retries 5                 # more attempts

Detached:
    nohup python3 -u stage_a_paper01.py > stage_a_paper01.log 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CLOUD_HOST = os.environ.get("GPU_HOST", "gpu-node")
CLOUD_ROOT = "/mnt/vdc/laser-plasma-research"
# CONDA_INIT kept as a no-op so the ssh_run/ssh_launch_detached wrappers can
# still execute lightweight host-side commands (stat, ls, grep, rm) without
# entering the container.  The container is invoked explicitly inside
# launch_sim() and launch_analysis() below.
CONDA_INIT = "true"

# Container settings — used by launch_sim() and launch_analysis()
CONTAINER_IMAGE = "ring-reconnection-fusion:v1.2.1"
HOST_RUNS_DIR   = "/mnt/vdc/test_runs"  # mounted into container as /work/runs
# MPI transport hardening: forces single-node-only paths so OpenMPI's
# network init doesn't try to discover non-existent IB/UCX interfaces from
# inside the container.  Required for the v1.2.0 image.
MPI_ENV = ("OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm")

SIM_SCRIPT = "simulation/pb11_ring_reconnection_v15_pulsed.py"
ANALYSIS_SCRIPT = "analysis_scripts/run_full_analysis.sh"
FIRST_TRANSIT_SCRIPT = "analysis_scripts/pb11_first_transit_fusion.py"
ZONE_ANALYSIS_SCRIPT = "analysis_scripts/pb11_zone_analysis_paper3.py"

LOCAL_RUNS      = Path.home() / "LaserFusionResearch/research/laser-plasma-research/runs"
LOCAL_PAPER_DIR = LOCAL_RUNS / "paper01"

# Geometry / fuel configurations
# Paper 1 LD baseline: 8-spot ring at R=2.4mm, sigma=300um, B=85T, ch_bn at 5e24
LD_GEOMETRY = (
    "--base-fuel ch_bn --base-density 5e24 "
    "--ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --b-seed 85"
)

DIAG_PRODUCTION = "--diag-profile production"

# Per-job definitions
# Tuple format: (sub_tag, freq_hz, dump_period, max_steps, script_id, custom_flags)
JOBS = [
    # ---- LD baseline UUF 256^2 (matched-physics partner) ----------------
    # 1500 steps x dt=0.77 ps = 1.16 ns @ 256^2 with NPPC=200
    # Manuscript v0.8.3 headline G_FT = 2.02
    # Expected runtime: ~50 min (~5 min sim + ~45 min analysis)
    ("p1_ld_uuf", "0", 20, 1500, "v15",
     f"--test {LD_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),

    # ---- LD convergence partner UUF 512^2 (HEADLINE DATA) ---------------
    # SAME physics as p1_ld_uuf but 512^2 grid (4x memory, ~4x compute).
    # Manuscript v0.8.3 headline G_FT = 3.63 (4.4x Maxwellian baseline of 0.82)
    #
    # IMPORTANT: --test flag DROPPED (vs the 256^2 run above) because of CFL.
    # The --test profile uses DT=1e-3, which is ~5x larger than production's
    # DT=1.94e-4. At 512^2 the cell size halves, so the CFL-limited dt also
    # halves - making the --test DT roughly 10x over the stability limit at
    # this grid resolution. Empirically observed failure mode: no E-field
    # NaN crash, but the field solver effectively skips work, B never
    # collapses, no reconnection event develops. Production DT (auto-set
    # when --test absent) plus --substeps 160 (cloud v15 auto-scales to
    # this at NX=512 anyway, but stated explicitly here for visibility)
    # is the correct combination.
    #
    # --nppc 200 matches the 256^2 baseline apples-to-apples - without
    # this override, production-mode defaults to NPPC=400 which at 512^2
    # OOMs the H100 (81 GB total).
    #
    # MAX_STEPS / DUMP_PERIOD: Production DT at 512^2 resolves empirically
    # to ~0.15 ps per step. max_steps=7700 covers 7700 x 0.15 = 1155 ps,
    # matching the 256^2 LD baseline's 1158 ps total window apples-to-apples
    # (1153 ps is the diagnostic-window number used in the manuscript).
    # dump_period=100 -> ~77 dumps total at ~15 ps cadence, matching 256^2
    # LD. Keeping dump count constant means Stage B analysis cost is
    # unchanged from the 256^2 baseline (analysis scales with dump count,
    # not step count).
    #
    # METHODOLOGY NOTE FOR THE MANUSCRIPT: the 256^2/512^2 comparison is NOT
    # a pure grid-only convergence check because of the DT change. It is
    # better characterized as "matched-physics-at-each-resolution" - both
    # runs use the most accurate DT their grid allows. Section 3.7 of
    # v0.8.3 reports the resulting behavior: Maxwellian baseline and
    # lobe-field equilibrium agree to 0.2% across resolutions, while
    # G_FT shifts +80% (2.02 -> 3.63) consistent with better resolution
    # of the suprathermal tail. A 1024^2 checkpoint to establish the
    # asymptote is listed as Section 5.6 (i) future work.
    #
    # Expected runtime: ~4-4.5 hours (sim ~64 min at ~120 steps/min + ~3 hr analysis)
    ("p1_ld_uuf_512", "0", 100, 7700, "v15",
     f"{LD_GEOMETRY} "
     f"--nx 512 --nz 512 --nppc 200 --substeps 160 "
     f"{DIAG_PRODUCTION}"),

    # ========================================================================
    # NPPC=100 CONVERGENCE SERIES (256² / 512² / 1024²)
    # ========================================================================
    # Establishes the grid-convergence asymptote noted as Section 5.6 (i)
    # future work in v0.8.3 of the manuscript.  All three runs at NPPC=100 so
    # grid convergence is tested at fixed NPPC; existing NPPC=200 baselines
    # (p1_ld_uuf, p1_ld_uuf_512 above) supply the apples-to-apples reference
    # for the published G_FT=2.02 / 3.63 values.
    #
    # 2x2 convergence matrix plus 256² anchor:
    #   256² @ NPPC=200 (existing, G_FT=2.02 published)
    #   256² @ NPPC=100 (NEW: NPPC sensitivity at fixed grid)
    #   512² @ NPPC=200 (existing, G_FT=3.63 published)
    #   512² @ NPPC=100 (NEW: NPPC sensitivity + grid convergence anchor)
    #   1024² @ NPPC=100 (NEW: convergence asymptote)
    #
    # All three new runs use the SAME DT scaling, substeps, max_steps, and
    # ~15 ps dump cadence as their NPPC=200 counterparts.  Only NPPC changes.
    # Matched ~15 ps cadence at every resolution is essential: the earlier
    # convergence attempt used coarser cadence (30 ps at 512², 60 ps at
    # 1024²) which undersampled the reconnection collapse window and
    # produced artifactual t_cut values.
    #
    # Disk budget at NPPC=100 (peak during run on /mnt/vdc/test_runs/):
    #   256² ~ 300 GB
    #   512² ~ 1.2 TB
    #   1024² ~ 5 TB     ← drives the 8 TB volume requirement

    # ---- 256² @ NPPC=100 -------------------------------------------------
    # Same recipe as p1_ld_uuf (above): --test flag, max_steps=1500,
    # dump_period=20, ~1.16 ns total at ~15 ps cadence.  Only NPPC changes.
    # Expected runtime: ~50 min
    ("p1_ld_uuf_n100", "0", 20, 1500, "v15",
     f"--test {LD_GEOMETRY} "
     f"--nppc 100 "
     f"{DIAG_PRODUCTION}"),

    # ---- 512² @ NPPC=100 -------------------------------------------------
    # Same recipe as p1_ld_uuf_512 (above): no --test, substeps=160,
    # max_steps=7700, dump_period=100, ~1.16 ns total at ~15 ps cadence.
    # Only NPPC changes (200 -> 100).
    # Expected runtime: ~3.5-4 hours
    ("p1_ld_uuf_512_n100", "0", 100, 7700, "v15",
     f"{LD_GEOMETRY} "
     f"--nx 512 --nz 512 --nppc 100 --substeps 160 "
     f"{DIAG_PRODUCTION}"),

    # ---- 1024² @ NPPC=100 (convergence asymptote) ------------------------
    # NPPC=100 (not 200) because 1024² @ NPPC=200 OOMs the 81 GB H100.
    # substeps=320 maintains CFL at 1024² (cell size halves vs 512², so
    # dt must halve; substeps doubles to compensate).  Production DT at
    # 1024² resolves empirically to ~0.075 ps/step.
    #
    # max_steps=15400 covers 15400 x 0.075 = 1155 ps, matching the 256²
    # and 512² windows apples-to-apples.  dump_period=200 -> ~77 dumps at
    # ~15 ps cadence, again matching the other resolutions.
    #
    # DISK: ~5 TB peak required at 77 dumps x 1024² x NPPC=100 (each
    # particle dump ~65 GB).  This is why an 8 TB /mnt/vdc volume is
    # required for this run; the previous 4 TB volume was insufficient.
    #
    # Expected runtime: ~10-14 hours (sim ~5 hr at ~50 steps/min + ~7 hr analysis)
    ("p1_ld_uuf_1024_n100", "0", 200, 15400, "v15",
     f"{LD_GEOMETRY} "
     f"--nx 1024 --nz 1024 --nppc 100 --substeps 320 "
     f"{DIAG_PRODUCTION}"),
]

# ============================================================================
# Sensitivity sweep (symmetry-perturbation referee response)
# ============================================================================
# Separate stage (--stage sensitivity). 12 runs at the SAME 256^2 LD operating
# point as p1_ld_uuf, each identical except for a per-spot --spot-spec
# perturbation that relaxes the imposed 8-fold symmetry along three axes
# (energy imbalance / position jitter / beams on-off). Answers Bonasera's
# "are you assuming some symmetry?" ahead of submission.
#
# Requirements on the cloud before launching this stage:
#   - patched simulation/pb11_ring_reconnection_v15_pulsed.py (adds --spot-spec)
#   - specs/<name>.json  for every non-baseline run, at ~/laser-plasma-research/specs/
#
# s01_baseline carries NO --spot-spec -> byte-identical to p1_ld_uuf. It is run
# FRESH here (not reused) because the centroid metric needs its ~30 ps particle
# dump as the n=0 reference, and p1_ld_uuf's particles were long since cleaned.
# Particles are KEPT on the cloud for all 12 (KEEP_PARTICLES), so the centroid
# can be computed cloud-side across the whole set. ~50 min/run, ~10 hr total.
_SENS_BASE = f"--test {LD_GEOMETRY} {DIAG_PRODUCTION}"

# (sub_tag, spec_path_or_None) — sub_tags MUST match the spec filenames and the
# names in run_manifest.csv that analyze_sensitivity.py reads.
_SENS_SPECS = [
    ("s01_baseline",          None),
    ("s00_baseline_specform", "specs/s00_baseline_specform.json"),
    ("s_energy_05pct",        "specs/s_energy_05pct.json"),
    ("s_energy_10pct",        "specs/s_energy_10pct.json"),
    ("s_energy_20pct",        "specs/s_energy_20pct.json"),
    ("s_jitter_050um",        "specs/s_jitter_050um.json"),
    ("s_jitter_150um",        "specs/s_jitter_150um.json"),
    ("s_jitter_300um",        "specs/s_jitter_300um.json"),
    ("s_drop1_4plus3",        "specs/s_drop1_4plus3.json"),
    ("s_drop2_adjacent",      "specs/s_drop2_adjacent.json"),
    ("s_drop2_opposite",      "specs/s_drop2_opposite.json"),
    ("s_combined_realistic",  "specs/s_combined_realistic.json"),
]

# Same tuple format as JOBS: (sub_tag, freq_hz, dump_period, max_steps, script_id, custom_flags)
SENSITIVITY_JOBS = [
    (tag, "0", 20, 1500, "v15",
     _SENS_BASE if spec is None else f"{_SENS_BASE} --spot-spec {spec}")
    for tag, spec in _SENS_SPECS
]

POLL_INTERVAL_SEC = 60
SIM_TIMEOUT_HOURS = 6
ANALYSIS_TIMEOUT_HOURS = 4
# The 512^2 case legitimately needs >600s on Stage B at 4x the 256^2
# particle count; 1800s is the empirically validated headroom.
NO_PROGRESS_TIMEOUT_SEC = 1800

DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SEC = [30, 120, 300]

TRANSIENT_RCS = {124, 255}

# Set True by --stage sensitivity. When True, clean_pre_attempt() will NOT
# reclaim OTHER runs' particle dumps, so the whole sensitivity set's particles
# persist on the cloud (4 TB drive) for cross-run centroid analysis afterward.
KEEP_PARTICLES = False


# ============================================================================
# Helpers
# ============================================================================

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str, prefix: str = "  ") -> None:
    print(f"[{ts()}] {prefix}{msg}", flush=True)

def banner(msg: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {ts()} {msg}\n{line}", flush=True)


def ssh_run(remote_cmd: str, timeout: int = 60, check: bool = True):
    full = f"ssh -n {CLOUD_HOST} '{CONDA_INIT} && cd {CLOUD_ROOT} && {remote_cmd}'"
    try:
        r = subprocess.run(
            full, shell=True, capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
        if check and r.returncode != 0:
            log(f"SSH command failed (rc={r.returncode}): {remote_cmd[:80]}", "X ")
            log(f"  stderr: {r.stderr.strip()[:300]}", "  ")
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        log(f"SSH command timed out: {remote_cmd[:80]}", "! ")
        return 124, "", "timeout"


def ssh_launch_detached(launch_cmd: str, ssh_timeout: int = 15) -> None:
    full = f"ssh -n {CLOUD_HOST} '{CONDA_INIT} && cd {CLOUD_ROOT} && {launch_cmd}'"
    try:
        subprocess.run(
            full, shell=True, capture_output=True, text=True,
            timeout=ssh_timeout, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        pass


# ============================================================================
# Pre-attempt cleanup
# ============================================================================

def kill_orphaned_gpu_processes() -> None:
    log("Sweeping for orphaned GPU processes", "  ")
    cmd = (
        'PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader '
        '2>/dev/null); for pid in $PIDS; do kill -9 $pid 2>/dev/null; done; '
        'pkill -9 -f pb11_ring_reconnection 2>/dev/null; '
        'pkill -9 -f prterun 2>/dev/null; '
        'pkill -9 -f mpirun 2>/dev/null; sleep 3; '
        'nvidia-smi --query-gpu=memory.free --format=csv'
    )
    rc, out, err = ssh_run(cmd, timeout=30, check=False)
    if rc == 0 and out:
        free_mb_line = out.splitlines()[-1] if out.splitlines() else "?"
        log(f"GPU after sweep: {free_mb_line}", "  ")


def clean_other_runs_particles(current_sub_tag: str) -> None:
    """Free cloud disk by removing particles/fields from OTHER completed runs.

    Policy: cleanup_cloud() no longer runs after each successful job (see
    run_one_job_attempt). That keeps the most recently completed run's
    particle dumps on disk so they can be re-analyzed (e.g. for the f(E)
    histogram extraction stage that consumes particle dumps directly).
    However, particles from runs PRIOR to the most recent one are no
    longer useful and must be cleared before the next sim launches, or
    disk fills up during multi-job campaigns.

    This function clears particles+fields from every run dir EXCEPT
    current_sub_tag. The current_sub_tag's directory is wiped wholesale
    by clean_pre_attempt() itself, so we exclude it here to avoid
    fighting that logic.
    """
    cmd = (
        f"for d in {HOST_RUNS_DIR}/*/; do "
        f"  name=$(basename \"$d\"); "
        f"  if [ \"$name\" != \"{current_sub_tag}\" ]; then "
        f"    if [ -d \"$d/particles\" ] || [ -d \"$d/fields\" ]; then "
        f"      rm -rf \"$d/particles\" \"$d/fields\" 2>/dev/null; "
        f"      echo \"freed: $name\"; "
        f"    fi; "
        f"  fi; "
        f"done"
    )
    rc, out, _ = ssh_run(cmd, timeout=180, check=False)
    if rc == 0 and out.strip():
        n_freed = len([ln for ln in out.splitlines() if ln.startswith("freed:")])
        if n_freed > 0:
            log(f"Reclaimed disk from {n_freed} prior run(s)", "  ")
            rc2, disk, _ = ssh_run(f"df -h {HOST_RUNS_DIR} | tail -1",
                                    timeout=15, check=False)
            log(f"Cloud disk after reclaim: {disk}", "  ")


def clean_pre_attempt(sub_tag: str) -> None:
    run_dir = f"{HOST_RUNS_DIR}/{sub_tag}"
    log(f"Pre-attempt cleanup: wiping {run_dir} on cloud + local", "  ")
    ssh_run(f"rm -rf {run_dir}", timeout=120, check=False)

    # Free disk from OTHER completed runs (keeps the just-finished one's
    # particles available until the NEXT job starts, enabling post-hoc
    # analysis like f(E) histogram extraction).
    if KEEP_PARTICLES:
        log("KEEP_PARTICLES set - preserving other runs' particle dumps for "
            "cross-run analysis", "  ")
    else:
        clean_other_runs_particles(sub_tag)

    local_dir = LOCAL_PAPER_DIR / sub_tag
    if local_dir.exists():
        import shutil
        try:
            shutil.rmtree(local_dir)
            log(f"Removed local {local_dir}", "  ")
        except OSError as e:
            log(f"Could not remove {local_dir}: {e}", "! ")

    kill_orphaned_gpu_processes()


# ============================================================================
# Per-job pipeline stages
# ============================================================================

def launch_sim(sub_tag: str, dump_period: int, max_steps: int,
               custom_flags: str) -> bool:
    # Sim runs inside the docker container.  Outputs land in
    # /mnt/vdc/test_runs/<sub_tag>/ on the host (mounted as /work/runs/<sub_tag>
    # inside the container).  Logs land in CLOUD_ROOT (host) via shell redir.
    run_dir_host = f"{HOST_RUNS_DIR}/{sub_tag}"
    run_dir_container = f"/work/runs/{sub_tag}"
    sim_log = f"{sub_tag}.log"

    flags = (f"{custom_flags} --max-steps {max_steps} "
             f"--dump-period {dump_period} --outdir {run_dir_container}")

    # Build the docker invocation.  Bind mounts override the baked-in image
    # source so the reverted-v0.7 analysis scripts and current sim driver
    # at /mnt/vdc/laser-plasma-research/ are used instead of whatever was
    # captured at v1.2.0 build time.  --gpus all gives the container the
    # single H100; mpirun uses -n 1 because oversubscribing a single GPU
    # causes context contention (one rank per GPU is the right model).
    docker_cmd = (
        f"docker run --rm --gpus all "
        f"--user $(id -u):$(id -g) "
        f"-v {HOST_RUNS_DIR}:/work/runs "
        f"-v {CLOUD_ROOT}/simulation:/work/simulation "
        f"-v {CLOUD_ROOT}/analysis_scripts:/work/analysis_scripts "
        f"-v {CLOUD_ROOT}/specs:/work/specs "
        f"-e CUPY_CACHE_DIR=/work/runs/.cupy_cache "
        f"-e HOME=/work/runs "
        f"{CONTAINER_IMAGE} "
        f"bash -c \"cd /work && export {MPI_ENV} && "
        f"mpirun --allow-run-as-root -n 1 "
        f"python -u /work/{SIM_SCRIPT} {flags}\""
    )

    launch_cmd = (
        f"mkdir -p {run_dir_host} && "
        f"setsid nohup {docker_cmd} "
        f"< /dev/null > {sim_log} 2>&1 & disown; echo started"
    )

    log(f"Launching sim (container): max_steps={max_steps}, dump_period={dump_period}", "-> ")
    ssh_launch_detached(launch_cmd, ssh_timeout=30)

    log("Verifying via log-file growth (up to 3 attempts)...", "  ")
    log_size_int = 0
    for attempt_i in range(3):
        time.sleep(15 if attempt_i == 0 else 10)
        rc, log_size, _ = ssh_run(f"stat -c %s {sim_log} 2>/dev/null || echo 0",
                                  timeout=30, check=False)
        if rc == 0 and log_size.isdigit() and int(log_size) > 0:
            log_size_int = int(log_size)
            break
    if log_size_int == 0:
        log("Sim log empty or missing after 3 verify attempts - launch failed", "X ")
        return False
    log(f"Sim log is {log_size_int} bytes - sim is running", "OK ")
    return True


def poll_sim(sub_tag: str, max_steps: int) -> bool:
    sim_log = f"{sub_tag}.log"
    deadline = time.time() + SIM_TIMEOUT_HOURS * 3600
    last_step = -1
    last_progress_at = time.time()
    last_log_size = -1

    log(f"Polling sim - expected ~{SIM_TIMEOUT_HOURS}h max, max_steps={max_steps}", "  ")

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)

        rc, step_line, _ = ssh_run(
            f'grep "^STEP " {sim_log} 2>/dev/null | grep "ends" | tail -1',
            timeout=20, check=False,
        )
        rc_size, log_size, _ = ssh_run(
            f"stat -c %s {sim_log} 2>/dev/null || echo 0",
            timeout=15, check=False,
        )
        log_size_int = int(log_size) if log_size.isdigit() else 0

        cur_step = -1
        if step_line.startswith("STEP "):
            try:
                cur_step = int(step_line.split()[1])
            except (ValueError, IndexError):
                pass

        if cur_step >= 0:
            log(f"step {cur_step}/{max_steps} ({100*cur_step//max_steps}%)", "  ")

        if cur_step >= max_steps:
            time.sleep(10)
            log(f"Sim reached max_steps={max_steps} - completion confirmed", "OK ")
            return True

        if log_size_int == last_log_size and cur_step == last_step:
            stuck_for = time.time() - last_progress_at
            if stuck_for > NO_PROGRESS_TIMEOUT_SEC:
                rc, tail, _ = ssh_run(f"tail -5 {sim_log}", timeout=15, check=False)
                log(f"No progress for {int(stuck_for)}s - sim may have crashed", "X ")
                log(f"Log tail:\n{tail}", "  ")
                return False
        else:
            last_progress_at = time.time()
            last_log_size = log_size_int
            last_step = cur_step

    log(f"Sim exceeded {SIM_TIMEOUT_HOURS}h timeout", "X ")
    return False


def launch_analysis(sub_tag: str) -> bool:
    # Analysis runs inside the docker container, mounted the same way as the
    # sim but WITHOUT --gpus (analysis is CPU-only).  run_paper1_chain.sh
    # reads particle dumps from /work/runs/<sub_tag>/ and writes summary
    # CSVs/text/figures alongside them.
    run_dir_container = f"/work/runs/{sub_tag}"
    analysis_log = f"{sub_tag}_analysis.log"

    docker_cmd = (
        f"docker run --rm "
        f"--user $(id -u):$(id -g) "
        f"-v {HOST_RUNS_DIR}:/work/runs "
        f"-v {CLOUD_ROOT}/simulation:/work/simulation "
        f"-v {CLOUD_ROOT}/analysis_scripts:/work/analysis_scripts "
        f"-v {CLOUD_ROOT}/specs:/work/specs "
        f"-e CUPY_CACHE_DIR=/work/runs/.cupy_cache "
        f"-e HOME=/work/runs "
        f"{CONTAINER_IMAGE} "
        f"bash -c \"cd /work && "
        f"bash analysis_scripts/run_paper1_chain.sh {run_dir_container}\""
    )

    launch_cmd = (
        f"setsid nohup {docker_cmd} "
        f"< /dev/null > {analysis_log} 2>&1 & disown; echo started"
    )

    log(f"Launching analysis chain (container, detached) for {sub_tag}", "-> ")
    ssh_launch_detached(launch_cmd, ssh_timeout=30)

    log("Verifying via analysis log growth (up to 3 attempts)...", "  ")
    log_size_int = 0
    for attempt_i in range(3):
        time.sleep(10 if attempt_i == 0 else 8)
        rc, log_size, _ = ssh_run(f"stat -c %s {analysis_log} 2>/dev/null || echo 0",
                                  timeout=30, check=False)
        if rc == 0 and log_size.isdigit() and int(log_size) > 0:
            log_size_int = int(log_size)
            break
    if log_size_int == 0:
        log("Analysis log empty or missing after 3 verify attempts - launch failed", "X ")
        return False
    log(f"Analysis log is {log_size_int} bytes - analysis is running", "OK ")
    return True


def poll_analysis(sub_tag: str) -> bool:
    analysis_log = f"{sub_tag}_analysis.log"
    deadline = time.time() + ANALYSIS_TIMEOUT_HOURS * 3600
    last_log_size = -1
    last_progress_at = time.time()

    log(f"Polling analysis chain - expected ~30 min, max {ANALYSIS_TIMEOUT_HOURS}h", "  ")

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)

        rc, marker, _ = ssh_run(
            f'grep -c "ALL P1 ANALYSIS STAGES COMPLETE" {analysis_log} 2>/dev/null || echo 0',
            timeout=15, check=False,
        )
        if marker.isdigit() and int(marker) > 0:
            rc2, summary, _ = ssh_run(
                f'tail -10 {analysis_log}',
                timeout=15, check=False,
            )
            failed_stages = []
            for line in summary.splitlines():
                if "rc=" in line and "rc=0" not in line:
                    failed_stages.append(line.strip())
            if failed_stages:
                log("Analysis chain finished but some stages failed:", "! ")
                for fs in failed_stages:
                    log(f"  {fs}", "  ")
                return False
            log("Analysis chain complete - all stages OK", "OK ")
            return True

        rc_size, log_size, _ = ssh_run(
            f"stat -c %s {analysis_log} 2>/dev/null || echo 0",
            timeout=15, check=False,
        )
        log_size_int = int(log_size) if log_size.isdigit() else 0

        rc, stage_line, _ = ssh_run(
            f'grep -E "=== STAGE [A-D]" {analysis_log} 2>/dev/null | tail -1',
            timeout=15, check=False,
        )
        if stage_line and ("STAGE " in stage_line):
            try:
                stage = stage_line.split("STAGE ")[1].split(":")[0].strip()
                log(f"analysis stage {stage}", "  ")
            except (ValueError, IndexError):
                pass

        if log_size_int > last_log_size:
            last_log_size = log_size_int
            last_progress_at = time.time()
        else:
            stuck_for = time.time() - last_progress_at
            if stuck_for > NO_PROGRESS_TIMEOUT_SEC:
                rc, tail, _ = ssh_run(f"tail -10 {analysis_log}", timeout=15, check=False)
                log(f"Analysis stuck for {int(stuck_for)}s", "X ")
                log(f"Log tail:\n{tail}", "  ")
                return False

    log(f"Analysis exceeded {ANALYSIS_TIMEOUT_HOURS}h timeout", "X ")
    return False


def run_analysis(sub_tag: str) -> bool:
    if not launch_analysis(sub_tag):
        return False
    return poll_analysis(sub_tag)


def rsync_down(sub_tag: str) -> bool:
    # Container variant: sim outputs land at HOST_RUNS_DIR/<sub_tag>/, not
    # CLOUD_ROOT/runs/.  Pull from there.
    local_dir = LOCAL_PAPER_DIR / sub_tag
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"rsync -avh "
        f"--include='*/' "
        f"--include='*.png' --include='*.mp4' --include='*.txt' "
        f"--include='*.csv' --include='*.npz' "
        f"--exclude='*' "
        f"{CLOUD_HOST}:{HOST_RUNS_DIR}/{sub_tag}/ {local_dir}/"
    )

    transient_rcs = {23, 24, 30, 35}

    for attempt in range(1, 4):
        log(f"Rsyncing results to {local_dir} (attempt {attempt}/3)", "-> ")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=900)
        except subprocess.TimeoutExpired:
            log(f"Rsync timed out after 900s on attempt {attempt}/3", "! ")
            if attempt < 3:
                wait = 30 * attempt
                log(f"Waiting {wait}s before retry...", "  ")
                time.sleep(wait)
                continue
            return False
        except Exception as e:
            log(f"Rsync raised unexpected exception: {type(e).__name__}: {e}", "X ")
            return False

        if r.returncode == 0:
            log("Rsync complete", "OK ")
            return True
        if r.returncode in transient_rcs and attempt < 3:
            log(f"Rsync rc={r.returncode} (transient) on attempt {attempt}/3: "
                f"{r.stderr.strip()[:200]}", "! ")
            wait = 30 * attempt
            log(f"Waiting {wait}s before retry...", "  ")
            time.sleep(wait)
            continue
        log(f"Rsync failed (rc={r.returncode}): {r.stderr.strip()[:300]}", "X ")
        return False

    return False


def cleanup_cloud(sub_tag: str) -> None:
    run_dir = f"{HOST_RUNS_DIR}/{sub_tag}"
    log("Cleaning cloud particles + fields", "-> ")
    ssh_run(f"rm -rf {run_dir}/particles {run_dir}/fields", timeout=60, check=False)
    rc, disk, _ = ssh_run(f"df -h {HOST_RUNS_DIR} | tail -1",
                          timeout=15, check=False)
    log(f"Cloud disk after cleanup: {disk}", "  ")


# ============================================================================
# Per-job pipeline orchestration with retries
# ============================================================================

def run_one_job_attempt(sub_tag: str, dump_period: int, max_steps: int,
                        custom_flags: str):
    if not launch_sim(sub_tag, dump_period, max_steps, custom_flags):
        return False, "sim_launch"
    if not poll_sim(sub_tag, max_steps):
        return False, "sim_poll"
    if not run_analysis(sub_tag):
        return False, "analysis"
    if not rsync_down(sub_tag):
        return False, "rsync"
    # Post-run cleanup_cloud() intentionally NOT called here. The
    # just-completed run's particles+fields stay on cloud until the NEXT
    # job's clean_pre_attempt() reclaims their disk space. This preserves
    # particle dumps for ad-hoc f(E) histogram re-analysis without forcing
    # a re-simulation. cleanup_cloud() is still defined above and can be
    # invoked manually if disk pressure demands it.
    return True, "complete"


def run_one_job(sub_tag: str, freq_hz: str, dump_period: int,
                max_steps: int, max_retries: int,
                script_id: str, custom_flags: str) -> bool:
    banner(f"Job: {sub_tag} (script={script_id}, "
           f"dump_period={dump_period}, max_steps={max_steps})")

    if script_id != "v15":
        log(f"WARNING: Paper 1 expects script_id=v15, got {script_id}", "! ")

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            backoff = RETRY_BACKOFF_SEC[min(attempt - 2, len(RETRY_BACKOFF_SEC) - 1)]
            log(f"Retry {attempt}/{max_retries} for {sub_tag} after {backoff}s backoff", "+ ")
            time.sleep(backoff)

        clean_pre_attempt(sub_tag)

        try:
            ok, stage = run_one_job_attempt(sub_tag, dump_period, max_steps, custom_flags)
        except Exception as e:
            import traceback
            log(f"UNHANDLED EXCEPTION in attempt {attempt}/{max_retries}: "
                f"{type(e).__name__}: {e}", "X ")
            log("Traceback:\n" + traceback.format_exc(), "  ")
            ok, stage = False, "unhandled_exception"

        if ok:
            if attempt > 1:
                log(f"Job {sub_tag}: COMPLETE on attempt {attempt}", "OK ")
            else:
                log(f"Job {sub_tag}: COMPLETE", "OK ")
            return True
        log(f"Attempt {attempt}/{max_retries} failed at stage: {stage}", "X ")

    log(f"Job {sub_tag}: ALL {max_retries} ATTEMPTS FAILED", "X ")
    return False


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", help="Comma-separated subset of sub_tags to run")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Retries per job before declaring failure "
                        f"(default {DEFAULT_MAX_RETRIES})")
    p.add_argument("--abort-on-failure", action="store_true",
                   help="Abort remaining jobs after first failure (legacy behavior). "
                        "Default is to continue past failures.")
    p.add_argument("--stage", choices=["paper1", "sensitivity"], default="paper1",
                   help="Which campaign to run. 'paper1' (default) runs the two "
                        "baseline/convergence jobs. 'sensitivity' runs the 12 "
                        "symmetry-perturbation jobs and KEEPS all particle dumps "
                        "on the cloud for cross-run centroid analysis.")
    args = p.parse_args()

    global KEEP_PARTICLES
    if args.stage == "sensitivity":
        KEEP_PARTICLES = True
        all_jobs = SENSITIVITY_JOBS
        stage_name = "Paper 1 SENSITIVITY runner"
    else:
        all_jobs = JOBS
        stage_name = "Paper 1 runner"

    jobs = all_jobs
    if args.jobs:
        wanted = set(args.jobs.split(","))
        jobs = [j for j in all_jobs if j[0] in wanted]
        if not jobs:
            print(f"No matching jobs in {args.jobs}. "
                  f"Available: {[j[0] for j in all_jobs]}")
            return 1

    banner(f"{stage_name} - {len(jobs)} jobs: {', '.join(j[0] for j in jobs)} "
           f"(max_retries={args.max_retries})")

    succeeded, failed = [], []
    for job in jobs:
        sub_tag, freq, dump_period, max_steps, script_id, custom_flags = job
        ok = run_one_job(sub_tag, freq, dump_period, max_steps, args.max_retries,
                         script_id=script_id, custom_flags=custom_flags)
        (succeeded if ok else failed).append(sub_tag)
        if not ok and args.abort_on_failure:
            log(f"--abort-on-failure set; aborting remaining jobs after {sub_tag}", "X ")
            break

    banner("FINAL SUMMARY")
    print(f"  Succeeded: {len(succeeded)} - {succeeded}", flush=True)
    print(f"  Failed:    {len(failed)} - {failed}", flush=True)
    if not failed:
        if args.stage == "sensitivity":
            print(f"\n  All sensitivity runs complete. Particles are KEPT on the "
                  f"cloud for cross-run analysis.", flush=True)
            print(f"  Next: run the cross-run analyzer ON THE CLOUD (that is where "
                  f"the particle dumps live):", flush=True)
            print(f"    ssh {CLOUD_HOST} '{CONDA_INIT} && cd {CLOUD_ROOT} && \\", flush=True)
            print(f"      python analysis_scripts/analyze_sensitivity.py \\", flush=True)
            print(f"        --runs-root runs --manifest specs/run_manifest.csv \\", flush=True)
            print(f"        --baseline s01_baseline --out runs/sensitivity_results.csv'", flush=True)
            print(f"  Then bring just the small CSV down:", flush=True)
            print(f"    scp {CLOUD_HOST}:{CLOUD_ROOT}/runs/sensitivity_results.csv .", flush=True)
            print(f"  Sanity check first: s01_baseline must reproduce G_FT 2.02 / "
                  f"Maxwellian 0.82 / |B|min 21.46 T.\n", flush=True)
        else:
            print(f"\n  Next step: review the post_analysis_report.txt and "
                  f"first_transit_summary_p11b.txt files in each run directory.", flush=True)
            print(f"  Local runs: {LOCAL_PAPER_DIR}", flush=True)
            print(f"  These provide the data for Paper 1's three contributions:", flush=True)
            print(f"    (1) Geometry of reconnection (zones_paper3.txt, "
                  f"reconnection_summary.txt)", flush=True)
            print(f"    (2) Fusion yield/gain via sigma-weighted first-transit "
                  f"(first_transit_summary_p11b.txt)", flush=True)
            print(f"    (3) Comparison with current approaches (literature, "
                  f"no extra data needed)\n", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
