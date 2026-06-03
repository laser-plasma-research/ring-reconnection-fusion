#!/usr/bin/env python3
"""
stage_a_multilaser_container.py - Commodity-laser demonstration campaign
runner (CONTAINER variant).

Same laser->simulation mapping as stage_a_multilaser.py, but the execution
backbone is ported from stage_a_paper01_container.py: the sim and analysis
chain run inside the ring-reconnection-fusion docker container on the GPU
host, not against a native conda env.

      launch sim (docker --gpus all) -> poll log for completion ->
        run_paper1_chain.sh (docker, CPU) ->
          run_full_analysis.sh ->
          pb11_first_transit_fusion.py --reaction p11b ->
          pb11_zone_analysis_paper3.py ->
      rsync down from /mnt/vdc/test_runs -> cleanup -> next

Container execution model (identical to stage_a_paper01_container.py):
  - CLOUD_HOST  = os.environ.get("GPU_HOST", "gpu-node")
  - CLOUD_ROOT  = "/mnt/vdc/laser-plasma-research"  (volume-mounted source)
  - CONDA_INIT  = "true" (no-op; env baked into the image)
  - launch_sim() wraps mpirun in `docker run --gpus all` with bind mounts for
    /work/simulation, /work/analysis_scripts, /work/specs, /work/runs, so the
    on-disk source overrides the in-image baked copies.
  - launch_analysis() wraps the analysis chain in `docker run` (no --gpus).
  - Sim writes to /mnt/vdc/test_runs/<sub_tag>/; rsync_down pulls from there.
  - MPI uses single rank + explicit transport selection (OMPI_MCA_pml=ob1,
    BTL=self,vader; UCX_TLS=self,sm) to avoid container network-init failures.

PURPOSE / LASER MAPPING / LASER-ENERGY NOTE
-------------------------------------------
Identical to stage_a_multilaser.py - see that file's docstring. In brief:
the PIC run's laser fingerprint is the Biermann seed --b-seed (derived from
each laser's on-target intensity, anchored so the 5 J / 2 ns baseline = 85 T);
laser energy is only the gain denominator. PASS_LASER_ENERGY_FLAG defaults
OFF so the baseline stays byte-identical to p1_ld_uuf and runs report
"gain vs 5 J" (true gain = reported * 5/E, printed per job).

USAGE
-----
    python3 stage_a_multilaser_container.py --preview      # derivation table, no launch
    python3 stage_a_multilaser_container.py                # run all laser jobs
    python3 stage_a_multilaser_container.py --jobs ml_fiber_mopa,ml_ps_fast
    python3 stage_a_multilaser_container.py --abort-on-failure
    python3 stage_a_multilaser_container.py --max-retries 5

Detached:
    nohup python3 -u stage_a_multilaser_container.py > stage_a_multilaser_container.log 2>&1 &
"""

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration  (container execution model; from stage_a_paper01_container.py)
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
# MPI transport hardening: forces single-node-only paths so OpenMPI's network
# init doesn't try to discover non-existent IB/UCX interfaces inside the
# container.  Required for the v1.2.x image.
MPI_ENV = ("OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_TLS=self,sm")

SIM_SCRIPT = "simulation/pb11_ring_reconnection_v15_pulsed.py"
ANALYSIS_SCRIPT = "analysis_scripts/run_full_analysis.sh"
FIRST_TRANSIT_SCRIPT = "analysis_scripts/pb11_first_transit_fusion.py"
ZONE_ANALYSIS_SCRIPT = "analysis_scripts/pb11_zone_analysis_paper3.py"

LOCAL_RUNS      = Path.home() / "LaserFusionResearch/research/laser-plasma-research/runs"
LOCAL_PAPER_DIR = LOCAL_RUNS / "multilaser"

DIAG_PRODUCTION = "--diag-profile production"

# ----------------------------------------------------------------------------
# Fixed on-grid geometry (the validated Paper-1 baseline; NOT varied per laser)
# ----------------------------------------------------------------------------
SIM_FUEL          = "ch_bn"
SIM_DENSITY       = "5e24"
SIM_RING_RADIUS_UM = 2400
SIM_SPOT_SIGMA_UM  = 300     # on-grid Gaussian sigma (~13 cells at 256^2); NOT the focal spot
SIM_N_SPOTS        = 8

def geometry_flags(b_seed_t: float) -> str:
    """Baseline grid geometry with a per-laser Biermann seed.

    geometry_flags(85.0) reproduces LD_GEOMETRY exactly:
      --base-fuel ch_bn --base-density 5e24 --ring-radius-um 2400
      --spot-radius-um 300 --n-spots 8 --b-seed 85
    """
    return (
        f"--base-fuel {SIM_FUEL} --base-density {SIM_DENSITY} "
        f"--ring-radius-um {SIM_RING_RADIUS_UM} --spot-radius-um {SIM_SPOT_SIGMA_UM} "
        f"--n-spots {SIM_N_SPOTS} --b-seed {b_seed_t:g}"
    )

# ============================================================================
# Laser -> simulation mapping (all knobs tunable; see module docstring)
# ============================================================================

# Biermann seed = B_BASELINE_T * (I/I0)^INTENSITY_EXP, clamped to [floor, cap].
B_BASELINE_T  = 85.0     # anchor: the validated Paper-1 LD operating point
INTENSITY_EXP = 0.5      # B ~ sqrt(I); conservative middle of patent's 1e12->1 T .. 1e16->1000 T
B_FLOOR_T     = 40.0     # below this the plasma is too high-beta for reconnection to organize
B_CAP_T       = 1000.0   # patent operative ceiling

# Baseline beta for the indicative beta ~ B^-2 estimate printed in the preview.
BETA_AT_BASELINE = 13.7

# dt is set by the sim as time_step = DT * 2*pi / omega_ci, with omega_ci ~ B,
# so dt ~ 1/B. From v15 (DT=1.94e-4 production, 1e-3 in --test) and the proton
# cyclotron frequency:
#   production: dt[fs] = 12726 / B[T]   (verified: 85 T -> 149.7 fs, matches run_meta)
#   --test    : dt[fs] = 65600 / B[T]   (verified: 85 T -> 0.772 ps, matches baseline)
DT_FS_TIMES_B_PROD = 12726.0
DT_FS_TIMES_B_TEST = 65600.0

# --test profile is only CFL-safe at low seed fields. The baseline (85 T) uses
# it; the 100 T sweep already needed production dt. Use --test at/below this.
B_TEST_MAX_T = 90.0

# Test-mode run shape (matches p1_ld_uuf: ~1.16 ns at ~15 ps cadence, ~50 min).
TEST_MAX_STEPS   = 1500
TEST_DUMP_PERIOD = 20

# Production demo runs: aim for a short but reconnection-resolving window.
# (A publication-grade run would extend to the full ~1 ns window; this is a
#  "does it run / does the mechanism appear" demonstration, per the brief.)
PROD_TARGET_WINDOW_PS = 120.0
PROD_MAX_STEPS_CAP    = 8000      # bound demo runtime
PROD_TARGET_DUMPS     = 70        # ~ matches the baseline dump count for equal analysis cost

# See LASER ENERGY NOTE. Default OFF for a byte-identical baseline; gain
# rescale (5/E) is printed instead.
PASS_LASER_ENERGY_FLAG = False
LASER_ENERGY_FLAG_NAME = "--laser-energy-j"
GAIN_DENOMINATOR_DEFAULT_J = 5.0


@dataclass
class LaserSystem:
    """A representative commodity / near-commodity driver.

    Specs are representative of the named product class (mirroring the
    provisional's Section XV envelope and named drivers), not a vendor
    datasheet. Edit freely - everything downstream is derived.
    """
    key: str                 # sub_tag used for run dir + --jobs selection
    label: str               # human description
    energy_j: float          # total pulse energy on target (gain denominator)
    pulse_s: float           # pulse duration
    wavelength_nm: float
    focal_diam_um: float     # PHYSICAL focal spot diameter (feeds intensity only)
    rep_rate_hz: float


# Representative commodity-laser set. Ordered cheap-first / expensive-last so a
# late interruption costs the longest (highest-B, most-steps) run, not the rest
# - same philosophy as putting the 512^2 run last in the Paper-1 runner.
LASERS = [
    # Validated Paper-1 anchor. By construction this derives to b_seed=85 T and
    # (with PASS_LASER_ENERGY_FLAG off) is byte-identical to p1_ld_uuf, so it
    # reproduces the published G_FT=2.02 / Maxwellian 0.82 as a sanity check.
    LaserSystem("ml_baseline",   "Paper-1 reference (hybrid Yb:fiber+Nd:YAG, anchor)",
                energy_j=5.0,  pulse_s=2.0e-9,  wavelength_nm=1030, focal_diam_um=75, rep_rate_hz=10),

    # Low-power commodity: industrial Yb fiber MOPA, sub-J, tightly focused.
    LaserSystem("ml_fiber_mopa", "Yb fiber MOPA, sub-J (IPG/SPI-class)",
                energy_j=0.2,  pulse_s=2.0e-9,  wavelength_nm=1064, focal_diam_um=25, rep_rate_hz=1000),

    # Workhorse J-class Q-switched Nd:YAG (Ekspla NL303-class). Longer pulse +
    # looser focus => lower PEAK intensity than the fiber MOPA above despite 5x
    # the energy - clean illustration that the Biermann seed tracks intensity,
    # not pulse energy.
    LaserSystem("ml_ndyag_1j",   "Q-switched Nd:YAG, ~1 J (Ekspla NL303-class)",
                energy_j=1.0,  pulse_s=5.0e-9,  wavelength_nm=1064, focal_diam_um=40, rep_rate_hz=10),

    # Upper-commodity hybrid driver (the Paper-1 baseline scaled to 10 J).
    LaserSystem("ml_hybrid_10j", "Hybrid Yb:fiber+Nd:YAG, ~10 J",
                energy_j=10.0, pulse_s=2.0e-9,  wavelength_nm=1064, focal_diam_um=75, rep_rate_hz=10),

    # High-end commodity FAST PULSE: ps-regime Nd:YAG. Short pulse -> high peak
    # intensity -> high Biermann seed, hence production dt + short demo window.
    LaserSystem("ml_ps_fast",    "ps-regime Nd:YAG, ~1 J / 50 ps (fast pulse)",
                energy_j=1.0,  pulse_s=50.0e-12, wavelength_nm=1064, focal_diam_um=50, rep_rate_hz=100),
]


def peak_intensity_wcm2(ls: LaserSystem) -> float:
    """On-target per-spot peak intensity in W/cm^2."""
    r_cm = (ls.focal_diam_um * 1e-4) / 2.0
    area_cm2 = math.pi * r_cm * r_cm
    power_per_spot_w = (ls.energy_j / SIM_N_SPOTS) / ls.pulse_s
    return power_per_spot_w / area_cm2


# Baseline intensity used to anchor the Biermann map (computed once).
_I0 = peak_intensity_wcm2(LASERS[0])


@dataclass
class DerivedRun:
    ls: LaserSystem
    intensity: float
    intensity_ratio: float
    b_seed_t: float
    b_clamped: str           # "", "floor", or "cap"
    beta_est: float
    mode: str                # "test" or "production"
    max_steps: int
    dump_period: int
    dt_fs: float
    window_ps: float
    gain_rescale: float      # multiply reported "gain vs 5 J" by this for true gain
    custom_flags: str


def derive(ls: LaserSystem) -> DerivedRun:
    I = peak_intensity_wcm2(ls)
    ratio = I / _I0
    b_raw = B_BASELINE_T * (ratio ** INTENSITY_EXP)
    b = b_raw
    clamp = ""
    if b < B_FLOOR_T:
        b, clamp = B_FLOOR_T, "floor"
    elif b > B_CAP_T:
        b, clamp = B_CAP_T, "cap"
    b = round(b)

    beta_est = BETA_AT_BASELINE * (B_BASELINE_T / b) ** 2

    energy_flag = (f"{LASER_ENERGY_FLAG_NAME} {ls.energy_j:g} "
                   if PASS_LASER_ENERGY_FLAG else "")

    if b <= B_TEST_MAX_T:
        mode = "test"
        dt_fs = DT_FS_TIMES_B_TEST / b   # --test DT=1e-3
        max_steps = TEST_MAX_STEPS
        dump_period = TEST_DUMP_PERIOD
        # --test auto-sets 256^2 / NPPC=200. Window follows from the per-B dt:
        # lower B -> larger dt -> LONGER physical window for the same 1500 steps
        # (runtime is per-step, so wall-clock is ~unchanged across these).
        window_ps = max_steps * dt_fs / 1000.0
        flags = f"--test {geometry_flags(b)} {energy_flag}{DIAG_PRODUCTION}".strip()
    else:
        mode = "production"
        dt_fs = DT_FS_TIMES_B_PROD / b   # production DT=1.94e-4
        steps = math.ceil(PROD_TARGET_WINDOW_PS * 1000.0 / dt_fs)
        max_steps = min(steps, PROD_MAX_STEPS_CAP)
        window_ps = max_steps * dt_fs / 1000.0
        dump_period = max(1, max_steps // PROD_TARGET_DUMPS)
        # Production default grid is 384^2 / NPPC=400; pin to the 256^2 / NPPC=200
        # baseline so these stay apples-to-apples with the test-mode runs.
        flags = (f"{geometry_flags(b)} --nx 256 --nz 256 --nppc 200 "
                 f"{energy_flag}{DIAG_PRODUCTION}").strip()

    gain_rescale = GAIN_DENOMINATOR_DEFAULT_J / ls.energy_j

    return DerivedRun(
        ls=ls, intensity=I, intensity_ratio=ratio, b_seed_t=float(b),
        b_clamped=clamp, beta_est=beta_est, mode=mode, max_steps=max_steps,
        dump_period=dump_period, dt_fs=dt_fs, window_ps=window_ps,
        gain_rescale=gain_rescale, custom_flags=flags,
    )


# Build the JOBS table in the exact tuple format the runner expects:
#   (sub_tag, freq_hz, dump_period, max_steps, script_id, custom_flags)
# freq_hz is "0" (static) for every run - no rotational drive (Paper 2 closed).
DERIVED = {ls.key: derive(ls) for ls in LASERS}
JOBS = [
    (ls.key, "0", DERIVED[ls.key].dump_period, DERIVED[ls.key].max_steps,
     "v15", DERIVED[ls.key].custom_flags)
    for ls in LASERS
]


def print_preview() -> None:
    banner("Multi-laser campaign - derivation preview (CONTAINER variant)")
    print(f"  Image: {CONTAINER_IMAGE}  host: {CLOUD_HOST}  runs: {HOST_RUNS_DIR}", flush=True)
    print(f"  Anchor I0 (baseline) = {_I0:.3e} W/cm^2  ->  b_seed = {B_BASELINE_T:g} T", flush=True)
    print(f"  Biermann map: B = {B_BASELINE_T:g} * (I/I0)^{INTENSITY_EXP:g}, "
          f"clamped [{B_FLOOR_T:g}, {B_CAP_T:g}] T", flush=True)
    print(f"  Laser energy flag: {'PASSED (' + LASER_ENERGY_FLAG_NAME + ')' if PASS_LASER_ENERGY_FLAG else 'OFF -> reported gain is vs 5 J; multiply by gain x'}",
          flush=True)
    print("", flush=True)
    hdr = (f"  {'key':<14} {'E(J)':>6} {'tau':>7} {'Phi_um':>6} "
           f"{'I[W/cm2]':>10} {'I/I0':>8} {'b_seed':>7} {'beta~':>7} "
           f"{'mode':>5} {'steps':>6} {'win_ps':>7} {'gain x':>7}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for ls in LASERS:
        d = DERIVED[ls.key]
        tau = (f"{ls.pulse_s*1e9:g}ns" if ls.pulse_s >= 1e-9 else f"{ls.pulse_s*1e12:g}ps")
        bcl = "" if not d.b_clamped else f"({d.b_clamped})"
        print(f"  {ls.key:<14} {ls.energy_j:>6g} {tau:>7} {ls.focal_diam_um:>6g} "
              f"{d.intensity:>10.2e} {d.intensity_ratio:>8.3g} "
              f"{d.b_seed_t:>5g}{bcl:<2} {d.beta_est:>7.1f} "
              f"{d.mode:>5} {d.max_steps:>6} {d.window_ps:>7.0f} "
              f"{d.gain_rescale:>7.2g}", flush=True)
    print("", flush=True)
    for ls in LASERS:
        d = DERIVED[ls.key]
        print(f"  {ls.key}: {ls.label}", flush=True)
        print(f"      flags: {d.custom_flags}", flush=True)
        if d.b_clamped == "floor":
            print(f"      NOTE: intensity below baseline; Biermann seed FLOORED to "
                  f"{B_FLOOR_T:g} T to stay in the reconnection-organizing regime. "
                  f"Physically this driver would need tighter focus or pulse-stacking "
                  f"to reach this seed.", flush=True)
        if d.b_clamped == "cap":
            print(f"      NOTE: intensity very high; Biermann seed CAPPED at {B_CAP_T:g} T "
                  f"(patent operative ceiling).", flush=True)
    print("", flush=True)


POLL_INTERVAL_SEC = 60
SIM_TIMEOUT_HOURS = 6
ANALYSIS_TIMEOUT_HOURS = 4
NO_PROGRESS_TIMEOUT_SEC = 1800

DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SEC = [30, 120, 300]

TRANSIENT_RCS = {124, 255}

KEEP_PARTICLES = False


# ============================================================================
# Helpers  (verbatim from stage_a_paper01_container.py)
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
# Pre-attempt cleanup  (container paths; from stage_a_paper01_container.py)
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
# Per-job pipeline stages  (container-wrapped; from stage_a_paper01_container.py)
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

    # Bind mounts override the baked-in image source so the on-disk analysis
    # scripts and sim driver at /mnt/vdc/laser-plasma-research/ are used instead
    # of whatever was captured at image-build time.  --gpus all gives the
    # container the single H100; mpirun uses -n 1 because oversubscribing a
    # single GPU causes context contention (one rank per GPU is the right model).
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
    return True, "complete"


def run_one_job(sub_tag: str, freq_hz: str, dump_period: int,
                max_steps: int, max_retries: int,
                script_id: str, custom_flags: str) -> bool:
    banner(f"Job: {sub_tag} (script={script_id}, "
           f"dump_period={dump_period}, max_steps={max_steps})")

    if script_id != "v15":
        log(f"WARNING: multilaser campaign expects script_id=v15, got {script_id}", "! ")

    d = DERIVED.get(sub_tag)
    if d is not None:
        log(f"{d.ls.label}", "  ")
        log(f"E={d.ls.energy_j:g} J  I={d.intensity:.2e} W/cm^2  -> b_seed={d.b_seed_t:g} T "
            f"(beta~{d.beta_est:.0f}, {d.mode}); reported-gain x{d.gain_rescale:.3g} = true gain",
            "  ")

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
    p.add_argument("--preview", action="store_true",
                   help="Print the laser->sim derivation table and exit (no launch).")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Retries per job before declaring failure "
                        f"(default {DEFAULT_MAX_RETRIES})")
    p.add_argument("--abort-on-failure", action="store_true",
                   help="Abort remaining jobs after first failure. "
                        "Default is to continue past failures.")
    args = p.parse_args()

    if args.preview:
        print_preview()
        return 0

    jobs = JOBS
    if args.jobs:
        wanted = set(args.jobs.split(","))
        jobs = [j for j in JOBS if j[0] in wanted]
        if not jobs:
            print(f"No matching jobs in {args.jobs}. "
                  f"Available: {[j[0] for j in JOBS]}")
            return 1

    banner(f"Multi-laser runner (CONTAINER) - {len(jobs)} jobs: "
           f"{', '.join(j[0] for j in jobs)} (max_retries={args.max_retries})")
    print_preview()

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
    if succeeded and not PASS_LASER_ENERGY_FLAG:
        print(f"\n  Reminder: runs report 'gain vs 5 J'. For the true per-laser gain,", flush=True)
        print(f"  multiply each run's reported gain by (5 / E_laser):", flush=True)
        for tag in succeeded:
            d = DERIVED.get(tag)
            if d is not None:
                print(f"    {tag:<14} x {d.gain_rescale:.3g}  (E={d.ls.energy_j:g} J)", flush=True)
    if not failed:
        print(f"\n  Next step: review post_analysis_report.txt and "
              f"first_transit_summary_p11b.txt in each run directory.", flush=True)
        print(f"  Local runs: {LOCAL_PAPER_DIR}", flush=True)
        print(f"  The cross-laser comparison (b_seed, fast-fraction, G_FT vs driver class)", flush=True)
        print(f"  is the deliverable for the 'works across commodity lasers' argument.\n", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
