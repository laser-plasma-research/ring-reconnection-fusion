#!/usr/bin/env python3
"""
cloud_run_all.py — Cloud orchestrator for ring-reconnection plasma research.

Reads a program config YAML and for each sub_job:
  1. Checks the cloud environment is clean (no stale processes, enough disk)
  2. Launches simulation on cloud via SSH
  3. Monitors until completion, verifying AMReX finalized + dump counts
  4. Retries up to max_retries if unsuccessful
  5. Runs the 8-stage analysis pipeline (run_full_analysis.sh)
  6. Verifies all required analysis products are present and non-empty
  7. Downloads analysis products to Mac (rsync)
  8. Repeats from clean-check for the next job

Usage:
    python cloud_run_all.py --config program_config_paper2.yaml --paper A2
    python cloud_run_all.py --config program_config_paper2.yaml --paper A2 --dry-run
    python cloud_run_all.py --config program_config_paper2.yaml --paper A2 --skip-sim
    python cloud_run_all.py --config program_config_paper2.yaml --paper A2 --jobs p2_ghz_28
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml


# ============================================================================
# Constants
# ============================================================================

SSH_HOST          = os.environ.get("GPU_HOST", "gpu-node")
CLOUD_ROOT        = "~/laser-plasma-research"
CONDA_ENV         = "plasma"
CONDA_INIT        = f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate {CONDA_ENV}"
SIM_SCRIPT        = "simulation/pb11_ring_reconnection_v12_fuel_center_outer.py"
ANALYSIS_SCRIPT   = "analysis_scripts/run_full_analysis.sh"
CHECK_SCRIPT      = "check_analysis_status.sh"

# ============================================================================
# Required outputs — organized by pipeline stage
# Each entry: (relative_path, min_size_bytes, stage_name)
# ============================================================================

REQUIRED_OUTPUTS = {
    # Pre-existing from simulation (must exist before analysis runs)
    "simulation": [
        ("run_meta.txt",                                    100,    "sim"),
        ("fusion_rate_power_by_iter.csv",                   500,    "sim"),
        ("reconnection_rate_by_iter.csv",                   500,    "sim"),
        ("step_time_index.csv",                             500,    "sim"),
    ],
    # Stage 1: visualize_all.py
    "stage1_reports": [],
    "stage1_figures": [
        ("figures/composite_snapshots.png",                100_000, "visualize_all"),
        ("figures/composite_evolution.png",                 50_000, "visualize_all"),
        ("figures/b_evolution.mp4",                         50_000, "visualize_all"),
        ("figures/individual/snap_01_b_magnitude.png",      20_000, "visualize_all"),
        ("figures/individual/snap_02_b_streamlines_jy.png", 20_000, "visualize_all"),
        ("figures/individual/snap_03_plasma_density.png",   20_000, "visualize_all"),
        ("figures/individual/snap_04_particle_ke.png",      20_000, "visualize_all"),
        ("figures/individual/snap_05_energy_spectrum.png",  20_000, "visualize_all"),
        ("figures/individual/snap_06_zone_e95_bar.png",     20_000, "visualize_all"),
        ("figures/individual/evol_07_fast_frac_fusion_power.png", 20_000, "visualize_all"),
        ("figures/individual/evol_08_cumulative_yield.png", 20_000, "visualize_all"),
        ("figures/individual/evol_09_zone_e95.png",         20_000, "visualize_all"),
        ("figures/individual/evol_10_centre_spot_ratio.png",20_000, "visualize_all"),
        ("figures/individual/evol_11_zone_counts.png",      20_000, "visualize_all"),
        ("figures/individual/evol_12_b_evolution.png",      20_000, "visualize_all"),
    ],
    "stage1_cache": [
        ("figures/cache/zone_timeseries.npz",                5_000, "visualize_all"),
        ("figures/cache/snapshot_data.npz",                100_000, "visualize_all"),
        ("figures/cache/field_timeseries.npz",               2_000, "visualize_all"),
    ],
    # Stage 2: pb11_consolidate_reports.py
    "stage2": [
        ("post_analysis_report.txt",                         3_000, "consolidate_reports"),
        ("zone_report.txt",                                  1_500, "consolidate_reports"),
    ],
    # Stage 3: pb11_phase_analysis.py
    "stage3": [
        ("phase_analysis_report.txt",                        2_000, "phase_analysis"),
    ],
    # Stage 4: pb11_reconnection_rate_offline.py
    "stage4": [
        ("reconnection_rate_offline.csv",                   10_000, "reconnection_rate_offline"),
        ("reconnection_summary.txt",                         1_500, "reconnection_rate_offline"),
    ],
    # Stage 5: pb11_fusion_diagnostics.py
    "stage5_csv": [
        ("fusion_diagnostics.csv",                           2_000, "fusion_diagnostics"),
        ("fusion_accounting_notes.txt",                        500, "fusion_diagnostics"),
    ],
    "stage5_figures": [
        ("figures/fusion_grade_fractions_vs_time.png",      30_000, "fusion_diagnostics"),
        ("figures/spectrum_evolution_heatmap.png",          30_000, "fusion_diagnostics"),
        ("figures/peak_snapshot_fusion_grade.png",         200_000, "fusion_diagnostics"),
        ("figures/energy_distribution_by_zone_at_peak.png", 30_000, "fusion_diagnostics"),
    ],
    # Stage 6: pb11_particle_animations.py
    "stage6": [
        ("figures/particle_density_by_energy.mp4",         500_000, "particle_animations"),
        ("figures/acceleration_tracer.mp4",                 50_000, "particle_animations"),
        ("figures/energy_spectrum_evolution.mp4",           30_000, "particle_animations"),
    ],
    # Stage 7: pb11_particle_trajectories.py
    "stage7": [
        ("figures/trajectory_summary.png",                 200_000, "particle_trajectories"),
        ("figures/trajectory_energy_evolution.png",         50_000, "particle_trajectories"),
        ("figures/particle_trajectories.mp4",              500_000, "particle_trajectories"),
    ],
}

# Stages that must pass for a run to be considered complete
CRITICAL_STAGES = ["simulation", "stage1_figures", "stage2", "stage3", "stage4",
                   "stage5_csv", "stage5_figures", "stage6", "stage7"]
# stage1_cache is informational; needed for re-runs but not for the deliverable

MIN_DISK_FREE_GB  = 300   # abort if less than this is available
POLL_INTERVAL_SEC = 30    # how often to poll simulation progress
MAX_RETRIES       = 2     # max simulation + analysis attempts per job
STATE_FILE        = "cloud_run_state.json"


# ============================================================================
# Progress tracking
# ============================================================================

class OrchestratorState:
    """Tracks current orchestrator state for progress display."""

    PHASES = ["env_check", "simulation", "verify_sim", "analysis",
              "verify_analysis", "download", "cleanup", "complete"]

    def __init__(self, total_jobs: int):
        self.total_jobs       = total_jobs
        self.current_job_idx  = 0   # 1-indexed when active
        self.current_sub_tag  = ""
        self.current_phase    = "env_check"
        self.current_attempt  = 1
        self.start_time       = datetime.now()
        self.completed_jobs   = []
        self.failed_jobs      = []
        # Per-phase sub-progress (e.g. "step 5,000 / 200,000" within simulation)
        self.phase_detail     = ""
        self.phase_pct        = 0.0    # 0–100 within current phase

    def begin_job(self, idx: int, sub_tag: str, attempt: int = 1):
        self.current_job_idx = idx
        self.current_sub_tag = sub_tag
        self.current_attempt = attempt
        self.current_phase   = "env_check"
        self.phase_detail    = ""
        self.phase_pct       = 0.0

    def set_phase(self, phase: str, detail: str = "", pct: float = 0.0):
        self.current_phase = phase
        self.phase_detail  = detail
        self.phase_pct     = pct

    def finish_job(self, success: bool):
        target = self.completed_jobs if success else self.failed_jobs
        if self.current_sub_tag and self.current_sub_tag not in target:
            target.append(self.current_sub_tag)

    @property
    def overall_pct(self) -> float:
        """Overall completion percentage (jobs completed + current job partial)."""
        if self.total_jobs == 0:
            return 0.0
        # Count completed jobs as 1.0, current job as fraction based on phase
        phase_weights = {
            "env_check":         0.02,
            "simulation":        0.70,
            "verify_sim":        0.72,
            "analysis":          0.95,
            "verify_analysis":   0.97,
            "download":          0.99,
            "cleanup":           1.00,
            "complete":          1.00,
        }
        base = phase_weights.get(self.current_phase, 0.0)
        # Fold in within-phase progress (only meaningful for simulation/analysis)
        if self.current_phase == "simulation":
            base = 0.02 + 0.68 * (self.phase_pct / 100.0)
        elif self.current_phase == "analysis":
            base = 0.72 + 0.23 * (self.phase_pct / 100.0)
        n_done = len(self.completed_jobs) + len(self.failed_jobs)
        # Job in progress
        if self.current_job_idx > 0 and self.current_job_idx > n_done:
            n_done_partial = n_done + base
        else:
            n_done_partial = n_done
        return 100.0 * n_done_partial / self.total_jobs

    @property
    def elapsed_str(self) -> str:
        delta = datetime.now() - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}h{m:02d}m"

    @property
    def eta_str(self) -> str:
        pct = self.overall_pct
        if pct < 1:
            return "calculating..."
        elapsed = (datetime.now() - self.start_time).total_seconds()
        total_estimated = elapsed * 100 / pct
        remaining = total_estimated - elapsed
        if remaining < 0:
            return "imminent"
        h, rem = divmod(int(remaining), 3600)
        m, _ = divmod(rem, 60)
        return f"{h:d}h{m:02d}m"

    def banner(self) -> str:
        """Render a compact one-line status banner."""
        bar_width = 30
        filled = int(self.overall_pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        n_done = len(self.completed_jobs)
        n_failed = len(self.failed_jobs)
        return (
            f"[{bar}] {self.overall_pct:5.1f}%  "
            f"job {self.current_job_idx}/{self.total_jobs}: "
            f"{self.current_sub_tag or '—':<24s} "
            f"{self.current_phase:<16s} "
            f"({n_done}✓ {n_failed}✗) "
            f"elapsed {self.elapsed_str} ETA {self.eta_str}"
        )

    def detail_line(self) -> str:
        """Optional detail line for current sub-progress (e.g. step counter)."""
        if not self.phase_detail:
            return ""
        return f"      └─ {self.phase_detail}"


# Global state singleton (set by run_paper)
STATE: OrchestratorState | None = None


def banner():
    """Print the current state banner. Safe to call when STATE is None."""
    if STATE is None:
        return
    print(f"\n  {STATE.banner()}", flush=True)
    detail = STATE.detail_line()
    if detail:
        print(detail, flush=True)



# ============================================================================
# Utilities
# ============================================================================

def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg, level="INFO"):
    marker = {"INFO": "  ", "OK": "✓ ", "WARN": "⚠ ", "ERR": "✗ ", "HEAD": "══"}
    print(f"[{ts()}] {marker.get(level,'  ')}{msg}", flush=True)


def sep(title=""):
    if title:
        print(f"\n{'='*70}\n  {title}\n{'='*70}", flush=True)
    else:
        print(f"\n{'─'*70}", flush=True)


def ssh(cmd: str, capture=True, timeout=30) -> tuple[int, str, str]:
    """Run a command on the cloud host via SSH.

    Uses `ssh -n` (redirect ssh's stdin from /dev/null) AND passes
    stdin=subprocess.DEVNULL to subprocess. Both are required to prevent
    SSH from hanging when the remote command backgrounds a process —
    even with full file-redirection on the backgrounded child, SSH
    waits for the connection's file descriptors to close on the remote
    side, blocking until subprocess timeout.
    """
    full = f"ssh -n {SSH_HOST} '{CONDA_INIT} && cd {CLOUD_ROOT} && {cmd}'"
    result = subprocess.run(
        full, shell=True, capture_output=capture,
        text=True, timeout=timeout, stdin=subprocess.DEVNULL
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def ssh_raw(cmd: str, timeout=30) -> tuple[int, str, str]:
    """Run a raw SSH command without conda init or cd. See ssh() for -n / DEVNULL rationale."""
    full = f"ssh -n {SSH_HOST} '{cmd}'"
    result = subprocess.run(
        full, shell=True, capture_output=True,
        text=True, timeout=timeout, stdin=subprocess.DEVNULL
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def rsync_down(remote_path: str, local_path: str, include_patterns=None) -> bool:
    """Rsync from cloud to Mac."""
    Path(local_path).mkdir(parents=True, exist_ok=True)
    if include_patterns:
        inc = " ".join(f"--include='{p}'" for p in include_patterns)
        cmd = f"rsync -avh {inc} --exclude='*' {SSH_HOST}:'{remote_path}/' '{local_path}/'"
    else:
        cmd = f"rsync -avh {SSH_HOST}:'{remote_path}' '{local_path}/'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


# ============================================================================
# Step 1 — Environment check
# ============================================================================

def check_cloud_environment(run_dir: Optional[str] = None) -> bool:
    """Verify cloud is clean: no stale processes, enough disk space."""
    log("Checking cloud environment...", "HEAD")

    # Check for running simulations.
    # Match only the actual python process (cmdline starts with `python ...`).
    # The orphan `bash -c` wrapper from a hung SSH session contains the script
    # path in its cmdline but is harmless and should NOT block new jobs.
    rc, out, _ = ssh_raw("pgrep -af 'python.*ring_reconnection' | grep -v 'bash -c' || true")
    if out.strip():
        log(f"Stale simulation process found:\n  {out}", "ERR")
        log("Kill it with: ssh gpu-node 'pkill -f ring_reconnection'", "WARN")
        return False
    log("No stale simulation processes", "OK")

    # Also clean up any orphan bash wrappers from previous SSH-hangs (harmless,
    # but they pile up and confuse pgrep). Best-effort, no failure on error.
    ssh_raw("pkill -f 'bash -c.*ring_reconnection' 2>/dev/null; true")

    # Check for running analysis (exclude bash -c orphans same as sim check)
    rc, out, _ = ssh_raw("pgrep -af 'run_full_analysis|visualize_all|pb11_phase|pb11_fusion' | grep -v 'bash -c' | grep -v grep || true")
    if out.strip():
        log(f"Stale analysis process found:\n  {out}", "WARN")
        log("Run: ssh gpu-node 'bash ~/laser-plasma-research/check_analysis_status.sh --clean'", "WARN")
        return False
    log("No stale analysis processes", "OK")

    # Check disk space
    # NOTE: Do NOT pipe through awk here — embedded single quotes collide
    # with ssh_raw's outer single-quote wrapping. Parse fields in Python.
    rc, out, _ = ssh_raw("df -BG ~/laser-plasma-research | tail -1")
    fields = out.split()
    free_gb = None
    if len(fields) >= 4:
        m = re.match(r"(\d+)", fields[3])
        if m:
            free_gb = int(m.group(1))
    if free_gb is not None:
        if free_gb < MIN_DISK_FREE_GB:
            log(f"Insufficient disk: {free_gb}GB free < {MIN_DISK_FREE_GB}GB required", "ERR")
            return False
        log(f"Disk OK: {free_gb}GB free", "OK")
    else:
        log(f"Could not parse disk space: {out!r}", "WARN")

    # Check if run dir already exists (warn, don't fail)
    if run_dir:
        rc, out, _ = ssh(f"ls -d '{run_dir}' 2>/dev/null || echo 'absent'")
        if "absent" not in out:
            log(f"Run dir already exists: {run_dir}", "WARN")
            log("Will overwrite if sim re-runs. Existing analysis preserved.", "WARN")

    log("Cloud environment: CLEAN", "OK")
    return True


# ============================================================================
# Step 2 — Launch simulation
# ============================================================================

def launch_simulation(sub_job: dict, paper: dict, dry_run=False) -> tuple[bool, str]:
    """
    Launch simulation on cloud. Returns (success, log_file_path).

    Uses fire-and-forget pattern: launches the sim with setsid to fully
    decouple from SSH's session, accepts that SSH may hang briefly even
    so, and verifies launch by polling pgrep for the unique --outdir flag.
    This is more robust than trying to capture $! synchronously over SSH.
    """
    sub_tag = sub_job["sub_tag"]
    # NOTE: .strip() is critical — YAML folded-scalar (>) preserves a trailing \n
    flags   = sub_job["flags_override"].strip()
    log_file = f"{sub_tag}.log"
    run_dir = f"runs/{sub_tag}"

    # `setsid` creates a new session for python, fully decoupling it from
    # SSH's tty/connection so SSH can close cleanly. The trailing `&` on the
    # whole setsid command + redirections + `disown` is belt-and-suspenders.
    launch_cmd = (
        f"setsid nohup python -u {SIM_SCRIPT} "
        f"{flags} "
        f"--outdir {run_dir} "
        f"< /dev/null > {log_file} 2>&1 & disown; echo started"
    )

    if dry_run:
        log(f"[DRY-RUN] Would launch: {launch_cmd}", "WARN")
        return True, log_file

    log(f"Launching simulation: {sub_tag}")
    log(f"  Flags: {flags}")

    # Fire-and-forget. If SSH hangs, we accept it; the sim is running anyway.
    # Use a short timeout (15s) so we don't waste time waiting.
    try:
        rc, out, err = ssh(launch_cmd, timeout=15)
        if rc != 0 and "started" not in out:
            log(f"Launch SSH non-zero rc={rc}, will verify via pgrep anyway", "WARN")
    except subprocess.TimeoutExpired:
        log("Launch SSH hung after sending command (sim still launches anyway)", "WARN")

    # Verify the sim is actually running on cloud by matching its unique --outdir flag
    log("  Verifying sim started on cloud...")
    time.sleep(5)  # give nohup/python a moment to fully start
    # Match the actual python process (not the bash wrapper that contains the
    # full launch_cmd in its cmdline). The python process's cmdline starts
    # with the script path, so anchor on that.
    pgrep_pattern = f"python.*pb11_ring_reconnection.*--outdir {run_dir}( |$)"
    rc, pid_out, _ = ssh(
        f"pgrep -af '{pgrep_pattern}' | grep -v 'bash -c' | awk '{{print $1}}' | head -1",
        timeout=20
    )
    pid_out = pid_out.strip()
    if pid_out and pid_out.isdigit():
        log(f"Simulation launched — PID {pid_out}, log: {CLOUD_ROOT}/{log_file}", "OK")
        return True, log_file
    else:
        log(f"Simulation not detected after launch (pgrep returned: {pid_out!r})", "ERR")
        # Try to show the log if it exists
        rc, log_tail, _ = ssh(f"tail -20 {log_file} 2>/dev/null", timeout=10)
        if log_tail:
            log(f"Last lines of {log_file}:\n{log_tail}", "ERR")
        return False, log_file


# ============================================================================
# Step 3 — Monitor simulation
# ============================================================================

def monitor_simulation(log_file: str, expected_steps: int, timeout_hours: float = 30) -> bool:
    """
    Poll simulation log until complete or timeout.
    Returns True if simulation completed successfully.
    """
    deadline = time.time() + timeout_hours * 3600
    last_step = 0
    last_step_time = time.time()
    stall_timeout = 600  # 10 min without progress = stalled

    log(f"Monitoring simulation (timeout={timeout_hours}h)...")

    while time.time() < deadline:
        # Check if process still running
        rc, out, _ = ssh_raw(f"pgrep -af 'ring_reconnection' | grep -v grep || true")
        process_running = bool(out.strip())

        # Check log for progress
        rc, log_tail, _ = ssh(f"tail -8 {log_file} 2>/dev/null", timeout=15)

        # Extract current step
        step_match = re.search(r"STEP\s+(\d+)\s+ends", log_tail)
        current_step = int(step_match.group(1)) if step_match else last_step

        # Extract simulation time
        time_match = re.search(r"TIME\s*=\s*([0-9.e+\-]+)", log_tail)
        sim_time_ns = float(time_match.group(1)) * 1e9 if time_match else 0

        # Extract avg step time
        avg_match = re.search(r"Avg\. per step\s*=\s*([0-9.]+)", log_tail)
        avg_step_sec = float(avg_match.group(1)) if avg_match else None

        # Progress tracking
        if current_step > last_step:
            last_step = current_step
            last_step_time = time.time()

        pct = 100 * current_step / expected_steps if expected_steps else 0

        # ETA
        eta_str = "unknown"
        if avg_step_sec and current_step > 0:
            remaining = (expected_steps - current_step) * avg_step_sec
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_str = eta.strftime("%H:%M")

        log(f"Step {current_step:,}/{expected_steps:,} ({pct:.1f}%) "
            f"| t={sim_time_ns:.2f}ns | ETA {eta_str}")

        # Update orchestrator state and show banner
        if STATE is not None:
            STATE.set_phase(
                "simulation",
                detail=f"step {current_step:,}/{expected_steps:,} | "
                       f"t={sim_time_ns:.2f}ns | step ETA {eta_str}",
                pct=pct,
            )
            banner()

        # Check for completion
        if not process_running:
            # Process exited — check if it finished or crashed
            rc, amrex_out, _ = ssh(f"tail -5 {log_file} | grep 'AMReX.*finalized' || echo ''")
            if "finalized" in amrex_out:
                rc2, step_out, _ = ssh(f"grep 'STEP.*ends' {log_file} | tail -1")
                final_step_match = re.search(r"STEP\s+(\d+)\s+ends", step_out)
                final_step = int(final_step_match.group(1)) if final_step_match else 0
                if final_step >= expected_steps:
                    log(f"Simulation COMPLETE at step {final_step:,} / {sim_time_ns:.2f}ns", "OK")
                    return True
                else:
                    log(f"Simulation exited early at step {final_step:,} (expected {expected_steps:,})", "ERR")
                    return False
            else:
                log("Process exited without AMReX finalization — likely crashed", "ERR")
                rc, crash, _ = ssh(f"tail -20 {log_file}")
                log(f"Last log lines:\n{crash}", "ERR")
                return False

        # Stall detection
        if time.time() - last_step_time > stall_timeout and current_step == last_step:
            log(f"No progress in {stall_timeout//60} min — simulation may be stalled", "ERR")
            return False

        time.sleep(POLL_INTERVAL_SEC)

    log(f"Simulation timed out after {timeout_hours}h", "ERR")
    return False


# ============================================================================
# Step 4 — Verify simulation outputs
# ============================================================================

def verify_simulation_outputs(run_dir: str, expected_dumps: int) -> bool:
    """Check AMReX finalized and particle dumps exist."""
    log("Verifying simulation outputs...")

    # Check particle dump count
    rc, out, _ = ssh(f"ls {run_dir}/particles 2>/dev/null | wc -l")
    actual_dumps = int(out.strip()) if out.strip().isdigit() else 0
    min_dumps = int(expected_dumps * 0.95)  # allow 5% tolerance

    if actual_dumps < min_dumps:
        log(f"Insufficient particle dumps: {actual_dumps} (expected ~{expected_dumps})", "ERR")
        return False
    log(f"Particle dumps: {actual_dumps} (expected ~{expected_dumps})", "OK")

    # Check run_meta.txt exists
    rc, out, _ = ssh(f"test -f {run_dir}/run_meta.txt && echo OK || echo MISSING")
    if "MISSING" in out:
        log("run_meta.txt missing — simulation may not have initialized", "ERR")
        return False
    log("run_meta.txt present", "OK")

    return True


# ============================================================================
# Step 5 — Run analysis pipeline
# ============================================================================

def run_analysis(run_dir: str, dry_run=False) -> tuple[bool, str]:
    """Launch the 8-stage analysis pipeline. Returns (success, log_file)."""
    log_file = f"{Path(run_dir).name}_analysis.log"

    if dry_run:
        log(f"[DRY-RUN] Would run analysis on {run_dir}", "WARN")
        return True, log_file

    log(f"Launching 8-stage analysis pipeline on {run_dir}...")
    # See launch_simulation for explanation of setsid + pgrep verify pattern
    launch_cmd = (
        f"setsid nohup bash {ANALYSIS_SCRIPT} {run_dir} "
        f"< /dev/null > {log_file} 2>&1 & disown; echo started"
    )
    try:
        rc, out, err = ssh(launch_cmd, timeout=15)
        if rc != 0 and "started" not in out:
            log(f"Analysis launch SSH non-zero rc={rc}, will verify via pgrep", "WARN")
    except subprocess.TimeoutExpired:
        log("Analysis launch SSH hung after sending command (analysis still launches)", "WARN")

    log("  Verifying analysis started on cloud...")
    time.sleep(5)
    run_dir_basename = Path(run_dir).name
    rc, pid_out, _ = ssh(
        f"pgrep -af 'bash.*run_full_analysis.*{run_dir_basename}( |$)' "
        f"| grep -v 'bash -c' | awk '{{print $1}}' | head -1",
        timeout=20
    )
    pid_out = pid_out.strip()
    if pid_out and pid_out.isdigit():
        log(f"Analysis launched — PID {pid_out}", "OK")
    else:
        log(f"Analysis not detected after launch (pgrep returned: {pid_out!r})", "ERR")
        return False, log_file

    # Monitor analysis
    deadline = time.time() + 3 * 3600  # 3 hour timeout
    while time.time() < deadline:
        # Check if still running
        rc, procs, _ = ssh_raw(
            f"pgrep -af 'run_full_analysis|visualize_all|pb11_phase|pb11_fusion|pb11_particle' "
            f"| grep -v grep || true"
        )
        running = bool(procs.strip())

        # Get stage progress
        rc, stages, _ = ssh(f"grep -E 'STAGE|COMPLETE|FAILED|ALL 8' {log_file} | tail -5")

        if stages:
            for line in stages.strip().split("\n"):
                log(f"  {line.strip()}")

        # Update state with current stage
        if STATE is not None and stages:
            # Parse "STAGE N/8" or "STAGE N COMPLETE" from latest line
            stage_match = re.search(r"STAGE\s+(\d+)/8:\s+(\S+)", stages)
            complete_count = len(re.findall(r"STAGE\s+\d+\s+COMPLETE", stages))
            if stage_match:
                stage_num = int(stage_match.group(1))
                stage_name = stage_match.group(2)
                pct = 100.0 * (complete_count / 8)
                STATE.set_phase(
                    "analysis",
                    detail=f"stage {stage_num}/8: {stage_name}  "
                           f"({complete_count} stages complete)",
                    pct=pct,
                )
                banner()

        if not running:
            # Check completion
            rc, done, _ = ssh(f"grep 'ALL 8 STAGES COMPLETE' {log_file} || echo ''")
            if "ALL 8 STAGES COMPLETE" in done:
                log("Analysis pipeline COMPLETE", "OK")
                if STATE is not None:
                    STATE.set_phase("analysis", "all 8 stages complete", 100.0)
                    banner()
                return True, log_file

            rc, failed, _ = ssh(f"grep 'FAILED' {log_file} || echo ''")
            if "FAILED" in failed:
                log(f"Analysis pipeline FAILED:\n{failed}", "ERR")
                return False, log_file

            log("Analysis process exited unexpectedly", "ERR")
            return False, log_file

        time.sleep(POLL_INTERVAL_SEC)

    log("Analysis timed out", "ERR")
    return False, log_file


# ============================================================================
# Step 6 — Verify analysis products (comprehensive, by stage)
# ============================================================================

def _format_size(size_bytes: int) -> str:
    """Format bytes as human readable (B, KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    return f"{size_bytes/(1024*1024):.1f}MB"


def get_remote_file_sizes(run_dir: str, file_list: list) -> dict:
    """
    Single SSH call to stat all required files at once.
    Returns: {filename: size_in_bytes_or_None_if_missing}
    """
    # Build a shell command that prints "<path>|<size>" for each file
    paths = " ".join(f"'{run_dir}/{rel_path}'" for rel_path, _, _ in file_list)
    cmd = (
        f'for f in {paths}; do '
        f'  if [ -f "$f" ]; then echo "$f|$(stat -c %s "$f")"; '
        f'  else echo "$f|MISSING"; fi; '
        f'done'
    )
    rc, out, _ = ssh(cmd, timeout=30)
    sizes = {}
    for line in out.strip().split("\n"):
        if "|" not in line:
            continue
        path, size = line.split("|", 1)
        # Strip the run_dir prefix to get back to the relative path
        rel = path.split(f"{run_dir}/", 1)[-1].rstrip("'")
        if size == "MISSING":
            sizes[rel] = None
        else:
            try:
                sizes[rel] = int(size)
            except ValueError:
                sizes[rel] = None
    return sizes


def verify_stage(stage_name: str, run_dir: str, file_list: list,
                 strict: bool = True) -> tuple[bool, dict]:
    """
    Verify all files in one stage. Returns (all_ok, results_dict).
    If strict=False, treats missing files as warnings, not failures.
    """
    if not file_list:
        return True, {}

    sizes = get_remote_file_sizes(run_dir, file_list)
    all_ok = True
    results = {}

    for rel_path, min_size, owner in file_list:
        actual_size = sizes.get(rel_path)
        results[rel_path] = (actual_size, min_size, owner)

        if actual_size is None:
            level = "ERR" if strict else "WARN"
            log(f"  [{owner:24s}] {rel_path:55s}  MISSING", level)
            if strict:
                all_ok = False
        elif actual_size < min_size:
            log(f"  [{owner:24s}] {rel_path:55s}  "
                f"too small: {_format_size(actual_size)} < {_format_size(min_size)}", "WARN")
            if strict:
                all_ok = False
        else:
            log(f"  [{owner:24s}] {rel_path:55s}  {_format_size(actual_size)}", "OK")

    return all_ok, results


def verify_analysis_products(run_dir: str) -> tuple[bool, dict]:
    """
    Comprehensive verification of all 8 stages.
    Returns: (all_critical_ok, full_results)
    """
    log("Verifying analysis products (comprehensive)...", "HEAD")

    full_results = {}
    failures_by_stage = {}

    # Group display by section header
    section_titles = {
        "simulation":      "Pre-existing simulation outputs",
        "stage1_figures":  "Stage 1: visualize_all — figures",
        "stage1_cache":    "Stage 1: visualize_all — cache (informational)",
        "stage2":          "Stage 2: pb11_consolidate_reports",
        "stage3":          "Stage 3: pb11_phase_analysis",
        "stage4":          "Stage 4: pb11_reconnection_rate_offline",
        "stage5_csv":      "Stage 5: pb11_fusion_diagnostics — reports",
        "stage5_figures":  "Stage 5: pb11_fusion_diagnostics — figures",
        "stage6":          "Stage 6: pb11_particle_animations",
        "stage7":          "Stage 7: pb11_particle_trajectories",
    }

    for stage_key, file_list in REQUIRED_OUTPUTS.items():
        if not file_list:
            continue
        title = section_titles.get(stage_key, stage_key)
        log(f"--- {title} ---")
        strict = stage_key in CRITICAL_STAGES
        ok, results = verify_stage(stage_key, run_dir, file_list, strict=strict)
        full_results[stage_key] = results
        if not ok and strict:
            failures_by_stage[stage_key] = [
                p for p, (sz, _, _) in results.items() if sz is None
            ]

    # Summary
    n_total = sum(len(v) for v in REQUIRED_OUTPUTS.values())
    n_present = sum(
        1 for stage_results in full_results.values()
        for sz, _, _ in stage_results.values() if sz is not None
    )
    n_critical_failed = sum(len(v) for v in failures_by_stage.values())

    log(f"Verification: {n_present}/{n_total} files present, "
        f"{n_critical_failed} critical failures", "HEAD")

    if failures_by_stage:
        log("Critical failures by stage:", "ERR")
        for stage, files in failures_by_stage.items():
            log(f"  {stage}: {', '.join(files)}", "ERR")
        return False, full_results

    return True, full_results


# ============================================================================
# Step 7 — Download to Mac
# ============================================================================

def download_results(run_dir: str, local_base: str, sub_tag: str) -> bool:
    """rsync analysis products from cloud to Mac.

    Downloads:
      - All figures (png/gif/mp4) from figures/ and figures/individual/
      - All required reports + CSVs (everything in REQUIRED_OUTPUTS except cache)
      - Pre-existing simulation outputs (run_meta, fusion_rate_power_by_iter, etc.)
    """
    log(f"Downloading results to Mac...")

    local_run_dir = Path(local_base) / sub_tag
    local_run_dir.mkdir(parents=True, exist_ok=True)
    (local_run_dir / "figures" / "individual").mkdir(parents=True, exist_ok=True)

    # ── Download figures (top-level + individual subdir) ──
    log("  Downloading figures (png/gif/mp4) from figures/...")
    ok1 = rsync_down(
        f"{SSH_HOST}:{CLOUD_ROOT}/{run_dir}/figures/",
        str(local_run_dir / "figures"),
        include_patterns=["*.png", "*.gif", "*.mp4"]
    )

    log("  Downloading individual panels...")
    ok2 = rsync_down(
        f"{SSH_HOST}:{CLOUD_ROOT}/{run_dir}/figures/individual/",
        str(local_run_dir / "figures" / "individual"),
        include_patterns=["*.png"]
    )

    # ── Build complete file list from REQUIRED_OUTPUTS ──
    # Skip cache (npz files) since those are large and only useful on cloud,
    # and skip stage1_figures since we already pulled all figures above.
    skip_stages = {"stage1_figures", "stage1_cache", "stage5_figures",
                   "stage6", "stage7"}  # all figure-only stages
    text_files = []
    for stage_key, file_list in REQUIRED_OUTPUTS.items():
        if stage_key in skip_stages:
            continue
        for rel_path, _, _ in file_list:
            # Only pull text/csv files via this path (figures came via rsync above)
            if rel_path.endswith((".txt", ".csv", ".log")):
                text_files.append(f"'{CLOUD_ROOT}/{run_dir}/{rel_path}'")

    if text_files:
        cmd = f"rsync -avh {' '.join(text_files)} '{str(local_run_dir)}/'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        ok3 = result.returncode == 0
    else:
        ok3 = True

    if ok1 and ok2 and ok3:
        log(f"Download complete → {local_run_dir}", "OK")
        return True
    else:
        log(f"Download partial: figures_top={ok1}, figures_individual={ok2}, reports={ok3}", "WARN")
        return False


# ============================================================================
# Step 8 — Cloud cleanup
#   Removes ONLY particles/ and fields/ (raw simulation data, ~400 GB-1.6 TB).
#   Keeps everything else: figures, reports, CSVs, AND the cache (~2 MB)
#   so figures can be regenerated from cache if ever needed.
# ============================================================================

def cleanup_cloud(run_dir: str, dry_run=False) -> bool:
    """Remove only particle + field dumps. Preserves all analysis artifacts."""
    log(f"Cleaning up cloud (particles + fields only) for {run_dir}...")

    if dry_run:
        log("[DRY-RUN] Would remove particles/ and fields/", "WARN")
        return True

    # Show sizes of what we're about to remove (and what we're keeping)
    rc, out, _ = ssh(
        f"echo '  Removing:'; "
        f"du -sh {run_dir}/particles {run_dir}/fields 2>/dev/null; "
        f"echo '  Keeping:'; "
        f"du -sh {run_dir}/figures {run_dir}/figures/cache 2>/dev/null; "
        f"du -sb {run_dir}/*.txt {run_dir}/*.csv 2>/dev/null | "
        f"awk '{{s+=$1}} END {{printf \"  %.1f KB total in reports/CSVs\\n\", s/1024}}'"
    )
    if out:
        for line in out.splitlines():
            log(line)

    # Remove ONLY particles and fields. Keep figures/, figures/cache/, all reports.
    rc, _, err = ssh(
        f"rm -rf {run_dir}/particles {run_dir}/fields && echo OK"
    )
    if rc != 0:
        log(f"Cleanup failed: {err}", "WARN")
        return False

    rc, disk, _ = ssh_raw(f"df -h ~/laser-plasma-research | tail -1")
    log(f"  Disk after cleanup: {disk}", "OK")
    return True


# ============================================================================
# State management (resumable runs)
# ============================================================================

def load_state(config_path: str) -> dict:
    state_file = Path(config_path).parent / STATE_FILE
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "attempts": {}}


def save_state(config_path: str, state: dict):
    state_file = Path(config_path).parent / STATE_FILE
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# Config parsing
# ============================================================================

def load_config(config_path: str, paper_id: str) -> tuple[dict, dict]:
    """Load YAML config. Returns (program_config, paper_config)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    papers = cfg.get("papers", [])
    paper = next((p for p in papers if p["id"] == paper_id), None)
    if paper is None:
        ids = [p["id"] for p in papers]
        print(f"ERROR: paper '{paper_id}' not found. Available: {ids}", file=sys.stderr)
        sys.exit(1)

    return cfg, paper


# ============================================================================
# Main orchestration loop
# ============================================================================

def run_paper(cfg: dict, paper: dict, args) -> dict:
    """
    Orchestrate all sub_jobs for a paper.
    Returns summary dict.
    """
    sub_jobs = paper.get("simulation", {}).get("sub_jobs", [])
    local_base = Path(args.local_runs_dir).expanduser()
    state = load_state(args.config)

    # Filter to specific jobs if requested
    if args.jobs:
        requested = set(args.jobs.split(","))
        sub_jobs = [j for j in sub_jobs if j["sub_tag"] in requested]
        if not sub_jobs:
            log(f"No matching jobs for --jobs {args.jobs}", "ERR")
            sys.exit(1)

    # Capture the (possibly --jobs-filtered) list before the skip-completed
    # filter, so the early-exit path can report which jobs were already done.
    original_sub_jobs = list(sub_jobs)

    # Skip already completed
    if not args.rerun:
        pending = [j for j in sub_jobs if j["sub_tag"] not in state["completed"]]
        skipped = len(sub_jobs) - len(pending)
        if skipped:
            log(f"Skipping {skipped} already-completed job(s). Use --rerun to force.")
        sub_jobs = pending

    if not sub_jobs:
        log("All jobs already complete!", "OK")
        # Return a results dict with the same shape as the normal-exit path.
        # All matching jobs are already in state["completed"], so report them
        # as successes for the FINAL SUMMARY (no work done this invocation).
        already_done = [
            j["sub_tag"] for j in original_sub_jobs
            if j["sub_tag"] in state["completed"]
        ]
        return {"success": already_done, "failed": []}

    # Initialize global state tracker
    global STATE
    STATE = OrchestratorState(total_jobs=len(sub_jobs))
    log(f"Initialized progress tracker for {len(sub_jobs)} jobs")
    banner()

    results = {"success": [], "failed": []}
    sim_cfg = paper.get("simulation", {})
    expected_steps = sim_cfg.get("expected_steps", 200000)
    timeout_hours = sim_cfg.get("estimated_runtime_hours", 10)
    expected_dumps = sim_cfg.get("expected_dumps", expected_steps // sim_cfg.get("dump_period", 2000) + 1)

    for job_idx, sub_job in enumerate(sub_jobs):
        sub_tag = sub_job["sub_tag"]
        run_dir = f"runs/{sub_tag}"
        note    = sub_job.get("note", "")

        STATE.begin_job(job_idx + 1, sub_tag)
        sep(f"Job {job_idx+1}/{len(sub_jobs)}: {sub_tag}")
        if note:
            log(f"Note: {note}")
        banner()

        # Honor per-sub-job runtime override if present
        job_timeout_hours = sub_job.get("estimated_runtime_hours", timeout_hours)
        job_max_steps = expected_steps
        # Try to extract --max-steps from flags_override for accurate ETA
        flags = sub_job.get("flags_override", "").strip()
        ms = re.search(r"--max-steps\s+(\d+)", flags)
        if ms:
            job_max_steps = int(ms.group(1))
        dp = re.search(r"--dump-period\s+(\d+)", flags)
        job_expected_dumps = (job_max_steps // int(dp.group(1)) + 1) if dp else expected_dumps

        # Track attempts
        attempts = state["attempts"].get(sub_tag, 0)
        job_success = False

        while attempts < MAX_RETRIES:
            attempts += 1
            state["attempts"][sub_tag] = attempts
            STATE.current_attempt = attempts
            if not args.dry_run:
                save_state(args.config, state)

            log(f"Attempt {attempts}/{MAX_RETRIES}", "HEAD")

            # ── Step 1: Environment check ──
            STATE.set_phase("env_check", "checking cloud environment")
            banner()
            if not check_cloud_environment(run_dir):
                log("Environment not clean — manual intervention needed", "ERR")
                if attempts < MAX_RETRIES:
                    log(f"Waiting 60s before retry...", "WARN")
                    time.sleep(60)
                continue

            # ── Step 2: Launch simulation ──
            if args.skip_sim:
                log("Skipping simulation (--skip-sim)", "WARN")
                sim_ok = True
                log_file = f"{sub_tag}.log"
            else:
                STATE.set_phase("simulation", "launching")
                banner()
                sim_ok, log_file = launch_simulation(sub_job, paper, dry_run=args.dry_run)
                if not sim_ok:
                    log("Simulation launch failed", "ERR")
                    continue

                # ── Step 3: Monitor simulation ──
                if not args.dry_run:
                    sim_ok = monitor_simulation(log_file, job_max_steps, job_timeout_hours)
                    if not sim_ok:
                        log("Simulation did not complete successfully", "ERR")
                        continue

            # ── Step 4: Verify simulation outputs ──
            if not args.dry_run and not args.skip_sim:
                STATE.set_phase("verify_sim", "checking dumps")
                banner()
                if not verify_simulation_outputs(run_dir, job_expected_dumps):
                    log("Simulation output verification failed", "ERR")
                    continue

            # ── Step 5: Run analysis pipeline ──
            if args.skip_analysis:
                log("Skipping analysis (--skip-analysis)", "WARN")
                analysis_ok = True
            else:
                STATE.set_phase("analysis", "running 8-stage pipeline")
                banner()
                analysis_ok, analysis_log = run_analysis(run_dir, dry_run=args.dry_run)
                if not analysis_ok:
                    log("Analysis pipeline failed", "ERR")
                    continue

            # ── Step 6: Verify analysis products ──
            if not args.dry_run and not args.skip_analysis:
                STATE.set_phase("verify_analysis", "checking outputs")
                banner()
                verify_ok, verify_results = verify_analysis_products(run_dir)
                if not verify_ok:
                    log("Analysis product verification failed", "ERR")
                    continue

            # ── Step 7: Download to Mac ──
            if not args.dry_run:
                STATE.set_phase("download", "rsync to Mac")
                banner()
                download_results(run_dir, str(local_base), sub_tag)

            # ── Step 8: Cloud cleanup (particles + fields only) ──
            if args.cleanup and not args.dry_run:
                STATE.set_phase("cleanup", "removing particles + fields")
                banner()
                cleanup_cloud(run_dir, dry_run=args.dry_run)

            STATE.set_phase("complete", "done")
            banner()
            job_success = True
            break

        # Record outcome.
        # In dry-run, populate the in-memory `results` so the FINAL SUMMARY
        # reflects what *would* have happened, but do NOT mutate `state` or
        # write to cloud_run_state.json — dry-runs must be side-effect-free.
        STATE.finish_job(job_success)
        if job_success:
            if not args.dry_run:
                state["completed"].append(sub_tag)
            results["success"].append(sub_tag)
            log(f"Job {sub_tag}: COMPLETE ✓", "OK")
        else:
            if not args.dry_run:
                state["failed"].append(sub_tag)
            results["failed"].append(sub_tag)
            log(f"Job {sub_tag}: FAILED after {attempts} attempts", "ERR")

        if not args.dry_run:
            save_state(args.config, state)
        banner()

    # ── Post-program: cross-run aggregation analysis (e.g. freq_resonance.py) ──
    post_scripts = paper.get("post_program_analysis", [])
    if post_scripts and results["success"] and not args.skip_analysis:
        sep("Post-program cross-run analysis")
        for script_cfg in post_scripts:
            script_path = script_cfg["script"]
            args_str    = script_cfg.get("args", "")
            description = script_cfg.get("description", script_path)
            output_file = script_cfg.get("output_file")

            cmd = f"python -u analysis_scripts/{script_path} {args_str}"
            if args.dry_run:
                log(f"[DRY-RUN] Would run {description}: {cmd}", "WARN")
                continue

            log(f"Running {description}...")
            rc, out, err = ssh(cmd, timeout=300)
            if rc == 0:
                log(f"  Cross-run analysis complete", "OK")
                # Save output to local cross-run reports dir
                cross_dir = Path(args.local_runs_dir).expanduser() / "_cross_run_reports" / paper["id"]
                cross_dir.mkdir(parents=True, exist_ok=True)
                out_local = cross_dir / (output_file or f"{Path(script_path).stem}_output.txt")
                out_local.write_text(out)
                log(f"  Output saved → {out_local}", "OK")
            else:
                log(f"  Cross-run analysis FAILED: {err}", "ERR")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cloud orchestrator for ring-reconnection research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="program_config_paper2.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--paper", required=True,
                        help="Paper ID to run (e.g. A2)")
    parser.add_argument("--jobs",
                        help="Comma-separated list of sub_tag names to run (default: all)")
    parser.add_argument("--local-runs-dir",
                        default="~/LaserFusionResearch/research/laser-plasma-research/runs",
                        help="Local Mac directory to download results into")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation, go straight to analysis")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis pipeline")
    parser.add_argument("--cleanup", action="store_true", default=True,
                        help="Delete cloud particles/fields after successful download (default: on)")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                        help="Keep particles/fields on cloud after download")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run even jobs already marked complete in state file")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                        help=f"Max attempts per job (default: {MAX_RETRIES})")
    args = parser.parse_args()

    # Expand local dir
    args.local_runs_dir = str(Path(args.local_runs_dir).expanduser())

    # Load config
    cfg, paper = load_config(args.config, args.paper)

    # Show plan
    sub_jobs = paper.get("simulation", {}).get("sub_jobs", [])
    sep(f"Cloud Orchestrator — Paper {args.paper}: {paper.get('label','')}")
    log(f"Config:     {args.config}")
    log(f"SSH host:   {SSH_HOST}")
    log(f"Cloud root: {CLOUD_ROOT}")
    log(f"Local dir:  {args.local_runs_dir}")
    log(f"Jobs ({len(sub_jobs)}): {', '.join(j['sub_tag'] for j in sub_jobs)}")
    if args.dry_run:
        log("DRY-RUN MODE — no commands will execute", "WARN")

    # Run
    results = run_paper(cfg, paper, args)

    # Final summary
    sep("FINAL SUMMARY")
    if STATE is not None:
        log(f"Total elapsed:  {STATE.elapsed_str}")
        log(f"Final progress: {STATE.overall_pct:.1f}%")
        log(f"Banner:         {STATE.banner()}")
        log("")
    log(f"Successful: {len(results['success'])} — {results['success']}")
    log(f"Failed:     {len(results['failed'])} — {results['failed']}")
    log(f"Local dir:  {args.local_runs_dir}")

    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
