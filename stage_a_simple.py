#!/usr/bin/env python3
"""
stage_a_simple.py — Paper 2 Stage A serial runner.

A focused replacement for cloud_run_all.py for the 4-job Stage A sweep.
No state file, no retries, no pgrep verification. Each job runs in sequence:

    launch sim → poll log for completion → run analysis → rsync down → cleanup → next

If something breaks, edit JOBS to remove already-completed entries and re-run.

Usage:
    python3 stage_a_simple.py
    python3 stage_a_simple.py --jobs p2_ghz_36,p2_ghz_38   # subset

Detached:
    nohup python3 -u stage_a_simple.py > stage_a.log 2>&1 &
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration — edit JOBS to add/remove frequencies
# ============================================================================

CLOUD_HOST = "substrate-gpu"
CLOUD_ROOT = "~/laser-plasma-research"
CONDA_INIT = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma"
SIM_SCRIPT = "simulation/pb11_ring_reconnection_v12_fuel_center_outer.py"
ANALYSIS_SCRIPT = "analysis_scripts/run_full_analysis.sh"
RESONANCE_SCRIPT = "analysis_scripts/freq_resonance_ghz.py"
LOCAL_RUNS = Path.home() / "LaserFusionResearch/research/laser-plasma-research/runs"
LOCAL_PAPER_DIR = LOCAL_RUNS / "paper02"  # Local target for Paper 2 deliverables.
                                          # Cloud side stays flat at runs/<sub_tag>/.

# Common simulation flags (per Paper 2 YAML)
COMMON_FLAGS = (
    "--base-fuel ch_bn --base-density 5e24 --b-seed 85 --n-spots 8 "
    "--lx-min-um 9600 --nx 256 --nz 256 --max-steps 4500 --diag-profile custom --rotate"
)
MAX_STEPS = 4500

# Per-job definitions: (sub_tag, freq_hz, dump_period)
JOBS = [
    ("p2_ghz_18", "18e9", 50),  # ultrafine_default cadence
    ("p2_ghz_33", "33e9", 50),  # ultrafine_default cadence
    ("p2_ghz_36", "36e9", 33),  # ultrafine_high_freq cadence
    ("p2_ghz_38", "38e9", 33),  # ultrafine_high_freq cadence
]

# Polling cadence
POLL_INTERVAL_SEC = 60          # check sim progress every minute
SIM_TIMEOUT_HOURS = 3           # abort if sim takes longer than 3h
NO_PROGRESS_TIMEOUT_SEC = 600   # abort if step counter stuck for 10 min

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

def ssh_run(remote_cmd: str, timeout: int = 60, check: bool = True) -> tuple[int, str, str]:
    """Run a foreground command via SSH. Returns (rc, stdout, stderr)."""
    full = f"ssh -n {CLOUD_HOST} '{CONDA_INIT} && cd {CLOUD_ROOT} && {remote_cmd}'"
    try:
        r = subprocess.run(
            full, shell=True, capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
        if check and r.returncode != 0:
            log(f"SSH command failed (rc={r.returncode}): {remote_cmd[:80]}", "✗ ")
            log(f"  stderr: {r.stderr.strip()[:300]}", "  ")
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        log(f"SSH command timed out: {remote_cmd[:80]}", "⚠ ")
        return 124, "", "timeout"

def ssh_launch_detached(launch_cmd: str, ssh_timeout: int = 15) -> None:
    """Fire-and-forget launch via SSH. Accepts SSH hanging because setsid detaches."""
    full = f"ssh -n {CLOUD_HOST} '{CONDA_INIT} && cd {CLOUD_ROOT} && {launch_cmd}'"
    try:
        subprocess.run(
            full, shell=True, capture_output=True, text=True,
            timeout=ssh_timeout, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        # Expected — SSH hangs because of inherited fds, but the python is
        # already running via setsid on cloud
        pass

# ============================================================================
# Per-job pipeline
# ============================================================================

def launch_sim(sub_tag: str, freq_hz: str, dump_period: int) -> bool:
    """Launch sim on cloud. Returns True if sim is running afterward."""
    run_dir = f"runs/{sub_tag}"
    sim_log = f"{sub_tag}.log"

    flags = (
        f"{COMMON_FLAGS} --freq {freq_hz} --dump-period {dump_period} "
        f"--outdir {run_dir}"
    )
    launch_cmd = (
        f"setsid nohup python -u {SIM_SCRIPT} {flags} "
        f"< /dev/null > {sim_log} 2>&1 & disown; echo started"
    )

    log(f"Launching sim: freq={freq_hz}, dump-period={dump_period}", "→ ")
    ssh_launch_detached(launch_cmd, ssh_timeout=15)

    # Verify by checking that the log file is being written
    log("Verifying via log-file growth...", "  ")
    time.sleep(15)
    rc, log_size, _ = ssh_run(f"stat -c %s {sim_log} 2>/dev/null || echo 0", timeout=15, check=False)
    if rc != 0 or not log_size.isdigit() or int(log_size) == 0:
        log(f"Sim log empty or missing — launch failed", "✗ ")
        return False
    log(f"Sim log is {log_size} bytes — sim is running", "✓ ")
    return True

def poll_sim(sub_tag: str) -> bool:
    """Poll sim until completion or failure. Returns True on clean completion."""
    sim_log = f"{sub_tag}.log"
    deadline = time.time() + SIM_TIMEOUT_HOURS * 3600
    last_step = -1
    last_progress_at = time.time()
    last_log_size = -1

    log(f"Polling sim — expected ~{SIM_TIMEOUT_HOURS}h, MAX_STEPS={MAX_STEPS}", "  ")

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)

        # Check current step from log
        # NOTE: using double-quotes around the regex is critical. Single
        # quotes would collide with ssh_run's outer single-quote wrapping
        # (same bug pattern as awk-with-single-quotes earlier).
        # Also avoid character classes like [0-9]+ in the regex — extract
        # the step number in Python instead.
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
            log(f"step {cur_step}/{MAX_STEPS} ({100*cur_step//MAX_STEPS}%)", "  ")

        # Detect completion
        if cur_step >= MAX_STEPS:
            # Wait briefly for python to finish writing post-run output
            time.sleep(10)
            log(f"Sim reached MAX_STEPS — completion confirmed", "✓ ")
            return True

        # Detect process death (log not growing AND no further step progress)
        if log_size_int == last_log_size and cur_step == last_step:
            stuck_for = time.time() - last_progress_at
            if stuck_for > NO_PROGRESS_TIMEOUT_SEC:
                # Verify by checking process really is dead via log-tail content
                rc, tail, _ = ssh_run(f"tail -5 {sim_log}", timeout=15, check=False)
                log(f"No progress for {int(stuck_for)}s — sim may have crashed", "✗ ")
                log(f"Log tail:\n{tail}", "  ")
                return False
        else:
            last_progress_at = time.time()
            last_log_size = log_size_int
            last_step = cur_step

    log(f"Sim exceeded {SIM_TIMEOUT_HOURS}h timeout", "✗ ")
    return False

def run_analysis(sub_tag: str) -> bool:
    """Run the 8-stage analysis pipeline synchronously (foreground SSH, no hang issue)."""
    run_dir = f"runs/{sub_tag}"
    analysis_log = f"{sub_tag}_analysis.log"

    log("Running 8-stage analysis pipeline...", "→ ")
    # Run analysis with a long timeout (3h max). SSH foreground is fine here
    # because the analysis script doesn't background anything.
    cmd = f"bash {ANALYSIS_SCRIPT} {run_dir} > {analysis_log} 2>&1"
    rc, _, _ = ssh_run(cmd, timeout=3 * 3600, check=False)
    if rc != 0:
        log(f"Analysis failed (rc={rc}). Tail of {analysis_log}:", "✗ ")
        _, tail, _ = ssh_run(f"tail -30 {analysis_log}", timeout=15, check=False)
        log(tail, "  ")
        return False
    log("Analysis pipeline complete", "✓ ")
    return True

def rsync_down(sub_tag: str) -> bool:
    """Pull figures, reports, and cache npz files down to Mac."""
    run_dir = f"runs/{sub_tag}"
    local_dir = LOCAL_PAPER_DIR / sub_tag
    local_dir.mkdir(parents=True, exist_ok=True)

    log(f"Rsyncing results to {local_dir}", "→ ")
    cmd = (
        f"rsync -avh "
        f"--include='*/' "
        f"--include='*.png' --include='*.mp4' --include='*.txt' "
        f"--include='*.csv' --include='*.npz' "
        f"--exclude='*' "
        f"{CLOUD_HOST}:{CLOUD_ROOT}/{run_dir}/ {local_dir}/"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        log(f"Rsync failed: {r.stderr.strip()[:300]}", "✗ ")
        return False
    log("Rsync complete", "✓ ")
    return True

def cleanup_cloud(sub_tag: str) -> None:
    """Delete cloud particles/fields to free disk."""
    run_dir = f"runs/{sub_tag}"
    log("Cleaning cloud particles + fields", "→ ")
    ssh_run(f"rm -rf {run_dir}/particles {run_dir}/fields", timeout=60, check=False)
    rc, disk, _ = ssh_run("df -h ~/laser-plasma-research | tail -1", timeout=15, check=False)
    log(f"Cloud disk after cleanup: {disk}", "  ")

def sim_already_complete(sub_tag: str) -> bool:
    """Return True if cloud already has a completed sim for this sub_tag.

    Detection: particles/ directory has at least 50 .h5 dumps. A complete
    4500-step run with dump_period=33 produces ~136 dumps; with
    dump_period=50, ~91 dumps. Partial/early-aborted sims have far fewer.
    50 is a safe lower bound that distinguishes complete from partial.

    We deliberately do NOT check the sim log for AMReX finalization,
    because a re-launched sim overwrites the log file even if the
    underlying particle data from the previous run is still intact.
    """
    run_dir = f"runs/{sub_tag}"
    rc, particle_count, _ = ssh_run(
        f"ls {run_dir}/particles 2>/dev/null | wc -l",
        timeout=15, check=False,
    )
    try:
        n_dumps = int(particle_count) if particle_count.isdigit() else 0
    except ValueError:
        return False
    return n_dumps >= 50

def run_one_job(sub_tag: str, freq_hz: str, dump_period: int) -> bool:
    banner(f"Job: {sub_tag} (freq={freq_hz}, dump-period={dump_period})")
    if sim_already_complete(sub_tag):
        log(f"Sim data already complete on cloud — skipping launch + polling", "↪ ")
    else:
        if not launch_sim(sub_tag, freq_hz, dump_period):
            return False
        if not poll_sim(sub_tag):
            return False
    if not run_analysis(sub_tag):
        return False
    if not rsync_down(sub_tag):
        return False
    cleanup_cloud(sub_tag)
    log(f"Job {sub_tag}: COMPLETE", "✓ ")
    return True

def cross_run_analysis() -> None:
    banner("Cross-run resonance analysis")
    cmd = (
        f"python -u {RESONANCE_SCRIPT} "
        f"--base-dir runs "
        f"--static-ultrafine runs/p1_ld_512_4500_ultrafine "
        f"--static-long runs/p1_ld_512_200k_long"
    )
    rc, out, err = ssh_run(cmd, timeout=600, check=False)
    if rc != 0:
        log(f"Cross-run analysis failed (rc={rc})", "✗ ")
        log(err[:500], "  ")
        return
    cross_dir = LOCAL_PAPER_DIR / "_cross_run_reports" / "A2"
    cross_dir.mkdir(parents=True, exist_ok=True)
    out_file = cross_dir / "freq_resonance_GHz_sweep.txt"
    out_file.write_text(out)
    log(f"Cross-run output saved: {out_file}", "✓ ")

# ============================================================================
# Main
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", help="Comma-separated subset of sub_tags to run")
    args = p.parse_args()

    jobs = JOBS
    if args.jobs:
        wanted = set(args.jobs.split(","))
        jobs = [j for j in JOBS if j[0] in wanted]
        if not jobs:
            print(f"No matching jobs in {args.jobs}. Available: {[j[0] for j in JOBS]}")
            return 1

    banner(f"Stage A simple runner — {len(jobs)} jobs: {', '.join(j[0] for j in jobs)}")

    succeeded, failed = [], []
    for (sub_tag, freq, dump_period) in jobs:
        ok = run_one_job(sub_tag, freq, dump_period)
        (succeeded if ok else failed).append(sub_tag)
        if not ok:
            log(f"Aborting remaining jobs after {sub_tag} failure.", "✗ ")
            log(f"To resume, edit JOBS to start from where it left off.", "  ")
            break

    if succeeded:
        cross_run_analysis()

    banner("FINAL SUMMARY")
    print(f"  Succeeded: {len(succeeded)} — {succeeded}", flush=True)
    print(f"  Failed:    {len(failed)} — {failed}", flush=True)
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
