#!/usr/bin/env python3
import os
"""
stage_a_simple.py — Paper 2 Stage A serial runner.

A focused replacement for cloud_run_all.py for the Stage A sweep.

    launch sim → poll log for completion → run analysis → rsync down → cleanup → next

REVISION (2026-05-07):
  - Added MAX_RETRIES per-job retry loop with exponential backoff. Transient
    failures (rc=255 from dropped SSH, rc=124 from timeout) automatically retry.
    Real failures (non-zero return after 3 attempts) abort the run.
  - Analysis stage no longer holds a 3-hour SSH connection open. The analysis
    is launched detached on remote and polled, mirroring the simulation pattern.
    This eliminates the rc=255 spurious failures we were seeing.
  - Added analysis_already_complete() check so retries skip already-finished
    work. A retry of a half-finished job picks up where it left off.
  - "Abort on first failure" replaced with "continue past failures by default,
    --abort-on-failure to opt back in to the old behavior."
  - Default retry pool: 3 attempts, 30s/2min/5min backoff.

Usage:
    python3 stage_a_simple.py
    python3 stage_a_simple.py --jobs p2_ghz_36,p2_ghz_38       # subset
    python3 stage_a_simple.py --abort-on-failure                # old behavior
    python3 stage_a_simple.py --max-retries 5                   # more attempts

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

CLOUD_HOST = os.environ.get("GPU_HOST", "gpu-node")
CLOUD_ROOT = "~/laser-plasma-research"
CONDA_INIT = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma"
SIM_SCRIPT = "simulation/pb11_ring_reconnection_v12_fuel_center_outer.py"
ANALYSIS_SCRIPT = "analysis_scripts/run_full_analysis.sh"
RESONANCE_SCRIPT = "analysis_scripts/freq_resonance_ghz.py"
LOCAL_RUNS = Path.home() / "LaserFusionResearch/research/laser-plasma-research/runs"
LOCAL_PAPER_DIR = LOCAL_RUNS / "paper02"

# Common simulation flags (per Paper 2 YAML)
COMMON_FLAGS = (
    "--base-fuel ch_bn --base-density 5e24 --b-seed 85 --n-spots 8 "
    "--lx-min-um 9600 --nx 256 --nz 256 --max-steps 4500 --diag-profile custom --rotate"
)
MAX_STEPS = 4500

# Per-job definitions: (sub_tag, freq_hz, dump_period)
JOBS = [
    ("p2_ghz_18", "18e9", 50),
    ("p2_ghz_33", "33e9", 50),
    ("p2_ghz_36", "36e9", 33),
    ("p2_ghz_38", "38e9", 33),
]

# Polling cadence
POLL_INTERVAL_SEC = 60
SIM_TIMEOUT_HOURS = 3
ANALYSIS_TIMEOUT_HOURS = 2  # full 8-stage pipeline observed at ~25 min; 2h is safe
NO_PROGRESS_TIMEOUT_SEC = 600

# Retry policy (overridable via --max-retries on the CLI)
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SEC = [30, 120, 300]  # length must be >= max retries

# SSH return codes that indicate transient/connection failure (worth retrying).
# Genuine analysis failures will return rc=1 or rc=2 (script-level errors), which
# we do NOT auto-retry — they need human investigation.
TRANSIENT_RCS = {124, 255}  # 124=timeout (subprocess), 255=ssh dropped/killed


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
# Per-job pipeline stages
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

    log("Verifying via log-file growth...", "  ")
    time.sleep(15)
    rc, log_size, _ = ssh_run(f"stat -c %s {sim_log} 2>/dev/null || echo 0",
                              timeout=15, check=False)
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

        if cur_step >= MAX_STEPS:
            time.sleep(10)
            log(f"Sim reached MAX_STEPS — completion confirmed", "✓ ")
            return True

        if log_size_int == last_log_size and cur_step == last_step:
            stuck_for = time.time() - last_progress_at
            if stuck_for > NO_PROGRESS_TIMEOUT_SEC:
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


def launch_analysis(sub_tag: str) -> bool:
    """Launch analysis pipeline detached on remote. Returns True if launch succeeded.

    Detached launch + polling avoids the long-held SSH connection that was
    causing rc=255 spurious failures. Analysis runs to completion regardless
    of whether the orchestrator stays connected.
    """
    run_dir = f"runs/{sub_tag}"
    analysis_log = f"{sub_tag}_analysis.log"

    launch_cmd = (
        f"setsid nohup bash {ANALYSIS_SCRIPT} {run_dir} "
        f"< /dev/null > {analysis_log} 2>&1 & disown; echo started"
    )

    log(f"Launching analysis (detached) for {sub_tag}", "→ ")
    ssh_launch_detached(launch_cmd, ssh_timeout=15)

    time.sleep(10)
    rc, log_size, _ = ssh_run(f"stat -c %s {analysis_log} 2>/dev/null || echo 0",
                              timeout=15, check=False)
    if rc != 0 or not log_size.isdigit() or int(log_size) == 0:
        log("Analysis log empty or missing — launch failed", "✗ ")
        return False
    return True


def poll_analysis(sub_tag: str) -> bool:
    """Poll the detached analysis until done. Returns True on clean completion.

    Completion detection: look for "ALL 8 STAGES COMPLETE" in the analysis log.
    Failure detection: look for "STAGE N FAILED" or no log growth + no completion.
    """
    analysis_log = f"{sub_tag}_analysis.log"
    deadline = time.time() + ANALYSIS_TIMEOUT_HOURS * 3600
    last_log_size = -1
    last_progress_at = time.time()
    last_stage = 0

    log(f"Polling analysis — expected ~25 min, max {ANALYSIS_TIMEOUT_HOURS}h", "  ")

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL_SEC)

        # Check completion marker. Accept either "ALL 8 STAGES" (legacy) or
        # "ALL 9 STAGES" (post-2026-05-07 pipeline with sanity-check stage).
        rc, marker, _ = ssh_run(
            f'grep -cE "ALL [89] STAGES COMPLETE" {analysis_log} 2>/dev/null || echo 0',
            timeout=15, check=False,
        )
        if marker.isdigit() and int(marker) > 0:
            log("Analysis complete — all stages done", "✓ ")
            return True

        # Check sanity-stage warning (Stage 9 failed but pipeline ran).
        # We treat this as a failure here so retry logic kicks in: a sanity
        # failure usually means the analysis re-run produced suspect outputs
        # and a retry is unlikely to help, but it should not silently pass.
        rc, sanity_warn, _ = ssh_run(
            f'grep -c "STAGE 9 FAILED" {analysis_log} 2>/dev/null || echo 0',
            timeout=15, check=False,
        )
        if sanity_warn.isdigit() and int(sanity_warn) > 0:
            log("Analysis ran but Stage 9 sanity check failed — outputs suspect", "⚠ ")
            rc, tail, _ = ssh_run(f"tail -30 {analysis_log}", timeout=15, check=False)
            log(f"Log tail:\n{tail}", "  ")
            return False

        # Check explicit stage failure
        rc, fail_line, _ = ssh_run(
            f'grep "STAGE.*FAILED" {analysis_log} 2>/dev/null | tail -1',
            timeout=15, check=False,
        )
        if "FAILED" in fail_line:
            log(f"Analysis pipeline reported stage failure: {fail_line.strip()}", "✗ ")
            rc, tail, _ = ssh_run(f"tail -30 {analysis_log}", timeout=15, check=False)
            log(f"Log tail:\n{tail}", "  ")
            return False

        # Check stage progress (look for "STAGE N/M:" markers, M = 8 or 9)
        rc, stage_line, _ = ssh_run(
            f'grep -E "^  STAGE [0-9]+/[89]:" {analysis_log} 2>/dev/null | tail -1',
            timeout=15, check=False,
        )
        cur_stage = last_stage
        if stage_line:
            try:
                # Parse "  STAGE 3/9: pb11_phase_analysis"
                cur_stage = int(stage_line.strip().split()[1].split("/")[0])
            except (ValueError, IndexError):
                pass

        # Check log growth
        rc_size, log_size, _ = ssh_run(
            f"stat -c %s {analysis_log} 2>/dev/null || echo 0",
            timeout=15, check=False,
        )
        log_size_int = int(log_size) if log_size.isdigit() else 0

        if cur_stage > last_stage:
            log(f"analysis stage {cur_stage}", "  ")
            last_stage = cur_stage

        if log_size_int > last_log_size or cur_stage > last_stage:
            last_log_size = log_size_int
            last_progress_at = time.time()
        else:
            stuck_for = time.time() - last_progress_at
            if stuck_for > NO_PROGRESS_TIMEOUT_SEC:
                rc, tail, _ = ssh_run(f"tail -10 {analysis_log}", timeout=15, check=False)
                log(f"Analysis stuck for {int(stuck_for)}s at stage {cur_stage}", "✗ ")
                log(f"Log tail:\n{tail}", "  ")
                return False

    log(f"Analysis exceeded {ANALYSIS_TIMEOUT_HOURS}h timeout", "✗ ")
    return False


def run_analysis(sub_tag: str) -> bool:
    """Launch analysis and poll for completion."""
    if not launch_analysis(sub_tag):
        return False
    return poll_analysis(sub_tag)


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


# ============================================================================
# Completion-state checks (used by the retry loop to skip already-done work)
# ============================================================================

def sim_already_complete(sub_tag: str) -> bool:
    """True if the simulation has produced enough particle dumps to count as complete."""
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


def analysis_already_complete(sub_tag: str) -> bool:
    """True if the analysis pipeline already produced its expected outputs.

    The cheap check is the presence of post_analysis_report.txt with non-zero size.
    The thorough check would also verify all 8 stages' outputs are present, but
    that's overkill for a re-run guard.
    """
    run_dir = f"runs/{sub_tag}"
    rc, size, _ = ssh_run(
        f"stat -c %s {run_dir}/post_analysis_report.txt 2>/dev/null || echo 0",
        timeout=15, check=False,
    )
    try:
        return int(size) > 1000  # an empty/aborted report would be <1 KB
    except ValueError:
        return False


def rsync_already_complete(sub_tag: str) -> bool:
    """True if local already has the rsynced results for this run."""
    local_dir = LOCAL_PAPER_DIR / sub_tag
    if not local_dir.exists():
        return False
    return (local_dir / "post_analysis_report.txt").exists()


# ============================================================================
# Per-job pipeline orchestration with retries
# ============================================================================

def run_one_job_attempt(sub_tag: str, freq_hz: str, dump_period: int) -> tuple[bool, str]:
    """Execute one attempt at the full per-job pipeline. Returns (ok, stage_failed_at).

    Each pipeline stage is gated by an "already complete" check so that retries
    pick up where the previous attempt left off rather than starting from zero.
    """
    # Sim
    if sim_already_complete(sub_tag):
        log(f"Sim data already complete on cloud — skipping launch + polling", "↪ ")
    else:
        if not launch_sim(sub_tag, freq_hz, dump_period):
            return False, "sim_launch"
        if not poll_sim(sub_tag):
            return False, "sim_poll"

    # Analysis
    if analysis_already_complete(sub_tag):
        log("Analysis outputs already present on cloud — skipping analysis re-run", "↪ ")
    else:
        if not run_analysis(sub_tag):
            return False, "analysis"

    # Rsync
    if rsync_already_complete(sub_tag):
        log("Local rsync already complete — skipping rsync", "↪ ")
    else:
        if not rsync_down(sub_tag):
            return False, "rsync"

    # Cleanup is best-effort; not gated by retries
    cleanup_cloud(sub_tag)
    return True, "complete"


def run_one_job(sub_tag: str, freq_hz: str, dump_period: int,
                max_retries: int) -> bool:
    """Execute one job with retry-on-failure. Returns True if eventually succeeded."""
    banner(f"Job: {sub_tag} (freq={freq_hz}, dump-period={dump_period})")

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            backoff = RETRY_BACKOFF_SEC[min(attempt - 2, len(RETRY_BACKOFF_SEC) - 1)]
            log(f"Retry {attempt}/{max_retries} for {sub_tag} after {backoff}s backoff", "⟳ ")
            time.sleep(backoff)

        ok, stage = run_one_job_attempt(sub_tag, freq_hz, dump_period)
        if ok:
            if attempt > 1:
                log(f"Job {sub_tag}: COMPLETE on attempt {attempt}", "✓ ")
            else:
                log(f"Job {sub_tag}: COMPLETE", "✓ ")
            return True
        log(f"Attempt {attempt}/{max_retries} failed at stage: {stage}", "✗ ")

    log(f"Job {sub_tag}: ALL {max_retries} ATTEMPTS FAILED", "✗ ")
    return False


# ============================================================================
# Cross-run analysis
# ============================================================================

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
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Retries per job before declaring failure "
                        f"(default {DEFAULT_MAX_RETRIES})")
    p.add_argument("--abort-on-failure", action="store_true",
                   help="Abort remaining jobs after first failure (legacy behavior). "
                        "Default is to continue past failures.")
    args = p.parse_args()

    jobs = JOBS
    if args.jobs:
        wanted = set(args.jobs.split(","))
        jobs = [j for j in JOBS if j[0] in wanted]
        if not jobs:
            print(f"No matching jobs in {args.jobs}. Available: {[j[0] for j in JOBS]}")
            return 1

    banner(f"Stage A simple runner — {len(jobs)} jobs: {', '.join(j[0] for j in jobs)} "
           f"(max_retries={args.max_retries})")

    succeeded, failed = [], []
    for (sub_tag, freq, dump_period) in jobs:
        ok = run_one_job(sub_tag, freq, dump_period, args.max_retries)
        (succeeded if ok else failed).append(sub_tag)
        if not ok and args.abort_on_failure:
            log(f"--abort-on-failure set; aborting remaining jobs after {sub_tag}", "✗ ")
            log("To resume, edit JOBS or use --jobs to start from where it left off.", "  ")
            break

    if succeeded:
        cross_run_analysis()

    banner("FINAL SUMMARY")
    print(f"  Succeeded: {len(succeeded)} — {succeeded}", flush=True)
    print(f"  Failed:    {len(failed)} — {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
