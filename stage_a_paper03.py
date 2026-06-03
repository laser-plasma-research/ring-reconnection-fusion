#!/usr/bin/env python3
"""
stage_a_paper03.py - Paper 3 multi-zone target campaign runner (unified v0.8).

v0.8 (2026-05-17): Option G geometry fix. Outer annulus moved from
3.0 mm to 3.7 mm (center), giving 600 µm clean buffer past spot 2σ tail.
Eliminates 4.2% direct laser deposit into the catcher that contaminated
v0.7 outer/rod_plus_outer results. Paper 1 anchors (R=2.4 mm, σ=300 µm)
unchanged. Only one config line touched: OUTER_GEOMETRY (line 164).
Jobs affected by re-launch: p3_outer_only_p11b_uuf, p3_outer_only_p7li_uuf,
p3_rod_plus_outer_p11b_uuf, p3_rod_plus_outer_p7li_uuf,
p3_rod_plus_outer_p11b_512, p3_rod_plus_outer_p7li_512 (six total).
Jobs unaffected: p3_ring_only_uuf, p3_rod_only_uuf, p3_rod_sweep_5e25_uuf,
p3_ring_only_512 (no annulus, no overlap with spots).

A focused per-paper orchestrator that runs the full Paper 3 campaign:
ultraultrafine factorial decomposition, rod-density linearity check, and
512² convergence partners. Consolidates the previous stage_a_paper03.py +
stage_a_paper3_uuf.py + retired rod-density sweep orchestrator into one
file with explicit tier metadata.

    launch sim -> poll log for completion ->
       run_full_analysis.sh ->
       pb11_first_transit_fusion.py --reaction p11b ->
       pb11_first_transit_fusion.py --reaction p7li (optional) ->
       pb11_zone_analysis_paper3.py ->
    rsync down -> cleanup -> next

PAPER 3 OBJECTIVES (v0.7)
-------------------------
Demonstrate additive multi-zone fusion gain via:
  * 8-spot ring reconnection geometry as the shared base (matches Paper 1).
  * Geometrically matched central rod (R=400 µm, sized to inflow convergence).
  * Outer catcher placed OUTSIDE the laser ring (R=3700 µm, t=200 µm).
    Option G geometry: annulus inner edge at 3.6 mm gives a 600 µm clean
    buffer past the spot 2σ tail at 3.0 mm. Cf. v0.7 used R=3000 µm
    which placed the annulus inner edge at 2.9 mm — 100 µm INSIDE the
    spot 2σ tail, causing ~4.2% of laser energy to deposit directly
    into the catcher via the spot footprint rather than via reconnection.
  * Factorial decomposition (ring / rod / outer / hybrid) at 1500 steps × 15.4 ps cadence.
  * Single-fuel (all p-¹¹B) vs hybrid (p-¹¹B rod + p-⁷Li outer) comparison.
  * 2-point rod-density linearity check (5e24 from Tier 1 + 5e25 in Tier 2).
  * 512² grid-convergence partners for the three headline configs.
  * Spatially-resolved per-particle target-density accounting (the v0.7
    methodology — requires the SR patch to pb11_first_transit_fusion.py
    before the gain numbers are correct; see "PRE-CAMPAIGN GATE" below).

WHAT'S DROPPED FROM PREVIOUS VERSIONS
--------------------------------------
  * Tier 0 preflight  — UUF Tier 1 runs are also fast (50 min) and validate
                        more of the pipeline. No additional smoke step needed.
  * Tier 2 production — was 6 × 18000-step long-time runs. v0.7 methodology
                        terminates first-transit accumulation at t_cut
                        (B-collapse > 50% time, typically 60-200 ps), so
                        simulating past 1.16 ns gives no additional gain
                        integration. The thermal-equilibration regime past
                        t_cut is descriptive only and is already documented
                        by Paper 1's existing long-time data at the
                        ring-only baseline.
  * Rod density sweep — was 4 points (5e23 / 5e24 / 5e25 / 5e26). v0.7 uses
                        2-point linearity check instead: if linear scaling
                        holds, 2 points suffice to demonstrate; if it breaks,
                        4 points still wouldn't tell you why. The 2 points
                        are 5e24 (Tier 1 rod_only) and 5e25 (Tier 2 sweep).
  * Old 'ultrafine' set (dump-period 33 / 25.5 ps cadence) — strictly
                        coarser than UUF (dump-period 20 / 15.4 ps cadence).
                        Redundant; UUF is kept.

TIER LAYOUT (11 jobs total, ~20 hours unattended)
-------------------------------------------------
  Tier 1  (6 jobs, ~5 hr)    UUF factorial @ 1500 steps × dp=20 (15.4 ps)
                              ring_only, rod_only, outer_only_{p11b,p7li},
                              rod_plus_outer_{p11b,p7li}
  Tier 2  (1 job,  ~50 min)  Rod density linearity, "high" point @ 5e25
                              Combine with p3_rod_only_uuf (5e24, Tier 1)
                              for the 2-point sweep in the manuscript.
  Tier 3  (4 jobs, ~18 hr)   512² convergence @ 7700 steps × dp=100 (15.4 ps)
                              ring_only, rod_plus_outer_p11b, rod_plus_outer_p7li
                              MATCHES Tier 1's 1.16 ns physical duration exactly
                              so first-transit gain comparisons are apples-to-
                              apples. Production DT required at 512² (see
                              detailed Tier 3 comment in JOBS section).

PRE-CAMPAIGN GATE (READ BEFORE DISPATCHING)
-------------------------------------------
Paper 3 v0.7 requires the spatially-resolved per-particle density lookup
patch to be applied to pb11_first_transit_fusion.py FIRST. Without it,
Stage B reports incomplete (zone-only) gain numbers — particles in the
CORE/INTER/X_LINE/SPOT regions are uncounted even though they ARE fusing
against base_n_b11 = 5e24. The reconnection-accelerated fast protons
concentrate at the X-line annulus (NOT inside the rod), so the missing
contribution is potentially LARGER than the rod contribution itself.

This orchestrator is structurally ready for dispatch — it just calls
run_paper3_chain.sh which will be patched separately to use --mode global
with spatially-resolved density. Once that's in place, all 11 jobs here
will produce SR-correct gain numbers.

If you dispatch this orchestrator BEFORE the SR patch lands:
  - Sim outputs (run_meta, reconnection_summary, etc.) will be correct.
  - Stage A standard analysis will be correct.
  - Stage B first-transit gain will be INCOMPLETE (zone-only methodology).
  - Stage D zone-analysis E95 / per-zone rates will be correct.
You can re-run Stage B alone after the patch by manually re-invoking the
analysis chain against the on-cloud particle data. NOTE (2026-05-14 v0.7.2):
cleanup_cloud() no longer auto-runs after each job — the most recently
completed run's particles+fields persist on the cloud until the NEXT
job's clean_pre_attempt() reclaims their disk space. Cheapest path: wait
for the SR patch, then re-dispatch the Stage B analyzer (no re-sim needed).

JOB ORDERING RATIONALE (fail-fast)
-----------------------------------
Within each tier, cheapest validation runs first, longest convergence
last. Across tiers, the order is:
  Tier 1 → Tier 2 → Tier 3
so a fast factorial failure shows up in <1 hour, and the slow 512²
convergence partners only run after Tier 1 + 2 have succeeded.

RETRY POLICY
------------
On failure: clean nuke of remote runs/<sub_tag>/ and local <sub_tag>/,
then retry from scratch. No "resume from where we left off" partial-
state recovery — each attempt starts with a guaranteed-clean data space.

USAGE
-----
    python3 stage_a_paper03.py                       # all 11 jobs
    python3 stage_a_paper03.py --tier 1              # just UUF factorial
    python3 stage_a_paper03.py --tier 1,2            # UUF + density sweep
    python3 stage_a_paper03.py --tier 3              # just 512² convergence
    python3 stage_a_paper03.py --jobs p3_ring_only_uuf,p3_rod_only_uuf
    python3 stage_a_paper03.py --tier 1 --max-retries 5
    python3 stage_a_paper03.py --abort-on-failure    # legacy behaviour

Detached (recommended for long campaigns):
    nohup python3 -u stage_a_paper03.py --tier 1 > stage_a_paper3_tier1.log 2>&1 &
    disown

DENSITY MODEL (apples-to-apples extension of Paper 1)
-----------------------------------------------------
  - Base plasma:    ch_bn at 5e24 (matches Paper 1 LD: n_H=8.15e24,
                                    n_B11=5e24, n_Bheavy=1.185e25, β=13.7).
  - Rod insert:     p11b at 5e24 default (5e25 in Tier 2 sweep) on top of base.
  - Outer insert:   p11b or p7li at 5e24 (single-fuel or hybrid) on top of base.
  - Catcher zones therefore have ADDITIVE density. The spatially-resolved
    first-transit script (post-SR-patch) correctly uses base everywhere,
    base+rod_addl inside rod, base+outer_addl inside outer.
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
CLOUD_ROOT = "~/laser-plasma-research"
CONDA_INIT = "source ~/miniforge3/etc/profile.d/conda.sh && conda activate plasma"

SIM_SCRIPT = "simulation/pb11_ring_reconnection_v15_pulsed.py"
ANALYSIS_SCRIPT = "analysis_scripts/run_full_analysis.sh"
FIRST_TRANSIT_SCRIPT = "analysis_scripts/pb11_first_transit_fusion.py"
ZONE_ANALYSIS_SCRIPT = "analysis_scripts/pb11_zone_analysis_paper3.py"

LOCAL_RUNS      = Path.home() / "LaserFusionResearch/research/laser-plasma-research/runs"
LOCAL_PAPER_DIR = LOCAL_RUNS / "paper03"

# ── Geometry / fuel building blocks ─────────────────────────────────────────
COMMON_GEOMETRY = (
    "--base-fuel ch_bn --base-density 5e24 "
    "--ring-radius-um 2400 --spot-radius-um 300 --n-spots 8 --b-seed 85"
)
ROD_GEOMETRY = "--rod-radius-um 400 --fuel-rod"
OUTER_GEOMETRY = "--outer-radius-um 3700 --outer-thickness-um 200 --fuel-ring-full"

DIAG_PRODUCTION = "--diag-profile production"


# ============================================================================
# Job definitions — grouped by tier
# ============================================================================
# Tuple format (preserved from previous orchestrators for chain compatibility):
#   (sub_tag, freq_hz, dump_period, max_steps, script_id, custom_flags)
#
# Paper 3 is fully static (no rotation), so freq_hz="0" everywhere.
# script_id is informational — only "v15" is currently supported and a
# warning fires if anything else is passed.


# ── Tier 1: UUF factorial @ 1500 steps × dp=20 (15.4 ps cadence) ────────────
# 1.16 ns simulated time, ~75 dumps. Six configs cover the full factorial
# decomposition needed for the multi-zone synergy claim:
#
#   ring_only           — baseline; pure ring reconnection, no inserts
#                          (scientifically same setup as Paper 1 LD)
#   rod_only            — adds rod insert only (no outer catcher)
#   outer_only_p11b     — adds p-¹¹B outer catcher only (no rod)
#   outer_only_p7li     — adds p-⁷Li outer catcher only (no rod)
#   rod_plus_outer_p11b — single-fuel multi-zone (p-¹¹B rod + p-¹¹B outer)
#   rod_plus_outer_p7li — HYBRID HEADLINE: p-¹¹B rod + p-⁷Li outer catcher
#
# All six run independently to support the additive-gain decomposition.
# Stage C of the analysis chain runs only when "p7li" appears in sub_tag.
#
# Cadence rationale: at dt=0.772 ps, dump_period=20 gives 15.4 ps cadence.
# Fine enough to resolve the ~22-28 GHz natural reconnection cycle (35 ps
# period) that Paper 1 identified — Nyquist-safe with ~2× oversampling.
#
# Expected runtime per job: ~50 min (≈5 min sim + ≈45 min analysis chain)
# Disk per run: ~45 GB peak (auto-cleaned after rsync)
TIER1_UUF_FACTORIAL = [
    ("p3_ring_only_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),

    ("p3_rod_only_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"--rod-fuel p11b --rod-density 5e24 {ROD_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),

    ("p3_outer_only_p11b_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"--ring-fuel p11b --ring-density 5e24 {OUTER_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),

    ("p3_outer_only_p7li_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"--ring-fuel p7li --ring-density 5e24 {OUTER_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),

    ("p3_rod_plus_outer_p11b_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"--rod-fuel p11b --rod-density 5e24 {ROD_GEOMETRY} "
     f"--ring-fuel p11b --ring-density 5e24 {OUTER_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),

    ("p3_rod_plus_outer_p7li_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"--rod-fuel p11b --rod-density 5e24 {ROD_GEOMETRY} "
     f"--ring-fuel p7li --ring-density 5e24 {OUTER_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),
]


# ── Tier 2: Rod-density linearity check, "high" point @ 5e25 ───────────────
# 1 dedicated job. Combined with p3_rod_only_uuf (rod_density=5e24, Tier 1)
# this gives a 2-point sweep across the rod-density range that bracket the
# physically interesting regime — enough to demonstrate whether the linear-
# scaling assumption holds, without committing 4 jobs to redundant data
# that wouldn't answer "why" if linearity breaks.
#
# Same geometry as p3_rod_only_uuf except for rod-density=5e25 (10× the
# Tier 1 default). Manuscript reports gain at 5e24 vs 5e25 and tests for
# linear ∝ rod_n_addl scaling.
TIER2_ROD_DENSITY_SWEEP = [
    ("p3_rod_sweep_5e25_uuf", "0", 20, 1500, "v15",
     f"--test {COMMON_GEOMETRY} "
     f"--rod-fuel p11b --rod-density 5e25 {ROD_GEOMETRY} "
     f"{DIAG_PRODUCTION}"),
]


# ── Tier 3: 512² convergence partners @ 7700 steps × dp=100 (15.4 ps) ───────
# Four configs covering the headline data points: ring-only baseline, the
# catcher-fuel comparison (outer-only p7li), and the two multi-zone configs
# (single-fuel and hybrid). The outer-only-p7li_512 entry pairs with
# p3_rod_plus_outer_p11b_512 at the same resolution to isolate the
# catcher-fuel choice effect from the rod-presence effect.
#
# LESSONS BAKED IN FROM PAPER 1 LD 512² (2026-05-13 v0.7.1 debug saga):
# ---------------------------------------------------------------------
#
# 1. --test FLAG DROPPED. The --test flag sets DT=1e-3 which is ~5x larger
#    than production's DT=1.94e-4 (and at 512² resolves empirically to
#    ~0.15 ps per step). At 512² the cell size halves so the CFL-limited
#    dt also halves, making --test DT roughly 10x over the stability
#    limit. Empirically observed failure mode: no E-field NaN crash,
#    but the field solver effectively skips work, B never collapses,
#    no reconnection event develops. Production DT (when --test absent)
#    plus --substeps 160 is the correct combination.
#
# 2. --nppc 200 EXPLICIT ON p11b CONFIGS. Without --test present, the
#    NPPC default switches to "production=400" per the v15 --help. At
#    512² with proton+B11 species, NPPC=400 pushes total particle
#    storage past the 80 GB H100 envelope. The p7li config explicitly
#    sets --nppc 100 (further reduced because Li-7 adds a third species).
#
# 3. max_steps=7700 (was 2000) at production DT covers 7700 × 0.15 ps =
#    1155 ps. This matches Tier 1's --test-DT 1500 × 0.77 ps = 1158 ps
#    almost exactly. WITHOUT this bump, Tier 3 would simulate only
#    300 ps — well short of Tier 1's 1.16 ns and before the first-
#    transit accumulation window closes, breaking apples-to-apples
#    gain comparison. Still safely below the empirically-observed
#    hybrid-PIC instability ceiling at t≈1.898 ns at 512².
#
# 4. dump_period=100 (was 50) at the new step count gives 77 dumps at
#    15.4 ps cadence — identical to Tier 1. Stage B runtime scales with
#    dump count, not step count, so this keeps analysis cost flat
#    relative to Tier 1 jobs.
#
# CLOUD-SIDE PREREQUISITE (CHECK BEFORE DISPATCHING):
# ---------------------------------------------------
# The production diag-profile's LATE_PERIOD_OVERRIDE must be > 7700
# (Paper 1's hotfix was 99999). With the default of 5000, the v15
# script's late_diag callback at step 5000 invokes deref('proton')
# which materializes ~100M particles in Python memory on top of an
# already-91%-full H100, triggering an OOM hang at step 5000 exactly.
# Verify with:
#   ssh gpu-node 'grep "LATE_PERIOD_OVERRIDE   = " \
#     ~/laser-plasma-research/simulation/pb11_ring_reconnection_v15_pulsed.py'
# The production branch must show LATE_PERIOD_OVERRIDE = 99999 (or higher).
#
# METHODOLOGY NOTE FOR THE MANUSCRIPT:
# ------------------------------------
# Tier 1 (256²) and Tier 3 (512²) use different DT (--test 0.77 ps vs
# production 0.15 ps) because of the CFL constraint at 512². This is NOT
# a pure grid-only convergence check — it is "matched-physics-at-each-
# resolution": both runs use the most accurate DT their grid allows over
# the same 1.16 ns physical duration. The convergence question is whether
# the headline observables (B-collapse fraction, t_cut location, E95 peak
# time, G_FT) are stable across the resolution change; Paper 1 v0.7.1
# showed B_initial, B_min, B_collapse_pct match within 2% at both
# resolutions, so the reconnection physics is grid-converged. Reviewers
# can be referred to the Paper 1 convergence section.
#
# Expected runtime per job: ~4-4.5 hr
#   sim:      ~36 min  (7700 steps × ~0.28 sec/step at 512² production DT)
#   Stage A:  ~20 min  (Stage 9 emits an rc=2 sanity warning treated as soft-OK)
#   Stage B:  ~2.5-3 hr (77 dumps × ~2-3 min/dump; hybrid p7li slightly slower
#                        because Stage C adds an extra reaction pass)
#   Stage D:  ~10 min
#   Rsync + cleanup: ~10 min
# Disk per run: ~1.3 TB peak before cleanup (77 dumps × ~17 GB at 512²)
TIER3_CONVERGENCE_512 = [
    ("p3_ring_only_512", "0", 100, 7700, "v15",
     f"{COMMON_GEOMETRY} "
     f"--nx 512 --nz 512 --nppc 200 --substeps 160 "
     f"{DIAG_PRODUCTION}"),

    # Catcher-fuel comparison partner at 512² resolution. Pair with
    # p3_rod_plus_outer_p11b_512 (p-11B catcher) for like-for-like
    # σ-folded yield comparison across catcher fuel choice without
    # the rod-presence confound. NPPC=100 because Li-7 species adds
    # memory load at 512².
    ("p3_outer_only_p7li_512", "0", 100, 7700, "v15",
     f"{COMMON_GEOMETRY} "
     f"--ring-fuel p7li --ring-density 5e24 {OUTER_GEOMETRY} "
     f"--nx 512 --nz 512 --nppc 100 --substeps 160 "
     f"{DIAG_PRODUCTION}"),

    ("p3_rod_plus_outer_p11b_512", "0", 100, 7700, "v15",
     f"{COMMON_GEOMETRY} "
     f"--rod-fuel p11b --rod-density 5e24 {ROD_GEOMETRY} "
     f"--ring-fuel p11b --ring-density 5e24 {OUTER_GEOMETRY} "
     f"--nx 512 --nz 512 --nppc 200 --substeps 160 "
     f"{DIAG_PRODUCTION}"),

    ("p3_rod_plus_outer_p7li_512", "0", 100, 7700, "v15",
     f"{COMMON_GEOMETRY} "
     f"--rod-fuel p11b --rod-density 5e24 {ROD_GEOMETRY} "
     f"--ring-fuel p7li --ring-density 5e24 {OUTER_GEOMETRY} "
     f"--nx 512 --nz 512 --nppc 100 --substeps 160 "
     f"{DIAG_PRODUCTION}"),
]


# Aggregate all jobs in dispatch order (fail-fast: Tier 1 first, Tier 3 last)
JOBS = TIER1_UUF_FACTORIAL + TIER2_ROD_DENSITY_SWEEP + TIER3_CONVERGENCE_512

# Tier metadata for --tier filtering. Keys are integer tier numbers; values
# are lists of sub_tag strings from the corresponding tier.
TIERS = {
    1: [j[0] for j in TIER1_UUF_FACTORIAL],
    2: [j[0] for j in TIER2_ROD_DENSITY_SWEEP],
    3: [j[0] for j in TIER3_CONVERGENCE_512],
}


POLL_INTERVAL_SEC = 60
SIM_TIMEOUT_HOURS = 5        # generous; UUF takes ~5 min, 512² takes ~36 min
ANALYSIS_TIMEOUT_HOURS = 4   # 512² analysis takes ~3 hr; 4 hr is safety margin
# Bumped from 600 to 1800 (2026-05-13 v0.7.1): Stage B at 512² is legitimately
# slow per dump (77 dumps × 2-3 min each), and the watchdog needs to tolerate
# the gap between dump-process log lines without false-positive aborting.
# Same fix Paper 1's stage_a_paper01.py adopted today.
NO_PROGRESS_TIMEOUT_SEC = 1800

DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SEC = [30, 120, 300]

TRANSIENT_RCS = {124, 255}


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

    Policy change 2026-05-14 v0.7.2: cleanup_cloud() no longer runs after
    each successful job (see run_one_job_attempt). That keeps the most
    recently completed run's particle dumps on disk so they can be
    re-analyzed (e.g. for the f(E) histogram extraction stage that
    consumes particle dumps directly). However, particles from runs
    PRIOR to the most recent one are no longer useful and must be cleared
    before the next sim launches, or disk fills up during multi-job
    campaigns (~1.3 TB per 512^2 run × 3.5 TB total = OOM by job 3).

    This function clears particles+fields from every run dir EXCEPT
    current_sub_tag. The current_sub_tag's directory is wiped wholesale
    by clean_pre_attempt() itself, so we exclude it here to avoid
    fighting that logic.
    """
    cmd = (
        f"for d in runs/*/; do "
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
            rc2, disk, _ = ssh_run("df -h ~/laser-plasma-research | tail -1",
                                    timeout=15, check=False)
            log(f"Cloud disk after reclaim: {disk}", "  ")


def clean_pre_attempt(sub_tag: str) -> None:
    run_dir = f"runs/{sub_tag}"
    log(f"Pre-attempt cleanup: wiping {run_dir} on cloud + local", "  ")
    ssh_run(f"rm -rf {run_dir}", timeout=120, check=False)

    # Free disk from OTHER completed runs (keeps the just-finished one's
    # particles available until the NEXT job starts, enabling post-hoc
    # analysis like f(E) histogram extraction).
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
    run_dir = f"runs/{sub_tag}"
    sim_log = f"{sub_tag}.log"

    flags = (f"{custom_flags} --max-steps {max_steps} "
             f"--dump-period {dump_period} --outdir {run_dir}")

    launch_cmd = (
        f"mkdir -p {run_dir} && "
        f"setsid nohup mpirun -n 8 python -u {SIM_SCRIPT} {flags} "
        f"< /dev/null > {sim_log} 2>&1 & disown; echo started"
    )

    log(f"Launching sim: max_steps={max_steps}, dump_period={dump_period}", "-> ")
    ssh_launch_detached(launch_cmd, ssh_timeout=30)

    # Verify by polling the log size up to 3 times. SSH is sometimes slow to
    # establish the second connection, and the remote bash also takes a
    # moment to flush its first output. A single check at 15s can race with
    # that, so retry a few times with backoff before declaring the launch dead.
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


def launch_analysis(sub_tag: str, has_p7li: bool) -> bool:
    """Launch the Paper 3 analysis chain via run_paper3_chain.sh on cloud.

    Two positional args: run_dir and "yes"|"no" for whether Stage C runs.
    The chain itself is unchanged in this orchestrator update — the SR
    methodology patch (when it lands) modifies the underlying first-transit
    script, not the chain. So this call site does not need to change to
    benefit from the patch.
    """
    run_dir = f"runs/{sub_tag}"
    analysis_log = f"{sub_tag}_analysis.log"
    has_p7li_arg = "yes" if has_p7li else "no"

    launch_cmd = (
        f"setsid nohup bash analysis_scripts/run_paper3_chain.sh "
        f"{run_dir} {has_p7li_arg} "
        f"< /dev/null > {analysis_log} 2>&1 & disown; echo started"
    )

    log(f"Launching analysis chain (detached) for {sub_tag}", "-> ")
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
            f'grep -c "ALL P3 ANALYSIS STAGES COMPLETE" {analysis_log} 2>/dev/null || echo 0',
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
    # has_p7li is derived from the sub_tag: any job whose name contains
    # "p7li" has p-7Li fuel somewhere and needs the Stage C analysis pass.
    has_p7li = "p7li" in sub_tag
    if not launch_analysis(sub_tag, has_p7li):
        return False
    return poll_analysis(sub_tag)


def rsync_down(sub_tag: str) -> bool:
    """Pull results from cloud to local. Retries up to 3 times on transient
    failures (timeout, rc=23/24/30/35 = network/IO errors).
    """
    run_dir = f"runs/{sub_tag}"
    local_dir = LOCAL_PAPER_DIR / sub_tag
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"rsync -avh "
        f"--include='*/' "
        f"--include='*.png' --include='*.mp4' --include='*.txt' "
        f"--include='*.csv' --include='*.npz' "
        f"--exclude='*' "
        f"{CLOUD_HOST}:{CLOUD_ROOT}/{run_dir}/ {local_dir}/"
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
    run_dir = f"runs/{sub_tag}"
    log("Cleaning cloud particles + fields", "-> ")
    ssh_run(f"rm -rf {run_dir}/particles {run_dir}/fields", timeout=60, check=False)
    rc, disk, _ = ssh_run("df -h ~/laser-plasma-research | tail -1",
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
    # Policy change 2026-05-14 v0.7.2: post-run cleanup_cloud() removed.
    # The just-completed run's particles+fields stay on cloud until the
    # NEXT job's clean_pre_attempt() reclaims their disk space. This
    # preserves particle dumps for ad-hoc f(E) histogram re-analysis
    # without forcing a re-simulation. cleanup_cloud() is still defined
    # below and can be invoked manually if disk pressure demands it.
    return True, "complete"


def run_one_job(sub_tag: str, freq_hz: str, dump_period: int,
                max_steps: int, max_retries: int,
                script_id: str, custom_flags: str) -> bool:
    banner(f"Job: {sub_tag} (script={script_id}, "
           f"dump_period={dump_period}, max_steps={max_steps})")

    if script_id != "v15":
        log(f"WARNING: Paper 3 expects script_id=v15, got {script_id}", "! ")

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

def resolve_jobs(jobs_arg: str, tier_arg: str):
    """Resolve --tier and --jobs into a final ordered job list.

    Filters are intersected: if both --tier and --jobs are given, the result
    is the intersection (jobs that are in the named tier AND named explicitly).

    Returns: list of job tuples in dispatch order, or None on error.
    """
    selected_tags = None  # None means "all jobs"

    if tier_arg:
        try:
            tier_ids = [int(t.strip()) for t in tier_arg.split(",")]
        except ValueError:
            print(f"ERROR: --tier expects comma-separated integers, got: {tier_arg}",
                  flush=True)
            return None
        bad = [t for t in tier_ids if t not in TIERS]
        if bad:
            print(f"ERROR: unknown tier(s): {bad}. "
                  f"Available tiers: {sorted(TIERS.keys())}", flush=True)
            return None
        tier_tags = set()
        for t in tier_ids:
            tier_tags.update(TIERS[t])
        selected_tags = tier_tags

    if jobs_arg:
        jobs_tags = set(jobs_arg.split(","))
        if selected_tags is None:
            selected_tags = jobs_tags
        else:
            selected_tags = selected_tags & jobs_tags

    if selected_tags is None:
        return JOBS  # no filter
    out = [j for j in JOBS if j[0] in selected_tags]
    if not out:
        available = [j[0] for j in JOBS]
        print(f"ERROR: no jobs match filters --tier={tier_arg} --jobs={jobs_arg}.",
              flush=True)
        print(f"Available jobs: {available}", flush=True)
        return None
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper 3 multi-zone target campaign runner (unified v0.7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tiers:\n"
               "  1  UUF factorial decomposition (6 jobs, ~5 hr)\n"
               "  2  Rod density linearity check (1 job, ~50 min)\n"
               "  3  512² grid convergence (4 jobs, ~18 hr)",
    )
    p.add_argument("--jobs", default="",
                   help="Comma-separated sub_tags to run "
                        "(intersected with --tier if both given)")
    p.add_argument("--tier", default="",
                   help="Comma-separated tier numbers (1,2,3); "
                        "intersected with --jobs if both given")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Retries per job before declaring failure "
                        f"(default {DEFAULT_MAX_RETRIES})")
    p.add_argument("--abort-on-failure", action="store_true",
                   help="Abort remaining jobs after first failure "
                        "(default: continue past failures)")
    args = p.parse_args()

    jobs = resolve_jobs(args.jobs, args.tier)
    if jobs is None:
        return 1

    tier_filter = f"tier={args.tier} " if args.tier else ""
    jobs_filter = f"jobs={args.jobs} " if args.jobs else ""
    filter_desc = f"({tier_filter}{jobs_filter}max_retries={args.max_retries})"
    banner(f"Paper 3 runner v0.7 - {len(jobs)} jobs: "
           f"{', '.join(j[0] for j in jobs)} {filter_desc}")

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
        print(f"\n  Next step: python3 paper3_summarize.py", flush=True)
        print(f"  This will read all rsync'd runs in {LOCAL_PAPER_DIR}", flush=True)
        print(f"  and produce the master multi-zone decomposition table.\n", flush=True)
        print(f"  REMINDER: gain figures depend on the spatially-resolved", flush=True)
        print(f"  patch to pb11_first_transit_fusion.py landing first.", flush=True)
        print(f"  See the PRE-CAMPAIGN GATE section in this script's docstring.",
              flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
