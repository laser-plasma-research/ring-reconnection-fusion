#!/usr/bin/env python3
"""
sanity_check_run.py — Lightweight smoke-test for a completed analysis run.

Validates that an analysis pipeline output directory contains numerically
plausible results. Designed to be wired in as the final stage of
run_full_analysis.sh, catching obviously-wrong outputs before they
propagate into cross-run aggregation or papers.

Checks performed:

  1. post_analysis_report.txt exists and has non-trivial size (> 1 KB)
  2. The "Maximum core/spot" line is present and parses as a number
  3. All E95 values in the zone table are in the plausible keV range
     (this catches eV/keV mismatches at the source)
  4. Times are monotonically non-decreasing
  5. fusion_rate_power_by_iter.csv exists with expected columns
  6. cum_alpha_yield_p11b is non-negative and monotonically non-decreasing

Returns:
    0 — all checks passed
    1 — usage error (missing args, dir doesn't exist)
    2 — one or more sanity checks failed (analysis output is suspect)

Usage:
    python tools/sanity_check_run.py runs/p2_ghz_36
    python tools/sanity_check_run.py runs/p2_ghz_36 --verbose
"""

import argparse
import os
import sys
from pathlib import Path

# Sanity bounds (keV). Loosely derived from physics — the simulation should
# never produce values far outside these.
E95_MIN_PLAUSIBLE_KEV = 0.05      # 50 eV — colder than thermal would mean a bug
E95_MAX_PLAUSIBLE_KEV = 50000.0   # 50 MeV — anything higher is unit confusion
ALPHA_MAX_PLAUSIBLE = 1e20        # cum_alpha values above this would be unphysical


def check_post_analysis_report(run_dir: Path, verbose: bool) -> list:
    """Return a list of failure-message strings (empty if all checks pass)."""
    failures = []
    report = run_dir / "post_analysis_report.txt"

    if not report.exists():
        # Try zone_report.txt as fallback
        report = run_dir / "zone_report.txt"
        if not report.exists():
            failures.append(f"neither post_analysis_report.txt nor zone_report.txt found in {run_dir}")
            return failures
        if verbose:
            print(f"  using fallback {report.name}")

    size = report.stat().st_size
    if size < 1000:
        failures.append(f"{report.name} is suspiciously small ({size} bytes < 1 KB)")
        return failures
    if verbose:
        print(f"  {report.name} OK ({size} bytes)")

    # Parse the zone table and validate
    e95_count = 0
    e95_oor = 0  # out of range
    e95_min_seen = float("inf")
    e95_max_seen = float("-inf")
    times = []
    found_max_core_spot = False

    with open(report) as f:
        in_table = False
        for line in f:
            if "Maximum core/spot" in line:
                found_max_core_spot = True
                # Try to parse the number
                # Expected format: "  Maximum core/spot:        1.730     at t = 14.97 ps"
                tokens = line.split()
                try:
                    # Find token that's "core/spot:" then take the next
                    for i, tok in enumerate(tokens):
                        if tok.endswith("core/spot:"):
                            float(tokens[i + 1])
                            break
                except (ValueError, IndexError):
                    failures.append("Maximum core/spot line found but value didn't parse")

            if "core/spot" in line and "xline/spot" in line:
                in_table = True
                continue
            if not in_table:
                continue
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("="):
                continue
            if stripped == "" or "TREND" in line or "PEAK" in line:
                in_table = False
                continue

            parts = line.split("|")
            if len(parts) != 3:
                continue
            try:
                left = parts[0].split()
                if len(left) < 1:
                    continue
                t_ps = float(left[0])
                times.append(t_ps)

                e95s = parts[1].split()
                if len(e95s) >= 4:
                    for v_str in e95s[:4]:
                        v = float(v_str)
                        e95_count += 1
                        e95_min_seen = min(e95_min_seen, v) if v > 0 else e95_min_seen
                        e95_max_seen = max(e95_max_seen, v)
                        if v > 0 and not (E95_MIN_PLAUSIBLE_KEV <= v <= E95_MAX_PLAUSIBLE_KEV):
                            e95_oor += 1
            except (ValueError, IndexError):
                continue

    if not found_max_core_spot:
        failures.append("'Maximum core/spot' line not found in report")

    if e95_count == 0:
        failures.append("no E95 values found in zone table")
    elif e95_oor > 0:
        failures.append(
            f"{e95_oor}/{e95_count} E95 values out of plausible range "
            f"[{E95_MIN_PLAUSIBLE_KEV}, {E95_MAX_PLAUSIBLE_KEV}] keV "
            f"(min seen: {e95_min_seen:.3g}, max seen: {e95_max_seen:.3g}). "
            f"This is the classic eV/keV unit mix-up — check whether the "
            f"report writer is dividing by 1000 correctly."
        )
    elif verbose:
        print(f"  {e95_count} E95 values OK, range [{e95_min_seen:.1f}, {e95_max_seen:.1f}] keV")

    # Monotonicity of times
    if times:
        violations = sum(1 for i in range(1, len(times)) if times[i] < times[i - 1])
        if violations > 0:
            failures.append(f"timestamps non-monotonic: {violations} backward jumps in {len(times)} samples")
        elif verbose:
            print(f"  {len(times)} timestamps monotonically non-decreasing OK")

    return failures


def check_fusion_csv(run_dir: Path, verbose: bool) -> list:
    """Validate fusion_rate_power_by_iter.csv."""
    failures = []
    csv = run_dir / "fusion_rate_power_by_iter.csv"
    if not csv.exists():
        failures.append("fusion_rate_power_by_iter.csv not found")
        return failures

    with open(csv) as f:
        lines = f.readlines()

    if len(lines) < 2:
        failures.append("fusion CSV has no data rows")
        return failures

    header = [h.strip() for h in lines[0].strip().split(",")]
    required = ["time_ns", "cum_alpha_yield_p11b"]
    for col in required:
        if col not in header:
            failures.append(f"fusion CSV missing required column: {col}")
            return failures

    idx_t = header.index("time_ns")
    idx_a = header.index("cum_alpha_yield_p11b")

    last_t = -float("inf")
    last_a = -float("inf")
    t_violations = 0
    a_violations = 0
    a_max_seen = 0.0
    n_rows = 0
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) <= max(idx_t, idx_a):
            continue
        try:
            t = float(parts[idx_t])
            a = float(parts[idx_a])
        except ValueError:
            continue
        n_rows += 1
        if t < last_t:
            t_violations += 1
        if a < last_a - 1e-6:  # tiny float-noise tolerance
            a_violations += 1
        last_t = t
        last_a = a
        a_max_seen = max(a_max_seen, a)

    if n_rows == 0:
        failures.append("fusion CSV had header but no parseable rows")
    if t_violations > 0:
        failures.append(f"fusion CSV time_ns non-monotonic: {t_violations} backward jumps")
    if a_violations > 0:
        failures.append(
            f"fusion CSV cum_alpha_yield_p11b non-monotonic: {a_violations} backward jumps "
            f"(cumulative quantities should never decrease)"
        )
    if a_max_seen > ALPHA_MAX_PLAUSIBLE:
        failures.append(
            f"fusion CSV cum_alpha_yield_p11b reaches {a_max_seen:.3e}, "
            f"which exceeds the {ALPHA_MAX_PLAUSIBLE:.0e} sanity ceiling — "
            f"likely a units or accumulation bug"
        )
    if a_max_seen < 0:
        failures.append(f"fusion CSV cum_alpha_yield_p11b is negative ({a_max_seen:.3e})")
    if not failures and verbose:
        print(f"  fusion CSV OK ({n_rows} rows, max cum_alpha={a_max_seen:.3e})")

    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="Path to the run directory to validate")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-check status, not just failures")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"Sanity-checking {run_dir}", flush=True)

    all_failures = []
    all_failures.extend(check_post_analysis_report(run_dir, args.verbose))
    all_failures.extend(check_fusion_csv(run_dir, args.verbose))

    if all_failures:
        print(f"\n  ✗ {len(all_failures)} CHECK(S) FAILED:", flush=True)
        for f in all_failures:
            print(f"    - {f}", flush=True)
        return 2

    print(f"  ✓ All sanity checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
