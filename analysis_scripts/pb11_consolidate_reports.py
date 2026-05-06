#!/usr/bin/env python3
"""
pb11_consolidate_reports.py

Generate the consolidated post_analysis_report.txt and zone_report.txt from
cached data produced by visualize_all.py. This replaces the old
pb11_post_analysis.py + pb11_analyse_rotating.py scripts that were lost.

Reads:
    runs/<run>/figures/cache/zone_timeseries.npz   (per-dump zone particle data)
    runs/<run>/figures/cache/field_timeseries.npz  (per-dump B-field data)
    runs/<run>/run_meta.txt                        (run metadata)
    runs/<run>/fusion_rate_power_by_iter.csv       (inline diagnostic CSV)

Writes:
    runs/<run>/post_analysis_report.txt   (consolidated 7-section report)
    runs/<run>/zone_report.txt            (zone-only re-analysis with geometry header)

Usage:
    python pb11_consolidate_reports.py --dir runs/<run_name>
"""

import argparse
import os
import sys
import csv
import math
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

SEP = "=" * 92
SUB = "-" * 86

P_PB11_THRESHOLD_KEV = 500.0   # p-11B fusion threshold


def read_run_meta(run_dir):
    """Parse run_meta.txt into a dict. Tolerates either 'key=value' or 'key: value'."""
    path = Path(run_dir) / "run_meta.txt"
    meta = {}
    if not path.exists():
        return meta
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            elif ":" in line:
                k, v = line.split(":", 1)
            else:
                continue
            meta[k.strip()] = v.strip()
    return meta


def read_inline_csv(run_dir):
    """Read fusion_rate_power_by_iter.csv. Returns list of dicts or empty list."""
    path = Path(run_dir) / "fusion_rate_power_by_iter.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def safe_float(s, default=float("nan")):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def fmt_num(x, sig=4):
    """Format number compactly."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if abs(x) >= 1e5 or (abs(x) < 1e-2 and x != 0):
        return f"{x:.3e}"
    return f"{x:,.1f}"


def fmt_count(x):
    """Format a particle count (physical, with weights) in scientific notation."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.3e}"


# -----------------------------------------------------------------------------
# Main report builder
# -----------------------------------------------------------------------------

def build_post_analysis_report(run_dir):
    """Build the consolidated post_analysis_report.txt content."""
    run_dir = Path(run_dir)
    run_name = run_dir.name
    cache_dir = run_dir / "figures" / "cache"

    if not (cache_dir / "zone_timeseries.npz").exists():
        sys.exit(f"ERROR: {cache_dir}/zone_timeseries.npz not found. "
                 "Run visualize_all.py first.")

    # ---- Load cached data
    zd = np.load(cache_dir / "zone_timeseries.npz", allow_pickle=True)
    fd_path = cache_dir / "field_timeseries.npz"
    fd = np.load(fd_path, allow_pickle=True) if fd_path.exists() else None

    times_ps = zd["times_ps"]
    n_dumps = len(times_ps)
    t_max_ps = float(times_ps[-1]) if n_dumps else 0.0

    # ---- Read metadata
    meta = read_run_meta(run_dir)
    inline = read_inline_csv(run_dir)

    # ---- Begin report
    lines = []
    lines.append(SEP)
    lines.append(f"  POST-RUN ANALYSIS — {run_dir.as_posix()}")
    lines.append(SEP)
    lines.append("")

    # =============================================================
    # SECTION 1: RUN METADATA
    # =============================================================
    lines.append("  RUN METADATA")
    lines.append("  " + SUB)
    for k in ("version", "timestamp", "Geometry", "Base fuel", "B-seed",
              "Plasma beta", "Current support", "Steps × dt", "Completion"):
        if k in meta:
            lines.append(f"  {k}={meta[k]}")
    # Fall back: dump everything we have if standard keys missing
    if not any(k in meta for k in ("version", "Geometry")):
        for k, v in meta.items():
            lines.append(f"  {k}={v}")
    lines.append(f"  Snapshots:       {n_dumps}")
    lines.append(f"  Time range:      0.00 to {t_max_ps:.2f} ps")
    lines.append("")

    # =============================================================
    # SECTION 2: INLINE DIAGNOSTIC TIME SERIES
    # =============================================================
    lines.append(SEP)
    lines.append("  INLINE DIAGNOSTIC TIME SERIES (fusion_rate_power_by_iter.csv)")
    lines.append(SEP)
    lines.append("  NOTE: this CSV uses the legacy R_CENTRE=200 um diagnostic which is")
    lines.append("        usually wrong for ring geometries. Use ZONE ANALYSIS below for")
    lines.append("        physically meaningful core/X-line/spot zones.")
    lines.append("")

    if inline:
        lines.append(f"  {len(inline)} diagnostic samples in CSV")
        # Get time range
        t_first = safe_float(inline[0].get("t_ps", inline[0].get("time_ps", "0")))
        t_last = safe_float(inline[-1].get("t_ps", inline[-1].get("time_ps", "0")))
        lines.append(f"  Time range: {t_first:.2f} → {t_last:.2f} ps")
        lines.append("")
        lines.append(f"     t(ps)    fast_frac    fusion_rate        E_cum_J       gain")
        lines.append("  " + "-" * 70)

        # Sample evenly: first, ~25%, ~50%, ~75%, last
        n = len(inline)
        sample_idx = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        for i in sample_idx:
            r = inline[i]
            t = safe_float(r.get("t_ps", r.get("time_ps", "0")))
            ff = safe_float(r.get("fast_frac", "nan"))
            rate = safe_float(r.get("fusion_rate", "nan"))
            ecum = safe_float(r.get("E_cum_J", r.get("e_cum_j", "nan")))
            gain = safe_float(r.get("gain", "nan"))
            lines.append(f"   {t:7.2f}   {ff:8.4f}   {rate:12.3e}   {ecum:12.3e}  {gain:.3e}")
        lines.append("")

        # Final summary
        last = inline[-1]
        e_cum_final = safe_float(last.get("E_cum_J", last.get("e_cum_j", "nan")))
        gain_final = safe_float(last.get("gain", "nan"))
        alpha_final = safe_float(last.get("alpha_yield", last.get("alpha_cumulative", "nan")))
        lines.append(f"  FINAL  cumulative fusion energy: {e_cum_final:.3e} J")
        lines.append(f"  FINAL  gain vs 5 J laser:        {gain_final:.3e}")
        if not math.isnan(alpha_final):
            lines.append(f"  FINAL  cumulative alpha yield:   {alpha_final:.3e}")
    else:
        lines.append("  (CSV not found — inline diagnostics unavailable)")
    lines.append("")

    # =============================================================
    # SECTION 3: ZONE-BASED PARTICLE ANALYSIS
    # =============================================================
    lines.append(SEP)
    lines.append("  ZONE-BASED PARTICLE ANALYSIS (geometry-aware)")
    lines.append(SEP)
    lines.append("")
    lines.append("  Zone boundaries:")
    lines.append("    CORE     : r <  1200 um             (50% of ring radius)")
    lines.append("    INNER    : 1200 <= r < 2097 um")
    lines.append("    X-LINE   : 2097 <= r <= 2337 um  (annulus around X-line at r=2217 um)")
    lines.append("    SPOT     : r >= 1800 um  (within 2σ of spots)")
    lines.append("")
    lines.append(f"  Processing {n_dumps} snapshots from {run_dir.as_posix()}/particles...")
    lines.append("")
    lines.append("  Note: N values are PHYSICAL particle counts (using simulation weights).")
    lines.append("        E95 values are 95th-percentile energies in keV.")
    lines.append("")

    # Header for zone table
    hdr = "    t(ps)   N_core      N_inner     N_xline     N_spot   |   E95_core   E95_inner   E95_xline   E95_spot |  core/spot  xline/spot"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) + 8))

    # Fields
    N_core = zd["N_core"]
    N_inner = zd["N_inner"]
    N_xline = zd["N_xline"]
    N_spot = zd["N_spot"]
    E95_core = zd["E95_core"]
    E95_inner = zd["E95_inner"]
    E95_xline = zd["E95_xline"]
    E95_spot = zd["E95_spot"]

    # Compute ratios safely
    eps = 1e-30
    ratio_core_spot = E95_core / (E95_spot + eps)
    ratio_xline_spot = E95_xline / (E95_spot + eps)

    for i in range(n_dumps):
        t = times_ps[i]
        # Convert eV → keV (cache stores eV)
        line = (
            f"   {t:7.2f}  {N_core[i]:.2e}  {N_inner[i]:.2e}  {N_xline[i]:.2e}  "
            f"{N_spot[i]:.2e}   |  {E95_core[i]/1000:9.1f}  {E95_inner[i]/1000:10.1f}  "
            f"{E95_xline[i]/1000:10.1f}  {E95_spot[i]/1000:9.1f} |  "
            f"{ratio_core_spot[i]:8.4f}  {ratio_xline_spot[i]:8.4f}"
        )
        lines.append(line)
    lines.append("")

    # =============================================================
    # SECTION 4: PEAK DETECTION
    # =============================================================
    lines.append(SEP)
    lines.append("  PEAK DETECTION (across full run)")
    lines.append(SEP)

    # Find max E95_core and max core/spot ratio
    e95_core_keV = E95_core / 1000.0
    idx_max_e = int(np.argmax(e95_core_keV))
    idx_max_r = int(np.argmax(ratio_core_spot))

    lines.append(f"  Maximum core E95:       {e95_core_keV[idx_max_e]:,.0f} keV  at t = {times_ps[idx_max_e]:.2f} ps")
    lines.append(f"  Maximum core/spot:        {ratio_core_spot[idx_max_r]:.3f}     at t = {times_ps[idx_max_r]:.2f} ps")

    # Threshold check
    if e95_core_keV[idx_max_e] > P_PB11_THRESHOLD_KEV:
        lines.append(f"  ✓ TRANSIENT FUSION-GRADE CORE: peak E95 = {e95_core_keV[idx_max_e]:,.0f} keV exceeds")
        lines.append(f"    p-11B threshold (500 keV) at t = {times_ps[idx_max_e]:.1f} ps")
    else:
        lines.append(f"  ~ Peak core E95 below p-11B threshold ({e95_core_keV[idx_max_e]:.0f} < 500 keV)")

    if ratio_core_spot[idx_max_r] > 1.5:
        lines.append(f"  ✓ STRONG CORE INVERSION: max ratio = {ratio_core_spot[idx_max_r]:.3f} at t = {times_ps[idx_max_r]:.1f} ps")
    elif ratio_core_spot[idx_max_r] > 1.0:
        lines.append(f"  ~ Mild core inversion: max ratio = {ratio_core_spot[idx_max_r]:.3f} at t = {times_ps[idx_max_r]:.1f} ps")
    else:
        lines.append(f"  ~ No core inversion detected (max ratio < 1.0)")
    lines.append("")

    # =============================================================
    # SECTION 5: TREND ANALYSIS
    # =============================================================
    lines.append(SEP)
    lines.append("  TREND ANALYSIS — early third vs late third")
    lines.append(SEP)
    lines.append("  WARNING: Window averaging may hide transient peaks.")
    lines.append("  See PEAK DETECTION above for true maxima.")
    lines.append("")

    # Use early/late thirds
    third = n_dumps // 3
    if third < 2:
        third = 1
    early_idx = slice(0, third)
    late_idx = slice(n_dumps - third, n_dumps)

    t_early_lo = times_ps[0]
    t_early_hi = times_ps[third - 1] if third > 0 else times_ps[0]
    t_late_lo = times_ps[n_dumps - third]
    t_late_hi = times_ps[-1]

    lines.append(f"  Early window: t = {t_early_lo:.1f} → {t_early_hi:.1f} ps")
    lines.append(f"  Late  window: t = {t_late_lo:.1f} → {t_late_hi:.1f} ps")
    lines.append("")
    lines.append("  Zone          E95 early       E95 late          Δ E95     N early      N late")
    lines.append("  " + "-" * 80)

    zones = [
        ("core", E95_core, N_core),
        ("inner", E95_inner, N_inner),
        ("xline", E95_xline, N_xline),
        ("spot", E95_spot, N_spot),
    ]
    early_avgs = {}
    late_avgs = {}
    for name, e95, n in zones:
        e_early = float(np.mean(e95[early_idx]) / 1000.0)  # eV → keV
        e_late = float(np.mean(e95[late_idx]) / 1000.0)
        n_early = float(np.mean(n[early_idx]))
        n_late = float(np.mean(n[late_idx]))
        delta = e_late - e_early
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {name:5s}        {e_early*1000:11,.1f}      {e_late*1000:11,.1f}    {sign}{delta*1000:11,.1f}   {n_early:.2e}    {n_late:.2e}"
        )
        early_avgs[name] = (e_early, n_early)
        late_avgs[name] = (e_late, n_late)
    lines.append("")

    # =============================================================
    # SECTION 6: RECONNECTION-NARRATIVE DETECTION
    # =============================================================
    lines.append(SEP)
    lines.append("  RECONNECTION-NARRATIVE DETECTION")
    lines.append(SEP)
    lines.append("")

    e_core_early = early_avgs["core"][0]
    e_core_late = late_avgs["core"][0]
    e_xline_early = early_avgs["xline"][0]
    e_xline_late = late_avgs["xline"][0]
    n_core_early = early_avgs["core"][1]
    n_core_late = late_avgs["core"][1]
    e_core_peak = float(np.max(e95_core_keV))
    n_core_peak = float(np.max(N_core))

    # Core warming peak-detected
    if e_core_peak > 500:
        lines.append(f"  ✓ STRONG CORE HEATING (peak-detected):")
    elif e_core_peak > 200:
        lines.append(f"  ~ MODERATE CORE HEATING (peak-detected):")
    else:
        lines.append(f"  ~ Core warming gradually:")
    lines.append(f"    Peak: {e_core_peak:,.0f} keV at t = {times_ps[idx_max_e]:.1f} ps")
    lines.append(f"    Early-window avg: {e_core_early*1000:,.0f} keV")
    lines.append(f"    Late-window avg:  {e_core_late*1000:,.0f} keV")

    # Particle inflow
    if n_core_peak > n_core_early * 1.5:
        lines.append(f"  ✓ PARTICLE INFLOW: N_core peaked at {n_core_peak:.3e} (early avg: {n_core_early:.3e}, late avg: {n_core_late:.3e})")
    else:
        lines.append(f"  ~ Core particle count: {n_core_early:.3e} → {n_core_late:.3e}  (no significant inflow)")

    # X-line acceleration
    e_xline_peak = float(np.max(E95_xline) / 1000.0)
    if e_xline_peak > 200:
        lines.append(f"  ✓ X-LINE ACCELERATION: peak {e_xline_peak:,.0f} keV (early avg: {e_xline_early*1000:,.0f}, late avg: {e_xline_late*1000:,.0f})")
    else:
        lines.append(f"  ~ X-line activity: early avg {e_xline_early*1000:,.0f} keV → late avg {e_xline_late*1000:,.0f} keV")

    # Core/spot inversion
    peak_ratio = ratio_core_spot[idx_max_r]
    if peak_ratio > 1.5:
        lines.append(f"  ✓ STRONG CORE/SPOT INVERSION: peak ratio = {peak_ratio:.3f}× at t = {times_ps[idx_max_r]:.1f} ps (core much hotter than spots)")
    elif peak_ratio > 1.0:
        lines.append(f"  ~ Mild core/spot inversion: peak ratio = {peak_ratio:.3f}× at t = {times_ps[idx_max_r]:.1f} ps")
    else:
        lines.append(f"  ~ core/spot ratio (peak) = {peak_ratio:.2f}× (core cooler than spots)")

    # Fusion threshold
    if e_core_peak > 500:
        ratio_to_threshold = e_core_peak / P_PB11_THRESHOLD_KEV
        lines.append(f"  ✓ FUSION-RELEVANT CORE: peak E95 = {e_core_peak:,.0f} keV  ({ratio_to_threshold:.1f}× p-11B threshold) at t = {times_ps[idx_max_e]:.1f} ps")
    else:
        lines.append(f"  ~ Peak core E95 = {e_core_peak:,.0f} keV (below p-11B 500 keV threshold)")
    lines.append("")

    # =============================================================
    # SECTION 7: B-FIELD EVOLUTION
    # =============================================================
    lines.append(SEP)
    lines.append("  B-FIELD EVOLUTION CHECK")
    lines.append(SEP)
    lines.append("")

    if fd is not None:
        f_times = fd["times_ps"]
        b_max = fd["B_max"]
        b_mean = fd["B_mean"]
        jy_max = fd["Jy_max"]
        n_field = len(f_times)

        lines.append(f"  Field dumps from: {run_dir.as_posix()}/fields")
        lines.append(f"     t(ps)   |B|_max(T)    |B|_mean(T)    |J|_max(A/m²)")
        lines.append("  " + "-" * 60)

        # Sample evenly
        sample_idx = sorted(set([0] + list(range(n_field // 8, n_field, max(1, n_field // 8))) + [n_field - 1]))
        for i in sample_idx[:12]:  # cap at 12 rows
            lines.append(f"   {f_times[i]:8.2f}     {b_max[i]:8.2f}     {b_mean[i]:9.4f}      {jy_max[i]:.3e}")

        # B-collapse
        b_collapse_pct = (1 - b_max[-1] / b_max[0]) * 100 if b_max[0] > 0 else 0
        lines.append("")
        lines.append(f"  B-collapse (peak): {b_collapse_pct:.1f}% (|B|_max: {b_max[0]:.1f}T → {b_max[-1]:.1f}T)")
    else:
        lines.append("  (field_timeseries.npz not found — B-field analysis unavailable)")
    lines.append("")

    lines.append(SEP)
    lines.append("  END POST-RUN ANALYSIS")
    lines.append(SEP)
    lines.append(f"  Report saved to: {run_dir.as_posix()}/post_analysis_report.txt")

    return "\n".join(lines) + "\n"


def build_zone_report(run_dir):
    """Build the zone_report.txt content (zone-only re-analysis with geometry header)."""
    run_dir = Path(run_dir)
    cache_dir = run_dir / "figures" / "cache"

    if not (cache_dir / "zone_timeseries.npz").exists():
        sys.exit(f"ERROR: {cache_dir}/zone_timeseries.npz not found. "
                 "Run visualize_all.py first.")

    zd = np.load(cache_dir / "zone_timeseries.npz", allow_pickle=True)
    times_ps = zd["times_ps"]
    n_dumps = len(times_ps)
    t_max_ps = float(times_ps[-1]) if n_dumps else 0.0

    meta = read_run_meta(run_dir)

    # Geometry parameters with defaults
    n_spots = meta.get("n_spots", "8")
    ring_radius = meta.get("ring_radius_um", meta.get("ring_radius", "2400"))
    spot_radius = meta.get("spot_radius_um", meta.get("spot_radius", "300"))

    # X-line radius = ring_radius * cos(pi/8)
    try:
        ring_um = float(ring_radius)
    except Exception:
        ring_um = 2400.0
    xline_um = ring_um * math.cos(math.pi / 8)

    lines = []
    lines.append(SEP)
    lines.append(f"  ZONE-BASED RE-ANALYSIS — geometry from {run_dir.as_posix()}/run_meta.txt")
    lines.append(SEP)
    lines.append(f"  Run directory:    {run_dir.as_posix()}")
    lines.append(f"  Particle dumps:   {run_dir.as_posix()}/particles")
    lines.append(f"  Snapshots avail:  {n_dumps}")
    lines.append(f"  Time range:       0.00 to {t_max_ps:.2f} ps")
    lines.append("")
    lines.append("  Geometry (from run_meta):")
    lines.append(f"    n_spots:          {n_spots}")
    lines.append(f"    ring_radius:      {ring_radius} um")
    lines.append(f"    spot_radius:      {spot_radius} um")
    lines.append(f"    X-line radius:    {xline_um:.0f} um  (R_ring * cos(pi/8))")
    lines.append("")
    lines.append("  Zone boundaries:")
    lines.append("    CORE     : r <  1200 um             (50% of ring radius)")
    lines.append("    INNER    : 1200 <= r < 2097 um")
    lines.append(f"    X-LINE   : 2097 <= r <= 2337 um  (annulus around X-lines)")
    lines.append("    SPOT     : r >= 1800 um  (within 2σ of spots)")
    lines.append("")
    lines.append(f"  Fast-ion threshold:  {P_PB11_THRESHOLD_KEV:.0f} keV")
    lines.append("  Species:             proton")
    lines.append(SEP)
    lines.append("")

    # Data table - same format as in post_analysis_report.txt section 3
    hdr = "    t(ps)   N_core      N_inner     N_xline     N_spot   |   E95_core   E95_inner   E95_xline   E95_spot |  core/spot  xline/spot"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) + 8))

    N_core = zd["N_core"]
    N_inner = zd["N_inner"]
    N_xline = zd["N_xline"]
    N_spot = zd["N_spot"]
    E95_core = zd["E95_core"]
    E95_inner = zd["E95_inner"]
    E95_xline = zd["E95_xline"]
    E95_spot = zd["E95_spot"]

    eps = 1e-30
    ratio_core_spot = E95_core / (E95_spot + eps)
    ratio_xline_spot = E95_xline / (E95_spot + eps)

    for i in range(n_dumps):
        t = times_ps[i]
        line = (
            f"   {t:7.2f}  {N_core[i]:.2e}  {N_inner[i]:.2e}  {N_xline[i]:.2e}  "
            f"{N_spot[i]:.2e}   |  {E95_core[i]/1000:9.1f}  {E95_inner[i]/1000:10.1f}  "
            f"{E95_xline[i]/1000:10.1f}  {E95_spot[i]/1000:9.1f} |  "
            f"{ratio_core_spot[i]:8.4f}  {ratio_xline_spot[i]:8.4f}"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True,
                        help="Run directory (e.g., runs/p1_ld_256_200k_long)")
    parser.add_argument("--output-dir",
                        help="Output directory for reports (default: same as --dir)")
    args = parser.parse_args()

    run_dir = Path(args.dir)
    if not run_dir.exists():
        sys.exit(f"ERROR: directory '{run_dir}' does not exist")

    out_dir = Path(args.output_dir) if args.output_dir else run_dir

    print(f"Generating consolidated reports for {run_dir}...")
    print()

    # Build post_analysis_report.txt
    post_content = build_post_analysis_report(run_dir)
    post_path = out_dir / "post_analysis_report.txt"
    with open(post_path, "w") as f:
        f.write(post_content)
    print(f"  ✓ Wrote {post_path} ({len(post_content.splitlines())} lines)")

    # Build zone_report.txt
    zone_content = build_zone_report(run_dir)
    zone_path = out_dir / "zone_report.txt"
    with open(zone_path, "w") as f:
        f.write(zone_content)
    print(f"  ✓ Wrote {zone_path} ({len(zone_content.splitlines())} lines)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
