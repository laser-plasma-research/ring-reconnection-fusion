#!/usr/bin/env python3
"""
freq_resonance_ghz.py — Cross-run aggregator for Paper 2 GHz frequency sweep.

UPDATED: 2026-05-05 with refined natural frequency 30 GHz and new sweep:
   10 / 20 / 25 / 28 / 30 / 33 / 35 / 40 / 50 / 100 GHz (Phase 1)
   10 / 30 / 100 GHz at 512²                            (Phase 2)
   30 GHz long 256² + 512²                              (Phase 3)

Aggregates results across the 15 sub_jobs of Paper 2 in three sections:

  (1) FREQUENCY SWEEP @ 256² (Phase 1)
      10 frequencies at 7.5 ps cadence (4.9 ps for 40-50 GHz, 2.4 ps for 100 GHz)
      Compared against static ultrafine baseline (Paper 1 LD 512² ultrafine)

  (2) RESOLUTION CONVERGENCE (Phase 2)
      256² vs 512² ultrafine at 10/30/100 GHz
      Validates resonance peak is not a resolution artefact

  (3) LONG-TIME COMPARISON (Phase 3)
      30 GHz long (256² + 512²) vs static long baseline
      Demonstrates sustained operation over 30 ns

Usage:
    python freq_resonance_ghz.py --base-dir runs \
        --static-ultrafine runs/p1_ld_512_4500_ultrafine \
        --static-long runs/p1_ld_256_200k_long_v2

    python freq_resonance_ghz.py --base-dir runs \
        --output freq_resonance_GHz_sweep.txt
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# Run definitions — UPDATED for refined natural frequency
# ============================================================================

# Phase 1: GHz frequency sweep at 256² (densely sampled around 28-33 GHz)
SWEEP_256 = [
    ("p2_ghz_10",  1.0e10,  "10 GHz",  "sub-resonant control"),
    ("p2_ghz_20",  2.0e10,  "20 GHz",  "approaching lower flank"),
    ("p2_ghz_25",  2.5e10,  "25 GHz",  "lower flank"),
    ("p2_ghz_28",  2.8e10,  "28 GHz",  "near peak (low estimate)"),
    ("p2_ghz_30",  3.0e10,  "30 GHz",  "AT RESONANCE (central)"),
    ("p2_ghz_33",  3.3e10,  "33 GHz",  "near peak (high estimate)"),
    ("p2_ghz_35",  3.5e10,  "35 GHz",  "upper flank"),
    ("p2_ghz_40",  4.0e10,  "40 GHz",  "supra-resonant transition"),
    ("p2_ghz_50",  5.0e10,  "50 GHz",  "supra-resonant"),
    ("p2_ghz_100", 1.0e11,  "100 GHz", "deep supra-resonant"),
]

# Phase 2: ultrafine convergence at 512²
CONVERGENCE_512 = [
    ("p2_ghz_10_512",   1.0e10,  "10 GHz (512²)"),
    ("p2_ghz_30_512",   3.0e10,  "30 GHz (512²)"),
    ("p2_ghz_100_512",  1.0e11,  "100 GHz (512²)"),
]

# Phase 3: long flagship runs at 30 GHz
LONG_FLAGSHIP = [
    ("p2_ghz_30_long_256", 3.0e10, "30 GHz LONG (256²)", 256),
    ("p2_ghz_30_long_512", 3.0e10, "30 GHz LONG (512²)", 512),
]

# Refined natural frequency from LD 512² ultrafine analysis
NATURAL_FREQUENCY_HZ      = 3.0e10           # 30 GHz central estimate
NATURAL_FREQUENCY_RANGE   = (2.7e10, 3.7e10) # 27-37 GHz uncertainty window
P11B_THRESHOLD_KEV        = 500.0
LASER_ENERGY_J            = 5.0


# ============================================================================
# Data extraction
# ============================================================================

def csv_last_value(csv_path, column):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[0].strip().split(",")]
    try:
        idx = header.index(column)
    except ValueError:
        return None
    last = lines[-1].strip().split(",")
    try:
        return float(last[idx])
    except (ValueError, IndexError):
        return None


def csv_max_value(csv_path, column):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[0].strip().split(",")]
    try:
        idx = header.index(column)
    except ValueError:
        return None
    values = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) <= idx:
            continue
        try:
            values.append(float(parts[idx]))
        except ValueError:
            continue
    return max(values) if values else None


def parse_zone_report(report_path):
    if not os.path.exists(report_path):
        return {"peak_e95_core_keV": None,
                "peak_core_spot_ratio": None,
                "peak_xline_spot_ratio": None}
    peak_e95_core = peak_cs = peak_xs = 0.0
    with open(report_path) as f:
        in_table = False
        for line in f:
            if "core/spot" in line and "xline/spot" in line:
                in_table = True
                continue
            if not in_table:
                continue
            if line.strip().startswith("-") or line.strip().startswith("="):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                if line.strip() == "" or "TREND" in line or "PEAK" in line:
                    break
                continue
            try:
                e95s = parts[1].split()
                if len(e95s) >= 4:
                    peak_e95_core = max(peak_e95_core, float(e95s[0]))
                ratios = parts[2].split()
                if len(ratios) >= 2:
                    peak_cs = max(peak_cs, float(ratios[0]))
                    peak_xs = max(peak_xs, float(ratios[1]))
            except (ValueError, IndexError):
                continue
    return {
        "peak_e95_core_keV":     peak_e95_core if peak_e95_core > 0 else None,
        "peak_core_spot_ratio":  peak_cs       if peak_cs > 0       else None,
        "peak_xline_spot_ratio": peak_xs       if peak_xs > 0       else None,
    }


def fetch_run_metrics(run_dir):
    if not os.path.isdir(run_dir):
        return {"error": f"directory not found: {run_dir}"}
    csv_path = os.path.join(run_dir, "fusion_rate_power_by_iter.csv")
    zone_path = os.path.join(run_dir, "zone_report.txt")
    metrics = {
        "run_dir":             run_dir,
        "cum_alpha":           csv_last_value(csv_path, "cum_alpha_yield_p11b"),
        "fusion_rate":         csv_last_value(csv_path, "fusion_rate_p11b_s^-1"),
        "max_fusion_rate":     csv_max_value(csv_path, "fusion_rate_p11b_s^-1"),
        "fusion_power_w":      csv_last_value(csv_path, "fusion_power_p11b_w"),
        "Q":                   csv_last_value(csv_path, "gain_vs_laser_energy"),
        "fast_frac":           csv_last_value(csv_path, "fast_fraction_gt_500kev_p11b"),
        "max_fast_frac":       csv_max_value(csv_path, "fast_fraction_gt_500kev_p11b"),
        "final_time_ns":       csv_last_value(csv_path, "time_ns"),
    }
    metrics.update(parse_zone_report(zone_path))
    metrics["alpha_per_sr"] = (metrics["cum_alpha"] / (4 * np.pi)
                               if metrics["cum_alpha"] else None)
    return metrics


# ============================================================================
# Reporting helpers
# ============================================================================

def fmt(v, fstr="{:.3e}"):
    if v is None:
        return "n/a"
    try:
        return fstr.format(v)
    except (ValueError, TypeError):
        return "n/a"


def print_section_header(title, out=sys.stdout):
    print(file=out)
    print("═" * 130, file=out)
    print(f"  {title}", file=out)
    print("═" * 130, file=out)


def print_phase1_table(rows, baseline, out=sys.stdout):
    f_low_ghz = NATURAL_FREQUENCY_RANGE[0] / 1e9
    f_hi_ghz = NATURAL_FREQUENCY_RANGE[1] / 1e9
    print_section_header(
        "PHASE 1 — GHz FREQUENCY SWEEP @ 256² (resonance physics)", out)
    print(f"  Cadence: 7.5 ps (default), 4.9 ps (40-50 GHz), 2.4 ps (100 GHz)", file=out)
    print(f"  Duration: 674 ps  |  Natural cycle: {NATURAL_FREQUENCY_HZ/1e9:.0f} GHz "
          f"(window {f_low_ghz:.0f}-{f_hi_ghz:.0f} GHz)", file=out)
    if baseline:
        print(f"  Static baseline:  {baseline['run_dir']} (LD 512² ultrafine)", file=out)
    print(file=out)

    hdr = (f"{'sub_tag':<14s} {'freq':>9s} {'f/f_nat':>8s} {'regime':<28s} | "
           f"{'cum_α':>10s} {'peak_E95':>10s} {'core/spot':>10s} "
           f"{'fast_frac':>10s} {'Q':>8s}")
    print(hdr, file=out)
    print("─" * len(hdr), file=out)

    if baseline:
        b = baseline
        print(f"{'(static_ulf)':<14s} {'static':>9s} {'—':>8s} "
              f"{'baseline (no rotation)':<28s} | "
              f"{fmt(b.get('cum_alpha')):>10s} "
              f"{fmt(b.get('peak_e95_core_keV'), '{:.0f}'):>10s} "
              f"{fmt(b.get('peak_core_spot_ratio'), '{:.2f}'):>10s} "
              f"{fmt(b.get('max_fast_frac'), '{:.3f}'):>10s} "
              f"{fmt(b.get('Q'), '{:.4f}'):>8s}", file=out)
        print("─" * len(hdr), file=out)

    for sub_tag, freq_hz, freq_label, regime in SWEEP_256:
        m = rows.get(sub_tag)
        if not m or "error" in m:
            print(f"{sub_tag:<14s} {freq_label:>9s} "
                  f"{freq_hz/NATURAL_FREQUENCY_HZ:>7.2f}× {regime:<28s} | "
                  f"  *** missing ***", file=out)
            continue
        f_ratio = freq_hz / NATURAL_FREQUENCY_HZ
        # Mark window
        in_window = (NATURAL_FREQUENCY_RANGE[0] <= freq_hz <= NATURAL_FREQUENCY_RANGE[1])
        marker = " ◀" if in_window else ""
        print(f"{sub_tag:<14s} {freq_label:>9s} {f_ratio:>7.2f}× {regime:<28s} | "
              f"{fmt(m.get('cum_alpha')):>10s} "
              f"{fmt(m.get('peak_e95_core_keV'), '{:.0f}'):>10s} "
              f"{fmt(m.get('peak_core_spot_ratio'), '{:.2f}'):>10s} "
              f"{fmt(m.get('max_fast_frac'), '{:.3f}'):>10s} "
              f"{fmt(m.get('Q'), '{:.4f}'):>8s}{marker}", file=out)
    print("─" * len(hdr), file=out)
    print(f"  ◀ = within natural cycle frequency window ({f_low_ghz:.0f}-{f_hi_ghz:.0f} GHz)",
          file=out)


def print_phase1_resonance_summary(rows, baseline, out=sys.stdout):
    valid = [(t, h, l, r, rows[t]) for t, h, l, r in SWEEP_256
             if t in rows and "error" not in rows[t]
             and rows[t].get("peak_e95_core_keV")]

    print(file=out)
    print(f"  RESONANCE ANALYSIS:", file=out)
    if not valid:
        print(f"    No valid data points yet.", file=out)
        return

    peak_e95 = max(valid, key=lambda x: x[4]["peak_e95_core_keV"])
    print(f"    Peak E95_core:        {peak_e95[2]} "
          f"({peak_e95[4]['peak_e95_core_keV']:,.0f} keV)", file=out)

    valid_cs = [r for r in valid if r[4].get("peak_core_spot_ratio")]
    if valid_cs:
        peak_cs = max(valid_cs, key=lambda x: x[4]["peak_core_spot_ratio"])
        print(f"    Peak core/spot:       {peak_cs[2]} "
              f"({peak_cs[4]['peak_core_spot_ratio']:.2f}×)", file=out)

    valid_y = [r for r in valid if r[4].get("cum_alpha")]
    if valid_y:
        peak_y = max(valid_y, key=lambda x: x[4]["cum_alpha"])
        print(f"    Peak fusion yield:    {peak_y[2]} "
              f"(α = {peak_y[4]['cum_alpha']:.3e})", file=out)

    if baseline and baseline.get("peak_e95_core_keV"):
        b_e95 = baseline["peak_e95_core_keV"]
        print(file=out)
        print(f"  Enhancement factors (E95_core ratio vs static ultrafine baseline):",
              file=out)
        for tag, freq_hz, label, _ in SWEEP_256:
            r = rows.get(tag)
            if r and r.get("peak_e95_core_keV"):
                ratio = r["peak_e95_core_keV"] / b_e95
                in_window = (NATURAL_FREQUENCY_RANGE[0] <= freq_hz <= NATURAL_FREQUENCY_RANGE[1])
                marker = " ◀ in window" if in_window else ""
                bar = "█" * min(int(ratio * 10), 50)
                print(f"    {label:>10s}: {ratio:5.2f}×  {bar}{marker}", file=out)


def print_phase2_convergence(rows_256, rows_512, out=sys.stdout):
    print_section_header(
        "PHASE 2 — RESOLUTION CONVERGENCE @ 256² vs 512² (ultrafine cadence)", out)
    print(f"  Validates resonance peak is not a resolution artefact", file=out)
    print(file=out)

    pairs = [
        ("10 GHz",  "p2_ghz_10",   "p2_ghz_10_512"),
        ("30 GHz",  "p2_ghz_30",   "p2_ghz_30_512"),
        ("100 GHz", "p2_ghz_100",  "p2_ghz_100_512"),
    ]

    hdr = (f"{'frequency':<10s} | {'metric':<22s} | "
           f"{'256²':>14s} {'512²':>14s} {'ratio (512/256)':>18s}")
    print(hdr, file=out)
    print("─" * len(hdr), file=out)

    for label, tag_256, tag_512 in pairs:
        m256 = rows_256.get(tag_256)
        m512 = rows_512.get(tag_512)
        if not (m256 and m512):
            present = []
            if m256: present.append("256²")
            if m512: present.append("512²")
            present_str = ", ".join(present) if present else "neither"
            print(f"  {label}: incomplete (have: {present_str})", file=out)
            continue
        for metric_name, key, fstr in [
            ("peak E95_core (keV)",    "peak_e95_core_keV",     "{:.0f}"),
            ("peak core/spot",          "peak_core_spot_ratio",  "{:.2f}"),
            ("cum alpha",               "cum_alpha",             "{:.3e}"),
            ("Q",                       "Q",                     "{:.4f}"),
        ]:
            v256 = m256.get(key)
            v512 = m512.get(key)
            if v256 is None or v512 is None or v256 == 0:
                continue
            ratio = v512 / v256
            converged = "✓" if 0.85 <= ratio <= 1.15 else "⚠"
            print(f"{label:<10s} | {metric_name:<22s} | "
                  f"{fstr.format(v256):>14s} {fstr.format(v512):>14s} "
                  f"{ratio:>14.3f} {converged}", file=out)
        print("─" * len(hdr), file=out)
    print("  ✓ converged (within ±15%)   ⚠ resolution-dependent", file=out)


def print_phase3_long_comparison(rows_long, static_long, out=sys.stdout):
    print_section_header(
        "PHASE 3 — LONG-TIME COMPARISON @ 30 GHz vs STATIC LONG BASELINE", out)
    print(f"  30 ns runs at flagship frequency, demonstrate sustained operation", file=out)
    if static_long:
        print(f"  Static baseline:  {static_long['run_dir']} "
              f"(Paper 1 LD 256² long v2)", file=out)
    print(file=out)

    hdr = (f"{'config':<28s} | {'cum_α':>10s} {'final_α/s':>12s} {'fusion_W':>12s} "
           f"{'Q':>8s} {'peak_E95':>10s} {'fast_frac':>10s}")
    print(hdr, file=out)
    print("─" * len(hdr), file=out)

    if static_long:
        s = static_long
        print(f"{'STATIC LONG (256²)':<28s} | "
              f"{fmt(s.get('cum_alpha')):>10s} "
              f"{fmt(s.get('fusion_rate'), '{:.3e}'):>12s} "
              f"{fmt(s.get('fusion_power_w'), '{:.3e}'):>12s} "
              f"{fmt(s.get('Q'), '{:.4f}'):>8s} "
              f"{fmt(s.get('peak_e95_core_keV'), '{:.0f}'):>10s} "
              f"{fmt(s.get('max_fast_frac'), '{:.3f}'):>10s}", file=out)
        print("─" * len(hdr), file=out)

    for tag, freq, label, res in LONG_FLAGSHIP:
        m = rows_long.get(tag)
        if not m or "error" in m:
            print(f"{label:<28s} |  *** missing ***", file=out)
            continue
        print(f"{label:<28s} | "
              f"{fmt(m.get('cum_alpha')):>10s} "
              f"{fmt(m.get('fusion_rate'), '{:.3e}'):>12s} "
              f"{fmt(m.get('fusion_power_w'), '{:.3e}'):>12s} "
              f"{fmt(m.get('Q'), '{:.4f}'):>8s} "
              f"{fmt(m.get('peak_e95_core_keV'), '{:.0f}'):>10s} "
              f"{fmt(m.get('max_fast_frac'), '{:.3f}'):>10s}", file=out)
    print("─" * len(hdr), file=out)

    # Enhancement vs static long
    if static_long and static_long.get("cum_alpha"):
        s_alpha = static_long["cum_alpha"]
        print(file=out)
        print(f"  ENHANCEMENT vs static long baseline (cum_α ratio):", file=out)
        for tag, freq, label, res in LONG_FLAGSHIP:
            m = rows_long.get(tag)
            if m and m.get("cum_alpha"):
                ratio = m["cum_alpha"] / s_alpha
                bar = "█" * min(int(ratio * 5), 50)
                print(f"    {label:<28s}: {ratio:5.2f}×  {bar}", file=out)

        # Long convergence check (256² vs 512² long)
        m256 = rows_long.get("p2_ghz_30_long_256")
        m512 = rows_long.get("p2_ghz_30_long_512")
        if m256 and m512 and m256.get("cum_alpha") and m512.get("cum_alpha"):
            ratio = m512["cum_alpha"] / m256["cum_alpha"]
            converged = "✓ converged" if 0.85 <= ratio <= 1.15 else "⚠ resolution-dependent"
            print(file=out)
            print(f"  LONG-RUN convergence (30 GHz, 512² / 256² cum_α ratio): "
                  f"{ratio:.3f}  {converged}", file=out)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default="runs",
                        help="Directory containing the p2_* run subdirs")
    parser.add_argument("--static-ultrafine",
                        default="runs/p1_ld_512_4500_ultrafine",
                        help="Static reference at ultrafine cadence (Paper 1)")
    parser.add_argument("--static-long",
                        default="runs/p1_ld_256_200k_long_v2",
                        help="Static reference at long duration (Paper 1)")
    parser.add_argument("--output",
                        help="Save report to this file (in addition to stdout)")
    args = parser.parse_args()

    base = Path(args.base_dir)

    # Phase 1: 256² sweep
    rows_256 = {tag: fetch_run_metrics(str(base / tag))
                for tag, *_ in SWEEP_256}

    # Phase 2: 512² convergence
    rows_512 = {tag: fetch_run_metrics(str(base / tag))
                for tag, *_ in CONVERGENCE_512}

    # Phase 3: long flagship
    rows_long = {tag: fetch_run_metrics(str(base / tag))
                 for tag, *_ in LONG_FLAGSHIP}

    # Static baselines
    static_ultrafine = (fetch_run_metrics(args.static_ultrafine)
                        if os.path.isdir(args.static_ultrafine) else None)
    static_long = (fetch_run_metrics(args.static_long)
                   if os.path.isdir(args.static_long) else None)

    if not static_ultrafine:
        print(f"WARNING: ultrafine baseline not found at {args.static_ultrafine}",
              file=sys.stderr)
    if not static_long:
        print(f"WARNING: long baseline not found at {args.static_long}",
              file=sys.stderr)

    streams = [sys.stdout]
    fout = None
    if args.output:
        fout = open(args.output, "w")
        streams.append(fout)

    for stream in streams:
        print("█" * 130, file=stream)
        print(f"  PAPER 2 — GHz FREQUENCY RESONANCE SCAN — FULL REPORT", file=stream)
        print(f"  Natural cycle frequency: ~30 GHz (range 27-37 GHz)  |  "
              f"p-11B threshold: 500 keV", file=stream)
        print("█" * 130, file=stream)

        print_phase1_table(rows_256, static_ultrafine, out=stream)
        print_phase1_resonance_summary(rows_256, static_ultrafine, out=stream)
        print_phase2_convergence(rows_256, rows_512, out=stream)
        print_phase3_long_comparison(rows_long, static_long, out=stream)

        print(file=stream)
        print("█" * 130, file=stream)
        print("  END OF REPORT", file=stream)
        print("█" * 130, file=stream)

    if fout:
        fout.close()
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
