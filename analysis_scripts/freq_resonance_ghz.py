#!/usr/bin/env python3
"""
freq_resonance_ghz.py — Cross-run aggregator for Paper 2 GHz frequency sweep.

REVISION (2026-05-07):
  - Frequency lists no longer hardcoded. Phase 1/2/3 sub_jobs are read from
    program_config_paper2.yaml so the orchestrator and analyzer cannot
    silently disagree about which runs exist. Single source of truth.
  - Unit safety: all E95 values are normalized to keV at the read boundary.
    Heuristic: any value > 10,000 is assumed eV and divided by 1000.
    This catches the YAML-baked static baseline of 502478 (eV) so it
    plots alongside per-run values that are correctly in keV.
  - Added post_pinch_peak_core_spot metric: the maximum core/spot ratio
    AFTER t > 100 ps. The original peak_core_spot_ratio is dominated by
    the early initial-pinch transient (which is essentially identical
    across rotating runs because it's set by static seed B-field, not
    by modulation). The post-pinch metric is what actually distinguishes
    modulation effects.
  - Phase 1 table now shows BOTH the raw peak (early-pinch) and the
    post-pinch peak side by side, with the post-pinch peak clearly
    labeled as the modulation-relevant metric.

Usage:
    python freq_resonance_ghz.py --base-dir runs \
        --config program_config_paper2.yaml \
        --static-ultrafine runs/p1_ld_512_4500_ultrafine \
        --static-long runs/p1_ld_256_200k_long_v2

    python freq_resonance_ghz.py --base-dir runs --output report.txt
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML required. Install with: pip install pyyaml")


# ============================================================================
# Constants
# ============================================================================

P11B_THRESHOLD_KEV = 500.0
LASER_ENERGY_J = 5.0
POST_PINCH_T_THRESHOLD_PS = 100.0  # snapshots before this are dominated by initial pinch

# Sanity bounds for E95 values (keV). Any value outside this is suspicious.
E95_MIN_PLAUSIBLE_KEV = 0.1     # 100 eV — colder than thermal would be a bug
E95_MAX_PLAUSIBLE_KEV = 50000.0 # 50 MeV — anything higher is almost certainly mis-unit


# ============================================================================
# Configuration loader — pull frequency lists from YAML instead of hardcoding
# ============================================================================

def load_sub_jobs_from_yaml(yaml_path):
    """Parse program_config_paper2.yaml and return Phase 1, 2, 3 sub_jobs.

    Returns:
        sweep_256:        list of (sub_tag, freq_hz, label, regime) tuples
        convergence_512:  list of (sub_tag, freq_hz, label) tuples
        long_flagship:    list of (sub_tag, freq_hz, label, resolution) tuples
        natural_freq_hz:  central estimate of natural reconnection frequency
        natural_window:   (low_hz, high_hz) tuple defining the natural-cycle window
    """
    if not os.path.exists(yaml_path):
        sys.exit(f"ERROR: config file not found: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Find the Paper A2 entry
    papers = cfg.get("papers", [])
    paper_a2 = next((p for p in papers if p.get("id") == "A2"), None)
    if not paper_a2:
        sys.exit(f"ERROR: paper id 'A2' not found in {yaml_path}")

    sub_jobs = paper_a2.get("simulation", {}).get("sub_jobs", [])
    if not sub_jobs:
        sys.exit(f"ERROR: no sub_jobs in paper A2 of {yaml_path}")

    sweep_256 = []
    convergence_512 = []
    long_flagship = []

    for j in sub_jobs:
        sub_tag = j.get("sub_tag")
        freq_hz = float(j.get("freq_hz", 0))
        label = j.get("freq_label", sub_tag)
        regime = j.get("regime", "")
        phase = j.get("phase", 1)
        resolution = j.get("resolution", 256)

        if phase == 1:
            sweep_256.append((sub_tag, freq_hz, label, regime))
        elif phase == 2:
            convergence_512.append((sub_tag, freq_hz, label))
        elif phase == 3:
            long_flagship.append((sub_tag, freq_hz, label, resolution))

    # Natural frequency (prefer sub-cycle estimate; YAML schema has both)
    nat = paper_a2.get("natural_frequency", {})
    nat_freq_ghz = nat.get("sub_cycle_estimate_ghz") or nat.get("big_cycle_estimate_ghz") or 30.0
    nat_window_ghz = nat.get("sub_cycle_window_ghz") or nat.get("big_cycle_window_ghz") or [27.0, 37.0]
    natural_freq_hz = nat_freq_ghz * 1e9
    natural_window = (nat_window_ghz[0] * 1e9, nat_window_ghz[1] * 1e9)

    return {
        "sweep_256": sweep_256,
        "convergence_512": convergence_512,
        "long_flagship": long_flagship,
        "natural_freq_hz": natural_freq_hz,
        "natural_window": natural_window,
    }


# ============================================================================
# Unit-safe numeric reader
# ============================================================================

def normalize_e95_to_kev(value):
    """Coerce an E95 value to keV with heuristic unit detection.

    The static baseline value in the YAML (502478) is in eV but labeled keV.
    Per-run values from zone_report.txt are in keV. To prevent the units from
    being mixed across rows of the same column, we normalize at the read
    boundary: anything above the plausible keV ceiling is assumed eV.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    if v == 0:
        return None
    if v > E95_MAX_PLAUSIBLE_KEV:
        # Almost certainly eV — convert
        return v / 1000.0
    return v


# ============================================================================
# Per-run data extraction
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
    """Parse the zone-table from post_analysis_report.txt or zone_report.txt.

    Returns peak metrics including BOTH the raw peak (whole-run argmax) and
    the post-pinch peak (argmax restricted to t > POST_PINCH_T_THRESHOLD_PS).
    Per-run E95 values are emitted in keV; we keep them in keV here.
    """
    if not os.path.exists(report_path):
        return {
            "peak_e95_core_keV": None,
            "peak_core_spot_ratio": None,
            "peak_xline_spot_ratio": None,
            "post_pinch_peak_core_spot": None,
            "post_pinch_peak_e95_core_keV": None,
        }

    # Walk the table line by line, capturing (t_ps, e95_core, ratio_cs, ratio_xs)
    # The header line of the zone table contains "core/spot" AND "xline/spot"
    rows = []
    with open(report_path) as f:
        in_table = False
        for line in f:
            if "core/spot" in line and "xline/spot" in line:
                in_table = True
                continue
            if not in_table:
                continue
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("="):
                continue
            if stripped == "" or "TREND" in line or "PEAK" in line:
                break
            parts = line.split("|")
            if len(parts) != 3:
                continue
            try:
                # Left segment: " 14.97   1.23e+04   ..."
                #   first value is t_ps
                left = parts[0].split()
                if len(left) < 1:
                    continue
                t_ps = float(left[0])

                # Middle segment: "  E95_core   E95_inner   E95_xline   E95_spot"
                e95s = parts[1].split()
                if len(e95s) < 4:
                    continue
                e95_core_keV = float(e95s[0])

                # Right segment: "  core/spot  xline/spot"
                ratios = parts[2].split()
                if len(ratios) < 2:
                    continue
                ratio_cs = float(ratios[0])
                ratio_xs = float(ratios[1])

                rows.append((t_ps, e95_core_keV, ratio_cs, ratio_xs))
            except (ValueError, IndexError):
                continue

    if not rows:
        return {
            "peak_e95_core_keV": None,
            "peak_core_spot_ratio": None,
            "peak_xline_spot_ratio": None,
            "post_pinch_peak_core_spot": None,
            "post_pinch_peak_e95_core_keV": None,
        }

    arr = np.array(rows)  # shape (N, 4): t_ps, e95_core, ratio_cs, ratio_xs

    # Whole-run peaks
    peak_e95 = float(arr[:, 1].max())
    peak_cs = float(arr[:, 2].max())
    peak_xs = float(arr[:, 3].max())

    # Post-pinch peaks (t > threshold)
    mask = arr[:, 0] > POST_PINCH_T_THRESHOLD_PS
    if mask.any():
        post_peak_cs = float(arr[mask, 2].max())
        post_peak_e95 = float(arr[mask, 1].max())
    else:
        post_peak_cs = None
        post_peak_e95 = None

    return {
        "peak_e95_core_keV":              peak_e95 if peak_e95 > 0 else None,
        "peak_core_spot_ratio":           peak_cs if peak_cs > 0 else None,
        "peak_xline_spot_ratio":          peak_xs if peak_xs > 0 else None,
        "post_pinch_peak_core_spot":      post_peak_cs,
        "post_pinch_peak_e95_core_keV":   post_peak_e95,
    }


def fetch_run_metrics(run_dir):
    """Fetch metrics for one run. Tries post_analysis_report.txt then zone_report.txt."""
    if not os.path.isdir(run_dir):
        return {"error": f"directory not found: {run_dir}"}
    csv_path = os.path.join(run_dir, "fusion_rate_power_by_iter.csv")

    # Prefer post_analysis_report.txt; fall back to zone_report.txt
    report_path = os.path.join(run_dir, "post_analysis_report.txt")
    if not os.path.exists(report_path):
        report_path = os.path.join(run_dir, "zone_report.txt")

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
    metrics.update(parse_zone_report(report_path))

    # Unit normalization: peak_e95_core_keV may have come from a YAML-baked
    # value in eV. Normalize defensively.
    metrics["peak_e95_core_keV"] = normalize_e95_to_kev(metrics["peak_e95_core_keV"])
    metrics["post_pinch_peak_e95_core_keV"] = normalize_e95_to_kev(
        metrics["post_pinch_peak_e95_core_keV"])

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


def print_phase1_table(rows, baseline, sweep_jobs, natural_freq_hz, natural_window, out=sys.stdout):
    f_low_ghz = natural_window[0] / 1e9
    f_hi_ghz = natural_window[1] / 1e9
    print_section_header(
        "PHASE 1 — GHz FREQUENCY SWEEP @ 256² (resonance physics)", out)
    print(f"  Cadence: 7.5 ps (default), 4.9 ps (>33 GHz), 2.4 ps (100 GHz)", file=out)
    print(f"  Duration: 674 ps  |  Natural cycle estimate: "
          f"{natural_freq_hz/1e9:.0f} GHz (window {f_low_ghz:.0f}-{f_hi_ghz:.0f} GHz)",
          file=out)
    print(f"  Post-pinch threshold: t > {POST_PINCH_T_THRESHOLD_PS:.0f} ps "
          f"(rules out initial-pinch transient)", file=out)
    if baseline:
        print(f"  Static baseline:  {baseline['run_dir']} (LD 512² ultrafine)", file=out)
    print(file=out)
    print(f"  Note: 'core/spot (raw)' is whole-run peak — DOMINATED BY EARLY PINCH.",
          file=out)
    print(f"        'core/spot (post-pinch)' is the modulation-relevant metric.",
          file=out)
    print(file=out)

    hdr = (f"{'sub_tag':<14s} {'freq':>9s} {'f/f_nat':>8s} {'regime':<28s} | "
           f"{'cum_α':>10s} {'pk_E95(raw)':>12s} {'cs(raw)':>9s} "
           f"{'cs(post-pinch)':>16s} {'pk_E95(post)':>13s} "
           f"{'fast_frac':>10s} {'Q':>8s}")
    print(hdr, file=out)
    print("─" * len(hdr), file=out)

    if baseline:
        b = baseline
        print(f"{'(static_ulf)':<14s} {'static':>9s} {'—':>8s} "
              f"{'baseline (no rotation)':<28s} | "
              f"{fmt(b.get('cum_alpha')):>10s} "
              f"{fmt(b.get('peak_e95_core_keV'), '{:.0f}'):>12s} "
              f"{fmt(b.get('peak_core_spot_ratio'), '{:.2f}'):>9s} "
              f"{fmt(b.get('post_pinch_peak_core_spot'), '{:.2f}'):>16s} "
              f"{fmt(b.get('post_pinch_peak_e95_core_keV'), '{:.0f}'):>13s} "
              f"{fmt(b.get('max_fast_frac'), '{:.3f}'):>10s} "
              f"{fmt(b.get('Q'), '{:.4f}'):>8s}", file=out)
        print("─" * len(hdr), file=out)

    for sub_tag, freq_hz, freq_label, regime in sweep_jobs:
        m = rows.get(sub_tag)
        if not m or "error" in m:
            print(f"{sub_tag:<14s} {freq_label:>9s} "
                  f"{freq_hz/natural_freq_hz:>7.2f}× {regime:<28s} | "
                  f"  *** missing ***", file=out)
            continue
        f_ratio = freq_hz / natural_freq_hz
        in_window = (natural_window[0] <= freq_hz <= natural_window[1])
        marker = " ◀" if in_window else ""
        print(f"{sub_tag:<14s} {freq_label:>9s} {f_ratio:>7.2f}× {regime:<28s} | "
              f"{fmt(m.get('cum_alpha')):>10s} "
              f"{fmt(m.get('peak_e95_core_keV'), '{:.0f}'):>12s} "
              f"{fmt(m.get('peak_core_spot_ratio'), '{:.2f}'):>9s} "
              f"{fmt(m.get('post_pinch_peak_core_spot'), '{:.2f}'):>16s} "
              f"{fmt(m.get('post_pinch_peak_e95_core_keV'), '{:.0f}'):>13s} "
              f"{fmt(m.get('max_fast_frac'), '{:.3f}'):>10s} "
              f"{fmt(m.get('Q'), '{:.4f}'):>8s}{marker}", file=out)
    print("─" * len(hdr), file=out)
    print(f"  ◀ = within natural cycle frequency window ({f_low_ghz:.0f}-{f_hi_ghz:.0f} GHz)",
          file=out)


def print_phase1_resonance_summary(rows, baseline, sweep_jobs, natural_freq_hz,
                                   natural_window, out=sys.stdout):
    valid = [(t, h, l, r, rows[t]) for t, h, l, r in sweep_jobs
             if t in rows and "error" not in rows[t]
             and rows[t].get("post_pinch_peak_e95_core_keV")]

    print(file=out)
    print(f"  RESONANCE ANALYSIS (post-pinch metrics — t > {POST_PINCH_T_THRESHOLD_PS:.0f} ps):",
          file=out)
    if not valid:
        print(f"    No valid data points yet.", file=out)
        return

    peak_e95 = max(valid, key=lambda x: x[4]["post_pinch_peak_e95_core_keV"])
    print(f"    Peak post-pinch E95_core:   {peak_e95[2]} "
          f"({peak_e95[4]['post_pinch_peak_e95_core_keV']:,.0f} keV)", file=out)

    valid_cs = [r for r in valid if r[4].get("post_pinch_peak_core_spot")]
    if valid_cs:
        peak_cs = max(valid_cs, key=lambda x: x[4]["post_pinch_peak_core_spot"])
        print(f"    Peak post-pinch core/spot:  {peak_cs[2]} "
              f"({peak_cs[4]['post_pinch_peak_core_spot']:.2f}×)", file=out)

    valid_y = [r for r in valid if r[4].get("cum_alpha")]
    if valid_y:
        peak_y = max(valid_y, key=lambda x: x[4]["cum_alpha"])
        print(f"    Peak fusion yield:          {peak_y[2]} "
              f"(α = {peak_y[4]['cum_alpha']:.3e})", file=out)

    if baseline and baseline.get("post_pinch_peak_e95_core_keV"):
        b_e95 = baseline["post_pinch_peak_e95_core_keV"]
        print(file=out)
        print(f"  Enhancement factors (post-pinch E95_core ratio vs static baseline):",
              file=out)
        for tag, freq_hz, label, _ in sweep_jobs:
            r = rows.get(tag)
            if r and r.get("post_pinch_peak_e95_core_keV"):
                ratio = r["post_pinch_peak_e95_core_keV"] / b_e95
                in_window = (natural_window[0] <= freq_hz <= natural_window[1])
                marker = " ◀ in window" if in_window else ""
                bar = "█" * min(int(ratio * 10), 50)
                print(f"    {label:>10s}: {ratio:5.2f}×  {bar}{marker}", file=out)


def print_phase2_convergence(rows_256, rows_512, sweep_jobs_256,
                              convergence_jobs_512, out=sys.stdout):
    print_section_header(
        "PHASE 2 — RESOLUTION CONVERGENCE @ 256² vs 512² (ultrafine cadence)", out)
    print(f"  Validates resonance peak is not a resolution artefact", file=out)
    print(file=out)

    # Build pairs by matching freq_hz between 256² and 512² lists
    pairs = []
    by_freq_256 = {f: t for t, f, *_ in sweep_jobs_256}
    for tag_512, freq_hz, label in convergence_jobs_512:
        tag_256 = by_freq_256.get(freq_hz)
        if tag_256:
            pairs.append((label.replace(" (512² convergence)", "").replace(" — DEFAULT", ""),
                         tag_256, tag_512))

    if not pairs:
        print(f"  No matched 256²/512² pairs found.", file=out)
        return

    hdr = (f"{'frequency':<12s} | {'metric':<25s} | "
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
            ("post-pinch E95_core (keV)",  "post_pinch_peak_e95_core_keV", "{:.0f}"),
            ("post-pinch core/spot",        "post_pinch_peak_core_spot",   "{:.2f}"),
            ("cum alpha",                   "cum_alpha",                    "{:.3e}"),
            ("Q",                           "Q",                            "{:.4f}"),
        ]:
            v256 = m256.get(key)
            v512 = m512.get(key)
            if v256 is None or v512 is None or v256 == 0:
                continue
            ratio = v512 / v256
            converged = "✓" if 0.85 <= ratio <= 1.15 else "⚠"
            print(f"{label:<12s} | {metric_name:<25s} | "
                  f"{fstr.format(v256):>14s} {fstr.format(v512):>14s} "
                  f"{ratio:>14.3f} {converged}", file=out)
        print("─" * len(hdr), file=out)
    print("  ✓ converged (within ±15%)   ⚠ resolution-dependent", file=out)


def print_phase3_long_comparison(rows_long, static_long, long_jobs, out=sys.stdout):
    print_section_header(
        "PHASE 3 — LONG-TIME COMPARISON @ FLAGSHIP FREQUENCY vs STATIC LONG", out)
    print(f"  30 ns runs at flagship frequency, demonstrate sustained operation", file=out)
    if static_long:
        print(f"  Static baseline:  {static_long['run_dir']}", file=out)
    print(file=out)

    hdr = (f"{'config':<28s} | {'cum_α':>10s} {'final_α/s':>12s} {'fusion_W':>12s} "
           f"{'Q':>8s} {'pk_E95(post)':>13s} {'fast_frac':>10s}")
    print(hdr, file=out)
    print("─" * len(hdr), file=out)

    if static_long:
        s = static_long
        print(f"{'STATIC LONG':<28s} | "
              f"{fmt(s.get('cum_alpha')):>10s} "
              f"{fmt(s.get('fusion_rate'), '{:.3e}'):>12s} "
              f"{fmt(s.get('fusion_power_w'), '{:.3e}'):>12s} "
              f"{fmt(s.get('Q'), '{:.4f}'):>8s} "
              f"{fmt(s.get('post_pinch_peak_e95_core_keV'), '{:.0f}'):>13s} "
              f"{fmt(s.get('max_fast_frac'), '{:.3f}'):>10s}", file=out)
        print("─" * len(hdr), file=out)

    for tag, freq, label, res in long_jobs:
        m = rows_long.get(tag)
        if not m or "error" in m:
            print(f"{label:<28s} |  *** missing ***", file=out)
            continue
        print(f"{label:<28s} | "
              f"{fmt(m.get('cum_alpha')):>10s} "
              f"{fmt(m.get('fusion_rate'), '{:.3e}'):>12s} "
              f"{fmt(m.get('fusion_power_w'), '{:.3e}'):>12s} "
              f"{fmt(m.get('Q'), '{:.4f}'):>8s} "
              f"{fmt(m.get('post_pinch_peak_e95_core_keV'), '{:.0f}'):>13s} "
              f"{fmt(m.get('max_fast_frac'), '{:.3f}'):>10s}", file=out)
    print("─" * len(hdr), file=out)

    if static_long and static_long.get("cum_alpha"):
        s_alpha = static_long["cum_alpha"]
        print(file=out)
        print(f"  ENHANCEMENT vs static long baseline (cum_α ratio):", file=out)
        for tag, freq, label, res in long_jobs:
            m = rows_long.get(tag)
            if m and m.get("cum_alpha"):
                ratio = m["cum_alpha"] / s_alpha
                bar = "█" * min(int(ratio * 5), 50)
                print(f"    {label:<28s}: {ratio:5.2f}×  {bar}", file=out)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default="runs",
                        help="Directory containing the p2_* run subdirs")
    parser.add_argument("--config", default="program_config_paper2.yaml",
                        help="YAML config to load sub_jobs from "
                             "(default: program_config_paper2.yaml)")
    parser.add_argument("--static-ultrafine",
                        default="runs/p1_ld_512_4500_ultrafine",
                        help="Static reference at ultrafine cadence (Paper 1)")
    parser.add_argument("--static-long",
                        default="runs/p1_ld_256_200k_long_v2",
                        help="Static reference at long duration (Paper 1)")
    parser.add_argument("--output",
                        help="Save report to this file (in addition to stdout)")
    args = parser.parse_args()

    # Load sub_jobs from YAML
    yaml_data = load_sub_jobs_from_yaml(args.config)
    sweep_256 = yaml_data["sweep_256"]
    convergence_512 = yaml_data["convergence_512"]
    long_flagship = yaml_data["long_flagship"]
    natural_freq_hz = yaml_data["natural_freq_hz"]
    natural_window = yaml_data["natural_window"]

    base = Path(args.base_dir)

    rows_256 = {tag: fetch_run_metrics(str(base / tag))
                for tag, *_ in sweep_256}
    rows_512 = {tag: fetch_run_metrics(str(base / tag))
                for tag, *_ in convergence_512}
    rows_long = {tag: fetch_run_metrics(str(base / tag))
                 for tag, *_ in long_flagship}

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

    f_low_ghz = natural_window[0] / 1e9
    f_hi_ghz = natural_window[1] / 1e9
    for stream in streams:
        print("█" * 130, file=stream)
        print(f"  PAPER 2 — GHz FREQUENCY RESONANCE SCAN — FULL REPORT", file=stream)
        print(f"  Natural cycle frequency: ~{natural_freq_hz/1e9:.0f} GHz "
              f"(range {f_low_ghz:.0f}-{f_hi_ghz:.0f} GHz)  |  "
              f"p-11B threshold: 500 keV", file=stream)
        print(f"  Loaded {len(sweep_256)} Phase-1, {len(convergence_512)} Phase-2, "
              f"{len(long_flagship)} Phase-3 sub_jobs from {args.config}", file=stream)
        print("█" * 130, file=stream)

        print_phase1_table(rows_256, static_ultrafine, sweep_256,
                          natural_freq_hz, natural_window, out=stream)
        print_phase1_resonance_summary(rows_256, static_ultrafine, sweep_256,
                                       natural_freq_hz, natural_window, out=stream)
        print_phase2_convergence(rows_256, rows_512, sweep_256, convergence_512, out=stream)
        print_phase3_long_comparison(rows_long, static_long, long_flagship, out=stream)

        print(file=stream)
        print("█" * 130, file=stream)
        print("  END OF REPORT", file=stream)
        print("█" * 130, file=stream)

    if fout:
        fout.close()
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
