#!/usr/bin/env python3
"""
freq_resonance.py — Analyze the Paper A2 frequency scan.

Looks at the 7 rotating-frequency simulations:
  100, 208, 300, 416, 500, 750, 1000 MHz

Project hypothesis: peaks at 208 MHz and 416 MHz (Alfvén sub-harmonics)
because these match the natural reconnection oscillation modes.

For each frequency:
  - Read fusion CSV
  - Extract end-of-run alpha yield, fusion rate, fusion power, Q
  - Extract peak CORE/SPOT ratio from zone analysis (need to integrate or use
    pre-generated zone reports)

Then plot/print the resonance curve: yield vs frequency.

Usage:
    python freq_resonance.py --base-dir runs/paper02/rotating
"""

import os, sys, argparse, glob, re
import numpy as np

# Frequencies in the scan
FREQUENCIES_MHZ = [100, 208, 300, 416, 500, 750, 1000]

# Predicted resonance peaks (project doc)
PREDICTED_PEAKS = {208: '1st Alfvén sub-harmonic', 416: '2nd Alfvén sub-harmonic'}


def get_csv_metric(csv_path, col_name):
    """Read a metric from the last row of a fusion CSV."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    header = lines[0].strip().split(',')
    last = lines[-1].strip().split(',')
    try:
        return float(last[header.index(col_name)])
    except (ValueError, IndexError):
        return None


def get_csv_max_metric(csv_path, col_name):
    """Get max value of a metric across all CSV rows."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    header = lines[0].strip().split(',')
    idx = header.index(col_name)
    values = []
    for line in lines[1:]:
        try:
            values.append(float(line.strip().split(',')[idx]))
        except (ValueError, IndexError):
            continue
    return max(values) if values else None


def parse_zone_report(report_path):
    """Read a zone analysis report and extract peak CORE/SPOT ratio."""
    if not os.path.exists(report_path):
        return None, None
    with open(report_path) as f:
        text = f.read()

    # Look for the data table — find lines that look like time data
    # Format: "    t(ps)   N_core  ... | E95_core ... | core/spot  xline/spot"
    peak_core_spot = 0
    peak_xline_spot = 0
    peak_core_e95 = 0

    in_table = False
    for line in text.splitlines():
        if 'core/spot' in line and 'xline/spot' in line:
            in_table = True
            continue
        if in_table and line.strip().startswith('-'):
            continue
        if in_table:
            parts = line.split('|')
            if len(parts) == 4:
                # First column has t(ps) and N_*
                # Second has E95s
                # Third has ratios
                try:
                    e95s = parts[1].split()
                    if len(e95s) >= 4:
                        core_e95 = float(e95s[0])
                        peak_core_e95 = max(peak_core_e95, core_e95)
                    ratios = parts[2].split()
                    if len(ratios) >= 2:
                        cs = float(ratios[0])
                        xs = float(ratios[1])
                        peak_core_spot = max(peak_core_spot, cs)
                        peak_xline_spot = max(peak_xline_spot, xs)
                except (ValueError, IndexError):
                    pass
            else:
                # End of table
                if line.strip().startswith('=') or 'TREND' in line:
                    break

    return peak_core_spot, peak_xline_spot, peak_core_e95


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='runs/paper02/rotating',
                        help='Directory containing p2_freq_*MHz subdirs')
    parser.add_argument('--zone-reports-dir', default='zone_reports',
                        help='Directory containing pre-generated zone reports')
    parser.add_argument('--report', default=None, help='Save report to this file')
    args = parser.parse_args()

    rows = []
    for freq in FREQUENCIES_MHZ:
        sub_tag = f'p2_freq_{freq}MHz'
        run_dir = os.path.join(args.base_dir, sub_tag)
        csv_path = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')
        zone_path = os.path.join(args.zone_reports_dir, f'{sub_tag}_zones.txt')

        if not os.path.exists(run_dir):
            rows.append({'freq': freq, 'error': 'run dir missing'})
            continue

        cum_alpha = get_csv_metric(csv_path, 'cum_alpha_yield_p11b')
        fusion_rate = get_csv_metric(csv_path, 'fusion_rate_p11b_s^-1')
        fusion_power = get_csv_metric(csv_path, 'fusion_power_p11b_w')
        Q = get_csv_metric(csv_path, 'gain_vs_laser_energy')
        fast_frac = get_csv_metric(csv_path, 'fast_fraction_gt_500kev_p11b')
        max_fusion_rate = get_csv_max_metric(csv_path, 'fusion_rate_p11b_s^-1')

        zone_data = parse_zone_report(zone_path) if os.path.exists(zone_path) else (None, None, None)
        peak_cs, peak_xs, peak_e95 = zone_data

        rows.append({
            'freq': freq,
            'cum_alpha': cum_alpha,
            'alpha_per_sr': cum_alpha / (4 * np.pi) if cum_alpha else None,
            'fusion_rate': fusion_rate,
            'max_fusion_rate': max_fusion_rate,
            'fusion_power': fusion_power,
            'Q': Q,
            'fast_frac': fast_frac,
            'peak_core_spot': peak_cs,
            'peak_xline_spot': peak_xs,
            'peak_core_e95_keV': peak_e95,
        })

    # Print resonance curve
    print('=' * 120)
    print('  PAPER A2 — FREQUENCY RESONANCE SCAN')
    print('=' * 120)
    print(f"  Predicted Alfvén sub-harmonic peaks: 208 MHz, 416 MHz")
    print()
    print(f"{'Freq':>6s} {'cum_α':>10s} {'α/sr':>10s} {'rate':>10s} "
          f"{'max_rate':>10s} {'power_W':>10s} {'Q':>8s} {'fast':>6s} "
          f"{'CORE/SPOT':>10s} {'X/SPOT':>8s} {'CORE_keV':>10s}")
    print('-' * 120)

    for r in rows:
        if 'error' in r:
            print(f"{r['freq']:>4d}MHz   ERROR: {r['error']}")
            continue
        marker = ''
        if r['freq'] in PREDICTED_PEAKS:
            marker = ' *'
        print(f"{r['freq']:>4d}MHz {r['cum_alpha']:>10.2e} {r['alpha_per_sr']:>10.2e} "
              f"{r['fusion_rate']:>10.2e} {r['max_fusion_rate']:>10.2e} "
              f"{r['fusion_power']:>10.2e} {r['Q']:>8.4f} {r['fast_frac']:>6.3f} "
              f"{r['peak_core_spot'] or 0:>10.2f} {r['peak_xline_spot'] or 0:>8.2f} "
              f"{r['peak_core_e95_keV'] or 0:>10.0f}{marker}")

    print('-' * 120)
    print('  * = predicted Alfvén sub-harmonic peak')

    # Identify actual peak
    valid_rows = [r for r in rows if 'error' not in r and r['cum_alpha']]
    if valid_rows:
        peak_alpha = max(valid_rows, key=lambda r: r['cum_alpha'])
        peak_Q = max(valid_rows, key=lambda r: r['Q'] or 0)
        peak_cs = max(valid_rows, key=lambda r: r['peak_core_spot'] or 0)

        print()
        print('  KEY RESULTS:')
        print(f"  Peak alpha yield:       {peak_alpha['freq']} MHz "
              f"({peak_alpha['cum_alpha']:.2e} alphas)")
        print(f"  Peak Q:                 {peak_Q['freq']} MHz "
              f"(Q={peak_Q['Q']:.4f})")
        print(f"  Peak CORE/SPOT ratio:   {peak_cs['freq']} MHz "
              f"({peak_cs['peak_core_spot']:.2f}×)")
        print()

        # Did predicted peaks match?
        for predicted_freq, label in PREDICTED_PEAKS.items():
            row = next((r for r in valid_rows if r['freq'] == predicted_freq), None)
            if row:
                print(f"  At {predicted_freq} MHz ({label}):")
                print(f"    cum_alpha = {row['cum_alpha']:.2e}, Q = {row['Q']:.4f}, "
                      f"CORE/SPOT = {row['peak_core_spot'] or 0:.2f}×")

    print('=' * 120)

    if args.report:
        # Recall print into file
        with open(args.report, 'w') as f:
            from contextlib import redirect_stdout
            with redirect_stdout(f):
                main_print = lambda: None  # already printed
                # Redirect would re-run; instead just write the rows
                pass


if __name__ == '__main__':
    main()
