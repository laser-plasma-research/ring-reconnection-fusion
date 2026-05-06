#!/usr/bin/env python3
"""
rod_compare.py — Analyze the Paper A3 rod fuel target sweep.

Compares 7 simulations:
  p3_no_inserts                   - baseline (no rod)
  p3_rod_ammonia                  - ammonia borane rod
  p3_rod_p11b                     - p-11B fuel rod
  p3_rod_sweep_5e23/5e24/5e25/5e26 - rod density sweep with ammonia

Question: does adding a central fuel rod boost yield by providing
  reaction substrate for inward-accelerated protons?

Usage:
    python rod_compare.py --base-dir runs/paper03/static
"""

import os, sys, argparse
import numpy as np

# Sims to compare, in logical groups
COMPOSITION_SIMS = [
    ('p3_no_inserts',    'no rod (baseline)',     None,    None),
    ('p3_rod_ammonia',   'ammonia borane @5e24',  'NH3BH3', 5e24),
    ('p3_rod_p11b',      'p-11B fuel rod @5e24',  'p11B',   5e24),
]

DENSITY_SCAN_SIMS = [
    ('p3_rod_sweep_5e23', 'NH3BH3 @5e23', 5e23),
    ('p3_rod_sweep_5e24', 'NH3BH3 @5e24', 5e24),
    ('p3_rod_sweep_5e25', 'NH3BH3 @5e25', 5e25),
    ('p3_rod_sweep_5e26', 'NH3BH3 @5e26', 5e26),
]


def get_csv_metric(csv_path, col_name, agg='last'):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    header = lines[0].strip().split(',')
    try:
        idx = header.index(col_name)
    except ValueError:
        return None
    values = []
    for line in lines[1:]:
        try:
            values.append(float(line.strip().split(',')[idx]))
        except (ValueError, IndexError):
            continue
    if not values:
        return None
    return values[-1] if agg == 'last' else max(values) if agg == 'max' else None


def parse_zone_report(report_path):
    if not os.path.exists(report_path):
        return None, None, None
    with open(report_path) as f:
        text = f.read()
    peak_cs, peak_xs, peak_e95 = 0, 0, 0
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
                try:
                    e95s = parts[1].split()
                    if len(e95s) >= 4:
                        peak_e95 = max(peak_e95, float(e95s[0]))
                    ratios = parts[2].split()
                    if len(ratios) >= 2:
                        peak_cs = max(peak_cs, float(ratios[0]))
                        peak_xs = max(peak_xs, float(ratios[1]))
                except (ValueError, IndexError):
                    pass
            else:
                if line.strip().startswith('=') or 'TREND' in line:
                    break
    return peak_cs, peak_xs, peak_e95


def fetch_metrics(run_dir, sub_tag, zone_dir):
    csv_path = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')
    zone_path = os.path.join(zone_dir, f'{sub_tag}_zones.txt')
    if not os.path.exists(csv_path):
        return None
    return {
        'cum_alpha':    get_csv_metric(csv_path, 'cum_alpha_yield_p11b'),
        'fusion_rate':  get_csv_metric(csv_path, 'fusion_rate_p11b_s^-1'),
        'fusion_power': get_csv_metric(csv_path, 'fusion_power_p11b_w'),
        'Q':            get_csv_metric(csv_path, 'gain_vs_laser_energy'),
        'fast_frac':    get_csv_metric(csv_path, 'fast_fraction_gt_500kev_p11b'),
        'max_rate':     get_csv_metric(csv_path, 'fusion_rate_p11b_s^-1', agg='max'),
        'zone':         parse_zone_report(zone_path),
    }


def fmt_metrics(m):
    if not m:
        return '   --- (no data)'
    cs, xs, e95 = m['zone']
    return (f"  cum_α={m['cum_alpha'] or 0:.2e}  Q={m['Q'] or 0:.4f}  "
            f"fast={m['fast_frac'] or 0:.3f}  power={m['fusion_power'] or 0:.2e}W\n"
            f"  CORE/SPOT={cs:.2f}×  X-LINE/SPOT={xs:.2f}×  CORE_E95={e95:.0f} keV")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='runs/paper03/static')
    parser.add_argument('--zone-reports-dir', default='zone_reports')
    args = parser.parse_args()

    print('=' * 100)
    print('  PAPER A3 — ROD FUEL TARGET COMPARISON')
    print('=' * 100)
    print()
    print('  PART 1: Composition comparison (no rod vs ammonia rod vs p-11B rod)')
    print('-' * 100)

    for sub_tag, label, _, _ in COMPOSITION_SIMS:
        run_dir = os.path.join(args.base_dir, sub_tag)
        m = fetch_metrics(run_dir, sub_tag, args.zone_reports_dir)
        print(f"\n  {label:<35s} ({sub_tag})")
        print(fmt_metrics(m))

    # Compute boost factor
    base_dir = os.path.join(args.base_dir, 'p3_no_inserts')
    base_m = fetch_metrics(base_dir, 'p3_no_inserts', args.zone_reports_dir)

    if base_m and base_m['cum_alpha']:
        print()
        print('  Boost factors (vs no_inserts baseline):')
        for sub_tag, label, _, _ in COMPOSITION_SIMS:
            if sub_tag == 'p3_no_inserts':
                continue
            m = fetch_metrics(os.path.join(args.base_dir, sub_tag), sub_tag, args.zone_reports_dir)
            if m and m['cum_alpha']:
                ratio = m['cum_alpha'] / base_m['cum_alpha']
                print(f"    {label:<35s}: {ratio:.2f}× alpha yield boost")

    print()
    print('=' * 100)
    print('  PART 2: Rod density sweep (NH3BH3, varying density)')
    print('-' * 100)

    rows = []
    for sub_tag, label, density in DENSITY_SCAN_SIMS:
        run_dir = os.path.join(args.base_dir, sub_tag)
        m = fetch_metrics(run_dir, sub_tag, args.zone_reports_dir)
        rows.append((sub_tag, label, density, m))

    print(f"{'density (m^-3)':<15s} {'cum_α':>10s} {'rate':>10s} "
          f"{'power_W':>10s} {'Q':>8s} {'CORE_E95_keV':>12s}")
    print('-' * 80)
    for sub_tag, label, density, m in rows:
        if not m:
            print(f"{density:<15.1e}   ERROR")
            continue
        cs, xs, e95 = m['zone']
        print(f"{density:<15.1e} {m['cum_alpha'] or 0:>10.2e} "
              f"{m['fusion_rate'] or 0:>10.2e} {m['fusion_power'] or 0:>10.2e} "
              f"{m['Q'] or 0:>8.4f} {e95:>12.0f}")

    print('-' * 80)

    valid = [(d, m) for s, l, d, m in rows if m and m['cum_alpha']]
    if len(valid) >= 2:
        best_d, best_m = max(valid, key=lambda x: x[1]['cum_alpha'])
        print(f"\n  Optimal rod density: {best_d:.1e}  (cum_α = {best_m['cum_alpha']:.2e})")

        # Density scaling
        print('\n  Density scaling: alpha yield per increment in rod density')
        for i in range(1, len(valid)):
            d_prev, m_prev = valid[i-1]
            d_curr, m_curr = valid[i]
            dr = d_curr / d_prev
            ar = (m_curr['cum_alpha'] / m_prev['cum_alpha']) if m_prev['cum_alpha'] else float('inf')
            print(f"    {d_prev:.0e} → {d_curr:.0e}: density {dr:.0f}×, "
                  f"alpha yield {ar:.2f}×")

    print('=' * 100)


if __name__ == '__main__':
    main()
