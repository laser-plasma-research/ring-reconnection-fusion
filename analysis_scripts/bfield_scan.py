#!/usr/bin/env python3
"""
bfield_scan.py — Analyze the Paper A2 B-field sensitivity scan.

Looks at 6 static simulations with different seed B-fields:
  100, 150, 200, 300, 400 T (and the static_baseline at 300T)

Question: How does the seed B-field amplitude affect:
  - Reconnection acceleration (CORE energies)
  - Final fusion yield
  - Whether physics regime crosses from soft (low-β) to stiff (high-β)?

Project doc note: 85T is well-conditioned (β≈4), 300T is stiff (β<1).
This scan explores both regimes.

Usage:
    python bfield_scan.py --base-dir runs/paper02/static
"""

import os, sys, argparse
import numpy as np

# B-field values in scan (T)
B_VALUES_T = [100, 150, 200, 300, 400]

# Compute Alfvén speed and beta for each
N_PEAK = 5e24      # m^-3
T_ION_J = 660 * 1.602e-19   # 660 eV in Joules
T_ELEC_J = 2200 * 1.602e-19 # 2200 eV in Joules
PROTON_MASS = 1.67262e-27
B11_MASS = 11.0093 * 1.66054e-27
MU_0 = 4 * np.pi * 1e-7

def alfven_speed(B, n_p, n_b11):
    rho = n_p * PROTON_MASS + n_b11 * B11_MASS
    return B / np.sqrt(MU_0 * rho)

def plasma_beta(B, n_p, n_b11, T_p, T_b11, T_e):
    p_thermal = n_p * T_p + n_b11 * T_b11 + (n_p + 5*n_b11) * T_e
    p_magnetic = B**2 / (2 * MU_0)
    return p_thermal / p_magnetic


def get_csv_metric(csv_path, col_name, agg='last'):
    """Read a metric from CSV. agg can be 'last' or 'max'."""
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
            v = float(line.strip().split(',')[idx])
            values.append(v)
        except (ValueError, IndexError):
            continue
    if not values:
        return None
    if agg == 'last':
        return values[-1]
    elif agg == 'max':
        return max(values)
    elif agg == 'mean':
        return sum(values) / len(values)
    return None


def parse_zone_report(report_path):
    """Extract peak CORE/SPOT, XLINE/SPOT, and CORE E95 from zone report."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-dir', default='runs/paper02/static',
                        help='Directory containing p2_bseed_*T subdirs')
    parser.add_argument('--zone-reports-dir', default='zone_reports')
    args = parser.parse_args()

    n_p = N_PEAK
    n_b11 = N_PEAK / 5

    print('=' * 130)
    print('  PAPER A2 — B-FIELD SEED SENSITIVITY SCAN')
    print('=' * 130)
    print(f"  Plasma parameters (peak): n_p={n_p:.1e}, n_B11={n_b11:.1e}, "
          f"T_p={660} eV, T_e={2200} eV")
    print()
    print(f"{'B (T)':>6s} {'v_Alfven':>10s} {'β':>8s} | "
          f"{'cum_α':>10s} {'α/sr':>10s} {'rate':>10s} {'power':>10s} {'Q':>8s} | "
          f"{'CORE/SPOT':>10s} {'X/SPOT':>8s} {'CORE_keV':>10s}")
    print('-' * 130)

    rows = []
    for B in B_VALUES_T:
        sub_tag = f'p2_bseed_{B}T'
        run_dir = os.path.join(args.base_dir, sub_tag)
        csv_path = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')
        zone_path = os.path.join(args.zone_reports_dir, f'{sub_tag}_zones.txt')

        if not os.path.exists(run_dir):
            print(f"{B:>4d}T   ERROR: missing")
            continue

        cum_alpha = get_csv_metric(csv_path, 'cum_alpha_yield_p11b')
        fusion_rate = get_csv_metric(csv_path, 'fusion_rate_p11b_s^-1')
        fusion_power = get_csv_metric(csv_path, 'fusion_power_p11b_w')
        Q = get_csv_metric(csv_path, 'gain_vs_laser_energy')

        peak_cs, peak_xs, peak_e95 = parse_zone_report(zone_path) if os.path.exists(zone_path) else (0, 0, 0)

        v_a = alfven_speed(B, n_p, n_b11) / 1e6  # Mm/s
        beta = plasma_beta(B, n_p, n_b11, T_ION_J, T_ION_J, T_ELEC_J)

        alpha_per_sr = cum_alpha / (4 * np.pi) if cum_alpha else 0

        regime = '(stiff)' if beta < 1 else '(soft)'

        print(f"{B:>4d}T  {v_a:>9.2f}M  {beta:>6.2f}{regime:>4s} | "
              f"{cum_alpha or 0:>10.2e} {alpha_per_sr:>10.2e} "
              f"{fusion_rate or 0:>10.2e} {fusion_power or 0:>10.2e} {Q or 0:>8.4f} | "
              f"{peak_cs:>10.2f} {peak_xs:>8.2f} {peak_e95:>10.0f}")

        rows.append({'B': B, 'beta': beta, 'cum_alpha': cum_alpha,
                     'Q': Q, 'peak_cs': peak_cs})

    print('-' * 130)
    print('  Regime: β > 1 = soft (well-conditioned), β < 1 = stiff (magnetic-pressure dominant)')
    print()

    valid = [r for r in rows if r.get('cum_alpha')]
    if valid:
        # Sweet spot
        best_alpha = max(valid, key=lambda r: r['cum_alpha'])
        best_cs = max(valid, key=lambda r: r['peak_cs'] or 0)
        print(f'  Peak alpha yield:    B = {best_alpha["B"]} T  ({best_alpha["cum_alpha"]:.2e} alphas)')
        print(f'  Peak CORE/SPOT:      B = {best_cs["B"]} T  ({best_cs["peak_cs"]:.2f}×)')

        # β vs Q correlation
        print()
        print('  β scaling:')
        for r in valid:
            print(f'    B={r["B"]}T:  β={r["beta"]:.2f}, Q={r["Q"]:.4f}, '
                  f'CORE/SPOT={r["peak_cs"]:.2f}')

    print('=' * 130)


if __name__ == '__main__':
    main()
