#!/usr/bin/env python3
"""
alpha_yield_per_sr.py — Compute alpha particle yield per steradian.

For comparison against published p-11B experimental yields:
  Belyaev (2005):    1e5 α/sr/shot
  Labaune (2013):    1e6 α/sr/shot
  Picciotto (2014):  1e9 α/sr/shot
  Giuffrida (2020):  1e10 α/sr/shot
  Bonvalet (2021):   1e11 α/sr/shot   <- current record

For each sim, this:
  1. Reads the cumulative alpha yield from the last row of fusion_rate_power_by_iter.csv
     (this is the total volume-integrated alpha count over the simulation duration)
  2. Reads alpha angular distribution at end-of-simulation from particle dump
  3. Computes α/sr at end of run AND total cumulative α/sr
  4. Normalizes per Joule of effective laser energy

Note: simulation alphas are released into 4π solid angle from each fusion event.
For uniform emission: α/sr = total_alphas / (4π).
For anisotropic: bin by direction, find peak.

Usage:
    python alpha_yield_per_sr.py --dir runs/paper01/static/p1_low_density
    python alpha_yield_per_sr.py --batch runs/paper0*/static/* runs/paper0*/rotating/*
"""

import os, sys, argparse, glob
import numpy as np

PROTON_MASS_KG = 1.67262e-27
ALPHA_MASS_KG  = 6.6446573e-27
Q_E            = 1.602e-19
KEV_TO_J       = Q_E * 1e3
LASER_ENERGY_J = 5.0  # 5 J total across 8 spots (per project doc)

LITERATURE = {
    'Belyaev (2005)':    1e5,
    'Labaune (2013)':    1e6,
    'Picciotto (2014)':  1e9,
    'Giuffrida (2020)':  1e10,
    'Bonvalet (2021)':   1e11,
}


def analyze_one(run_dir):
    """Compute alpha yield metrics for a single run."""
    sub_tag = os.path.basename(run_dir.rstrip('/'))

    # Read cumulative alpha yield from CSV
    csv_path = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')
    if not os.path.exists(csv_path):
        return {'sub_tag': sub_tag, 'error': 'no fusion CSV'}

    with open(csv_path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return {'sub_tag': sub_tag, 'error': 'CSV has no data rows'}

    header = lines[0].strip().split(',')
    last = lines[-1].strip().split(',')

    def col(name):
        try:
            return float(last[header.index(name)])
        except (ValueError, IndexError):
            return None

    cum_alpha = col('cum_alpha_yield_p11b')
    if cum_alpha is None or cum_alpha < 1:
        # Try the alternate column name
        cum_alpha = col('alpha_yield_step_total')

    final_time_ns = col('time_ns')
    fast_frac = col('fast_fraction_gt_500kev_p11b')
    fusion_rate = col('fusion_rate_p11b_s^-1')
    fusion_power = col('fusion_power_p11b_w')
    Q_value = col('gain_vs_laser_energy')

    # Compute α/sr assuming isotropic 4π emission (good first approximation)
    # In reality, p-11B alphas have some anisotropy but isotropic-equivalent
    # is the standard normalization for comparison.
    if cum_alpha is None:
        alpha_per_sr = None
    else:
        alpha_per_sr = cum_alpha / (4 * np.pi)
        alpha_per_sr_per_J = alpha_per_sr / LASER_ENERGY_J

    return {
        'sub_tag': sub_tag,
        'cum_alpha': cum_alpha,
        'alpha_per_sr': alpha_per_sr,
        'alpha_per_sr_per_J': alpha_per_sr_per_J if cum_alpha else None,
        'final_time_ns': final_time_ns,
        'fast_frac': fast_frac,
        'fusion_rate': fusion_rate,
        'fusion_power_w': fusion_power,
        'Q': Q_value,
    }


def print_one(r):
    if 'error' in r:
        print(f"{r['sub_tag']:30s}: ERROR: {r['error']}")
        return
    print(f"\n{'='*80}")
    print(f"  {r['sub_tag']}")
    print(f"{'='*80}")
    print(f"  Final simulation time:      {r['final_time_ns']:.3f} ns")
    print(f"  Cumulative alpha yield:     {r['cum_alpha']:.3e} alphas (volume-integrated)")
    print(f"  Alpha per steradian:        {r['alpha_per_sr']:.3e} α/sr  (assuming 4π isotropic)")
    print(f"  Alpha per sr per Joule:     {r['alpha_per_sr_per_J']:.3e} α/sr/J  (laser=5J)")
    print(f"  Fast-ion fraction (>500keV):{r['fast_frac']:.3f}")
    print(f"  Fusion rate:                {r['fusion_rate']:.3e} /s")
    print(f"  Fusion power:               {r['fusion_power_w']:.3e} W")
    print(f"  Q (gain vs laser energy):   {r['Q']:.4f}")

    # Compare to literature
    print(f"\n  vs Literature (α/sr/shot):")
    for ref, ref_value in LITERATURE.items():
        ratio = r['alpha_per_sr'] / ref_value
        flag = '  +' if ratio >= 1 else '  -'
        print(f"  {flag} {ref:20s}: {ref_value:.0e} α/sr/shot   ({ratio:.2g}× our value)")


def print_summary_table(results):
    """Print compact comparison table across all sims."""
    print(f"\n{'='*120}")
    print(f"  SUMMARY TABLE — {len(results)} simulations")
    print(f"{'='*120}")
    print(f"{'sub_tag':<30s} {'time_ns':>8s} {'cum_α':>10s} {'α/sr':>10s} {'α/sr/J':>10s} "
          f"{'fast_frac':>10s} {'fusion_W':>10s} {'Q':>8s}")
    print(f"{'-'*120}")
    for r in sorted(results, key=lambda x: x.get('alpha_per_sr', 0) or 0, reverse=True):
        if 'error' in r:
            print(f"{r['sub_tag']:<30s}   ERROR: {r['error']}")
            continue
        print(f"{r['sub_tag']:<30s} {r['final_time_ns']:>8.3f} "
              f"{r['cum_alpha']:>10.2e} {r['alpha_per_sr']:>10.2e} "
              f"{r['alpha_per_sr_per_J']:>10.2e} {r['fast_frac']:>10.3f} "
              f"{r['fusion_power_w']:>10.2e} {r['Q']:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', help='Single run directory')
    parser.add_argument('--batch', nargs='+', help='Multiple run directories (glob ok)')
    parser.add_argument('--report', default=None, help='Save summary to this file')
    args = parser.parse_args()

    if args.dir:
        dirs = [args.dir]
    elif args.batch:
        dirs = []
        for pattern in args.batch:
            dirs.extend(glob.glob(pattern))
        dirs = sorted(set(dirs))
    else:
        print('ERROR: Provide --dir or --batch')
        sys.exit(1)

    print(f"\nAnalyzing {len(dirs)} run directories...")

    results = []
    for d in dirs:
        if os.path.isdir(d):
            r = analyze_one(d)
            results.append(r)

    # Detail per sim
    for r in results:
        print_one(r)

    # Summary table
    print_summary_table(results)

    if args.report:
        # Re-run as a write to file
        with open(args.report, 'w') as f:
            from contextlib import redirect_stdout
            with redirect_stdout(f):
                for r in results:
                    print_one(r)
                print_summary_table(results)
        print(f"\nReport saved to {args.report}")


if __name__ == '__main__':
    main()
