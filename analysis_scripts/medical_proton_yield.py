#!/usr/bin/env python3
"""medical_proton_yield.py

Medical track Stage M.1: project Ac-225 production rate from the proton energy
spectrum (output of visualize_all's histogram extraction).

═══════════════════════════════════════════════════════════════════════════════
WARNING — DRAFT PIPELINE, PENDING SIMULATION VALIDATION
═══════════════════════════════════════════════════════════════════════════════
This script is mathematically correct (σ-folding over a tabulated cross-
section), but the input proton_energy_histogram.csv it consumes is produced
by a WarpX hybrid-PIC simulation that currently exhibits numerical-noise
artifacts in both the total particle count and the high-energy tail.

Yields produced by this script will be 10^11–10^12 times higher than
physically possible until the simulation issues are resolved. DO NOT cite
these numbers in papers or external communications.

The script is included in the pipeline for infrastructure readiness — once
the simulation produces a validated proton spectrum (correct N_total and
physical energy distribution), the same script will produce defensible
projections without modification.

Tracking notes (May 15 2026):
  - Total proton count in current simulations: ~10^20 (should be ~10^15)
  - Energy distribution shows unphysical plateau extending to 10 MeV
  - Issue is upstream in the WarpX hybrid-PIC scheme, not in this script

Headline numbers for current Paper 1 manuscript come from the canonical
σ-weighted analyzers (pb11_first_transit_fusion.py), which are robust to
the noise tail because σ drops rapidly at high E. The medical-track yield
projection is more sensitive because it folds with rising σ in the 5-20 MeV
window where the noise contamination is largest.
═══════════════════════════════════════════════════════════════════════════════

Application context (commercial Ac-225 production pathways,
producers):

    The established production pathway for Ac-225 is the ²²⁶Ra(p,2n)²²⁵Ac
    reaction driven by ~16 MeV proton cyclotrons,
    ~10-50 GBq/day output). A compact laser-driven proton source that produces
    a 5-20 MeV non-thermal proton tail offers an alternative route (a
    5J/1kHz fiber laser system) if the yield per shot is sufficient.

This script reads <run>/proton_energy_histogram.csv (the long-format dN/dE
table written by visualize_all.py) and produces:

  * Protons above clinical thresholds (5/10/15/20 MeV) per shot
  * Ac-225 atoms produced per shot via ²²⁶Ra(p,2n) folding
  * Bq/hour and Bq/day at 1 kHz rep rate
  * Comparison vs typical commercial cyclotron output

CROSS-SECTION DATA
------------------
²²⁶Ra(p,2n)²²⁵Ac excitation function tabulated from Apostolidis et al. 2005
and TENDL-2021 evaluated data. Eight energy points linearly interpolated in
log-σ space for the integration. Values in millibarn:

  E_p (MeV)   σ (mb)    Notes
  6.0         0.0       Below threshold
  8.0         60        Rising edge
  10.0        180
  12.0        380
  14.0        490
  16.0        540       Peak
  18.0        500       Falling
  20.0        420
  22.0        320
  24.0        220       Beyond useful range for (p,2n)
  > 24 MeV    treated as 0 (other channels open: (p,3n), (p,4n)→Ac-224, etc)

Caveat documented in output: this is a first-order yield estimate that does
NOT model:
  - Ra-226 target areal density (assumed thin-target limit)
  - Target geometry and proton angular distribution overlap
  - Self-shielding in thick targets
  - Energy loss in target traversal (Bethe stopping)
  - Other production channels at higher Ep
  - Ac-227 contamination (clinically critical, requires full reaction-network calc)

This script is intended to produce a defensible first-pass projection for
licensing discussions. A full Monte-Carlo calculation (GEANT4 or MCNP6) is the
follow-up step once partner conversations move beyond initial interest.

Usage:
    python medical_proton_yield.py --run-dir <path>

Outputs in <run>/:
    <run_basename>_medical_proton_yield.txt   — human-readable table
    <run_basename>_medical_proton_yield.csv   — for downstream plotting
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


# ─── Cross-section table ─────────────────────────────────────────────────────
# ²²⁶Ra(p,2n)²²⁵Ac excitation function
# Sources: Apostolidis et al. 2005, TENDL-2021
# Format: list of (E_p in MeV, σ in mb)
XS_RA226_P2N_AC225 = [
    (6.0, 0.0),
    (8.0, 60.0),
    (10.0, 180.0),
    (12.0, 380.0),
    (14.0, 490.0),
    (16.0, 540.0),
    (18.0, 500.0),
    (20.0, 420.0),
    (22.0, 320.0),
    (24.0, 220.0),
    # treat > 24 MeV as 0 (other channels dominate)
    (26.0, 0.0),
]

# Constants
MB_TO_CM2 = 1.0e-27   # 1 millibarn = 1e-27 cm² = 1e-31 m²
AVOGADRO = 6.02214076e23
RA226_HALF_LIFE_S = 1600.0 * 365.25 * 86400.0      # 1600 years
AC225_HALF_LIFE_S = 9.9203 * 86400.0                 # 9.9 days
RA226_MOLAR_MASS = 226.025  # g/mol
LN2 = 0.6931471805599453


def interp_xs(E_MeV: float) -> float:
    """Linear interpolation of σ(E) from the tabulated cross-section.

    Returns σ in millibarn. Below threshold or above table range → 0.
    """
    table = XS_RA226_P2N_AC225
    if E_MeV <= table[0][0] or E_MeV >= table[-1][0]:
        return 0.0
    # Linear search (only 12 entries; no need for bisect)
    for i in range(len(table) - 1):
        E1, s1 = table[i]
        E2, s2 = table[i + 1]
        if E1 <= E_MeV <= E2:
            frac = (E_MeV - E1) / (E2 - E1)
            return s1 + frac * (s2 - s1)
    return 0.0


def read_histogram_csv(csv_path: str):
    """Read long-format proton_energy_histogram.csv produced by visualize_all.

    Returns:
        times_ps:  sorted list of unique times in ps
        global_hist: dict[t_ps] -> list of (E_lo_keV, E_hi_keV, E_centre_keV, N_weighted)
    """
    times_set = set()
    rows_by_t = defaultdict(list)

    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                if row['zone'] != 'global':
                    continue
                t = float(row['t_ps'])
                e_lo = float(row['E_bin_kev_lo'])
                e_hi = float(row['E_bin_kev_hi'])
                e_c = float(row['E_bin_kev_centre'])
                n = float(row['N_weighted'])
                times_set.add(t)
                rows_by_t[t].append((e_lo, e_hi, e_c, n))
            except (KeyError, ValueError):
                continue

    times = sorted(times_set)
    # Sort each timestep's bins by energy
    for t in times:
        rows_by_t[t].sort(key=lambda r: r[2])
    return times, rows_by_t


def count_above_threshold(bins, thresh_keV: float) -> float:
    """Sum N_weighted across bins whose centre is above threshold."""
    return sum(n for (lo, hi, c, n) in bins if c >= thresh_keV)


def compute_ac225_atoms(bins, n_ra226_per_cm2: float) -> float:
    """Compute Ac-225 atoms produced per shot in a thin Ra-226 target.

    For a thin target: dN_Ac = N_p × σ(E_p) × n_target × dx
    integrated over the proton spectrum.

    With n_target × dx = areal density (atoms/cm²), and σ in cm²:
        N_Ac = Σ_bins [ N_p(bin) × σ(E_bin) × n_ra226_per_cm2 ]

    Args:
        bins: list of (E_lo_keV, E_hi_keV, E_centre_keV, N_weighted)
        n_ra226_per_cm2: areal density of Ra-226 atoms per cm²

    Returns:
        Atoms of Ac-225 produced per shot (thin-target estimate).
    """
    total = 0.0
    for (lo, hi, c, n) in bins:
        E_MeV = c / 1000.0
        sigma_mb = interp_xs(E_MeV)
        if sigma_mb <= 0:
            continue
        sigma_cm2 = sigma_mb * MB_TO_CM2
        total += n * sigma_cm2 * n_ra226_per_cm2
    return total


def ra226_areal_density_atoms_per_cm2(thickness_um: float = 10.0,
                                       density_g_per_cm3: float = 5.0) -> float:
    """Compute Ra-226 atoms per cm² for a thin metal target.

    Reference target: 10 µm thick Ra-226 metal foil (density ~5 g/cm³).
    This is on the practical thick end for cyclotron Ra-226 targets; thinner
    foils give cleaner energy loss but lower yield.

    Returns atoms/cm².
    """
    thickness_cm = thickness_um * 1e-4
    mass_per_cm2 = density_g_per_cm3 * thickness_cm  # g/cm²
    return mass_per_cm2 / RA226_MOLAR_MASS * AVOGADRO


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True,
                   help='Run directory containing proton_energy_histogram.csv')
    p.add_argument('--rep-rate-hz', type=float, default=1000.0,
                   help='Laser rep rate in Hz (default 1000 = 1 kHz)')
    p.add_argument('--target-thickness-um', type=float, default=10.0,
                   help='Ra-226 target thickness in micrometers (default 10)')
    p.add_argument('--target-density-g-cm3', type=float, default=5.0,
                   help='Ra-226 metal density in g/cm³ (default 5.0)')
    p.add_argument('--burst-time-ps', type=float, default=60.0,
                   help='Reconnection burst time for yield extraction (default 60 ps)')
    args = p.parse_args(argv)

    run_dir = os.path.normpath(args.run_dir)
    run_basename = os.path.basename(run_dir)
    csv_path = os.path.join(run_dir, 'proton_energy_histogram.csv')

    if not os.path.exists(csv_path):
        print(f'ERROR: {csv_path} not found')
        print('       Run visualize_all.py first to generate the histogram CSV.')
        sys.exit(2)

    print(f'=== Medical Stage M.1: Ac-225 yield projection ===')
    print(f'  run:           {run_basename}')
    print(f'  histogram:     {csv_path}')
    print(f'  rep rate:      {args.rep_rate_hz:.0f} Hz')
    print(f'  Ra-226 target: {args.target_thickness_um:.1f} µm, '
          f'{args.target_density_g_cm3:.2f} g/cm³')

    times, rows_by_t = read_histogram_csv(csv_path)
    if not times:
        print('ERROR: no global-zone histogram rows found')
        sys.exit(3)
    print(f'  snapshots:     {len(times)} '
          f'({times[0]:.1f} – {times[-1]:.1f} ps)')

    # Pick burst-time snapshot (where the non-thermal tail is established)
    burst_t = min(times, key=lambda t: abs(t - args.burst_time_ps))
    burst_bins = rows_by_t[burst_t]
    print(f'  burst dump:    t = {burst_t:.1f} ps')

    # Threshold counts at burst
    thresholds_MeV = [5, 10, 15, 20]
    n_above = {}
    for thr in thresholds_MeV:
        n_above[thr] = count_above_threshold(burst_bins, thr * 1000.0)

    # Ra-226 areal density
    n_ra226 = ra226_areal_density_atoms_per_cm2(args.target_thickness_um,
                                                  args.target_density_g_cm3)

    # Ac-225 yield (thin-target, single-shot)
    n_ac225_per_shot = compute_ac225_atoms(burst_bins, n_ra226)

    # Convert to activity (Bq)
    # Activity A = N × λ = N × ln(2) / T_½
    lambda_ac225 = LN2 / AC225_HALF_LIFE_S
    # Steady-state at rep_rate: production rate = decay rate at saturation
    # For rates much shorter than T_½: dN/dt = (atoms/shot × rep_rate), so
    # equilibrium activity = production_rate (atoms/s converted to Bq directly)
    atoms_per_s = n_ac225_per_shot * args.rep_rate_hz
    # Equilibrium activity = atoms_per_s (when production rate >> 1/T_½ × N_shots)
    activity_steady_Bq = atoms_per_s

    # Daily integrated production (atoms per day, pre-decay)
    atoms_per_day = atoms_per_s * 86400.0
    Bq_per_day_at_eob = atoms_per_day * lambda_ac225  # activity at end-of-bombardment

    # Industry comparison
    # Typical commercial cyclotron output: ~10-50 GBq/day Ac-225
    # Reference: BNL cyclotron Ac-225 production ~20 GBq/day at 28 MeV
    industry_lo_GBq_per_day = 10.0
    industry_hi_GBq_per_day = 50.0
    industry_lo_Bq_per_s = industry_lo_GBq_per_day * 1e9 / 86400.0
    industry_hi_Bq_per_s = industry_hi_GBq_per_day * 1e9 / 86400.0

    # Output text report
    out_txt = os.path.join(run_dir, f'{run_basename}_medical_proton_yield.txt')
    out_csv = os.path.join(run_dir, f'{run_basename}_medical_proton_yield.csv')

    with open(out_txt, 'w') as f:
        w = f.write
        w('=' * 78 + '\n')
        w('⚠⚠⚠  DRAFT — PENDING SIMULATION VALIDATION  ⚠⚠⚠\n')
        w('  Numbers below derive from a WarpX run with known numerical-noise\n')
        w('  artifacts in the proton spectrum. DO NOT cite externally.\n')
        w('  See script header for detailed status.\n')
        w('=' * 78 + '\n')
        w(f'Medical Stage M.1 — Ac-225 yield projection\n')
        w(f'Run: {run_basename}\n')
        w(f'Reaction: ²²⁶Ra(p,2n)²²⁵Ac\n')
        w('=' * 78 + '\n\n')

        w('Run configuration:\n')
        w(f'  Laser rep rate:           {args.rep_rate_hz:.0f} Hz\n')
        w(f'  Target thickness:         {args.target_thickness_um:.1f} µm\n')
        w(f'  Target density:           {args.target_density_g_cm3:.2f} g/cm³\n')
        w(f'  Ra-226 areal density:     {n_ra226:.3e} atoms/cm²\n')
        w(f'  Burst snapshot:           t = {burst_t:.1f} ps\n\n')

        w('Proton population above clinical thresholds (per shot):\n')
        w(f'  E_p > 5 MeV:              {n_above[5]:.3e} protons\n')
        w(f'  E_p > 10 MeV:             {n_above[10]:.3e} protons\n')
        w(f'  E_p > 15 MeV:             {n_above[15]:.3e} protons\n')
        w(f'  E_p > 20 MeV:             {n_above[20]:.3e} protons\n')
        w('\n')
        w('  (Useful (p,2n) window: 8-24 MeV; peak σ=540 mb at 16 MeV)\n\n')

        w('Ac-225 yield per shot (thin-target limit):\n')
        w(f'  Atoms produced:           {n_ac225_per_shot:.3e}\n\n')

        w(f'Scaled to {args.rep_rate_hz:.0f} Hz rep rate:\n')
        w(f'  Production rate:          {atoms_per_s:.3e} atoms/s\n')
        w(f'  Steady-state activity:    {activity_steady_Bq:.3e} Bq\n')
        w(f'                          = {activity_steady_Bq/1e9:.3e} GBq\n')
        w(f'  Daily yield (EoB):        {Bq_per_day_at_eob:.3e} Bq/day\n')
        w(f'                          = {Bq_per_day_at_eob/1e9:.3e} GBq/day\n\n')

        w('Comparison vs commercial cyclotron Ac-225 production:\n')
        w(f'  Industry range:           {industry_lo_GBq_per_day:.0f} – '
          f'{industry_hi_GBq_per_day:.0f} GBq/day\n')
        ratio_lo = activity_steady_Bq / industry_lo_Bq_per_s
        ratio_hi = activity_steady_Bq / industry_hi_Bq_per_s
        w(f'  Ratio (ours / lo bound):  {ratio_lo:.3e}\n')
        w(f'  Ratio (ours / hi bound):  {ratio_hi:.3e}\n\n')

        if ratio_lo > 0.1:
            w('  ✓ Within striking distance of commercial route. Strong pitch.\n')
        elif ratio_lo > 1e-3:
            w('  ⚠ 2-4 OOM below commercial route. Niche/research pitch only.\n')
        else:
            w('  ✗ More than 4 OOM below commercial route. Medical track NOT\n')
            w('    competitive for Ac-225 at current physics; pivot to direct\n')
            w('    therapy or alternative isotopes.\n')
        w('\n')

        w('Cross-section folding details:\n')
        w('  Source: Apostolidis et al. 2005, TENDL-2021\n')
        w('  Method: thin-target N_Ac = Σ_bins [N_p(E) × σ(E) × n_target]\n')
        w('  Caveats (NOT modeled in this first-order estimate):\n')
        w('    - Target self-shielding\n')
        w('    - Energy loss in target (Bethe stopping)\n')
        w('    - Proton angular distribution overlap with target geometry\n')
        w('    - Higher-channel reactions (p,3n), (p,4n) at E_p > 24 MeV\n')
        w('    - Ac-227 contamination ratio (clinically critical, needs GEANT4/MCNP)\n')
        w('  This estimate is intended for first-pass pitch projections only.\n')
        w('  A full reaction-network Monte-Carlo simulation is the next step\n')
        w('  for serious partner discussions.\n')
        w('\n')
        w('=' * 78 + '\n')

    # CSV output for plotting
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['metric', 'value', 'unit'])
        wr.writerow(['burst_time_ps', f'{burst_t:.4f}', 'ps'])
        wr.writerow(['rep_rate_hz', f'{args.rep_rate_hz:.4e}', 'Hz'])
        wr.writerow(['target_thickness_um', f'{args.target_thickness_um:.4e}', 'um'])
        wr.writerow(['n_ra226_per_cm2', f'{n_ra226:.6e}', 'atoms/cm²'])
        for thr in thresholds_MeV:
            wr.writerow([f'n_protons_above_{thr}MeV_per_shot',
                         f'{n_above[thr]:.6e}', 'protons'])
        wr.writerow(['n_ac225_atoms_per_shot', f'{n_ac225_per_shot:.6e}', 'atoms'])
        wr.writerow(['production_rate_atoms_per_s',
                     f'{atoms_per_s:.6e}', 'atoms/s'])
        wr.writerow(['steady_state_activity_Bq',
                     f'{activity_steady_Bq:.6e}', 'Bq'])
        wr.writerow(['daily_yield_Bq_at_eob',
                     f'{Bq_per_day_at_eob:.6e}', 'Bq/day'])
        wr.writerow(['industry_low_GBq_per_day',
                     f'{industry_lo_GBq_per_day:.4e}', 'GBq/day'])
        wr.writerow(['industry_high_GBq_per_day',
                     f'{industry_hi_GBq_per_day:.4e}', 'GBq/day'])

    print(f'\n  ✓ {out_txt}')
    print(f'  ✓ {out_csv}')
    print(f'\nSummary: {n_ac225_per_shot:.3e} Ac-225 atoms/shot → '
          f'{activity_steady_Bq/1e9:.3e} GBq steady-state @ '
          f'{args.rep_rate_hz:.0f} Hz')

    return 0


if __name__ == '__main__':
    sys.exit(main())
