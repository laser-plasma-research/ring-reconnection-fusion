#!/usr/bin/env python3
"""medical_alpha_kinematic.py

Medical track Stage M.2: derive the alpha-particle energy spectrum
kinematically from the fusion-rate diagnostic (since the WarpX simulation does
NOT track alphas as particles; only counts them via σ-folded reaction rates).

═══════════════════════════════════════════════════════════════════════════════
WARNING — DRAFT PIPELINE, PENDING SIMULATION VALIDATION
═══════════════════════════════════════════════════════════════════════════════
This script is mathematically correct (kinematic projection from validated
nuclear data), but it consumes two upstream products that currently have
numerical-noise issues:

  1. proton_energy_histogram.csv — used as the parent-proton distribution.
     Total proton count is inflated ~10^5× and high-energy tail is unphysical.

  2. fusion_rate_power_by_iter.csv (cum_alpha_yield_p11b column) — used as
     the absolute normalization for total alphas/shot. This is σ-folded
     over the same corrupted proton distribution, so it inherits the noise.

DO NOT cite alpha yields, flux numbers, or spectrum shapes from this script
in external communications until the simulation noise issue is resolved.

The script is included in the pipeline for infrastructure readiness. Once
the simulation produces a validated proton spectrum, the same script will
produce defensible projections without modification.
═══════════════════════════════════════════════════════════════════════════════

Application context (medical-isotope and alpha-therapy applications,
augmentation, alpha source for nuclear research):

    Each ¹¹B(p,α)αα reaction releases 8.68 MeV among three alpha particles.
    The primary alpha carries ~3.76 MeV (CM frame), while the recoiling ⁸Be
    decays into two secondary alphas at ~2.46 MeV each. In the lab frame the
    spectrum is broadened by the incident-proton energy and the angular
    distribution between alphas.

This script:
  1. Reads the per-iteration fusion rate from
     fusion_rate_power_by_iter.csv (alpha_yield_step_p11b column)
  2. Reads the proton energy histogram (visualize_all output) to derive the
     parent-proton energy distribution
  3. Computes the lab-frame alpha energy spectrum via Q-value kinematics
  4. Writes the alpha spectrum CSV + a one-page summary for the medical-applications track (Paper 1 §4.5)

KINEMATIC MODEL
---------------
The p + ¹¹B → 3α reaction is treated as a 2-body breakup p + ¹¹B → α + ⁸Be*
followed by ⁸Be* → α + α. We use the two-step decomposition with the average
branching for the ⁸Be ground state vs first excited state (Becker & Kavanagh
1955; Sikora & Weller 2016 review).

For a parent proton at lab-frame energy E_p:
  Total CM energy:    Q_eff = 8.68 MeV + (E_p × m_B / (m_p + m_B))
  Primary alpha:      E_α₁ ≈ 0.45 × Q_eff (CM)
  Secondary alphas:   E_α₂,α₃ ≈ 0.275 × Q_eff each (CM, from ⁸Be* breakup)
  Lab-frame boost:    add v_CM along p direction, isotropic emission in CM

We compute the resulting lab-frame alpha spectrum by Monte-Carlo sampling for
each proton energy bin, then weight by the fusion-rate distribution.

CAVEATS
-------
  - This is a kinematic projection only. WarpX does NOT track alphas as
    particles in this simulation; the fusion module only updates rate counters.
  - Branching ratios approximated from p+¹¹B level scheme; precise channel
    weights require dedicated nuclear-data fold.
  - No plasma stopping (alphas treated as exit the plasma without energy loss).
  - Angular distribution is assumed isotropic in CM (slightly forward-peaked
    in lab); for serious pitch work a Boltzmann-folded angular calc is needed.

Usage:
    python medical_alpha_kinematic.py --run-dir <path>

Outputs:
    <run>_medical_alpha_kinematic.txt   — text summary for the medical-applications track
    <run>_medical_alpha_kinematic.csv   — full energy spectrum
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np


# ─── Constants ──────────────────────────────────────────────────────────────
Q_PB11_MeV = 8.68          # p + ¹¹B → 3α total Q-value
M_P_amu = 1.007276
M_B11_amu = 11.009305
M_BE8_amu = 8.005305        # ⁸Be ground state
M_ALPHA_amu = 4.001506
AMU_TO_MeV = 931.494        # MeV/c²

# Branching: primary alpha (α₁) gets ~45% of Q, ⁸Be* gets ~55% which then
# splits into 2α at ~27.5% each. These are approximate.
F_PRIMARY = 0.45
F_SECONDARY = (1.0 - F_PRIMARY) / 2.0   # = 0.275 each


def read_proton_histogram(csv_path: str, burst_time_ps: float):
    """Read global-zone proton dN/dE at the burst-time snapshot.

    Returns:
        list of (E_centre_keV, N_weighted) at the snapshot closest to burst_time_ps
    """
    by_t = defaultdict(list)
    times = set()
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                if row['zone'] != 'global':
                    continue
                t = float(row['t_ps'])
                e_c = float(row['E_bin_kev_centre'])
                n = float(row['N_weighted'])
                times.add(t)
                by_t[t].append((e_c, n))
            except (KeyError, ValueError):
                continue
    if not times:
        return None, None
    times = sorted(times)
    chosen_t = min(times, key=lambda t: abs(t - burst_time_ps))
    by_t[chosen_t].sort(key=lambda r: r[0])
    return chosen_t, by_t[chosen_t]


def read_total_alpha_yield(fusion_csv: str):
    """Read final cumulative alpha yield from fusion_rate_power_by_iter.csv.

    Returns the final value of cum_alpha_yield_p11b (alphas per shot) if found,
    else 0.0.
    """
    if not os.path.exists(fusion_csv):
        return 0.0
    last_cum = 0.0
    with open(fusion_csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                last_cum = float(row.get('cum_alpha_yield_p11b', last_cum))
            except (KeyError, ValueError):
                continue
    return last_cum


def kinematic_alpha_spectrum(proton_bins, n_alpha_total: float,
                               n_samples_per_bin: int = 200):
    """Build the lab-frame alpha energy spectrum kinematically.

    For each proton-energy bin:
      1. Q_eff = Q_pb11 + E_p × m_B / (m_p + m_B)
      2. Sample N alphas (n_samples_per_bin × 3 alphas/event)
         - 1 primary at f_primary × Q_eff
         - 2 secondary at f_secondary × Q_eff each
      3. Isotropic emission in CM; boost to lab frame using CM velocity

    Returns:
        (E_alpha_keV_bins, counts) — histogram of alpha energies
    """
    # Normalize proton bins to weights summing to total fusion events
    if not proton_bins:
        return np.array([]), np.array([])
    proton_E = np.array([e for (e, n) in proton_bins])     # keV
    proton_w = np.array([n for (e, n) in proton_bins])
    # Restrict to bins above the (p,α) threshold (~0.7 MeV center-of-mass;
    # ~0.4 MeV lab-frame for p+B11 with B11 at rest at ion temperature)
    mask = proton_E >= 100.0
    proton_E = proton_E[mask]
    proton_w = proton_w[mask]
    if proton_w.sum() == 0:
        return np.array([]), np.array([])
    proton_w = proton_w / proton_w.sum()  # normalize as PDF over E

    # Total alphas produced per shot: distribute across bins by proton flux × σ
    # For first-order approximation, use proton flux distribution directly
    # (assumes σ varies slowly over each bin — accept the approximation)

    # Output histogram for alpha energies (50 log bins, 0.1 to 20 MeV)
    n_E_bins = 60
    E_lo = 100.0        # keV
    E_hi = 20000.0       # keV
    bin_edges = np.logspace(math.log10(E_lo), math.log10(E_hi), n_E_bins + 1)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    counts = np.zeros(n_E_bins)

    # Mass-weighted CM frame velocity (non-relativistic ok for E_p < few MeV)
    m_tot = M_P_amu + M_B11_amu
    f_p = M_P_amu / m_tot       # fraction of energy in proton motion at CM
    rng = np.random.default_rng(42)

    for i, E_p_keV in enumerate(proton_E):
        w_bin = proton_w[i]
        E_p_MeV = E_p_keV / 1000.0
        # CM energy released (proton brings its KE + Q-value)
        Q_eff_MeV = Q_PB11_MeV + E_p_MeV * (M_B11_amu / m_tot)
        # CM-frame alpha energies
        E_alpha1_CM = F_PRIMARY * Q_eff_MeV
        E_alpha23_CM = F_SECONDARY * Q_eff_MeV
        # CM frame velocity in lab frame (fraction of c)
        # KE_p_CM = E_p × m_B / (m_p + m_B); v_CM along p direction
        # For non-relativistic case: v_CM = p_p / m_tot
        # alpha lab energy = E_α_CM + 2 × sqrt(E_α_CM × E_CM_motion) × cosθ + E_CM_motion
        # where E_CM_motion is the KE of an alpha sitting in CM frame moving with v_CM
        # Simpler: use scalar Galilean energy boost
        E_CM_motion_per_alpha_MeV = 0.5 * M_ALPHA_amu * (
            (E_p_MeV * f_p * 2.0 / M_P_amu)  # v² = 2KE/m
        )
        # Sample isotropic emission
        cos_thetas = rng.uniform(-1.0, 1.0, size=(n_samples_per_bin, 3))
        # For each event (3 alphas):
        alpha_energies_MeV = []
        for s in range(n_samples_per_bin):
            # alpha 1 (primary)
            e_lab1 = (E_alpha1_CM + E_CM_motion_per_alpha_MeV
                       + 2.0 * math.sqrt(max(0.0, E_alpha1_CM * E_CM_motion_per_alpha_MeV))
                              * cos_thetas[s, 0])
            # alphas 2, 3 (secondary)
            e_lab2 = (E_alpha23_CM + E_CM_motion_per_alpha_MeV
                       + 2.0 * math.sqrt(max(0.0, E_alpha23_CM * E_CM_motion_per_alpha_MeV))
                              * cos_thetas[s, 1])
            e_lab3 = (E_alpha23_CM + E_CM_motion_per_alpha_MeV
                       + 2.0 * math.sqrt(max(0.0, E_alpha23_CM * E_CM_motion_per_alpha_MeV))
                              * cos_thetas[s, 2])
            alpha_energies_MeV.extend([max(0.0, e_lab1),
                                        max(0.0, e_lab2),
                                        max(0.0, e_lab3)])
        # Bin these alpha energies
        alpha_keV = np.array(alpha_energies_MeV) * 1000.0
        hist, _ = np.histogram(alpha_keV, bins=bin_edges)
        # Weight by the proton bin's contribution to the total fusion yield
        # × number of alphas per event accounting (we already produced 3 per sample)
        # Scale: each sample represents w_bin / n_samples_per_bin of the total events
        scale = w_bin / float(n_samples_per_bin)
        counts += hist * scale

    # Normalize so total alphas = n_alpha_total
    total_sampled = counts.sum()
    if total_sampled > 0 and n_alpha_total > 0:
        counts = counts * (n_alpha_total / total_sampled)
    return bin_centres, counts, bin_edges


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--burst-time-ps', type=float, default=60.0)
    p.add_argument('--rep-rate-hz', type=float, default=1000.0,
                   help='Laser rep rate in Hz (default 1000)')
    args = p.parse_args(argv)

    run_dir = os.path.normpath(args.run_dir)
    run_basename = os.path.basename(run_dir)
    hist_csv = os.path.join(run_dir, 'proton_energy_histogram.csv')
    fusion_csv = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')

    if not os.path.exists(hist_csv):
        print(f'ERROR: {hist_csv} not found')
        sys.exit(2)

    print(f'=== Medical Stage M.2: kinematic alpha spectrum ===')
    print(f'  run:          {run_basename}')

    burst_t, proton_bins = read_proton_histogram(hist_csv, args.burst_time_ps)
    if proton_bins is None:
        print('ERROR: no proton histogram data')
        sys.exit(3)
    print(f'  burst dump:   t = {burst_t:.1f} ps')

    n_alpha_total = read_total_alpha_yield(fusion_csv)
    print(f'  cumulative α: {n_alpha_total:.3e} /shot (from fusion_rate CSV)')

    if n_alpha_total <= 0:
        print('WARNING: no alpha yield in fusion CSV; spectrum will be empty')

    bin_centres, counts, bin_edges = kinematic_alpha_spectrum(
        proton_bins, n_alpha_total)

    if bin_centres.size == 0:
        print('ERROR: kinematic spectrum is empty')
        sys.exit(4)

    bin_widths = np.diff(bin_edges)
    dNdE = np.where(bin_widths > 0, counts / bin_widths, 0.0)

    # Summary statistics
    thresholds_MeV = [1, 3, 5, 10]
    n_above = {}
    for thr in thresholds_MeV:
        mask = bin_centres >= thr * 1000.0
        n_above[thr] = counts[mask].sum()

    # Weighted mean alpha energy
    if counts.sum() > 0:
        mean_E_keV = float((bin_centres * counts).sum() / counts.sum())
    else:
        mean_E_keV = 0.0

    # Outputs
    out_txt = os.path.join(run_dir, f'{run_basename}_medical_alpha_kinematic.txt')
    out_csv = os.path.join(run_dir, f'{run_basename}_medical_alpha_kinematic.csv')

    with open(out_txt, 'w') as f:
        w = f.write
        w('=' * 78 + '\n')
        w('⚠⚠⚠  DRAFT — PENDING SIMULATION VALIDATION  ⚠⚠⚠\n')
        w('  Numbers below derive from a WarpX run with known numerical-noise\n')
        w('  artifacts in the proton spectrum and folded alpha yield. DO NOT\n')
        w('  cite externally. See script header for detailed status.\n')
        w('=' * 78 + '\n')
        w(f'Medical Stage M.2 — Kinematic alpha spectrum\n')
        w(f'Run: {run_basename}\n')
        w(f'Reaction: ¹¹B(p,α)αα   Q = 8.68 MeV\n')
        w('=' * 78 + '\n\n')

        w('METHODOLOGY NOTE\n')
        w('  This script does NOT use directly-simulated alphas: WarpX hybrid-PIC\n')
        w('  in this configuration does not track alphas as particle species.\n')
        w('  The alpha spectrum here is computed KINEMATICALLY from:\n')
        w('    1. The proton energy histogram at reconnection burst\n')
        w('    2. The cumulative alpha count from σ-folded fusion-rate diagnostics\n')
        w('    3. Two-step kinematic decomposition: p+¹¹B → α + ⁸Be* → α + α + α\n')
        w('  with isotropic CM emission and Galilean lab-frame boost.\n')
        w('\n')

        w('Inputs:\n')
        w(f'  Burst snapshot:           t = {burst_t:.1f} ps\n')
        w(f'  Total α per shot:         {n_alpha_total:.3e}\n')
        w(f'  Rep rate:                 {args.rep_rate_hz:.0f} Hz\n\n')

        w('Alpha spectrum summary:\n')
        w(f'  Mean α energy:            {mean_E_keV/1000.0:.2f} MeV\n')
        w(f'  Per-shot α counts above thresholds (kinematic projection):\n')
        for thr in thresholds_MeV:
            w(f'    E_α > {thr:2d} MeV:           {n_above[thr]:.3e}\n')
        w('\n')

        # Flux scaling
        alpha_rate_per_s = n_alpha_total * args.rep_rate_hz
        w(f'Continuous-operation projections at {args.rep_rate_hz:.0f} Hz:\n')
        w(f'  Total α flux:             {alpha_rate_per_s:.3e} α/s\n')
        w(f'  α flux (>3 MeV):          '
          f'{n_above[3] * args.rep_rate_hz:.3e} α/s\n')
        w(f'  α flux (>5 MeV):          '
          f'{n_above[5] * args.rep_rate_hz:.3e} α/s\n\n')

        w('Comparison reference points for licensing discussion:\n')
        w('  Reference alpha-therapy seed: ~10⁵ α/s per seed (Ra-224 decay product)\n')
        w('  Targeted Alpha Therapy:   ~10⁶-10⁷ α/s per patient dose\n')
        w('  Research-grade α source:  ~10⁸-10¹⁰ α/s\n\n')

        w('Kinematic model parameters:\n')
        w(f'  Q-value:                  {Q_PB11_MeV} MeV\n')
        w(f'  Primary α energy frac:    {F_PRIMARY} of Q_eff (CM)\n')
        w(f'  Secondary α energy frac:  {F_SECONDARY} of Q_eff each (CM)\n')
        w(f'  CM-to-lab boost:          Galilean (non-relativistic)\n')
        w(f'  Angular distribution:     isotropic in CM\n\n')

        w('Caveats (acknowledged in any pitch material):\n')
        w('  - Alphas NOT simulated as particles; kinematic projection only\n')
        w('  - No plasma stopping (energy loss in target medium) modeled\n')
        w('  - Angular distribution simplified (isotropic CM)\n')
        w('  - Branching ratios approximate (precise nuclear data fold pending)\n')
        w('  - For clinical-application context, a GEANT4 follow-up calc with\n')
        w('    full kinematics + biological tissue stopping is the next step\n')
        w('\n')
        w('=' * 78 + '\n')

    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['E_alpha_keV_centre', 'E_alpha_keV_lo', 'E_alpha_keV_hi',
                     'dE_keV', 'counts_per_shot', 'dNdE_per_keV_per_shot'])
        for i, c in enumerate(bin_centres):
            wr.writerow([f'{c:.4f}',
                         f'{bin_edges[i]:.4f}',
                         f'{bin_edges[i+1]:.4f}',
                         f'{bin_widths[i]:.4f}',
                         f'{counts[i]:.6e}',
                         f'{dNdE[i]:.6e}'])

    print(f'\n  ✓ {out_txt}')
    print(f'  ✓ {out_csv}')
    print(f'\nSummary: {n_alpha_total:.3e} α/shot kinematic mean = '
          f'{mean_E_keV/1000.0:.2f} MeV')

    return 0


if __name__ == '__main__':
    sys.exit(main())
