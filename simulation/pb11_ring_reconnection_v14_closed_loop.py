#!/usr/bin/env python3
"""
p-11B / p-7Li / LiB Aneutronic Fusion — Ring Geometry with Rotating Modulation
WarpX Hybrid-PIC @ 5J optimal laser energy

v13.0 — ROTATION PHYSICS TESTBED — adds two new modes to test whether
        same-handedness Biermann-like topology (vortex_same) and same-sign
        rotating drive produce physics distinct from the legacy alternating
        Harris approximation. Default behavior is bit-for-bit identical to
        v12.4.4.

WHAT'S NEW IN v13.0:
  Two new CLI flags introduce alternative physics regimes alongside the
  legacy v12.4.4 behavior. With no new flags specified, v13 reproduces v12.4.4
  exactly — existing JOBS, YAML configs, and analysis pipelines work unchanged.

  --seed-topology vortex_same
      Replaces the alternating-polarity By bumps (legacy harris) with same-
      handedness in-plane vortices. Each spot is A_y(x,z) Gaussian, B = curl(A)
      produces in-plane Bx, Bz with By=0. ALL 8 spots have the same circulation
      sense, modeling the physical reality that all spots see the same
      laser-driven grad(n) x grad(T) sign for Biermann battery action.
      X-lines are NOT pre-seeded; whether they form depends on plasma response.

  --rotate-mode driving
      Same-sign rotating Jy drive at full Alfvenic amplitude (j_scale ~ 0.5-1.0).
      Produces an azimuthal current loop circulating around the ring, generating
      an axial guide-field component via Ampere's law. Distinct from legacy
      perturbative rotation which uses 1%% amplitude with alternating polarity.

PHYSICS NOTE — "evolved" mode and current support:
  WarpX hybrid-PIC's Ohm solver provides automatic plasma current support
  through the electron drift response. There is no Python-level switch to
  disable this. Whether a seeded topology is sustained depends on whether
  the plasma's natural drift response generates the supporting current
  pattern the topology requires:

  - harris (alternating By bumps): plasma naturally generates supporting
    Jx, Jz currents. Topology is stable. v12 confirms Q ~ 0.48 across runs.

  - vortex_same (same-handedness in-plane vortices): the supporting current
    pattern is azimuthal Jy. The plasma's spontaneous response may NOT
    generate this pattern. If so, the seeded vortex topology will spread and
    weaken without external drive. This is what the test investigates.

  - vortex_same + rotate-mode driving: the rotating same-sign Jy drive
    explicitly provides the azimuthal current pattern that the vortex
    topology needs. If the bare vortex_same case decays and the driven case
    sustains, the rotation hypothesis is supported.

  The functional difference between "static decay" and "driven sustain" is
  thus mediated by topology choice, NOT by a special "evolved" field-mode.
  The --field-mode flag retains its v12 meaning (applied/current/none) and
  doesn't gain new semantics in v13.

DESIGNED COMBINATIONS:
  Legacy reproduction (default):
    --field-mode applied --seed-topology harris --rotate-mode perturbative
    Reproduces v12.4.4 bit-for-bit.

  Topology decay test (new):
    --field-mode applied --seed-topology vortex_same --rotate-mode none
    Tests whether vortex_same decays when plasma response can't support it.

  Topology sustain test (new):
    --field-mode applied --seed-topology vortex_same --rotate-mode driving
    Tests whether same-sign rotating drive sustains the vortex topology.

  Bridge run (new):
    --field-mode applied --seed-topology harris --rotate-mode driving
    Same-sign drive on legacy harris seed (control for drive-mode effect).

INHERITED FROM v12.4.4:
v12.4.4 — B-seed default raised to 300T (Biermann battery at ~6e13 W/cm2)

WHAT'S NEW IN v12.4.4:
  - Default --b-seed raised from 150T to 300T
    Consistent with published laser-plasma measurements (Gao 2016, Santos 2018)
    at comparable intensity ~6e13 W/cm2 and 75 um spot size
    Prior 150T was conservative; 300T is the experimentally validated value
    that produced our best result (centre E_95th 470 keV, ratio 1.98x)
    Use --b-seed 150 to reproduce lower-field runs for sensitivity scan

v12.3 — per-region configurable fuel system with Li-7 / Li-6 / B-11 support

WHAT'S NEW IN v12.3:
  - Three independently configurable fuel regions, each with their own material:
      base  — ring plasma driving reconnection (default: p11b)
      rod   — central secondary-laser-ionized rod (default: ammonia_borane)
      ring  — outer catcher annulus (default: lib_equal)
  - Four particle species supported: proton, boron-11, lithium-7, lithium-6
    Species only created if at least one region uses them (saves memory)
  - 15 material presets covering p-11B, p-7Li, p-6Li, LiB hybrid, pure targets
  - Per-region ratio overrides for custom compositions
  - Multi-channel fusion diagnostics: separate rate/power columns per reaction
    (p+11B, p+7Li, p+6Li) with correct cross sections and energy yields
  - All v11.4 fixes preserved: n0, j_scale, field_mode

REACTION CHANNELS:
  p + 11B -> 3 alpha  + 8.68 MeV   (fully aneutronic)
  p + 7Li -> 2 alpha  + 17.35 MeV  (fully aneutronic, high yield)
  p + 6Li -> alpha+He3 + 4.02 MeV  (nearly aneutronic)

MATERIAL PRESETS (--base-fuel / --rod-fuel / --ring-fuel):
  p11b            H:B11=1:1         pure p-11B
  ammonia_borane  H:B11=6:1         NH3BH3, H-rich p-11B
  decaborane      H:B11=1.4:1       B10H14
  b18h22          H:B11=1.22:1      B18H22
  bn_plasma       H:B11=1:1         BN plasma
  p7li            H:Li7=1:1         pure p-7Li
  p6li            H:Li6=1:1         pure p-6Li
  nat_li          H:Li(natural)     natural abundance Li (92.6% Li7)
  lib_equal       H:Li7:B11=1:1:1   equal LiB hybrid
  lib_li_rich     H:Li7:B11=1:2:1   Li-heavy hybrid
  lib_b_rich      H:Li7:B11=1:0.5:1 B-heavy hybrid
  lib_nat_li      H:natLi:B11=1:1:1 natural Li + B11
  b11_target      B11 only          pure B11, protons from ring plasma
  li7_target      Li7 only          pure Li7, protons from ring plasma
  lib_target      Li7:B11=1:1       LiB no added H
  custom          use per-region ratio flags

RUN EXAMPLES:
  # Baseline p-11B (no inserts):
  mpirun -n 8 python pb11_v12.3.py --test

  # p-11B base + NH3BH3 rod + LiB catcher:
  mpirun -n 8 python pb11_v12.3.py --test --fuel-rod --fuel-ring-full

  # LiB hybrid everywhere:
  mpirun -n 8 python pb11_v12.3.py --test --base-fuel lib_equal \\
    --fuel-rod --rod-fuel lib_equal --fuel-ring-full --ring-fuel lib_equal

  # Pure p-7Li experiment:
  mpirun -n 8 python pb11_v12.3.py --test --base-fuel p7li \\
    --fuel-rod --rod-fuel p7li --fuel-ring-full --ring-fuel li7_target

  # Rotating + LiB rod:
  mpirun -n 8 python pb11_v12.3.py --test --rotate \\
    --fuel-rod --rod-fuel lib_equal
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
from mpi4py import MPI as mpi
from pywarpx import callbacks, picmi
import pywarpx
import warnings
# Suppress deprecated ParticleContainerWrapper API warning
# (still functional; will be updated when sim.particles.get() is stable)
warnings.filterwarnings('ignore',
                        message='.*ParticleContainerWrapper.*',
                        category=UserWarning)


# ============================================================================
# MATERIAL PRESETS
# ============================================================================
# Each preset defines ratios H : Li7 : Li6 : B11
# Ratios are relative — actual density set by per-region --*-density flag
# 'custom' means use per-region ratio override flags

FUEL_PRESETS = {
    # p-11B fuels
    'p11b':           {'h':1.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'p-11B H:B=1:1'},
    # 5:1 H:B11 — the "Holy Grail program" design composition. Hydrogenic-
    # dominant plasma keeps v_A high (driving fast reconnection) while still
    # providing sufficient B11 areal density for the p+B reaction. Gives
    # n_p = 5e24 and n_B11 = 1e24 at the standard base_density of 5e24.
    'p11b_holygrail': {'h':5.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'p-11B H:B=5:1 (Holy Grail program design)'},
    'ammonia_borane': {'h':6.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'NH3BH3 H:B=6:1'},
    'decaborane':     {'h':1.4,      'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'B10H14 H:B=1.4:1'},
    'b18h22':         {'h':22/18,    'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'B18H22 H:B=1.22:1'},
    'bn_plasma':      {'h':1.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'BN plasma H:B=1:1'},
    # v12.11: PERLA/HiLASE comparison target (Istokskaia 2023 Comm Phys).
    # CH-BN target = 600nm CH plasma polymer (H/C=1.88) on 3mm BN substrate.
    # Pre-plasma plume composition after 14ns prepulse expansion:
    #   H: 0.326, C: 0.174, N: 0.250, B-11: 0.200, B-10: 0.050 (atom fractions)
    # Lumping C+N+B-10 into 'bheavy' (avg mass 12.85 amu, charge +6):
    #   H : B-11 : bheavy = 1.63 : 1.00 : 2.37 (normalized so b11_ratio = 1)
    # Convention: --base-density sets n_B-11 (the species with ratio=1).
    # To match Istokskaia's pre-plasma B-11 density of 5e25 /m³, run with:
    #   --base-fuel ch_bn --base-density 5e25
    # This will create:  n_H = 8.15e25, n_B-11 = 5.00e25, n_bheavy = 1.185e26
    # → total ion density 2.5e26 /m³, mass-loaded as in Istokskaia PIC.
    'ch_bn':          {'h':1.63,     'li7':0.0,    'li6':0.0,    'b11':1.00,   'bheavy':2.37,
                       'label':'CH-BN (PERLA target, H+B11+spectator-heavy)'},
    # p-Li fuels
    'p7li':           {'h':1.0,      'li7':1.0,    'li6':0.0,    'b11':0.0,    'bheavy':0.0,
                       'label':'p-7Li H:Li7=1:1'},
    'p6li':           {'h':1.0,      'li7':0.0,    'li6':1.0,    'b11':0.0,    'bheavy':0.0,
                       'label':'p-6Li H:Li6=1:1'},
    'nat_li':         {'h':1.0,      'li7':0.926,  'li6':0.074,  'b11':0.0,    'bheavy':0.0,
                       'label':'natural Li (92.6% Li7)'},
    # LiB hybrid fuels
    'lib_equal':      {'h':1.0,      'li7':1.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'LiB equal H:Li7:B11=1:1:1'},
    'lib_li_rich':    {'h':1.0,      'li7':2.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'LiB Li-heavy H:Li7:B11=1:2:1'},
    'lib_b_rich':     {'h':1.0,      'li7':0.5,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'LiB B-heavy H:Li7:B11=1:0.5:1'},
    'lib_nat_li':     {'h':1.0,      'li7':0.926,  'li6':0.074,  'b11':1.0,    'bheavy':0.0,
                       'label':'natural Li + B11'},
    # Pure target fuels (no added H — receive protons from ring plasma)
    'b11_target':     {'h':0.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'pure B11 target'},
    'li7_target':     {'h':0.0,      'li7':1.0,    'li6':0.0,    'b11':0.0,    'bheavy':0.0,
                       'label':'pure Li7 target'},
    'lib_target':     {'h':0.0,      'li7':1.0,    'li6':0.0,    'b11':1.0,    'bheavy':0.0,
                       'label':'Li7+B11 target no H'},
    'custom':         {'h':-1.0,     'li7':-1.0,   'li6':-1.0,   'b11':-1.0,   'bheavy':-1.0,
                       'label':'custom — use ratio flags'},
}

PRESET_NAMES = list(FUEL_PRESETS.keys())

# Fusion reaction constants
REACTIONS = {
    'p11b': {
        'label':       'p + 11B -> 3 alpha',
        'energy_mev':  8.68,
        'threshold_kev': 500.0,
        'peak_kev':    675.0,
        'sigma_mb':    1.2,
        'n_alphas':    3,
        'target_species': 'boron11',
    },
    'p7li': {
        'label':       'p + 7Li -> 2 alpha',
        'energy_mev':  17.35,
        'threshold_kev': 300.0,
        'peak_kev':    441.0,
        'sigma_mb':    0.5,
        'n_alphas':    2,
        'target_species': 'li7',
    },
    'p6li': {
        'label':       'p + 6Li -> alpha + He3',
        'energy_mev':  4.02,
        'threshold_kev': 150.0,
        'peak_kev':    200.0,
        'sigma_mb':    0.15,
        'n_alphas':    1,
        'target_species': 'li6',
    },
}


# ============================================================================
# CROSS-SECTION CURVES — energy-resolved σ(E) for proper yield integration
# ============================================================================
# These tables replace the constant 'sigma_mb' values above when computing
# yield from the actual proton energy distribution. The constants above are
# retained for the legacy 'rate_per_ff' calculation (which assumed all
# fast-fraction protons sit at threshold). For paper-quality yield numbers
# we fold these σ(E) curves over the simulation's proton f(E).
#
# Data sources:
#   p+11B  : Sikora & Weller (2014) NIFS data; Becker et al. (1987) for low E
#   p+7Li  : NACRE evaluation
#   p+6Li  : NACRE evaluation
#
# Format: (E_kev, sigma_mb) pairs; we linearly interpolate in log-log space
# (which is more accurate than linear-linear for tunneling-dominated cross
# sections).

SIGMA_CURVE_KEV_MB = {
    'p11b': [
        # E_kev,    sigma_mb        # comment
        ( 200.0,    0.0001),        # well below threshold (Coulomb suppressed)
        ( 300.0,    0.005),         # rising from sub-threshold
        ( 400.0,    0.05),
        ( 500.0,    0.20),          # nominal threshold
        ( 600.0,    0.7),
        ( 675.0,    1.3),           # narrow resonance peak (Becker 1987)
        ( 750.0,    0.9),
        ( 1000.0,   0.4),
        ( 1500.0,   0.6),           # broad secondary peak
        ( 2000.0,   1.0),           # broad resonance ~2 MeV
        ( 3000.0,   0.6),
        ( 5000.0,   0.3),
        ( 10000.0,  0.1),
    ],
    'p7li': [
        ( 100.0,    0.001),
        ( 200.0,    0.05),
        ( 300.0,    0.2),
        ( 441.0,    0.5),           # peak
        ( 600.0,    0.4),
        ( 1000.0,   0.2),
        ( 2000.0,   0.08),
        ( 5000.0,   0.02),
    ],
    'p6li': [
        ( 50.0,     0.001),
        ( 100.0,    0.02),
        ( 150.0,    0.08),
        ( 200.0,    0.15),          # peak
        ( 300.0,    0.10),
        ( 500.0,    0.05),
        ( 1000.0,   0.02),
        ( 2000.0,   0.005),
    ],
}


def sigma_from_curve(rxn_key, energy_kev_array):
    """Return σ(E) in m² for an array of energies (keV), via log-log
    interpolation on the tabulated curve. Below the lowest tabulated point
    we extrapolate the Gamow tunneling shape (σ ∝ exp(-B/sqrt(E))/E); above
    the highest point we use a 1/E^2 fall-off (high-energy approximation).
    Energies outside the table are not the concern of this calculation."""
    e_arr  = np.asarray(energy_kev_array, dtype=float)
    table  = SIGMA_CURVE_KEV_MB.get(rxn_key)
    if not table:
        return np.zeros_like(e_arr)
    e_tab  = np.array([row[0] for row in table])
    s_tab  = np.array([row[1] for row in table])
    # Log-log interp (clip energies to table range so interp doesn't extrapolate
    # in linear; below-table is dominated by Coulomb suppression which we
    # approximate by using the lowest table value scaled by tunneling factor;
    # above-table we assume 1/E² for the high-energy tail).
    out = np.full_like(e_arr, 1e-12)  # near-zero default
    valid = (e_arr > 0) & np.isfinite(e_arr)

    # In-range: log-log interp
    in_range = valid & (e_arr >= e_tab[0]) & (e_arr <= e_tab[-1])
    if np.any(in_range):
        out[in_range] = np.exp(np.interp(np.log(e_arr[in_range]),
                                          np.log(e_tab), np.log(s_tab)))
    # Below table: tunneling extrapolation
    below = valid & (e_arr > 0) & (e_arr < e_tab[0])
    if np.any(below):
        out[below] = s_tab[0] * np.exp(-(np.sqrt(e_tab[0]/e_arr[below]) - 1) * 5.0)
    # Above table: 1/E² fall-off from last point
    above = valid & (e_arr > e_tab[-1])
    if np.any(above):
        out[above] = s_tab[-1] * (e_tab[-1] / e_arr[above])**2

    return out * 1e-31   # mb -> m²


def compute_energy_resolved_rate(rxn_key, ke_kev_protons, weights_protons,
                                  n_target_m3, plasma_volume_m3,
                                  physical_protons_per_macro):
    """Compute fusion rate by folding actual proton energy distribution over
    the σ(E) curve. Returns reactions/second (volume-integrated, instantaneous).

    Args:
      rxn_key: 'p11b' / 'p7li' / 'p6li'
      ke_kev_protons: array of macro-proton kinetic energies (keV)
      weights_protons: corresponding macro-particle weights (= number of
                       physical protons each macro-particle represents)
      n_target_m3:    target ion number density (m^-3)
      plasma_volume_m3: volume over which target density applies (m^3)
      physical_protons_per_macro: typically 1 (since weights already encode this)

    Math:
      For each macro-proton i with energy E_i, weight w_i:
        per-macro rate = w_i * sigma(E_i) * v(E_i) * n_target
        (this is rate per unit time per unit volume, integrated over the
         macro's representative volume which is implicit in the weight)
      Total rate = Σ_i w_i * σ(E_i) * v(E_i) * n_target

    The plasma_volume_m3 doesn't directly appear because the weights already
    encode the physical-particle count integrated over volume. We do however
    cap at the plasma volume to ensure we're not double-counting if particles
    leave the target region (out-of-region protons see n_target=0).
    """
    if ke_kev_protons is None or len(ke_kev_protons) == 0:
        return 0.0
    if n_target_m3 <= 0:
        return 0.0

    ke = np.asarray(ke_kev_protons, dtype=float)
    w  = np.asarray(weights_protons, dtype=float)
    if len(w) != len(ke):
        return float('nan')

    # σ(E) in m²
    sig = sigma_from_curve(rxn_key, ke)
    # v(E) in m/s, non-relativistic OK for ≤ 1 MeV protons
    v = np.sqrt(2.0 * ke * 1e3 * constants.q_e / PROTON_MASS_KG)

    # Per-macro rate contribution: w * σ * v * n_target
    rate_per_macro = w * sig * v * n_target_m3
    # Sum over all macros = total reactions per second
    return float(np.sum(rate_per_macro))


# Module-level constant: cap the σ(E) integration at energies above this
# threshold to avoid divergent low-energy contributions where the table
# extrapolation is unreliable.
SIGMA_INTEGRATION_MIN_KEV = 100.0


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(
    description='p-11B/p-7Li/LiB ring reconnection — v12.14 (Harris By topology restored)'
)

# ── Simulation control ───────────────────────────────────────────────────────
parser.add_argument('--test',         action='store_true',
                    help='Quick test run (128x128, 10 cyclotron periods)')
parser.add_argument('--freq',         type=float, default=500e6,
                    help='Rotation frequency Hz (default: 500 MHz)')
parser.add_argument('--rotate',       action='store_true',
                    help='Enable rotating Jy modulation (default: static)')
parser.add_argument('--nx',           type=int, default=0,
                    help='Override grid NX (default 0 = use test/production default)')
parser.add_argument('--nz',           type=int, default=0,
                    help='Override grid NZ (default 0 = use test/production default)')
parser.add_argument('--lx-min-um',    type=float, default=0.0,
                    help='Minimum domain size in microns (default 0 = use d_i scaling). Forces ring to fit.')
parser.add_argument('--ramp-steps',   type=int,   default=100)
parser.add_argument('--sigma-scale',  type=float, default=1.15)
parser.add_argument('--eta-scale',    type=float, default=1.0)
parser.add_argument('--te-ev',        type=float, default=2200.0)
parser.add_argument('--b-seed',       type=float, default=300.0,
                    help='Biermann battery seed field in T (default: 300T, v12.10). '
                         'At 300T, magnetic pressure dominates thermal pressure (β<1) '
                         'so simulation is in stiff regime — but Path A2 geometry '
                         '(σ=300µm, R=2400µm) puts σ>>d_i and reduces the spurious-E '
                         'init artefact 16× via larger spot radius. The combination '
                         'should keep peak proton acceleration at fusion-relevant '
                         'values (~vA·B ~ 10⁸ V/m) while the larger σ keeps the '
                         'simulation tractable. Use --b-seed 30 for low-β baseline.')
parser.add_argument('--field-mode',   type=str,   default='applied',
                    choices=['current','applied','none'],
                    help='Field initialization strategy (legacy v12.4.4 semantics):\n'
                         '  applied  - (default) AnalyticInitialField at t=0, topology\n'
                         '             controlled by --seed-topology. Plasma response\n'
                         '             automatically maintains the field if compatible\n'
                         '             with the topology.\n'
                         '  current  - ramped Jy seed, no B-init\n'
                         '  none     - no field at all, baseline mode')
parser.add_argument('--seed-topology', type=str, default='harris',
                    choices=['harris','vortex_same','vortex_alt'],
                    help='Initial B-field topology (v13):\n'
                         '  harris      - (default, legacy) alternating-polarity By bumps\n'
                         '                Adjacent spots have opposite By, X-lines pre-formed\n'
                         '                at chord midpoints. Mathematically 8 mini-Harris-sheets\n'
                         '                arrayed around a ring. NOT physical Biermann fields.\n'
                         '  vortex_same - (NEW) same-handedness in-plane vortices\n'
                         '                A_y(x,z) Gaussian about each spot, B = curl(A) gives\n'
                         '                Bx, Bz components, By=0. ALL 8 spots same circulation.\n'
                         '                Models same-direction laser-Biermann: closed-loop B at\n'
                         '                t=0, X-lines form DYNAMICALLY at chord midpoints as\n'
                         '                fields interact (~130 ps Alfvén transit).\n'
                         '  vortex_alt  - alternating-handedness in-plane vortices (v12.5 form)\n'
                         '                Same vortex math as vortex_same but s_k = (-1)^k.\n'
                         '                Provided for comparison; not the physical Biermann case.')
parser.add_argument('--rotate-mode', type=str, default='perturbative',
                    choices=['none','perturbative','driving'],
                    help='Rotating Jy drive (v13, only active when --rotate is set):\n'
                         '  perturbative - (default, legacy) j_scale * 1%% small perturbation\n'
                         '                 with alternating (-1)^k polarity per spot.\n'
                         '                 Reproduces v12.4.4 rotating behavior exactly.\n'
                         '  none         - rotation flag is honored but no Jy is produced.\n'
                         '                 Use to run vortex_same+evolved as a static control.\n'
                         '  driving      - (NEW) j_scale * full Alfvenic amplitude with same-sign\n'
                         '                 drive at all 8 spots, rotating around the ring.\n'
                         '                 Produces an azimuthal current loop (axial guide field\n'
                         '                 via Ampere). Tests sustained-topology hypothesis.')
parser.add_argument('--j-scale',      type=float, default=0.15)
# v14 STEP 1: closed-loop feasibility test flags. With these flags, the
# simulation runs unchanged but installs an extra observation-only callback
# that reads |B|_max from the field arrays every N steps and writes a CSV.
# No source modification is attempted at this stage. The test answers two
# questions: (1) can WarpX callbacks read field state at runtime?, (2) what
# is the |B|(t) trajectory we'd be triggering on for closed-loop control?
parser.add_argument('--monitor-bfield', action='store_true',
                    help='STEP 1 FEASIBILITY TEST: install an after-step callback '
                         'that reads |B|_max from the simulation domain every N steps '
                         '(see --monitor-interval). Writes runs/<outdir>/'
                         'closed_loop_monitor.csv with columns (step, t_ps, '
                         'B_max_T, B_min_T, B_mean_T, dB_dt_T_per_ns, wall_t_s). '
                         'Pure observation - simulation behaviour unchanged. '
                         'Used to verify callback architecture before building '
                         'closed-loop drive control.')
parser.add_argument('--monitor-interval', type=int, default=50,
                    help='Step interval between |B|_max samples when --monitor-bfield '
                         'is active. Default 50 steps (~7.5 ps at dt=150fs). '
                         'Lower = finer resolution, higher = less callback overhead.')
parser.add_argument('--no-bfield', action='store_true',
                    help='Run with B = 0 everywhere (artefact baseline mode). '
                         'Disables the analytic vortex initialization. Used to '
                         'distinguish reconnection-driven energization from '
                         'initialization-induced numerical artefacts. The hybrid '
                         'PIC solver still evolves B from currents, so any non-zero '
                         'B that emerges is purely from the simulated plasma '
                         'dynamics, not the seed field. Pair a vortex-on run with '
                         'a --no-bfield run at identical settings to subtract the '
                         'artefact and isolate the reconnection signal.')
parser.add_argument('--outdir',       type=str,   default='./pb11_diags')
parser.add_argument('--early-diag-period', type=int, default=5)
parser.add_argument('--early-diag-steps',  type=int, default=200)
parser.add_argument('--diag-profile', type=str, default='custom',
                    choices=['preflight', 'test', 'production', 'custom'],
                    help='Diagnostic strategy preset:\n'
                         '  preflight  - sparse early only (~10 dumps), abort fast (~5K steps)\n'
                         '  test       - dense early only (~100 dumps), no late, validate parameters\n'
                         '  production - balanced early + sparse late (~25-30 dumps), paper-quality\n'
                         '  custom     - (default) honour explicit --early-* / --late-* / --max-steps flags\n'
                         'When set to anything other than custom, this flag overrides the early/late\n'
                         'period flags. Use custom or omit the flag to keep manual control.')
parser.add_argument('--physical-protons',  type=float, default=5.74e19)
parser.add_argument('--max-steps',         type=int,   default=0,
                    help='Hard cap on total simulation steps (0=use test/prod default). '
                         'Used by preflight runner to limit run length exactly.')
parser.add_argument('--dump-period',       type=int,   default=2500,
                    help='Period (in simulation steps) between particle/field dumps. '
                         'Default 2500 (~375 ps at dt=150fs) is paper-quality cadence. '
                         'Use --dump-period 200 for a fine-grained exploration run '
                         '(captures reconnection event at ~30 ps resolution). '
                         'WARNING: at 512^2 each dump is ~17 GB. Disk usage scales as '
                         'n_steps / dump_period * 17 GB.')
parser.add_argument('--no-current-support', action='store_true',
                    help='Disable curl(B)/mu_0 supporting current in static mode '
                         '(v12.13: default ON). The supporting current makes the '
                         'seed B-field self-consistent in the Ohm solver, preventing '
                         'spurious E-field generation at high beta. Use this flag '
                         'only for diagnostic/baseline comparison runs.')

# ── Fuel region enable/disable ───────────────────────────────────────────────
parser.add_argument('--fuel-rod',        action='store_true',
                    help='Enable central rod plasma at convergence zone')
parser.add_argument('--fuel-ring-inner', action='store_true',
                    help='Enable outer ring inner face plasma')
parser.add_argument('--fuel-ring-outer', action='store_true',
                    help='Enable outer ring outer face plasma')
parser.add_argument('--fuel-ring-full',  action='store_true',
                    help='Enable full outer ring annulus')

# ── Per-region fuel selection ────────────────────────────────────────────────
parser.add_argument('--base-fuel',  type=str, default='p11b',
                    choices=PRESET_NAMES,
                    help='Ring plasma (reconnection base) fuel (default: p11b)')
parser.add_argument('--rod-fuel',   type=str, default='ammonia_borane',
                    choices=PRESET_NAMES,
                    help='Central rod fuel (default: ammonia_borane)')
parser.add_argument('--ring-fuel',  type=str, default='lib_equal',
                    choices=PRESET_NAMES,
                    help='Outer catcher ring fuel (default: lib_equal)')

# ── Per-region density ───────────────────────────────────────────────────────
parser.add_argument('--base-density', type=float, default=5e24,
                    help='Ring plasma peak density m^-3 (default: 5e24)')
parser.add_argument('--rod-density',  type=float, default=5e24,
                    help='Central rod density m^-3 (default: 5e24)')
parser.add_argument('--ring-density', type=float, default=5e24,
                    help='Outer ring density m^-3 (default: 5e24)')

# ── Per-region ratio overrides (negative = use preset) ──────────────────────
for region in ['base', 'rod', 'ring']:
    for species in ['h', 'li7', 'li6', 'b11', 'bheavy']:
        parser.add_argument(f'--{region}-{species}-ratio', type=float, default=-1.0,
                            help=f'{region} {species} ratio override (negative=use preset)')

# ── Geometry ─────────────────────────────────────────────────────────────────
parser.add_argument('--rod-radius-um',        type=float, default=75.0)
parser.add_argument('--outer-radius-um',      type=float, default=1050.0)
parser.add_argument('--outer-thickness-um',   type=float, default=100.0)
parser.add_argument('--face-edge-um',         type=float, default=15.0)
parser.add_argument('--n-spots',              type=int,   default=8,
                    help='Number of laser spots in ring (default: 8). '
                         'Use --n-spots 1 for single-spot TNSA-equivalent baseline '
                         '(forces ring radius to 0; disables X-line diagnostics).')
parser.add_argument('--ring-radius-um',       type=float, default=2400.0,
                    help='Ring radius in microns (default: 2400, i.e. 2.4 mm). '
                         'Reduce for smaller-domain well-resolved runs at high '
                         'density. For HD (n=5e25, d_i=32.2um) at 512^2 grid '
                         'with auto-scaled domain (120*d_i = 3.86 mm), use '
                         '--ring-radius-um 1000 to fit ring with margin and '
                         'achieve cells/d_i ~4.27. Scale --spot-radius-um '
                         'proportionally to preserve R/sigma ratio.')
parser.add_argument('--spot-radius-um',       type=float, default=300.0,
                    help='Spot radius (Gaussian sigma) in microns (default: 300). '
                         'Scale proportionally with --ring-radius-um for '
                         'self-similar geometry. Minimum is set by sigma/d_i '
                         '>= 2.95 (script convention for physics resolution).')

args = parser.parse_args()

IS_ROTATING = args.rotate
constants   = picmi.constants
comm        = mpi.COMM_WORLD
rank        = comm.Get_rank()

ORIG_STDOUT = sys.stdout
ORIG_STDERR = sys.stderr


# ============================================================================
# RESOLVE PER-REGION FUEL COMPOSITIONS
# ============================================================================
def resolve_region(region_name, preset_name, density):
    """
    Resolve final H/Li7/Li6/B11/bheavy ratios and densities for a named region.
    Per-region ratio overrides (if positive) replace preset values.
    Returns dict with species densities in m^-3.

    v12.11: Added 'bheavy' species — mass-weighted lump of C+N+B-10
    used for the ch_bn fuel mode (PERLA/HiLASE comparison).
    """
    p = FUEL_PRESETS[preset_name].copy()
    for sp in ['h', 'li7', 'li6', 'b11', 'bheavy']:
        override = getattr(args, f'{region_name}_{sp}_ratio')
        if override >= 0.0:
            p[sp] = override

    # Normalise so max ratio = 1 then scale by density
    ratios = {sp: max(0.0, p[sp]) for sp in ['h','li7','li6','b11','bheavy']}
    label  = FUEL_PRESETS[preset_name]['label']

    return {
        'preset':   preset_name,
        'label':    label,
        'density':  density,
        'ratios':   ratios,
        'n_h':      density * ratios['h'],
        'n_li7':    density * ratios['li7'],
        'n_li6':    density * ratios['li6'],
        'n_b11':    density * ratios['b11'],
        'n_bheavy': density * ratios['bheavy'],
    }

BASE = resolve_region('base',  args.base_fuel,  args.base_density)
ROD  = resolve_region('rod',   args.rod_fuel,   args.rod_density)
RING = resolve_region('ring',  args.ring_fuel,  args.ring_density)

# Determine which species need to be created
FUEL_ROD        = args.fuel_rod
FUEL_RING_INNER = args.fuel_ring_inner
FUEL_RING_OUTER = args.fuel_ring_outer
FUEL_RING_FULL  = args.fuel_ring_full
FUEL_ANY        = any([FUEL_ROD, FUEL_RING_INNER, FUEL_RING_OUTER, FUEL_RING_FULL])

def species_needed(sp_key):
    """True if any active region has non-zero ratio for this species."""
    if sp_key == 'h':
        return True  # always need protons
    if BASE['ratios'][sp_key] > 0:
        return True
    if FUEL_ROD   and ROD['ratios'][sp_key]  > 0: return True
    if FUEL_ANY   and RING['ratios'][sp_key] > 0: return True
    return False

NEED_B11    = species_needed('b11')
NEED_LI7    = species_needed('li7')
NEED_LI6    = species_needed('li6')
NEED_BHEAVY = species_needed('bheavy')   # v12.11: spectator-heavy for ch_bn

# Collect effective target densities for fusion accounting
# (averaged over active regions, weighted by density)
def effective_target_density(sp_key):
    """Effective number density of target species seen by fast protons."""
    total_n = BASE['n_' + sp_key]
    if FUEL_ROD:   total_n += ROD['n_'  + sp_key]
    if FUEL_ANY:   total_n += RING['n_' + sp_key]
    return total_n

EFF_N_B11    = effective_target_density('b11')
EFF_N_LI7    = effective_target_density('li7')
EFF_N_LI6 = effective_target_density('li6')


# ============================================================================
# FUSION ACCOUNTING — per reaction channel
# ============================================================================
AMU_KG         = 1.66054e-27
PROTON_MASS_KG = 1.00728 * AMU_KG
LI7_MASS_KG    = 7.01601 * AMU_KG
LI6_MASS_KG    = 6.01512 * AMU_KG
B11_MASS_KG    = 11.0093 * AMU_KG
# v12.11: mass-weighted-average heavy spectator (C+N+B-10) for CH-BN fuel mode.
# CH-BN composition (H polymer + BN substrate, mixed in pre-plasma plume):
#   m_C = 12.011, m_N = 14.007, m_B-10 = 10.013 amu, weighted ~12.85 amu.
# Rounded to 13.0 amu for clean numerical handling.
BHEAVY_MASS_KG = 13.0  * AMU_KG

def build_reaction_constants(rxn_key, eff_n_target, physical_protons):
    r = REACTIONS[rxn_key]
    q_e   = constants.q_e
    E_thr_j = r['threshold_kev'] * 1e3 * q_e
    v_thr   = np.sqrt(2.0 * E_thr_j / PROTON_MASS_KG)
    sigma_m2 = r['sigma_mb'] * 1e-31
    E_rxn_j  = r['energy_mev'] * 1e6 * q_e
    rate_per_ff = physical_protons * eff_n_target * sigma_m2 * v_thr
    return {
        'key':            rxn_key,
        'label':          r['label'],
        'threshold_kev':  r['threshold_kev'],
        'peak_kev':       r['peak_kev'],
        'n_alphas':       r['n_alphas'],
        'target_species': r['target_species'],
        'sigma_m2':       sigma_m2,
        'v_thr':          v_thr,
        'E_rxn_j':        E_rxn_j,
        'eff_n_target':   eff_n_target,
        'rate_per_ff':    rate_per_ff,
        'power_per_ff':   rate_per_ff * E_rxn_j,
        'active':         eff_n_target > 0.0,
    }

PHYSICAL_PROTONS = args.physical_protons

RXN_P11B = build_reaction_constants('p11b', EFF_N_B11, PHYSICAL_PROTONS)
RXN_P7LI = build_reaction_constants('p7li', EFF_N_LI7, PHYSICAL_PROTONS)
RXN_P6LI = build_reaction_constants('p6li', EFF_N_LI6, PHYSICAL_PROTONS)
ALL_RXNS  = [RXN_P11B, RXN_P7LI, RXN_P6LI]

fusion_diag_state = {
    'last_step':   None,
    'last_time_s': None,
    'cumulative_energy_j': 0.0,
    'warned_particle_access': False,
}


# ============================================================================
# OUTPUT DIRECTORY + LOGGING
# ============================================================================
if args.outdir == './pb11_diags':
    mode_tag   = 'rot' if IS_ROTATING else 'static'
    test_tag   = 'test' if args.test else 'prod'
    base_tag   = args.base_fuel
    rod_tag    = args.rod_fuel if FUEL_ROD else 'norod'
    ring_parts = ([f'inner_{args.ring_fuel}'] if FUEL_RING_INNER else []) + \
                 ([f'outer_{args.ring_fuel}'] if FUEL_RING_OUTER else []) + \
                 ([f'full_{args.ring_fuel}']  if FUEL_RING_FULL  else [])
    ring_tag   = '_'.join(ring_parts) if ring_parts else 'noring'
    stamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTDIR     = (f'./runs/pb11_{mode_tag}_base_{base_tag}'
                  f'_rod_{rod_tag}_ring_{ring_tag}_{test_tag}_{stamp}')
else:
    OUTDIR = args.outdir.rstrip('/')

if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
comm.Barrier()

LOG_FILE               = os.path.join(OUTDIR, 'run.log')
META_FILE              = os.path.join(OUTDIR, 'run_meta.txt')
STEP_INDEX_FILE        = os.path.join(OUTDIR, 'step_time_index.csv')
FUSION_DIAG_FILE       = os.path.join(OUTDIR, 'fusion_rate_power_by_iter.csv')
FUSION_ACCOUNTING_FILE = os.path.join(OUTDIR, 'fusion_accounting_notes.txt')

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, o):
        for f in self.files: f.write(o); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_fh = None
if rank == 0:
    log_fh = open(LOG_FILE, 'w', buffering=1)
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)


# ============================================================================
# PHYSICAL PARAMETERS
# ============================================================================
LASER_ENERGY_J    = 5.0
ROTATION_FREQ_HZ  = args.freq
N_SPOTS           = args.n_spots
if N_SPOTS < 1:
    raise ValueError(f'--n-spots must be >= 1 (got {N_SPOTS})')
SINGLE_SPOT_MODE  = (N_SPOTS == 1)
# v12.10 Path A2: bigger spots and bigger ring for hybrid-PIC tractability
# σ=300µm  (was 75µm — now σ/d_i = 2.95 instead of 0.74; physics resolved)
# R=2400µm (was 800µm — keeps chord_half/σ = 3.06 for clean X-lines)
# B=300T (kept) — fusion-relevant, but artefact 16× smaller due to larger σ
# v12.12: --n-spots 1 forces RING_RADIUS_M=0 (single spot at origin = TNSA-equivalent
# baseline with no ring topology and no reconnection X-lines).
# v12.15: ring/spot radii now CLI-configurable via --ring-radius-um / --spot-radius-um
#         (defaults preserve v12.10/12.14 geometry: 2400um ring, 300um spot).
RING_RADIUS_M     = 0.0 if SINGLE_SPOT_MODE else args.ring_radius_um * 1e-6
SPOT_RADIUS_M     = args.spot_radius_um * 1e-6

N_BACKGROUND_M3   = BASE['density'] * 0.30      # 30% of base peak as background
N_PEAK_M3         = BASE['density']
T_ION_EV          = 660.0
T_ELEC_EV         = args.te_ev
B_SEED_T          = args.b_seed
# Vortex-prefactor renormalisation so peak |B| at r=σ equals user-facing B_SEED_T.
# (Detailed derivation in the comment block above build_B_vortex() further down.)
B_SEED_NORM       = float(np.exp(0.5))   # 1.6487 -- so peak |B| equals B_SEED_T

ROD_RADIUS_M      = args.rod_radius_um      * 1e-6
OUTER_RADIUS_M    = args.outer_radius_um    * 1e-6
OUTER_THICKNESS_M = args.outer_thickness_um * 1e-6
FACE_EDGE_M       = max(args.face_edge_um, 1.0) * 1e-6

omega_ci = constants.q_e * B_SEED_T / PROTON_MASS_KG
omega_pi = np.sqrt(N_PEAK_M3 * constants.q_e**2 / (constants.ep0 * PROTON_MASS_KG))
d_i      = constants.c / omega_pi
# v_A computed from TOTAL ion mass density (protons + B11 + Li7 + Li6 if present),
# not just protons. The base region's composition is what mostly determines this
# in the bulk plasma; rod/ring regions are perturbations. Using just m_p here
# overestimates v_A by sqrt(1 + sum(n_i*m_i/n_p/m_p)) — for a 1:1 H:B11 plasma
# that's a factor of ~3.5× error.
_ion_mass_density = (
    BASE['n_h']      * PROTON_MASS_KG +
    BASE['n_b11']    * B11_MASS_KG +
    BASE['n_li7']    * LI7_MASS_KG +
    BASE['n_li6']    * LI6_MASS_KG +
    BASE['n_bheavy'] * BHEAVY_MASS_KG    # v12.11: ch_bn spectator-heavy
)
# Guard against zero (e.g., empty config); fall back to proton-only.
if _ion_mass_density <= 0:
    _ion_mass_density = N_PEAK_M3 * PROTON_MASS_KG
v_A      = B_SEED_T / np.sqrt(constants.mu0 * _ion_mass_density)
# Also expose the proton-only v_A for legacy comparisons / diagnostics
v_A_proton_only = B_SEED_T / np.sqrt(constants.mu0 * N_PEAK_M3 * PROTON_MASS_KG)
v_th_p      = np.sqrt(T_ION_EV * constants.q_e / PROTON_MASS_KG)
v_th_B11    = np.sqrt(T_ION_EV * constants.q_e / B11_MASS_KG)
v_th_Li7    = np.sqrt(T_ION_EV * constants.q_e / LI7_MASS_KG)
v_th_Li6    = np.sqrt(T_ION_EV * constants.q_e / LI6_MASS_KG)
v_th_bheavy = np.sqrt(T_ION_EV * constants.q_e / BHEAVY_MASS_KG)  # v12.11

eta_norm = 6e-3
eta0     = d_i * v_A / (constants.ep0 * constants.c**2)
eta_SI   = eta_norm * eta0 * args.eta_scale

J_amp_check   = N_PEAK_M3 * constants.q_e * v_A * args.j_scale
J_thermal     = N_PEAK_M3 * constants.q_e * v_th_p
j_scale_ratio = J_amp_check / J_thermal

# v12.13: Plasma beta = thermal pressure / magnetic pressure.
# beta >> 1 = thermal-dominated (high-beta), spot-edge density gradients
# in the Ohm solver create spurious E without supporting current.
# beta ~ 1 = comparable, well-conditioned for hybrid-PIC.
# beta << 1 = magnetic-dominated, particle pusher may struggle.
# At ch_bn density (n_e ~ 1e27/m^3), beta > 100 for B < 200T → support REQUIRED.
_n_e_estimate = (BASE['n_h']*1.0 + BASE['n_b11']*5.0 + BASE['n_li7']*3.0
                 + BASE['n_li6']*3.0 + BASE['n_bheavy']*6.0)
if _n_e_estimate <= 0:
    _n_e_estimate = N_PEAK_M3
_total_ion_n = (BASE['n_h'] + BASE['n_b11'] + BASE['n_li7']
                + BASE['n_li6'] + BASE['n_bheavy'])
if _total_ion_n <= 0:
    _total_ion_n = N_PEAK_M3
_P_th  = (_total_ion_n * T_ION_EV + _n_e_estimate * T_ELEC_EV) * constants.q_e
_P_mag = B_SEED_T**2 / (2.0 * constants.mu0)
plasma_beta = _P_th / _P_mag if _P_mag > 0 else float('inf')

# Domain and timestepping
# v12.10 Path A2: domain expanded to 120 d_i (~12 mm) to fit R=2400µm with margin.
# Grid resolution increased proportionally to keep ≥ 6 cells per σ.
LX_DI = LZ_DI = 120
if args.test:
    NX, NZ = 256, 256; NPPC = 200; LT = 10;  DT = 1e-3
else:
    NX, NZ = 384, 384; NPPC = 400; LT = 25;  DT = 1.94e-4

# CLI override for grid size
if args.nx > 0:
    NX = args.nx
if args.nz > 0:
    NZ = args.nz

# Domain: max of d_i-based and minimum specified
LX_M_DI = LX_DI * d_i
LZ_M_DI = LZ_DI * d_i
LX_M_MIN = args.lx_min_um * 1e-6 if args.lx_min_um > 0 else 0.0
LX_M = max(LX_M_DI, LX_M_MIN)
LZ_M = max(LZ_M_DI, LX_M_MIN)
time_step_s  = DT * 2.0 * np.pi / omega_ci   # cyclotron-period units (was DT/omega_ci, missing 2pi)
total_time_s = LT * 2.0 * np.pi / omega_ci   # cyclotron-period units (was LT/omega_ci, missing 2pi)
n_steps      = int(total_time_s / time_step_s)
# v12.4.3: hard cap for preflight — limits simulation to exactly N steps
if args.max_steps > 0:
    n_steps = args.max_steps  # v12.x: allow override beyond LT-based default
RAMP_STEPS   = max(1, args.ramp_steps)
RAMP_TIME_S  = RAMP_STEPS * time_step_s
SIGMA_INIT_M = SPOT_RADIUS_M * args.sigma_scale

# ── Diagnostic-profile presets ────────────────────────────────────────────────
# Each preset overrides early/late diagnostic periods AND (for preflight only)
# also caps n_steps to limit runtime. 'custom' leaves all values untouched.
#
# preflight  : sparse early only, no late, hard step cap   -> ~10 dumps, ~5 GB
# test       : dense early only, no late                   -> ~100 dumps, ~5-8 GB
# production : moderate early + sparse late                -> ~25-30 dumps, ~10-15 GB
# custom     : honour explicit --early-* / --late-* / --max-steps flags
DIAG_PROFILE = args.diag_profile
if DIAG_PROFILE == 'preflight':
    # 5,000-step survival check. Dense early to spot init artefacts, no late.
    if args.max_steps == 0:
        n_steps = min(n_steps, 5000)
    args.early_diag_period = 50    # ~10 dumps in first 500 steps
    args.early_diag_steps  = 500
    LATE_PERIOD_OVERRIDE   = 0     # disable late by setting to 0 (sentinel)
elif DIAG_PROFILE == 'test':
    # Validate parameters before production. Dense early diagnostics, no late
    # because the question we're answering is "do these settings work" not
    # "what does steady state look like".
    args.early_diag_period = 5
    args.early_diag_steps  = 500   # ~100 dumps in first 500 steps
    LATE_PERIOD_OVERRIDE   = 0
elif DIAG_PROFILE == 'production':
    # Paper-quality run. NO early dumps (EARLY_DIAG_STEPS doesnt cap WarpX dumps).
    # Late dumps every 5000 steps: 100K steps = 20 dumps at 512^2 = ~1 TB peak.
    args.early_diag_period = 0     # DISABLED - was causing disk-full crashes
    args.early_diag_steps  = 0
    LATE_PERIOD_OVERRIDE   = 5000  # late dumps every 5000 steps
else:  # 'custom'
    LATE_PERIOD_OVERRIDE   = None  # use the existing automatic computation

EARLY_DIAG_PERIOD = max(0, args.early_diag_period)
EARLY_DIAG_STEPS  = max(0, args.early_diag_steps)
if LATE_PERIOD_OVERRIDE is None:
    LATE_DIAG_PERIOD = max(25, n_steps // 80)
else:
    LATE_DIAG_PERIOD = LATE_PERIOD_OVERRIDE   # 0 disables late dumps


# ============================================================================
# PARAMETER REPORT
# ============================================================================
if rank == 0:
    print('=' * 70)
    print('  p-11B/p-7Li/LiB RING RECONNECTION  v14.0-step1 (closed-loop monitor)')
    print('=' * 70)
    geom_label = (f'SINGLE SPOT (TNSA-equivalent baseline, R=0)' if SINGLE_SPOT_MODE
                  else f'{N_SPOTS}-spot ring at R={RING_RADIUS_M*1e6:.0f} um')
    print(f'  Geometry:     {geom_label}')
    print(f'  Mode:         {"ROTATING" if IS_ROTATING else "STATIC BASELINE"}')
    print(f'  Diag profile: {DIAG_PROFILE}'
          + (' (preflight: 5K steps cap, no late dumps)' if DIAG_PROFILE=='preflight'
             else ' (test: dense early, no late)' if DIAG_PROFILE=='test'
             else ' (production: paper-quality)' if DIAG_PROFILE=='production'
             else ' (manual)'))
    print(f'  Field mode:   {args.field_mode}'
          + ('  (B-field DISABLED: --no-bfield)' if args.no_bfield else f'  (j_scale={args.j_scale:.3f})'))
    print(f'  B-seed:       {B_SEED_T:.1f} T  |  v_A={v_A:.2e} m/s  |  d_i={d_i*1e6:.1f} um')
    print(f'  eta_SI:       {eta_SI:.3e} Ohm m  |  J_ext/J_th={j_scale_ratio:.3f}')
    # v12.13 regime indicators: beta and current-support status
    if args.no_bfield:
        regime_note = '(no B-field)'
    elif plasma_beta > 100:
        regime_note = 'HIGH-BETA — current support REQUIRED'
    elif plasma_beta > 10:
        regime_note = 'thermal-dominated — current support recommended'
    elif plasma_beta > 1:
        regime_note = 'comparable — well-conditioned regime'
    else:
        regime_note = 'magnetic-dominated'
    print(f'  Plasma beta:  {plasma_beta:.2f}  ({regime_note})')
    # v12.14: Jy support is not used with the By topology (the support current
    # for By out-of-plane is in the (x,z) plane, not Jy). The plasma drifts
    # self-consistently to support the seed By within the first few timesteps.
    if IS_ROTATING:
        print(f'  External Jy:  rotational drive at {ROTATION_FREQ_HZ/1e6:.0f} MHz '
              f'(j_scale={args.j_scale:.3f})')
    else:
        print(f'  External Jy:  none (Harris By is supported by particle drift)')
    if j_scale_ratio < 0.10 and IS_ROTATING:
        print('  WARNING: j_scale low — rotation may be invisible vs static')
    print(f'  n0 (Ohm):     {N_PEAK_M3+N_BACKGROUND_M3:.3e} m^-3  [v11.4 FIX1]')
    print()
    print('  FUEL REGIONS:')
    print(f'  BASE  plasma: {args.base_fuel:18s} ({BASE["label"]})')
    print(f'    density={BASE["density"]:.2e}  H={BASE["n_h"]:.2e}  Li7={BASE["n_li7"]:.2e}  Li6={BASE["n_li6"]:.2e}  B11={BASE["n_b11"]:.2e}  bheavy={BASE["n_bheavy"]:.2e}')
    rod_status = "ENABLED" if FUEL_ROD else "disabled"
    print(f'  ROD   region: {rod_status}  fuel={args.rod_fuel:18s} ({ROD["label"]})')
    if FUEL_ROD:
        print(f'    density={ROD["density"]:.2e}  H={ROD["n_h"]:.2e}  Li7={ROD["n_li7"]:.2e}  Li6={ROD["n_li6"]:.2e}  B11={ROD["n_b11"]:.2e}  bheavy={ROD["n_bheavy"]:.2e}')
        print(f'    radius={ROD_RADIUS_M*1e6:.0f} um')
    ring_modes = []
    if FUEL_RING_INNER: ring_modes.append('inner')
    if FUEL_RING_OUTER: ring_modes.append('outer')
    if FUEL_RING_FULL:  ring_modes.append('full')
    ring_status = '+'.join(ring_modes) if ring_modes else 'disabled'
    print(f'  RING  region: {ring_status}  fuel={args.ring_fuel:18s} ({RING["label"]})')
    if ring_modes:
        print(f'    density={RING["density"]:.2e}  H={RING["n_h"]:.2e}  Li7={RING["n_li7"]:.2e}  Li6={RING["n_li6"]:.2e}  B11={RING["n_b11"]:.2e}  bheavy={RING["n_bheavy"]:.2e}')
        print(f'    radius={OUTER_RADIUS_M*1e6:.0f} um  thickness={OUTER_THICKNESS_M*1e6:.0f} um')
    print()
    print('  ACTIVE SPECIES:')
    print(f'    proton:  always  (reconnection driver)')
    print(f'    boron11: {"YES" if NEED_B11 else "no"}   (p-11B channel active={RXN_P11B["active"]})')
    print(f'    li7:     {"YES" if NEED_LI7 else "no"}   (p-7Li channel active={RXN_P7LI["active"]})')
    print(f'    li6:     {"YES" if NEED_LI6 else "no"}   (p-6Li channel active={RXN_P6LI["active"]})')
    print(f'    bheavy:  {"YES" if NEED_BHEAVY else "no"}   (spectator C+N+B-10, no fusion channel)')
    print()
    print('  FUSION REACTION CHANNELS:')
    for rxn in ALL_RXNS:
        if rxn['active']:
            print(f'    {rxn["label"]}')
            print(f'      n_target={rxn["eff_n_target"]:.2e} m^-3  '
                  f'sigma={rxn["sigma_m2"]*1e31:.2f} mb  '
                  f'E={rxn["E_rxn_j"]/constants.q_e/1e6:.2f} MeV')
            print(f'      rate_per_ff={rxn["rate_per_ff"]:.3e}/s  '
                  f'power_per_ff={rxn["power_per_ff"]:.3e} W')
    print('=' * 70)
    print(f'  Grid: {NX}x{NZ}  NPPC={NPPC}  dt={time_step_s*1e12:.4f}ps  '
          f'T={total_time_s*1e9:.3f}ns  steps={n_steps:,}')
    print(f'  Output: {OUTDIR}')
    if args.max_steps > 0:
        print(f'  [PREFLIGHT CAP: {n_steps} steps only]')
    print('=' * 70)


# ============================================================================
# SIMULATION SETUP
# ============================================================================
simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=1)

grid = picmi.Cartesian2DGrid(
    number_of_cells=[NX, NZ],
    lower_bound=[-LX_M/2, -LZ_M/2],
    upper_bound=[ LX_M/2,  LZ_M/2],
    lower_boundary_conditions=['periodic','periodic'],
    upper_boundary_conditions=['periodic','periodic'],
    lower_boundary_conditions_particles=['periodic','periodic'],
    upper_boundary_conditions_particles=['periodic','periodic'],
    # Domain decomposition: tuned for single-GPU runs.
    # Default WarpX decomposition would split into 32x32 boxes (64 boxes for 256x256),
    # which causes severe kernel-launch overhead on GPU. Use one large box per GPU.
    # On CPU-MPI runs, AMReX still subdivides as needed via blocking_factor.
    warpx_max_grid_size=max(NX, NZ),
    warpx_blocking_factor=32,
)

def smooth_ramp(t='t'):
    return f'(1.0 - exp(-{t} / {RAMP_TIME_S:.8e}))'


# ── Jy external current ──────────────────────────────────────────────────────
def build_Jy(rotating=False):
    """Construct the externally-applied Jy(x,z,t) for the hybrid-PIC Ohm solver.

    Two regimes (v12.14 — Harris-style By topology):
      - Static (not rotating): NO external Jy. The seed By field is supported
        self-consistently by particle drift — at low/moderate density the
        plasma adjusts its drift velocity to carry curl(B)/mu_0 within the
        first few timesteps. External Jy is unnecessary for this topology
        because the supporting current for By is in the (x,z) plane (Jx, Jz),
        not Jy, and pywarpx only allows Jy_external.
      - Rotating: small rotational perturbation at args.freq. This is the
        time-varying equivalent of "shake the spots a little" — it does not
        carry the supporting current; rotation just modulates the drive
        on top of whatever the plasma is already doing. j_scale controls
        the perturbation amplitude as fraction of thermal current.

    HISTORY:
      v12.13 added a curl(B_vortex)/mu_0 Jy supporting current. That was
      designed for the v12.5 in-plane vortex topology (which we have now
      reverted). For the original By topology, that Jy formula is incorrect
      and would inject current in the wrong direction. v12.14 removes it.
    """
    if args.field_mode == 'none' or args.field_mode == 'applied' and not rotating:
        return '0.0'

    # v13: respect rotate-mode when rotating is active
    rotate_mode = getattr(args, 'rotate_mode', 'perturbative')

    # Special case: rotate-mode = none — return zero even when --rotate set
    # (lets us run vortex_same+evolved as a static decay control while keeping
    #  --rotate flag plumbing untouched for diagnostics)
    if rotating and rotate_mode == 'none':
        return '0.0'

    # Same-sign ('driving') vs alternating ('perturbative') drive
    # Driving: all 8 spots have same Jy sign; pattern rotates as a unit.
    #          Produces an azimuthal current loop -> axial Bz guide field.
    # Perturbative: legacy (-1)^k alternation; matches harris-seed polarity.
    use_alternating_polarity = (rotate_mode == 'perturbative')

    # Rotating drive: amplitude scales with j_scale; in driving mode it is the
    # primary current source (j_scale ~ 0.5-1.0 expected); in perturbative mode
    # the legacy intent is a small fraction (j_scale ~ 0.01-0.15).
    J  = N_PEAK_M3 * constants.q_e * v_A * args.j_scale
    f  = ROTATION_FREQ_HZ
    R  = RING_RADIUS_M; s = SPOT_RADIUS_M
    terms = []
    for k in range(N_SPOTS):
        ba   = 2*np.pi*k/N_SPOTS
        sgn  = (-1)**k if use_alternating_polarity else 1
        ae   = f'({ba:.8f} + 2*pi*{f:.8e}*t)' if rotating else f'({ba:.8f})'
        cx   = f'({R:.8e}*cos({ae}))'; cz = f'({R:.8e}*sin({ae}))'
        terms.append(
            f'({sgn:.1f}*{J:.8e}*exp(-(((x-{cx})**2+(z-{cz})**2)/(2*{s:.8e}**2))))'
        )
    return smooth_ramp() + ' * (' + ' + '.join(terms) + ')'

Jy_expr = build_Jy(rotating=IS_ROTATING)

# ── Hybrid-PIC solver (FIX1: n0 = peak + background) ────────────────────────
solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T_ELEC_EV * constants.q_e / constants.kb,
    n0=N_PEAK_M3 + N_BACKGROUND_M3,
    n_floor=0.1 * N_BACKGROUND_M3,
    plasma_resistivity=eta_SI,
    substeps=80,
    Jy_external_function=Jy_expr,
)
simulation.solver = solver


# ── Fuel shape expressions ───────────────────────────────────────────────────
def fuel_shape_expr(rod, ring_inner, ring_outer, ring_full):
    if not any([rod, ring_inner, ring_outer, ring_full]): return '0.0'
    r  = 'sqrt(x**2 + z**2)'
    r0 = OUTER_RADIUS_M; st = OUTER_THICKNESS_M; sr = ROD_RADIUS_M; e = FACE_EDGE_M
    terms = []
    if rod:
        terms.append(f'exp(-((x**2+z**2)/(2.0*{sr:.8e}**2)))')
    rg = f'exp(-(({r}-{r0:.8e})**2)/(2.0*{st:.8e}**2))'
    iw = f'(0.5*(1.0-tanh(({r}-{r0:.8e})/{e:.8e})))'
    ow = f'(0.5*(1.0+tanh(({r}-{r0:.8e})/{e:.8e})))'
    if ring_full:
        terms.append(rg)
    else:
        if ring_inner: terms.append(f'({rg}*{iw})')
        if ring_outer: terms.append(f'({rg}*{ow})')
    return f'min(1.0, ({" + ".join(terms)}))'

_fuel_shape = fuel_shape_expr(FUEL_ROD, FUEL_RING_INNER, FUEL_RING_OUTER, FUEL_RING_FULL)


# ── Density expression builder ───────────────────────────────────────────────
def build_density(base_peak, base_bg, sigma, rod_density=0.0, ring_density=0.0):
    """
    Ring spot Gaussians (reconnection base) + rod fuel + ring fuel.
    Rod and ring use the shared _fuel_shape but with independent densities.
    """
    spots = ' + '.join(
        f'exp(-(((x-{RING_RADIUS_M*np.cos(2*np.pi*k/N_SPOTS):.8e})**2'
        f'+(z-{RING_RADIUS_M*np.sin(2*np.pi*k/N_SPOTS):.8e})**2)'
        f'/(2.0*{sigma:.8e}**2)))'
        for k in range(N_SPOTS)
    )
    expr = f'{base_bg:.8e} + {base_peak:.8e} * ({spots})'

    # Rod and ring can have different densities for the same species
    # We build separate shape terms to allow independent density scaling
    def rod_shape():
        sr = ROD_RADIUS_M
        return f'exp(-((x**2+z**2)/(2.0*{sr:.8e}**2)))'

    def ring_shape():
        r  = 'sqrt(x**2 + z**2)'
        r0 = OUTER_RADIUS_M; st = OUTER_THICKNESS_M; e = FACE_EDGE_M
        rg = f'exp(-(({r}-{r0:.8e})**2)/(2.0*{st:.8e}**2))'
        iw = f'(0.5*(1.0-tanh(({r}-{r0:.8e})/{e:.8e})))'
        ow = f'(0.5*(1.0+tanh(({r}-{r0:.8e})/{e:.8e})))'
        parts = []
        if FUEL_RING_FULL:  parts.append(rg)
        if FUEL_RING_INNER: parts.append(f'({rg}*{iw})')
        if FUEL_RING_OUTER: parts.append(f'({rg}*{ow})')
        return f'min(1.0, ({" + ".join(parts)}))' if parts else '0.0'

    if FUEL_ROD and rod_density > 0.0:
        expr += f' + ({rod_density:.8e}) * ({rod_shape()})'
    if any([FUEL_RING_INNER, FUEL_RING_OUTER, FUEL_RING_FULL]) and ring_density > 0.0:
        expr += f' + ({ring_density:.8e}) * ({ring_shape()})'

    return expr


# Build density for each species
proton_expr = build_density(
    base_peak=BASE['n_h'],    base_bg=N_BACKGROUND_M3,
    sigma=SIGMA_INIT_M,
    rod_density=ROD['n_h'],   ring_density=RING['n_h'],
)

b11_expr = build_density(
    base_peak=BASE['n_b11'],  base_bg=BASE['n_b11']*0.3 if BASE['n_b11']>0 else 0,
    sigma=SIGMA_INIT_M,
    rod_density=ROD['n_b11'], ring_density=RING['n_b11'],
) if NEED_B11 else None


# ============================================================================
# OPTION 1 — CURRENT-SUPPORTING ION DRIFT (v12.7)
# ============================================================================
# Hybrid PIC interprets ∇×B/μ₀ as plasma current. If we seed a B-field with
# nonzero curl (which our vortex pattern has), the solver immediately tries to
# produce that current via electron flow, generating large compensating
# E-fields that overwhelm any reconnection-driven dynamics.
#
# Fix: at t=0, give the ion species a directed velocity that produces exactly
# the J needed to support B. Then ∇×B is supported by real ion current rather
# than electron rush, and no spurious E develops.
#
# Math (for our 2D-XZ, A_y-only vector potential):
#
#   B = ∇ × (A_y ŷ)
#   A_y = Σ_k sg_k · A0 · σ · exp(-r_k²/(2σ²))
#   ∇²A_y = (A0/σ) · sg_k · (r_k²/σ² - 2) · exp(-r_k²/(2σ²))
#   μ₀ J_y = -∇²A_y = (A0/σ) · sg_k · (2 - r_k²/σ²) · exp(-r_k²/(2σ²))
#
# At spot centre (r_k=0): peak |J_y| = 2·A0/(μ₀·σ)
# At r_k = σ·√2: J_y = 0 (current ring-shaped around each spot)
# At r_k > σ·√2: J_y reverses sign (counter-current beyond the spot)
#
# Translating to ion drift velocity:
#   J_y = e · Σ_i Z_i · n_i · v_drift,i
# We assign the same drift to all ion species so the current is shared in
# proportion to charge density. v_drift = J_y / (e · n_total_charge_per_volume)
#
# IMPORTANT: drift velocities are typically << thermal velocities for our
# parameters (~1-3 km/s vs ~360 km/s thermal at 660 eV). So this is a small
# perturbation on top of the Maxwellian, not a beam.

def build_J_curl_expression():
    """Return the analytic expression for J_y(x,z) = (curl B / mu0)_y at t=0.

    This matches the curl of build_B_vortex() exactly. Used to derive the
    ion drift velocity that supports B without a transient electron rush.

    Sign convention: assumes B = ∇×(A_y ŷ) with A_y = sg·A0·σ·exp(-r²/(2σ²)).
    Then -∇²A_y/μ₀ gives J_y in A/m².
    """
    A0       = B_SEED_T * B_SEED_NORM
    mu0_sym  = constants.mu0
    sigma    = SPOT_RADIUS_M
    # prefactor for J_y at spot centre: 2·sg·A0/(μ₀·σ)
    # general:                          (sg·A0/(μ₀·σ)) · (2 - r²/σ²) · exp(-r²/(2σ²))
    j_terms = []
    for k in range(N_SPOTS):
        a  = 2*np.pi*k/N_SPOTS
        cx = RING_RADIUS_M*np.cos(a); cz = RING_RADIUS_M*np.sin(a)
        # v12.8 FIX: all spots same circulation direction (was alternating).
        # Same-direction circulation produces anti-parallel field overlap at
        # adjacent-spot chord midpoints → 8 X-lines (the reconnection sites).
        # Alternating signs (the previous bug) made fields ADD at midpoints,
        # which suppressed reconnection topology entirely.
        sg = +1
        prefactor = sg * A0 / (mu0_sym * sigma)   # SI A/m²
        env = (f'exp(-0.5*((x-{cx:.8e})**2+(z-{cz:.8e})**2)/{sigma:.8e}**2)')
        radial = f'(2.0 - ((x-{cx:.8e})**2+(z-{cz:.8e})**2)/{sigma:.8e}**2)'
        j_terms.append(f'({prefactor:.8e}*{radial}*{env})')
    return ' + '.join(j_terms)


# ── Current-support history (v12.7 → v12.13 → v12.14) ───────────────────────
# v12.7 introduced curl(B)/mu_0 ion drift to support a seed B field. v12.9
# removed it. v12.13 restored it as external Jy current. None of these were
# strictly necessary for the Harris-style By topology used in the original
# 696-ps successful run; v12.14 has reverted to that simpler arrangement and
# removed Jy support entirely for the static case. With By out-of-plane,
# the supporting current is in (Jx, Jz) and develops automatically from
# particle drift. build_J_curl_expression() below is preserved for diagnostic
# inspection but is no longer called by build_Jy().

li7_expr = build_density(
    base_peak=BASE['n_li7'],  base_bg=BASE['n_li7']*0.3 if BASE['n_li7']>0 else 0,
    sigma=SIGMA_INIT_M,
    rod_density=ROD['n_li7'], ring_density=RING['n_li7'],
) if NEED_LI7 else None

li6_expr = build_density(
    base_peak=BASE['n_li6'],  base_bg=BASE['n_li6']*0.3 if BASE['n_li6']>0 else 0,
    sigma=SIGMA_INIT_M,
    rod_density=ROD['n_li6'], ring_density=RING['n_li6'],
) if NEED_LI6 else None

# v12.11: bheavy is the lumped (C+N+B-10) spectator species for ch_bn fuel mode.
# Mass-weighted average ~13 amu, charge state +6 (between C+6 and N+7), used to
# correctly model: (1) plasma mass loading for Alfvén-speed scaling, (2) Coulomb
# stopping of fast protons, (3) charge neutrality. Does NOT participate in any
# fusion reaction channel (p-bheavy is not in REACTIONS).
bheavy_expr = build_density(
    base_peak=BASE['n_bheavy'],  base_bg=BASE['n_bheavy']*0.3 if BASE['n_bheavy']>0 else 0,
    sigma=SIGMA_INIT_M,
    rod_density=ROD['n_bheavy'], ring_density=RING['n_bheavy'],
) if NEED_BHEAVY else None


# ── Species ──────────────────────────────────────────────────────────────────
proton_species = picmi.Species(
    particle_type='H', name='proton',
    charge='q_e', mass=PROTON_MASS_KG,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=proton_expr,
        rms_velocity=[v_th_p, v_th_p, v_th_p],
        directed_velocity=[0.0, 0.0, 0.0],
    ), warpx_do_not_push=False,
)
simulation.add_species(proton_species,
    layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC))

all_species = [proton_species]

if NEED_B11:
    b11_species = picmi.Species(
        name='boron11', charge=5*constants.q_e, mass=B11_MASS_KG,
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=b11_expr,
            rms_velocity=[v_th_B11, v_th_B11, v_th_B11],
            directed_velocity=[0.0, 0.0, 0.0],
        ), warpx_do_not_push=False,
    )
    simulation.add_species(b11_species,
        layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC))
    all_species.append(b11_species)

if NEED_LI7:
    li7_species = picmi.Species(
        name='li7', charge=3*constants.q_e, mass=LI7_MASS_KG,
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=li7_expr,
            rms_velocity=[v_th_Li7, v_th_Li7, v_th_Li7],
            directed_velocity=[0.0, 0.0, 0.0],
        ), warpx_do_not_push=False,
    )
    simulation.add_species(li7_species,
        layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC))
    all_species.append(li7_species)

if NEED_LI6:
    li6_species = picmi.Species(
        name='li6', charge=3*constants.q_e, mass=LI6_MASS_KG,
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=li6_expr,
            rms_velocity=[v_th_Li6, v_th_Li6, v_th_Li6],
            directed_velocity=[0.0, 0.0, 0.0],
        ), warpx_do_not_push=False,
    )
    simulation.add_species(li6_species,
        layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC))
    all_species.append(li6_species)

# v12.11: spectator-heavy species (C+N+B-10 lump) for ch_bn fuel mode (PERLA
# comparison). Charge +6, mass 13 amu. Pushed by E,B but does NOT undergo any
# fusion reaction in this simulation (no p-bheavy channel registered).
if NEED_BHEAVY:
    bheavy_species = picmi.Species(
        name='bheavy', charge=6*constants.q_e, mass=BHEAVY_MASS_KG,
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=bheavy_expr,
            rms_velocity=[v_th_bheavy, v_th_bheavy, v_th_bheavy],
            directed_velocity=[0.0, 0.0, 0.0],
        ), warpx_do_not_push=False,
    )
    simulation.add_species(bheavy_species,
        layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=NPPC))
    all_species.append(bheavy_species)


# ── B-field initialization ────────────────────────────────────────────────────
#
# v12.5 PHYSICS FIX: replace scalar B_y bumps with divergence-free 2D vortex
# fields per spot.
#
# OLD field (v12.4 and earlier):
#     B_y(r_k) = ±B0 · (r_k/σ) · exp(-r_k²/2σ²),  B_x = B_z = 0
#   Problems:
#     - B_y is a 1D scalar pattern, not a Biermann-battery topology.
#     - Field is zero AT the spot centre and peaks at r_k = σ — a hole, not a peak.
#     - No anti-parallel field structure between adjacent spots → no genuine
#       reconnection X-lines. The "alternating polarity" was a sign flip on a
#       scalar, not on a circulating vector field.
#     - Steep |dBy/dr| at the spot core produced J ~ 3·10¹² A/m² inside the
#       spots at t=0, driving a Hall E ~ 10⁹ V/m that energised the protons
#       sitting in the spot rings (the "outer" zone) rather than the centre.
#
# NEW field (v12.5):
#   Each spot is a 2D vortex generated from a vector potential A_y(x,z) that's
#   a Gaussian about the spot centre. With A = (0, A_y, 0) and B = ∇×A:
#       B_x =  s_k · A0 · (z-z_k)/σ · exp(-r_k²/2σ²)
#       B_z = -s_k · A0 · (x-x_k)/σ · exp(-r_k²/2σ²)
#       B_y = 0
#   where s_k = (-1)^k toggles vortex circulation between adjacent spots.
#
#   |B|(r_k) = A0 · (r_k/σ) · exp(-r_k²/2σ²),  peaks at r_k = σ with
#   |B|_peak = A0 · e^(-1/2) = 0.6065 · A0.
#
#   To get the user-facing seed parameter B_SEED_T = actual peak |B|,
#   the prefactor is renormalised: A0 = B_SEED_T / 0.6065 = B_SEED_T · e^(1/2).
#
# Why this works for ring reconnection:
#   - Adjacent vortices have opposite circulation, so at the chord midpoint
#     between them the contributions superpose to a true B ~ 0 null line.
#   - On either side of that null, B reverses sign — genuine anti-parallel
#     field, hence a genuine X-line reconnection topology.
#   - ∇·B = 0 by construction (B = ∇×A).
#   - Peak |J| = |∇×B|/μ₀ now sits in the spot-edge regions where reconnection
#     inflows are physically expected, NOT in the spot cores.
#
# B_SEED_NORM is defined at the top of the file alongside B_SEED_T.

def build_By_expression():
    """Return the alternating-polarity dipolar By(x,z) formula.

    Each spot contributes a term:
        sign_k * B_SEED_T * (r_k / sigma) * exp(-0.5 * r_k^2 / sigma^2)
    where r_k is distance from spot k centre, and sign_k = (-1)^k alternates
    around the ring. Adjacent spots therefore have opposite By polarity, so
    By passes through zero at the chord midpoints — these are the X-lines.

    This is mathematically 8 mini-Harris-sheets arrayed around a ring.
    The reconnection inflow at each X-line drives ions toward the ring core.

    HISTORY:
      v12.5 introduced an in-plane vortex topology (Bx, Bz with By=0). The
      vortex was divergence-free but produced closed-loop B around each spot
      with NO X-lines, which is the wrong topology for ring reconnection.
      The original By-only formula in this function is the geometry that
      produced the successful 696-ps run with measurable centre acceleration.
    """
    terms = []
    for k in range(N_SPOTS):
        angle = 2.0 * np.pi * k / N_SPOTS
        cx    = RING_RADIUS_M * np.cos(angle)
        cz    = RING_RADIUS_M * np.sin(angle)
        sign  = (-1)**k
        sigma = SPOT_RADIUS_M
        terms.append(
            f"({sign:.1f} * {B_SEED_T:.4f}"
            f" * sqrt((x - {cx:.8e})**2 + (z - {cz:.8e})**2)"
            f" / {sigma:.6e}"
            f" * exp(-0.5 * ((x - {cx:.8e})**2 + (z - {cz:.8e})**2)"
            f" / {sigma:.6e}**2))"
        )
    return " + ".join(terms)


# v13: vortex topology builders for in-plane Bx, Bz from A_y vector potential.
# Each spot k contributes A_y(x,z) = s_k * A0 * exp(-r_k^2/2 sigma^2)
# Then B = curl(A_y * y_hat) = (dA_y/dz) x_hat - (dA_y/dx) z_hat:
#   Bx =  s_k * A0 * (-(z-z_k)/sigma^2) * exp(-r_k^2/2 sigma^2)
#   Bz = -s_k * A0 * (-(x-x_k)/sigma^2) * exp(-r_k^2/2 sigma^2)
#       = s_k * A0 * (x-x_k)/sigma^2 * exp(...)
# Wait — let me redo this with sign conventions consistent with v12.5 doc:
#   A_y is a Gaussian with width sigma and amplitude A0 around (x_k, z_k).
#   B_x = +dA_y/dz, B_z = -dA_y/dx (right-hand rule for curl with A in +y)
#   dA_y/dz at spot k = -s_k * A0 * (z-z_k)/sigma^2 * exp(-r_k^2/2 sigma^2)
#   dA_y/dx at spot k = -s_k * A0 * (x-x_k)/sigma^2 * exp(-r_k^2/2 sigma^2)
# So:
#   B_x = -s_k * A0 * (z-z_k)/sigma^2 * exp(...)
#   B_z = +s_k * A0 * (x-x_k)/sigma^2 * exp(...)
#
# |B|^2 = (A0/sigma^2)^2 * (r_k^2) * exp(-r_k^2/sigma^2) * 1
# |B| peaks at r_k = sigma, value |B|_peak = (A0/sigma) * exp(-1/2).
# So to get user-facing peak |B| = B_SEED_T:
#   A0 = B_SEED_T * sigma * exp(1/2)
#
# vortex_same: s_k = +1 for all spots — physical Biermann (same drive direction)
# vortex_alt:  s_k = (-1)^k (legacy v12.5 form, kept for comparison)


def build_vortex_field_expressions(same_handedness=True):
    """Return (Bx_expr, By_expr, Bz_expr) for in-plane vortex topology.

    Each spot is a 2D vortex generated from a Gaussian vector potential A_y.
    With A = (0, A_y, 0), B = curl(A) gives in-plane Bx, Bz; By = 0.

    same_handedness=True   : all 8 spots same circulation (physical Biermann)
                              X-lines form dynamically at chord midpoints
                              as fields interact (~130 ps Alfven transit time)
    same_handedness=False  : adjacent spots have opposite circulation
                              (legacy v12.5 form)
    """
    sigma2 = SPOT_RADIUS_M ** 2
    A0     = B_SEED_T * SPOT_RADIUS_M * np.exp(0.5)  # gives peak |B|=B_SEED_T

    bx_terms = []
    bz_terms = []
    for k in range(N_SPOTS):
        angle = 2.0 * np.pi * k / N_SPOTS
        cx    = RING_RADIUS_M * np.cos(angle)
        cz    = RING_RADIUS_M * np.sin(angle)
        sk    = 1 if same_handedness else (-1)**k
        # Common Gaussian envelope expression
        gauss = (
            f"exp(-0.5 * ((x - {cx:.8e})**2 + (z - {cz:.8e})**2)"
            f" / {sigma2:.8e})"
        )
        # Bx = -s_k * A0 * (z-z_k)/sigma^2 * gauss
        bx_terms.append(
            f"({-sk * A0:.8e}"
            f" * (z - {cz:.8e}) / {sigma2:.8e}"
            f" * {gauss})"
        )
        # Bz = +s_k * A0 * (x-x_k)/sigma^2 * gauss
        bz_terms.append(
            f"({sk * A0:.8e}"
            f" * (x - {cx:.8e}) / {sigma2:.8e}"
            f" * {gauss})"
        )

    return (" + ".join(bx_terms),
            "0.0",
            " + ".join(bz_terms))


def get_initial_field_expressions():
    """Dispatch to the right (Bx, By, Bz) expressions based on seed_topology."""
    topo = getattr(args, 'seed_topology', 'harris')
    if topo == 'harris':
        # Legacy: scalar By bumps with alternating sign
        return ('0.0', build_By_expression(), '0.0')
    elif topo == 'vortex_same':
        return build_vortex_field_expressions(same_handedness=True)
    elif topo == 'vortex_alt':
        return build_vortex_field_expressions(same_handedness=False)
    else:
        raise ValueError(f"Unknown seed-topology: {topo}")


if args.no_bfield:
    # Baseline mode: skip analytic B-field init entirely. All particles see
    # B = 0 at t=0; the hybrid PIC solver will evolve B only from simulated
    # currents. Any energization observed in this run is from initialization
    # transients / density gradients / numerical artefacts — NOT from the
    # seed field. Subtract this from a B-on run to isolate the reconnection
    # signal.
    if rank == 0:
        print('  B-field init: DISABLED (--no-bfield baseline mode)')
        print('                Particles will see B = 0 at t=0; only simulation-'
              'evolved B is present')
elif args.field_mode == 'applied':
    bx_expr, by_expr, bz_expr = get_initial_field_expressions()
    topo = getattr(args, 'seed_topology', 'harris')
    if rank == 0:
        if topo == 'harris':
            print(f'  B-field init: 8 alternating-polarity By dipoles (Harris-style ring)')
            print(f'                Adjacent spots have opposite By polarity → 8 X-lines '
                  'at chord midpoints (PRE-SEEDED)')
            print(f'                peak |By| = {B_SEED_T:.1f} T at r = sigma from each '
                  'spot centre')
            preview = by_expr[:120].replace('\n',' ')
            print(f'  By preview:   {preview}...')
        elif topo == 'vortex_same':
            print(f'  B-field init: 8 SAME-HANDEDNESS in-plane vortices (Biermann-like)')
            print(f'                Bx, Bz from curl(A_y); By = 0 at t=0')
            print(f'                ALL 8 vortices same circulation; X-lines NOT pre-seeded')
            print(f'                peak |B| = {B_SEED_T:.1f} T at r = sigma from each spot')
            print(f'                Plasma may NOT naturally support this topology — expect')
            print(f'                decay unless --rotate-mode driving provides supporting Jy')
        elif topo == 'vortex_alt':
            print(f'  B-field init: 8 ALTERNATING-handedness in-plane vortices (v12.5 form)')
            print(f'                Bx, Bz from curl(A_y); adjacent spots opposite circulation')
            print(f'                peak |B| = {B_SEED_T:.1f} T at r = sigma from each spot')
    simulation.add_applied_field(picmi.AnalyticInitialField(
        Bx_expression=bx_expr,
        By_expression=by_expr,
        Bz_expression=bz_expr))


# ============================================================================
# FUSION DIAGNOSTIC — multi-channel
# ============================================================================
def diagnostic_step_set():
    steps = {0, n_steps}
    if EARLY_DIAG_PERIOD > 0:
        for s in range(0, min(EARLY_DIAG_STEPS, n_steps)+1, EARLY_DIAG_PERIOD):
            steps.add(s)
    if LATE_DIAG_PERIOD > 0:
        for s in range(0, n_steps+1, LATE_DIAG_PERIOD):
            steps.add(s)
    return sorted(steps)

def write_fusion_diag_header():
    """
    FIX I: Full header for all papers.
    Columns cover: physical time, per-reaction rates/powers/alphas,
    cumulative alphas per channel, centre/outer spatial metrics,
    wall-clock timing.
    """
    if rank != 0: return
    header = (
        # Time mapping (FIX B+C)
        ['iter', 'step_index', 'time_s', 'time_ps', 'time_ns',
         'delta_t_s', 'n_macro_protons', 'wall_clock_s']
        # Per-reaction fast fractions (FIX J — correct threshold per channel)
        + [f'fast_fraction_gt_{r["threshold_kev"]:.0f}kev_{r["key"]}'
           for r in ALL_RXNS if r['active']]
        # Per-reaction rates
        + [f'fusion_rate_{r["key"]}_s^-1' for r in ALL_RXNS if r['active']]
        # Per-reaction rates — ENERGY-RESOLVED via σ(E) (v12.6 FIX 4)
        # These are paper-quality numbers; use these for figures, NOT
        # the legacy rate_per_ff-times-fast-fraction columns above.
        + [f'fusion_rate_sigma_{r["key"]}_s^-1' for r in ALL_RXNS if r['active']]
        # Per-reaction powers
        + [f'fusion_power_{r["key"]}_w'   for r in ALL_RXNS if r['active']]
        # FIX D: alpha yield per step per channel
        + [f'alpha_yield_step_{r["key"]}' for r in ALL_RXNS if r['active']]
        # Energy-resolved alpha yield per step (v12.6 FIX 4)
        + [f'alpha_yield_step_sigma_{r["key"]}' for r in ALL_RXNS if r['active']]
        # FIX E: cumulative alpha count per channel
        + [f'cum_alpha_yield_{r["key"]}' for r in ALL_RXNS if r['active']]
        # Totals
        + ['alpha_yield_step_total',
           'fusion_power_total_w',
           'fusion_energy_step_j',
           'cumulative_fusion_energy_j',
           'gain_vs_laser_energy']
        # FIX F: centre/outer spatial metrics (Papers 1-5)
        + ['centre_E_mean_kev',
           'centre_E_95th_kev',
           'N_centre_particles',
           'outer_E_95th_kev',
           'centre_outer_ratio']
    )
    with open(FUSION_DIAG_FILE, 'w', newline='', encoding='utf-8') as fh:
        csv.writer(fh).writerow(header)

def write_fusion_accounting():
    if rank != 0: return
    with open(FUSION_ACCOUNTING_FILE, 'w', encoding='utf-8') as fh:
        fh.write('p-11B/p-7Li/LiB fusion accounting — v12.4.4\n' + '='*70 + '\n')
        fh.write(f'base_fuel={args.base_fuel}  rod_fuel={args.rod_fuel}  ring_fuel={args.ring_fuel}\n')
        fh.write(f'time_step_s={time_step_s:.16e}\n')
        fh.write(f'total_time_s={total_time_s:.16e}\n')
        fh.write(f'n_steps={n_steps}\n\n')
        fh.write('Fuel region densities:\n')
        for region, cfg in [('base',BASE),('rod',ROD),('ring',RING)]:
            fh.write(f'  {region}: density={cfg["density"]:.3e}  ')
            fh.write(f'H={cfg["n_h"]:.3e}  Li7={cfg["n_li7"]:.3e}  ')
            fh.write(f'Li6={cfg["n_li6"]:.3e}  B11={cfg["n_b11"]:.3e}\n')
        fh.write('\nReaction channels:\n')
        for rxn in ALL_RXNS:
            if rxn['active']:
                fh.write(f'[{rxn["key"]}] {rxn["label"]}\n')
                fh.write(f'  threshold_kev={rxn["threshold_kev"]}  ')
                fh.write(f'peak_kev={rxn["peak_kev"]}  ')
                fh.write(f'sigma_m2={rxn["sigma_m2"]:.3e}\n')
                fh.write(f'  eff_n_target={rxn["eff_n_target"]:.3e} m^-3\n')
                fh.write(f'  energy_mev={rxn["E_rxn_j"]/constants.q_e/1e6:.3f}\n')
                fh.write(f'  n_alphas={rxn["n_alphas"]}\n')
                fh.write(f'  rate_per_ff={rxn["rate_per_ff"]:.3e} rxn/s\n')
                fh.write(f'  power_per_ff={rxn["power_per_ff"]:.3e} W\n')
                fh.write(f'  alphas_per_ff={rxn["rate_per_ff"]*rxn["n_alphas"]:.3e} alphas/s\n\n')
        fh.write('Formulas:\n')
        fh.write('  R_i(rxn)     = N_p_total * ff_i(>E_thr_rxn) * n_target * sigma * v_thr\n')
        fh.write('  P_i(rxn)     = R_i * E_rxn_j\n')
        fh.write('  alphas_i(rxn)= R_i * n_alphas * delta_t_i\n')
        fh.write('  E_step_i     = sum_rxn P_i(rxn) * delta_t_i\n')
        fh.write('  E_fusion     = sum_i E_step_i\n')
        fh.write('  gain         = E_fusion / laser_energy_j\n\n')
        fh.write('Physical time mapping:\n')
        fh.write('  t_physical_s = step_index * time_step_s\n')
        fh.write('  iter = step_index (1:1 mapping preserved)\n\n')
        fh.write('CAVEAT: Net energy requires integrating P*dt over physical confinement time.\n')


def write_step_time_index_csv():
    """
    FIX A: Maps every diagnostic step to its physical time.
    This is the canonical clock-time reference for all papers.
    iter == step_index (1:1) — physical time = step * time_step_s
    """
    if rank != 0: return
    header = [
        'iter', 'step_index', 'time_s', 'time_ps', 'time_ns',
        'dt_s', 'dt_ps', 'delta_t_s', 'pct_of_run',
        'early_diag_step', 'late_diag_step', 'final_step',
        'laser_energy_j', 'physical_protons',
    ] + [f'eff_n_{r["key"]}_target_m3' for r in ALL_RXNS if r['active']]       + [f'rate_per_ff_{r["key"]}' for r in ALL_RXNS if r['active']]       + [f'alphas_per_ff_{r["key"]}' for r in ALL_RXNS if r['active']]       + ['net_energy_caveat']

    early_steps = ({s for s in range(0, min(EARLY_DIAG_STEPS,n_steps)+1, EARLY_DIAG_PERIOD)}
                   if EARLY_DIAG_PERIOD > 0 else set())
    late_steps  = ({s for s in range(0, n_steps+1, LATE_DIAG_PERIOD)}
                   if LATE_DIAG_PERIOD > 0 else set())
    steps = sorted({0, n_steps} | early_steps | late_steps)
    prev = None
    with open(STEP_INDEX_FILE, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(header)
        for step in steps:
            t_s = step * time_step_s
            dt  = 0.0 if prev is None else (step - prev) * time_step_s
            prev = step
            row = [
                step, step,
                f'{t_s:.16e}', f'{t_s*1e12:.9e}', f'{t_s*1e9:.9e}',
                f'{time_step_s:.16e}', f'{time_step_s*1e12:.9e}',
                f'{dt:.16e}',
                f'{100.0*step/n_steps:.6f}' if n_steps>0 else '0.0',
                int(EARLY_DIAG_PERIOD > 0 and step<=EARLY_DIAG_STEPS and step%EARLY_DIAG_PERIOD==0),
                int(LATE_DIAG_PERIOD > 0 and step%LATE_DIAG_PERIOD==0),
                int(step==n_steps),
                f'{LASER_ENERGY_J:.6e}',
                f'{PHYSICAL_PROTONS:.6e}',
            ] + [f'{r["eff_n_target"]:.6e}' for r in ALL_RXNS if r['active']]               + [f'{r["rate_per_ff"]:.6e}'   for r in ALL_RXNS if r['active']]               + [f'{r["rate_per_ff"]*r["n_alphas"]:.6e}' for r in ALL_RXNS if r['active']]               + ['t_physical=step*time_step_s; net_energy_requires_time_integration']
            w.writerow(row)


write_fusion_diag_header()
write_fusion_accounting()
write_step_time_index_csv()


# ── Live particle access helpers ─────────────────────────────────────────────
def _to_cpu(x):
    """Convert cupy array to numpy if needed, else pass through.
    Required for WarpX-CUDA: particle data lives on GPU as cupy arrays.
    Implicit conversion via np.asarray(cupy_array) raises TypeError;
    we must call .get() explicitly."""
    if x is None:
        return None
    # cupy arrays have a .get() method that copies to CPU
    if hasattr(x, 'get') and hasattr(x, 'device'):
        try:
            return x.get()
        except Exception:
            pass
    return x

def _flatten(parts):
    if parts is None: return None
    parts = _to_cpu(parts)
    if isinstance(parts, np.ndarray): return np.asarray(parts).ravel()
    if isinstance(parts, (list,tuple)):
        arrs = []
        for i in parts:
            if i is None: continue
            i = _to_cpu(i)
            if np.asarray(i).size > 0:
                arrs.append(np.asarray(i).ravel())
        return np.concatenate(arrs) if arrs else np.asarray([], dtype=float)
    try:
        parts = _to_cpu(parts)
        return np.asarray(parts).ravel()
    except: return None

def _call_first(obj, names, **kw):
    """Try each method name in order. Records last error for diagnostics."""
    last_err = None
    for n in names:
        fn = getattr(obj, n, None)
        if fn is None:
            last_err = f'{n}:no_attr'
            continue
        try:
            return fn(**kw)
        except TypeError as e:
            try:
                return fn()
            except Exception as e2:
                last_err = f'{n}:{type(e2).__name__}:{e2}'
                continue
        except Exception as e:
            last_err = f'{n}:{type(e).__name__}:{e}'
            continue
    # Stash on the function so callers can inspect why we failed
    _call_first.last_err = last_err
    return None
_call_first.last_err = None

def _resolve_proton_container():
    """Locate the per-species proton particle container.

    pywarpx 26.04 changed the layout: simulation.particles.get('proton')
    returns the MultiParticleContainer (the registry), not the per-species
    container. The actual proton container has to be fetched via
    multi_pc.get_particle_container_from_name('proton') OR via the deprecated
    pywarpx.particle_containers.ParticleContainerWrapper('proton'). This
    function tries both routes in order and returns (pc, route_label, err).
    """
    err_chain = []

    # Route A: simulation.particles -> get_particle_container_from_name
    try:
        multi_pc = simulation.particles.get('proton')
    except Exception as e:
        multi_pc = None
        err_chain.append(f'sim.particles.get:{type(e).__name__}:{e}')

    if multi_pc is not None:
        # If it already has per-particle accessors, use it directly (covers
        # any future API change where simulation.particles.get returns the
        # per-species container).
        if hasattr(multi_pc, 'get_particle_x') or hasattr(multi_pc, 'get_particle_ux'):
            return multi_pc, 'simulation.particles.get(direct)', None

        # Otherwise dereference via the documented method
        deref = getattr(multi_pc, 'get_particle_container_from_name', None)
        if deref is not None:
            try:
                pc = deref('proton')
                if pc is not None:
                    return pc, 'multi.get_particle_container_from_name', None
                err_chain.append('multi.get_particle_container_from_name:returned_None')
            except Exception as e:
                err_chain.append(f'multi.get_particle_container_from_name:{type(e).__name__}:{e}')
        else:
            err_chain.append('multi.get_particle_container_from_name:no_attr')

    # Route B: deprecated wrapper (still works in pywarpx 26.04, just warns)
    try:
        from pywarpx import particle_containers
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            pc = particle_containers.ParticleContainerWrapper('proton')
        if pc is not None:
            return pc, 'ParticleContainerWrapper', None
        err_chain.append('ParticleContainerWrapper:returned_None')
    except Exception as e:
        err_chain.append(f'ParticleContainerWrapper:{type(e).__name__}:{e}')

    return None, None, '|'.join(err_chain)


def get_proton_ke_kev():
    """Fetch proton kinetic energies from the live particle container.

    Returns (ke_kev, weights, status) — ke_kev/weights are None on failure
    and status is a descriptive string explaining what happened. The status
    string is logged once per failure mode to avoid spam.
    """
    pc, pc_route, pc_err = _resolve_proton_container()
    if pc is None:
        return None, None, f'no_container:{pc_err}'

    # Stage 2: pull arrays. Try three kwarg variants for level selection.
    ux = uy = uz = w = None
    last_kw_err = None
    for kw in [{'lev':0}, {'level':0}, {}]:
        ux = _flatten(_call_first(pc, ['get_particle_ux','get_particle_px'], **kw))
        ux_err = _call_first.last_err
        uy = _flatten(_call_first(pc, ['get_particle_uy','get_particle_py'], **kw))
        uz = _flatten(_call_first(pc, ['get_particle_uz','get_particle_pz'], **kw))
        w  = _flatten(_call_first(pc, ['get_particle_weight','get_particle_weighting'], **kw))
        if ux is not None and uy is not None and uz is not None:
            break
        last_kw_err = f'kw={kw} ux_err={ux_err}'

    if ux is None or uy is None or uz is None:
        # List what methods the container actually has, so we can fix the names
        avail = [m for m in dir(pc) if 'particle' in m.lower() and not m.startswith('_')]
        return None, None, (f'no_arrays via {pc_route}; last_err={last_kw_err}; '
                            f'available_methods={avail[:25]}')

    n = min(ux.size, uy.size, uz.size)
    if n == 0:
        return None, None, f'empty_arrays via {pc_route}'

    ux = ux[:n].astype(float, copy=False)
    uy = uy[:n].astype(float, copy=False)
    uz = uz[:n].astype(float, copy=False)
    w  = (w[:n].astype(float, copy=False)
          if w is not None and w.size >= n else np.ones(n))
    u2    = ux*ux + uy*uy + uz*uz
    gamma = np.sqrt(1.0 + u2/(constants.c*constants.c))
    ke    = (gamma-1.0)*PROTON_MASS_KG*constants.c*constants.c/(1e3*constants.q_e)
    valid = np.isfinite(ke) & np.isfinite(w) & (w>0)
    return ke[valid], w[valid], f'ok({pc_route},n={n})'


def should_diag(step):
    if step<0 or step>n_steps: return False
    if step==0 or step==n_steps: return True
    if EARLY_DIAG_PERIOD > 0 and step<=EARLY_DIAG_STEPS and step%EARLY_DIAG_PERIOD==0:
        return True
    if LATE_DIAG_PERIOD > 0 and step%LATE_DIAG_PERIOD==0:
        return True
    return False


def fusion_diagnostic_callback():
    """
    v12.4 complete diagnostic callback.
    FIX B: t_s = step * time_step_s (was missing)
    FIX C: time_ns column added
    FIX D: alpha_yield per reaction per step (rate * n_alphas * dt)
    FIX E: cumulative alpha count per channel
    FIX F: live centre/outer proton E_95th and ratio
    FIX J: per-reaction fast_fraction with correct threshold per channel
    FIX K: wall-clock time column
    """
    step = simulation.extension.warpx.getistep(lev=0)
    if not should_diag(step) or fusion_diag_state['last_step'] == step:
        return

    import time as _time
    wall_t = _time.perf_counter()

    # FIX B: correct physical time calculation
    t_s     = step * time_step_s
    delta_t = 0.0 if fusion_diag_state['last_time_s'] is None \
              else max(0.0, t_s - fusion_diag_state['last_time_s'])

    ke_kev, weights, status = get_proton_ke_kev()
    n_macro = 0

    if ke_kev is None or weights is None:
        # Track unique failure modes so we get one log line per distinct mode,
        # not one per step. The previous implementation logged only the first
        # failure ever, which silently hid 10000 NaN rows behind a single line.
        seen = fusion_diag_state.setdefault('seen_failures', set())
        if rank == 0 and status not in seen:
            seen.add(status)
            print(f'  [fusion_diag] step={step} particle access FAILED: {status}',
                  flush=True)
        ffs    = {r['key']: float('nan') for r in ALL_RXNS}
        rates  = ffs.copy()
        powers = ffs.copy()
        alphas_step = ffs.copy()
    else:
        # MPI reduce weight totals
        wt    = comm.allreduce(float(np.sum(weights)), op=mpi.SUM)
        n_mac = comm.allreduce(int(len(weights)),       op=mpi.SUM)
        n_macro = n_mac

        # FIX J: per-reaction fast_fraction with correct threshold per channel
        ffs = {}; rates = {}; powers = {}; alphas_step = {}
        for rxn in ALL_RXNS:
            if rxn['active']:
                fast_mask = ke_kev >= rxn['threshold_kev']
                wf = comm.allreduce(float(np.sum(weights[fast_mask])), op=mpi.SUM)
                ff = (wf / wt) if wt > 0 else float('nan')
                ffs[rxn['key']]         = ff
                rates[rxn['key']]       = rxn['rate_per_ff'] * ff if ff == ff else float('nan')
                powers[rxn['key']]      = rxn['power_per_ff'] * ff if ff == ff else float('nan')
                # FIX D: alpha yield per step per channel
                alphas_step[rxn['key']] = (rates[rxn['key']] * rxn['n_alphas'] * delta_t
                                           if ff == ff else float('nan'))

                # ── ENERGY-RESOLVED RATE (FIX 4 / v12.6) ─────────────────────
                # Fold actual proton f(E) over σ(E) curve. This is the
                # paper-quality number, not the rate_per_ff approximation.
                # Only includes protons above the integration floor to avoid
                # the unreliable extrapolation region of the σ(E) table.
                cut = ke_kev >= SIGMA_INTEGRATION_MIN_KEV
                if np.any(cut):
                    rate_sigma_local = compute_energy_resolved_rate(
                        rxn_key                     = rxn['key'],
                        ke_kev_protons              = ke_kev[cut],
                        weights_protons             = weights[cut],
                        n_target_m3                 = rxn['eff_n_target'],
                        plasma_volume_m3            = 0.0,  # unused
                        physical_protons_per_macro  = 1.0,
                    )
                else:
                    rate_sigma_local = 0.0
                # MPI-reduce across ranks (each rank holds a subset of particles)
                rate_sigma_total = comm.allreduce(rate_sigma_local, op=mpi.SUM)
                rates[rxn['key'] + '_sigma'] = rate_sigma_total
                # Energy-resolved alpha yield per step
                alphas_step[rxn['key'] + '_sigma'] = (
                    rate_sigma_total * rxn['n_alphas'] * delta_t)
            else:
                ffs[rxn['key']] = rates[rxn['key']] = powers[rxn['key']] = float('nan')
                alphas_step[rxn['key']] = float('nan')
                rates[rxn['key'] + '_sigma'] = float('nan')
                alphas_step[rxn['key'] + '_sigma'] = float('nan')

    total_power = sum(p for p in powers.values() if p == p) or float('nan')
    step_energy = total_power * delta_t if total_power == total_power else float('nan')
    total_alphas_step = sum(a for a in alphas_step.values() if a == a) or float('nan')

    if step_energy == step_energy:
        fusion_diag_state['cumulative_energy_j'] += step_energy
    cum_energy = fusion_diag_state['cumulative_energy_j']

    # FIX E: cumulative alpha count per channel
    for rxn in ALL_RXNS:
        k = rxn['key']
        if alphas_step.get(k, float('nan')) == alphas_step.get(k, float('nan')):
            fusion_diag_state.setdefault(f'cum_alphas_{k}', 0.0)
            fusion_diag_state[f'cum_alphas_{k}'] += alphas_step.get(k, 0.0)

    cum_alphas = {r['key']: fusion_diag_state.get(f'cum_alphas_{r["key"]}', 0.0)
                  for r in ALL_RXNS}
    gain = cum_energy / LASER_ENERGY_J if LASER_ENERGY_J > 0 else float('nan')

    # FIX F: live centre/outer proton energy metrics
    # R_CENTRE scales as 0.25 * R_RING (matches Mac reference 800 um -> 200 um).
    # At current 2400 um ring this gives R_CENTRE = 600 um.
    # Captures truly converged ions (centre population) rather than X-line ions.
    R_CENTRE = 0.25 * RING_RADIUS_M if RING_RADIUS_M > 0 else 200e-6
    centre_metrics = {'E_mean': float('nan'), 'E_95th': float('nan'),
                      'N_centre': 0, 'outer_E_95th': float('nan'), 'ratio': float('nan')}
    if ke_kev is not None and weights is not None:
        try:
            pc, _route, _err = _resolve_proton_container()
            if pc is None:
                raise RuntimeError(f'no_container_for_centre_outer:{_err}')
            px_ = uz_ = None
            for kw in [{'lev':0}, {'level':0}, {}]:
                px_ = _flatten(_call_first(pc, ['get_particle_x'], **kw))
                pz_ = _flatten(_call_first(pc, ['get_particle_z'], **kw))
                if px_ is not None and pz_ is not None:
                    break
            if px_ is not None and pz_ is not None:
                n    = min(px_.size, pz_.size, ke_kev.size)
                r_   = np.sqrt(px_[:n]**2 + pz_[:n]**2)
                c_   = r_ < R_CENTRE
                if c_.sum() > 10:
                    E_c  = ke_kev[:n][c_];  E_o = ke_kev[:n][~c_]
                    e95c = comm.allreduce(float(np.percentile(E_c, 95)) if E_c.size>0 else 0.0, op=mpi.MAX)
                    e95o = comm.allreduce(float(np.percentile(E_o, 95)) if E_o.size>0 else 0.0, op=mpi.MAX)
                    nc   = comm.allreduce(int(c_.sum()), op=mpi.SUM)
                    centre_metrics = {
                        'E_mean':     float(np.mean(E_c)) if E_c.size>0 else float('nan'),
                        'E_95th':     e95c,
                        'N_centre':   nc,
                        'outer_E_95th': e95o,
                        'ratio':      (e95c/e95o) if e95o > 0 else float('nan'),
                    }
        except Exception:
            pass

    # FIX K: wall-clock elapsed
    wall_elapsed = _time.perf_counter() - wall_t

    def fmt(v):
        return f'{v:.10e}' if (v == v and v != float('inf')) else 'nan'

    if rank == 0:
        row = (
            # FIX B+C: iter, step_index, time_s, time_ps, time_ns, delta_t
            [step, step, f'{t_s:.16e}', f'{t_s*1e12:.6e}', f'{t_s*1e9:.6e}',
             f'{delta_t:.16e}', n_macro,
             # FIX K: wall-clock
             f'{wall_elapsed:.4f}']
            # FIX J: per-reaction fast_fraction
            + [fmt(ffs.get(r['key'], float('nan'))) for r in ALL_RXNS if r['active']]
            # reaction rates
            + [fmt(rates.get(r['key'], float('nan')))  for r in ALL_RXNS if r['active']]
            # reaction rates — ENERGY-RESOLVED (v12.6 FIX 4)
            + [fmt(rates.get(r['key'] + '_sigma', float('nan'))) for r in ALL_RXNS if r['active']]
            # reaction powers
            + [fmt(powers.get(r['key'], float('nan'))) for r in ALL_RXNS if r['active']]
            # FIX D: alpha yield per step per channel
            + [fmt(alphas_step.get(r['key'], float('nan'))) for r in ALL_RXNS if r['active']]
            # alpha yield — ENERGY-RESOLVED (v12.6 FIX 4)
            + [fmt(alphas_step.get(r['key'] + '_sigma', float('nan'))) for r in ALL_RXNS if r['active']]
            # FIX E: cumulative alphas per channel
            + [fmt(cum_alphas.get(r['key'], 0.0)) for r in ALL_RXNS if r['active']]
            # totals
            + [fmt(total_alphas_step), fmt(total_power), fmt(step_energy),
               fmt(cum_energy), fmt(gain)]
            # FIX F: centre/outer metrics
            + [fmt(centre_metrics['E_mean']),   fmt(centre_metrics['E_95th']),
               str(centre_metrics['N_centre']), fmt(centre_metrics['outer_E_95th']),
               fmt(centre_metrics['ratio'])]
        )
        with open(FUSION_DIAG_FILE, 'a', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(row)

        if step % max(50, LATE_DIAG_PERIOD) == 0 and total_power == total_power:
            rxn_str = '  '.join(
                f'{r["key"]}:{fmt(rates.get(r["key"],float("nan")))}/s'
                for r in ALL_RXNS if r['active']
            )
            ratio_str = fmt(centre_metrics['ratio'])
            print(
                f'  [diag] iter={step:6d} t={t_s*1e12:.1f}ps'
                f' | {rxn_str}'
                f' | P={fmt(total_power)}W E_cum={fmt(cum_energy)}J'
                f' | centre/outer={ratio_str}'
                f' | gain={fmt(gain)}',
                flush=True
            )

    fusion_diag_state['last_step']   = step
    fusion_diag_state['last_time_s'] = t_s


def status_callback():
    # FIX G: rotation phase properly logged
    step = simulation.extension.warpx.getistep(lev=0)
    if step % 50 == 0 and rank == 0:
        t    = step * time_step_s
        pct  = 100.0 * step / n_steps
        ramp = 1.0 - np.exp(-t/RAMP_TIME_S) if RAMP_TIME_S>0 else 1.0
        msg  = (f'  Step {step:6d} | iter={step:6d} | t={t*1e12:8.2f}ps'
                f' | {pct:5.1f}% | ramp={ramp:5.3f} | {args.field_mode}')
        if IS_ROTATING:
            full_rots = int(ROTATION_FREQ_HZ * t)
            phase_deg = (ROTATION_FREQ_HZ * t % 1.0) * 360.0
            print(msg + f' | rotation {full_rots}+{phase_deg:5.1f}deg', flush=True)
        else:
            print(msg + ' [STATIC]', flush=True)

callbacks.installafterstep(status_callback)
callbacks.installafterstep(fusion_diagnostic_callback)


# ============================================================================
# RECONNECTION RATE DIAGNOSTIC
# ============================================================================
# Measures the canonical normalized reconnection rate at each X-line:
#
#     R = E_y_xline / (v_A * B_lobe)
#
# where E_y_xline is the reconnecting electric field component perpendicular
# to the simulation plane sampled at the X-line, B_lobe is the upstream
# lobe field magnitude (sampled at the spot centre adjacent to the X-line),
# and v_A is the local Alfvén speed in the lobe.
#
# N_SPOTS X-lines are located at the midpoints between adjacent spots on the
# ring, at angle (2π(k+0.5)/N_SPOTS) and radius R*cos(π/N_SPOTS).
# v12.12: skipped entirely when N_SPOTS=1 (no reconnection topology with one spot).
#
# Fast reconnection literature: R ≈ 0.1 in collisionless regimes (Birn+2001,
# Comisso+ 2016). Slow Sweep-Parker reconnection: R ~ 1/sqrt(S_Lundquist).
# For a single-pulse paper this number, sampled at every diagnostic step,
# is the canonical reviewer-expected measurement.

# Pre-compute X-line positions (and their adjacent spot positions) once.
# v12.12: skip entirely when SINGLE_SPOT_MODE — there are no X-lines with one spot.
_xline_positions = []  # list of (k_idx, x_xl, z_xl, x_spot_left, z_spot_left, x_spot_right, z_spot_right)
if not SINGLE_SPOT_MODE:
    for _k in range(N_SPOTS):
        # X-line k is between spot k and spot k+1 (mod N_SPOTS)
        _angle_xl = 2 * np.pi * (_k + 0.5) / N_SPOTS
        _r_xl     = RING_RADIUS_M * np.cos(np.pi / N_SPOTS)
        _x_xl, _z_xl = _r_xl * np.cos(_angle_xl), _r_xl * np.sin(_angle_xl)

        _angle_l, _angle_r = 2*np.pi*_k/N_SPOTS, 2*np.pi*((_k+1) % N_SPOTS)/N_SPOTS
        _x_sl, _z_sl = RING_RADIUS_M*np.cos(_angle_l), RING_RADIUS_M*np.sin(_angle_l)
        _x_sr, _z_sr = RING_RADIUS_M*np.cos(_angle_r), RING_RADIUS_M*np.sin(_angle_r)

        _xline_positions.append((_k, _x_xl, _z_xl, _x_sl, _z_sl, _x_sr, _z_sr))

RECONNECTION_DIAG_FILE = os.path.join(OUTDIR, 'reconnection_rate_by_iter.csv')


def _sample_field_at(field_array, x_target, z_target, lo_x, lo_z, dx, dz):
    """Return scalar field value at (x_target, z_target) using nearest-cell
    sampling. field_array is a 2D numpy array indexed [iz, ix] (WarpX convention).
    Returns NaN if the target is outside the grid."""
    if field_array is None:
        return float('nan')
    nz, nx = field_array.shape
    ix = int(round((x_target - lo_x) / dx))
    iz = int(round((z_target - lo_z) / dz))
    if 0 <= ix < nx and 0 <= iz < nz:
        return float(field_array[iz, ix])
    return float('nan')


def _try_get_field_array(component):
    """Attempt to fetch a 2D field array (component in 'Bx','By','Bz','Ex','Ey','Ez')
    from the running WarpX simulation. Returns (array, lo_x, lo_z, dx, dz) or
    (None,...) if the field can't be accessed (e.g. early in init, GPU build
    differences, etc.). Designed to never raise — diagnostic is best-effort."""
    try:
        wx = simulation.extension.warpx
        # PyWarpX field-access API: getter names follow pattern getEx, getBx, etc.
        getter_name = f'get{component}'
        if not hasattr(wx, getter_name):
            return None, 0, 0, 0, 0
        # Level 0, full domain (include guards = False)
        getter = getattr(wx, getter_name)
        try:
            arr = getter(lev=0, include_ghost=False)
        except TypeError:
            # Older API may use different keyword
            try:
                arr = getter(0)
            except Exception:
                return None, 0, 0, 0, 0
        # arr can be a list of subdomain blocks (one per MPI rank). Concatenate
        # if needed; for our 8-rank Cartesian2D case rank 0 typically has the
        # full domain in unified-memory builds.
        if isinstance(arr, (list, tuple)) and len(arr) > 0:
            arr = arr[0]
        arr = _to_cpu(arr)
        arr = np.asarray(arr)
        if arr.ndim != 2:
            # Unexpected shape, give up gracefully
            return None, 0, 0, 0, 0

        # Get grid extents from the simulation grid object
        lo_x = grid.lower_bound[0]
        lo_z = grid.lower_bound[1]
        hi_x = grid.upper_bound[0]
        hi_z = grid.upper_bound[1]
        nx_grid = grid.number_of_cells[0]
        nz_grid = grid.number_of_cells[1]
        dx = (hi_x - lo_x) / nx_grid
        dz = (hi_z - lo_z) / nz_grid
        return arr, lo_x, lo_z, dx, dz
    except Exception:
        return None, 0, 0, 0, 0


_recon_diag_state = {'last_step': -1, 'header_written': False}


def reconnection_rate_callback():
    """Sample E_y and |B_lobe| at each of the 8 X-lines, compute the
    normalized reconnection rate, and append to CSV. Best-effort: silently
    writes NaNs if the field arrays can't be accessed at this step."""
    step = simulation.extension.warpx.getistep(lev=0)
    if not should_diag(step):
        return
    if _recon_diag_state['last_step'] == step:
        return
    _recon_diag_state['last_step'] = step

    if rank != 0:
        # Only rank 0 writes the CSV. Field arrays may be MPI-distributed,
        # but for this diagnostic we accept the rank-0-local view; X-lines
        # close to the rank-0 subdomain will be sampled, others will be NaN.
        # In a tight-budget paper-quality measurement you'd MPI-reduce; here
        # we accept this limitation since the X-line count is small (8) and
        # a fully-replicated field on rank 0 (typical for unified-memory
        # GPU builds) gives correct numbers.
        return

    t_s = step * time_step_s

    # Fetch fields (best-effort; may return None at very early steps)
    Ey_arr, lo_x, lo_z, dx, dz = _try_get_field_array('Ey')
    Bx_arr, *_                  = _try_get_field_array('Bx')
    Bz_arr, *_                  = _try_get_field_array('Bz')

    # Header write (lazy — first time only)
    if not _recon_diag_state['header_written']:
        with open(RECONNECTION_DIAG_FILE, 'w', newline='', encoding='utf-8') as fh:
            cols = ['step', 't_ps']
            for k in range(N_SPOTS):
                cols += [f'Ey_xl{k}_Vm', f'B_lobe_l{k}_T', f'B_lobe_r{k}_T',
                         f'rate_xl{k}']
            cols += ['rate_mean', 'rate_max', 'rate_min']
            fh.write(','.join(cols) + '\n')
        _recon_diag_state['header_written'] = True

    # Sample at each X-line + adjacent spot pair
    rates = []
    row = [step, t_s * 1e12]
    for (k, x_xl, z_xl, x_sl, z_sl, x_sr, z_sr) in _xline_positions:
        Ey_xl  = _sample_field_at(Ey_arr, x_xl, z_xl, lo_x, lo_z, dx, dz)

        # B_lobe magnitude at the two adjacent spot centres (use the larger
        # of the two as the upstream value, since reconnection drives between
        # the lobes and either spot can be the relevant lobe).
        Bx_l, Bz_l = _sample_field_at(Bx_arr, x_sl, z_sl, lo_x, lo_z, dx, dz), \
                     _sample_field_at(Bz_arr, x_sl, z_sl, lo_x, lo_z, dx, dz)
        Bx_r, Bz_r = _sample_field_at(Bx_arr, x_sr, z_sr, lo_x, lo_z, dx, dz), \
                     _sample_field_at(Bz_arr, x_sr, z_sr, lo_x, lo_z, dx, dz)

        B_lobe_l = np.sqrt(Bx_l**2 + Bz_l**2) if not (np.isnan(Bx_l) or np.isnan(Bz_l)) else float('nan')
        B_lobe_r = np.sqrt(Bx_r**2 + Bz_r**2) if not (np.isnan(Bx_r) or np.isnan(Bz_r)) else float('nan')
        B_lobe = max(B_lobe_l, B_lobe_r) if not (np.isnan(B_lobe_l) and np.isnan(B_lobe_r)) else float('nan')

        # Normalized reconnection rate. v_A here is the global Alfvén speed
        # (computed once at startup). For more accuracy you'd compute a local
        # v_A from the local density and B_lobe; this global value is a
        # good first approximation for paper-quality diagnostics.
        if not np.isnan(Ey_xl) and not np.isnan(B_lobe) and B_lobe > 0 and v_A > 0:
            rate = abs(Ey_xl) / (v_A * B_lobe)
        else:
            rate = float('nan')
        rates.append(rate)

        row += [f'{Ey_xl:.6e}', f'{B_lobe_l:.6e}', f'{B_lobe_r:.6e}', f'{rate:.6e}']

    valid_rates = [r for r in rates if not np.isnan(r)]
    if valid_rates:
        row += [f'{np.mean(valid_rates):.6e}', f'{np.max(valid_rates):.6e}',
                f'{np.min(valid_rates):.6e}']
    else:
        row += ['nan', 'nan', 'nan']

    with open(RECONNECTION_DIAG_FILE, 'a', newline='', encoding='utf-8') as fh:
        fh.write(','.join(str(v) for v in row) + '\n')


# v12.12: only install reconnection-rate diagnostic when there are X-lines to sample.
# In SINGLE_SPOT_MODE (--n-spots 1), no reconnection topology exists.
if not SINGLE_SPOT_MODE:
    callbacks.installafterstep(reconnection_rate_callback)


# ============================================================================
# v14 STEP 1: CLOSED-LOOP FEASIBILITY TEST (observation-only)
# ============================================================================
# Installs a callback that reads B-field state every N steps and writes a CSV.
# Pure diagnostic - no source modification. Used to verify (1) WarpX callbacks
# can read field arrays at runtime, and (2) what the |B|(t) trajectory looks
# like for triggering decisions in the eventual closed-loop drive.
#
# Output columns: step, t_ps, B_max_T, B_min_T, B_mean_T, dB_dt_T_per_ns, wall_t_s
# Diagnostic-only; raises no exceptions, writes NaN if field arrays unavailable.

if args.monitor_bfield:
    import time as _time_mod

    MONITOR_DIAG_FILE = os.path.join(OUTDIR, 'closed_loop_monitor.csv')
    MONITOR_INTERVAL  = max(1, args.monitor_interval)

    _monitor_state = {
        'last_step': -1,
        'header_written': False,
        'last_b_max': float('nan'),
        'last_t_s': 0.0,
        't_callback_start': _time_mod.time(),
        'samples_taken': 0,
    }

    def closed_loop_monitor_callback():
        """Step 1 feasibility test: read |B| state, write CSV. Observation-only.

        At each invocation:
          - Reads Bx, By, Bz field arrays from the running WarpX simulation
          - Computes |B| = sqrt(Bx^2 + By^2 + Bz^2) on the rank-0 subdomain
          - Calculates B_max, B_min, B_mean across that subdomain
          - Computes dB/dt from the previous sample
          - Writes one CSV row

        On any failure (field array unavailable, MPI race, etc.) writes NaN
        instead of raising. Designed to never crash the simulation.
        """
        try:
            step = simulation.extension.warpx.getistep(lev=0)
        except Exception:
            return
        if step % MONITOR_INTERVAL != 0:
            return
        if _monitor_state['last_step'] == step:
            return
        _monitor_state['last_step'] = step

        # Only rank 0 writes; field arrays are taken from rank 0's local view
        # (matches the pattern used by reconnection_rate_callback).
        if rank != 0:
            return

        t_s  = step * time_step_s
        t_ps = t_s * 1e12
        wall_t_s = _time_mod.time() - _monitor_state['t_callback_start']

        # Header write (lazy - first time only)
        if not _monitor_state['header_written']:
            with open(MONITOR_DIAG_FILE, 'w', newline='', encoding='utf-8') as fh:
                fh.write('step,t_ps,B_max_T,B_min_T,B_mean_T,'
                         'dB_dt_T_per_ns,wall_t_s,n_cells_sampled\n')
            _monitor_state['header_written'] = True

        # Read field arrays (best-effort)
        Bx_arr, *_ = _try_get_field_array('Bx')
        By_arr, *_ = _try_get_field_array('By')
        Bz_arr, *_ = _try_get_field_array('Bz')

        b_max = float('nan')
        b_min = float('nan')
        b_mean = float('nan')
        n_cells = 0

        try:
            if Bx_arr is not None and By_arr is not None and Bz_arr is not None:
                # Compute |B| on the rank-0 subdomain
                B_mag_sq = (np.asarray(Bx_arr)**2 +
                            np.asarray(By_arr)**2 +
                            np.asarray(Bz_arr)**2)
                # Filter out any NaN/Inf values that might appear at boundaries
                B_mag_sq_finite = B_mag_sq[np.isfinite(B_mag_sq)]
                if B_mag_sq_finite.size > 0:
                    B_mag = np.sqrt(B_mag_sq_finite)
                    b_max  = float(np.max(B_mag))
                    b_min  = float(np.min(B_mag))
                    b_mean = float(np.mean(B_mag))
                    n_cells = int(B_mag_sq_finite.size)
        except Exception:
            # Compute failed - write NaNs but don't crash
            pass

        # Compute dB/dt (T/ns) from last sample
        dt_ns = (t_s - _monitor_state['last_t_s']) * 1e9
        if (not np.isnan(b_max) and not np.isnan(_monitor_state['last_b_max'])
                and dt_ns > 0):
            db_dt = (b_max - _monitor_state['last_b_max']) / dt_ns
        else:
            db_dt = float('nan')

        _monitor_state['last_b_max'] = b_max
        _monitor_state['last_t_s']   = t_s
        _monitor_state['samples_taken'] += 1

        # Write row
        with open(MONITOR_DIAG_FILE, 'a', newline='', encoding='utf-8') as fh:
            fh.write(f'{step},{t_ps:.6e},{b_max:.6e},{b_min:.6e},'
                     f'{b_mean:.6e},{db_dt:.6e},{wall_t_s:.3f},{n_cells}\n')

    callbacks.installafterstep(closed_loop_monitor_callback)
    if rank == 0:
        print(f'  v14 STEP 1: closed-loop monitor ENABLED')
        print(f'              monitor interval: every {MONITOR_INTERVAL} steps '
              f'(~{MONITOR_INTERVAL * time_step_s * 1e12:.1f} ps)')
        print(f'              output: {MONITOR_DIAG_FILE}')


# ============================================================================
# DIAGNOSTICS
# ============================================================================
if rank == 0:
    profile_label = f' (profile={DIAG_PROFILE})' if DIAG_PROFILE != 'custom' else ''
    if LATE_DIAG_PERIOD == 0:
        print(f'  Early diag: period={EARLY_DIAG_PERIOD} for first {EARLY_DIAG_STEPS} steps'
              f'  |  Late diag: DISABLED{profile_label}')
    else:
        print(f'  Early diag: period={EARLY_DIAG_PERIOD}  Late diag: period={LATE_DIAG_PERIOD}{profile_label}')
    print(f'  Fusion CSV: {FUSION_DIAG_FILE}')

# Field + particle diagnostics. The 'fields_early' / 'particles_early' diagnostics
# fire every EARLY_DIAG_PERIOD steps and are scoped to the first EARLY_DIAG_STEPS
# steps via the existing simulation step-bound logic. The 'fields' / 'particles'
# diagnostics handle late-time dumps; setting LATE_DIAG_PERIOD=0 disables them
# entirely (used by preflight and test profiles).

# v12.17 FIX: DUMP_PERIOD is now CLI-configurable via --dump-period.
# Default is 2500 steps (~375 ps at dt=150fs) for paper-quality late-time runs.
# Use --dump-period 200 for fine-grained early-time exploration (~30 ps cadence).
DUMP_PERIOD = args.dump_period

if rank == 0:
    print(f'  Diagnostic period: every {DUMP_PERIOD} steps (~{DUMP_PERIOD * 150e-3:.0f} ps at dt=150fs)')
    print(f'  Total dumps for {n_steps}-step run: {n_steps // DUMP_PERIOD + 1}')
    _dump_size_gb = 17  # measured per-dump size at 512^2
    print(f'  Estimated peak disk: ~{(n_steps // DUMP_PERIOD + 1) * _dump_size_gb} GB')

simulation.add_diagnostic(picmi.FieldDiagnostic(
    name='fields', grid=grid, period=DUMP_PERIOD,
    data_list=['B','E','J','rho'],
    write_dir=OUTDIR, warpx_format='openpmd',
    warpx_openpmd_backend='h5'))

simulation.add_diagnostic(picmi.ParticleDiagnostic(
    name='particles', period=DUMP_PERIOD, species=all_species,
    data_list=['position','momentum','weighting'],
    write_dir=OUTDIR, warpx_format='openpmd',
    warpx_openpmd_backend='h5'))


# ============================================================================
# METADATA
# ============================================================================
if rank == 0:
    with open(META_FILE, 'w', encoding='utf-8') as fh:
        fh.write('v12.14 per-region fuel config run metadata\n' + '='*60 + '\n')
        fh.write(f'version=12.14\ntimestamp={datetime.now().isoformat()}\n')
        fh.write(f'outdir={OUTDIR}\n')
        fh.write('\n[geometry]\n')
        fh.write(f'n_spots={N_SPOTS}\n')
        fh.write(f'single_spot_mode={SINGLE_SPOT_MODE}\n')
        fh.write(f'ring_radius_m={RING_RADIUS_M:.3e}\n')
        fh.write(f'spot_radius_m={SPOT_RADIUS_M:.3e}\n')
        fh.write('\n[fuel_regions]\n')
        for region, cfg, flag in [('base',BASE,'always'),
                                   ('rod', ROD, str(FUEL_ROD)),
                                   ('ring',RING,str(FUEL_ANY))]:
            fh.write(f'{region}_fuel={cfg["preset"]}  enabled={flag}\n')
            fh.write(f'{region}_density={cfg["density"]:.3e}\n')
            for sp in ['h','li7','li6','b11','bheavy']:
                fh.write(f'{region}_n_{sp}={cfg["n_"+sp]:.3e}\n')
        fh.write('\n[reactions]\n')
        for rxn in ALL_RXNS:
            fh.write(f'{rxn["key"]}_active={rxn["active"]}\n')
            fh.write(f'{rxn["key"]}_rate_per_ff={rxn["rate_per_ff"]:.6e}\n')
        fh.write('\n[physics]\n')
        fh.write(f'b_seed_t={B_SEED_T}\neta_si={eta_SI:.6e}\n')
        fh.write(f'no_bfield={args.no_bfield}\n')
        fh.write(f'plasma_beta={plasma_beta:.4e}\n')
        fh.write(f'current_support_enabled={not args.no_current_support}\n')
        fh.write(f'time_step_s={time_step_s:.16e}\ntotal_time_s={total_time_s:.16e}\n')
        fh.write(f'n_steps={n_steps}\n')
        fh.write(f'\n[diagnostics]\n')
        fh.write(f'diag_profile={DIAG_PROFILE}\n')
        fh.write(f'early_diag_period={EARLY_DIAG_PERIOD}\n')
        fh.write(f'early_diag_steps={EARLY_DIAG_STEPS}\n')
        fh.write(f'late_diag_period={LATE_DIAG_PERIOD}\n')


# ============================================================================
# RUN
# ============================================================================
simulation.time_step_size = time_step_s
simulation.max_steps      = n_steps

if rank == 0:
    active_sp = ['proton'] + (['boron11'] if NEED_B11 else []) + \
                (['li7'] if NEED_LI7 else []) + (['li6'] if NEED_LI6 else [])
    print()
    print(f'  Species active: {", ".join(active_sp)}')
    print(f'  Starting {n_steps:,} steps -> {OUTDIR}/')
    print()

# Probe: run 1 step, then check whether the particle container is reachable.
# If this fails, every subsequent fusion-diag row will be NaN. Better to know
# now than to discover it after a multi-hour run.
simulation.step(1)
if rank == 0:
    ke_test, w_test, status_test = get_proton_ke_kev()
    print('  ' + '=' * 68)
    print(f'  [particle access probe] status: {status_test}')
    if ke_test is not None and w_test is not None:
        print(f'  [particle access probe] OK -- {ke_test.size} protons reachable, '
              f'mean KE = {ke_test.mean():.3f} keV')
    else:
        print('  [particle access probe] FAILED. Fusion CSV will be NaN.')
        print('  [particle access probe] Check the status string above for the')
        print('  [particle access probe] available_methods list and update')
        print('  [particle access probe] _call_first() name lists in get_proton_ke_kev().')
    print('  ' + '=' * 68)

simulation.step(n_steps - 1)

if rank == 0:
    print()
    print('=' * 70)
    print('  SIMULATION COMPLETE — v12.14')
    print('=' * 70)
    geom_done = ('single-spot baseline (TNSA-equivalent)' if SINGLE_SPOT_MODE
                 else f'{N_SPOTS}-spot ring')
    print(f'  Geometry:   {geom_done}')
    print(f'  Base fuel:  {args.base_fuel}')
    print(f'  Rod fuel:   {args.rod_fuel if FUEL_ROD else "none"}')
    ring_modes = ([f'inner/{args.ring_fuel}'] if FUEL_RING_INNER else []) + \
                 ([f'outer/{args.ring_fuel}'] if FUEL_RING_OUTER else []) + \
                 ([f'full/{args.ring_fuel}']  if FUEL_RING_FULL  else [])
    print(f'  Ring fuel:  {", ".join(ring_modes) if ring_modes else "none"}')
    print(f'  Output:     {OUTDIR}/')
    print(f'  Fusion CSV: {FUSION_DIAG_FILE}')
    print('  CAVEAT: integrate P*dt for net energy assessment.')
    print('=' * 70)

if rank == 0 and log_fh is not None:
    try:
        sys.stdout = ORIG_STDOUT; sys.stderr = ORIG_STDERR
        log_fh.flush(); log_fh.close()
    except: pass
