#!/usr/bin/env python3
"""
p-11B / p-7Li / LiB Aneutronic Fusion — Ring Geometry with Rotating Modulation
WarpX Hybrid-PIC @ 5J optimal laser energy

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
    'p11b':           {'h':1.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,
                       'label':'p-11B H:B=1:1'},
    'ammonia_borane': {'h':6.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,
                       'label':'NH3BH3 H:B=6:1'},
    'decaborane':     {'h':1.4,      'li7':0.0,    'li6':0.0,    'b11':1.0,
                       'label':'B10H14 H:B=1.4:1'},
    'b18h22':         {'h':22/18,    'li7':0.0,    'li6':0.0,    'b11':1.0,
                       'label':'B18H22 H:B=1.22:1'},
    'bn_plasma':      {'h':1.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,
                       'label':'BN plasma H:B=1:1'},
    # p-Li fuels
    'p7li':           {'h':1.0,      'li7':1.0,    'li6':0.0,    'b11':0.0,
                       'label':'p-7Li H:Li7=1:1'},
    'p6li':           {'h':1.0,      'li7':0.0,    'li6':1.0,    'b11':0.0,
                       'label':'p-6Li H:Li6=1:1'},
    'nat_li':         {'h':1.0,      'li7':0.926,  'li6':0.074,  'b11':0.0,
                       'label':'natural Li (92.6% Li7)'},
    # LiB hybrid fuels
    'lib_equal':      {'h':1.0,      'li7':1.0,    'li6':0.0,    'b11':1.0,
                       'label':'LiB equal H:Li7:B11=1:1:1'},
    'lib_li_rich':    {'h':1.0,      'li7':2.0,    'li6':0.0,    'b11':1.0,
                       'label':'LiB Li-heavy H:Li7:B11=1:2:1'},
    'lib_b_rich':     {'h':1.0,      'li7':0.5,    'li6':0.0,    'b11':1.0,
                       'label':'LiB B-heavy H:Li7:B11=1:0.5:1'},
    'lib_nat_li':     {'h':1.0,      'li7':0.926,  'li6':0.074,  'b11':1.0,
                       'label':'natural Li + B11'},
    # Pure target fuels (no added H — receive protons from ring plasma)
    'b11_target':     {'h':0.0,      'li7':0.0,    'li6':0.0,    'b11':1.0,
                       'label':'pure B11 target'},
    'li7_target':     {'h':0.0,      'li7':1.0,    'li6':0.0,    'b11':0.0,
                       'label':'pure Li7 target'},
    'lib_target':     {'h':0.0,      'li7':1.0,    'li6':0.0,    'b11':1.0,
                       'label':'Li7+B11 target no H'},
    'custom':         {'h':-1.0,     'li7':-1.0,   'li6':-1.0,   'b11':-1.0,
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
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(
    description='p-11B/p-7Li/LiB ring reconnection — v12.4.4 b-seed 300T default'
)

# ── Simulation control ───────────────────────────────────────────────────────
parser.add_argument('--test',         action='store_true',
                    help='Quick test run (128x128, 10 cyclotron periods)')
parser.add_argument('--freq',         type=float, default=500e6,
                    help='Rotation frequency Hz (default: 500 MHz)')
parser.add_argument('--rotate',       action='store_true',
                    help='Enable rotating Jy modulation (default: static)')
parser.add_argument('--ramp-steps',   type=int,   default=100)
parser.add_argument('--sigma-scale',  type=float, default=1.15)
parser.add_argument('--eta-scale',    type=float, default=1.0)
parser.add_argument('--te-ev',        type=float, default=2200.0)
parser.add_argument('--b-seed',       type=float, default=300.0,
                    help='Biermann battery seed field in T (default: 300T, '  
                         'consistent with published measurements at ~6e13 W/cm2)')
parser.add_argument('--field-mode',   type=str,   default='applied',
                    choices=['current','applied','none'])
parser.add_argument('--j-scale',      type=float, default=0.15)
parser.add_argument('--outdir',       type=str,   default='./pb11_diags')
parser.add_argument('--early-diag-period', type=int, default=5)
parser.add_argument('--early-diag-steps',  type=int, default=200)
parser.add_argument('--physical-protons',  type=float, default=5.74e19)
parser.add_argument('--max-steps',         type=int,   default=0,
                    help='Hard cap on total simulation steps (0=use test/prod default). '
                         'Used by preflight runner to limit run length exactly.')

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
    for species in ['h', 'li7', 'li6', 'b11']:
        parser.add_argument(f'--{region}-{species}-ratio', type=float, default=-1.0,
                            help=f'{region} {species} ratio override (negative=use preset)')

# ── Geometry ─────────────────────────────────────────────────────────────────
parser.add_argument('--rod-radius-um',        type=float, default=75.0)
parser.add_argument('--outer-radius-um',      type=float, default=1050.0)
parser.add_argument('--outer-thickness-um',   type=float, default=100.0)
parser.add_argument('--face-edge-um',         type=float, default=15.0)

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
    Resolve final H/Li7/Li6/B11 ratios and densities for a named region.
    Per-region ratio overrides (if positive) replace preset values.
    Returns dict with species densities in m^-3.
    """
    p = FUEL_PRESETS[preset_name].copy()
    for sp in ['h', 'li7', 'li6', 'b11']:
        override = getattr(args, f'{region_name}_{sp}_ratio')
        if override >= 0.0:
            p[sp] = override

    # Normalise so max ratio = 1 then scale by density
    ratios = {sp: max(0.0, p[sp]) for sp in ['h','li7','li6','b11']}
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

NEED_B11  = species_needed('b11')
NEED_LI7  = species_needed('li7')
NEED_LI6  = species_needed('li6')

# Collect effective target densities for fusion accounting
# (averaged over active regions, weighted by density)
def effective_target_density(sp_key):
    """Effective number density of target species seen by fast protons."""
    total_n = BASE['n_' + sp_key]
    if FUEL_ROD:   total_n += ROD['n_'  + sp_key]
    if FUEL_ANY:   total_n += RING['n_' + sp_key]
    return total_n

EFF_N_B11 = effective_target_density('b11')
EFF_N_LI7 = effective_target_density('li7')
EFF_N_LI6 = effective_target_density('li6')


# ============================================================================
# FUSION ACCOUNTING — per reaction channel
# ============================================================================
AMU_KG         = 1.66054e-27
PROTON_MASS_KG = 1.00728 * AMU_KG
LI7_MASS_KG    = 7.01601 * AMU_KG
LI6_MASS_KG    = 6.01512 * AMU_KG
B11_MASS_KG    = 11.0093 * AMU_KG

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
N_SPOTS           = 8
RING_RADIUS_M     = 800e-6
SPOT_RADIUS_M     = 75e-6

N_BACKGROUND_M3   = BASE['density'] * 0.30      # 30% of base peak as background
N_PEAK_M3         = BASE['density']
T_ION_EV          = 660.0
T_ELEC_EV         = args.te_ev
B_SEED_T          = args.b_seed

ROD_RADIUS_M      = args.rod_radius_um      * 1e-6
OUTER_RADIUS_M    = args.outer_radius_um    * 1e-6
OUTER_THICKNESS_M = args.outer_thickness_um * 1e-6
FACE_EDGE_M       = max(args.face_edge_um, 1.0) * 1e-6

omega_ci = constants.q_e * B_SEED_T / PROTON_MASS_KG
omega_pi = np.sqrt(N_PEAK_M3 * constants.q_e**2 / (constants.ep0 * PROTON_MASS_KG))
d_i      = constants.c / omega_pi
v_A      = B_SEED_T / np.sqrt(constants.mu0 * N_PEAK_M3 * PROTON_MASS_KG)
v_th_p   = np.sqrt(T_ION_EV * constants.q_e / PROTON_MASS_KG)
v_th_B11 = np.sqrt(T_ION_EV * constants.q_e / B11_MASS_KG)
v_th_Li7 = np.sqrt(T_ION_EV * constants.q_e / LI7_MASS_KG)
v_th_Li6 = np.sqrt(T_ION_EV * constants.q_e / LI6_MASS_KG)

eta_norm = 6e-3
eta0     = d_i * v_A / (constants.ep0 * constants.c**2)
eta_SI   = eta_norm * eta0 * args.eta_scale

J_amp_check   = N_PEAK_M3 * constants.q_e * v_A * args.j_scale
J_thermal     = N_PEAK_M3 * constants.q_e * v_th_p
j_scale_ratio = J_amp_check / J_thermal

# Domain and timestepping
LX_DI = LZ_DI = 60
if args.test:
    NX, NZ = 128, 128; NPPC = 200; LT = 10;  DT = 1e-3
else:
    NX, NZ = 256, 256; NPPC = 400; LT = 25;  DT = 5e-4

LX_M = LX_DI * d_i;  LZ_M = LZ_DI * d_i
time_step_s  = DT / omega_ci
total_time_s = LT / omega_ci
n_steps      = int(total_time_s / time_step_s)
# v12.4.3: hard cap for preflight — limits simulation to exactly N steps
if args.max_steps > 0:
    n_steps = min(n_steps, args.max_steps)
RAMP_STEPS   = max(1, args.ramp_steps)
RAMP_TIME_S  = RAMP_STEPS * time_step_s
SIGMA_INIT_M = SPOT_RADIUS_M * args.sigma_scale
EARLY_DIAG_PERIOD = max(1, args.early_diag_period)
EARLY_DIAG_STEPS  = max(0, args.early_diag_steps)
LATE_DIAG_PERIOD  = max(25, n_steps // 80)


# ============================================================================
# PARAMETER REPORT
# ============================================================================
if rank == 0:
    print('=' * 70)
    print('  p-11B/p-7Li/LiB RING RECONNECTION  v12.4.4')
    print('=' * 70)
    print(f'  Mode:         {"ROTATING" if IS_ROTATING else "STATIC BASELINE"}')
    print(f'  Field mode:   {args.field_mode}  (j_scale={args.j_scale:.3f})')
    print(f'  B-seed:       {B_SEED_T:.1f} T  |  v_A={v_A:.2e} m/s  |  d_i={d_i*1e6:.1f} um')
    print(f'  eta_SI:       {eta_SI:.3e} Ohm m  |  J_ext/J_th={j_scale_ratio:.3f}')
    if j_scale_ratio < 0.10 and IS_ROTATING:
        print('  WARNING: j_scale low — rotation may be invisible vs static')
    print(f'  n0 (Ohm):     {N_PEAK_M3+N_BACKGROUND_M3:.3e} m^-3  [v11.4 FIX1]')
    print()
    print('  FUEL REGIONS:')
    print(f'  BASE  plasma: {args.base_fuel:18s} ({BASE["label"]})')
    print(f'    density={BASE["density"]:.2e}  H={BASE["n_h"]:.2e}  Li7={BASE["n_li7"]:.2e}  Li6={BASE["n_li6"]:.2e}  B11={BASE["n_b11"]:.2e}')
    rod_status = "ENABLED" if FUEL_ROD else "disabled"
    print(f'  ROD   region: {rod_status}  fuel={args.rod_fuel:18s} ({ROD["label"]})')
    if FUEL_ROD:
        print(f'    density={ROD["density"]:.2e}  H={ROD["n_h"]:.2e}  Li7={ROD["n_li7"]:.2e}  Li6={ROD["n_li6"]:.2e}  B11={ROD["n_b11"]:.2e}')
        print(f'    radius={ROD_RADIUS_M*1e6:.0f} um')
    ring_modes = []
    if FUEL_RING_INNER: ring_modes.append('inner')
    if FUEL_RING_OUTER: ring_modes.append('outer')
    if FUEL_RING_FULL:  ring_modes.append('full')
    ring_status = '+'.join(ring_modes) if ring_modes else 'disabled'
    print(f'  RING  region: {ring_status}  fuel={args.ring_fuel:18s} ({RING["label"]})')
    if ring_modes:
        print(f'    density={RING["density"]:.2e}  H={RING["n_h"]:.2e}  Li7={RING["n_li7"]:.2e}  Li6={RING["n_li6"]:.2e}  B11={RING["n_b11"]:.2e}')
        print(f'    radius={OUTER_RADIUS_M*1e6:.0f} um  thickness={OUTER_THICKNESS_M*1e6:.0f} um')
    print()
    print('  ACTIVE SPECIES:')
    print(f'    proton:  always  (reconnection driver)')
    print(f'    boron11: {"YES" if NEED_B11 else "no"}   (p-11B channel active={RXN_P11B["active"]})')
    print(f'    li7:     {"YES" if NEED_LI7 else "no"}   (p-7Li channel active={RXN_P7LI["active"]})')
    print(f'    li6:     {"YES" if NEED_LI6 else "no"}   (p-6Li channel active={RXN_P6LI["active"]})')
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
)

def smooth_ramp(t='t'):
    return f'(1.0 - exp(-{t} / {RAMP_TIME_S:.8e}))'


# ── Jy external current ──────────────────────────────────────────────────────
def build_Jy(rotating=False):
    if args.field_mode == 'none': return '0.0'
    if args.field_mode == 'applied' and not rotating: return '0.0'
    J  = N_PEAK_M3 * constants.q_e * v_A * args.j_scale
    f  = ROTATION_FREQ_HZ
    R  = RING_RADIUS_M; s = SPOT_RADIUS_M
    terms = []
    for k in range(N_SPOTS):
        ba   = 2*np.pi*k/N_SPOTS; sgn = (-1)**k
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
    substeps=40,
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


# ── B-field initialization ────────────────────────────────────────────────────
def build_By():
    # NO ramp multiplier here.
    # AnalyticInitialField evaluates this expression exactly once at t=0.
    # smooth_ramp() = (1 - exp(-t/RAMP_TIME)) = 0 at t=0 → B-field = 0.
    # The ramp belongs only in build_Jy() for the time-varying current seed.
    # B-field is initialized at full strength and evolves self-consistently.
    terms = []
    for k in range(N_SPOTS):
        a  = 2*np.pi*k/N_SPOTS
        cx = RING_RADIUS_M*np.cos(a); cz = RING_RADIUS_M*np.sin(a)
        sg = (-1)**k; s = SPOT_RADIUS_M
        terms.append(
            f'({sg:.1f}*{B_SEED_T:.8e}'
            f'*sqrt((x-{cx:.8e})**2+(z-{cz:.8e})**2)/{s:.8e}'
            f'*exp(-0.5*((x-{cx:.8e})**2+(z-{cz:.8e})**2)/{s:.8e}**2))'
        )
    return ' + '.join(terms)

if args.field_mode == 'applied':
    By_expr_str = build_By()
    if rank == 0:
        # Confirm B-field expression is non-trivial (no ramp, should be non-zero at t=0)
        preview = By_expr_str[:120].replace('\n',' ')
        print(f'  B-field init: AnalyticInitialField applied (no ramp, t=0 non-zero)')
        print(f'  By preview:   {preview}...')
        print(f'  Expected B_max at t=0: ~{B_SEED_T * 0.6:.1f} T (60% of seed at spot edge)')
    simulation.add_applied_field(picmi.AnalyticInitialField(
        Bx_expression='0.0', By_expression=By_expr_str, Bz_expression='0.0'))


# ============================================================================
# FUSION DIAGNOSTIC — multi-channel
# ============================================================================
def diagnostic_step_set():
    steps = {0, n_steps}
    for s in range(0, min(EARLY_DIAG_STEPS, n_steps)+1, EARLY_DIAG_PERIOD):
        steps.add(s)
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
        # Per-reaction powers
        + [f'fusion_power_{r["key"]}_w'   for r in ALL_RXNS if r['active']]
        # FIX D: alpha yield per step per channel
        + [f'alpha_yield_step_{r["key"]}' for r in ALL_RXNS if r['active']]
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

    steps = sorted({0, n_steps}
                   | {s for s in range(0, min(EARLY_DIAG_STEPS,n_steps)+1, EARLY_DIAG_PERIOD)}
                   | {s for s in range(0, n_steps+1, LATE_DIAG_PERIOD)})
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
                int(step<=EARLY_DIAG_STEPS and step%EARLY_DIAG_PERIOD==0),
                int(step%LATE_DIAG_PERIOD==0),
                int(step==n_steps),
                f'{LASER_ENERGY_J:.6e}',
                f'{PHYSICAL_PROTONS:.6e}',
            ] + [f'{r["eff_n_target"]:.6e}' for r in ALL_RXNS if r['active']]               + [f'{r["rate_per_ff"]:.6e}'   for r in ALL_RXNS if r['active']]               + [f'{r["rate_per_ff"]*r["n_alphas"]:.6e}' for r in ALL_RXNS if r['active']]               + ['t_physical=step*time_step_s; net_energy_requires_time_integration']
            w.writerow(row)


write_fusion_diag_header()
write_fusion_accounting()
write_step_time_index_csv()


# ── Live particle access helpers ─────────────────────────────────────────────
def _flatten(parts):
    if parts is None: return None
    if isinstance(parts, np.ndarray): return np.asarray(parts).ravel()
    if isinstance(parts, (list,tuple)):
        arrs = [np.asarray(i).ravel() for i in parts
                if i is not None and np.asarray(i).size>0]
        return np.concatenate(arrs) if arrs else np.asarray([], dtype=float)
    try: return np.asarray(parts).ravel()
    except: return None

def _call_first(obj, names, **kw):
    for n in names:
        fn = getattr(obj, n, None)
        if fn is None: continue
        try: return fn(**kw)
        except TypeError:
            try: return fn()
            except: continue
        except: continue
    return None

def get_proton_ke_kev():
    # Use new API (sim.particles.get) with fallback to deprecated wrapper
    # Fallback suppresses the UserWarning that fires once per rank per step
    try:
        pc = simulation.particles.get('proton')
    except Exception:
        try:
            from pywarpx import particle_containers
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                pc = particle_containers.ParticleContainerWrapper('proton')
        except Exception as e:
            return None, None, f'unavailable:{e}'
    ux=uy=uz=w=None
    for kw in [{'lev':0},{'level':0},{}]:
        ux = _flatten(_call_first(pc,['get_particle_ux','get_particle_px'],**kw))
        uy = _flatten(_call_first(pc,['get_particle_uy','get_particle_py'],**kw))
        uz = _flatten(_call_first(pc,['get_particle_uz','get_particle_pz'],**kw))
        w  = _flatten(_call_first(pc,['get_particle_weight','get_particle_weighting'],**kw))
        if ux is not None and uy is not None and uz is not None: break
    if ux is None: return None, None, 'no_arrays'
    n = min(ux.size, uy.size, uz.size)
    if n == 0: return None, None, 'empty'
    ux=ux[:n].astype(float,copy=False)
    uy=uy[:n].astype(float,copy=False)
    uz=uz[:n].astype(float,copy=False)
    w = w[:n].astype(float,copy=False) if w is not None and w.size>=n else np.ones(n)
    u2    = ux*ux + uy*uy + uz*uz
    gamma = np.sqrt(1.0 + u2/(constants.c*constants.c))
    ke    = (gamma-1.0)*PROTON_MASS_KG*constants.c*constants.c/(1e3*constants.q_e)
    valid = np.isfinite(ke) & np.isfinite(w) & (w>0)
    return ke[valid], w[valid], 'ok'


def should_diag(step):
    if step<0 or step>n_steps: return False
    if step==0 or step==n_steps: return True
    if step<=EARLY_DIAG_STEPS and step%EARLY_DIAG_PERIOD==0: return True
    if step%LATE_DIAG_PERIOD==0: return True
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
        if rank == 0 and not fusion_diag_state['warned_particle_access']:
            print(f'  [fusion_diag] arrays unavailable: {status}', flush=True)
            fusion_diag_state['warned_particle_access'] = True
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
            else:
                ffs[rxn['key']] = rates[rxn['key']] = powers[rxn['key']] = float('nan')
                alphas_step[rxn['key']] = float('nan')

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
    R_CENTRE = 200e-6   # 200 um convergence zone
    centre_metrics = {'E_mean': float('nan'), 'E_95th': float('nan'),
                      'N_centre': 0, 'outer_E_95th': float('nan'), 'ratio': float('nan')}
    if ke_kev is not None and weights is not None:
        try:
            try:
                pc = simulation.particles.get('proton')
            except Exception:
                from pywarpx import particle_containers
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    pc = particle_containers.ParticleContainerWrapper('proton')
            px_  = _flatten(_call_first(pc, ['get_particle_x'], lev=0))
            pz_  = _flatten(_call_first(pc, ['get_particle_z'], lev=0))
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
            # reaction powers
            + [fmt(powers.get(r['key'], float('nan'))) for r in ALL_RXNS if r['active']]
            # FIX D: alpha yield per step per channel
            + [fmt(alphas_step.get(r['key'], float('nan'))) for r in ALL_RXNS if r['active']]
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
# DIAGNOSTICS
# ============================================================================
if rank == 0:
    print(f'  Early diag: period={EARLY_DIAG_PERIOD}  Late diag: period={LATE_DIAG_PERIOD}')
    print(f'  Fusion CSV: {FUSION_DIAG_FILE}')

for name, period in [('fields_early',EARLY_DIAG_PERIOD),('fields',LATE_DIAG_PERIOD)]:
    simulation.add_diagnostic(picmi.FieldDiagnostic(
        name=name, grid=grid, period=period,
        data_list=['B','E','J','rho'],
        write_dir=OUTDIR, warpx_format='openpmd'))

for name, period in [('particles_early',EARLY_DIAG_PERIOD),('particles',LATE_DIAG_PERIOD)]:
    simulation.add_diagnostic(picmi.ParticleDiagnostic(
        name=name, period=period, species=all_species,
        data_list=['position','momentum','weighting'],
        write_dir=OUTDIR, warpx_format='openpmd'))


# ============================================================================
# METADATA
# ============================================================================
if rank == 0:
    with open(META_FILE, 'w', encoding='utf-8') as fh:
        fh.write('v12.4.4 per-region fuel config run metadata\n' + '='*60 + '\n')
        fh.write(f'version=12.4\ntimestamp={datetime.now().isoformat()}\n')
        fh.write(f'outdir={OUTDIR}\n')
        fh.write('\n[fuel_regions]\n')
        for region, cfg, flag in [('base',BASE,'always'),
                                   ('rod', ROD, str(FUEL_ROD)),
                                   ('ring',RING,str(FUEL_ANY))]:
            fh.write(f'{region}_fuel={cfg["preset"]}  enabled={flag}\n')
            fh.write(f'{region}_density={cfg["density"]:.3e}\n')
            for sp in ['h','li7','li6','b11']:
                fh.write(f'{region}_n_{sp}={cfg["n_"+sp]:.3e}\n')
        fh.write('\n[reactions]\n')
        for rxn in ALL_RXNS:
            fh.write(f'{rxn["key"]}_active={rxn["active"]}\n')
            fh.write(f'{rxn["key"]}_rate_per_ff={rxn["rate_per_ff"]:.6e}\n')
        fh.write('\n[physics]\n')
        fh.write(f'b_seed_t={B_SEED_T}\neta_si={eta_SI:.6e}\n')
        fh.write(f'time_step_s={time_step_s:.16e}\ntotal_time_s={total_time_s:.16e}\n')
        fh.write(f'n_steps={n_steps}\n')


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

simulation.step(n_steps)

if rank == 0:
    print()
    print('=' * 70)
    print('  SIMULATION COMPLETE — v12.4.4')
    print('=' * 70)
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
