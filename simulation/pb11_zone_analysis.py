#!/usr/bin/env python3
"""
pb11_zone_analysis.py — Geometry-aware re-analysis for ring-reconnection runs.

Replaces the hardcoded R_CENTRE = 200 µm in pb11_text_analysis.py and the
inline simulation diagnostic. Reads ring_radius_m and n_spots from run_meta.txt
and defines four physically meaningful zones:

  CORE      : r < 0.5 * R_ring         — true vacuum core, expected reconnection
                                         convergence target
  INNER     : 0.5 * R_ring <= r < 0.85 * R_ring
                                       — region between core and X-lines, where
                                         inflow from X-lines should be visible
  X-LINE    : 0.85 * R_ring <= r <= 0.95 * R_ring
                                       — annulus around the X-line radius
                                         (geometric: R_ring * cos(pi/N))
  SPOT      : r > 0.95 * R_ring        — spot region, hottest by initialization

For each particle dump iteration, prints particle count, mean energy, 95th
percentile, and max energy in each zone. Also reports core/spot ratio (the
right replacement for the centre/outer ratio in the inline diagnostic).

Usage:
    python pb11_zone_analysis.py --dir runs/v1213_1_p11b_b300_1000

Optional flags:
    --core-frac     fraction of R_ring defining core boundary (default 0.5)
    --xline-frac    half-width of X-line annulus as fraction of R_ring
                    (default 0.05; X-line annulus = [(cos(pi/N) - frac), (cos(pi/N) + frac)] * R_ring)
    --species       species name (default proton)
    --max-snapshots cap on number of snapshots to process (default all)
    --thresh-kev    fast-ion threshold for fast_fraction (default 500)
"""

import os, sys, argparse, configparser
import numpy as np

PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir',     required=True, help='Run output directory')
parser.add_argument('--report',  default=None,
                    help='Save report to this path. If "auto" or unspecified, '
                         'defaults to <dir>/zone_report.txt. Use "" or "stdout" '
                         'to disable file output.')
parser.add_argument('--core-frac',     type=float, default=0.5,
                    help='Core boundary as fraction of ring_radius (default 0.5)')
parser.add_argument('--xline-frac',    type=float, default=0.05,
                    help='Half-width of X-line annulus (default 0.05)')
parser.add_argument('--species',       default='proton')
parser.add_argument('--max-snapshots', type=int, default=0,
                    help='Cap on snapshots (0 = all)')
parser.add_argument('--thresh-kev',    type=float, default=500.0)
args = parser.parse_args()

run_dir = args.dir.rstrip('/')

# Default report path: <run_dir>/zone_report.txt unless user said otherwise
if args.report is None or args.report == 'auto':
    report_path = os.path.join(run_dir, 'zone_report.txt')
elif args.report in ('', 'stdout', 'none', '-'):
    report_path = None
else:
    report_path = args.report

# Tee stdout to report file if one is set
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()

if report_path:
    # Make sure the parent dir exists
    parent = os.path.dirname(report_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    _report_fh = open(report_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _report_fh)

# ─── Load geometry from run_meta.txt ────────────────────────────────────────
meta_path = os.path.join(run_dir, 'run_meta.txt')
if not os.path.exists(meta_path):
    print(f'ERROR: run_meta.txt not found at {meta_path}', file=sys.stderr)
    sys.exit(1)

# run_meta.txt is INI-like but the leading lines aren't in a section. Skip them.
meta_lines = open(meta_path).read().splitlines()
section_start = None
for i, line in enumerate(meta_lines):
    if line.strip().startswith('['):
        section_start = i
        break
if section_start is None:
    print(f'ERROR: no [section] markers in {meta_path}', file=sys.stderr)
    sys.exit(1)

cp = configparser.ConfigParser()
cp.read_string('\n'.join(meta_lines[section_start:]))

try:
    N_SPOTS       = int(cp['geometry']['n_spots'])
    RING_RADIUS_M = float(cp['geometry']['ring_radius_m'])
    SPOT_RADIUS_M = float(cp['geometry']['spot_radius_m'])
except KeyError as e:
    print(f'ERROR: missing geometry field {e} in run_meta.txt', file=sys.stderr)
    sys.exit(1)

# ─── Compute zone boundaries from geometry ─────────────────────────────────
if N_SPOTS == 1 or RING_RADIUS_M <= 0:
    # single-spot mode: zones don't apply the same way
    print('Single-spot mode detected — zone analysis not meaningful for R=0.')
    print('Use the standard analysis script instead.')
    sys.exit(0)

R_xline       = RING_RADIUS_M * np.cos(np.pi / N_SPOTS)  # geometric X-line radius
R_core        = args.core_frac * RING_RADIUS_M
R_xline_inner = (np.cos(np.pi / N_SPOTS) - args.xline_frac) * RING_RADIUS_M
R_xline_outer = (np.cos(np.pi / N_SPOTS) + args.xline_frac) * RING_RADIUS_M
R_spot_inner  = RING_RADIUS_M - 2 * SPOT_RADIUS_M

# ─── Load particle data via openpmd_viewer ────────────────────────────────
try:
    import openpmd_viewer as ov
except ImportError:
    print('Installing openpmd-viewer...')
    os.system('pip install openpmd-viewer --break-system-packages')
    import openpmd_viewer as ov

# Find particle directory: try test/production layouts
particle_dir_candidates = [
    os.path.join(run_dir, 'particles_early'),
    os.path.join(run_dir, 'particles'),
    os.path.join(run_dir, 'diags', 'particles_early'),
    os.path.join(run_dir, 'diags', 'particles'),
    os.path.join(run_dir, 'debug_particles'),
]
particle_dir = next((p for p in particle_dir_candidates if os.path.isdir(p)), None)
if particle_dir is None:
    print(f'ERROR: no particle dumps found in any of: {particle_dir_candidates}',
          file=sys.stderr)
    sys.exit(1)

ts = ov.OpenPMDTimeSeries(particle_dir)

# ─── Banner ─────────────────────────────────────────────────────────────────
print('=' * 92)
print(f'  ZONE-BASED RE-ANALYSIS — geometry from {meta_path}')
print('=' * 92)
print(f'  Run directory:    {run_dir}')
print(f'  Particle dumps:   {particle_dir}')
print(f'  Snapshots avail:  {len(ts.iterations)}')
print(f'  Time range:       {ts.t[0]*1e12:.2f} to {ts.t[-1]*1e12:.2f} ps')
print()
print(f'  Geometry (from run_meta):')
print(f'    n_spots:          {N_SPOTS}')
print(f'    ring_radius:      {RING_RADIUS_M*1e6:.0f} um')
print(f'    spot_radius:      {SPOT_RADIUS_M*1e6:.0f} um')
print(f'    X-line radius:    {R_xline*1e6:.0f} um  (R_ring * cos(pi/{N_SPOTS}))')
print()
print(f'  Zone boundaries:')
print(f'    CORE     : r <  {R_core*1e6:.0f} um             ({args.core_frac*100:.0f}% of ring radius)')
print(f'    INNER    : {R_core*1e6:.0f} <= r < {R_xline_inner*1e6:.0f} um')
print(f'    X-LINE   : {R_xline_inner*1e6:.0f} <= r <= {R_xline_outer*1e6:.0f} um  (annulus around X-lines)')
print(f'    SPOT     : r >= {R_spot_inner*1e6:.0f} um  (within 2σ of spots)')
print()
print(f'  Fast-ion threshold:  {args.thresh_kev:.0f} keV')
print(f'  Species:             {args.species}')
print('=' * 92)

# ─── Per-snapshot zone analysis ─────────────────────────────────────────────
mass_kg = (B11_MASS_KG if args.species == 'boron11' else PROTON_MASS_KG)

iterations = list(ts.iterations)
if args.max_snapshots > 0 and len(iterations) > args.max_snapshots:
    # subsample evenly
    step = max(1, len(iterations) // args.max_snapshots)
    iterations = iterations[::step][:args.max_snapshots]

print()
print(f'  {"t(ps)":>7} {"N_core":>8} {"N_inner":>8} {"N_xline":>8} {"N_spot":>8} | '
      f'{"E95_core":>10} {"E95_inner":>11} {"E95_xline":>11} {"E95_spot":>10} | '
      f'{"core/spot":>10} {"xline/spot":>11}')
print('  ' + '-' * 130)

results = []
for it in iterations:
    try:
        x, z, ux, uy, uz = ts.get_particle(
            ['x', 'z', 'ux', 'uy', 'uz'], species=args.species, iteration=it)
    except Exception as e:
        print(f'  iter {it}: particle read failed: {e}')
        continue

    # WarpX stores u = gamma*v / c (dimensionless momentum)
    u2 = ux*ux + uy*uy + uz*uz
    gamma = np.sqrt(1.0 + u2)
    ke_kev = (gamma - 1.0) * mass_kg * C * C / KEV_TO_J

    r = np.sqrt(x*x + z*z)
    in_core  = r < R_core
    in_inner = (r >= R_core) & (r < R_xline_inner)
    in_xline = (r >= R_xline_inner) & (r <= R_xline_outer)
    in_spot  = r >= R_spot_inner

    def zone_stat(mask, label):
        n = int(mask.sum())
        if n < 5:
            return n, float('nan'), float('nan'), float('nan'), float('nan')
        ek = ke_kev[mask]
        return (n,
                float(ek.mean()),
                float(np.percentile(ek, 95)),
                float(ek.max()),
                float((ek > args.thresh_kev).sum() / n))

    n_c, _,  e95_c, _, ff_c = zone_stat(in_core,  'core')
    n_i, _,  e95_i, _, ff_i = zone_stat(in_inner, 'inner')
    n_x, _,  e95_x, _, ff_x = zone_stat(in_xline, 'xline')
    n_s, _,  e95_s, _, ff_s = zone_stat(in_spot,  'spot')

    t_ps = ts.t[list(ts.iterations).index(it)] * 1e12

    cs = e95_c / e95_s if (e95_s > 0 and not np.isnan(e95_s)) else float('nan')
    xs = e95_x / e95_s if (e95_s > 0 and not np.isnan(e95_s)) else float('nan')

    print(f'  {t_ps:>7.2f} {n_c:>8d} {n_i:>8d} {n_x:>8d} {n_s:>8d} | '
          f'{e95_c:>10.1f} {e95_i:>11.1f} {e95_x:>11.1f} {e95_s:>10.1f} | '
          f'{cs:>10.4f} {xs:>11.4f}')

    results.append({
        't_ps': t_ps, 'iter': it,
        'core':  {'n': n_c, 'e95': e95_c, 'ff': ff_c},
        'inner': {'n': n_i, 'e95': e95_i, 'ff': ff_i},
        'xline': {'n': n_x, 'e95': e95_x, 'ff': ff_x},
        'spot':  {'n': n_s, 'e95': e95_s, 'ff': ff_s},
    })

# ─── Summary trends (last 30% of run vs first 30%) ─────────────────────────
if len(results) >= 6:
    # === v12.18 FIX: Add peak detection alongside window averaging ===
    # Bug fix: early/late window averaging hides transient peaks. We now detect
    # peaks in core E95 and report max value + time of max.
    t_arr = np.array([r['t_ps'] for r in results])
    e95_core_arr = np.array([r['core']['e95'] for r in results])
    e95_core_clean = np.where(np.isnan(e95_core_arr), 0.0, e95_core_arr)
    e95_spot_arr = np.array([r['spot']['e95'] for r in results])
    ratio_arr = np.where(e95_spot_arr > 0, e95_core_arr / np.maximum(e95_spot_arr, 1.0), 1.0)
    ratio_clean = np.where(np.isnan(ratio_arr), 1.0, ratio_arr)

    peak_idx_core = int(np.argmax(e95_core_clean))
    e95_core_max = float(e95_core_clean[peak_idx_core])
    t_core_max = float(t_arr[peak_idx_core])
    peak_idx_ratio = int(np.argmax(ratio_clean))
    ratio_max = float(ratio_clean[peak_idx_ratio])
    t_ratio_max = float(t_arr[peak_idx_ratio])

    n_third = max(2, len(results) // 3)
    early = results[:n_third]
    late  = results[-n_third:]

    def avg_e95(window, zone):
        vals = [w[zone]['e95'] for w in window if not np.isnan(w[zone]['e95'])]
        return np.mean(vals) if vals else float('nan')

    def avg_n(window, zone):
        return np.mean([w[zone]['n'] for w in window])

    print()
    print('=' * 92)
    print('  PEAK DETECTION (across full run)')
    print('=' * 92)
    print(f'  Maximum core E95:    {e95_core_max:>10,.0f} keV  at t = {t_core_max:.2f} ps')
    print(f'  Maximum core/spot:   {ratio_max:>10.3f}     at t = {t_ratio_max:.2f} ps')
    if e95_core_max > 500:
        print(f'  ✓ TRANSIENT FUSION-GRADE CORE: peak {e95_core_max:,.0f} keV at t={t_core_max:.1f} ps')
    if ratio_max > 1.5:
        print(f'  ✓ STRONG CORE INVERSION: max ratio {ratio_max:.3f} at t={t_ratio_max:.1f} ps')

    print()
    print('=' * 92)
    print('  TREND: early third vs late third')
    print('=' * 92)
    print(f'  WARNING: Window averaging may hide transient peaks.')
    print(f'  See PEAK DETECTION above for true maxima.')
    print()
    t_e_start, t_e_end = early[0]['t_ps'], early[-1]['t_ps']
    t_l_start, t_l_end = late[0]['t_ps'],  late[-1]['t_ps']
    print(f'  Early window: t = {t_e_start:.1f} to {t_e_end:.1f} ps')
    print(f'  Late  window: t = {t_l_start:.1f} to {t_l_end:.1f} ps')
    print()
    print(f'  {"Zone":<8} {"E95 early":>12} {"E95 late":>12} {"Δ":>12} {"N early":>10} {"N late":>10}')
    print('  ' + '-' * 70)
    for zone in ['core', 'inner', 'xline', 'spot']:
        e_e = avg_e95(early, zone)
        e_l = avg_e95(late, zone)
        d   = e_l - e_e
        n_e = avg_n(early, zone)
        n_l = avg_n(late, zone)
        print(f'  {zone:<8} {e_e:>12.1f} {e_l:>12.1f} {d:>+12.1f} {n_e:>10.0f} {n_l:>10.0f}')

    print()
    print('  KEY QUESTIONS (peak-aware):')
    e95_core_late = avg_e95(late, 'core')
    e95_core_early = avg_e95(early, 'core')
    e95_xline_late = avg_e95(late, 'xline')
    e95_xline_early = avg_e95(early, 'xline')
    e95_spot_late = avg_e95(late, 'spot')

    # === v12.18 FIX: Peak-aware core heating detection ===
    if e95_core_max > 500 and e95_core_max > 5 * max(e95_core_early, 1.0):
        print(f'  ✓ STRONG CORE HEATING (peak-detected):')
        print(f'    Peak {e95_core_max:,.0f} keV at t = {t_core_max:.1f} ps')
        print(f'    Early avg: {e95_core_early:.1f} keV; Late avg: {e95_core_late:.1f} keV')
    elif e95_core_max > 1.5 * max(e95_core_early, 1.0):
        print(f'  ~ MODERATE CORE HEATING (peak {e95_core_max:,.0f} keV at t={t_core_max:.1f} ps)')
        print(f'    Early avg: {e95_core_early:.1f}, Late avg: {e95_core_late:.1f} keV')
    else:
        print(f'  ✗ Core not heating significantly: peak {e95_core_max:,.0f} keV')

    if e95_xline_late > e95_xline_early * 1.5:
        print(f'  ✓ X-LINE region heating: {e95_xline_early:.1f} → {e95_xline_late:.1f} keV  '
              '(consistent with reconnection acceleration)')
    else:
        print(f'  ~ X-LINE evolution:    {e95_xline_early:.1f} → {e95_xline_late:.1f} keV')

    if e95_spot_late > 1000:
        print(f'  • SPOT E95 = {e95_spot_late:.1f} keV  (this is mostly the support-current heating)')

print()
print('=' * 92)
