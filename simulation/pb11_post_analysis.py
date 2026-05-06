#!/usr/bin/env python3
"""
pb11_post_analysis.py — Unified post-run analysis for ring-reconnection simulations.

Runs ALL the analyses on a completed simulation and produces a single
consolidated report. Replaces having to remember which separate script to
run for which kind of question.

Includes:
  1. Run summary from run_meta.txt + run.log
  2. Inline-diagnostic time series review (fusion_rate_power_by_iter.csv)
  3. Geometry-aware zone analysis (the breakthrough diagnostic — replaces
     the broken hardcoded R_CENTRE=200um)
  4. B-field evolution check (peak, mean, X-line region)
  5. Reconnection narrative detection (auto-flags the four-phase pattern)

Usage:
    python pb11_post_analysis.py --dir runs/<run_name>
    python pb11_post_analysis.py --dir runs/<run_name> --report-file out.txt

Optional flags:
    --core-frac      core boundary as fraction of ring_radius (default 0.5)
    --xline-frac     X-line annulus half-width as fraction (default 0.05)
    --species        species name (default proton)
    --skip-fields    skip B-field evolution check (faster on big runs)
    --max-snapshots  cap on snapshots to process (default all)
"""

import os, sys, argparse, configparser
import numpy as np

PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3

# ─── Argparse ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir',           required=True, help='Run output directory')
parser.add_argument('--report-file',   default=None,
                    help='Save report to this path. If unspecified or "auto", '
                         'defaults to <dir>/post_analysis_report.txt. Use "stdout" '
                         'or "" or "-" to disable file output.')
parser.add_argument('--core-frac',     type=float, default=0.5)
parser.add_argument('--xline-frac',    type=float, default=0.05)
parser.add_argument('--species',       default='proton')
parser.add_argument('--skip-fields',   action='store_true')
parser.add_argument('--max-snapshots', type=int, default=0,
                    help='Cap on snapshots (0 = all)')
parser.add_argument('--thresh-kev',    type=float, default=500.0)
args = parser.parse_args()

run_dir = args.dir.rstrip('/')

# ─── Tee output to file (default: inside run dir) ──────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()

# Resolve report path: default to <run_dir>/post_analysis_report.txt
if args.report_file is None or args.report_file == 'auto':
    report_path = os.path.join(run_dir, 'post_analysis_report.txt')
elif args.report_file in ('', 'stdout', 'none', '-'):
    report_path = None
else:
    report_path = args.report_file

if report_path:
    parent = os.path.dirname(report_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    rf = open(report_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, rf)

# ─── Section 0: Run summary ────────────────────────────────────────────────
print('=' * 92)
print(f'  POST-RUN ANALYSIS — {run_dir}')
print('=' * 92)

meta_path = os.path.join(run_dir, 'run_meta.txt')
if not os.path.exists(meta_path):
    print(f'ERROR: run_meta.txt not found at {meta_path}', file=sys.stderr)
    sys.exit(1)

# Parse run_meta.txt (skip pre-section header)
meta_lines = open(meta_path).read().splitlines()
section_start = next((i for i, line in enumerate(meta_lines)
                      if line.strip().startswith('[')), None)
if section_start is None:
    print(f'ERROR: no [section] markers in {meta_path}', file=sys.stderr)
    sys.exit(1)

cp = configparser.ConfigParser()
cp.read_string('\n'.join(meta_lines[section_start:]))

# Pre-section header lines
header = meta_lines[:section_start]
version_line = next((l for l in header if l.startswith('version=')), '')
timestamp_line = next((l for l in header if l.startswith('timestamp=')), '')

print()
print('  RUN METADATA')
print('  ' + '-' * 86)
print(f'  {version_line}')
print(f'  {timestamp_line}')
try:
    N_SPOTS       = int(cp['geometry']['n_spots'])
    RING_RADIUS_M = float(cp['geometry']['ring_radius_m'])
    SPOT_RADIUS_M = float(cp['geometry']['spot_radius_m'])
    print(f'  Geometry:        {N_SPOTS} spots at R={RING_RADIUS_M*1e6:.0f} um, '
          f'spot σ={SPOT_RADIUS_M*1e6:.0f} um')
except KeyError:
    print(f'  WARNING: incomplete geometry in run_meta.txt')
    N_SPOTS = 8; RING_RADIUS_M = 2400e-6; SPOT_RADIUS_M = 300e-6

if 'fuel_regions' in cp:
    base_fuel = cp['fuel_regions'].get('base_fuel', 'unknown').split()[0]
    base_density = float(cp['fuel_regions'].get('base_density', '0'))
    print(f'  Base fuel:       {base_fuel} at {base_density:.2e} m^-3')

if 'physics' in cp:
    b_seed = float(cp['physics'].get('b_seed_t', '0'))
    plasma_beta = float(cp['physics'].get('plasma_beta', '0'))
    current_support = cp['physics'].get('current_support_enabled', 'unknown')
    n_steps = int(cp['physics'].get('n_steps', '0'))
    dt_s = float(cp['physics'].get('time_step_s', '0'))
    print(f'  B-seed:          {b_seed:.1f} T')
    print(f'  Plasma beta:     {plasma_beta:.2f}')
    print(f'  Current support: {current_support}')
    print(f'  Steps × dt:      {n_steps} × {dt_s*1e15:.2f} fs = {n_steps*dt_s*1e12:.2f} ps')

# Check completion via run.log
log_path = os.path.join(run_dir, 'run.log')
completion_status = 'unknown'
if not os.path.exists(log_path):
    # Try alternate log locations
    for candidate in ['../' + os.path.basename(run_dir) + '.log',
                      run_dir + '.log']:
        if os.path.exists(candidate):
            log_path = candidate
            break
if os.path.exists(log_path):
    log_text = open(log_path).read()
    if 'SIMULATION COMPLETE' in log_text:
        completion_status = 'COMPLETE'
    elif 'SIGABRT' in log_text or 'Killed' in log_text or 'Segfault' in log_text:
        completion_status = 'CRASHED'
    else:
        completion_status = 'incomplete (no completion marker)'
print(f'  Completion:      {completion_status}')

# ─── Section 1: Inline diagnostic CSV review ───────────────────────────────
print()
print('=' * 92)
print('  INLINE DIAGNOSTIC TIME SERIES (fusion_rate_power_by_iter.csv)')
print('=' * 92)
print('  NOTE: this CSV uses the legacy R_CENTRE=200 um diagnostic which is')
print('        usually wrong for ring geometries. Use ZONE ANALYSIS below for')
print('        physically meaningful core/X-line/spot zones.')
print()

import csv as csv_module
fusion_csv = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')
if os.path.exists(fusion_csv):
    rows = list(csv_module.DictReader(open(fusion_csv)))
    print(f'  {len(rows)} diagnostic samples in CSV')
    if rows:
        first = rows[0]; last = rows[-1]
        t_first = float(first['time_ps'])
        t_last  = float(last['time_ps'])
        print(f'  Time range: {t_first:.2f} → {t_last:.2f} ps')
        print()
        print(f'  {"t(ps)":>8} {"fast_frac":>12} {"fusion_rate":>14} {"E_cum_J":>14} {"gain":>10}')
        print('  ' + '-' * 70)
        # Sample a few rows: first, every 25%, last
        n = len(rows)
        sample_idxs = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
        for i in sample_idxs:
            r = rows[i]
            t = float(r['time_ps'])
            ff = float(r.get('fast_fraction_gt_500kev_p11b', 'nan'))
            rate = float(r.get('fusion_rate_p11b_s^-1', 'nan'))
            cum = float(r.get('cumulative_fusion_energy_j', 'nan'))
            gain = float(r.get('gain_vs_laser_energy', 'nan'))
            print(f'  {t:>8.2f} {ff:>12.4f} {rate:>14.3e} {cum:>14.3e} {gain:>10.3e}')
        print()
        # Final cumulative numbers
        print(f'  FINAL  cumulative fusion energy: {float(last.get("cumulative_fusion_energy_j", 0)):.3e} J')
        print(f'  FINAL  gain vs 5 J laser:        {float(last.get("gain_vs_laser_energy", 0)):.3e}')
        try:
            cum_alphas_p11b = float(last.get('cum_alpha_yield_p11b', 0))
            print(f'  FINAL  cumulative alpha yield:   {cum_alphas_p11b:.3e}')
        except (KeyError, ValueError):
            pass
else:
    print(f'  WARNING: {fusion_csv} not found')

# ─── Section 2: Zone analysis ──────────────────────────────────────────────
print()
print('=' * 92)
print('  ZONE-BASED PARTICLE ANALYSIS (geometry-aware)')
print('=' * 92)

if N_SPOTS == 1 or RING_RADIUS_M <= 0:
    print('  Single-spot mode: zone analysis not applicable.')
    do_zones = False
else:
    do_zones = True
    R_xline       = RING_RADIUS_M * np.cos(np.pi / N_SPOTS)
    R_core        = args.core_frac * RING_RADIUS_M
    R_xline_inner = (np.cos(np.pi / N_SPOTS) - args.xline_frac) * RING_RADIUS_M
    R_xline_outer = (np.cos(np.pi / N_SPOTS) + args.xline_frac) * RING_RADIUS_M
    R_spot_inner  = RING_RADIUS_M - 2 * SPOT_RADIUS_M

    print()
    print(f'  Zone boundaries:')
    print(f'    CORE     : r <  {R_core*1e6:.0f} um             ({args.core_frac*100:.0f}% of ring radius)')
    print(f'    INNER    : {R_core*1e6:.0f} <= r < {R_xline_inner*1e6:.0f} um')
    print(f'    X-LINE   : {R_xline_inner*1e6:.0f} <= r <= {R_xline_outer*1e6:.0f} um  '
          f'(annulus around X-line at r={R_xline*1e6:.0f} um)')
    print(f'    SPOT     : r >= {R_spot_inner*1e6:.0f} um  (within 2σ of spots)')

    try:
        import openpmd_viewer as ov
    except ImportError:
        print('  Installing openpmd-viewer...')
        os.system('pip install openpmd-viewer --break-system-packages')
        import openpmd_viewer as ov

    particle_dir_candidates = [
        os.path.join(run_dir, 'particles_early'),
        os.path.join(run_dir, 'particles'),
        os.path.join(run_dir, 'diags', 'particles_early'),
        os.path.join(run_dir, 'diags', 'particles'),
        os.path.join(run_dir, 'debug_particles'),
    ]
    particle_dir = next((p for p in particle_dir_candidates if os.path.isdir(p)), None)
    if particle_dir is None:
        print(f'  ERROR: no particle dumps found in {particle_dir_candidates}')
        do_zones = False

    if do_zones:
        ts = ov.OpenPMDTimeSeries(particle_dir)
        mass_kg = (B11_MASS_KG if args.species == 'boron11' else PROTON_MASS_KG)
        iterations = list(ts.iterations)
        if args.max_snapshots > 0 and len(iterations) > args.max_snapshots:
            step = max(1, len(iterations) // args.max_snapshots)
            iterations = iterations[::step][:args.max_snapshots]

        print()
        print(f'  Processing {len(iterations)} snapshots from {particle_dir}...')
        print()
        print(f'  {"t(ps)":>7} {"N_core":>8} {"N_inner":>8} {"N_xline":>8} {"N_spot":>9} | '
              f'{"E95_core":>10} {"E95_inner":>11} {"E95_xline":>11} {"E95_spot":>10} | '
              f'{"core/spot":>10} {"xline/spot":>11}')
        print('  ' + '-' * 130)

        results = []
        for it in iterations:
            try:
                x, z, ux, uy, uz = ts.get_particle(
                    ['x', 'z', 'ux', 'uy', 'uz'], species=args.species, iteration=it)
            except Exception as e:
                continue

            u2 = ux*ux + uy*uy + uz*uz
            gamma = np.sqrt(1.0 + u2)
            ke_kev = (gamma - 1.0) * mass_kg * C * C / KEV_TO_J

            r = np.sqrt(x*x + z*z)
            in_core  = r < R_core
            in_inner = (r >= R_core) & (r < R_xline_inner)
            in_xline = (r >= R_xline_inner) & (r <= R_xline_outer)
            in_spot  = r >= R_spot_inner

            def zs(mask):
                n = int(mask.sum())
                if n < 5: return n, float('nan'), float('nan'), float('nan'), float('nan')
                ek = ke_kev[mask]
                return (n, float(ek.mean()), float(np.percentile(ek, 95)),
                        float(ek.max()), float((ek > args.thresh_kev).sum() / n))

            n_c, _, e95_c, _, ff_c = zs(in_core)
            n_i, _, e95_i, _, _    = zs(in_inner)
            n_x, _, e95_x, _, _    = zs(in_xline)
            n_s, _, e95_s, _, _    = zs(in_spot)

            t_ps = ts.t[list(ts.iterations).index(it)] * 1e12
            cs = e95_c / e95_s if (e95_s > 0 and not np.isnan(e95_s)) else float('nan')
            xs = e95_x / e95_s if (e95_s > 0 and not np.isnan(e95_s)) else float('nan')

            print(f'  {t_ps:>7.2f} {n_c:>8d} {n_i:>8d} {n_x:>8d} {n_s:>9d} | '
                  f'{e95_c:>10.1f} {e95_i:>11.1f} {e95_x:>11.1f} {e95_s:>10.1f} | '
                  f'{cs:>10.4f} {xs:>11.4f}')
            results.append({'t_ps': t_ps, 'iter': it, 'n_core': n_c, 'e95_core': e95_c,
                            'n_inner': n_i, 'e95_inner': e95_i,
                            'n_xline': n_x, 'e95_xline': e95_x,
                            'n_spot': n_s, 'e95_spot': e95_s, 'ff_core': ff_c})

        # ─── Section 3: Trend analysis + reconnection narrative detection ─────
        if len(results) >= 6:
            # === v12.18 FIX: Add peak detection alongside window averaging ===
            # Bug fix: early/late window averaging hides transient peaks (e.g., 537
            # keV transient at t=54 ps in HD ultrafine gets averaged out). We now
            # also detect peaks in core E95 and report max value + time of max.
            t_arr = np.array([r['t_ps'] for r in results])
            e95_core_arr = np.array([r['e95_core'] for r in results])
            e95_core_arr_clean = np.where(np.isnan(e95_core_arr), 0.0, e95_core_arr)
            ratio_arr = np.array([r['e95_core'] / max(r['e95_spot'], 1.0) if r['e95_spot'] > 0 else 1.0
                                  for r in results])

            peak_idx_core = int(np.argmax(e95_core_arr_clean))
            e95_core_max = float(e95_core_arr_clean[peak_idx_core])
            t_core_max = float(t_arr[peak_idx_core])
            peak_idx_ratio = int(np.argmax(ratio_arr))
            ratio_max = float(ratio_arr[peak_idx_ratio])
            t_ratio_max = float(t_arr[peak_idx_ratio])

            n_third = max(2, len(results) // 3)
            early = results[:n_third]
            late  = results[-n_third:]

            def avg(window, key):
                vals = [w[key] for w in window if not np.isnan(w[key])]
                return np.mean(vals) if vals else float('nan')

            print()
            print('=' * 92)
            print('  PEAK DETECTION (across full run)')
            print('=' * 92)
            print(f'  Maximum core E95:    {e95_core_max:>10,.0f} keV  at t = {t_core_max:.2f} ps')
            print(f'  Maximum core/spot:   {ratio_max:>10.3f}     at t = {t_ratio_max:.2f} ps')
            if e95_core_max > 500:
                print(f'  ✓ TRANSIENT FUSION-GRADE CORE: peak E95 = {e95_core_max:,.0f} keV exceeds')
                print(f'    p-11B threshold (500 keV) at t = {t_core_max:.1f} ps')
            if ratio_max > 1.5:
                print(f'  ✓ STRONG CORE INVERSION: max ratio = {ratio_max:.3f} at t = {t_ratio_max:.1f} ps')
            elif ratio_max > 1.10:
                print(f'  ✓ MODERATE CORE INVERSION: max ratio = {ratio_max:.3f} at t = {t_ratio_max:.1f} ps')

            print()
            print('=' * 92)
            print('  TREND ANALYSIS — early third vs late third')
            print('=' * 92)
            print(f'  WARNING: Window averaging may hide transient peaks.')
            print(f'  See PEAK DETECTION above for true maxima.')
            print()
            print(f'  Early window: t = {early[0]["t_ps"]:.1f} → {early[-1]["t_ps"]:.1f} ps')
            print(f'  Late  window: t = {late[0]["t_ps"]:.1f} → {late[-1]["t_ps"]:.1f} ps')
            print()
            print(f'  {"Zone":<8} {"E95 early":>14} {"E95 late":>14} {"Δ E95":>14} '
                  f'{"N early":>11} {"N late":>11}')
            print('  ' + '-' * 80)
            for zone, e95_key, n_key in [('core', 'e95_core', 'n_core'),
                                          ('inner', 'e95_inner', 'n_inner'),
                                          ('xline', 'e95_xline', 'n_xline'),
                                          ('spot', 'e95_spot', 'n_spot')]:
                e_e = avg(early, e95_key); e_l = avg(late, e95_key)
                n_e = avg(early, n_key);   n_l = avg(late, n_key)
                print(f'  {zone:<8} {e_e:>14.1f} {e_l:>14.1f} {e_l-e_e:>+14.1f} '
                      f'{n_e:>11.0f} {n_l:>11.0f}')

            # ─── Reconnection narrative auto-detection ─────────────────────────
            # === v12.18 FIX: Use peak data, not just window averages ===
            print()
            print('=' * 92)
            print('  RECONNECTION-NARRATIVE DETECTION')
            print('=' * 92)

            e95_core_late  = avg(late, 'e95_core')
            e95_core_early = avg(early, 'e95_core')
            e95_xline_late = avg(late, 'e95_xline')
            e95_xline_early = avg(early, 'e95_xline')
            e95_spot_late  = avg(late, 'e95_spot')
            n_core_late    = avg(late, 'n_core')
            n_core_early   = avg(early, 'n_core')

            print()
            # Test 1: core heating (PEAK-AWARE: use max value, not late-window avg)
            # This is the key fix: e95_core_max captures transient peaks that the
            # late-window average misses.
            if e95_core_max > 500 and e95_core_max > 5 * max(e95_core_early, 1.0):
                print(f'  ✓ STRONG CORE HEATING (peak-detected):')
                print(f'    Peak: {e95_core_max:,.0f} keV at t = {t_core_max:.1f} ps')
                print(f'    Early-window avg: {e95_core_early:,.0f} keV')
                print(f'    Late-window avg:  {e95_core_late:,.0f} keV')
                print(f'    Suggests transient reconnection-driven acceleration')
            elif e95_core_max > 100 and e95_core_max > 1.5 * max(e95_core_early, 1.0):
                print(f'  ~ MODERATE CORE HEATING (peak-detected):')
                print(f'    Peak: {e95_core_max:,.0f} keV at t = {t_core_max:.1f} ps')
                print(f'    Early-window avg: {e95_core_early:,.0f} keV')
                print(f'    Late-window avg:  {e95_core_late:,.0f} keV')
            else:
                print(f'  ✗ Core not heating significantly')
                print(f'    Peak: {e95_core_max:,.0f} keV; Early avg: {e95_core_early:,.0f} keV; '
                      f'Late avg: {e95_core_late:,.0f} keV')

            # Test 2: particle inflow to core (also use max)
            n_core_max = max(r['n_core'] for r in results)
            if n_core_max > n_core_early * 1.5:
                print(f'  ✓ PARTICLE INFLOW: N_core peaked at {n_core_max:,.0f} '
                      f'(early avg: {n_core_early:,.0f}, late avg: {n_core_late:,.0f})')
            else:
                print(f'  ~ Core particle count: peak {n_core_max:,.0f}, '
                      f'early {n_core_early:,.0f}, late {n_core_late:,.0f}')

            # Test 3: X-line acceleration
            e95_xline_max = max(r['e95_xline'] for r in results
                                if not np.isnan(r['e95_xline']))
            if e95_xline_max > 100 and e95_xline_max > 1.5 * max(e95_xline_early, 1.0):
                print(f'  ✓ X-LINE ACCELERATION: peak {e95_xline_max:,.0f} keV '
                      f'(early avg: {e95_xline_early:,.0f}, late avg: {e95_xline_late:,.0f})')
            else:
                print(f'  ~ X-line evolution: early {e95_xline_early:,.0f} → late {e95_xline_late:,.0f} keV')

            # Test 4: core/spot energy inversion (use max ratio)
            if ratio_max > 1.50:
                print(f'  ✓ STRONG CORE/SPOT INVERSION: peak ratio = {ratio_max:.3f}× '
                      f'at t = {t_ratio_max:.1f} ps (core much hotter than spots)')
            elif ratio_max > 1.05:
                print(f'  ~ Core/spot inversion: peak ratio = {ratio_max:.3f}× '
                      f'at t = {t_ratio_max:.1f} ps')
            else:
                print(f'  ~ Core/spot ratio peaks at {ratio_max:.3f}× (no inversion detected)')

            # p-11B fusion threshold check (peak-aware)
            if e95_core_max >= 500:
                print(f'  ✓ FUSION-RELEVANT CORE: peak E95 = {e95_core_max:,.0f} keV  '
                      f'({e95_core_max/500:.1f}× p-11B threshold) at t = {t_core_max:.1f} ps')
            else:
                print(f'  ~ Core peak E95 = {e95_core_max:,.0f} keV (below 500 keV p-11B threshold)')

# ─── Section 4: B-field evolution ──────────────────────────────────────────
if not args.skip_fields:
    print()
    print('=' * 92)
    print('  B-FIELD EVOLUTION CHECK')
    print('=' * 92)
    field_dir_candidates = [
        os.path.join(run_dir, 'fields_early'),
        os.path.join(run_dir, 'fields'),
        os.path.join(run_dir, 'diags', 'fields_early'),
        os.path.join(run_dir, 'diags', 'fields'),
        os.path.join(run_dir, 'debug_fields'),
    ]
    field_dir = next((p for p in field_dir_candidates if os.path.isdir(p)), None)
    if field_dir is None:
        print(f'  No field dumps found.')
    else:
        try:
            import openpmd_viewer as ov
            ts_f = ov.OpenPMDTimeSeries(field_dir)
            print()
            print(f'  Field dumps from: {field_dir}')
            print(f'  {"t(ps)":>8} {"|B|_max(T)":>12} {"|B|_mean(T)":>14} {"|J|_max(A/m²)":>16}')
            print('  ' + '-' * 60)
            f_iters = list(ts_f.iterations)
            sample = f_iters[::max(1, len(f_iters)//8)]
            if f_iters[-1] not in sample:
                sample.append(f_iters[-1])
            for it in sample:
                try:
                    Bx, _ = ts_f.get_field('B', 'x', iteration=it)
                    By, _ = ts_f.get_field('B', 'y', iteration=it)
                    Bz, _ = ts_f.get_field('B', 'z', iteration=it)
                    Bm = np.sqrt(Bx**2 + By**2 + Bz**2)
                    try:
                        Jy, _ = ts_f.get_field('j', 'y', iteration=it)
                        j_max = float(np.abs(Jy).max())
                    except Exception:
                        j_max = float('nan')
                    t_ps = ts_f.t[list(ts_f.iterations).index(it)] * 1e12
                    print(f'  {t_ps:>8.2f} {Bm.max():>12.2f} {Bm.mean():>14.4f} '
                          f'{j_max:>16.3e}')
                except Exception as e:
                    print(f'  iter {it}: read failed ({e})')
        except ImportError:
            print(f'  openpmd_viewer not available — skipping field check')

# ─── End ───────────────────────────────────────────────────────────────────
print()
print('=' * 92)
print('  END POST-RUN ANALYSIS')
print('=' * 92)
if report_path:
    print(f'  Report saved to: {report_path}')
