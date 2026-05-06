#!/usr/bin/env python3
"""
pb11_reconnection_rate_offline.py — Reconnection rate analyzer for 2D-XZ
hybrid-PIC simulations with guide-field geometry.

Physics context: In 2D-XZ hybrid-PIC, only By (out-of-plane) is non-zero by
symmetry. Bx, Bz, Ey are all constrained to zero. "Reconnection" in this
geometry happens via topology changes of By(x,z) — X-lines are where By
crosses zero, and lobes are regions of opposite-sign By that flux-anneal.

This script computes THREE complementary reconnection metrics:

1. LOBE FLUX ANNIHILATION RATE
   Phi_lobe(t) = integral(|By|) over each lobe region
   rate_flux = -dPhi/dt / (v_A * B_lobe * L_lobe)
   Direct measurement of flux loss.

2. RECONNECTION ELECTRIC FIELD
   At each X-line, find the local in-plane E component perpendicular to grad(By)
   rate_E = E_perp / (v_A * B_lobe)
   Local reconnection rate.

3. B-COLLAPSE RATE (global)
   rate_global = -d|B|_max/dt / |B|_max
   Already in zone analysis but reported here for context.

USAGE:
    python pb11_reconnection_rate_offline.py --dir runs/<run_name>

OUTPUTS:
    <run_dir>/reconnection_rate_offline.csv     - per-iteration time series
    <run_dir>/reconnection_summary.txt          - physical interpretation
"""

import os, sys, argparse, configparser
import numpy as np

PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
MU0            = 4.0e-7 * np.pi
EPS0           = 8.854e-12
Q_E            = 1.602e-19

# CLI
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir', required=True, help='Run directory')
parser.add_argument('--lobe-radius-um', type=float, default=400.0,
                    help='Radius (um) for lobe region around each X-line (default 400)')
parser.add_argument('--xline-search-radius-um', type=float, default=300.0,
                    help='Search radius (um) around expected X-line position (default 300)')
parser.add_argument('--report', default=None,
                    help='Output CSV path (default <dir>/reconnection_rate_offline.csv)')
parser.add_argument('--summary', default=None,
                    help='Summary text path (default <dir>/reconnection_summary.txt)')
parser.add_argument('--debug', action='store_true', help='Verbose output')
args = parser.parse_args()

run_dir = args.dir.rstrip('/')

# Load metadata
meta_path = os.path.join(run_dir, 'run_meta.txt')
if not os.path.exists(meta_path):
    print('ERROR: ' + meta_path + ' not found', file=sys.stderr)
    sys.exit(1)

meta_lines = open(meta_path).read().splitlines()
section_start = next((i for i, l in enumerate(meta_lines) if l.strip().startswith('[')), None)
cp = configparser.ConfigParser()
cp.read_string('\n'.join(meta_lines[section_start:]))

N_SPOTS       = int(cp['geometry']['n_spots'])
RING_RADIUS_M = float(cp['geometry']['ring_radius_m'])
SPOT_RADIUS_M = float(cp['geometry']['spot_radius_m'])

base_density = float(cp['fuel_regions'].get('base_density', '5e25'))
b_seed       = float(cp['physics'].get('b_seed_t', '85'))

# X-line positions (geometric expectation)
R_xline = RING_RADIUS_M * np.cos(np.pi / N_SPOTS)
xline_angles = np.array([2.0 * np.pi * k / N_SPOTS + np.pi / N_SPOTS for k in range(N_SPOTS)])
xline_x_expected = R_xline * np.cos(xline_angles)
xline_z_expected = R_xline * np.sin(xline_angles)

LOBE_RADIUS = args.lobe_radius_um * 1e-6
XLINE_SEARCH_RADIUS = args.xline_search_radius_um * 1e-6

# Load field dumps
try:
    import openpmd_viewer as ov
except ImportError:
    print('Installing openpmd-viewer...')
    os.system('pip install openpmd-viewer --break-system-packages')
    import openpmd_viewer as ov

field_dir_candidates = [
    os.path.join(run_dir, 'fields_early'),
    os.path.join(run_dir, 'fields'),
    os.path.join(run_dir, 'diags', 'fields_early'),
    os.path.join(run_dir, 'diags', 'fields'),
]
field_dir = next((p for p in field_dir_candidates if os.path.isdir(p)), None)
if field_dir is None:
    print('ERROR: no field dumps found in any of: ' + str(field_dir_candidates), file=sys.stderr)
    sys.exit(1)

ts = ov.OpenPMDTimeSeries(field_dir)
iterations = list(ts.iterations)
times_s = np.asarray(ts.t)

# Build coordinate grids from first dump
B_test, info_test = ts.get_field('B', 'y', iteration=iterations[0])
nx, nz = B_test.shape

x_axis = np.asarray(info_test.x)
z_axis = np.asarray(info_test.z)
dx = x_axis[1] - x_axis[0]
dz = z_axis[1] - z_axis[0]
cell_area = dx * dz

X, Z = np.meshgrid(x_axis, z_axis, indexing='ij')
R_grid = np.sqrt(X**2 + Z**2)

print('=' * 92)
print('  RECONNECTION ANALYZER - guide-field 2D-XZ geometry')
print('=' * 92)
print('  Run:            ' + run_dir)
print('  Field dumps:    ' + field_dir)
print('  Snapshots:      ' + str(len(iterations)))
print('  Time range:     {:.2f} -> {:.2f} ps'.format(times_s[0]*1e12, times_s[-1]*1e12))
print('  Geometry:       {} expected X-lines at R = {:.0f} um'.format(N_SPOTS, R_xline*1e6))
print('  Grid:           {}x{}, dx={:.1f} um, dz={:.1f} um'.format(nx, nz, dx*1e6, dz*1e6))
print('  Lobe radius:    {:.0f} um'.format(args.lobe_radius_um))
print('  Base density:   {:.2e} m^-3'.format(base_density))
print('  B-seed:         {:.1f} T'.format(b_seed))
print('=' * 92)

# Bilinear sampling helper
def bilinear_sample(field_2d, x_axis, z_axis, x_query, z_query):
    """Bilinear interpolation. Returns NaN for queries outside grid bounds."""
    nx, nz = field_2d.shape
    dx = x_axis[1] - x_axis[0]
    dz = z_axis[1] - z_axis[0]
    fx = (x_query - x_axis[0]) / dx
    fz = (z_query - z_axis[0]) / dz

    out = np.zeros_like(np.atleast_1d(np.asarray(x_query, dtype=float)))
    flat_fx = np.atleast_1d(fx).astype(float)
    flat_fz = np.atleast_1d(fz).astype(float)
    for i in range(len(out)):
        a = flat_fx[i]; b = flat_fz[i]
        if a < 0 or a >= nx - 1 or b < 0 or b >= nz - 1:
            out[i] = np.nan
            continue
        i0 = int(a); j0 = int(b)
        wa = a - i0; wb = b - j0
        v = (field_2d[i0,   j0  ] * (1 - wa) * (1 - wb) +
             field_2d[i0+1, j0  ] *      wa  * (1 - wb) +
             field_2d[i0,   j0+1] * (1 - wa) *      wb  +
             field_2d[i0+1, j0+1] *      wa  *      wb)
        out[i] = v
    return out

# Find local X-line position
def find_xline_local(By, x0, z0, search_radius):
    """
    Find actual X-line position (local minimum of |By|) in a search circle
    around expected position (x0, z0).
    Returns (x_xline, z_xline, By_at_xline) or (None, None, None) if not found.
    """
    mask = (X - x0)**2 + (Z - z0)**2 < search_radius**2
    if not mask.any():
        return None, None, None
    abs_By_masked = np.where(mask, np.abs(By), np.inf)
    min_idx = np.unravel_index(np.argmin(abs_By_masked), abs_By_masked.shape)
    x_xl = X[min_idx]
    z_xl = Z[min_idx]
    By_xl = By[min_idx]
    return x_xl, z_xl, By_xl


# Per-iteration computation
header = ['step', 't_ps']
for k in range(N_SPOTS):
    header += ['xline{}_x_um'.format(k), 'xline{}_z_um'.format(k),
               'xline{}_By_T'.format(k), 'xline{}_Bcollapse_T'.format(k),
               'xline{}_Phi_lobe_Tm2'.format(k), 'xline{}_E_perp_Vm'.format(k),
               'xline{}_v_in_ms'.format(k), 'xline{}_v_in_over_vA'.format(k),
               'xline{}_rate_E'.format(k), 'xline{}_rate_flux'.format(k)]
header += ['B_max_T', 'B_collapse_pct', 'rate_global',
           'rate_E_mean', 'rate_flux_mean']

rows = []
flux_history = [[] for _ in range(N_SPOTS)]

print()
print('  {:>6s} {:>9s} {:>9s} {:>9s} {:>11s} {:>11s} {:>11s}'.format(
    'step', 't(ps)', '|B|_max', '<B_xl>', '<E_perp>', '<v_in/vA>', '<rate_E>'))
print('  ' + '-' * 72)

for it_idx, it in enumerate(iterations):
    t_ps = float(times_s[it_idx] * 1e12)
    t_s  = float(times_s[it_idx])

    try:
        By, _ = ts.get_field('B', 'y', iteration=it)
        Ex, _ = ts.get_field('E', 'x', iteration=it)
        Ez, _ = ts.get_field('E', 'z', iteration=it)
    except Exception as e:
        print('  iter {}: field read failed ({})'.format(it, e))
        continue

    By_mag = np.abs(By)
    B_max = float(By_mag.max())

    row = [int(it), t_ps]

    rates_E = []
    rates_flux = []

    for k in range(N_SPOTS):
        x_xl, z_xl, By_xl = find_xline_local(By, xline_x_expected[k], xline_z_expected[k],
                                              XLINE_SEARCH_RADIUS)

        if x_xl is None:
            row += [np.nan]*10
            continue

        lobe_mask = (X - x_xl)**2 + (Z - z_xl)**2 < LOBE_RADIUS**2
        lobe_By = By[lobe_mask]
        if lobe_By.size == 0:
            row += [np.nan]*10
            continue

        B_lobe = float(np.percentile(np.abs(lobe_By), 90))
        Phi_lobe = float(np.sum(np.abs(lobe_By)) * cell_area)
        B_collapse = b_seed - B_lobe

        # Use base mass density (proton number density × proton mass).
        # NOTE: We do NOT read the rho field from WarpX because WarpX stores
        # rho as CHARGE density in C/m^3, not mass density in kg/m^3. For a
        # quasineutral plasma the charge density fluctuations are tiny, which
        # would give artificially huge v_A values. Using base_density * m_p
        # gives the correct ion mass density.
        rho_lobe = base_density * PROTON_MASS_KG

        if B_lobe > 0.1 and rho_lobe > 1e-9:
            v_A = B_lobe / np.sqrt(MU0 * rho_lobe)
        else:
            v_A = 0.0

        Ex_xl = float(bilinear_sample(Ex, x_axis, z_axis,
                                      np.array([x_xl]), np.array([z_xl]))[0])
        Ez_xl = float(bilinear_sample(Ez, x_axis, z_axis,
                                      np.array([x_xl]), np.array([z_xl]))[0])

        delta = max(dx, dz)
        By_xp = bilinear_sample(By, x_axis, z_axis, np.array([x_xl + delta]), np.array([z_xl]))[0]
        By_xm = bilinear_sample(By, x_axis, z_axis, np.array([x_xl - delta]), np.array([z_xl]))[0]
        By_zp = bilinear_sample(By, x_axis, z_axis, np.array([x_xl]), np.array([z_xl + delta]))[0]
        By_zm = bilinear_sample(By, x_axis, z_axis, np.array([x_xl]), np.array([z_xl - delta]))[0]

        if any(np.isnan([By_xp, By_xm, By_zp, By_zm])):
            row += [x_xl*1e6, z_xl*1e6, float(By_xl), B_collapse, Phi_lobe,
                    np.nan, np.nan, np.nan, np.nan, np.nan]
            continue

        gradBy_x = (By_xp - By_xm) / (2.0 * delta)
        gradBy_z = (By_zp - By_zm) / (2.0 * delta)
        grad_mag = float(np.sqrt(gradBy_x**2 + gradBy_z**2))

        if grad_mag < 1e-3:
            row += [x_xl*1e6, z_xl*1e6, float(By_xl), B_collapse, Phi_lobe,
                    np.nan, np.nan, np.nan, np.nan, np.nan]
            continue

        E_along_grad = (Ex_xl * gradBy_x + Ez_xl * gradBy_z) / grad_mag
        E_total_sq = Ex_xl**2 + Ez_xl**2
        E_perp_sq = E_total_sq - E_along_grad**2
        E_perp = float(np.sqrt(max(E_perp_sq, 0.0)))

        v_in = np.sqrt(E_total_sq) / B_lobe if B_lobe > 0.1 else 0.0
        v_in_over_vA = v_in / v_A if v_A > 0 else 0.0

        rate_E = E_perp / (v_A * B_lobe) if (v_A > 0 and B_lobe > 0.1) else np.nan

        flux_history[k].append((t_s, Phi_lobe, B_lobe, v_A))

        rate_flux = np.nan
        if len(flux_history[k]) >= 2:
            t_prev, Phi_prev, B_prev, vA_prev = flux_history[k][-2]
            dt = t_s - t_prev
            if dt > 0 and B_prev > 0.1 and vA_prev > 0:
                dPhi_dt = (Phi_lobe - Phi_prev) / dt
                L_lobe = 2 * LOBE_RADIUS
                avg_vA = 0.5 * (v_A + vA_prev)
                avg_B  = 0.5 * (B_lobe + B_prev)
                rate_flux = -dPhi_dt / (avg_vA * avg_B * L_lobe)

        row += [x_xl*1e6, z_xl*1e6, float(By_xl), float(B_collapse), Phi_lobe,
                E_perp, float(v_in), float(v_in_over_vA),
                float(rate_E) if not np.isnan(rate_E) else np.nan,
                float(rate_flux) if not np.isnan(rate_flux) else np.nan]

        if not np.isnan(rate_E):  rates_E.append(rate_E)
        if not (isinstance(rate_flux, float) and np.isnan(rate_flux)): rates_flux.append(rate_flux)

    B_collapse_pct = (1.0 - B_max / b_seed) * 100.0
    rate_global = -1.0  # placeholder, computed below

    rate_E_mean = float(np.mean(rates_E)) if rates_E else np.nan
    rate_flux_mean = float(np.mean(rates_flux)) if rates_flux else np.nan

    row += [B_max, B_collapse_pct, rate_global, rate_E_mean, rate_flux_mean]
    rows.append(row)

    e_perp_vals = [row[2 + 10*k + 5] for k in range(N_SPOTS)
                   if not (isinstance(row[2 + 10*k + 5], float) and np.isnan(row[2 + 10*k + 5]))]
    avg_e_perp = float(np.mean(e_perp_vals)) if e_perp_vals else 0.0

    v_ratio_vals = [row[2 + 10*k + 7] for k in range(N_SPOTS)
                    if not (isinstance(row[2 + 10*k + 7], float) and np.isnan(row[2 + 10*k + 7]))]
    avg_v_ratio = float(np.mean(v_ratio_vals)) if v_ratio_vals else 0.0

    by_at_xls = [abs(row[2 + 10*k + 2]) for k in range(N_SPOTS)
                 if not (isinstance(row[2 + 10*k + 2], float) and np.isnan(row[2 + 10*k + 2]))]
    avg_byxl = float(np.mean(by_at_xls)) if by_at_xls else 0.0

    print('  {:>6d} {:>9.2f} {:>9.2f} {:>9.3f} {:>11.3e} {:>11.2f} {:>11.4f}'.format(
        it, t_ps, B_max, avg_byxl, avg_e_perp, avg_v_ratio,
        rate_E_mean if not np.isnan(rate_E_mean) else float('nan')))

# Compute global B-collapse rate as time series
B_max_series = np.array([row[2 + 10*N_SPOTS] for row in rows])
t_arr = np.array([row[1] for row in rows]) * 1e-12
for i in range(len(rows)):
    if i == 0:
        rows[i][2 + 10*N_SPOTS + 2] = 0.0
    else:
        if t_arr[i] - t_arr[i-1] > 0:
            dB_dt = (B_max_series[i] - B_max_series[i-1]) / (t_arr[i] - t_arr[i-1])
            rows[i][2 + 10*N_SPOTS + 2] = -dB_dt / max(B_max_series[i], 0.1)
        else:
            rows[i][2 + 10*N_SPOTS + 2] = 0.0

# Write CSV
import csv as csv_module
out_path = args.report or os.path.join(run_dir, 'reconnection_rate_offline.csv')
with open(out_path, 'w', newline='') as f:
    w = csv_module.writer(f)
    w.writerow(header)
    for row in rows:
        w.writerow(row)

print()
print('  CSV written: ' + out_path)
print()

# Summary report
summary_path = args.summary or os.path.join(run_dir, 'reconnection_summary.txt')

all_rate_E = [r[-2] for r in rows if not (isinstance(r[-2], float) and np.isnan(r[-2]))]
all_rate_flux = [r[-1] for r in rows if not (isinstance(r[-1], float) and np.isnan(r[-1]))]
B_max_vals = [r[2 + 10*N_SPOTS] for r in rows]
B_collapse_pct_vals = [r[2 + 10*N_SPOTS + 1] for r in rows]
rate_global_vals = [r[2 + 10*N_SPOTS + 2] for r in rows[1:]]

with open(summary_path, 'w') as sf:
    sf.write('=' * 92 + '\n')
    sf.write('  RECONNECTION ANALYSIS SUMMARY\n')
    sf.write('=' * 92 + '\n')
    sf.write('  Run:               ' + run_dir + '\n')
    sf.write('  Field snapshots:   {}\n'.format(len(iterations)))
    sf.write('  Time range:        {:.2f} -> {:.2f} ps\n'.format(times_s[0]*1e12, times_s[-1]*1e12))
    sf.write('  Geometry:          2D-XZ guide-field reconnection (By is the only B component)\n')
    sf.write('  Expected X-lines:  {} at R = {:.0f} um\n'.format(N_SPOTS, R_xline*1e6))
    sf.write('\n')

    sf.write('=' * 92 + '\n')
    sf.write('  GLOBAL B-FIELD EVOLUTION\n')
    sf.write('=' * 92 + '\n')
    sf.write('  Initial |By|_max:  {:.2f} T (seed={:.1f} T)\n'.format(B_max_vals[0], b_seed))
    sf.write('  Final |By|_max:    {:.2f} T\n'.format(B_max_vals[-1]))
    sf.write('  Min |By|_max:      {:.2f} T\n'.format(min(B_max_vals)))
    sf.write('  Final collapse:    {:.1f}%\n'.format(B_collapse_pct_vals[-1]))
    sf.write('  Peak collapse:     {:.1f}%\n'.format(max(B_collapse_pct_vals)))
    if rate_global_vals:
        sf.write('  Mean global rate:  {:.3e} /s\n'.format(np.mean(rate_global_vals)))
        sf.write('  Peak global rate:  {:.3e} /s\n'.format(max(rate_global_vals)))
    sf.write('\n')

    sf.write('=' * 92 + '\n')
    sf.write('  PER-X-LINE RECONNECTION METRIC: E-FIELD RATE\n')
    sf.write('=' * 92 + '\n')
    sf.write('  Definition: rate_E = E_perp / (v_A * B_lobe)\n')
    sf.write('              where E_perp is the in-plane E component perpendicular\n')
    sf.write('              to grad(By) at the X-line\n')
    sf.write('\n')
    sf.write('  Healthy fast reconnection: 0.05 - 0.15\n')
    sf.write('  Slow reconnection:         0.001 - 0.05\n')
    sf.write('  Diffusion-limited:         < 0.001\n')
    sf.write('\n')
    if all_rate_E:
        sf.write('  Valid measurements:    {} / {}\n'.format(len(all_rate_E), len(rows)*N_SPOTS))
        sf.write('  Mean rate_E:           {:.4f}\n'.format(np.mean(all_rate_E)))
        sf.write('  Median rate_E:         {:.4f}\n'.format(np.median(all_rate_E)))
        sf.write('  Maximum rate_E:        {:.4f}\n'.format(np.max(all_rate_E)))
        sf.write('  Minimum rate_E:        {:.4f}\n'.format(np.min(all_rate_E)))
        if np.mean(all_rate_E) > 0.05:
            sf.write('  -> FAST RECONNECTION REGIME\n')
        elif np.mean(all_rate_E) > 0.001:
            sf.write('  -> SLOW / SUB-ALFVENIC RECONNECTION\n')
        else:
            sf.write('  -> DIFFUSION-LIMITED OR NEGLIGIBLE RECONNECTION\n')
    else:
        sf.write('  No valid E-field rate measurements (likely no X-lines forming)\n')
    sf.write('\n')

    sf.write('=' * 92 + '\n')
    sf.write('  PER-X-LINE RECONNECTION METRIC: FLUX ANNIHILATION RATE\n')
    sf.write('=' * 92 + '\n')
    sf.write('  Definition: rate_flux = -dPhi_lobe/dt / (v_A * B_lobe * L_lobe)\n')
    sf.write('              where Phi_lobe = integral|By| dA over lobe region\n')
    sf.write('\n')
    if all_rate_flux:
        sf.write('  Valid measurements:    {}\n'.format(len(all_rate_flux)))
        sf.write('  Mean rate_flux:        {:.4f}\n'.format(np.mean(all_rate_flux)))
        sf.write('  Median rate_flux:      {:.4f}\n'.format(np.median(all_rate_flux)))
        sf.write('  Maximum |rate_flux|:   {:.4f}\n'.format(np.max(np.abs(all_rate_flux))))
    else:
        sf.write('  No valid flux annihilation measurements\n')
    sf.write('\n')

    sf.write('=' * 92 + '\n')
    sf.write('  PHYSICS INTERPRETATION\n')
    sf.write('=' * 92 + '\n')
    sf.write('  Geometry: 2D-XZ guide-field hybrid-PIC. By is the only B component;\n')
    sf.write('  Bx, Bz, Ey are constrained to zero by symmetry. Reconnection occurs\n')
    sf.write('  via topology changes of By(x,z): X-lines are zero-crossings of By.\n')
    sf.write('\n')
    sf.write('  The PRIMARY reconnection metric in this geometry is:\n')
    sf.write('    - rate_flux: direct measurement of dPhi/dt at lobes\n')
    sf.write('    - B-collapse: global flux annihilation\n')
    sf.write('\n')
    sf.write('  rate_E (E_perp/v_A/B_lobe) is reported but should be interpreted\n')
    sf.write('  carefully: hybrid-PIC E includes Hall (J×B/ne), pressure gradient,\n')
    sf.write('  and resistive terms beyond ideal MHD. Large rate_E values reflect\n')
    sf.write('  these non-ideal contributions, not ideal MHD reconnection rates.\n')
    sf.write('  In hybrid-PIC of low-beta plasmas, rate_E >> 0.1 is typical and\n')
    sf.write('  indicates Hall-dominated fast reconnection physics.\n')
    sf.write('\n')
    sf.write('  rate_flux is the most reliable metric: it directly measures how fast\n')
    sf.write('  magnetic flux is leaving the lobe regions. Healthy reconnection has\n')
    sf.write('  |rate_flux| in the range 0.01-0.5.\n')
    sf.write('\n')
    sf.write('=' * 92 + '\n')

print('  Summary written: ' + summary_path)
print()

# Print summary to stdout
print('=' * 92)
print('  SUMMARY')
print('=' * 92)
if all_rate_E:
    print('  Mean rate_E:        {:.4f}'.format(np.mean(all_rate_E)))
    print('  Median rate_E:      {:.4f}'.format(np.median(all_rate_E)))
    print('  Maximum rate_E:     {:.4f}'.format(np.max(all_rate_E)))
print('  B-collapse final:   {:.1f}%'.format(B_collapse_pct_vals[-1]))
print('  B-collapse peak:    {:.1f}%'.format(max(B_collapse_pct_vals)))
print()
print('=' * 92)
