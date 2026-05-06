#!/usr/bin/env python3
"""
pb11_paper_figures.py — Comprehensive publication figures + data extraction.

CRITICAL DESIGN: This script extracts EVERYTHING we'll ever want from the
particle dumps in a single pass, before the dumps are deleted. All derived
quantities are cached to compact .npz files in <run>/figures/cache/. After
this runs successfully, the particle dumps can be safely deleted and all
figures + future analyses can be regenerated from the cache.

PRODUCES:
  Two 6-panel composite figures (12 panels total):
    composite_snapshots.png — spatial diagnostics at saturation
      1. |B| field magnitude (overall topology)
      2. B-streamlines + Jy heatmap (X-line topology)
      3. Plasma density (cavitation/compression)
      4. Particle KE scatter (centre acceleration)
      5. Energy spectrum: centre vs outer
      6. Zone E95th comparison bar chart

    composite_evolution.png — time-resolved diagnostics
      7. Fast-ion fraction + fusion power (twin-axis timeseries)
      8. Cumulative fusion yield + gain
      9. Zone-resolved E95 over time (4 lines: core, inner, xline, spot)
      10. Centre/spot ratio over time (reconnection signature)
      11. Particle counts per zone over time
      12. |B|_max + |B|_xline over time (reconnection rate proxy)

  Individual PNGs for all 12 panels (talks/posters/supplements)
  Optional MP4 animation of |B| evolution
  Compact .npz cache (~1-10 MB) preserving all extracted data

USAGE:
    python pb11_paper_figures.py --dir runs/<run_name>
    python pb11_paper_figures.py --dir runs/<run_name> --animate
    python pb11_paper_figures.py --dir runs/<run_name> --style light --animate
    python pb11_paper_figures.py --dir runs/<run_name> --from-cache  # regen after deletion

After successful run, the run directory contains:
    figures/composite_snapshots.png
    figures/composite_evolution.png
    figures/individual/snap_01..snap_06_*.png
    figures/individual/evol_07..evol_12_*.png
    figures/b_evolution.mp4 (if --animate)
    figures/cache/zone_timeseries.npz
    figures/cache/snapshot_data.npz
    figures/cache/field_timeseries.npz
"""

import os
import sys
import argparse
import csv
import warnings
import time as time_module
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

# ─── Constants ─────────────────────────────────────────────────────────────
PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3
MU_0           = 4 * np.pi * 1e-7

# ─── Argparse ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir',             required=True, help='Run output directory')
parser.add_argument('--animate',         action='store_true', help='Generate MP4 animation')
parser.add_argument('--style',           default='dark', choices=['dark', 'light'])
parser.add_argument('--snapshot-time',   type=float, default=-1.0,
                    help='Time in ns for "saturation" panels (default: last available)')
parser.add_argument('--core-frac',       type=float, default=0.5)
parser.add_argument('--xline-frac',      type=float, default=0.05)
parser.add_argument('--species',         default='proton')
parser.add_argument('--max-particles-scatter', type=int, default=50000)
parser.add_argument('--max-snapshots',   type=int, default=0,
                    help='Cap on snapshots to process (0 = all)')
parser.add_argument('--from-cache',      action='store_true',
                    help='Skip particle data extraction; rebuild figures from .npz cache')
parser.add_argument('--no-composites',   action='store_true')
parser.add_argument('--no-individuals',  action='store_true')
parser.add_argument('--output-dir',      default=None)
args = parser.parse_args()

run_dir = args.dir.rstrip('/')
if not os.path.isdir(run_dir):
    print(f'ERROR: run dir not found: {run_dir}')
    sys.exit(1)

output_dir = args.output_dir or os.path.join(run_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)
individual_dir = os.path.join(output_dir, 'individual')
os.makedirs(individual_dir, exist_ok=True)
cache_dir = os.path.join(output_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)

# ─── Style ─────────────────────────────────────────────────────────────────
if args.style == 'dark':
    BG_COLOR    = '#0d1117'
    PANEL_COLOR = '#161b22'
    GRID_COLOR  = '#30363d'
    TEXT_COLOR  = '#e8eaed'
    MUTED_COLOR = '#8b949e'
    CMAP_DENSITY = 'magma'
else:
    BG_COLOR    = 'white'
    PANEL_COLOR = 'white'
    GRID_COLOR  = '#cccccc'
    TEXT_COLOR  = '#222222'
    MUTED_COLOR = '#666666'
    CMAP_DENSITY = 'viridis'

CMAP_BIPOLAR = 'RdBu_r'
CMAP_KE      = 'plasma'

ZONE_COLORS = {
    'core':  '#ff4d4d',
    'inner': '#f0c040',
    'xline': '#1D9E75',
    'spot':  '#4d9fff',
}

ACCENT_RED    = '#ff4d4d'
ACCENT_BLUE   = '#4d9fff'
ACCENT_YELLOW = '#f0c040'
ACCENT_GREEN  = '#1D9E75'

def style_axes(ax, title=None, fontsize=10):
    ax.set_facecolor(PANEL_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=fontsize, pad=8)
    ax.tick_params(colors=MUTED_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

def style_figure(fig):
    fig.patch.set_facecolor(BG_COLOR)

# ─── Load run metadata ─────────────────────────────────────────────────────
print(f'\n=== pb11_paper_figures.py ===')
print(f'Run dir:    {run_dir}')
print(f'Output:     {output_dir}')
print(f'Cache:      {cache_dir}')
print(f'Style:      {args.style}')
print(f'From cache: {args.from_cache}')

meta_path = os.path.join(run_dir, 'run_meta.txt')
N_SPOTS = 8
RING_RADIUS_M = 200e-6
SPOT_RADIUS_M = 50e-6
B_SEED_T = 85.0
LX_M = 9.6e-3
LZ_M = 9.6e-3

if os.path.exists(meta_path):
    # run_meta.txt is plain key=value pairs (no INI sections), with optional
    # comment lines starting with #. Parse it manually so we don't depend on
    # ConfigParser's [section] requirement.
    try:
        meta_kv = {}
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('['):
                    continue
                if '=' in line:
                    key, _, val = line.partition('=')
                    meta_kv[key.strip()] = val.strip()
        # Pull values with fallbacks to the defaults set above
        N_SPOTS       = int(meta_kv.get('n_spots', N_SPOTS))
        RING_RADIUS_M = float(meta_kv.get('ring_radius_m', RING_RADIUS_M))
        SPOT_RADIUS_M = float(meta_kv.get('spot_radius_m', SPOT_RADIUS_M))
        B_SEED_T      = float(meta_kv.get('b_seed_t', meta_kv.get('B_seed_T', B_SEED_T)))
        LX_M          = float(meta_kv.get('lx_m', meta_kv.get('LX_M', LX_M)))
        LZ_M          = float(meta_kv.get('lz_m', meta_kv.get('LZ_M', LZ_M)))
    except Exception as e:
        print(f'WARNING: run_meta parse failed ({e}); using defaults')

print(f'\nGeometry: N_SPOTS={N_SPOTS}, R_ring={RING_RADIUS_M*1e6:.0f} um, R_spot={SPOT_RADIUS_M*1e6:.0f} um')
print(f'Plasma:   B_seed={B_SEED_T:.1f} T, domain={LX_M*1e3:.1f}x{LZ_M*1e3:.1f} mm')

R_xline       = RING_RADIUS_M * np.cos(np.pi / N_SPOTS) if N_SPOTS > 1 else 0
R_core        = args.core_frac * RING_RADIUS_M
R_xline_inner = (np.cos(np.pi / N_SPOTS) - args.xline_frac) * RING_RADIUS_M if N_SPOTS > 1 else 0
R_xline_outer = (np.cos(np.pi / N_SPOTS) + args.xline_frac) * RING_RADIUS_M if N_SPOTS > 1 else 0
R_spot_inner  = RING_RADIUS_M - 2 * SPOT_RADIUS_M

print(f'Zones:    CORE r<{R_core*1e6:.0f}um  | INNER {R_core*1e6:.0f}-{R_xline_inner*1e6:.0f}um  | XLINE {R_xline_inner*1e6:.0f}-{R_xline_outer*1e6:.0f}um  | SPOT >={R_spot_inner*1e6:.0f}um')
print()

# ─── OpenPMD setup ─────────────────────────────────────────────────────────
field_dir = None
for cand in [os.path.join(run_dir, 'fields'),
             os.path.join(run_dir, 'diags', 'fields')]:
    if os.path.isdir(cand) and any(f.endswith('.h5') for f in os.listdir(cand)):
        field_dir = cand
        break

particle_dir = None
for cand in [os.path.join(run_dir, 'particles'),
             os.path.join(run_dir, 'diags', 'particles'),
             os.path.join(run_dir, 'particles_early'),
             os.path.join(run_dir, 'diags', 'particles_early')]:
    if os.path.isdir(cand) and any(f.endswith('.h5') for f in os.listdir(cand)):
        particle_dir = cand
        break

ts_field = None
ts_part  = None
iterations = []
times_s = np.array([])

if not args.from_cache:
    if field_dir is None:
        print(f'ERROR: no field dumps found in {run_dir}/fields')
        sys.exit(1)

    try:
        import openpmd_viewer as ov
    except ImportError:
        print('ERROR: openpmd-viewer not installed')
        sys.exit(1)

    ts_field = ov.OpenPMDTimeSeries(field_dir)
    iterations = ts_field.iterations
    times_s    = ts_field.t
    ts_part = ov.OpenPMDTimeSeries(particle_dir) if particle_dir else None

    print(f'Field snapshots:    {len(iterations)}, t = {times_s[0]*1e12:.1f} to {times_s[-1]*1e12:.1f} ps')
    if ts_part is not None:
        print(f'Particle snapshots: {len(ts_part.iterations)}')
    else:
        print(f'WARNING: no particle dumps found, particle-derived panels will be skipped')
    print()

if not args.from_cache:
    if args.snapshot_time > 0:
        target_s = args.snapshot_time * 1e-9
        sat_idx = int(np.argmin(np.abs(times_s - target_s)))
    else:
        sat_idx = len(iterations) - 1
    sat_iter = iterations[sat_idx]
    sat_t_ps = times_s[sat_idx] * 1e12
else:
    sat_idx = -1
    sat_iter = -1
    sat_t_ps = -1.0

# ─── Field loading helpers ─────────────────────────────────────────────────
def get_field_2d(name, comp, iteration):
    F, info = ts_field.get_field(name, comp, iteration=iteration)
    if F.ndim == 3:
        F = F[F.shape[0]//2]
    return F, info

def get_field_extent_mm(info):
    """Return [x_min, x_max, z_min, z_max] in mm.
    
    openpmd-viewer's FieldMetaInformation exposes axis coordinates as direct
    attributes (info.x, info.z, info.y for 3D), not as info.axes[name].
    info.axes is a dict mapping {0: 'x', 1: 'z'} (index -> name).
    """
    # info.axes maps int -> name (e.g., {0: 'x', 1: 'z'})
    axis_names = [info.axes[i] for i in sorted(info.axes.keys())]
    arrays = [getattr(info, name) for name in axis_names]
    x_min = float(arrays[0][0])
    x_max = float(arrays[0][-1])
    z_min = float(arrays[1][0])
    z_max = float(arrays[1][-1])
    return [x_min*1e3, x_max*1e3, z_min*1e3, z_max*1e3]

def overlay_geometry(ax):
    ring_r_mm = RING_RADIUS_M * 1e3
    spot_r_mm = SPOT_RADIUS_M * 1e3
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(ring_r_mm*np.cos(theta), ring_r_mm*np.sin(theta),
            color=ACCENT_YELLOW, lw=0.8, alpha=0.6, ls='--')
    if N_SPOTS > 1:
        for k in range(N_SPOTS):
            a = 2*np.pi*k/N_SPOTS
            cx = ring_r_mm * np.cos(a)
            cz = ring_r_mm * np.sin(a)
            ax.add_patch(plt.Circle((cx, cz), spot_r_mm,
                                    fill=False, ec=ACCENT_YELLOW, lw=0.5))
        for k in range(N_SPOTS):
            a = 2*np.pi*(k + 0.5)/N_SPOTS
            cx = R_xline * 1e3 * np.cos(a)
            cz = R_xline * 1e3 * np.sin(a)
            ax.plot(cx, cz, '+', color=ACCENT_RED, ms=8, mew=1.5)

def load_particles(p_iter):
    """Try common WarpX/openpmd particle field name variants."""
    for fields in (
        ['x', 'z', 'ux', 'uy', 'uz', 'w'],
        ['position_x', 'position_z', 'momentum_x', 'momentum_y', 'momentum_z', 'weighting'],
    ):
        try:
            data = ts_part.get_particle(fields, species=args.species, iteration=p_iter)
            if data is not None and len(data) >= 6 and data[0] is not None and len(data[0]) > 0:
                return data
        except Exception:
            continue
    return None, None, None, None, None, None

# ─── PASS 1: Zone timeseries ───────────────────────────────────────────────
zone_cache_path     = os.path.join(cache_dir, 'zone_timeseries.npz')
snapshot_cache_path = os.path.join(cache_dir, 'snapshot_data.npz')
field_ts_cache_path = os.path.join(cache_dir, 'field_timeseries.npz')

def extract_zone_timeseries():
    if ts_part is None:
        print('SKIP: zone timeseries (no particle dumps)')
        return None

    p_iters = list(ts_part.iterations)
    if args.max_snapshots > 0 and len(p_iters) > args.max_snapshots:
        step = max(1, len(p_iters) // args.max_snapshots)
        p_iters = p_iters[::step][:args.max_snapshots]

    print(f'PASS 1: Extracting zone timeseries from {len(p_iters)} particle snapshots...')
    mass_kg = B11_MASS_KG if args.species == 'boron11' else PROTON_MASS_KG

    out = {
        'times_ps': [], 'iterations': [],
        'N_core': [], 'N_inner': [], 'N_xline': [], 'N_spot': [],
        'E95_core': [], 'E95_inner': [], 'E95_xline': [], 'E95_spot': [],
        'Emean_core': [], 'Emean_inner': [], 'Emean_xline': [], 'Emean_spot': [],
        'fast_frac_core': [], 'fast_frac_inner': [],
        'fast_frac_xline': [], 'fast_frac_spot': [],
    }

    p_times_arr = np.asarray(ts_part.t)
    p_iters_arr = list(ts_part.iterations)

    for i, p_iter in enumerate(p_iters):
        x, z, ux, uy, uz, w = load_particles(p_iter)
        if x is None or len(x) == 0:
            continue

        u2 = ux**2 + uy**2 + uz**2
        # === v12.18 FIX: u is dimensionless momentum (gamma*v/c), not velocity ===
        # WarpX stores momentum as u = gamma*v/c (dimensionless). The Lorentz
        # factor is gamma = sqrt(1 + u^2). Previous code used u^2/C^2 which is
        # the wrong formula and gave gamma=1 -> all KE values became zero in
        # the cache file. This matches the convention in pb11_zone_analysis.py.
        gamma_1 = np.sqrt(1 + u2)
        KE_keV = (gamma_1 - 1) * mass_kg * C**2 / KEV_TO_J

        r = np.sqrt(x**2 + z**2)
        masks = {
            'core':  r < R_core,
            'inner': (r >= R_core) & (r < R_xline_inner),
            'xline': (r >= R_xline_inner) & (r <= R_xline_outer),
            'spot':  r >= R_spot_inner,
        }

        idx_in_arr = p_iters_arr.index(p_iter)
        t_ps = float(p_times_arr[idx_in_arr] * 1e12)
        out['times_ps'].append(t_ps)
        out['iterations'].append(int(p_iter))

        for zone in ('core', 'inner', 'xline', 'spot'):
            m = masks[zone]
            if m.any():
                ke_z = KE_keV[m]
                w_z  = w[m]
                w_sum = float(w_z.sum())
                out[f'N_{zone}'].append(w_sum)
                out[f'E95_{zone}'].append(float(np.percentile(ke_z, 95)))
                out[f'Emean_{zone}'].append(float(np.average(ke_z, weights=w_z)))
                fast = ke_z > 500.0
                out[f'fast_frac_{zone}'].append(float(w_z[fast].sum() / w_sum) if w_sum > 0 else 0.0)
            else:
                out[f'N_{zone}'].append(0.0)
                out[f'E95_{zone}'].append(0.0)
                out[f'Emean_{zone}'].append(0.0)
                out[f'fast_frac_{zone}'].append(0.0)

        if (i+1) % 5 == 0 or i+1 == len(p_iters):
            print(f'  processed {i+1}/{len(p_iters)} (t={t_ps:.0f} ps)')

    for k in out:
        out[k] = np.asarray(out[k])

    np.savez(zone_cache_path, **out)
    print(f'  -> {zone_cache_path}')
    return out

def extract_field_timeseries():
    if ts_field is None:
        return None
    print(f'PASS 1b: Extracting field timeseries from {len(iterations)} field snapshots...')
    out = {
        'times_ps': [], 'iterations': [],
        'B_max': [], 'B_mean': [],
        'B_xline_max': [], 'B_xline_min': [],
        'Jy_max': [],
    }

    Bx0, info0 = get_field_2d('B', 'x', iterations[0])
    extent = get_field_extent_mm(info0)
    nx, nz = Bx0.shape
    x = np.linspace(extent[0]*1e-3, extent[1]*1e-3, nx)
    z = np.linspace(extent[2]*1e-3, extent[3]*1e-3, nz)
    X, Z = np.meshgrid(x, z, indexing='ij')
    R = np.sqrt(X**2 + Z**2)
    xline_mask = (R >= R_xline_inner) & (R <= R_xline_outer)

    for i, it in enumerate(iterations):
        Bx, _ = get_field_2d('B', 'x', it)
        By, _ = get_field_2d('B', 'y', it)
        Bz, _ = get_field_2d('B', 'z', it)
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        try:
            Jy, _ = get_field_2d('j', 'y', it)
            jy_max = float(np.abs(Jy).max())
        except Exception:
            jy_max = 0.0

        out['times_ps'].append(float(times_s[i] * 1e12))
        out['iterations'].append(int(it))
        out['B_max'].append(float(Bmag.max()))
        out['B_mean'].append(float(Bmag.mean()))
        if xline_mask.any():
            out['B_xline_max'].append(float(Bmag[xline_mask].max()))
            out['B_xline_min'].append(float(Bmag[xline_mask].min()))
        else:
            out['B_xline_max'].append(0.0)
            out['B_xline_min'].append(0.0)
        out['Jy_max'].append(jy_max)

        if (i+1) % 10 == 0 or i+1 == len(iterations):
            print(f'  processed {i+1}/{len(iterations)} field snapshots')

    for k in out:
        out[k] = np.asarray(out[k])
    np.savez(field_ts_cache_path, **out)
    print(f'  -> {field_ts_cache_path}')
    return out

def extract_snapshot_data():
    print(f'PASS 2: Extracting snapshot data at sat_iter={sat_iter} (t={sat_t_ps:.0f} ps)...')
    out = {'sat_iter': sat_iter, 'sat_t_ps': sat_t_ps}

    Bx, info = get_field_2d('B', 'x', sat_iter)
    By, _    = get_field_2d('B', 'y', sat_iter)
    Bz, _    = get_field_2d('B', 'z', sat_iter)
    out['Bx'] = Bx.astype(np.float32)
    out['By'] = By.astype(np.float32)
    out['Bz'] = Bz.astype(np.float32)
    out['extent_mm'] = np.array(get_field_extent_mm(info))

    try:
        Jy, _ = get_field_2d('j', 'y', sat_iter)
        out['Jy'] = Jy.astype(np.float32)
    except Exception:
        out['Jy'] = np.zeros_like(Bx, dtype=np.float32)

    try:
        rho, _ = get_field_2d('rho', None, sat_iter)
        out['rho'] = rho.astype(np.float32)
    except Exception:
        out['rho'] = np.zeros_like(Bx, dtype=np.float32)

    if ts_part is not None:
        p_iters = list(ts_part.iterations)
        p_times_arr = np.asarray(ts_part.t)
        p_idx = int(np.argmin(np.abs(p_times_arr - times_s[sat_idx])))
        p_iter = p_iters[p_idx]
        out['p_sat_iter'] = int(p_iter)
        out['p_sat_t_ps'] = float(p_times_arr[p_idx] * 1e12)

        x, z, ux, uy, uz, w = load_particles(p_iter)
        if x is not None and len(x) > 0:
            mass_kg = B11_MASS_KG if args.species == 'boron11' else PROTON_MASS_KG
            u2 = ux**2 + uy**2 + uz**2
            # === v12.18 FIX: u is dimensionless momentum (gamma*v/c), not velocity ===
            # See note in extract_zone_timeseries above. Same bug fixed here.
            gamma_1 = np.sqrt(1 + u2)
            KE_keV = (gamma_1 - 1) * mass_kg * C**2 / KEV_TO_J

            n = len(x)
            if n > args.max_particles_scatter:
                idx = np.random.choice(n, args.max_particles_scatter, replace=False)
                out['scatter_x']  = x[idx].astype(np.float32)
                out['scatter_z']  = z[idx].astype(np.float32)
                out['scatter_KE'] = KE_keV[idx].astype(np.float32)
            else:
                out['scatter_x']  = x.astype(np.float32)
                out['scatter_z']  = z.astype(np.float32)
                out['scatter_KE'] = KE_keV.astype(np.float32)

            r = np.sqrt(x**2 + z**2)
            bins = np.logspace(0, 4, 80)
            for zone, mask in [
                ('core',  r < R_core),
                ('inner', (r >= R_core) & (r < R_xline_inner)),
                ('xline', (r >= R_xline_inner) & (r <= R_xline_outer)),
                ('spot',  r >= R_spot_inner),
            ]:
                if mask.any():
                    h, _ = np.histogram(KE_keV[mask], bins=bins, weights=w[mask])
                else:
                    h = np.zeros(len(bins)-1)
                out[f'spectrum_{zone}'] = h.astype(np.float32)
            out['spectrum_bins'] = bins

    np.savez(snapshot_cache_path, **out)
    print(f'  -> {snapshot_cache_path}')
    return out

def read_fusion_csv():
    csv_path = os.path.join(run_dir, 'fusion_rate_power_by_iter.csv')
    if not os.path.exists(csv_path):
        print(f'WARNING: {csv_path} not found')
        return None
    out = {
        'time_ps': [], 'fast_frac': [], 'fusion_power': [],
        'cum_energy_j': [], 'cum_alpha': [], 'gain': [],
    }
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            try:
                t = float(r['time_ps'])
                if not np.isfinite(t):
                    continue
                out['time_ps'].append(t)
                out['fast_frac'].append(float(r.get('fast_fraction_gt_500kev_p11b', 'nan')))
                out['fusion_power'].append(float(r.get('fusion_power_p11b_w', 'nan')))
                out['cum_energy_j'].append(float(r.get('cumulative_fusion_energy_j', 'nan')))
                out['cum_alpha'].append(float(r.get('cum_alpha_yield_p11b', 'nan')))
                out['gain'].append(float(r.get('gain_vs_laser_energy', 'nan')))
            except (ValueError, KeyError):
                continue
    for k in out:
        out[k] = np.asarray(out[k])
    return out

# ─── DRIVE: extract or load from cache ─────────────────────────────────────
zone_data = None
field_ts_data = None
snap_data = None
fusion_csv = read_fusion_csv()

if args.from_cache:
    print('LOADING from cache...')
    if os.path.exists(zone_cache_path):
        zone_data = dict(np.load(zone_cache_path))
        print(f'  loaded {zone_cache_path}')
    if os.path.exists(field_ts_cache_path):
        field_ts_data = dict(np.load(field_ts_cache_path))
        print(f'  loaded {field_ts_cache_path}')
    if os.path.exists(snapshot_cache_path):
        snap_data = dict(np.load(snapshot_cache_path))
        sat_iter = int(snap_data['sat_iter'])
        sat_t_ps = float(snap_data['sat_t_ps'])
        print(f'  loaded {snapshot_cache_path}')
else:
    t0 = time_module.time()
    zone_data     = extract_zone_timeseries()
    field_ts_data = extract_field_timeseries()
    snap_data     = extract_snapshot_data()
    print(f'Data extraction took {time_module.time()-t0:.1f} sec')
    print(f'Cache files written to {cache_dir}')

# ─── PANELS ────────────────────────────────────────────────────────────────
def panel_1_b_magnitude(ax):
    if snap_data is None:
        ax.text(0.5, 0.5, 'no snapshot data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, '|B| field (n/a)')
        return
    Bmag = np.sqrt(snap_data['Bx']**2 + snap_data['By']**2 + snap_data['Bz']**2)
    extent = snap_data['extent_mm']
    vmax = np.percentile(Bmag, 99)
    im = ax.imshow(Bmag.T, origin='lower', cmap=CMAP_DENSITY,
                   vmin=0, vmax=vmax, extent=extent, aspect='equal')
    overlay_geometry(ax)
    ax.set_xlabel('x (mm)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('z (mm)', color=MUTED_COLOR, fontsize=9)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=7)
    cb.set_label('|B| (T)', color=MUTED_COLOR, fontsize=8)
    style_axes(ax, f'|B| field at t = {sat_t_ps:.0f} ps')

def panel_2_streamlines(ax):
    if snap_data is None:
        ax.text(0.5, 0.5, 'no snapshot data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'B-streamlines + Jy (n/a)')
        return
    Bx = snap_data['Bx']
    Bz = snap_data['Bz']
    Jy = snap_data['Jy']
    extent = snap_data['extent_mm']
    nx, nz = Bx.shape
    x = np.linspace(extent[0], extent[1], nx)
    z = np.linspace(extent[2], extent[3], nz)
    vmax_j = np.percentile(np.abs(Jy), 95) if Jy.any() else 0
    if vmax_j > 0:
        im = ax.imshow(Jy.T, origin='lower', cmap=CMAP_BIPOLAR,
                       vmin=-vmax_j, vmax=vmax_j, extent=extent, aspect='equal')
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=7)
        cb.set_label('Jy (A/m²)', color=MUTED_COLOR, fontsize=8)
    try:
        # streamplot needs 1D x and z arrays, and U/V shaped (len(z), len(x))
        # Bx and Bz are stored as (nx, nz) indexed [i_x, i_z], so transpose to (nz, nx)
        ax.streamplot(x, z, Bx.T, Bz.T, color=TEXT_COLOR, linewidth=0.5,
                     density=1.5, arrowsize=0.8)
    except Exception as e:
        print(f'  streamplot failed: {e}')
    overlay_geometry(ax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('x (mm)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('z (mm)', color=MUTED_COLOR, fontsize=9)
    style_axes(ax, f'B-topology + Jy at t = {sat_t_ps:.0f} ps')

def panel_3_density(ax):
    if snap_data is None:
        ax.text(0.5, 0.5, 'no snapshot data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Plasma density (n/a)')
        return
    rho_abs = np.abs(snap_data['rho'])
    extent = snap_data['extent_mm']
    if (rho_abs > 0).any():
        rho_max = float(np.percentile(rho_abs[rho_abs > 0], 99))
        rho_min = max(rho_max * 1e-3, 1e-30)
        im = ax.imshow(rho_abs.T, origin='lower', cmap=CMAP_DENSITY,
                       norm=LogNorm(vmin=rho_min, vmax=rho_max),
                       extent=extent, aspect='equal')
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=7)
        cb.set_label('|ρ| (C/m³)', color=MUTED_COLOR, fontsize=8)
    overlay_geometry(ax)
    ax.set_xlabel('x (mm)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('z (mm)', color=MUTED_COLOR, fontsize=9)
    style_axes(ax, f'Plasma density at t = {sat_t_ps:.0f} ps')

def panel_4_particle_scatter(ax):
    if snap_data is None or 'scatter_x' not in snap_data:
        ax.text(0.5, 0.5, 'no particle data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Particle KE map (n/a)')
        return
    x_p  = snap_data['scatter_x']
    z_p  = snap_data['scatter_z']
    KE_p = snap_data['scatter_KE']
    sc = ax.scatter(x_p*1e3, z_p*1e3, c=KE_p, cmap=CMAP_KE,
                    s=0.5, alpha=0.6,
                    norm=LogNorm(vmin=max(1, np.percentile(KE_p, 5)),
                                 vmax=max(100, np.percentile(KE_p, 99.5))))
    extent = [-LX_M*0.5*1e3, LX_M*0.5*1e3, -LZ_M*0.5*1e3, LZ_M*0.5*1e3]
    overlay_geometry(ax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('z (mm)', color=MUTED_COLOR, fontsize=9)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=7)
    cb.set_label('KE (keV)', color=MUTED_COLOR, fontsize=8)
    p_t_ps = float(snap_data.get('p_sat_t_ps', sat_t_ps))
    style_axes(ax, f'Proton KE distribution at t = {p_t_ps:.0f} ps')

def panel_5_spectrum(ax):
    if snap_data is None or 'spectrum_bins' not in snap_data:
        ax.text(0.5, 0.5, 'no spectrum data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Energy spectrum (n/a)')
        return
    bins = snap_data['spectrum_bins']
    centers = 0.5*(bins[1:] + bins[:-1])
    for zone, color in ZONE_COLORS.items():
        h = snap_data.get(f'spectrum_{zone}', None)
        if h is not None and h.sum() > 0:
            h_norm = h / (h.sum() * np.diff(bins))
            label = {'core': f'core (r<{R_core*1e6:.0f}μm)',
                     'inner': 'inner', 'xline': 'X-line', 'spot': 'spot'}[zone]
            ax.step(centers, h_norm, where='mid', color=color, lw=1.5,
                    label=label, alpha=0.85)
    ax.axvline(500, color=ACCENT_YELLOW, ls='--', lw=0.8, alpha=0.7,
               label='500 keV')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('KE (keV)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('weighted PDF', color=MUTED_COLOR, fontsize=9)
    ax.legend(loc='best', fontsize=7, framealpha=0.7,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, f'Energy spectrum by zone (t={sat_t_ps:.0f} ps)')

def panel_6_zone_e95_bar(ax):
    if zone_data is None or len(zone_data['times_ps']) == 0:
        ax.text(0.5, 0.5, 'no zone data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Zone E95 comparison (n/a)')
        return
    last_idx = -1
    e95_vals = [
        float(zone_data['E95_core'][last_idx]),
        float(zone_data['E95_inner'][last_idx]),
        float(zone_data['E95_xline'][last_idx]),
        float(zone_data['E95_spot'][last_idx]),
    ]
    zones = ['core', 'inner', 'xline', 'spot']
    colors = [ZONE_COLORS[z] for z in zones]
    bars = ax.bar(zones, e95_vals, color=colors, alpha=0.8, edgecolor=GRID_COLOR)
    for bar, val in zip(bars, e95_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom',
                color=TEXT_COLOR, fontsize=8)
    ax.set_ylabel('E95th (keV)', color=MUTED_COLOR, fontsize=9)
    ax.set_xlabel('Zone', color=MUTED_COLOR, fontsize=9)
    ax.grid(True, axis='y', alpha=0.2, color=GRID_COLOR)
    t_final = float(zone_data['times_ps'][last_idx])
    style_axes(ax, f'95th-pct KE per zone (t={t_final:.0f} ps)')

def panel_7_fast_frac_fusion_power(ax):
    if fusion_csv is None or len(fusion_csv['time_ps']) == 0:
        ax.text(0.5, 0.5, 'no CSV data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Fast-frac & fusion power (n/a)')
        return
    t = fusion_csv['time_ps']
    ax.plot(t, fusion_csv['fast_frac'], color=ACCENT_RED, lw=1.5,
            label='Fast-ion fraction')
    ax.set_xlabel('Time (ps)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('Fast-ion fraction (>500 keV)', color=ACCENT_RED, fontsize=9)
    ax.tick_params(axis='y', labelcolor=ACCENT_RED)
    fp = fusion_csv['fusion_power']
    if (np.isfinite(fp) & (fp > 0)).any():
        ax2 = ax.twinx()
        ax2.semilogy(t, np.maximum(fp, 1e-3), color=ACCENT_BLUE, lw=1.5,
                     label='Fusion power')
        ax2.set_ylabel('Fusion power (W)', color=ACCENT_BLUE, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=ACCENT_BLUE)
        ax2.set_facecolor('none')
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, 'Fast-fraction & fusion power vs time')

def panel_8_cumulative_yield(ax):
    if fusion_csv is None or len(fusion_csv['time_ps']) == 0:
        ax.text(0.5, 0.5, 'no CSV data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Cumulative fusion yield (n/a)')
        return
    t = fusion_csv['time_ps']
    cum_e = fusion_csv['cum_energy_j']
    gain  = fusion_csv['gain']
    if (np.isfinite(cum_e) & (cum_e > 0)).any():
        ax.semilogy(t, np.maximum(cum_e, 1e-30), color=ACCENT_GREEN, lw=2,
                    label='Cumulative fusion energy (J)')
        finite = np.isfinite(cum_e) & (cum_e > 0)
        if finite.any():
            t_last = t[finite][-1]
            e_last = cum_e[finite][-1]
            ax.annotate(f'{e_last:.2e} J',
                        xy=(t_last, e_last),
                        xytext=(0.65, 0.3), textcoords='axes fraction',
                        color=ACCENT_GREEN, fontsize=10,
                        arrowprops=dict(arrowstyle='->', color=ACCENT_GREEN, lw=0.5))
    ax.set_xlabel('Time (ps)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('Cumulative fusion energy (J)', color=ACCENT_GREEN, fontsize=9)
    ax.tick_params(axis='y', labelcolor=ACCENT_GREEN)
    if (np.isfinite(gain) & (gain > 0)).any():
        ax2 = ax.twinx()
        ax2.semilogy(t, np.maximum(gain, 1e-30), color=ACCENT_YELLOW, lw=1.5,
                     ls='--', label='Gain vs laser')
        ax2.set_ylabel('Gain (Q vs 5J laser)', color=ACCENT_YELLOW, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=ACCENT_YELLOW)
        ax2.set_facecolor('none')
        ax2.axhline(1.0, color=ACCENT_YELLOW, ls=':', alpha=0.5)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, 'Cumulative fusion yield + gain')

def panel_9_zone_e95_evolution(ax):
    if zone_data is None or len(zone_data['times_ps']) == 0:
        ax.text(0.5, 0.5, 'no zone data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Zone E95 evolution (n/a)')
        return
    t = zone_data['times_ps']
    for zone, color in ZONE_COLORS.items():
        e95 = zone_data[f'E95_{zone}']
        ax.plot(t, e95, color=color, lw=1.5, label=zone, alpha=0.9)
    ax.set_xlabel('Time (ps)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('E95th (keV)', color=MUTED_COLOR, fontsize=9)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=7, framealpha=0.7,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, 'Zone-resolved E95 vs time')

def panel_10_centre_spot_ratio(ax):
    if zone_data is None or len(zone_data['times_ps']) == 0:
        ax.text(0.5, 0.5, 'no zone data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Centre/spot ratio (n/a)')
        return
    t = zone_data['times_ps']
    e95_core = zone_data['E95_core']
    e95_spot = zone_data['E95_spot']
    e95_xline = zone_data['E95_xline']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_cs = np.where(e95_spot > 0, e95_core / e95_spot, np.nan)
        ratio_xs = np.where(e95_spot > 0, e95_xline / e95_spot, np.nan)
    ax.plot(t, ratio_cs, color=ZONE_COLORS['core'], lw=2, label='core/spot')
    ax.plot(t, ratio_xs, color=ZONE_COLORS['xline'], lw=2, label='X-line/spot')
    ax.axhline(1.0, color=MUTED_COLOR, ls=':', alpha=0.5)
    ax.set_xlabel('Time (ps)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('E95th ratio', color=MUTED_COLOR, fontsize=9)
    ax.legend(loc='best', fontsize=8, framealpha=0.7,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, 'Reconnection signature: KE ratios')

def panel_11_zone_counts(ax):
    if zone_data is None or len(zone_data['times_ps']) == 0:
        ax.text(0.5, 0.5, 'no zone data', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, 'Zone counts (n/a)')
        return
    t = zone_data['times_ps']
    for zone, color in ZONE_COLORS.items():
        N = zone_data[f'N_{zone}']
        ax.semilogy(t, np.maximum(N, 1), color=color, lw=1.5,
                    label=zone, alpha=0.9)
    ax.set_xlabel('Time (ps)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('Particle count (weighted)', color=MUTED_COLOR, fontsize=9)
    ax.legend(loc='best', fontsize=7, framealpha=0.7,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, 'Particle counts per zone vs time')

def panel_12_b_field_evolution(ax):
    if field_ts_data is None or len(field_ts_data['times_ps']) == 0:
        ax.text(0.5, 0.5, 'no field timeseries', transform=ax.transAxes,
                ha='center', va='center', color=MUTED_COLOR)
        style_axes(ax, '|B| evolution (n/a)')
        return
    t = field_ts_data['times_ps']
    ax.plot(t, field_ts_data['B_max'], color=ACCENT_RED, lw=1.5, label='|B|_max')
    ax.plot(t, field_ts_data['B_xline_max'], color=ACCENT_GREEN, lw=1.5,
            label='|B|_xline_max', ls='--')
    ax.plot(t, field_ts_data['B_xline_min'], color=ACCENT_GREEN, lw=1.5,
            label='|B|_xline_min', ls=':')
    ax.plot(t, field_ts_data['B_mean'], color=ACCENT_BLUE, lw=1, label='|B|_mean',
            alpha=0.7)
    ax.axhline(B_SEED_T, color=MUTED_COLOR, ls=':', alpha=0.5,
               label=f'seed = {B_SEED_T:.0f} T')
    ax.set_xlabel('Time (ps)', color=MUTED_COLOR, fontsize=9)
    ax.set_ylabel('|B| (T)', color=MUTED_COLOR, fontsize=9)
    ax.legend(loc='best', fontsize=7, framealpha=0.7,
              facecolor=PANEL_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    style_axes(ax, '|B| evolution: peak, X-line, mean')

# ─── COMPOSITES ────────────────────────────────────────────────────────────
def make_snapshot_composite():
    print('Generating snapshot composite (panels 1-6)...')
    fig = plt.figure(figsize=(18, 12))
    style_figure(fig)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.32,
                           left=0.05, right=0.97, top=0.94, bottom=0.06)
    fig.suptitle('Spatial diagnostics at saturation',
                 color=TEXT_COLOR, fontsize=14, y=0.985)
    panel_1_b_magnitude(fig.add_subplot(gs[0, 0]))
    panel_2_streamlines(fig.add_subplot(gs[0, 1]))
    panel_3_density(fig.add_subplot(gs[0, 2]))
    panel_4_particle_scatter(fig.add_subplot(gs[1, 0]))
    panel_5_spectrum(fig.add_subplot(gs[1, 1]))
    panel_6_zone_e95_bar(fig.add_subplot(gs[1, 2]))
    out = os.path.join(output_dir, 'composite_snapshots.png')
    plt.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f'  -> {out}')

def make_evolution_composite():
    print('Generating evolution composite (panels 7-12)...')
    fig = plt.figure(figsize=(18, 12))
    style_figure(fig)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.96, top=0.94, bottom=0.07)
    fig.suptitle('Time-resolved diagnostics',
                 color=TEXT_COLOR, fontsize=14, y=0.985)
    panel_7_fast_frac_fusion_power(fig.add_subplot(gs[0, 0]))
    panel_8_cumulative_yield(fig.add_subplot(gs[0, 1]))
    panel_9_zone_e95_evolution(fig.add_subplot(gs[0, 2]))
    panel_10_centre_spot_ratio(fig.add_subplot(gs[1, 0]))
    panel_11_zone_counts(fig.add_subplot(gs[1, 1]))
    panel_12_b_field_evolution(fig.add_subplot(gs[1, 2]))
    out = os.path.join(output_dir, 'composite_evolution.png')
    plt.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f'  -> {out}')

def make_individual(panel_func, filename, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    style_figure(fig)
    panel_func(ax)
    out = os.path.join(individual_dir, filename)
    plt.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f'  -> {out}')

def make_individuals():
    print('\nGenerating individual panels...')
    make_individual(panel_1_b_magnitude,            'snap_01_b_magnitude.png')
    make_individual(panel_2_streamlines,            'snap_02_b_streamlines_jy.png')
    make_individual(panel_3_density,                'snap_03_plasma_density.png')
    make_individual(panel_4_particle_scatter,       'snap_04_particle_ke.png', figsize=(8, 7))
    make_individual(panel_5_spectrum,               'snap_05_energy_spectrum.png')
    make_individual(panel_6_zone_e95_bar,           'snap_06_zone_e95_bar.png')
    make_individual(panel_7_fast_frac_fusion_power, 'evol_07_fast_frac_fusion_power.png', figsize=(10, 5))
    make_individual(panel_8_cumulative_yield,       'evol_08_cumulative_yield.png', figsize=(10, 5))
    make_individual(panel_9_zone_e95_evolution,     'evol_09_zone_e95.png', figsize=(10, 5))
    make_individual(panel_10_centre_spot_ratio,     'evol_10_centre_spot_ratio.png', figsize=(10, 5))
    make_individual(panel_11_zone_counts,           'evol_11_zone_counts.png', figsize=(10, 5))
    make_individual(panel_12_b_field_evolution,     'evol_12_b_evolution.png', figsize=(10, 5))

def make_animation():
    if ts_field is None:
        print('SKIP animation: no field timeseries (running from cache)')
        return
    print('\nGenerating |B| evolution animation...')
    try:
        import imageio.v2 as imageio
    except ImportError:
        print('  imageio not installed: pip install "imageio[ffmpeg]"')
        return
    frames_dir = os.path.join(output_dir, 'animation_frames')
    os.makedirs(frames_dir, exist_ok=True)
    if field_ts_data is not None:
        vmax_global = float(np.percentile(field_ts_data['B_max'], 95))
    else:
        vmax_global = 200.0
    print(f'  Using global |B| vmax = {vmax_global:.1f} T')
    frame_paths = []
    for i, it in enumerate(iterations):
        fig, ax = plt.subplots(figsize=(8, 7))
        style_figure(fig)
        Bx, info = get_field_2d('B', 'x', it)
        By, _    = get_field_2d('B', 'y', it)
        Bz, _    = get_field_2d('B', 'z', it)
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        extent = get_field_extent_mm(info)
        im = ax.imshow(Bmag.T, origin='lower', cmap=CMAP_DENSITY,
                       vmin=0, vmax=vmax_global, extent=extent, aspect='equal')
        overlay_geometry(ax)
        ax.set_xlabel('x (mm)', color=MUTED_COLOR)
        ax.set_ylabel('z (mm)', color=MUTED_COLOR)
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label('|B| (T)', color=MUTED_COLOR)
        cb.ax.tick_params(colors=MUTED_COLOR)
        t_ps = times_s[i] * 1e12
        style_axes(ax, f't = {t_ps:.0f} ps  (frame {i+1}/{len(iterations)})')
        fp = os.path.join(frames_dir, f'frame_{i:04d}.png')
        plt.savefig(fp, dpi=120, bbox_inches='tight',
                    facecolor=BG_COLOR, edgecolor='none')
        plt.close(fig)
        frame_paths.append(fp)
        if (i+1) % 10 == 0:
            print(f'  rendered {i+1}/{len(iterations)} frames')
    out_mp4 = os.path.join(output_dir, 'b_evolution.mp4')
    try:
        with imageio.get_writer(out_mp4, fps=8, codec='h264') as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print(f'  -> {out_mp4}')
    except Exception as e:
        print(f'  MP4 failed ({e}); falling back to GIF')
        out_gif = os.path.join(output_dir, 'b_evolution.gif')
        with imageio.get_writer(out_gif, mode='I', duration=0.125) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print(f'  -> {out_gif}')

# ─── DRIVE ─────────────────────────────────────────────────────────────────
if not args.no_composites:
    make_snapshot_composite()
    make_evolution_composite()
if not args.no_individuals:
    make_individuals()
if args.animate:
    make_animation()

# ─── SUMMARY ───────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('DONE')
print('='*72)
print(f'Output:  {output_dir}')
print()
print('Composites:')
print('  composite_snapshots.png  (panels 1-6: spatial)')
print('  composite_evolution.png  (panels 7-12: time-resolved)')
print()
print('Individual panels: figures/individual/')
print('  Snapshot panels:')
print('    01 |B| field magnitude')
print('    02 B-streamlines + Jy heatmap')
print('    03 Plasma density')
print('    04 Particle KE scatter')
print('    05 Energy spectrum by zone')
print('    06 Zone E95 bar comparison')
print('  Evolution panels:')
print('    07 Fast-fraction + fusion power')
print('    08 Cumulative fusion yield + gain')
print('    09 Zone-resolved E95 evolution')
print('    10 Centre/spot ratio (reconnection signature)')
print('    11 Particle counts per zone')
print('    12 |B| evolution (peak, X-line, mean)')
if args.animate:
    print()
    print('Animation:')
    print('  b_evolution.mp4')
print()
print('Cache files (preserve to regenerate after particle deletion):')
print('  cache/zone_timeseries.npz')
print('  cache/snapshot_data.npz')
print('  cache/field_timeseries.npz')
print()
print('After verifying figures look correct, you can safely delete')
print(f'  {run_dir}/particles/')
print('and any panel can be regenerated from cache with: --from-cache')
print('='*72)