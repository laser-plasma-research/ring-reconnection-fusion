#!/usr/bin/env python3
"""
pb11_fusion_diagnostics.py - Static fusion-relevant diagnostic plots for Paper 1.

Produces three publication-ready static figures that complement the existing
visualize_all + phase_analysis suite by focusing on FUSION-RELEVANT particle
populations.

OUTPUTS:
    figures/fusion_grade_fractions_vs_time.png  - count of particles above
        each energy threshold (100, 300, 500, 1000 keV) vs time
    figures/peak_snapshot_fusion_grade.png      - spatial map of high-energy
        particles at peak acceleration time
    figures/energy_distribution_by_zone_at_peak.png - histogram of particle
        energies by zone (core, inner, xline, outer) at peak time
    fusion_diagnostics.csv                       - time-series CSV with
        fusion-grade counts per energy threshold

USAGE:
    python pb11_fusion_diagnostics.py --dir runs/<run_name>
    python pb11_fusion_diagnostics.py --dir runs/<run_name> --peak-time 54

OPTIONS:
    --peak-time T      Peak acceleration time in ps (auto-detect if not given)
    --species name     Particle species (default proton)
    --thresholds list  Comma-separated energy thresholds in keV
                       (default: 100,300,500,1000)
    --dpi N            Figure DPI (default 150)
"""

import os, sys, argparse, configparser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3

# Dark theme colors (match visualize_all.py)
BG_COLOR     = '#0a0a0a'
TEXT_COLOR   = '#e6e6e6'
MUTED_COLOR  = '#999999'
ACCENT       = '#5dadec'

# Energy bin colors
COLOR_COLD   = '#3b82f6'  # blue
COLOR_WARM   = '#facc15'  # yellow
COLOR_FUSION = '#fb923c'  # orange
COLOR_HOT    = '#ef4444'  # red

# Zone colors (match phase_analysis)
ZONE_COLORS = {
    'core':  '#fb923c',
    'inner': '#facc15',
    'xline': '#5dadec',
    'outer': '#94a3b8',
}

CMAP_KE = 'plasma'

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir', required=True, help='Run directory')
parser.add_argument('--peak-time', type=float, default=None,
                    help='Peak time in ps (auto-detect if not given)')
parser.add_argument('--species', default='proton')
parser.add_argument('--thresholds', default='100,1000,5000,20000,100000',
                    help='Energy thresholds in keV (default spans full range)')
parser.add_argument('--dpi', type=int, default=150)
parser.add_argument('--max-particles-per-dump', type=int, default=100000,
                    help='Cap particles read per dump (default 100k for speed). '
                         'Counts get scaled by total/sample for fusion-grade estimates.')
parser.add_argument('--max-snapshot-particles', type=int, default=200000,
                    help='Cap particles for peak snapshot (default 200000)')
args = parser.parse_args()

run_dir = args.dir.rstrip('/')
thresholds = [float(t) for t in args.thresholds.split(',')]

# Load metadata
meta_path = os.path.join(run_dir, 'run_meta.txt')
meta_lines = open(meta_path).read().splitlines()
section_start = next((i for i, l in enumerate(meta_lines) if l.strip().startswith('[')), None)
cp = configparser.ConfigParser()
cp.read_string('\n'.join(meta_lines[section_start:]))

N_SPOTS       = int(cp['geometry']['n_spots'])
RING_RADIUS_M = float(cp['geometry']['ring_radius_m'])
SPOT_RADIUS_M = float(cp['geometry']['spot_radius_m'])
R_xline = RING_RADIUS_M * np.cos(np.pi / N_SPOTS)

mass_kg = B11_MASS_KG if args.species == 'boron11' else PROTON_MASS_KG

# Zone definitions matching phase_analysis
ZONE_CORE_R   = 600e-6
ZONE_INNER_R  = 1500e-6  # between core and ring spots
ZONE_XLINE_R  = R_xline
ZONE_RING_R   = RING_RADIUS_M

# Load particle dumps
try:
    import openpmd_viewer as ov
except ImportError:
    print('Installing openpmd-viewer...')
    os.system('pip install openpmd-viewer --break-system-packages')
    import openpmd_viewer as ov

particle_dir_candidates = [
    os.path.join(run_dir, 'particles_early'),
    os.path.join(run_dir, 'particles'),
    os.path.join(run_dir, 'diags', 'particles_early'),
    os.path.join(run_dir, 'diags', 'particles'),
]
particle_dir = next((p for p in particle_dir_candidates if os.path.isdir(p)), None)
if particle_dir is None:
    print('ERROR: no particle dumps found', file=sys.stderr)
    sys.exit(1)

ts = ov.OpenPMDTimeSeries(particle_dir)
iterations = list(ts.iterations)
times_s = np.asarray(ts.t)
times_ps = times_s * 1e12

out_dir = os.path.join(run_dir, 'figures')
os.makedirs(out_dir, exist_ok=True)

print('=' * 92)
print('  FUSION DIAGNOSTICS - {}'.format(run_dir))
print('=' * 92)
print('  Particle dumps:  {}'.format(particle_dir))
print('  Snapshots:       {}'.format(len(iterations)))
print('  Time range:      {:.2f} -> {:.2f} ps'.format(times_ps[0], times_ps[-1]))
print('  Thresholds:      {} keV'.format(thresholds))
print('  Species:         {}'.format(args.species))
print('  Output:          {}'.format(out_dir))
print('=' * 92)

# Helper: compute KE from u
def ke_keV_from_u(ux, uy, uz, mass=mass_kg):
    u2 = ux*ux + uy*uy + uz*uz
    gamma = np.sqrt(1.0 + u2)
    return (gamma - 1.0) * mass * C * C / KEV_TO_J

# Helper: read particles with optional subsampling
def get_particles_subsampled(it, fields, max_n=None):
    """Read particle fields with optional fixed-seed subsampling.
    Returns (data_tuple, n_total, n_sampled).
    Counts scaled by total/sampled if subsampling occurred."""
    data = ts.get_particle(fields, species=args.species, iteration=it)
    if not isinstance(data, tuple) and not isinstance(data, list):
        data = (data,)
    n_total = len(data[0])
    if max_n is None or n_total <= max_n:
        return data, n_total, n_total
    np.random.seed(42)
    idx = np.random.choice(n_total, max_n, replace=False)
    sampled = tuple(d[idx] for d in data)
    return sampled, n_total, max_n


# Style helpers
def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED_COLOR)

def overlay_geometry(ax, scale=1e3):
    """Overlay ring + X-line markers on (x_mm, z_mm) plot."""
    theta = np.linspace(0, 2*np.pi, 100)
    ring_x = RING_RADIUS_M * scale * np.cos(theta)
    ring_z = RING_RADIUS_M * scale * np.sin(theta)
    ax.plot(ring_x, ring_z, color=MUTED_COLOR, linestyle='--', linewidth=0.6, alpha=0.5)
    for k in range(N_SPOTS):
        angle = 2*np.pi*k/N_SPOTS + np.pi/N_SPOTS
        xx = R_xline * scale * np.cos(angle)
        zz = R_xline * scale * np.sin(angle)
        ax.plot(xx, zz, marker='+', color=ACCENT, markersize=10, alpha=0.7,
                markeredgewidth=1.5)


# ============================================================================
# PASS 1: Compute time series (fusion-grade counts per threshold)
# ============================================================================
print()
print('Pass 1: computing fusion-grade fractions across all timesteps...')
print('  (subsampling to {} particles per dump for speed; counts are estimated)'.format(
    args.max_particles_per_dump))
print()
sys.stdout.flush()
counts_by_threshold = {th: np.zeros(len(iterations)) for th in thresholds}
total_counts = np.zeros(len(iterations))
peak_KE_per_frame = np.zeros(len(iterations))
mean_KE_per_frame = np.zeros(len(iterations))

# Accumulate energy spectrum per timestep for heatmap
N_E_BINS = 100
spectrum_E_edges = np.logspace(0, 6, N_E_BINS + 1)  # 1 keV to 1 GeV
spectrum_per_t = np.zeros((len(iterations), N_E_BINS))

for i, it in enumerate(iterations):
    try:
        (ux, uy, uz), n_total, n_sampled = get_particles_subsampled(
            it, ['ux', 'uy', 'uz'], max_n=args.max_particles_per_dump)
    except Exception as e:
        print('  iter {}: read failed ({})'.format(it, e))
        sys.stdout.flush()
        continue
    KE = ke_keV_from_u(ux, uy, uz)
    total_counts[i] = n_total
    if len(KE) > 0:
        peak_KE_per_frame[i] = float(np.percentile(KE, 99))
        mean_KE_per_frame[i] = float(np.mean(KE))
        scale = n_total / n_sampled
        for th in thresholds:
            counts_by_threshold[th][i] = int(np.sum(KE >= th) * scale)
        # Histogram for spectrum heatmap
        hist, _ = np.histogram(KE, bins=spectrum_E_edges)
        spectrum_per_t[i] = hist * scale
    print('  {:>3d}/{}: t={:7.2f} ps, N_total={:,}, N_sampled={:,}, peak={:6.0f} keV, mean={:6.0f} keV'.format(
        i+1, len(iterations), times_ps[i], int(n_total), int(n_sampled),
        peak_KE_per_frame[i], mean_KE_per_frame[i]))
    sys.stdout.flush()

# Auto-detect peak time if not specified
if args.peak_time is None:
    peak_idx = int(np.argmax(peak_KE_per_frame))
    peak_time_ps = times_ps[peak_idx]
    print()
    print('  Auto-detected peak time: t = {:.2f} ps (peak E99 = {:.1f} keV)'.format(
        peak_time_ps, peak_KE_per_frame[peak_idx]))
else:
    peak_time_ps = args.peak_time
    peak_idx = int(np.argmin(np.abs(times_ps - peak_time_ps)))
    print()
    print('  Using specified peak time: t = {:.2f} ps'.format(peak_time_ps))

# Write CSV
csv_path = os.path.join(run_dir, 'fusion_diagnostics.csv')
with open(csv_path, 'w') as f:
    header = ['step', 't_ps', 'total_count', 'peak_KE_keV_p99', 'mean_KE_keV']
    for th in thresholds:
        header.append('N_KE_above_{:.0f}_keV'.format(th))
    f.write(','.join(header) + '\n')
    for i, it in enumerate(iterations):
        row = ['{}'.format(it), '{:.4f}'.format(times_ps[i]),
               '{:.0f}'.format(total_counts[i]),
               '{:.4f}'.format(peak_KE_per_frame[i]),
               '{:.4f}'.format(mean_KE_per_frame[i])]
        for th in thresholds:
            row.append('{:.0f}'.format(counts_by_threshold[th][i]))
        f.write(','.join(row) + '\n')
print('  CSV written: {}'.format(csv_path))


# ============================================================================
# FIGURE 1: Fusion-grade fractions vs time (PERCENTAGE not absolute count)
# ============================================================================
print()
print('Generating Figure 1: fusion_grade_fractions_vs_time.png')
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
threshold_colors = ['#3b82f6', '#facc15', '#fb923c', '#ef4444', '#a855f7', '#ec4899']
for j, th in enumerate(thresholds):
    color = threshold_colors[j % len(threshold_colors)]
    # Fraction in PERCENT
    frac_pct = 100.0 * counts_by_threshold[th] / np.maximum(total_counts, 1)
    if th < 1000:
        label = 'KE > {:.0f} keV'.format(th)
    elif th < 1000000:
        label = 'KE > {:.1f} MeV'.format(th/1000)
    else:
        label = 'KE > {:.0f} MeV'.format(th/1000)
    ax.plot(times_ps, frac_pct, color=color, linewidth=2.0,
            label=label, marker='o', markersize=4, alpha=0.85)
ax.axvline(peak_time_ps, color='white', linestyle=':', alpha=0.4,
           label='peak (t={:.0f} ps)'.format(peak_time_ps))
ax.set_ylim(-2, 105)
style_axes(ax, title='Fraction of particles above each energy threshold (vs time)',
           xlabel='Time (ps)', ylabel='Fraction (%)')
leg = ax.legend(loc='center right', facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
                labelcolor=TEXT_COLOR, fontsize=10)
plt.tight_layout()
out_path = os.path.join(out_dir, 'fusion_grade_fractions_vs_time.png')
plt.savefig(out_path, dpi=args.dpi, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print('  -> {}'.format(out_path))


# ============================================================================
# FIGURE 1b: Spectrum evolution heatmap (time vs energy)
# ============================================================================
print()
print('Generating Figure 1b: spectrum_evolution_heatmap.png')

# Build time edges (centers + extrapolated endpoints) for pcolormesh
if len(times_ps) >= 2:
    dt = times_ps[1] - times_ps[0]
    time_edges = np.zeros(len(times_ps) + 1)
    time_edges[0] = times_ps[0] - dt / 2
    time_edges[1:-1] = (times_ps[:-1] + times_ps[1:]) / 2
    time_edges[-1] = times_ps[-1] + (times_ps[-1] - times_ps[-2]) / 2
else:
    time_edges = np.array([times_ps[0] - 1, times_ps[0] + 1])

# Prepare data for heatmap (replace zeros with NaN for log scale)
spectrum_safe = np.where(spectrum_per_t > 0, spectrum_per_t, np.nan)

fig, axes = plt.subplots(2, 1, figsize=(11, 9), facecolor=BG_COLOR)

# Top: raw counts
ax1 = axes[0]
im1 = ax1.pcolormesh(time_edges, spectrum_E_edges, spectrum_safe.T,
                      cmap='inferno', norm=LogNorm(vmin=1, vmax=np.nanmax(spectrum_safe)),
                      shading='flat')
ax1.set_yscale('log')
ax1.set_ylim(1, 1e6)
cb1 = plt.colorbar(im1, ax=ax1, pad=0.02)
cb1.set_label('estimated particle count', color=MUTED_COLOR)
cb1.ax.tick_params(colors=MUTED_COLOR)
for spine in cb1.ax.spines.values():
    spine.set_edgecolor(MUTED_COLOR)
# Reference lines
ax1.axhline(150, color='cyan', linestyle=':', alpha=0.6, label='Coulomb barrier (150 keV)')
ax1.axhline(675, color='lime', linestyle='--', alpha=0.6, label='peak fusion σ (675 keV)')
ax1.legend(loc='lower right', facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
           labelcolor=TEXT_COLOR, fontsize=8)
style_axes(ax1, title='Energy distribution evolution (raw count)',
           xlabel='Time (ps)', ylabel='Kinetic Energy (keV)')

# Bottom: normalized per timestep (highlights distribution shape)
spectrum_normalized = spectrum_per_t / np.maximum(spectrum_per_t.sum(axis=1, keepdims=True), 1)
spectrum_norm_safe = np.where(spectrum_normalized > 0, spectrum_normalized, np.nan)
ax2 = axes[1]
im2 = ax2.pcolormesh(time_edges, spectrum_E_edges, spectrum_norm_safe.T,
                      cmap='viridis', norm=LogNorm(vmin=1e-5, vmax=np.nanmax(spectrum_norm_safe)),
                      shading='flat')
ax2.set_yscale('log')
ax2.set_ylim(1, 1e6)
cb2 = plt.colorbar(im2, ax=ax2, pad=0.02)
cb2.set_label('fraction at each timestep', color=MUTED_COLOR)
cb2.ax.tick_params(colors=MUTED_COLOR)
for spine in cb2.ax.spines.values():
    spine.set_edgecolor(MUTED_COLOR)
ax2.axhline(150, color='cyan', linestyle=':', alpha=0.6)
ax2.axhline(675, color='lime', linestyle='--', alpha=0.6)
style_axes(ax2, title='Normalized distribution evolution (shape over time)',
           xlabel='Time (ps)', ylabel='Kinetic Energy (keV)')

plt.tight_layout()
out_path = os.path.join(out_dir, 'spectrum_evolution_heatmap.png')
plt.savefig(out_path, dpi=args.dpi, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print('  -> {}'.format(out_path))


# ============================================================================
# PASS 2: Read peak snapshot for spatial figures
# ============================================================================
print()
print('Pass 2: reading peak snapshot at iteration {}...'.format(iterations[peak_idx]))
sys.stdout.flush()
try:
    (x, z, ux, uy, uz), n_total_peak, n_sampled_peak = get_particles_subsampled(
        iterations[peak_idx], ['x', 'z', 'ux', 'uy', 'uz'],
        max_n=args.max_snapshot_particles)
    KE = ke_keV_from_u(ux, uy, uz)
    n_total = len(x)
    print('  Peak snapshot: {:,} total particles, sampling {:,} for figures'.format(
        int(n_total_peak), int(n_sampled_peak)))
    sys.stdout.flush()
except Exception as e:
    print('  ERROR: failed to read peak snapshot ({})'.format(e))
    sys.exit(1)


# ============================================================================
# FIGURE 2: Peak snapshot spatial map - fusion-grade particles
# ============================================================================
print()
print('Generating Figure 2: peak_snapshot_fusion_grade.png')

# Get fusion-grade subset (KE > 500 keV)
fusion_threshold = 500.0
mask_fusion = KE >= fusion_threshold
n_fusion = int(mask_fusion.sum())
print('  Particles above {:.0f} keV at peak: {}'.format(fusion_threshold, n_fusion))

# Show fusion-grade particles in main panel + cold/warm context in faint background
fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG_COLOR)

# Background context: subsample of all particles (faint)
if n_total > 20000:
    np.random.seed(42)
    bg_idx = np.random.choice(n_total, 20000, replace=False)
else:
    bg_idx = np.arange(n_total)
ax.scatter(x[bg_idx]*1e3, z[bg_idx]*1e3, c=MUTED_COLOR, s=0.3, alpha=0.15,
           label='all particles (background)')

# Fusion-grade particles in foreground (colored by KE)
if n_fusion > 0:
    sc = ax.scatter(x[mask_fusion]*1e3, z[mask_fusion]*1e3,
                   c=KE[mask_fusion], cmap=CMAP_KE,
                   norm=LogNorm(vmin=fusion_threshold, vmax=max(KE[mask_fusion].max(), fusion_threshold*2)),
                   s=8, alpha=0.85, edgecolors='white', linewidths=0.1,
                   label='KE >= {:.0f} keV ({} particles)'.format(fusion_threshold, n_fusion))
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label('Kinetic Energy (keV)', color=MUTED_COLOR)
    cb.ax.tick_params(colors=MUTED_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_edgecolor(MUTED_COLOR)
else:
    ax.text(0.5, 0.5, 'No particles above {:.0f} keV\nat peak time'.format(fusion_threshold),
            transform=ax.transAxes, ha='center', va='center', color=TEXT_COLOR, fontsize=14)

overlay_geometry(ax)
domain_max = RING_RADIUS_M * 1.8
ax.set_xlim(-domain_max*1e3, domain_max*1e3)
ax.set_ylim(-domain_max*1e3, domain_max*1e3)
ax.set_aspect('equal')
style_axes(ax,
    title='Fusion-grade particles at peak (t = {:.0f} ps)'.format(peak_time_ps),
    xlabel='x (mm)', ylabel='z (mm)')
leg = ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
                labelcolor=TEXT_COLOR, fontsize=9)
plt.tight_layout()
out_path = os.path.join(out_dir, 'peak_snapshot_fusion_grade.png')
plt.savefig(out_path, dpi=args.dpi, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print('  -> {}'.format(out_path))


# ============================================================================
# FIGURE 3: Energy distribution by zone at peak
# ============================================================================
print()
print('Generating Figure 3: energy_distribution_by_zone_at_peak.png')

# Classify particles by zone (matching phase_analysis zones)
r = np.sqrt(x*x + z*z)
mask_core  = r < ZONE_CORE_R
mask_inner = (r >= ZONE_CORE_R) & (r < ZONE_INNER_R)
mask_xline = (r >= ZONE_INNER_R) & (r < ZONE_RING_R)
mask_outer = r >= ZONE_RING_R

zones = {
    'core':  (mask_core,  ZONE_COLORS['core'],  'Core (r < {:.0f} um)'.format(ZONE_CORE_R*1e6)),
    'inner': (mask_inner, ZONE_COLORS['inner'], 'Inner ({:.0f}-{:.0f} um)'.format(
              ZONE_CORE_R*1e6, ZONE_INNER_R*1e6)),
    'xline': (mask_xline, ZONE_COLORS['xline'], 'X-line ({:.0f}-{:.0f} um)'.format(
              ZONE_INNER_R*1e6, ZONE_RING_R*1e6)),
    'outer': (mask_outer, ZONE_COLORS['outer'], 'Outer (r >= {:.0f} um)'.format(
              ZONE_RING_R*1e6)),
}

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)

# Use log-log histograms
KE_min = max(1.0, np.percentile(KE, 1))
KE_max = max(2000.0, np.percentile(KE, 99.9) * 1.2)
bins = np.logspace(np.log10(KE_min), np.log10(KE_max), 60)

for zone_name, (mask, color, label) in zones.items():
    if mask.sum() == 0:
        continue
    ax.hist(KE[mask], bins=bins, histtype='step', linewidth=2.0, color=color,
            label='{} (N={})'.format(label, int(mask.sum())))

# Add fusion-relevant energy markers
ax.axvline(150, color=COLOR_COLD,   linestyle=':', alpha=0.5,
           label='Coulomb barrier ~150 keV')
ax.axvline(500, color=COLOR_FUSION, linestyle='--', alpha=0.5,
           label='fusion-grade 500 keV')
ax.axvline(675, color=COLOR_HOT,    linestyle='--', alpha=0.5,
           label='peak cross-section ~675 keV')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(KE_min, KE_max)
style_axes(ax,
    title='Energy distribution by zone at peak (t = {:.0f} ps)'.format(peak_time_ps),
    xlabel='Kinetic Energy (keV)', ylabel='Particle count')
leg = ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
                labelcolor=TEXT_COLOR, fontsize=9)
plt.tight_layout()
out_path = os.path.join(out_dir, 'energy_distribution_by_zone_at_peak.png')
plt.savefig(out_path, dpi=args.dpi, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print('  -> {}'.format(out_path))


# ============================================================================
# Summary
# ============================================================================
print()
print('=' * 92)
print('  FUSION DIAGNOSTICS SUMMARY')
print('=' * 92)
print('  Peak time:           {:.2f} ps (frame {} of {})'.format(
    peak_time_ps, peak_idx+1, len(iterations)))
print('  Peak E99:            {:.1f} keV'.format(peak_KE_per_frame[peak_idx]))
print('  Mean KE at peak:     {:.1f} keV'.format(mean_KE_per_frame[peak_idx]))
print('  Total particles:     {:.0f}'.format(total_counts[peak_idx]))
print()
print('  Fusion-grade counts at peak:')
for th in thresholds:
    cnt = int(counts_by_threshold[th][peak_idx])
    pct = 100 * cnt / max(total_counts[peak_idx], 1)
    print('    KE > {:>5.0f} keV:  {:>10d}  ({:>5.2f}%)'.format(th, cnt, pct))
print()
print('  Zone counts at peak:')
for zone_name, (mask, _, label) in zones.items():
    n = int(mask.sum())
    n_fusion_zone = int((mask & (KE >= 500)).sum())
    print('    {:>5s}: N={:>8d}, N(>500 keV)={:>5d}'.format(zone_name, n, n_fusion_zone))
print()
print('=' * 92)
print('  DONE')
print('=' * 92)
