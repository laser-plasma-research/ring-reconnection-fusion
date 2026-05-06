#!/usr/bin/env python3
"""
pb11_particle_animations.py - Density-based particle animations for p-11B fusion.

Generates 2D-density heatmap animations per energy bin, plus tracer and spectrum
animations. Density heatmaps reveal spatial clustering that scatter plots miss
when there are millions of particles.

OUTPUTS:
    figures/particle_density_by_energy.mp4 - 4-panel density heatmaps per bin
    figures/acceleration_tracer.mp4        - high-energy particle tracer
    figures/energy_spectrum_evolution.mp4  - spectrum histogram over time

Energy bins (default, physics-relevant for p-11B):
    Sub-Coulomb:  < 100 keV     (below fusion threshold, rare in hot plasma)
    Pre-fusion:   100-500 keV   (rising cross-section)
    Fusion-peak:  500-5000 keV  (peak fusion cross-section regime)
    Hot tail:     > 5000 keV    (relativistic, less efficient fusion)

USAGE:
    python pb11_particle_animations.py --dir runs/<run_name>

OPTIONS:
    --bins list             Bin edges in keV (default 100,500,5000)
    --grid-size N           Heatmap resolution NxN (default 160)
    --tracer-threshold KE   Tracer threshold in keV (default 1000)
    --max-tracer-particles N (default 1500)
    --max-particles-per-dump N (default 80000)
    --species name          (default 'proton')
    --fps N                 (default 8)
    --skip-density          Skip density heatmap animation
    --skip-tracer           Skip tracer animation
    --skip-spectrum         Skip spectrum animation
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

BG_COLOR     = '#0a0a0a'
TEXT_COLOR   = '#e6e6e6'
MUTED_COLOR  = '#999999'
ACCENT       = '#5dadec'

BIN_LABELS = {
    'sub_coulomb':  'Sub-Coulomb',
    'pre_fusion':   'Pre-fusion',
    'fusion_peak':  'Fusion-peak',
    'hot_tail':     'Hot tail',
}
BIN_CMAPS = {
    'sub_coulomb': 'Blues',
    'pre_fusion':  'YlGn',
    'fusion_peak': 'YlOrRd',
    'hot_tail':    'hot',
}
CMAP_KE = 'plasma'

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir', required=True)
parser.add_argument('--bins', default='100,500,5000',
                    help='Bin edges in keV (default 100,500,5000)')
parser.add_argument('--grid-size', type=int, default=160,
                    help='Heatmap resolution NxN (default 160)')
parser.add_argument('--tracer-threshold', type=float, default=1000.0)
parser.add_argument('--max-tracer-particles', type=int, default=1500)
parser.add_argument('--max-particles-per-dump', type=int, default=80000)
parser.add_argument('--species', default='proton')
parser.add_argument('--fps', type=int, default=8)
parser.add_argument('--skip-density', action='store_true')
parser.add_argument('--skip-tracer', action='store_true')
parser.add_argument('--skip-spectrum', action='store_true')
parser.add_argument('--max-snapshots', type=int, default=0)
args = parser.parse_args()

run_dir = args.dir.rstrip('/')
bin_edges = [float(b) for b in args.bins.split(',')]
if len(bin_edges) != 3:
    print('ERROR: --bins must have exactly 3 values', file=sys.stderr)
    sys.exit(1)

bins_def = {
    'sub_coulomb': (0.0,           bin_edges[0]),
    'pre_fusion':  (bin_edges[0],  bin_edges[1]),
    'fusion_peak': (bin_edges[1],  bin_edges[2]),
    'hot_tail':    (bin_edges[2],  np.inf),
}

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

if args.max_snapshots > 0 and len(iterations) > args.max_snapshots:
    step = max(1, len(iterations) // args.max_snapshots)
    iterations = iterations[::step][:args.max_snapshots]
    times_s = np.array([ts.t[list(ts.iterations).index(it)] for it in iterations])

out_dir = os.path.join(run_dir, 'figures')
os.makedirs(out_dir, exist_ok=True)
domain_max = RING_RADIUS_M * 1.8

print('=' * 92)
print('  PARTICLE DENSITY ANIMATIONS - {}'.format(run_dir))
print('=' * 92)
print('  Particle dumps:    {}'.format(particle_dir))
print('  Snapshots:         {}'.format(len(iterations)))
print('  Time range:        {:.2f} -> {:.2f} ps'.format(times_s[0]*1e12, times_s[-1]*1e12))
print('  Species:           {}'.format(args.species))
print('  Energy bins (keV):')
print('    sub-Coulomb:  < {}'.format(bin_edges[0]))
print('    pre-fusion:   {} - {}'.format(bin_edges[0], bin_edges[1]))
print('    fusion-peak:  {} - {}'.format(bin_edges[1], bin_edges[2]))
print('    hot-tail:     > {}'.format(bin_edges[2]))
print('  Heatmap grid:      {}x{}'.format(args.grid_size, args.grid_size))
print('  Sample/dump:       {}'.format(args.max_particles_per_dump))
print('  Tracer threshold:  {} keV'.format(args.tracer_threshold))
print('  Output:            {}'.format(out_dir))
print('=' * 92)

def ke_keV_from_u(ux, uy, uz, mass=mass_kg):
    u2 = ux*ux + uy*uy + uz*uz
    gamma = np.sqrt(1.0 + u2)
    return (gamma - 1.0) * mass * C * C / KEV_TO_J

def style_axes(ax, title=None):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=11)

def overlay_geometry(ax, scale=1e3, alpha=0.5):
    theta = np.linspace(0, 2*np.pi, 100)
    ring_x = RING_RADIUS_M * scale * np.cos(theta)
    ring_z = RING_RADIUS_M * scale * np.sin(theta)
    ax.plot(ring_x, ring_z, color=TEXT_COLOR, linestyle='--', linewidth=0.7, alpha=alpha)
    for k in range(N_SPOTS):
        angle = 2*np.pi*k/N_SPOTS + np.pi/N_SPOTS
        xx = R_xline * scale * np.cos(angle)
        zz = R_xline * scale * np.sin(angle)
        ax.plot(xx, zz, marker='+', color=TEXT_COLOR, markersize=10,
                alpha=alpha+0.3, markeredgewidth=1.5)


# Pass 1: read particles with subsampling
print()
print('Pass 1: reading particle data with subsampling...')
sys.stdout.flush()
try_ids = True
all_particles = []
all_KE_global = []

def get_particles_capped(it, fields, max_n):
    data = ts.get_particle(fields, species=args.species, iteration=it)
    if not isinstance(data, (tuple, list)):
        data = (data,)
    n_total = len(data[0])
    if n_total <= max_n:
        return data, n_total, n_total
    np.random.seed(42)
    idx = np.random.choice(n_total, max_n, replace=False)
    return tuple(d[idx] for d in data), n_total, max_n

for i, it in enumerate(iterations):
    try:
        if try_ids:
            try:
                (x, z, ux, uy, uz, ids), n_total, n_sampled = get_particles_capped(
                    it, ['x', 'z', 'ux', 'uy', 'uz', 'id'],
                    args.max_particles_per_dump)
            except Exception:
                if i == 0:
                    print('  WARNING: particle IDs unavailable')
                    sys.stdout.flush()
                try_ids = False
                (x, z, ux, uy, uz), n_total, n_sampled = get_particles_capped(
                    it, ['x', 'z', 'ux', 'uy', 'uz'],
                    args.max_particles_per_dump)
                ids = None
        else:
            (x, z, ux, uy, uz), n_total, n_sampled = get_particles_capped(
                it, ['x', 'z', 'ux', 'uy', 'uz'],
                args.max_particles_per_dump)
            ids = None
    except Exception as e:
        print('  iter {}: failed ({})'.format(it, e))
        sys.stdout.flush()
        all_particles.append(None)
        continue

    KE = ke_keV_from_u(ux, uy, uz)
    all_particles.append({'x': x, 'z': z, 'KE': KE, 'ids': ids,
                          'n_total': n_total, 'n_sampled': n_sampled})
    if KE.size > 0:
        all_KE_global.append(np.percentile(KE, [50, 95, 99]))
    print('  {:>3d}/{}: t={:7.2f} ps, N_total={:,}, sampled={:,}'.format(
        i+1, len(iterations), times_s[i]*1e12, int(n_total), int(n_sampled)))
    sys.stdout.flush()

if not all_particles or all(p is None for p in all_particles):
    print('ERROR: no particle data', file=sys.stderr)
    sys.exit(1)

all_KE_arr = np.array(all_KE_global)
KE_max_global = float(np.percentile(all_KE_arr[:, 2], 95))
KE_min_global = max(1.0, float(np.percentile(all_KE_arr[:, 0], 5)))
KE_max_global = max(KE_max_global, bin_edges[2] * 2)

spectrum_bins = np.logspace(0, np.log10(KE_max_global * 2), 80)
spectrum_max_count = 0
for d in all_particles:
    if d is None:
        continue
    h, _ = np.histogram(d['KE'], bins=spectrum_bins)
    spectrum_max_count = max(spectrum_max_count, h.max())

# Build spatial extent for histograms
xz_edges_x = np.linspace(-domain_max, domain_max, args.grid_size + 1)
xz_edges_z = np.linspace(-domain_max, domain_max, args.grid_size + 1)

# Pre-compute global density max for each bin (for consistent colormap)
print()
print('Computing global density ranges per bin...')
sys.stdout.flush()
density_max = {b: 1.0 for b in bins_def}
for d in all_particles:
    if d is None:
        continue
    for bin_name, (lo, hi) in bins_def.items():
        mask = (d['KE'] >= lo) & (d['KE'] < hi) if hi != np.inf else (d['KE'] >= lo)
        if mask.sum() == 0:
            continue
        H, _, _ = np.histogram2d(d['x'][mask], d['z'][mask],
                                  bins=[xz_edges_x, xz_edges_z])
        density_max[bin_name] = max(density_max[bin_name], H.max())
print('  Density max per bin:')
for b, dmax in density_max.items():
    print('    {}: {:.0f}'.format(b, dmax))
sys.stdout.flush()

try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except ImportError:
    print('imageio not installed: pip install "imageio[ffmpeg]"')
    HAVE_IMAGEIO = False


def _normalize_frames(frame_paths):
    """Read frames and normalize to same even dimensions (required by h264)."""
    import numpy as np
    imgs = [imageio.imread(fp) for fp in frame_paths]
    # Find max dimensions, round up to even
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    h = h if h % 2 == 0 else h + 1
    w = w if w % 2 == 0 else w + 1
    # Pad all frames to same even size
    normalized = []
    for img in imgs:
        if img.shape[0] == h and img.shape[1] == w:
            normalized.append(img)
        else:
            pad = np.zeros((h, w) + img.shape[2:], dtype=img.dtype)
            pad[:img.shape[0], :img.shape[1]] = img
            normalized.append(pad)
    return normalized


def write_video(out_path, frame_paths, fps):
    if not HAVE_IMAGEIO or not frame_paths:
        return
    try:
        imgs = _normalize_frames(frame_paths)
        with imageio.get_writer(out_path, fps=fps, codec='h264',
                                macro_block_size=None,
                                output_params=['-pix_fmt', 'yuv420p']) as writer:
            for img in imgs:
                writer.append_data(img)
        print('  -> {}'.format(out_path))
    except Exception as e:
        print('  MP4 failed ({}); using GIF'.format(e))
        out_gif = out_path.replace('.mp4', '.gif')
        with imageio.get_writer(out_gif, mode='I', duration=1.0/fps) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print('  -> {}'.format(out_gif))


# Animation 1: 4-panel density heatmaps
if not args.skip_density:
    print()
    print('Animation 1: 4-panel density heatmaps...')
    sys.stdout.flush()
    density_dir = os.path.join(out_dir, 'density_anim_frames')
    os.makedirs(density_dir, exist_ok=True)
    density_frames = []
    extent = [-domain_max*1e3, domain_max*1e3, -domain_max*1e3, domain_max*1e3]

    for i, (it, d, t_s) in enumerate(zip(iterations, all_particles, times_s)):
        fig, axes = plt.subplots(2, 2, figsize=(13, 12), facecolor=BG_COLOR)
        if d is None:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color=TEXT_COLOR)
        else:
            x, z, KE = d['x'], d['z'], d['KE']
            # Scale factor: subsampled count -> total count
            scale_factor = d['n_total'] / d['n_sampled'] if d['n_sampled'] > 0 else 1.0

            bin_names = ['sub_coulomb', 'pre_fusion', 'fusion_peak', 'hot_tail']
            ax_positions = [(0,0), (0,1), (1,0), (1,1)]

            for bin_name, (row, col) in zip(bin_names, ax_positions):
                ax = axes[row, col]
                lo, hi = bins_def[bin_name]
                cmap = BIN_CMAPS[bin_name]
                if hi == np.inf:
                    label_e = 'KE > {:.0f} keV'.format(lo)
                    mask = KE >= lo
                else:
                    label_e = '{:.0f} - {:.0f} keV'.format(lo, hi)
                    mask = (KE >= lo) & (KE < hi)

                n_bin_sample = int(mask.sum())
                n_bin_total = int(n_bin_sample * scale_factor)

                if n_bin_sample > 0:
                    H, _, _ = np.histogram2d(x[mask], z[mask],
                                              bins=[xz_edges_x, xz_edges_z])
                    # Scale to estimated total
                    H_scaled = H * scale_factor
                    H_max = density_max[bin_name] * scale_factor
                    H_min = max(1.0, H_max / 1e4)
                    H_scaled_safe = np.where(H_scaled > 0, H_scaled, np.nan)
                    im = ax.imshow(H_scaled_safe.T, origin='lower', extent=extent,
                                   cmap=cmap, norm=LogNorm(vmin=H_min, vmax=H_max),
                                   aspect='equal', interpolation='nearest')
                    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
                    cb.set_label('estimated count/cell', color=MUTED_COLOR, fontsize=8)
                    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=8)
                    for spine in cb.ax.spines.values():
                        spine.set_edgecolor(MUTED_COLOR)
                else:
                    ax.text(0.5, 0.5, 'No particles in this bin',
                            transform=ax.transAxes, ha='center', va='center',
                            color=MUTED_COLOR, fontsize=10)

                overlay_geometry(ax, alpha=0.4)
                ax.set_xlim(-domain_max*1e3, domain_max*1e3)
                ax.set_ylim(-domain_max*1e3, domain_max*1e3)
                ax.set_aspect('equal')
                ax.set_xlabel('x (mm)', color=MUTED_COLOR, fontsize=9)
                ax.set_ylabel('z (mm)', color=MUTED_COLOR, fontsize=9)
                style_axes(ax, '{}: {} (~{:,} particles)'.format(
                    BIN_LABELS[bin_name], label_e, n_bin_total))

        t_ps = t_s * 1e12
        plt.suptitle('Particle DENSITY by energy band  -  t = {:.1f} ps  ({}/{})'.format(
            t_ps, i+1, len(iterations)),
            color=TEXT_COLOR, fontsize=13, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fp = os.path.join(density_dir, 'frame_{:04d}.png'.format(i))
        plt.savefig(fp, dpi=110, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)
        density_frames.append(fp)
        if (i+1) % 10 == 0:
            print('  rendered {}/{}'.format(i+1, len(iterations)))
            sys.stdout.flush()

    write_video(os.path.join(out_dir, 'particle_density_by_energy.mp4'),
                density_frames, args.fps)


# Animation 2: acceleration tracer
if not args.skip_tracer:
    print()
    print('Animation 2: acceleration tracer (KE > {:.0f} keV)...'.format(
        args.tracer_threshold))
    sys.stdout.flush()

    if try_ids and all_particles[0] and all_particles[0]['ids'] is not None:
        accelerated_ids = set()
        for d in all_particles:
            if d is None or d['ids'] is None:
                continue
            mask = d['KE'] >= args.tracer_threshold
            accelerated_ids.update(d['ids'][mask].tolist())

        n_accelerated = len(accelerated_ids)
        print('  Particles ever above {:.0f} keV: {}'.format(
            args.tracer_threshold, n_accelerated))
        sys.stdout.flush()

        if n_accelerated > args.max_tracer_particles:
            np.random.seed(42)
            tracer_ids_arr = np.array(sorted(np.random.choice(
                list(accelerated_ids), args.max_tracer_particles, replace=False)))
        elif n_accelerated > 0:
            tracer_ids_arr = np.array(sorted(accelerated_ids))
        else:
            tracer_ids_arr = None
    else:
        print('  WARNING: tracer needs particle IDs; per-frame fallback')
        tracer_ids_arr = None

    tracer_dir = os.path.join(out_dir, 'tracer_anim_frames')
    os.makedirs(tracer_dir, exist_ok=True)
    tracer_frames = []

    for i, (it, d, t_s) in enumerate(zip(iterations, all_particles, times_s)):
        fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG_COLOR)
        if d is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color=TEXT_COLOR)
        else:
            x, z, KE = d['x'], d['z'], d['KE']
            ids = d['ids']

            if tracer_ids_arr is not None and ids is not None:
                mask = np.isin(ids, tracer_ids_arr)
            else:
                mask = KE >= args.tracer_threshold

            x_t = x[mask]
            z_t = z[mask]
            KE_t = KE[mask]

            if len(x_t) > 0:
                sc = ax.scatter(x_t*1e3, z_t*1e3, c=KE_t, cmap=CMAP_KE,
                                norm=LogNorm(vmin=args.tracer_threshold,
                                             vmax=KE_max_global),
                                s=20, alpha=0.85,
                                edgecolors='white', linewidths=0.2)
                cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
                cb.set_label('KE (keV)', color=MUTED_COLOR)
                cb.ax.tick_params(colors=MUTED_COLOR)
                for spine in cb.ax.spines.values():
                    spine.set_edgecolor(MUTED_COLOR)

            overlay_geometry(ax, alpha=0.5)

        ax.set_xlim(-domain_max*1e3, domain_max*1e3)
        ax.set_ylim(-domain_max*1e3, domain_max*1e3)
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)', color=MUTED_COLOR)
        ax.set_ylabel('z (mm)', color=MUTED_COLOR)

        t_ps = t_s * 1e12
        n_visible = (d['KE'] >= args.tracer_threshold).sum() if d else 0
        style_axes(ax,
            'Particles > {:.0f} keV  -  t = {:.1f} ps  ({} visible)'.format(
                args.tracer_threshold, t_ps, int(n_visible)))

        fp = os.path.join(tracer_dir, 'frame_{:04d}.png'.format(i))
        plt.savefig(fp, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)
        tracer_frames.append(fp)
        if (i+1) % 10 == 0:
            print('  rendered {}/{}'.format(i+1, len(iterations)))
            sys.stdout.flush()

    write_video(os.path.join(out_dir, 'acceleration_tracer.mp4'),
                tracer_frames, args.fps)


# Animation 3: spectrum evolution
if not args.skip_spectrum:
    print()
    print('Animation 3: energy spectrum...')
    sys.stdout.flush()
    spec_dir = os.path.join(out_dir, 'spectrum_anim_frames')
    os.makedirs(spec_dir, exist_ok=True)
    spec_frames = []

    for i, (it, d, t_s) in enumerate(zip(iterations, all_particles, times_s)):
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
        if d is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color=TEXT_COLOR)
        else:
            KE = d['KE']
            ax.hist(KE, bins=spectrum_bins, color=ACCENT, edgecolor='none', alpha=0.8)
            ax.axvline(150, color='#3b82f6', linestyle=':', alpha=0.6,
                       label='Coulomb barrier (150 keV)')
            ax.axvline(500, color='#fb923c', linestyle='--', alpha=0.7,
                       label='fusion-grade (500 keV)')
            ax.axvline(675, color='#ef4444', linestyle='--', alpha=0.7,
                       label='peak cross-section (675 keV)')
            ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
                     labelcolor=TEXT_COLOR, fontsize=8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(KE_min_global, KE_max_global * 2)
        ax.set_ylim(1, max(spectrum_max_count, 10) * 1.5)
        ax.set_xlabel('Kinetic Energy (keV)', color=MUTED_COLOR)
        ax.set_ylabel('Particle count', color=MUTED_COLOR)
        t_ps = t_s * 1e12
        style_axes(ax, 'Energy spectrum  -  t = {:.1f} ps  ({}/{})'.format(
            t_ps, i+1, len(iterations)))
        fp = os.path.join(spec_dir, 'frame_{:04d}.png'.format(i))
        plt.savefig(fp, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
        plt.close(fig)
        spec_frames.append(fp)
        if (i+1) % 10 == 0:
            print('  rendered {}/{}'.format(i+1, len(iterations)))
            sys.stdout.flush()

    write_video(os.path.join(out_dir, 'energy_spectrum_evolution.mp4'),
                spec_frames, args.fps)

print()
print('=' * 92)
print('  DONE')
print('=' * 92)
