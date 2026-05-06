#!/usr/bin/env python3
"""
pb11_particle_trajectories.py - Trajectory animation showing 25 selected particles
with fading trails, organized by RADIAL SHELL at t=0.

Selection strategy:
  - Particles grouped by INITIAL RADIUS (5 concentric shells)
  - 5 particles per shell, angularly distributed (covers all sectors)
  - Total: 25 representative particles
  - Trail color = KE at each position (plasma colormap)
  - Marker shape = starting shell

Default shells (8-spot ring with R=2400um, R_xline=2218um):
  Core         r < 800 um           circle
  Inner        800-1800 um          square
  Mid-ring     1800-2300 um         triangle up
  Outer ring   2300-2700 um         diamond  (spot+X-line zone)
  Far outer    r > 2700 um          star

This reveals convergence vs expansion dynamics:
  - Inner shells initially cold, watch convergence ARRIVE
  - Mid shells see transit between center and ring
  - Outer shells see initial outward expansion + reconnection

Key implementation note:
  WarpX may reorder particles between dumps, so we cannot use seed-based
  index sampling. Instead we sample target IDs from the FIRST dump and then
  for every subsequent dump read all particles + filter by np.isin to those
  specific IDs. This guarantees the same particles are tracked across dumps.

OUTPUTS:
    figures/particle_trajectories.mp4         - animation
    figures/particle_trajectories.gif         - animation (GIF fallback)
    figures/trajectory_summary.png            - all 25 trajectories overlaid
    figures/trajectory_energy_evolution.png   - KE vs time, per shell

USAGE:
    python pb11_particle_trajectories.py --dir runs/<run_name>

OPTIONS:
    --shells list              Shell radii edges in um (default 800,1800,2300,2700)
    --n-per-shell N            Particles per shell (default 5)
    --target-pool-size N       Target IDs sampled from first dump (default 200000)
    --species name             (default 'proton')
    --fps N                    (default 6)
"""

import os, sys, argparse, configparser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

PROTON_MASS_KG = 1.67262e-27
B11_MASS_KG    = 11.0093 * 1.66054e-27
Q_E            = 1.602e-19
C              = 2.998e8
KEV_TO_J       = Q_E * 1e3

BG_COLOR     = '#0a0a0a'
TEXT_COLOR   = '#e6e6e6'
MUTED_COLOR  = '#999999'
ACCENT       = '#5dadec'
CMAP_KE      = 'plasma'

# Shell labels and markers
SHELL_LABELS = ['Core', 'Inner', 'Mid-ring', 'Outer ring', 'Far outer']
SHELL_MARKERS = ['o', 's', '^', 'D', '*']
SHELL_COLORS = ['#3b82f6', '#facc15', '#fb923c', '#ef4444', '#a855f7']

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dir', required=True)
parser.add_argument('--shells', default='800,1800,2300,2700',
                    help='Shell radii edges in um (default 800,1800,2300,2700)')
parser.add_argument('--n-per-shell', type=int, default=5)
parser.add_argument('--target-pool-size', type=int, default=200000,
                    help='Target IDs sampled from first dump (default 200000)')
parser.add_argument('--species', default='proton')
parser.add_argument('--fps', type=int, default=6)
parser.add_argument('--max-snapshots', type=int, default=0)
args = parser.parse_args()

run_dir = args.dir.rstrip('/')
shell_edges_um = [float(s) for s in args.shells.split(',')]
if len(shell_edges_um) != 4:
    print('ERROR: --shells must have exactly 4 values', file=sys.stderr)
    sys.exit(1)
shell_edges_m = [s * 1e-6 for s in shell_edges_um]

shells = [
    (0.0,                shell_edges_m[0]),  # Core
    (shell_edges_m[0],   shell_edges_m[1]),  # Inner
    (shell_edges_m[1],   shell_edges_m[2]),  # Mid-ring
    (shell_edges_m[2],   shell_edges_m[3]),  # Outer ring
    (shell_edges_m[3],   np.inf),             # Far outer
]
shells_um = [
    (0,                  shell_edges_um[0]),
    (shell_edges_um[0],  shell_edges_um[1]),
    (shell_edges_um[1],  shell_edges_um[2]),
    (shell_edges_um[2],  shell_edges_um[3]),
    (shell_edges_um[3],  np.inf),
]

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
times_ps = times_s * 1e12

if args.max_snapshots > 0 and len(iterations) > args.max_snapshots:
    step = max(1, len(iterations) // args.max_snapshots)
    iterations = iterations[::step][:args.max_snapshots]
    times_s = np.array([ts.t[list(ts.iterations).index(it)] for it in iterations])
    times_ps = times_s * 1e12

out_dir = os.path.join(run_dir, 'figures')
os.makedirs(out_dir, exist_ok=True)
domain_max = RING_RADIUS_M * 1.8
n_dumps = len(iterations)

print('=' * 92)
print('  PARTICLE TRAJECTORIES (radial shell selection) - {}'.format(run_dir))
print('=' * 92)
print('  Particle dumps:    {}'.format(particle_dir))
print('  Snapshots:         {}'.format(n_dumps))
print('  Time range:        {:.2f} -> {:.2f} ps'.format(times_ps[0], times_ps[-1]))
print('  Species:           {}'.format(args.species))
print('  Radial shells (um):')
for i, (lo, hi) in enumerate(shells_um):
    if hi == np.inf:
        print('    {} ({}): r > {} um   marker={}'.format(
            SHELL_LABELS[i], i, lo, SHELL_MARKERS[i]))
    else:
        print('    {} ({}): {}-{} um   marker={}'.format(
            SHELL_LABELS[i], i, lo, hi, SHELL_MARKERS[i]))
print('  Particles per shell: {}'.format(args.n_per_shell))
print('  Target pool size:    {} (sampled from first dump)'.format(args.target_pool_size))
print('  Output:              {}'.format(out_dir))
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
    # Ring
    ring_x = RING_RADIUS_M * scale * np.cos(theta)
    ring_z = RING_RADIUS_M * scale * np.sin(theta)
    ax.plot(ring_x, ring_z, color=TEXT_COLOR, linestyle='--', linewidth=0.7, alpha=alpha)
    # Shell boundaries (lighter)
    for r_m in shell_edges_m:
        sx = r_m * scale * np.cos(theta)
        sz = r_m * scale * np.sin(theta)
        ax.plot(sx, sz, color=MUTED_COLOR, linestyle=':', linewidth=0.5, alpha=alpha*0.6)
    # X-line markers
    for k in range(N_SPOTS):
        angle = 2*np.pi*k/N_SPOTS + np.pi/N_SPOTS
        xx = R_xline * scale * np.cos(angle)
        zz = R_xline * scale * np.sin(angle)
        ax.plot(xx, zz, marker='+', color=TEXT_COLOR, markersize=10,
                alpha=alpha+0.3, markeredgewidth=1.5)


# ============================================================================
# PASS 1: Read FIRST dump, sample target IDs based on initial radial shell
# ============================================================================
print()
print('Pass 1: reading first dump to select target particles by radial shell...')
sys.stdout.flush()

it0 = iterations[0]
try:
    x0_full, z0_full, ux0, uy0, uz0, ids_full = ts.get_particle(
        ['x', 'z', 'ux', 'uy', 'uz', 'id'],
        species=args.species, iteration=it0)
except Exception as e:
    print('ERROR: cannot read first dump: {}'.format(e), file=sys.stderr)
    sys.exit(1)

n_total = len(x0_full)
print('  First dump: N_total = {:,}'.format(n_total))
sys.stdout.flush()

# Compute initial radii
r0_full = np.sqrt(x0_full**2 + z0_full**2)
theta0_full = np.arctan2(z0_full, x0_full)

# Sample target_pool_size particles uniformly across whole dump as candidate pool
np.random.seed(42)
if n_total > args.target_pool_size:
    pool_idx = np.random.choice(n_total, args.target_pool_size, replace=False)
else:
    pool_idx = np.arange(n_total)

pool_ids = ids_full[pool_idx]
pool_x = x0_full[pool_idx]
pool_z = z0_full[pool_idx]
pool_r = r0_full[pool_idx]
pool_theta = theta0_full[pool_idx]

print('  Pool size: {:,}'.format(len(pool_ids)))
sys.stdout.flush()

# Select 5 particles per shell, angularly distributed
print()
print('Selecting representative particles by shell...')
selected_ids = []
selected_shell = []  # shell index
selected_x0 = []
selected_z0 = []

for shell_idx, (lo, hi) in enumerate(shells):
    if hi == np.inf:
        mask = pool_r >= lo
    else:
        mask = (pool_r >= lo) & (pool_r < hi)
    candidates = np.where(mask)[0]
    n_cand = len(candidates)
    n_pick = min(args.n_per_shell, n_cand)
    if n_cand == 0:
        print('  {}: NO candidates in shell ({} um < r < {} um)'.format(
            SHELL_LABELS[shell_idx], shells_um[shell_idx][0], shells_um[shell_idx][1]))
        continue
    # Sort candidates by angle, pick evenly spaced (use endpoint=False to avoid
    # picking both -180 and +180 which are the same angle)
    cand_theta = pool_theta[candidates]
    sort_idx = np.argsort(cand_theta)
    candidates_sorted = candidates[sort_idx]
    pick_positions = np.linspace(0, n_cand, n_pick, endpoint=False).astype(int)
    pick_positions = np.clip(pick_positions, 0, n_cand - 1)
    picked = candidates_sorted[pick_positions]

    print('  {} (n_cand={:,}): selected {} particles at angles:'.format(
        SHELL_LABELS[shell_idx], n_cand, n_pick))
    for p in picked:
        # Use Python int (handles arbitrary size) - will be converted to uint64 at array creation
        selected_ids.append(int(pool_ids[p]))
        selected_shell.append(shell_idx)
        selected_x0.append(float(pool_x[p]))
        selected_z0.append(float(pool_z[p]))
        print('    id={:>20d}  r={:6.0f} um  theta={:+6.1f} deg'.format(
            int(pool_ids[p]), pool_r[p]*1e6, np.degrees(pool_theta[p])))
    sys.stdout.flush()

selected_ids = np.array(selected_ids, dtype=np.uint64)
selected_shell = np.array(selected_shell, dtype=np.int32)
n_selected = len(selected_ids)
print()
print('  Total selected: {} particles'.format(n_selected))
sys.stdout.flush()

if n_selected == 0:
    print('ERROR: no particles selected. Check shell definitions.', file=sys.stderr)
    sys.exit(1)


# ============================================================================
# PASS 2: For each dump, read ALL particles, filter to selected IDs
# ============================================================================
print()
print('Pass 2: reading every dump and filtering to {} target IDs...'.format(n_selected))
print('  (this is slow - reading ~100M particles per dump)')
sys.stdout.flush()

traj_x = np.full((n_selected, n_dumps), np.nan)
traj_z = np.full((n_selected, n_dumps), np.nan)
traj_KE = np.full((n_selected, n_dumps), np.nan)

# For first dump, use the data we already have (don't re-read)
# Find positions of selected_ids in the full first dump
sort_idx_full = np.argsort(ids_full)
sorted_ids = ids_full[sort_idx_full]
pos_in_sorted = np.searchsorted(sorted_ids, selected_ids)
matches = sorted_ids[np.minimum(pos_in_sorted, len(sorted_ids)-1)] == selected_ids
actual_idx = sort_idx_full[pos_in_sorted]

if matches.all():
    traj_x[:, 0] = x0_full[actual_idx]
    traj_z[:, 0] = z0_full[actual_idx]
    KE0 = ke_keV_from_u(ux0[actual_idx], uy0[actual_idx], uz0[actual_idx])
    traj_KE[:, 0] = KE0
    print('  1/{}: t={:7.2f} ps - first dump filled (using cached data)'.format(
        n_dumps, times_ps[0]))
else:
    print('  WARNING: only {}/{} matches in first dump'.format(matches.sum(), n_selected))
sys.stdout.flush()

# Free first-dump bulk data, no longer needed
del x0_full, z0_full, ux0, uy0, uz0, ids_full
del r0_full, theta0_full, pool_idx, pool_ids, pool_x, pool_z, pool_r, pool_theta
del sort_idx_full, sorted_ids, pos_in_sorted, actual_idx, matches

# For subsequent dumps, read full data, filter by isin
selected_id_set = set(selected_ids.tolist())
import time
for i in range(1, n_dumps):
    it = iterations[i]
    t_start = time.time()
    try:
        x_full, z_full, ux_full, uy_full, uz_full, ids = ts.get_particle(
            ['x', 'z', 'ux', 'uy', 'uz', 'id'],
            species=args.species, iteration=it)
    except Exception as e:
        print('  iter {}: read failed ({})'.format(it, e))
        sys.stdout.flush()
        continue

    # Filter to selected IDs
    mask = np.isin(ids, selected_ids)
    if mask.sum() == 0:
        print('  {}/{}: NO matches found at iter {}'.format(i+1, n_dumps, it))
        sys.stdout.flush()
        continue

    matched_ids = ids[mask]
    matched_x = x_full[mask]
    matched_z = z_full[mask]
    matched_ux = ux_full[mask]
    matched_uy = uy_full[mask]
    matched_uz = uz_full[mask]

    # For each selected_id, find its position in matched arrays
    sort_idx = np.argsort(matched_ids)
    sorted_matched_ids = matched_ids[sort_idx]
    pos = np.searchsorted(sorted_matched_ids, selected_ids)
    valid_mask = (pos < len(sorted_matched_ids)) & \
                 (sorted_matched_ids[np.minimum(pos, len(sorted_matched_ids)-1)] == selected_ids)
    valid = np.where(valid_mask)[0]

    if len(valid) > 0:
        actual_pos = sort_idx[pos[valid]]
        traj_x[valid, i] = matched_x[actual_pos]
        traj_z[valid, i] = matched_z[actual_pos]
        traj_KE[valid, i] = ke_keV_from_u(matched_ux[actual_pos],
                                          matched_uy[actual_pos],
                                          matched_uz[actual_pos])

    # Free
    del x_full, z_full, ux_full, uy_full, uz_full, ids

    elapsed = time.time() - t_start
    print('  {:>3d}/{}: t={:7.2f} ps, matched {}/{} ({:.1f}s)'.format(
        i+1, n_dumps, times_ps[i], len(valid), n_selected, elapsed))
    sys.stdout.flush()

# Forward-fill any NaN cells (in case a particle was dropped from a dump)
n_filled = (~np.isnan(traj_x)).sum()
n_total_cells = n_selected * n_dumps
print()
print('  Trajectory completeness: {}/{} = {:.1%}'.format(
    n_filled, n_total_cells, n_filled/n_total_cells))
sys.stdout.flush()

# Color scale for trajectories
KE_valid = traj_KE[~np.isnan(traj_KE)]
if len(KE_valid) > 0:
    KE_color_max = float(np.percentile(KE_valid, 99))
    KE_color_min = max(1.0, float(np.percentile(KE_valid, 1)))
else:
    KE_color_min = 1.0
    KE_color_max = 1e5
norm = LogNorm(vmin=KE_color_min, vmax=KE_color_max)

# Compute peak KE per particle
peak_KE = np.nanmax(traj_KE, axis=1)
print()
print('  Peak KE summary by shell:')
for shell_idx in range(5):
    mask = selected_shell == shell_idx
    if mask.sum() > 0:
        sk = peak_KE[mask]
        sk = sk[~np.isnan(sk)]
        if len(sk) > 0:
            print('    {}: peak={:7.0f} keV mean,  range {:6.0f} - {:7.0f} keV'.format(
                SHELL_LABELS[shell_idx], np.mean(sk), np.min(sk), np.max(sk)))
sys.stdout.flush()


# ============================================================================
# FIGURE 1: Static summary - all 25 trajectories overlaid
# ============================================================================
print()
print('Generating Figure 1: trajectory_summary.png')
sys.stdout.flush()

fig, ax = plt.subplots(figsize=(11, 10), facecolor=BG_COLOR)
overlay_geometry(ax, alpha=0.5)

for j in range(n_selected):
    x_t = traj_x[j] * 1e3
    z_t = traj_z[j] * 1e3
    ke_t = traj_KE[j]
    valid = ~np.isnan(x_t)
    if valid.sum() < 2:
        continue
    x_t = x_t[valid]
    z_t = z_t[valid]
    ke_t = ke_t[valid]
    points = np.array([x_t, z_t]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=CMAP_KE, norm=norm, linewidth=1.2, alpha=0.75)
    lc.set_array(ke_t[:-1])
    ax.add_collection(lc)
    # Start position - lime circle
    ax.plot(x_t[0], z_t[0], marker='o', color='lime', markersize=4,
            markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)
    # End position - shell-shape marker
    ax.plot(x_t[-1], z_t[-1], marker=SHELL_MARKERS[selected_shell[j]],
            color='red', markersize=8, markeredgecolor='white',
            markeredgewidth=0.5, alpha=0.95)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=CMAP_KE, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
cb.set_label('Kinetic Energy (keV)', color=MUTED_COLOR)
cb.ax.tick_params(colors=MUTED_COLOR)
for spine in cb.ax.spines.values():
    spine.set_edgecolor(MUTED_COLOR)

# Legend
from matplotlib.lines import Line2D
legend_elements = []
for i, label in enumerate(SHELL_LABELS):
    n_in_shell = (selected_shell == i).sum()
    if n_in_shell > 0:
        legend_elements.append(Line2D([0], [0], marker=SHELL_MARKERS[i], color='red',
                                       markeredgecolor='white', markersize=8,
                                       linestyle='none',
                                       label='{} (N={})'.format(label, n_in_shell)))
legend_elements.append(Line2D([0], [0], marker='o', color='lime', markersize=4,
                               linestyle='none', label='start positions'))
ax.legend(handles=legend_elements, loc='upper right',
          facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
          labelcolor=TEXT_COLOR, fontsize=9)

ax.set_xlim(-domain_max*1e3, domain_max*1e3)
ax.set_ylim(-domain_max*1e3, domain_max*1e3)
ax.set_aspect('equal')
ax.set_xlabel('x (mm)', color=MUTED_COLOR)
ax.set_ylabel('z (mm)', color=MUTED_COLOR)
style_axes(ax, '{} representative trajectories  (color=KE, marker=initial shell)'.format(n_selected))

plt.tight_layout()
out_path = os.path.join(out_dir, 'trajectory_summary.png')
plt.savefig(out_path, dpi=140, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print('  -> {}'.format(out_path))


# ============================================================================
# FIGURE 2: KE evolution per shell
# ============================================================================
print()
print('Generating Figure 2: trajectory_energy_evolution.png')
sys.stdout.flush()

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, facecolor=BG_COLOR)

for shell_idx in range(5):
    ax = axes[shell_idx]
    in_shell = selected_shell == shell_idx
    if in_shell.sum() == 0:
        ax.text(0.5, 0.5, 'No particles in this shell',
                transform=ax.transAxes, ha='center', va='center',
                color=MUTED_COLOR)
        style_axes(ax, SHELL_LABELS[shell_idx])
        continue
    for j in np.where(in_shell)[0]:
        ke_t = traj_KE[j]
        valid = ~np.isnan(ke_t)
        ax.plot(times_ps[valid], ke_t[valid],
                color=SHELL_COLORS[shell_idx], linewidth=1.2, alpha=0.7)
    ax.set_yscale('log')
    ax.set_ylim(0.1, KE_color_max * 2)
    ax.axhline(150, color='cyan', linestyle=':', alpha=0.4, label='Coulomb barrier')
    ax.axhline(675, color='lime', linestyle='--', alpha=0.4, label='peak σ')
    ax.set_ylabel('KE (keV)', color=MUTED_COLOR)
    ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor=MUTED_COLOR,
             labelcolor=TEXT_COLOR, fontsize=8)
    style_axes(ax, '{} (N={})'.format(SHELL_LABELS[shell_idx], in_shell.sum()))

axes[-1].set_xlabel('Time (ps)', color=MUTED_COLOR)
plt.tight_layout()
out_path = os.path.join(out_dir, 'trajectory_energy_evolution.png')
plt.savefig(out_path, dpi=140, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print('  -> {}'.format(out_path))


# ============================================================================
# ANIMATION: trajectories with fading trails
# ============================================================================
print()
print('Generating animation: particle_trajectories...')
sys.stdout.flush()
anim_dir = os.path.join(out_dir, 'trajectory_anim_frames')
os.makedirs(anim_dir, exist_ok=True)
frame_paths = []

for frame_i in range(n_dumps):
    fig, ax = plt.subplots(figsize=(11, 10), facecolor=BG_COLOR)
    overlay_geometry(ax, alpha=0.4)

    for j in range(n_selected):
        x_full = traj_x[j, :frame_i+1] * 1e3
        z_full = traj_z[j, :frame_i+1] * 1e3
        ke_full = traj_KE[j, :frame_i+1]
        valid = ~np.isnan(x_full)
        if valid.sum() == 0:
            continue
        x_trail = x_full[valid]
        z_trail = z_full[valid]
        ke_trail = ke_full[valid]

        # Mark starting position (always visible)
        if not np.isnan(traj_x[j, 0]):
            ax.plot(traj_x[j, 0]*1e3, traj_z[j, 0]*1e3,
                    marker='o', color='lime', markersize=3, alpha=0.5)

        if frame_i == 0 or len(x_trail) < 2:
            # Just mark current position
            if len(x_trail) >= 1:
                cur_KE = ke_trail[-1]
                cur_color = plt.cm.plasma(norm(cur_KE))
                ax.plot(x_trail[-1], z_trail[-1],
                        marker=SHELL_MARKERS[selected_shell[j]], color=cur_color,
                        markersize=10, markeredgecolor='white',
                        markeredgewidth=0.7, alpha=0.95)
            continue

        # Build line segments with fading alpha
        points = np.array([x_trail, z_trail]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Alpha fades from 0.15 (oldest) to 0.95 (newest)
        alphas = np.linspace(0.15, 0.95, len(segments))
        lc = LineCollection(segments, cmap=CMAP_KE, norm=norm, linewidth=1.5)
        lc.set_array(ke_trail[:-1])
        lc.set_alpha(alphas.tolist())
        ax.add_collection(lc)

        # Current position: bright marker
        cur_KE = ke_trail[-1]
        cur_color = plt.cm.plasma(norm(cur_KE))
        ax.plot(x_trail[-1], z_trail[-1],
                marker=SHELL_MARKERS[selected_shell[j]], color=cur_color,
                markersize=10, markeredgecolor='white',
                markeredgewidth=0.7, alpha=0.95)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP_KE, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label('Kinetic Energy (keV)', color=MUTED_COLOR)
    cb.ax.tick_params(colors=MUTED_COLOR)
    for spine in cb.ax.spines.values():
        spine.set_edgecolor(MUTED_COLOR)

    ax.set_xlim(-domain_max*1e3, domain_max*1e3)
    ax.set_ylim(-domain_max*1e3, domain_max*1e3)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)', color=MUTED_COLOR)
    ax.set_ylabel('z (mm)', color=MUTED_COLOR)
    t_ps = times_ps[frame_i]
    style_axes(ax, '{} trajectories (radial shells)  -  t = {:.0f} ps  ({}/{})'.format(
        n_selected, t_ps, frame_i+1, n_dumps))

    fp = os.path.join(anim_dir, 'frame_{:04d}.png'.format(frame_i))
    plt.savefig(fp, dpi=120, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)
    frame_paths.append(fp)
    if (frame_i+1) % 10 == 0:
        print('  rendered {}/{}'.format(frame_i+1, n_dumps))
        sys.stdout.flush()

# Create video
try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except ImportError:
    print('imageio not installed')
    HAVE_IMAGEIO = False

if HAVE_IMAGEIO and frame_paths:
    out_mp4 = os.path.join(out_dir, 'particle_trajectories.mp4')
    out_gif = os.path.join(out_dir, 'particle_trajectories.gif')
    try:
        with imageio.get_writer(out_mp4, fps=args.fps, codec='h264',
                                 macro_block_size=1) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print('  -> {}'.format(out_mp4))
    except Exception as e:
        print('  MP4 failed ({}); using GIF only'.format(e))
    # Always also write a GIF
    try:
        with imageio.get_writer(out_gif, mode='I',
                                 duration=1.0/args.fps, loop=0) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print('  -> {}'.format(out_gif))
    except Exception as e:
        print('  GIF failed: {}'.format(e))

print()
print('=' * 92)
print('  DONE')
print('=' * 92)
