#!/usr/bin/env python3
"""
pb11_3d_anim_xz_slice.py — B-field magnitude animation, central xz slice.

Produces a 2D animation that is the direct 3D analog of the existing 2D
B-field montage in Paper 1 Figure 2. Reads field dumps from a 3D pilot run,
extracts the central y-slice at each iteration, and renders an MP4 (or PNG
sequence + ffmpeg) showing |B|(x, z, y=mid) evolving in time.

Purpose for the paper: demonstrate visually that the 3D simulation reproduces
the eight-spot ring structure and X-line compression collision phase that
Paper 1 Figure 2 shows in 2D — establishing that the 3D simulation is a
faithful extension of the 2D physics on the symmetry plane.

Usage:
  python pb11_3d_anim_xz_slice.py --run-dir runs/pb11_3d_pilot_<ts>
  # → produces 3d_diag/3d_anim_xz_slice.mp4 + per-frame PNGs

Dependencies: matplotlib, openpmd-api, ffmpeg (in PATH).
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import numpy as np

try:
    import openpmd_api as opmd
except ImportError:
    sys.exit('ERROR: openpmd-api not installed. Try: pip install openpmd-api')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out-dir', default=None, type=Path)
    p.add_argument('--n-spots', type=int, default=8)
    p.add_argument('--ring-radius-um', type=float, default=2400.0)
    p.add_argument('--vmax-T', type=float, default=25.0,
                   help='Color scale upper limit in T (default 25, matches 2D montage)')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--keep-frames', action='store_true',
                   help='Keep individual PNG frames after assembling MP4')
    return p.parse_args()


def find_field_series(run_dir: Path) -> Path:
    candidates = [
        run_dir / 'diags' / 'fields' / 'openpmd_%T.h5',
        run_dir / 'diags' / 'fields',
        run_dir / 'fields' / 'openpmd_%T.h5',
        run_dir / 'fields',
    ]
    for c in candidates:
        if c.is_file() or (c.is_dir() and any(c.iterdir())):
            return c
    sys.exit(f'ERROR: no field diagnostic found under {run_dir}')


def x_line_positions(n_spots, ring_radius_m):
    x_radius = ring_radius_m * np.cos(np.pi / n_spots)
    return [(x_radius * np.cos(2*np.pi*k/n_spots + np.pi/n_spots),
             x_radius * np.sin(2*np.pi*k/n_spots + np.pi/n_spots))
            for k in range(n_spots)]


def spot_positions(n_spots, ring_radius_m):
    return [(ring_radius_m * np.cos(2*np.pi*k/n_spots),
             ring_radius_m * np.sin(2*np.pi*k/n_spots))
            for k in range(n_spots)]


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / '3d_diag')
    frames_dir = out_dir / 'anim_xz_slice_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    series_path = find_field_series(args.run_dir)
    print(f'Reading field series from: {series_path}')
    if series_path.is_dir():
        series = opmd.Series(str(series_path / 'openpmd_%T.h5'),
                             opmd.Access.read_only)
    else:
        series = opmd.Series(str(series_path), opmd.Access.read_only)

    iters = sorted(series.iterations)
    print(f'Found {len(iters)} iterations')

    ring_r_m = args.ring_radius_um * 1e-6
    spots = spot_positions(args.n_spots, ring_r_m)
    xlines = x_line_positions(args.n_spots, ring_r_m)

    # 'inferno' colormap — same as the existing 2D montage in Paper 1 Fig 2
    cmap = plt.get_cmap('inferno')

    for frame_idx, it_idx in enumerate(iters):
        it = series.iterations[it_idx]
        time_ps = it.time * it.time_unit_SI * 1e12

        # Read all three B components, take magnitude on central y-slice
        try:
            B_mesh = it.meshes['B']
            Bx_rec = B_mesh['x']
            By_rec = B_mesh['y']
            Bz_rec = B_mesh['z']
        except KeyError:
            print(f'  iter {it_idx}: missing B record, skipping')
            continue

        # Read full 3D arrays (shape: [nz, ny, nx]).
        # Flush MUST come before np.asarray — slice returns a deferred chunk.
        Bx_chunk = Bx_rec[:, :, :]
        By_chunk = By_rec[:, :, :]
        Bz_chunk = Bz_rec[:, :, :]
        series.flush()
        Bx = np.asarray(Bx_chunk)
        By = np.asarray(By_chunk)
        Bz = np.asarray(Bz_chunk)

        nz, ny, nx = By.shape
        iy_mid = ny // 2
        Bx_slice = Bx[:, iy_mid, :]
        By_slice = By[:, iy_mid, :]
        Bz_slice = Bz[:, iy_mid, :]
        Bmag_slice = np.sqrt(Bx_slice**2 + By_slice**2 + Bz_slice**2)

        # Get extents in mm — these are properties of the Mesh, not the component
        dx, dy, dz = B_mesh.grid_spacing
        x0, y0, z0 = B_mesh.grid_global_offset
        x_mm = (x0 + np.arange(nx) * dx) * 1e3
        z_mm = (z0 + np.arange(nz) * dz) * 1e3
        extent_mm = [x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]]

        fig, ax = plt.subplots(figsize=(6.4, 5.8), facecolor='black')
        ax.set_facecolor('black')
        im = ax.imshow(Bmag_slice, extent=extent_mm, origin='lower',
                       cmap=cmap, vmin=0, vmax=args.vmax_T,
                       aspect='equal', interpolation='bilinear')

        # Overlay spot circles and X-line markers (same convention as 2D montage)
        for (sx, sz) in spots:
            ax.add_patch(plt.Circle((sx*1e3, sz*1e3), 0.30,
                                    fill=False, ec='white', lw=0.6, alpha=0.8))
        for (xx, xz) in xlines:
            ax.plot(xx*1e3, xz*1e3, marker='+', color='#ff6655',
                    ms=8, mew=1.4, alpha=0.9)

        ax.set_xlabel('x (mm)', color='white')
        ax.set_ylabel('z (mm)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        title = f't = {time_ps:.0f} ps   (3D pilot, central y-slice)'
        ax.set_title(title, color='white', fontsize=11)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('|B| (T)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

        plt.tight_layout()
        frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
        plt.savefig(frame_path, dpi=110, facecolor='black')
        plt.close()
        print(f'  frame {frame_idx+1}/{len(iters)} (t={time_ps:.0f} ps) → {frame_path.name}')

    # Assemble MP4 with ffmpeg
    mp4_path = out_dir / '3d_anim_xz_slice.mp4'
    if shutil.which('ffmpeg') is None:
        print(f'\nWARN: ffmpeg not in PATH. PNG frames at {frames_dir}.')
        print(f'      To assemble manually:')
        print(f'      ffmpeg -framerate {args.fps} -i {frames_dir}/frame_%04d.png \\')
        print(f'             -c:v libx264 -pix_fmt yuv420p -crf 22 {mp4_path}')
        return

    import subprocess
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps),
           '-i', str(frames_dir / 'frame_%04d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-crf', '22', str(mp4_path)]
    print(f'\nAssembling MP4: {mp4_path}')
    subprocess.run(cmd, check=False, capture_output=True)
    if mp4_path.exists():
        print(f'  Wrote {mp4_path} ({mp4_path.stat().st_size//1024} KB)')
    if not args.keep_frames:
        for f in frames_dir.glob('frame_*.png'):
            f.unlink()
        frames_dir.rmdir()
        print(f'  Removed PNG frames (use --keep-frames to retain)')


if __name__ == '__main__':
    main()
