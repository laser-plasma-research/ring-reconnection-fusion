#!/usr/bin/env python3
"""
pb11_3d_anim_yz_xline.py — yz-slice animation through one X-line.

Produces a yz-plane B-field animation at a fixed x-position chosen to pass
through one of the eight X-lines. This is the new view that 3D enables: it
shows whether the X-line stays straight along y (kink-stable) or develops a
sinusoidal bend along y (kink-unstable). Direct visual evidence of the
kink-growth diagnostic.

Purpose for the paper: paired with pb11_3d_kink_growth.py, this animation
provides the visual confirmation that the X-line topology is robust along
the third dimension. If the kink-growth verdict comes out STABLE, this
animation is the figure that demonstrates it.

Usage:
  python pb11_3d_anim_yz_xline.py --run-dir runs/pb11_3d_pilot_<ts> \\
      --xline-index 0       # which of the 8 X-lines to slice through

Dependencies: matplotlib, openpmd-api, ffmpeg.
"""

import argparse
import sys
import shutil
from pathlib import Path

import numpy as np

try:
    import openpmd_api as opmd
except ImportError:
    sys.exit('ERROR: openpmd-api not installed.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out-dir', default=None, type=Path)
    p.add_argument('--n-spots', type=int, default=8)
    p.add_argument('--ring-radius-um', type=float, default=2400.0)
    p.add_argument('--xline-index', type=int, default=0,
                   help='Which of the 8 X-lines to slice through (0..n_spots-1)')
    p.add_argument('--vmax-T', type=float, default=20.0)
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--keep-frames', action='store_true')
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


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / '3d_diag')
    frames_dir = out_dir / f'anim_yz_xline{args.xline_index}_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    series_path = find_field_series(args.run_dir)
    if series_path.is_dir():
        series = opmd.Series(str(series_path / 'openpmd_%T.h5'),
                             opmd.Access.read_only)
    else:
        series = opmd.Series(str(series_path), opmd.Access.read_only)

    iters = sorted(series.iterations)
    print(f'Found {len(iters)} iterations')

    # Determine the X-line position
    ring_r_m = args.ring_radius_um * 1e-6
    x_radius = ring_r_m * np.cos(np.pi / args.n_spots)
    k = args.xline_index
    angle = 2.0 * np.pi * k / args.n_spots + np.pi / args.n_spots
    x_target = x_radius * np.cos(angle)
    z_target = x_radius * np.sin(angle)
    print(f'X-line {k}: target (x,z) = ({x_target*1e3:.3f}, {z_target*1e3:.3f}) mm')

    cmap = plt.get_cmap('RdBu_r')   # diverging — shows By sign

    for frame_idx, it_idx in enumerate(iters):
        it = series.iterations[it_idx]
        time_ps = it.time * it.time_unit_SI * 1e12

        try:
            B_mesh = it.meshes['B']
            By_rec = B_mesh['y']
        except KeyError:
            continue

        By_chunk = By_rec[:, :, :]
        series.flush()
        By = np.asarray(By_chunk)
        nz, ny, nx = By.shape

        # Grid coords — properties of the Mesh, not the component
        dx, dy, dz = B_mesh.grid_spacing
        x0, y0, z0 = B_mesh.grid_global_offset
        # Find ix nearest to x_target — but since X-line is at oblique (x,z),
        # we slice along the LINE through z_target on the (y, x) plane that's
        # closest. Cleanest: slice By(y, z) at fixed ix = nearest(x_target),
        # i.e. show how B_y varies with z and y at the chord position.
        ix = int(round((x_target - x0) / dx))
        ix = max(0, min(nx - 1, ix))

        By_yz = By[:, :, ix]   # shape [nz, ny]

        y_mm = (y0 + np.arange(ny) * dy) * 1e3
        z_mm = (z0 + np.arange(nz) * dz) * 1e3
        extent_mm = [y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]]

        fig, ax = plt.subplots(figsize=(5.8, 5.4), facecolor='white')
        im = ax.imshow(By_yz, extent=extent_mm, origin='lower',
                       cmap=cmap, vmin=-args.vmax_T, vmax=args.vmax_T,
                       aspect='auto', interpolation='bilinear')

        # Mark the X-line z-position with a horizontal line
        ax.axhline(z_target * 1e3, color='black', ls='--', lw=0.6, alpha=0.6)
        ax.text(y_mm[0] + 0.05*(y_mm[-1]-y_mm[0]),
                z_target * 1e3 + 0.05,
                f'X-line {k}', fontsize=8, color='black', alpha=0.7)

        ax.set_xlabel('y (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title(f't = {time_ps:.0f} ps   (3D pilot, yz slice through X-line {k})',
                     fontsize=10)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'B$_y$ (T)')

        plt.tight_layout()
        frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
        plt.savefig(frame_path, dpi=110)
        plt.close()
        print(f'  frame {frame_idx+1}/{len(iters)} (t={time_ps:.0f} ps)')

    mp4_path = out_dir / f'3d_anim_yz_xline{args.xline_index}.mp4'
    if shutil.which('ffmpeg') is None:
        print(f'\nffmpeg not in PATH. PNG frames at {frames_dir}.')
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


if __name__ == '__main__':
    main()
