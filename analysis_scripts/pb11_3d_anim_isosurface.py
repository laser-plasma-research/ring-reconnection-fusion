#!/usr/bin/env python3
"""
pb11_3d_anim_isosurface.py — Volumetric proton density rendering with isosurfaces.

The headline 3D visualization. Computes proton density n(x, y, z, t) on a
moderate grid by binning particle positions, renders a single semi-transparent
isosurface colored by log(KE), with the 8-spot ring geometry as a visual
reference, camera orbiting through time.

This version (v2) addresses interpretability problems with the original:
  - Single isosurface (no onion-shell visual confusion)
  - Logarithmic KE colormap (gradient detail across keV→MeV range)
  - Ring geometry overlay: spots, X-lines, central marker, ring outline
  - Camera orbits 360° during the animation (each frame = time + slight rotation)
  - Two-light setup so the rear isn't pitch-black
  - Perceptually-uniform plasma colormap

Method:
  1. For each particle dump, bin protons onto a 64x32x64 grid.
  2. First pass scans all dumps for global peak density so the isosurface
     threshold is consistent across frames (otherwise the surface "breathes"
     in/out as time progresses, making the animation hard to interpret).
  3. Render a single isosurface at 25% of peak, semi-transparent, colored by
     mean log10(KE in eV) per bin. Add ring geometry markers as fixed reference.
  4. Camera orbits in azimuth across the frames so motion + time evolve together.

Dependencies: pyvista, numpy, openpmd-api. PyVista pulls in VTK; if not installed:
    pip install "pyvista[all]"   # ~200 MB

Usage:
    python pb11_3d_anim_isosurface.py --run-dir runs/pb11_3d_pilot_<ts>
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

try:
    import pyvista as pv
except ImportError:
    sys.exit('ERROR: pyvista not installed. Try: pip install "pyvista[all]"')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out-dir', default=None, type=Path)
    p.add_argument('--bin-x', type=int, default=64)
    p.add_argument('--bin-y', type=int, default=32)
    p.add_argument('--bin-z', type=int, default=64)
    p.add_argument('--iso-frac', type=float, default=0.25,
                   help='Isosurface threshold as fraction of global peak '
                        'density (default 0.25). Single surface, no nesting.')
    p.add_argument('--iso-opacity', type=float, default=0.65,
                   help='Isosurface opacity 0-1 (default 0.65 — semi-transparent '
                        'so internal structure is visible)')
    p.add_argument('--ke-log-min', type=float, default=3.0,
                   help='Lower bound for log10(KE in eV) colormap (default 3.0 = 1 keV)')
    p.add_argument('--ke-log-max', type=float, default=6.0,
                   help='Upper bound for log10(KE in eV) colormap (default 6.0 = 1 MeV)')
    p.add_argument('--orbit-degrees', type=float, default=360.0,
                   help='Camera azimuth sweep over the full animation (default 360°). '
                        'Set 0 to fix the camera angle.')
    p.add_argument('--start-azimuth', type=float, default=30.0,
                   help='Starting camera azimuth in degrees (default 30)')
    p.add_argument('--elevation', type=float, default=22.0,
                   help='Camera elevation in degrees (default 22 — slight downward tilt)')
    p.add_argument('--ring-radius-mm', type=float, default=2.4,
                   help='Ring radius in mm for the geometry overlay (default 2.4)')
    p.add_argument('--n-spots', type=int, default=8,
                   help='Number of laser spots / X-lines on the ring (default 8)')
    p.add_argument('--cmap', default='plasma',
                   help='Colormap (perceptually uniform recommended). '
                        'Options: plasma, viridis, magma, inferno, cividis')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--window-size', type=int, nargs=2, default=[1100, 950],
                   help='Render window size in pixels (default 1100x950)')
    p.add_argument('--keep-frames', action='store_true')
    return p.parse_args()


def find_particles_series(run_dir: Path) -> Path:
    candidates = [
        run_dir / 'diags' / 'particles' / 'openpmd_%T.h5',
        run_dir / 'diags' / 'particles',
        run_dir / 'particles' / 'openpmd_%T.h5',
        run_dir / 'particles',
    ]
    for c in candidates:
        if c.is_file() or (c.is_dir() and any(c.iterdir())):
            return c
    sys.exit(f'ERROR: no particle diagnostic found under {run_dir}')


def find_field_series(run_dir: Path) -> Path:
    """Used to determine domain extents."""
    candidates = [
        run_dir / 'diags' / 'fields' / 'openpmd_%T.h5',
        run_dir / 'diags' / 'fields',
        run_dir / 'fields' / 'openpmd_%T.h5',
        run_dir / 'fields',
    ]
    for c in candidates:
        if c.is_file() or (c.is_dir() and any(c.iterdir())):
            return c
    return None


def add_ring_geometry(plotter, ring_r_m, n_spots, y_extent_m):
    """Add the 8-spot ring geometry as a visual reference layer.

    Adds:
      - a thin tube tracing the ring at y = 0 (the geometric reference circle)
      - small spheres at the n_spots laser-spot positions (orange, on the ring)
      - vertical tubes at the n_spots X-line positions (blue, the X-lines extend
        along y in 3D)
      - a small bright sphere at the ring centre (target convergence point)

    All in metres; PyVista renders in whatever units are passed.
    """
    # The ring at y = 0
    theta = np.linspace(0, 2 * np.pi, 200)
    ring_x = ring_r_m * np.cos(theta)
    ring_z = ring_r_m * np.sin(theta)
    ring_y = np.zeros_like(theta)
    ring_pts = np.column_stack([ring_x, ring_y, ring_z])
    ring_polyline = pv.lines_from_points(ring_pts, close=True)
    tube = ring_polyline.tube(radius=ring_r_m * 0.005)
    plotter.add_mesh(tube, color='#888888', opacity=0.55,
                     show_scalar_bar=False)

    # Laser-spot positions
    spot_thetas = np.linspace(0, 2 * np.pi, n_spots, endpoint=False)
    spot_radius = ring_r_m * 0.025
    for th in spot_thetas:
        sx = ring_r_m * np.cos(th)
        sz = ring_r_m * np.sin(th)
        sphere = pv.Sphere(radius=spot_radius, center=(sx, 0, sz))
        plotter.add_mesh(sphere, color='#ff7700', opacity=0.85,
                         show_scalar_bar=False)

    # X-line positions: chord midpoints between adjacent spots, extending
    # along y as vertical tubes (X-lines extend along y in 3D)
    xline_r = ring_r_m * np.cos(np.pi / n_spots)
    xline_thetas = spot_thetas + np.pi / n_spots
    xline_radius = ring_r_m * 0.0072  # slimmer than the spot spheres
    for th in xline_thetas:
        xx = xline_r * np.cos(th)
        xz = xline_r * np.sin(th)
        line = pv.Line((xx, -y_extent_m / 2, xz),
                       (xx, +y_extent_m / 2, xz))
        tube_x = line.tube(radius=xline_radius)
        plotter.add_mesh(tube_x, color='#00aaff', opacity=0.7,
                         show_scalar_bar=False)

    # Ring centre (target convergence point)
    centre_marker = pv.Sphere(radius=ring_r_m * 0.020, center=(0, 0, 0))
    plotter.add_mesh(centre_marker, color='#ffffff', opacity=0.95,
                     show_scalar_bar=False)


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / '3d_diag')
    frames_dir = out_dir / 'anim_isosurface_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    pseries_path = find_particles_series(args.run_dir)
    print(f'Reading particle series from: {pseries_path}')
    if pseries_path.is_dir():
        pseries = opmd.Series(str(pseries_path / 'openpmd_%T.h5'),
                              opmd.Access.read_only)
    else:
        pseries = opmd.Series(str(pseries_path), opmd.Access.read_only)
    iters = sorted(pseries.iterations)
    print(f'Found {len(iters)} particle iterations')

    # --- Determine domain extents from a field record (more reliable than particles)
    fseries_path = find_field_series(args.run_dir)
    domain = None
    if fseries_path:
        if fseries_path.is_dir():
            fs = opmd.Series(str(fseries_path / 'openpmd_%T.h5'),
                             opmd.Access.read_only)
        else:
            fs = opmd.Series(str(fseries_path), opmd.Access.read_only)
        f_iters = sorted(fs.iterations)
        if f_iters:
            f_it = fs.iterations[f_iters[0]]
            try:
                B_mesh = f_it.meshes['B']
                rec = B_mesh['y']
                dx, dy, dz = B_mesh.grid_spacing
                x0, y0, z0 = B_mesh.grid_global_offset
                shape = rec.shape  # nz, ny, nx
                nz, ny, nx = shape
                domain = {
                    'x': (x0, x0 + nx * dx),
                    'y': (y0, y0 + ny * dy),
                    'z': (z0, z0 + nz * dz),
                }
                print(f'Domain from fields: x={domain["x"]}, '
                      f'y={domain["y"]}, z={domain["z"]}')
            except Exception:
                pass

    # --- First pass: bin all dumps, find global peak density for consistent threshold
    print('Scanning for global peak density...')
    peak_density = 0.0
    cached_grids = {}
    m_p = 1.6726219e-27
    eV_per_J = 6.241509e18

    for it_idx in iters:
        it = pseries.iterations[it_idx]
        try:
            protons = it.particles['proton']
            x_chunk = protons['position']['x'][:]
            y_chunk = protons['position']['y'][:]
            z_chunk = protons['position']['z'][:]
            ux_chunk = protons['momentum']['x'][:]
            uy_chunk = protons['momentum']['y'][:]
            uz_chunk = protons['momentum']['z'][:]
            w_chunk = protons['weighting'][:]
            pseries.flush()
            x = np.asarray(x_chunk)
            y = np.asarray(y_chunk)
            z = np.asarray(z_chunk)
            ux = np.asarray(ux_chunk)
            uy = np.asarray(uy_chunk)
            uz = np.asarray(uz_chunk)
            w = np.asarray(w_chunk)
        except Exception as e:
            print(f'  iter {it_idx}: error: {e}')
            continue

        if domain is None:
            # Fallback: infer from particle extents
            domain = {
                'x': (float(x.min()), float(x.max())),
                'y': (float(y.min()), float(y.max())),
                'z': (float(z.min()), float(z.max())),
            }

        # Bin onto 3D grid; weight by particle weighting; mean KE per bin
        edges_x = np.linspace(domain['x'][0], domain['x'][1], args.bin_x + 1)
        edges_y = np.linspace(domain['y'][0], domain['y'][1], args.bin_y + 1)
        edges_z = np.linspace(domain['z'][0], domain['z'][1], args.bin_z + 1)

        ke_eV = (ux**2 + uy**2 + uz**2) / (2.0 * m_p) * eV_per_J

        n_grid, _ = np.histogramdd((x, y, z), bins=(edges_x, edges_y, edges_z),
                                   weights=w)
        ew_grid, _ = np.histogramdd((x, y, z), bins=(edges_x, edges_y, edges_z),
                                    weights=w * ke_eV)
        with np.errstate(divide='ignore', invalid='ignore'):
            e_grid = np.where(n_grid > 0, ew_grid / n_grid, 0.0)
        # Take log10(KE) for the colormap. Clip to [10^ke_log_min, 10^ke_log_max].
        ke_min = 10 ** args.ke_log_min
        ke_max = 10 ** args.ke_log_max
        e_grid_clipped = np.clip(e_grid, ke_min, ke_max)
        log_e_grid = np.log10(e_grid_clipped)
        # Bins with no particles → set to log_min so they don't appear hot
        log_e_grid = np.where(n_grid > 0, log_e_grid, args.ke_log_min)

        cached_grids[it_idx] = (n_grid, log_e_grid, edges_x, edges_y, edges_z)
        peak_density = max(peak_density, float(n_grid.max()))
        print(f'  iter {it_idx}: peak n = {n_grid.max():.3e}')

    if peak_density == 0:
        sys.exit('ERROR: no particles in any dump')

    iso_threshold = args.iso_frac * peak_density
    print(f'\nGlobal peak density: {peak_density:.3e}')
    print(f'  Isosurface at: {iso_threshold:.3e} '
          f'({args.iso_frac * 100:.0f}% of peak)')
    print(f'  Coloring by log10(KE/eV) on [{args.ke_log_min}, {args.ke_log_max}]')

    # --- Render each frame
    n_frames = len(cached_grids)
    print(f'\nRendering {n_frames} frames...')
    pv.OFF_SCREEN = True

    ring_r_m = args.ring_radius_mm * 1e-3
    y_extent_m = domain['y'][1] - domain['y'][0]

    for frame_idx, (it_idx, (n_grid, log_e_grid, ex, ey, ez)) \
            in enumerate(cached_grids.items()):
        it = pseries.iterations[it_idx]
        time_ps = it.time * it.time_unit_SI * 1e12

        # Build PyVista grid from the binned data
        spacing = (ex[1] - ex[0], ey[1] - ey[0], ez[1] - ez[0])
        bin_centers_x = 0.5 * (ex[1:] + ex[:-1])
        bin_centers_y = 0.5 * (ey[1:] + ey[:-1])
        bin_centers_z = 0.5 * (ez[1:] + ez[:-1])
        origin = (bin_centers_x[0], bin_centers_y[0], bin_centers_z[0])

        try:
            grid = pv.ImageData(dimensions=(args.bin_x, args.bin_y, args.bin_z),
                                spacing=spacing, origin=origin)
        except AttributeError:
            grid = pv.UniformGrid(dimensions=(args.bin_x, args.bin_y, args.bin_z),
                                  spacing=spacing, origin=origin)

        # Histogramdd gives (x,y,z) ordering; PyVista expects Fortran flatten
        grid.point_data['density'] = n_grid.flatten(order='F')
        grid.point_data['log_KE'] = log_e_grid.flatten(order='F')

        plotter = pv.Plotter(off_screen=True, window_size=tuple(args.window_size))
        plotter.set_background('#0a0a14')  # near-black with slight blue cast

        # --- Two-light setup (one strong key, one fill at the rear)
        plotter.remove_all_lights()
        key_light = pv.Light(position=(ring_r_m * 4, ring_r_m * 4, ring_r_m * 4),
                             focal_point=(0, 0, 0),
                             color='white', intensity=0.9)
        fill_light = pv.Light(position=(-ring_r_m * 3, ring_r_m * 2, -ring_r_m * 3),
                              focal_point=(0, 0, 0),
                              color='#88aaff', intensity=0.45)
        plotter.add_light(key_light)
        plotter.add_light(fill_light)

        # --- Ring geometry overlay (drawn first so isosurface renders over it)
        add_ring_geometry(plotter, ring_r_m, args.n_spots, y_extent_m)

        # --- The single isosurface
        iso_mesh = grid.contour([iso_threshold], scalars='density')
        if iso_mesh.n_points > 0:
            # Sample the log_KE field onto the isosurface vertices
            iso_mesh = iso_mesh.sample(grid)
            plotter.add_mesh(
                iso_mesh,
                scalars='log_KE',
                cmap=args.cmap,
                clim=[args.ke_log_min, args.ke_log_max],
                opacity=args.iso_opacity,
                smooth_shading=True,
                specular=0.4,
                specular_power=15,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'log10(KE / eV)',
                    'color': 'white',
                    'title_font_size': 13,
                    'label_font_size': 11,
                    'n_labels': 4,
                    'fmt': '%.1f',
                    'position_x': 0.85,
                    'position_y': 0.20,
                    'width': 0.05,
                    'height': 0.55,
                },
            )

        # --- Annotations
        plotter.add_text(f't = {time_ps:.1f} ps',
                         position='upper_left', font_size=16, color='white',
                         shadow=True)
        plotter.add_text(
            f'3D pilot — proton density isosurface @ {args.iso_frac * 100:.0f}% of peak\n'
            f'8-spot ring R = {args.ring_radius_mm} mm, '
            f'colored by mean kinetic energy',
            position='lower_left', font_size=10, color='#cccccc',
            shadow=False,
        )

        # --- Subtle bounding box outline (thin so it doesn't dominate)
        plotter.add_mesh(grid.outline(), color='#444444', line_width=1.0,
                         show_scalar_bar=False)

        # --- Camera: orbit through the animation
        if args.orbit_degrees > 0 and n_frames > 1:
            azim = (args.start_azimuth
                    + args.orbit_degrees * frame_idx / max(n_frames - 1, 1))
        else:
            azim = args.start_azimuth

        # Place camera at fixed distance on a sphere parameterised by (azim, elev)
        cam_dist = ring_r_m * 4.5
        az_rad = np.radians(azim)
        el_rad = np.radians(args.elevation)
        cam_x = cam_dist * np.cos(el_rad) * np.cos(az_rad)
        cam_y = cam_dist * np.sin(el_rad)
        cam_z = cam_dist * np.cos(el_rad) * np.sin(az_rad)
        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.zoom(1.05)

        frame_path = frames_dir / f'frame_{frame_idx:04d}.png'
        plotter.screenshot(str(frame_path))
        plotter.close()
        print(f'  frame {frame_idx + 1}/{n_frames} (t={time_ps:.1f} ps, az={azim:.0f}°)')

    # --- Assemble MP4
    mp4_path = out_dir / '3d_anim_isosurface.mp4'
    if shutil.which('ffmpeg') is None:
        print(f'\nffmpeg not in PATH. PNG frames at {frames_dir}.')
        return

    import subprocess
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps),
           '-i', str(frames_dir / 'frame_%04d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-crf', '20',
           str(mp4_path)]
    print(f'\nAssembling MP4: {mp4_path}')
    result = subprocess.run(cmd, check=False, capture_output=True)
    if mp4_path.exists():
        print(f'  Wrote {mp4_path} ({mp4_path.stat().st_size // 1024} KB)')
    else:
        print(f'  ffmpeg failed:\n  stdout: {result.stdout.decode()[-500:]}\n'
              f'  stderr: {result.stderr.decode()[-500:]}')

    if not args.keep_frames:
        for f in frames_dir.glob('frame_*.png'):
            f.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass


if __name__ == '__main__':
    main()
