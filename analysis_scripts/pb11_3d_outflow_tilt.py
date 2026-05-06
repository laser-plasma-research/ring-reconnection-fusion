#!/usr/bin/env python3
"""
pb11_3d_outflow_tilt.py — Out-of-plane outflow tilt at X-lines.

Question this answers: do the proton outflow jets from each X-line stay in
the (x, z) plane (in-plane fraction high → 2D convergence picture survives),
or do they tilt out of plane along ±y?

Method:
  1. For each particle dump in the run, locate protons within an outflow
     region around each X-line (an annulus or sphere of radius ~ spot σ).
  2. Compute the mean velocity vector ⟨v_x⟩, ⟨v_y⟩, ⟨v_z⟩ over those protons.
  3. Decompose into in-plane magnitude √(⟨v_x⟩² + ⟨v_z⟩²) and out-of-plane
     magnitude |⟨v_y⟩|. Report the in-plane fraction.
  4. Threshold: in-plane fraction ≥ 0.85 means the convergence is robust to
     3D tilt; lower fractions need to be reported as a caveat.

Inputs:
  --run-dir   3D-pilot run directory containing the particles diag

Outputs:
  3d_outflow_tilt.csv   per-iter, per-X-line in-plane fraction
  3d_outflow_tilt.png   in-plane fraction vs time, one line per X-line
  3d_outflow_tilt.txt   summary report
"""

import argparse
import sys
import csv
import numpy as np
from pathlib import Path

try:
    import openpmd_api as opmd
except ImportError:
    sys.exit('ERROR: openpmd-api not installed. Try: pip install openpmd-api')

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
    p.add_argument('--spot-radius-um', type=float, default=300.0,
                   help='σ of the laser spots (defines outflow region size)')
    p.add_argument('--outflow-radius-factor', type=float, default=1.5,
                   help='Outflow region = spot_radius * factor (default 1.5)')
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


def x_line_positions(n_spots, ring_radius_m):
    x_radius = ring_radius_m * np.cos(np.pi / n_spots)
    out = []
    for k in range(n_spots):
        angle = 2.0 * np.pi * k / n_spots + np.pi / n_spots
        out.append((x_radius * np.cos(angle), x_radius * np.sin(angle)))
    return out


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / '3d_diag')
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = find_particles_series(args.run_dir)
    print(f'Reading particle series from: {series_path}')
    if series_path.is_dir():
        series = opmd.Series(str(series_path / 'openpmd_%T.h5'),
                             opmd.Access.read_only)
    else:
        series = opmd.Series(str(series_path), opmd.Access.read_only)

    iters = sorted(series.iterations)
    print(f'Found {len(iters)} iterations: {iters[0]} → {iters[-1]}')

    n_spots = args.n_spots
    ring_r_m = args.ring_radius_um * 1e-6
    spot_r_m = args.spot_radius_um * 1e-6
    r_outflow = spot_r_m * args.outflow_radius_factor
    xlines = x_line_positions(n_spots, ring_r_m)

    rows = []
    for it_idx in iters:
        it = series.iterations[it_idx]
        time_s = it.time * it.time_unit_SI
        time_ps = time_s * 1e12

        try:
            protons = it.particles['proton']
        except KeyError:
            print(f'  iter {it_idx}: no proton species, skipping')
            continue

        # Read positions and momenta for all protons
        # WarpX/openPMD: position in metres; momentum in kg·m/s
        try:
            x = protons['position']['x'][:]
            y = protons['position']['y'][:]
            z = protons['position']['z'][:]
            ux = protons['momentum']['x'][:]
            uy = protons['momentum']['y'][:]
            uz = protons['momentum']['z'][:]
        except Exception as e:
            print(f'  iter {it_idx}: error reading particles: {e}')
            continue
        series.flush()

        x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
        ux = np.asarray(ux); uy = np.asarray(uy); uz = np.asarray(uz)

        # Convert momentum → velocity (assume m_p; small corrections from
        # relativistic γ are negligible at these energies)
        m_p = 1.6726219e-27
        vx = ux / m_p
        vy = uy / m_p
        vz = uz / m_p

        per_xline = []
        for (x_t, z_t) in xlines:
            r_xz = np.sqrt((x - x_t)**2 + (z - z_t)**2)
            mask = r_xz < r_outflow
            n_in = int(mask.sum())
            if n_in == 0:
                per_xline.append((float('nan'),) * 4)
                continue
            mean_vx = float(np.mean(vx[mask]))
            mean_vy = float(np.mean(vy[mask]))
            mean_vz = float(np.mean(vz[mask]))
            v_in_plane = np.sqrt(mean_vx**2 + mean_vz**2)
            v_total = np.sqrt(v_in_plane**2 + mean_vy**2)
            in_plane_frac = v_in_plane / v_total if v_total > 0 else float('nan')
            per_xline.append((mean_vy, v_in_plane, v_total, in_plane_frac))

        rows.append((time_ps, it_idx, per_xline))
        ip_fracs = [pp[3] for pp in per_xline]
        print(f'  t = {time_ps:7.2f} ps  in-plane fraction: '
              f'mean={np.nanmean(ip_fracs):.3f}  min={np.nanmin(ip_fracs):.3f}')

    # Write CSV
    csv_path = out_dir / '3d_outflow_tilt.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        header = ['time_ps', 'iteration']
        for k in range(n_spots):
            header += [f'vy_xline_{k}', f'v_inplane_xline_{k}',
                       f'v_total_xline_{k}', f'in_plane_frac_xline_{k}']
        w.writerow(header)
        for (t, it, per) in rows:
            row = [f'{t:.4f}', it]
            for tup in per:
                row += [f'{v:.6e}' if np.isfinite(v) else 'nan' for v in tup]
            w.writerow(row)
    print(f'Wrote {csv_path}')

    # Plot in-plane fraction vs time, all X-lines
    times = np.array([r[0] for r in rows])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for k in range(n_spots):
        ip = np.array([r[2][k][3] for r in rows])
        ax.plot(times, ip, alpha=0.5, lw=0.8, label=f'X-line {k}' if k < 3 else None)
    means = np.array([np.nanmean([r[2][k][3] for k in range(n_spots)]) for r in rows])
    ax.plot(times, means, color='black', lw=1.6, label='mean across X-lines')
    ax.axhline(0.85, color='red', ls='--', lw=0.7, alpha=0.7, label='0.85 threshold')
    ax.set_xlabel('time (ps)')
    ax.set_ylabel('in-plane fraction')
    ax.set_title('Outflow tilt at X-lines (3D pilot)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / '3d_outflow_tilt.png'
    plt.savefig(png_path, dpi=140)
    print(f'Wrote {png_path}')

    # Verdict
    overall_mean_ip = float(np.nanmean(means))
    if overall_mean_ip >= 0.85:
        verdict = (f'ROBUST: mean in-plane fraction {overall_mean_ip:.3f} ≥ 0.85. '
                   '2D convergence picture survives 3D tilt.')
    elif overall_mean_ip >= 0.70:
        verdict = (f'PARTIAL: mean in-plane fraction {overall_mean_ip:.3f} between '
                   '0.70 and 0.85. Tilt is non-negligible but does not destroy '
                   'convergence; report as a quantified caveat.')
    else:
        verdict = (f'STRONG TILT: mean in-plane fraction {overall_mean_ip:.3f} < 0.70. '
                   '3D outflow significantly out of plane; convergence picture '
                   'needs revision.')

    txt_path = out_dir / '3d_outflow_tilt.txt'
    with open(txt_path, 'w') as fh:
        fh.write(f'3D outflow-tilt diagnostic\n')
        fh.write(f'{"="*60}\n')
        fh.write(f'run_dir       = {args.run_dir}\n')
        fh.write(f'n_iterations  = {len(rows)}\n')
        fh.write(f'mean_in_plane = {overall_mean_ip:.4f}\n')
        fh.write(f'\nVERDICT: {verdict}\n')
    print(f'Wrote {txt_path}')
    print()
    print(f'mean in-plane fraction: {overall_mean_ip:.4f}')
    print(f'VERDICT: {verdict}')


if __name__ == '__main__':
    main()
