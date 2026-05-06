"""
E-field diagnostic for the 3D pilot — check whether unphysical E_y is driving
the central-zone bulk-energy runaway observed in the central-convergence
diagnostic.

The hypothesis is that the alternating-polarity B_y Biermann seed creates a
strong gradient along y between adjacent ±B_y spots in 3D (a gradient that
doesn't exist in 2D-XZ). Through the Hall term E = -(J × B)/(n e), this can
drive a non-zero E_y component that accelerates protons axially without bound.

This diagnostic loads one or more field dumps at chosen iterations, computes
peak |E_x|, |E_y|, |E_z| values, and renders central slices of all three
components plus B-field magnitude for visual inspection.

Usage:
    python pb11_3d_efield_diagnostic.py --run-dir runs/pb11_3d_pilot_*/

Optional: --iter <N> to inspect a specific iteration. Default: pick the
iteration nearest the X-line acceleration peak (~22 ps in 2D).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import openpmd_api as opmd
except ImportError:
    print('ERROR: openpmd_api not installed in this env.')
    raise SystemExit(1)


def find_field_series(run_dir: Path) -> Path:
    """Locate the openPMD field series in the run directory.
    Same convention as pb11_3d_kink_growth.py."""
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out-dir', default=None, type=Path)
    p.add_argument('--iter', type=int, default=None,
                   help='Specific iteration to inspect. Default: nearest to '
                        'X-line peak ~22 ps (computed from time_unit_SI).')
    p.add_argument('--target-time-ps', type=float, default=22.0,
                   help='Target physical time for default iteration choice (ps)')
    p.add_argument('--n-iters', type=int, default=4,
                   help='Number of iterations to summarize in the timeseries '
                        '(spread across the run; default 4)')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / '3d_diag')
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = find_field_series(args.run_dir)
    print(f'Reading field series from: {series_path}')
    if series_path.is_dir():
        series = opmd.Series(str(series_path / 'openpmd_%T.h5'),
                             opmd.Access.read_only)
    else:
        series = opmd.Series(str(series_path), opmd.Access.read_only)

    iters = sorted(series.iterations)
    print(f'Found {len(iters)} field iterations: {iters[0]} → {iters[-1]}')

    # --- Compute time array for all iterations
    times_ps = []
    for it_idx in iters:
        it = series.iterations[it_idx]
        times_ps.append(it.time * it.time_unit_SI * 1e12)
    times_ps = np.array(times_ps)

    # --- Pick the inspection iteration
    if args.iter is not None:
        if args.iter not in iters:
            print(f'ERROR: iter {args.iter} not in available iterations')
            raise SystemExit(1)
        inspect_iter = args.iter
    else:
        idx = int(np.argmin(np.abs(times_ps - args.target_time_ps)))
        inspect_iter = iters[idx]
    inspect_time = times_ps[iters.index(inspect_iter)]
    print(f'\nInspection iteration: {inspect_iter} (t = {inspect_time:.2f} ps)')

    # --- Pick summary iterations: 4 spread across the run
    n_sum = min(args.n_iters, len(iters))
    sum_iter_idxs = np.linspace(0, len(iters)-1, n_sum, dtype=int)
    summary_iters = [iters[i] for i in sum_iter_idxs]
    print(f'Summary iterations: {summary_iters}')

    # --- Print the time-series summary table
    print()
    print(f'{"iter":>6}  {"t(ps)":>7}  {"|E_x|max":>11}  {"|E_y|max":>11}  '
          f'{"|E_z|max":>11}  {"|B|max":>10}  {"Ey/Ex":>8}  {"Ey/Ez":>8}')
    print('-' * 95)

    summary_rows = []
    for it_idx in summary_iters:
        it = series.iterations[it_idx]
        time_ps = it.time * it.time_unit_SI * 1e12

        try:
            E_mesh = it.meshes['E']
            B_mesh = it.meshes['B']
            Ex_chunk = E_mesh['x'][:, :, :]
            Ey_chunk = E_mesh['y'][:, :, :]
            Ez_chunk = E_mesh['z'][:, :, :]
            Bx_chunk = B_mesh['x'][:, :, :]
            By_chunk = B_mesh['y'][:, :, :]
            Bz_chunk = B_mesh['z'][:, :, :]
            series.flush()
            Ex = np.asarray(Ex_chunk, dtype=np.float64)
            Ey = np.asarray(Ey_chunk, dtype=np.float64)
            Ez = np.asarray(Ez_chunk, dtype=np.float64)
            Bx = np.asarray(Bx_chunk, dtype=np.float64)
            By = np.asarray(By_chunk, dtype=np.float64)
            Bz = np.asarray(Bz_chunk, dtype=np.float64)
        except Exception as e:
            print(f'  iter {it_idx}: error reading fields: {e}')
            continue

        Ex_max = float(np.max(np.abs(Ex)))
        Ey_max = float(np.max(np.abs(Ey)))
        Ez_max = float(np.max(np.abs(Ez)))
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        Bmag_max = float(np.max(Bmag))

        ratio_y_x = Ey_max / Ex_max if Ex_max > 0 else float('inf')
        ratio_y_z = Ey_max / Ez_max if Ez_max > 0 else float('inf')

        summary_rows.append({
            'iter': it_idx, 't_ps': time_ps,
            'Ex_max': Ex_max, 'Ey_max': Ey_max, 'Ez_max': Ez_max,
            'Bmag_max': Bmag_max,
            'ratio_y_x': ratio_y_x, 'ratio_y_z': ratio_y_z,
        })

        print(f'{it_idx:>6}  {time_ps:>7.2f}  {Ex_max:>11.3e}  {Ey_max:>11.3e}  '
              f'{Ez_max:>11.3e}  {Bmag_max:>10.3e}  {ratio_y_x:>8.2f}  {ratio_y_z:>8.2f}')

    # --- Now load the inspection iteration and render slices
    print(f'\nRendering slices for iteration {inspect_iter} ...')
    it = series.iterations[inspect_iter]
    E_mesh = it.meshes['E']
    B_mesh = it.meshes['B']

    Ex_chunk = E_mesh['x'][:, :, :]
    Ey_chunk = E_mesh['y'][:, :, :]
    Ez_chunk = E_mesh['z'][:, :, :]
    Bx_chunk = B_mesh['x'][:, :, :]
    By_chunk = B_mesh['y'][:, :, :]
    Bz_chunk = B_mesh['z'][:, :, :]
    series.flush()
    Ex = np.asarray(Ex_chunk, dtype=np.float64)
    Ey = np.asarray(Ey_chunk, dtype=np.float64)
    Ez = np.asarray(Ez_chunk, dtype=np.float64)
    Bx = np.asarray(Bx_chunk, dtype=np.float64)
    By = np.asarray(By_chunk, dtype=np.float64)
    Bz = np.asarray(Bz_chunk, dtype=np.float64)

    # Grid metadata from Mesh
    dx, dy, dz = E_mesh.grid_spacing
    x0, y0, z0 = E_mesh.grid_global_offset
    nz, ny, nx = Ex.shape  # WarpX/openPMD convention

    x_mm = (x0 + (np.arange(nx) + 0.5) * dx) * 1e3
    y_mm = (y0 + (np.arange(ny) + 0.5) * dy) * 1e3
    z_mm = (z0 + (np.arange(nz) + 0.5) * dz) * 1e3

    iy_mid = ny // 2
    ix_mid = nx // 2

    # --- Two slice planes:
    #     (1) central xz slice at iy = ny//2 — the "2D-equivalent" plane
    #     (2) central yz slice at ix = nx//2 — shows axial structure
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5),
                             gridspec_kw={'wspace': 0.35, 'hspace': 0.35})
    fig.suptitle(f'3D pilot — E-field diagnostic at iter {inspect_iter} '
                 f'(t = {inspect_time:.2f} ps)',
                 fontsize=12, y=0.99)

    # --- Top row: xz slice at central y
    cmap_E = 'RdBu_r'
    cmap_B = 'magma'

    extent_xz = [x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]]

    # E_x in xz plane
    ax = axes[0, 0]
    arr = Ex[:, iy_mid, :]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_xz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f'E_x (V/m), xz slice at y=0\nmax |E_x| = {vmax:.2e}')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # E_y in xz plane
    ax = axes[0, 1]
    arr = Ey[:, iy_mid, :]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_xz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f'E_y (V/m), xz slice at y=0\nmax |E_y| = {vmax:.2e}')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # E_z in xz plane
    ax = axes[0, 2]
    arr = Ez[:, iy_mid, :]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_xz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(f'E_z (V/m), xz slice at y=0\nmax |E_z| = {vmax:.2e}')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # |B| in xz plane
    ax = axes[0, 3]
    arr = Bmag[:, iy_mid, :]
    vmax = float(np.max(arr))
    im = ax.imshow(arr, extent=extent_xz, origin='lower',
                   cmap=cmap_B, vmin=0, vmax=vmax, aspect='equal')
    ax.set_title(f'|B| (T), xz slice at y=0\nmax |B| = {vmax:.2e}')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # --- Bottom row: yz slice at central x — shows axial (y) structure
    extent_yz = [y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]]

    ax = axes[1, 0]
    arr = Ex[:, :, ix_mid]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_yz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f'E_x (V/m), yz slice at x=0\nmax |E_x| = {vmax:.2e}')
    ax.set_xlabel('y (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    arr = Ey[:, :, ix_mid]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_yz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f'E_y (V/m), yz slice at x=0 ◀ KEY PANEL\nmax |E_y| = {vmax:.2e}')
    ax.set_xlabel('y (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    arr = Ez[:, :, ix_mid]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_yz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f'E_z (V/m), yz slice at x=0\nmax |E_z| = {vmax:.2e}')
    ax.set_xlabel('y (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 3]
    arr = By[:, :, ix_mid]
    vmax = float(np.max(np.abs(arr)))
    im = ax.imshow(arr, extent=extent_yz, origin='lower',
                   cmap=cmap_E, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f'B_y (T), yz slice at x=0\nmax |B_y| = {vmax:.2e}')
    ax.set_xlabel('y (mm)'); ax.set_ylabel('z (mm)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    png_path = out_dir / f'3d_efield_diagnostic_iter{inspect_iter:06d}.png'
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    print(f'\nWrote {png_path}')

    # --- Compute integrated metrics for the inspection iter
    Ex_max = float(np.max(np.abs(Ex)))
    Ey_max = float(np.max(np.abs(Ey)))
    Ez_max = float(np.max(np.abs(Ez)))
    Bmag_max = float(np.max(Bmag))

    # E-field energy density integrals
    eps0 = 8.854187817e-12  # F/m
    cell_vol = dx * dy * dz
    UEx = 0.5 * eps0 * np.sum(Ex**2) * cell_vol  # J
    UEy = 0.5 * eps0 * np.sum(Ey**2) * cell_vol
    UEz = 0.5 * eps0 * np.sum(Ez**2) * cell_vol
    UE_total = UEx + UEy + UEz
    UB_total = (1.0 / (2.0 * 4e-7 * np.pi)) * np.sum(Bmag**2) * cell_vol

    print()
    print(f'Inspection iteration {inspect_iter} (t = {inspect_time:.2f} ps) summary:')
    print(f'  Peak |E_x| = {Ex_max:.3e}  V/m')
    print(f'  Peak |E_y| = {Ey_max:.3e}  V/m   (target component for hypothesis)')
    print(f'  Peak |E_z| = {Ez_max:.3e}  V/m')
    print(f'  Peak |B|   = {Bmag_max:.3e}  T')
    print()
    print(f'  E_y / E_x peak ratio = {Ey_max/Ex_max:.3f}')
    print(f'  E_y / E_z peak ratio = {Ey_max/Ez_max:.3f}')
    print()
    print(f'  Field energy budget:')
    print(f'    U_E_x   = {UEx:.3e} J')
    print(f'    U_E_y   = {UEy:.3e} J   ← if dominant, hypothesis confirmed')
    print(f'    U_E_z   = {UEz:.3e} J')
    print(f'    U_E tot = {UE_total:.3e} J')
    print(f'    U_B tot = {UB_total:.3e} J')
    print(f'    U_E_y / U_E_total = {UEy/UE_total:.3f}')

    # --- Verdict on the hypothesis
    print()
    print('=' * 70)
    print('HYPOTHESIS TEST: is E_y unphysically dominant in 3D?')
    print('=' * 70)
    if Ey_max > 3 * max(Ex_max, Ez_max):
        verdict = ('CONFIRMED: |E_y| peak is more than 3× larger than max(|E_x|, |E_z|). '
                   'The alternating-polarity B_y seed is producing a strong axial gradient '
                   'that drives runaway E_y via the Hall term. The 3D pilot deck needs '
                   'reformulation: alternate B_z (in-plane) instead of B_y (out-of-plane), '
                   'or shrink Ly so the alternating patches do not develop axial structure.')
    elif UEy / UE_total > 0.5:
        verdict = ('CONFIRMED (energy basis): more than 50% of the E-field energy is in the '
                   'E_y component. Even if peak ratios look reasonable, the integrated E_y '
                   'is dominant, indicating a systemic axial-field problem.')
    elif Ey_max > 1.5 * max(Ex_max, Ez_max):
        verdict = ('PARTIAL: |E_y| is somewhat elevated relative to the in-plane components. '
                   'Suggests a real but non-dominant axial-field issue. The 3D pilot may '
                   'still be useful for the geometry result, but bulk-energy comparisons '
                   'with 2D should not be drawn from this data.')
    else:
        verdict = ('REJECTED: |E_y| is comparable to or smaller than |E_x|, |E_z|. The '
                   'central-zone runaway is not coming from an unphysical E_y. We need '
                   'to look elsewhere — likely candidates: numerical heating from low NPPC '
                   '(16) at high gradients, current-support BC scaling differently in 3D, '
                   'or a CFL marginality at the chosen dt. Run the same diagnostic at '
                   'multiple iterations to look for trends.')
    print(verdict)
    print('=' * 70)

    # --- Write text report
    txt_path = out_dir / f'3d_efield_diagnostic_iter{inspect_iter:06d}.txt'
    with open(txt_path, 'w') as fh:
        fh.write('3D pilot — E-field diagnostic\n')
        fh.write('=' * 60 + '\n')
        fh.write(f'run_dir = {args.run_dir}\n')
        fh.write(f'inspection iter = {inspect_iter} (t = {inspect_time:.2f} ps)\n')
        fh.write(f'\nField timeseries summary:\n')
        fh.write(f'{"iter":>6}  {"t(ps)":>7}  {"|E_x|max":>11}  {"|E_y|max":>11}  '
                 f'{"|E_z|max":>11}  {"|B|max":>10}\n')
        for r in summary_rows:
            fh.write(f'{r["iter"]:>6}  {r["t_ps"]:>7.2f}  {r["Ex_max"]:>11.3e}  '
                     f'{r["Ey_max"]:>11.3e}  {r["Ez_max"]:>11.3e}  '
                     f'{r["Bmag_max"]:>10.3e}\n')
        fh.write(f'\nInspection iteration totals:\n')
        fh.write(f'  Peak |E_x| = {Ex_max:.3e} V/m\n')
        fh.write(f'  Peak |E_y| = {Ey_max:.3e} V/m\n')
        fh.write(f'  Peak |E_z| = {Ez_max:.3e} V/m\n')
        fh.write(f'  Peak |B|   = {Bmag_max:.3e} T\n')
        fh.write(f'  E_y / E_x  = {Ey_max/Ex_max:.3f}\n')
        fh.write(f'  E_y / E_z  = {Ey_max/Ez_max:.3f}\n')
        fh.write(f'  U_E_x      = {UEx:.3e} J\n')
        fh.write(f'  U_E_y      = {UEy:.3e} J\n')
        fh.write(f'  U_E_z      = {UEz:.3e} J\n')
        fh.write(f'  U_E_y/U_E  = {UEy/UE_total:.3f}\n')
        fh.write(f'\nVERDICT: {verdict}\n')
    print(f'Wrote {txt_path}')


if __name__ == '__main__':
    main()
