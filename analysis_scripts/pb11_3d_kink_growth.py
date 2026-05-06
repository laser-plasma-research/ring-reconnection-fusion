#!/usr/bin/env python3
"""
pb11_3d_kink_growth.py — Kink-mode growth at the X-lines.

Question this answers: do the eight X-lines remain straight tubes along y over
the active phase of the run, or does the kink instability bend them on a
timescale shorter than the proton transit from X-line to ring center?

Method:
  1. For each field dump, locate the eight X-line zero-crossings of B_y(x, y=mid, z).
  2. At each X-line, extract B_y(y) along the y-axis at fixed (x, z) and
     compute the rms deviation from straight (i.e. from the y-mean).
  3. Track this rms vs time. If it grows exponentially, fit a growth rate
     gamma_kink and report gamma_kink * t_active where t_active is the duration
     of the active reconnection phase (~250 ps for LD).
  4. Threshold: gamma_kink * t_active < 1 means the kink instability does not
     develop within the active phase, i.e. the 2D reduction is robust.

Inputs:
  --run-dir    Path to the openPMD output directory of a 3D pilot run
  --out-dir    Where to write the diagnostic (default: <run-dir>/3d_diag)

Outputs:
  3d_kink_growth.csv    rms(B_y - <B_y>_y) at each X-line vs time
  3d_kink_growth.png    log-scale plot with growth-rate fit annotated
  3d_kink_growth.txt    summary report (gamma_kink, gamma * t_active, verdict)

Usage:
  python pb11_3d_kink_growth.py --run-dir runs/pb11_3d_pilot_*

This script is intentionally self-contained (no imports from the simulation
deck) so it runs anywhere that has openPMD-api and matplotlib.
"""

import argparse
import os
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


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path,
                   help='Path to 3D-pilot run directory containing diags/')
    p.add_argument('--out-dir', default=None, type=Path,
                   help='Output directory (default: <run-dir>/3d_diag)')
    p.add_argument('--n-spots', type=int, default=8,
                   help='Number of laser spots / X-lines (default: 8)')
    p.add_argument('--ring-radius-um', type=float, default=2400.0,
                   help='Ring radius in microns (default: 2400)')
    p.add_argument('--t-active-ps', type=float, default=250.0,
                   help='Active-phase duration in ps (default: 250)')
    return p.parse_args()


def find_field_series(run_dir: Path) -> Path:
    """Locate the openPMD field series in the run directory."""
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


def x_line_positions(n_spots: int, ring_radius_m: float):
    """Return list of (x, z) for the n_spots X-line locations.
    X-lines sit at angles offset by π/n_spots from the spots, on the chord
    midpoint circle of radius R cos(π/n_spots).
    """
    x_radius = ring_radius_m * np.cos(np.pi / n_spots)
    out = []
    for k in range(n_spots):
        angle = 2.0 * np.pi * k / n_spots + np.pi / n_spots
        out.append((x_radius * np.cos(angle), x_radius * np.sin(angle)))
    return out


def sample_along_y(B_y_3d, ix, iz):
    """Return B_y(y) at fixed (ix, iz). B_y_3d is indexed [iz, iy, ix]."""
    return B_y_3d[iz, :, ix]


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.run_dir / '3d_diag')
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = find_field_series(args.run_dir)
    print(f'Reading field series from: {series_path}')
    if series_path.is_dir():
        # OpenPMD pattern: directory containing per-iteration files
        series = opmd.Series(str(series_path / 'openpmd_%T.h5'),
                             opmd.Access.read_only)
    else:
        series = opmd.Series(str(series_path), opmd.Access.read_only)

    iters = sorted(series.iterations)
    print(f'Found {len(iters)} iterations: {iters[0]} → {iters[-1]}')

    n_spots = args.n_spots
    ring_r_m = args.ring_radius_um * 1e-6
    xlines = x_line_positions(n_spots, ring_r_m)

    rows = []  # (time_ps, iteration, [rms per X-line])

    for it_idx in iters:
        it = series.iterations[it_idx]
        time_s = it.time * it.time_unit_SI
        time_ps = time_s * 1e12

        try:
            B_mesh = it.meshes['B']
            By_record = B_mesh['y']
        except KeyError:
            print(f'  iter {it_idx}: no B/y record, skipping')
            continue

        # Read full 3D array. WarpX/openPMD: shape (nz, ny, nx).
        By = By_record[:, :, :]
        series.flush()
        By = np.asarray(By, dtype=np.float64)

        # Grid spacing — these are properties of the Mesh, not the Mesh_Record_Component
        dx, dy, dz = B_mesh.grid_spacing
        x0, y0, z0 = B_mesh.grid_global_offset
        nz, ny, nx = By.shape

        # rms deviation from y-mean at each X-line (proxy for kink amplitude)
        rms_per_xline = []
        for (x_t, z_t) in xlines:
            ix = int(round((x_t - x0) / dx))
            iz = int(round((z_t - z0) / dz))
            if 0 <= ix < nx and 0 <= iz < nz:
                profile = sample_along_y(By, ix, iz)
                rms = float(np.std(profile))
            else:
                rms = float('nan')
            rms_per_xline.append(rms)

        rows.append((time_ps, it_idx, rms_per_xline))
        print(f'  t = {time_ps:7.2f} ps  rms(B_y(y)) per X-line: '
              f'mean={np.nanmean(rms_per_xline):.3e}  '
              f'max={np.nanmax(rms_per_xline):.3e}')

    # Write CSV
    csv_path = out_dir / '3d_kink_growth.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['time_ps', 'iteration'] +
                   [f'rms_xline_{k}' for k in range(n_spots)])
        for (t, it, rms) in rows:
            w.writerow([f'{t:.4f}', it] + [f'{r:.6e}' for r in rms])
    print(f'Wrote {csv_path}')

    # Fit exponential growth on the mean rms across X-lines.
    # IMPORTANT: skip the X-line establishment phase. The seed B-field develops
    # along y over the first ~15-20 ps; rms growth in that window is field
    # establishment, not kink instability. We fit only the post-establishment
    # window where the X-lines are spatially resolved structures with a
    # well-defined baseline.
    times = np.array([r[0] for r in rows])
    means = np.array([np.nanmean(r[2]) for r in rows])
    t_run_max = float(times.max()) if len(times) else 0.0

    # Skip establishment phase: t > 20 ps OR last 75% of run, whichever leaves
    # more data. For short runs, this avoids dropping the fit entirely.
    fit_window_start = min(20.0, 0.25 * t_run_max)
    valid = (means > 0) & np.isfinite(means) & (times >= fit_window_start)

    if valid.sum() >= 3:
        log_means = np.log(means[valid])
        coef = np.polyfit(times[valid], log_means, 1)
        gamma_per_ps = float(coef[0])
        intercept = float(coef[1])
    else:
        gamma_per_ps = float('nan')
        intercept = float('nan')

    # Report TWO numbers:
    #   gamma_t_run     = γ × actual_run_duration  (what the data supports)
    #   gamma_t_active  = γ × t_active_arg          (extrapolation; flagged below)
    gamma_t_run = (gamma_per_ps * (t_run_max - fit_window_start)
                   if np.isfinite(gamma_per_ps) else float('nan'))
    gamma_t_active = (gamma_per_ps * args.t_active_ps
                      if np.isfinite(gamma_per_ps) else float('nan'))

    # Flag when the run is too short for a reliable kink-vs-reconnection
    # separation. Kink growth needs to be measured on a quasi-equilibrium
    # current sheet, which means the active reconnection phase (peaks ~22 ps
    # in 2D, equilibrates by ~150-200 ps) needs to be largely past. If the
    # run is shorter than 100 ps, the late-time rise is contaminated by the
    # active phase and the verdict is provisional.
    run_too_short = t_run_max < 100.0

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for k in range(n_spots):
        rms_k = np.array([r[2][k] for r in rows])
        ax.semilogy(times, rms_k, alpha=0.35, lw=0.7,
                    label=f'X-line {k}' if k < 3 else None)
    ax.semilogy(times, means, color='black', lw=1.5,
                label='mean across X-lines')
    # Shade the establishment window we excluded from the fit
    ax.axvspan(0, fit_window_start, color='gray', alpha=0.18,
               label=f'establishment phase (excluded from fit, t < {fit_window_start:.0f} ps)')
    if np.isfinite(gamma_per_ps):
        fit_t = np.linspace(fit_window_start, t_run_max, 50)
        fit_y = np.exp(intercept + gamma_per_ps * fit_t)
        ax.semilogy(fit_t, fit_y, '--', color='red', lw=1.2,
                    label=f'fit (post-establishment): γ = {gamma_per_ps:.4f} /ps')
    ax.set_xlabel('time (ps)')
    ax.set_ylabel(r'rms$\,(B_y(y)\,-\,\langle B_y\rangle_y)$  (T)')
    title_lines = ['Kink-mode amplitude at X-lines (3D pilot)']
    if np.isfinite(gamma_t_run):
        title_lines.append(
            f'γ × Δt_run = {gamma_t_run:.3f}  '
            f'(over {t_run_max - fit_window_start:.0f} ps fit window)'
        )
    ax.set_title('\n'.join(title_lines))
    ax.legend(loc='best', frameon=False, fontsize=7.5)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    png_path = out_dir / '3d_kink_growth.png'
    plt.savefig(png_path, dpi=140)
    print(f'Wrote {png_path}')

    # Verdict report. Two cases:
    #   1. Run is long enough (>= 100 ps) for kink to be separable from
    #      reconnection-active-phase reorganization → use γ × t_active for verdict
    #   2. Run is short (< 100 ps) → verdict is provisional, based on γ × Δt_run.
    #      In this case the late-time rise is contaminated by active reconnection
    #      and we cannot cleanly attribute it to kink instability.
    if not np.isfinite(gamma_per_ps):
        verdict = 'INDETERMINATE: insufficient data points or non-monotonic rms.'
    elif run_too_short:
        # Short run: report γ × Δt_run honestly, with explicit caveat
        if gamma_t_run < 0.5:
            verdict = (f'STABLE (provisional, short run): γ × Δt_run = {gamma_t_run:.3f} '
                       'after excluding the establishment phase. The X-line is not '
                       'developing kink modes on the resolved timescale, but the run '
                       f'covers only {t_run_max:.0f} ps which is shorter than the active '
                       'reconnection phase (~150-200 ps). A longer pilot is required to '
                       'cleanly separate kink instability from active-phase reorganization.')
        elif gamma_t_run < 1.0:
            verdict = (f'MARGINAL (provisional, short run): γ × Δt_run = {gamma_t_run:.3f}. '
                       f'Late-time rms growth observed over {t_run_max:.0f} ps but cannot be '
                       'cleanly distinguished from active-phase X-line reorganization. '
                       'A run extending to >= 200 ps is needed for a definitive verdict.')
        else:
            verdict = (f'GROWTH OBSERVED (provisional, short run): γ × Δt_run = {gamma_t_run:.3f}. '
                       f'Significant rms growth observed in the {t_run_max:.0f} ps window. '
                       'CAVEAT: this run is shorter than the active reconnection phase, so '
                       'this growth could reflect either kink instability OR active-phase '
                       'reorganization of the current sheet. A longer run is required to '
                       'distinguish; pure kink instability would continue growing once the '
                       'active phase ends, while reorganization should saturate.')
    else:
        # Long run: standard kink verdict using γ × t_active
        if gamma_t_active < 0.5:
            verdict = (f'STABLE: γ × t_active = {gamma_t_active:.3f}. Kink growth does '
                       'not develop on the active-phase timescale. 2D reduction is robust.')
        elif gamma_t_active < 1.0:
            verdict = (f'MARGINAL: γ × t_active = {gamma_t_active:.3f}. Kink growth is '
                       'sub-e-folding over the active phase. Effect is bounded but should '
                       'be reported as a limitation.')
        else:
            verdict = (f'UNSTABLE: γ × t_active = {gamma_t_active:.3f}. Kink growth has '
                       'time to develop ≥1 e-folding over the active phase. 3D effects '
                       'materially modify the 2D picture.')

    txt_path = out_dir / '3d_kink_growth.txt'
    with open(txt_path, 'w') as fh:
        fh.write(f'3D kink-growth diagnostic\n')
        fh.write(f'{"="*60}\n')
        fh.write(f'run_dir          = {args.run_dir}\n')
        fh.write(f'n_iterations     = {len(rows)}\n')
        fh.write(f'time_range_ps    = {times[0]:.2f} → {times[-1]:.2f}\n')
        fh.write(f'fit_window_start = {fit_window_start:.2f} ps '
                 f'(establishment phase excluded)\n')
        fh.write(f'fit_window_end   = {t_run_max:.2f} ps\n')
        fh.write(f'gamma_per_ps     = {gamma_per_ps:.6e}\n')
        fh.write(f'gamma * dt_run   = {gamma_t_run:.4f}  (data-supported)\n')
        fh.write(f't_active_ps      = {args.t_active_ps:.2f}\n')
        fh.write(f'gamma * t_active = {gamma_t_active:.4f}  '
                 f'(extrapolation; only meaningful for runs > 100 ps)\n')
        fh.write(f'run_too_short    = {run_too_short}\n')
        fh.write(f'\nVERDICT: {verdict}\n')
    print(f'Wrote {txt_path}')
    print()
    print(f'γ × Δt_run    = {gamma_t_run:.4f}  (data-supported, fit window {fit_window_start:.0f}–{t_run_max:.0f} ps)')
    if not run_too_short:
        print(f'γ × t_active  = {gamma_t_active:.4f}  (extrapolated to {args.t_active_ps:.0f} ps)')
    print(f'VERDICT: {verdict}')


if __name__ == '__main__':
    main()
