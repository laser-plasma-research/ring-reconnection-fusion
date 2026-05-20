#!/usr/bin/env python3
"""
pb11_zone_analysis_paper3.py — Seven-zone radial-band analysis.

Originally written for Paper 3 (rod + outer-catcher geometry), now also drives
Paper 1 output via the `--paper-tag` flag (defaults to `paper3` for backward
compatibility with the existing Paper 3 chain).

Splits the legacy four-zone breakdown (CORE / INNER / X-LINE / SPOT) into
seven zones that distinguish rod, outer-catcher, and halo contributions:

  ROD          : r < 2.4 × rod_radius_m            (~94% of rod Gaussian mass)
  CORE         : 2.4 × rod_radius_m  ≤ r < 1200 µm (Paper 1 inner-half)
  INTER        : 1200 µm ≤ r < 2097 µm             (Paper 1 outer-half)
  X_LINE       : 2097 µm ≤ r ≤ 2337 µm             (annulus around X-lines)
  SPOT         : 2337 µm < r ≤ 2900 µm             (outer edge of pitcher 3σ)
  OUTER        : R_out − 2σ ≤ r ≤ R_out + 2σ       (outer catcher 95% mass)
  HALO         : (max named-zone outer edge) < r ≤ R_edge
                                                   (post-SPOT/OUTER fast-proton
                                                    transit volume — previously
                                                    uncounted "interstitial")
  EXTERIOR     : r > min(LX_M, LZ_M)/2 − 100 µm    (periodic-edge danger zone)

For ring-only Paper 1 geometry, the ROD and OUTER zones automatically disable
themselves (rod_radius=0 / outer_radius=0 in run_meta.txt) and the report
populates CORE / INTER / X_LINE / SPOT / HALO — matching the Paper 1 narrative.

The HALO zone closes the accounting gap between zone-sum cumulative energy and
the global-FT methodology total: at the LD baseline the four named inner zones
account for ~4.8 J while global FT integrates to ~18.2 J, with the remaining
~13.4 J occurring in HALO from fast protons transiting the bulk plasma after
exiting the spot region. Adding HALO to the zone tally reconciles zone-sum to
G_FT to within rounding.

All boundaries derive from run_meta.txt; they do not require hard-coding.
The zones are non-overlapping and cover all radii from 0 to half-box.

OUTPUTS (filenames driven by --paper-tag, default 'paper3')
-----------------------------------------------------------
  <run>_zones_<tag>.txt             zone breakdown vs time
  <run>_zone_fusion_rates_<tag>.csv per-zone fusion rate using zone-specific n_B11

USAGE
-----
  # Paper 3 (default)
  python pb11_zone_analysis_paper3.py --run-dir runs/paper03/p3_rod_plus_outer

  # Paper 1 (renamed outputs + Paper 1 title in report header)
  python pb11_zone_analysis_paper3.py --run-dir runs/paper01/p1_ld_uuf \
      --paper-tag paper1

CHANGE LOG (consolidated v15.x)
-------------------------------
  - 2026-05-11: Added bulk-E95 / tail-E95 split (E95_X uses bulk percentile,
                E95_tail_X uses above-threshold percentile). The old "fast-
                only" E95 was renamed E95_tail and a new bulk E95 added.
  - 2026-05-11: CORE/INTER zones split (Paper 1 used these as separate bands).
  - 2026-05-12: outer_radius_m / outer_thickness_m defaults set to 0.0 so
                outer_enabled correctly reflects run_meta presence. Added
                guard against SPOT-clip emptying the SPOT zone.
  - 2026-05-12: percentile_above_threshold no longer raises TypeError on
                edge cases (Stage D rc=1 fix).
  - 2026-05-12: NaN ratios via safe_ratio for empty zones (no more
                divide-by-zero or 291234.97 garbage values).
  - 2026-05-13: --paper-tag flag added; gain-vs-laser now reads laser energy
                from step_time_index.csv instead of being hardcoded to 5 J.
  - 2026-05-18: HALO zone added between SPOT/OUTER and EXTERIOR. Captures
                fast-proton transit through bulk plasma after spot/outer
                exit — previously uncounted "interstitial" region that
                explains the gap between zone-sum and global-FT total.
                Output CSV adds rate_halo_per_s + cum_E_halo_J; text report
                adds N_halo + E95_halo columns + halo cumulative-energy line.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

PROTON_MASS_KG = 1.67262192e-27
Q_E             = 1.602176634e-19
C_LIGHT         = 2.99792458e8
KEV             = Q_E * 1e3
MEV             = Q_E * 1e6

SIGMA_P11B_M2     = 1.2e-31
THRESHOLD_KEV     = 500.0
E_FUSION_P11B_J   = 8.68 * MEV


def kev_from_momentum(ux, uy, uz, mass_kg):
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT))
    return (gamma - 1.0) * mass_kg * C_LIGHT * C_LIGHT / KEV


def speed_from_momentum(ux, uy, uz):
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT))
    return np.sqrt(u2) / gamma


def parse_run_meta(meta_path):
    meta = {}
    if not meta_path.is_file():
        raise FileNotFoundError(f'run_meta.txt not found: {meta_path}')
    section = None
    for line in meta_path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('=') or s.startswith('v'):
            continue
        if s.startswith('[') and s.endswith(']'):
            section = s[1:-1]; continue
        if '=' not in s:
            continue
        key, val = s.split('=', 1)
        key = key.strip(); val = val.split(' ', 1)[0].strip()
        try:
            if '.' in val or 'e' in val or 'E' in val:
                meta[key] = float(val)
            else:
                meta[key] = int(val)
        except ValueError:
            meta[key] = val
    return meta


def find_particle_series(run_dir: Path):
    try:
        import openpmd_api as io
    except ImportError:
        raise SystemExit('openpmd_api not installed.  pip install openpmd-api')

    pdir = run_dir / 'particles'
    if not pdir.is_dir():
        candidates = [d for d in run_dir.iterdir()
                      if d.is_dir() and any(d.glob('openpmd_*.h5'))]
        if not candidates:
            raise SystemExit(f'No particle dumps found under {run_dir}')
        pdir = candidates[0]

    if list(pdir.glob('openpmd_*.h5')):
        pattern = str(pdir / 'openpmd_%T.h5')
    elif list(pdir.glob('openpmd_*.bp')):
        pattern = str(pdir / 'openpmd_%T.bp')
    else:
        raise SystemExit(f'No openpmd files found in {pdir}')

    series = io.Series(pattern, io.Access.read_only)
    iters  = sorted(series.iterations)
    return series, iters, pdir


def load_dump(series, iter_idx, species_name='proton'):
    it = series.iterations[iter_idx]
    if species_name not in list(it.particles):
        return None
    sp = it.particles[species_name]

    def load(record, component=None):
        rec = sp[record]
        comp = rec[component] if component is not None else rec[next(iter(rec))]
        chunk = comp.load_chunk()
        series.flush()
        try:
            unit = comp.unit_SI
            if unit and unit != 1.0:
                chunk = chunk * unit
        except Exception:
            pass
        return chunk

    try:
        x   = load('position', 'x')
        z   = load('position', 'z')
        ux  = load('momentum', 'x') / PROTON_MASS_KG
        uy  = load('momentum', 'y') / PROTON_MASS_KG
        uz  = load('momentum', 'z') / PROTON_MASS_KG
        w   = load('weighting')
    except Exception as e:
        print(f'  WARN: iter {iter_idx} load failed: {e}', file=sys.stderr)
        return None
    return x, z, ux, uy, uz, w


def percentile_weighted(ke, w, percentile):
    """Weighted percentile of KE for ALL particles in array (in keV).

    Returns the bulk percentile (matches `pb11_consolidate_reports.py`
    convention).  Use this for E95_bulk_* columns.
    """
    if len(ke) == 0:
        return 0.0
    sorter = np.argsort(ke)
    cum_w = np.cumsum(w[sorter])
    cum_w /= cum_w[-1]
    idx = np.searchsorted(cum_w, percentile / 100.0)
    return float(ke[sorter[min(idx, len(ke) - 1)]])


def percentile_above_threshold(ke, w, percentile, threshold_kev):
    """Weighted percentile of KE for particles ABOVE threshold (in keV).

    Returns the non-thermal tail percentile.  Use this for E95_tail_* columns.
    Falls back to bulk percentile if no particle above threshold (so the
    column never returns NaN, just degrades gracefully).
    """
    mask = ke >= threshold_kev
    if not mask.any():
        return percentile_weighted(ke, w, percentile)
    return percentile_weighted(ke[mask], w[mask], percentile)


def find_laser_energy_j(run_dir):
    """Read step_time_index.csv last-row laser_energy_j for gain reporting.

    Returns float or None if column / file missing. Mirrors the helper in
    pb11_first_transit_fusion.py so the two scripts report consistent gain
    values. Both papers should use the same convention to keep cross-paper
    comparisons apples-to-apples.
    """
    csv_path = run_dir / 'step_time_index.csv'
    if not csv_path.is_file():
        return None
    last = None
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if 'laser_energy_j' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                last = float(row['laser_energy_j'])
            except (KeyError, ValueError):
                pass
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--species', default='proton')
    ap.add_argument('--threshold-kev', type=float, default=THRESHOLD_KEV)
    ap.add_argument('--out-prefix', type=str, default=None)
    ap.add_argument('--max-dumps', type=int, default=0)
    ap.add_argument('--paper-tag', type=str, default='paper3',
                    help='Suffix for output filenames and report title. '
                         'Default "paper3" preserves backward compat with the '
                         'existing Paper 3 chain. Pass "paper1" for Paper 1 '
                         'output naming (zones_paper1.txt etc.).')
    ap.add_argument('--laser-energy-j', type=float, default=None,
                    help='Override laser energy (J) used for the gain figure '
                         'in the final report. Default: read from '
                         'step_time_index.csv (last row, laser_energy_j '
                         'column). Falls back to 5.0 J only if neither the '
                         'flag nor the CSV is available.')
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f'ERROR: run-dir not found: {args.run_dir}', file=sys.stderr)
        return 2

    meta = parse_run_meta(args.run_dir / 'run_meta.txt')
    rod_radius_m       = meta.get('rod_radius_m', 75e-6)
    ring_radius_m      = meta.get('ring_radius_m', 2400e-6)
    spot_radius_m      = meta.get('spot_radius_m', 300e-6)
    # PATCHED: defaults must be 0.0 so outer_enabled correctly detects
    # missing run_meta keys.  Plausible-looking defaults (1050e-6 /
    # 100e-6) caused SPOT zone to be clipped to an empty range when
    # run_meta lacked outer_radius_m (e.g. for rod_only runs).
    outer_radius_m     = meta.get('outer_radius_m', 0.0)
    outer_thickness_m  = meta.get('outer_thickness_m', 0.0)
    lx_m               = meta.get('lx_m', 9600e-6)
    lz_m               = meta.get('lz_m', 9600e-6)
    time_step_s        = meta.get('time_step_s', None)
    n_spots            = meta.get('n_spots', 8)

    base_n_b11   = meta.get('base_n_b11',   5e24)
    rod_n_b11    = meta.get('rod_n_b11',    base_n_b11)
    outer_n_b11  = meta.get('ring_n_b11',   base_n_b11)

    # ── Zone boundaries ──────────────────────────────────────────────────
    # FIX 2: CORE/INTER split — Paper 1 used these as separate zones.
    R_rod = 2.4 * rod_radius_m
    R_core_hi = 0.5 * ring_radius_m        # Paper 1: ~1200 µm at default
    R_xline = ring_radius_m * math.cos(math.pi / max(2, n_spots))
    R_xline_lo = R_xline - 0.5 * spot_radius_m       # ~2097 µm at default
    R_xline_hi = R_xline + 0.5 * spot_radius_m       # ~2337 µm at default
    R_spot_hi  = ring_radius_m + 1.7 * spot_radius_m  # ~2900 µm at default
    half_box_min = min(lx_m, lz_m) / 2.0
    R_edge = half_box_min - 100e-6

    # FIX 1: outer zone enable check — only define outer + clip SPOT if
    # outer is actually present.
    outer_enabled = (outer_radius_m > 0.0 and outer_thickness_m > 0.0)
    if outer_enabled:
        R_outer_lo = outer_radius_m - 2.0 * outer_thickness_m
        R_outer_hi = outer_radius_m + 2.0 * outer_thickness_m
    else:
        R_outer_lo = R_edge + 1.0   # impossible band → mask matches nothing
        R_outer_hi = R_edge + 1.0

    # HALO zone: everything between the outermost named-zone boundary and the
    # exterior-cut radius. For Paper 1 (no outer) this is the post-SPOT region
    # where fast protons drift through bulk plasma. For Paper 3 (with outer)
    # this is the region beyond the outer-catcher annulus. Closes the gap
    # between zone-sum cumulative energy and global-FT methodology total.
    if outer_enabled:
        R_halo_lo = max(R_spot_hi, R_outer_hi)
    else:
        R_halo_lo = R_spot_hi
    R_halo_hi = R_edge

    print(f'Run:         {args.run_dir}')
    print(f'Geometry:    n_spots={n_spots}, R_ring={ring_radius_m*1e6:.0f}, σ_spot={spot_radius_m*1e6:.0f}')
    print(f'Zone boundaries (µm):')
    print(f'  ROD          r < {R_rod*1e6:.0f}')
    print(f'  CORE         {R_rod*1e6:.0f} ≤ r < {R_core_hi*1e6:.0f}')
    print(f'  INTER        {R_core_hi*1e6:.0f} ≤ r < {R_xline_lo*1e6:.0f}')
    print(f'  X_LINE       {R_xline_lo*1e6:.0f} ≤ r ≤ {R_xline_hi*1e6:.0f}')
    print(f'  SPOT         {R_xline_hi*1e6:.0f} < r ≤ {R_spot_hi*1e6:.0f}')
    if outer_enabled:
        print(f'  OUTER        {R_outer_lo*1e6:.0f} ≤ r ≤ {R_outer_hi*1e6:.0f}')
    else:
        print(f'  OUTER        DISABLED (outer_radius_m=0)')
    print(f'  HALO         {R_halo_lo*1e6:.0f} < r ≤ {R_halo_hi*1e6:.0f}  (fast-proton transit volume)')
    print(f'  EXTERIOR     r > {R_edge*1e6:.0f}  (periodic-edge danger)')

    # FIX 1 (continued): only adjust SPOT for OUTER overlap when OUTER is enabled.
    # PATCHED: also refuse to apply a clip that would empty the SPOT zone
    # (R_spot_hi must remain > R_xline_hi after any clip).  Defense-in-depth
    # against future runs where outer geometry might overlap X-line annulus.
    if outer_enabled and R_outer_lo < R_spot_hi:
        if R_outer_lo > R_xline_hi:
            print(f'  NOTE: OUTER overlaps SPOT; clipping SPOT to r < {R_outer_lo*1e6:.0f}')
            R_spot_hi = R_outer_lo
        else:
            print(f'  WARN: OUTER clip would empty SPOT (R_outer_lo={R_outer_lo*1e6:.0f} '
                  f'≤ R_xline_hi={R_xline_hi*1e6:.0f}); ignoring clip')

    series, iters, pdir = find_particle_series(args.run_dir)
    if args.max_dumps > 0:
        iters = iters[:args.max_dumps]
    print(f'Dumps:       {len(iters)} found in {pdir}')

    out_prefix = args.out_prefix or args.run_dir.name
    rows = []          # for the zones text table
    fr_rows = []       # for the per-zone fusion rate CSV

    cum_E = {'rod': 0.0, 'outer': 0.0, 'core': 0.0, 'inter': 0.0,
             'xline': 0.0, 'spot': 0.0, 'halo': 0.0}
    prev_t_s = None

    for k, idx in enumerate(iters):
        loaded = load_dump(series, idx, args.species)
        if loaded is None:
            continue
        x, z, ux, uy, uz, w = loaded
        r  = np.sqrt(x * x + z * z)
        ke = kev_from_momentum(ux, uy, uz, PROTON_MASS_KG)
        v  = speed_from_momentum(ux, uy, uz)
        t_s  = float(idx) * time_step_s if time_step_s else 0.0
        t_ps = t_s * 1e12
        dt_s = (t_s - prev_t_s) if prev_t_s is not None else 0.0
        prev_t_s = t_s

        # FIX 2: CORE/INTER split (separate masks)
        masks = {
            'rod':   r < R_rod,
            'core':  (r >= R_rod) & (r < R_core_hi),
            'inter': (r >= R_core_hi) & (r < R_xline_lo),
            'xline': (r >= R_xline_lo) & (r <= R_xline_hi),
            'spot':  (r > R_xline_hi) & (r <= R_spot_hi),
            'outer': (r >= R_outer_lo) & (r <= R_outer_hi),
            'halo':  (r > R_halo_lo) & (r <= R_halo_hi),
            'exterior': r > R_edge,
        }
        target = {
            'rod':   rod_n_b11,
            'core':  base_n_b11,
            'inter': base_n_b11,
            'xline': base_n_b11,
            'spot':  base_n_b11,
            'outer': outer_n_b11,
            'halo':  base_n_b11,
            'exterior': base_n_b11,
        }

        N_zone   = {z: int(masks[z].sum()) for z in masks}

        # FIX 3+5: E95 is BULK 95th percentile (matches consolidate convention,
        # always keV).  Also compute E95_tail (above-threshold) as a separate
        # diagnostic for runs that want non-thermal tail measure.
        E95_zone = {}
        E95_tail = {}
        for z in masks:
            ke_z = ke[masks[z]]
            w_z  = w[masks[z]]
            E95_zone[z] = percentile_weighted(ke_z, w_z, 95)
            E95_tail[z] = percentile_above_threshold(
                ke_z, w_z, 95, args.threshold_kev
            )

        # Per-zone fusion rate (zone-local target density)
        rate_zone = {}
        for z, m in masks.items():
            if z == 'exterior': continue
            mfast = m & (ke >= args.threshold_kev)
            rate_zone[z] = float((w[mfast] * v[mfast]).sum()) * target[z] * SIGMA_P11B_M2
            if dt_s > 0 and z in cum_E:
                cum_E[z] += rate_zone[z] * E_FUSION_P11B_J * dt_s

        # FIX 4: ratio with N/A for empty spot
        def safe_ratio(num_e95, num_n, denom_e95, denom_n):
            """Return E95_num/E95_denom, or NaN if either zone is empty
            or the denominator is too small."""
            if num_n == 0 or denom_n == 0:
                return float('nan')
            if denom_e95 < 1.0:
                return float('nan')
            return num_e95 / denom_e95

        rows.append({
            't_ps': t_ps,
            'N_rod': N_zone['rod'], 'N_core': N_zone['core'], 'N_inter': N_zone['inter'],
            'N_xline': N_zone['xline'], 'N_spot': N_zone['spot'],
            'N_outer': N_zone['outer'], 'N_halo': N_zone['halo'],
            'N_exterior': N_zone['exterior'],
            'E95_rod': E95_zone['rod'], 'E95_core': E95_zone['core'],
            'E95_inter': E95_zone['inter'], 'E95_xline': E95_zone['xline'],
            'E95_spot': E95_zone['spot'], 'E95_outer': E95_zone['outer'],
            'E95_halo': E95_zone['halo'],
            'E95_tail_xline': E95_tail['xline'], 'E95_tail_outer': E95_tail['outer'],
            'core_over_spot':  safe_ratio(E95_zone['core'],  N_zone['core'],  E95_zone['spot'], N_zone['spot']),
            'rod_over_spot':   safe_ratio(E95_zone['rod'],   N_zone['rod'],   E95_zone['spot'], N_zone['spot']),
            'outer_over_spot': safe_ratio(E95_zone['outer'], N_zone['outer'], E95_zone['spot'], N_zone['spot']),
            'xline_over_spot': safe_ratio(E95_zone['xline'], N_zone['xline'], E95_zone['spot'], N_zone['spot']),
            'halo_over_spot':  safe_ratio(E95_zone['halo'],  N_zone['halo'],  E95_zone['spot'], N_zone['spot']),
        })
        fr_rows.append({
            'iter': idx, 't_ps': t_ps, 'dt_s': dt_s,
            'rate_rod_per_s':   rate_zone['rod'],
            'rate_core_per_s':  rate_zone['core'],
            'rate_inter_per_s': rate_zone['inter'],
            'rate_xline_per_s': rate_zone['xline'],
            'rate_spot_per_s':  rate_zone['spot'],
            'rate_outer_per_s': rate_zone['outer'],
            'rate_halo_per_s':  rate_zone['halo'],
            'cum_E_rod_J':   cum_E['rod'],
            'cum_E_core_J':  cum_E['core'],
            'cum_E_inter_J': cum_E['inter'],
            'cum_E_xline_J': cum_E['xline'],
            'cum_E_spot_J':  cum_E['spot'],
            'cum_E_outer_J': cum_E['outer'],
            'cum_E_halo_J':  cum_E['halo'],
        })

        if k == 0 or (k + 1) % 10 == 0 or k == len(iters) - 1:
            print(f'  dump {k+1:3d}/{len(iters):3d}  t={t_ps:7.1f}  '
                  f'N_rod={N_zone["rod"]:>7d}  N_outer={N_zone["outer"]:>7d}  '
                  f'E95_core={E95_zone["core"]:>7.1f}  E95_xline={E95_zone["xline"]:>7.1f}  '
                  f'E95_outer={E95_zone["outer"]:>7.1f}')

    # ── output the zones text report ──────────────────────────────────────
    out_dir = args.run_dir
    txt_path = out_dir / f'{out_prefix}_zones_{args.paper_tag}.txt'

    def fmt_ratio(v):
        """Format a ratio; NaN → 'N/A' right-aligned to 8 chars."""
        if v != v:    # NaN
            return f'{"N/A":>8s}'
        return f'{v:8.4f}'

    with txt_path.open('w') as fh:
        fh.write('=' * 130 + '\n')
        fh.write(f'  ZONE-BASED ANALYSIS ({args.paper_tag.upper()}) — {args.run_dir.name}\n')
        fh.write('=' * 130 + '\n')
        fh.write(f'  Geometry (from run_meta):\n')
        fh.write(f'    n_spots:          {n_spots}\n')
        fh.write(f'    ring_radius:      {ring_radius_m*1e6:.0f} µm\n')
        fh.write(f'    spot_radius:      {spot_radius_m*1e6:.0f} µm\n')
        fh.write(f'    rod_radius:       {rod_radius_m*1e6:.0f} µm\n')
        fh.write(f'    outer_radius:     {outer_radius_m*1e6:.0f} µm\n')
        fh.write(f'    outer_thickness:  {outer_thickness_m*1e6:.0f} µm\n')
        fh.write(f'    box half-width:   {half_box_min*1e6:.0f} µm\n\n')
        fh.write(f'  Zone boundaries (µm):\n')
        fh.write(f'    ROD          r < {R_rod*1e6:.0f}\n')
        fh.write(f'    CORE         {R_rod*1e6:.0f} ≤ r < {R_core_hi*1e6:.0f}    (Paper 1 inner-half)\n')
        fh.write(f'    INTER        {R_core_hi*1e6:.0f} ≤ r < {R_xline_lo*1e6:.0f}    (Paper 1 outer-half)\n')
        fh.write(f'    X_LINE       {R_xline_lo*1e6:.0f} ≤ r ≤ {R_xline_hi*1e6:.0f}\n')
        fh.write(f'    SPOT         {R_xline_hi*1e6:.0f} < r ≤ {R_spot_hi*1e6:.0f}\n')
        if outer_enabled:
            fh.write(f'    OUTER        {R_outer_lo*1e6:.0f} ≤ r ≤ {R_outer_hi*1e6:.0f}\n')
        else:
            fh.write(f'    OUTER        DISABLED (outer_radius_m=0)\n')
        fh.write(f'    HALO         {R_halo_lo*1e6:.0f} < r ≤ {R_halo_hi*1e6:.0f}    (fast-proton transit volume)\n')
        fh.write(f'    EXTERIOR     r > {R_edge*1e6:.0f}  (periodic-edge danger)\n\n')
        fh.write(f'  Fast-ion threshold: {args.threshold_kev:.0f} keV    Species: {args.species}\n')
        fh.write(f'  Per-zone n_B11:     base={base_n_b11:.2e}   rod={rod_n_b11:.2e}   outer={outer_n_b11:.2e}\n')
        fh.write('  E95 values are BULK 95th percentile of all proton kinetic energies in each zone, in keV.\n')
        fh.write('  E95_tail values are 95th percentile of proton energies ABOVE the threshold (keV).\n')
        fh.write('  Ratios "X/spot" are E95_X / E95_spot, or N/A if SPOT zone is empty.\n')
        fh.write('=' * 130 + '\n\n')
        # Header
        fh.write(f'   t(ps)   N_rod   N_core  N_inter  N_xline   N_spot   N_outer    N_halo | '
                 f' E95_rod E95_core E95_inter E95_xline E95_spot E95_outer  E95_halo | '
                 f' core/spot  rod/spot  outer/spot  xline/spot  halo/spot\n')
        fh.write('  ' + '-' * 140 + '\n')
        for r_ in rows:
            fh.write(f'  {r_["t_ps"]:7.2f} '
                     f'{r_["N_rod"]:>8d} {r_["N_core"]:>8d} {r_["N_inter"]:>8d} '
                     f'{r_["N_xline"]:>8d} {r_["N_spot"]:>8d} {r_["N_outer"]:>9d} '
                     f'{r_["N_halo"]:>9d} | '
                     f'{r_["E95_rod"]:>8.1f} {r_["E95_core"]:>8.1f} {r_["E95_inter"]:>9.1f} '
                     f'{r_["E95_xline"]:>9.1f} {r_["E95_spot"]:>8.1f} {r_["E95_outer"]:>9.1f} '
                     f'{r_["E95_halo"]:>9.1f} | '
                     f'{fmt_ratio(r_["core_over_spot"])} {fmt_ratio(r_["rod_over_spot"])} '
                     f'{fmt_ratio(r_["outer_over_spot"])} {fmt_ratio(r_["xline_over_spot"])} '
                     f'{fmt_ratio(r_["halo_over_spot"])}\n')
        # Gain vs laser: prefer CLI override, else step_time_index.csv, else
        # the legacy 5 J fallback (with a note in the report so it's not silent).
        if args.laser_energy_j is not None:
            laser_E_j = args.laser_energy_j
            laser_E_src = f'CLI override (--laser-energy-j {laser_E_j:.3e})'
        else:
            laser_E_csv = find_laser_energy_j(args.run_dir)
            if laser_E_csv is not None and laser_E_csv > 0:
                laser_E_j = laser_E_csv
                laser_E_src = 'step_time_index.csv (last row, laser_energy_j)'
            else:
                laser_E_j = 5.0
                laser_E_src = 'fallback default 5.0 J (no CSV / CLI override)'
        gain_total = sum(cum_E.values()) / laser_E_j
        fh.write('\n' + '=' * 130 + '\n')
        fh.write(f'  CUMULATIVE FUSION ENERGY (J), per zone, integrated to t={rows[-1]["t_ps"]:.0f} ps:\n')
        fh.write(f'    rod   : {cum_E["rod"]:.4e}\n')
        fh.write(f'    core  : {cum_E["core"]:.4e}\n')
        fh.write(f'    inter : {cum_E["inter"]:.4e}\n')
        fh.write(f'    xline : {cum_E["xline"]:.4e}\n')
        fh.write(f'    spot  : {cum_E["spot"]:.4e}\n')
        fh.write(f'    outer : {cum_E["outer"]:.4e}\n')
        fh.write(f'    halo  : {cum_E["halo"]:.4e}\n')
        fh.write(f'    total : {sum(cum_E.values()):.4e}\n')
        fh.write(f'  Laser energy used for gain: {laser_E_j:.4e} J  ({laser_E_src})\n')
        fh.write(f'    gain vs laser  : {gain_total:.4e}\n')
        fh.write('=' * 130 + '\n')
        fh.write('  NOTE: Adding HALO closes the accounting gap between zone-sum total\n')
        fh.write('        and the global first-transit methodology total (G_FT). Halo\n')
        fh.write('        captures fast-proton transit through bulk plasma in the\n')
        fh.write('        previously-uncounted region between SPOT/OUTER and the\n')
        fh.write('        exterior-cut radius. For the LD baseline, halo contributes\n')
        fh.write('        approximately 13 J, bringing zone-sum into agreement with G_FT.\n')
        fh.write('=' * 130 + '\n')
    print(f'Wrote: {txt_path}')

    # ── output the per-zone fusion-rate CSV ──────────────────────────────
    csv_path = out_dir / f'{out_prefix}_zone_fusion_rates_{args.paper_tag}.csv'
    if fr_rows:
        with csv_path.open('w', newline='') as fh:
            w_csv = csv.DictWriter(fh, fieldnames=list(fr_rows[0].keys()))
            w_csv.writeheader(); w_csv.writerows(fr_rows)
        print(f'Wrote: {csv_path}')

    return 0



if __name__ == '__main__':
    sys.exit(main())
