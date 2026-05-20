#!/usr/bin/env python3
"""
pb11_first_transit_fusion.py — post-analysis: per-particle first-transit
fusion-rate accounting with periodic-boundary wrap detection.

Supports TWO modes via --mode:

  --mode zone   (default; Paper 3)
      Counts fusion contribution in THREE additive regions:
        (1) ROD   : while inside the placed rod, during first inward
                    crossing only, against rod_n = base_n + rod_n_addl
        (2) OUTER : while inside the placed outer band, during first
                    outward crossing only, against outer_n = base_n
                    + outer_n_addl
        (3) BASE  : everywhere outside both placed zones, while above
                    threshold AND not wrapped AND inside the exterior
                    cut, against base_n (the bulk plasma density).
      The BASE tally was added in v0.7 (SR patch) and is the headline
      correction: previously, particles in the base region — including
      the reconnection-accelerated population at the X-line annulus
      (r ≈ ring_radius - spot_radius, well outside any placed rod) —
      contributed nothing to zone-mode output even though they sit in
      the base ch_bn at base_n and are fusing against it. Total SR
      cumulative energy = rod + outer + base and is the value to use
      as a Paper 3 gain numerator.
      Zones automatically disable themselves when run_meta.txt indicates
      the corresponding fuel region was not placed; the base tally then
      expands to fill the disabled region. For ring-only runs (both
      zones disabled), the base tally captures the whole simulation
      volume and is equivalent to global mode without t_cut.

  --mode global (Paper 1: ring-only geometry, no inserts)
      Counts fusion contribution PER PARTICLE across the whole simulation
      volume, but only during the reconnection-event window:
        from each particle's super-threshold acceleration
        UNTIL min(periodic-wrap event, global B-collapse > 50%, end-of-run)
      Termination conditions:
        (a) wrap event       — particle crosses a periodic boundary
        (b) reconnection end — t_ps exceeds the first time the global
                                B-collapse (read from
                                reconnection_rate_offline.csv) crosses
                                50%, which marks the end of the lobe-
                                annihilation drive
      This addresses the wrap-artifact concern that affects the legacy
      pipeline's `fusion_rate_power_by_iter.csv` for ring-only geometries
      where no rod/outer insert is present and zone mode therefore
      produces zero output. Paper 1 v0.7 headline metric.

WHY FIRST-TRANSIT ACCOUNTING
----------------------------
The legacy pipeline (`fusion_rate_power_by_iter.csv`) computes fusion rate
per timestep as N_fast(t) × n_target × σ × v, where N_fast counts every
particle currently above the fusion threshold inside a region. This is the
right formula for an instantaneous rate, but with periodic boundaries a
single energetic proton can wrap around the box and re-cross a catcher
annulus (zone mode) or contribute to the bulk fusion rate (global mode)
multiple times during a long run. Each crossing gets counted again,
inflating the apparent fusion rate.

For PAPER 3 (rod + catcher inserts):
  * For the rod zone (r < R_rod), count fusion contribution only during
    the proton's *first inward* transit through the rod.
  * For the outer zone (R_outer ± dR), count contribution only during the
    proton's *first outward* crossing through the catcher annulus.

For PAPER 1 (ring-only, no inserts):
  * The reconnection event itself is the natural window. We count each
    proton's σ·v·n_B11 contribution only while:
        (i) it is currently above threshold
        (ii) it has not yet wrapped a periodic boundary
        (iii) t_ps ≤ t_cut (B-collapse-50% time)
    This is the correct "first transit" interpretation for a ring-only
    geometry where there is no zone boundary to cross.

The script reads particle dumps that already include 'id' as an openPMD
record (verified present in v15-produced dumps via pb11_particle_trajectories.py).

REACTIONS
---------
This script computes a single reaction at a time, selected by --reaction:

  --reaction p11b   p + 11B -> 3 alpha   (8.68 MeV, threshold 500 keV)
                    target nucleus = B-11 in each zone
  --reaction p7li   p + 7Li -> 2 alpha   (17.35 MeV, threshold 300 keV)
                    target nucleus = Li-7 in each zone

The Paper 3 pipeline runs this script TWICE per run (once for p11b, once
for p7li), and the local aggregator (paper3_summarize.py) zeroes out
zone contributions where the relevant target nucleus is absent. This
lets a single first-transit script generalize to any future fuel mix
without modification.

OUTPUTS (per --reaction)
------------------------
  Zone mode (default):
    <prefix>_fusion_rate_first_transit_<rxn>.csv         per-dump fusion rate
    <prefix>_wrap_artifact_metrics_<rxn>.csv             diagnostic: wrap counts
    <prefix>_first_transit_summary_<rxn>.txt             comparison vs legacy

  Global mode:
    <prefix>_fusion_rate_first_transit_global_<rxn>.csv  per-dump fusion rate
    <prefix>_wrap_artifact_metrics_global_<rxn>.csv      diagnostic: wrap counts
    <prefix>_first_transit_summary_global_<rxn>.txt      G_FT, G_legacy, t_cut

USAGE
-----
  # Paper 3 (zone mode, default)
  python pb11_first_transit_fusion.py --run-dir runs/paper03/p3_rod_plus_outer

  # Paper 1 (global mode, ring-only geometry)
  python pb11_first_transit_fusion.py --run-dir runs/paper01/p1_ld_uuf \
      --mode global

  # Override the reconnection-end cut time directly
  python pb11_first_transit_fusion.py --run-dir <dir> --mode global \
      --t-cut-ps 280

  # p-7Li accounting (run after p11b for hybrid runs)
  python pb11_first_transit_fusion.py --run-dir runs/paper03/p3_rod_plus_outer \
      --reaction p7li

  # Override zone radii (defaults pulled from run_meta.txt)
  python pb11_first_transit_fusion.py --run-dir <dir> \
      --r-rod-um 600 --r-outer-um 3500 --outer-thickness-um 600

The script defaults zone boundaries to:
  ROD     : r < 2.4 × rod_radius_m         (captures ~94% of rod Gaussian mass)
  OUTER   : outer_radius_m ± 2 × outer_thickness_m   (≈95% of catcher mass)
  EXTERIOR: r > min(LX_M/2, LZ_M/2) - 100µm buffer   (periodic-edge danger zone)

These can be overridden with --r-rod-um, --r-outer-um, --outer-thickness-um.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# ============================================================================
# Physical constants
# ============================================================================
PROTON_MASS_KG = 1.67262192e-27
Q_E             = 1.602176634e-19
C_LIGHT         = 2.99792458e8
KEV             = Q_E * 1e3
MEV             = Q_E * 1e6


# ============================================================================
# Reaction registry
# ============================================================================
# Each reaction provides:
#   sigma_m2:       effective cross-section above threshold (m²)
#   threshold_kev:  proton kinetic-energy threshold for fusion (keV)
#   n_alphas:       alphas produced per fusion event
#   energy_j:       fusion energy released per event (J)
#   target_isotope: 'b11' or 'li7' — selects which run_meta density key
#                   determines local target nucleus density per zone
#   meta_keys:      tuple of (base_key, rod_key, outer_key) names used to
#                   read per-zone target nucleus density from run_meta.txt
#
# Cross-section values match the existing pipeline's effective constants
# (σ × v at threshold, validated against fusion_rate_power_by_iter.csv).
# This keeps the first-transit / legacy comparison apples-to-apples and
# isolates the wrap-artifact effect rather than introducing new physics.
REACTIONS = {
    'p11b': {
        'label':           'p + 11B -> 3 alpha',
        'sigma_m2':        1.2e-31,
        'threshold_kev':   500.0,
        'n_alphas':        3,
        'energy_j':        8.68 * MEV,
        'target_isotope':  'b11',
        'meta_keys':       ('base_n_b11', 'rod_n_b11', 'ring_n_b11'),
    },
    'p7li': {
        'label':           'p + 7Li -> 2 alpha',
        'sigma_m2':        5.0e-31,            # 0.5 barn at 441 keV peak
        'threshold_kev':   300.0,
        'n_alphas':        2,
        'energy_j':        17.35 * MEV,
        'target_isotope':  'li7',
        'meta_keys':       ('base_n_li7', 'rod_n_li7', 'ring_n_li7'),
    },
}


# ============================================================================
# Helpers
# ============================================================================
def kev_from_momentum(ux, uy, uz, mass_kg):
    """Relativistic KE in keV from picmi-convention momentum (γβc per species)."""
    # WarpX/PICMI momentum convention: u = γv (relativistic velocity), so
    # u_total² has units of m²/s² and γ = sqrt(1 + (u/c)²)
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT))
    return (gamma - 1.0) * mass_kg * C_LIGHT * C_LIGHT / KEV


def speed_from_momentum(ux, uy, uz):
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 / (C_LIGHT * C_LIGHT))
    return np.sqrt(u2) / gamma


def parse_run_meta(meta_path):
    """Parse run_meta.txt and return a dict of geometry + density values."""
    meta = {}
    if not meta_path.is_file():
        raise FileNotFoundError(f'run_meta.txt not found: {meta_path}')
    section = None
    for line in meta_path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('=') or s.startswith('v'):
            continue
        if s.startswith('[') and s.endswith(']'):
            section = s[1:-1]
            continue
        if '=' not in s:
            continue
        key, val = s.split('=', 1)
        key = key.strip()
        val = val.split(' ', 1)[0].strip()  # strip trailing comments
        # try numeric, fall back to string
        try:
            if '.' in val or 'e' in val or 'E' in val:
                meta[key] = float(val)
            else:
                meta[key] = int(val)
        except ValueError:
            meta[key] = val
    return meta


def find_particle_series(run_dir: Path):
    """Locate the openPMD particle series and return (series, sorted_iters)."""
    try:
        import openpmd_api as io
    except ImportError:
        raise SystemExit('openpmd_api not installed.  pip install openpmd-api')

    pdir = run_dir / 'particles'
    if not pdir.is_dir():
        # Fall back to scanning subdirectories
        candidates = [d for d in run_dir.iterdir()
                      if d.is_dir() and (d / 'openpmd_000000.h5').exists()]
        if not candidates:
            raise SystemExit(f'No particle dumps found under {run_dir}')
        pdir = candidates[0]

    # Detect backend
    if list(pdir.glob('openpmd_*.h5')):
        pattern = str(pdir / 'openpmd_%T.h5')
    elif list(pdir.glob('openpmd_*.bp')):
        pattern = str(pdir / 'openpmd_%T.bp')
    else:
        raise SystemExit(f'No openpmd files found in {pdir}')

    series = io.Series(pattern, io.Access.read_only)
    iters = sorted(series.iterations)
    return series, iters, pdir


def load_dump(series, iter_idx, species_name):
    """Load (id, x, z, ux, uy, uz, w) arrays from a single openPMD iteration."""
    it = series.iterations[iter_idx]
    if species_name not in list(it.particles):
        return None
    sp = it.particles[species_name]

    def load(record, component=None):
        rec = sp[record]
        comp = rec[component] if component is not None else rec[next(iter(rec))]
        chunk = comp.load_chunk()
        series.flush()
        # Apply any unit_SI conversion stored on the component
        try:
            unit = comp.unit_SI
            if unit and unit != 1.0:
                chunk = chunk * unit
        except Exception:
            pass
        return chunk

    try:
        ids = load('id')
        x   = load('position', 'x')
        z   = load('position', 'z')
        ux  = load('momentum', 'x')
        uy  = load('momentum', 'y')
        uz  = load('momentum', 'z')
        w   = load('weighting')
    except Exception as e:
        print(f'  WARN: iter {iter_idx} load failed: {e}', file=sys.stderr)
        return None

    # WarpX stores momentum as γβmc (i.e. u_phys = u_record / mass).  We
    # divide by proton mass to get u in m/s × γ for use in kev_from_momentum.
    ux = ux / PROTON_MASS_KG
    uy = uy / PROTON_MASS_KG
    uz = uz / PROTON_MASS_KG
    return ids.astype(np.int64), x, z, ux, uy, uz, w


def find_b_minimum_t_cut(run_dir: Path):
    """Read reconnection_rate_offline.csv and return the time (in ps) at which
    the global lobe-field magnitude reaches its minimum.

    Returns:
        (t_cut_ps, b_min_T, b_initial_T, n_rows_read)
        Any element may be None if the CSV is missing or malformed.

    METHODOLOGY (Paper 1 v0.7, replaces threshold-based collapse cut):
    ----------------------------------------------------------------
    The earlier implementation used B_collapse_pct >= 50% (seed-B-relative)
    to mark "end of reconnection event". This was wrong because the Biermann
    ramping itself drops the field from seed (e.g. 85 T) to a post-ramp
    initial value (e.g. 51.6 T) over the first ~100 timesteps — already a
    39% "collapse" before any reconnection physics has begun. The 50%
    threshold then trips during the very first dump after t=0 (~15 ps),
    long before the reconnection event peaks at ~60 ps.

    The new definition is operationally unambiguous and threshold-free:
    "Reconnection has fully consumed the lobe field when the field magnitude
    reaches its minimum and begins to rebound." Specifically:

        t_cut = argmin_t(B_max_T(t))

    For a typical Paper 1 LD baseline (B_seed=85, post-ramp B=51.6),
    the minimum sits around 22-23 T at ~230 ps — right at the end of the
    phase_analysis "Sustained Reconnection" window (Phase 2 ends ~293 ps),
    which is the physical end of the reconnection event. After this point,
    the field oscillates in a flat well around 22-26 T while bulk thermal
    equilibration drives further fusion that is no longer reconnection-
    driven and therefore should be excluded from the first-transit headline.

    Reviewer note: this definition has no tunable parameter, sidestepping
    the "why 50% not 60%" critique entirely. The trade-off is that it
    requires a run long enough for B to actually reach a minimum and start
    rebounding — for very short runs that terminate inside the monotonic
    decay phase, the returned t_cut will be the final dump time (effectively
    end-of-run = no termination).
    """
    csv_path = run_dir / 'reconnection_rate_offline.csv'
    if not csv_path.is_file():
        return None, None, None, 0
    ts = []
    Bs = []
    n_rows = 0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if 't_ps' not in reader.fieldnames or 'B_max_T' not in reader.fieldnames:
            return None, None, None, 0
        for row in reader:
            n_rows += 1
            try:
                t_ps = float(row['t_ps'])
                B_T = float(row['B_max_T'])
            except (KeyError, ValueError):
                continue
            # float('NaN') and float('inf') don't raise — guard explicitly so
            # they don't corrupt argmin on the resulting array.
            if not (math.isfinite(t_ps) and math.isfinite(B_T)):
                continue
            # Skip pre-Biermann-seeding dumps where the lobe field has not
            # yet been initialized. This is a known WarpX intermittent: dump 0
            # at step 0 sometimes fires BEFORE the Biermann seeding hook runs,
            # leaving all fields at zero. The lobe field is physically strictly
            # positive once seeding completes, so B_T <= 0 is an upstream
            # artifact that must be filtered. Without this, argmin trivially
            # picks the pre-seeding dump and t_cut collapses to t=0.
            if B_T <= 0.0:
                continue
            ts.append(t_ps)
            Bs.append(B_T)
    if not ts:
        return None, None, None, n_rows
    ts_arr = np.asarray(ts, dtype=np.float64)
    Bs_arr = np.asarray(Bs, dtype=np.float64)
    idx_min = int(np.argmin(Bs_arr))
    t_cut_ps = float(ts_arr[idx_min])
    b_min_T = float(Bs_arr[idx_min])
    b_initial_T = float(Bs_arr[0])  # post-Biermann-ramp reference (NOT seed)
    return t_cut_ps, b_min_T, b_initial_T, n_rows


def find_laser_energy_j(run_dir: Path):
    """Read step_time_index.csv last row's laser_energy_j for gain reporting.

    Returns float or None if the column is missing / file unreadable.
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
    ap.add_argument('--run-dir', required=True, type=Path,
                    help='Run directory (must contain run_meta.txt + particles/)')
    ap.add_argument('--mode', default='zone', choices=['zone', 'global'],
                    help='Accounting mode (default: zone). '
                         'zone   = Paper 3 (rod / outer-catcher first-crossing). '
                         'global = Paper 1 v0.7 (whole-volume first-transit '
                         'until wrap, exterior, or B-field minimum).')
    ap.add_argument('--t-cut-ps', type=float, default=None,
                    help='Global-mode reconnection-end cut time in ps. If not '
                         'set, derived as the time of minimum lobe-field '
                         'magnitude (argmin of B_max_T in '
                         'reconnection_rate_offline.csv). Ignored in zone mode.')
    ap.add_argument('--b-collapse-threshold-pct', type=float, default=None,
                    help='DEPRECATED (Paper 1 v0.7). Previous threshold-based '
                         't_cut definition; ignored. v0.7 methodology uses '
                         'B-field argmin instead (threshold-free). Flag is '
                         'kept for backward compatibility with existing '
                         'analysis-chain shell scripts.')
    ap.add_argument('--exterior-margin-um', type=float, default=100.0,
                    help='Global-mode: exclude particles with r > (box_half_'
                         'width - margin_um) from all three global tallies '
                         '(first-transit, pre-t_cut, legacy). Particles in '
                         'this annulus are too close to the periodic boundary '
                         'for the bulk-density assumption to be reliable. '
                         'Matches the EXTERIOR zone convention used by '
                         'pb11_zone_analysis_paper3.py. Default 100 µm. '
                         'Set to 0 to disable (legacy v0.6 behaviour, includes '
                         'corner-region particles in the gain integral).')
    ap.add_argument('--reaction', default='p11b', choices=list(REACTIONS.keys()),
                    help='Fusion reaction to compute (default: p11b).  Run twice '
                         'with --reaction p11b and --reaction p7li for hybrid runs; '
                         'paper3_summarize.py applies per-zone fuel masking from '
                         'run_meta.txt to combine the outputs correctly.')
    ap.add_argument('--species', default='proton',
                    help='Particle species to analyze (default: proton)')
    ap.add_argument('--r-rod-um', type=float, default=None,
                    help='Override rod-zone outer radius in µm (default: 2.4 × rod_radius_m)')
    ap.add_argument('--r-outer-um', type=float, default=None,
                    help='Override outer-zone center radius in µm (default: outer_radius_m)')
    ap.add_argument('--outer-thickness-um', type=float, default=None,
                    help='Override outer-zone half-thickness in µm '
                         '(default: 2 × outer_thickness_m)')
    ap.add_argument('--threshold-kev', type=float, default=None,
                    help='Fusion threshold override (default: per-reaction value)')
    ap.add_argument('--max-dumps', type=int, default=0,
                    help='Limit to first N dumps for testing (0 = all)')
    ap.add_argument('--out-prefix', type=str, default=None,
                    help='Prefix for output CSVs (default: run_dir name)')
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f'ERROR: run-dir not found: {args.run_dir}', file=sys.stderr)
        return 2

    # ── Reaction parameters ───────────────────────────────────────────────
    rxn = REACTIONS[args.reaction]
    SIGMA_M2     = rxn['sigma_m2']
    THRESHOLD    = (args.threshold_kev if args.threshold_kev is not None
                    else rxn['threshold_kev'])
    E_FUSION_J   = rxn['energy_j']
    rxn_label    = rxn['label']
    rxn_tag      = args.reaction               # 'p11b' or 'p7li'
    target_iso   = rxn['target_isotope']       # 'b11' or 'li7'
    base_key, rod_key, outer_key = rxn['meta_keys']

    # ── geometry from run_meta ────────────────────────────────────────────
    # Defaults reflect v15 ring geometry; if rod/outer aren't in run_meta
    # (e.g. ring-only runs), the corresponding zone will be marked DISABLED
    # and the script will skip its accounting entirely (see "rod_enabled" /
    # "outer_enabled" below).
    meta = parse_run_meta(args.run_dir / 'run_meta.txt')
    rod_radius_m       = meta.get('rod_radius_m', None)        # None = disabled
    outer_radius_m     = meta.get('outer_radius_m', None)      # None = disabled
    outer_thickness_m  = meta.get('outer_thickness_m', 200e-6) # default v15 thickness
    lx_m               = meta.get('lx_m', 12220e-6)            # v15 default box (12.22 mm)
    lz_m               = meta.get('lz_m', 12220e-6)
    time_step_s        = meta.get('time_step_s', None)

    # Per-region target densities for THIS reaction's target nucleus.
    #
    # NOTE on additive model: v15's fuel placement is ADDITIVE — the base
    # plasma fills the entire box, and rod/outer fuel adds on top within
    # their respective regions. So `rod_n_b11` in run_meta is the rod's
    # ADDITIONAL contribution (not the total density in the rod region).
    # The actual total density a proton sees in the rod region is
    # base_n_b11 + rod_n_b11, and this total is what fusion rates must
    # be computed against.
    base_n        = meta.get(base_key,  0.0)
    rod_n_addl    = meta.get(rod_key,   0.0)   # additive contribution; 0 if rod disabled
    outer_n_addl  = meta.get(outer_key, 0.0)   # additive contribution; 0 if outer disabled
    # Total density in each region — what protons actually see
    rod_n   = base_n + rod_n_addl
    outer_n = base_n + outer_n_addl
    # "Placed" means this region has additive fuel on top of base
    rod_placed   = rod_n_addl   > 0.0
    outer_placed = outer_n_addl > 0.0

    # Zone enable flags: a zone is active only if BOTH the geometry is
    # defined in run_meta (so we know its boundary) AND the additive fuel
    # contribution is non-zero (otherwise there's nothing to track that the
    # base-region accounting wouldn't already cover).  A ring_only job has
    # no rod/outer fuel and no rod/outer geometry — in that case both zones
    # are disabled and the script reports baseline-only.
    rod_enabled   = rod_placed   and (rod_radius_m   is not None)
    outer_enabled = outer_placed and (outer_radius_m is not None)

    # Zone boundaries (only meaningful if the corresponding zone is enabled)
    if rod_enabled:
        R_rod = (args.r_rod_um * 1e-6) if args.r_rod_um is not None \
                else 2.4 * rod_radius_m
    else:
        R_rod = 0.0  # placeholder; not used when rod_enabled=False

    if outer_enabled:
        R_out = (args.r_outer_um * 1e-6) if args.r_outer_um is not None \
                else outer_radius_m
        dR_out = (args.outer_thickness_um * 1e-6) if args.outer_thickness_um is not None \
                 else 2.0 * outer_thickness_m
        R_out_lo = R_out - dR_out
        R_out_hi = R_out + dR_out
    else:
        R_out = R_out_lo = R_out_hi = 0.0  # placeholders; not used

    # Periodic-edge danger radius (smaller half-box dimension - 100µm buffer)
    half_box_min = min(lx_m, lz_m) / 2.0
    R_edge = half_box_min - 100e-6

    # Wrap detection threshold: a position jump > 0.4 × box_diagonal is a wrap.
    # 0.4 leaves margin for fast unwrapped travel between dumps.
    box_diag = math.sqrt(lx_m * lx_m + lz_m * lz_m)
    wrap_dist_threshold = 0.4 * box_diag

    # ── Global-mode setup (Paper 1 ring-only) ─────────────────────────────
    # In global mode we don't track rod/outer zones; instead, we count each
    # particle's fusion contribution against the bulk (base-region) target-
    # nucleus density while the particle is above threshold, unwrapped,
    # not in the periodic-boundary danger zone (r > R_edge), AND the
    # reconnection event is still ongoing (t ≤ t_cut).
    #
    # t_cut derivation precedence (v0.7):
    #   1. --t-cut-ps CLI override (explicit)
    #   2. argmin of B_max_T in reconnection_rate_offline.csv (B-field minimum)
    #   3. Fall back to +inf (no temporal termination; only wrap & exterior)
    #
    # The --b-collapse-threshold-pct flag is deprecated; if a value is passed,
    # we ignore it and print a one-line note so chained scripts continue to
    # work but operators see the change.
    if args.b_collapse_threshold_pct is not None:
        print(f'NOTE: --b-collapse-threshold-pct={args.b_collapse_threshold_pct} '
              f'is deprecated and ignored. Paper 1 v0.7 uses B-field argmin '
              f'for t_cut (threshold-free).')

    global_mode = (args.mode == 'global')
    if global_mode:
        if args.t_cut_ps is not None:
            t_cut_ps = float(args.t_cut_ps)
            t_cut_source = f'CLI override (--t-cut-ps {t_cut_ps:.1f})'
            b_min_T = None
            b_initial_T = None
            b_n_rows = 0
        else:
            t_cut_ps, b_min_T, b_initial_T, b_n_rows = find_b_minimum_t_cut(
                args.run_dir)
            if t_cut_ps is None:
                t_cut_source = ('NOT FOUND — reconnection_rate_offline.csv '
                                'missing or unparseable; falling back to '
                                'end-of-run (no temporal termination)')
                t_cut_ps = float('inf')
            else:
                ramp_pct = (((b_initial_T - b_min_T) / b_initial_T * 100.0)
                            if b_initial_T and b_initial_T > 0 else float('nan'))
                t_cut_source = (f'argmin(B_max_T) over {b_n_rows} dumps in '
                                f'reconnection_rate_offline.csv: '
                                f'B_min={b_min_T:.2f}T at t={t_cut_ps:.2f}ps '
                                f'(B_initial={b_initial_T:.2f}T post-Biermann-ramp, '
                                f'post-ramp collapse {ramp_pct:.1f}%)')
        laser_energy_j = find_laser_energy_j(args.run_dir)
        # Bulk target-nucleus density for global mode: base region density of
        # the reaction's target isotope (since ring-only runs have no zone
        # additives — base_n IS the density everywhere). For Paper 3 hybrid
        # global-mode use, this still represents the box-average bulk fuel.
        bulk_n = base_n
    else:
        t_cut_ps = float('inf')
        t_cut_source = 'n/a (zone mode)'
        b_min_T = None
        b_initial_T = None
        b_n_rows = 0
        # v0.7 SR patch: zone mode also reads laser_energy_j so the summary
        # can report G_SR (total_SR / laser_energy_j).
        laser_energy_j = find_laser_energy_j(args.run_dir)
        bulk_n = base_n

    # Exterior-exclusion radius (Paper 1 v0.7, extended in v0.7 SR patch to
    # zone mode for the BASE-region tally only).  Particles with r > R_exterior
    # are too close to the periodic boundary for the bulk-density assumption
    # to hold and are excluded from:
    #   - global-mode tallies (all three: FT / pre-t_cut / legacy), and
    #   - the zone-mode BASE-region tally (new in v0.7 SR patch).
    # The existing zone-mode rod / outer / legacy tallies are NOT exterior-cut.
    # Their masks (r < R_rod and R_out_lo <= r <= R_out_hi) live well inside
    # R_exterior anyway, and altering those tallies would change values that
    # pre-patch comparisons depend on.
    # The default 100 µm margin matches pb11_zone_analysis_paper3.py's
    # EXTERIOR zone. Setting --exterior-margin-um 0 disables the cut.
    if args.exterior_margin_um > 0:
        R_exterior = (min(lx_m, lz_m) / 2.0) - (args.exterior_margin_um * 1e-6)
        exterior_enabled = True
    else:
        R_exterior = float('inf')   # no particle ever exceeds → cut disabled
        exterior_enabled = False

    print(f'Run:               {args.run_dir}')
    print(f'Mode:              {args.mode.upper()}')
    print(f'Reaction:          {rxn_tag}  ({rxn_label})')
    print(f'  sigma            = {SIGMA_M2:.2e} m^2')
    print(f'  threshold        = {THRESHOLD:.0f} keV')
    print(f'  E_fusion         = {E_FUSION_J / MEV:.2f} MeV')
    print(f'  target nucleus   = {target_iso}')
    print(f'Species:           {args.species}')
    print(f'Rod placed:        {rod_placed}  '
          f'(base={base_n:.2e} + addl={rod_n_addl:.2e} = total={rod_n:.2e} m^-3)')
    print(f'Outer placed:      {outer_placed}  '
          f'(base={base_n:.2e} + addl={outer_n_addl:.2e} = total={outer_n:.2e} m^-3)')
    if rod_enabled:
        print(f'Rod zone:          r < {R_rod*1e6:.0f} um   ENABLED')
    else:
        print(f'Rod zone:          DISABLED (rod_radius_m={rod_radius_m}, rod_placed={rod_placed})')
    if outer_enabled:
        print(f'Outer zone:        {R_out_lo*1e6:.0f} <= r <= {R_out_hi*1e6:.0f} um   ENABLED')
    else:
        print(f'Outer zone:        DISABLED (outer_radius_m={outer_radius_m}, outer_placed={outer_placed})')
    print(f'Edge danger:       r > {R_edge*1e6:.0f} um  (wrap detection only)')
    print(f'Box half-width:    {half_box_min*1e6:.0f} um')
    print(f'Wrap |dr| trigger: {wrap_dist_threshold*1e6:.0f} um  (0.4 x box diag)')
    if not global_mode and exterior_enabled:
        # v0.7 SR patch: exterior cut applies to base-zone tally in zone mode.
        print(f'Exterior cut:      r > {R_exterior*1e6:.0f} um  '
              f'(applied to base-zone tally; rod/outer/legacy unaffected)')

    if global_mode:
        if t_cut_ps == float('inf'):
            t_cut_disp = 'inf (end-of-run)'
        else:
            t_cut_disp = f'{t_cut_ps:.1f} ps'
        print(f'GLOBAL MODE:       active')
        print(f'  Bulk density   = {bulk_n:.2e} m^-3  ({target_iso})')
        print(f'  t_cut          = {t_cut_disp}')
        print(f'  t_cut source   = {t_cut_source}')
        if exterior_enabled:
            print(f'  Exterior cut   = r > {R_exterior*1e6:.0f} um '
                  f'(box_half_width={min(lx_m,lz_m)/2.0*1e6:.0f}um - '
                  f'margin={args.exterior_margin_um:.0f}um)')
        else:
            print(f'  Exterior cut   = DISABLED (legacy v0.6 behaviour; '
                  f'includes corner-region particles)')
        print(f'  Laser energy   = '
              + (f'{laser_energy_j:.3e} J' if laser_energy_j else 'unknown'))

    # ── particle dumps ────────────────────────────────────────────────────
    series, iters, pdir = find_particle_series(args.run_dir)
    if args.max_dumps > 0:
        iters = iters[:args.max_dumps]
    print(f'Particle dumps:    {len(iters)} found in {pdir}')

    # ── streaming pass (VECTORIZED) ───────────────────────────────────────
    # Per-ID state is maintained as parallel NumPy arrays indexed by a
    # row number assigned the first time we see a given particle ID.
    # We use np.unique + searchsorted to vectorize the pid→row lookup
    # rather than per-particle dict access (which was the O(N) hotspot).
    #
    # State arrays grow as new IDs appear in subsequent dumps. Boolean
    # state arrays (wrapped, rod_first_in, etc.) default to False; float
    # state (prev_r) defaults to NaN.

    # Dynamic ID-to-row registry.  known_ids stays sorted for searchsorted.
    known_ids = np.empty(0, dtype=np.int64)        # sorted unique pids seen so far
    wrapped       = np.empty(0, dtype=bool)        # has this pid wrapped?
    rod_first_in  = np.empty(0, dtype=bool)        # has it had a first inward crossing?
    rod_first_out = np.empty(0, dtype=bool)        # has it had a first outward crossing?
    out_first_in  = np.empty(0, dtype=bool)
    out_first_out = np.empty(0, dtype=bool)
    prev_r_state  = np.empty(0, dtype=np.float64)  # last-known r, NaN for never-seen

    # Output rows (one per dump)
    ft_rows   = []
    wa_rows   = []

    # Cumulative integrals
    cum_E_rod_J          = 0.0
    cum_E_outer_J        = 0.0
    cum_E_legacy_rod_J   = 0.0
    cum_E_legacy_outer_J = 0.0
    # v0.7 SR patch: base-region first-transit cumulative (additive to rod
    # and outer, captures the previously-dropped X-line annulus + everything
    # outside placed zones, fusing against base_n).
    cum_E_base_J         = 0.0
    # Global-mode accumulators (only meaningful when global_mode=True)
    cum_E_global_J         = 0.0  # first-transit: above thr & not wrapped & t <= t_cut
    cum_E_global_legacy_J  = 0.0  # legacy: above thr only (no termination)
    cum_E_global_pretcut_J = 0.0  # post-wrap, pre-tcut (above thr & t <= t_cut, ignores wrap)
                                  # diagnostic isolating wrap effect vs t_cut effect
    n_above_thr_at_tcut    = 0    # captured once when t_ps first crosses t_cut

    prev_t_global = 0.0

    def _grow_state(new_count):
        """Extend all state arrays by `new_count` rows, default False/NaN."""
        nonlocal wrapped, rod_first_in, rod_first_out, out_first_in, out_first_out, prev_r_state
        if new_count <= 0:
            return
        wrapped       = np.concatenate([wrapped,       np.zeros(new_count, dtype=bool)])
        rod_first_in  = np.concatenate([rod_first_in,  np.zeros(new_count, dtype=bool)])
        rod_first_out = np.concatenate([rod_first_out, np.zeros(new_count, dtype=bool)])
        out_first_in  = np.concatenate([out_first_in,  np.zeros(new_count, dtype=bool)])
        out_first_out = np.concatenate([out_first_out, np.zeros(new_count, dtype=bool)])
        prev_r_state  = np.concatenate([prev_r_state,  np.full(new_count, np.nan)])

    for k, idx in enumerate(iters):
        loaded = load_dump(series, idx, args.species)
        if loaded is None:
            continue
        ids, x, z, ux, uy, uz, w = loaded
        ids = ids.astype(np.int64, copy=False)
        r = np.sqrt(x * x + z * z)
        ke = kev_from_momentum(ux, uy, uz, PROTON_MASS_KG)
        v  = speed_from_momentum(ux, uy, uz)
        t_s = float(idx) * time_step_s if time_step_s else 0.0
        t_ps = t_s * 1e12

        if k == 0:
            dt_s = 0.0
        else:
            dt_s = t_s - prev_t_global
        prev_t_global = t_s

        # ── Map pid → row index (vectorized).
        # Step 1: find which of this dump's IDs are new (not in known_ids)
        # Step 2: extend known_ids (and parallel state arrays) for new IDs
        # Step 3: rebuild row mapping for ALL of this dump's particles
        # Total cost: O(N log N) for searchsorted, no Python loops.
        if known_ids.size == 0:
            # First dump: all IDs are new
            sort_perm = np.argsort(ids)
            new_known_ids = ids[sort_perm]
            # Need unique in case of duplicates within one dump (shouldn't happen
            # for openPMD particle IDs, but defensive):
            unique_mask = np.concatenate([[True], np.diff(new_known_ids) != 0])
            new_known_ids = new_known_ids[unique_mask]
            n_new = new_known_ids.size
            known_ids = new_known_ids
            _grow_state(n_new)
        else:
            insert_pos = np.searchsorted(known_ids, ids)
            # Mask out positions where pid is actually present at that slot
            present = (insert_pos < known_ids.size) & \
                      (known_ids[np.minimum(insert_pos, known_ids.size - 1)] == ids)
            new_pids = np.unique(ids[~present])
            if new_pids.size > 0:
                # Insert new pids in sorted order; rebuild known_ids.
                merged = np.concatenate([known_ids, new_pids])
                merged_sort = np.argsort(merged, kind='stable')
                known_ids = merged[merged_sort]
                _grow_state(new_pids.size)
                # State arrays are appended in known_ids' new ordering. We
                # need to reorder existing entries to follow the new sorted
                # known_ids order. The trick: the new elements were appended
                # at the END of state arrays; sorting known_ids tells us how
                # to permute state arrays correspondingly.
                wrapped       = wrapped[merged_sort]
                rod_first_in  = rod_first_in[merged_sort]
                rod_first_out = rod_first_out[merged_sort]
                out_first_in  = out_first_in[merged_sort]
                out_first_out = out_first_out[merged_sort]
                prev_r_state  = prev_r_state[merged_sort]

        # Now resolve pid → row for THIS dump's particles
        rows = np.searchsorted(known_ids, ids)
        # Sanity check (defensive): every pid we just lookup should be present
        # at its searchsorted position.
        # (Skip the assert in production for speed; correctness proven by construction)

        # ── Wrap detection (vectorized) ──
        prev_r_arr = prev_r_state[rows]            # NaN for first time we see a pid
        dr = np.abs(r - prev_r_arr)
        new_wrap_mask = np.isfinite(prev_r_arr) & (dr > wrap_dist_threshold) & ~wrapped[rows]
        if new_wrap_mask.any():
            # Mark these rows as wrapped (np.unique handles duplicates within
            # a single dump, which shouldn't occur but is defensive)
            wrap_rows = np.unique(rows[new_wrap_mask])
            wrapped[wrap_rows] = True

        # ── Crossing detection (vectorized; only when zone enabled) ──
        if rod_enabled:
            inward_rod = (np.isfinite(prev_r_arr)
                          & (prev_r_arr >= R_rod) & (r < R_rod))
            outward_rod = (np.isfinite(prev_r_arr)
                           & (prev_r_arr < R_rod) & (r >= R_rod))
            # Mark first inward crossing (excluding wrapped)
            inrod_rows = rows[inward_rod & ~wrapped[rows]]
            if inrod_rows.size > 0:
                # only set those that haven't yet had a first_in
                inrod_unique = np.unique(inrod_rows)
                set_mask = ~rod_first_in[inrod_unique]
                rod_first_in[inrod_unique[set_mask]] = True
            # Mark first outward crossing AFTER first inward (excluding wrapped)
            outrod_rows = rows[outward_rod & ~wrapped[rows]]
            if outrod_rows.size > 0:
                outrod_unique = np.unique(outrod_rows)
                set_mask = rod_first_in[outrod_unique] & ~rod_first_out[outrod_unique]
                rod_first_out[outrod_unique[set_mask]] = True

        if outer_enabled:
            inward_outer = (np.isfinite(prev_r_arr)
                            & (prev_r_arr < R_out_lo) & (r >= R_out_lo)
                            & (r <= R_out_hi))
            outward_outer = (np.isfinite(prev_r_arr)
                             & ((prev_r_arr <= R_out_hi) & (r > R_out_hi)
                                | (prev_r_arr >= R_out_lo) & (r < R_out_lo)
                                  & (prev_r_arr <= R_out_hi)))
            inout_rows = rows[inward_outer & ~wrapped[rows]]
            if inout_rows.size > 0:
                inout_unique = np.unique(inout_rows)
                set_mask = ~out_first_in[inout_unique]
                out_first_in[inout_unique[set_mask]] = True
            outout_rows = rows[outward_outer & ~wrapped[rows]]
            if outout_rows.size > 0:
                outout_unique = np.unique(outout_rows)
                set_mask = out_first_in[outout_unique] & ~out_first_out[outout_unique]
                out_first_out[outout_unique[set_mask]] = True

        # ── First-transit fusion contribution this dump (vectorized) ──
        above_thr = ke >= THRESHOLD

        if rod_enabled:
            in_rod_mask = (r < R_rod)
            # rod_active = above-threshold AND in-rod AND has-first-in AND
            # NOT-yet-first-out AND NOT wrapped.  All vectorized.
            rod_active = (in_rod_mask & above_thr
                          & rod_first_in[rows] & ~rod_first_out[rows]
                          & ~wrapped[rows])
            legacy_rod_mask = in_rod_mask & above_thr
            rate_rod        = float((w[rod_active]      * v[rod_active]).sum()      * rod_n   * SIGMA_M2)
            rate_rod_legacy = float((w[legacy_rod_mask] * v[legacy_rod_mask]).sum() * rod_n   * SIGMA_M2)
            n_rod_active    = int(rod_active.sum())
            n_rod_legacy    = int(legacy_rod_mask.sum())
        else:
            rate_rod = rate_rod_legacy = 0.0
            n_rod_active = n_rod_legacy = 0

        if outer_enabled:
            in_outer_mask = (r >= R_out_lo) & (r <= R_out_hi)
            outer_active = (in_outer_mask & above_thr
                            & out_first_in[rows] & ~out_first_out[rows]
                            & ~wrapped[rows])
            legacy_outer_mask = in_outer_mask & above_thr
            rate_outer        = float((w[outer_active]      * v[outer_active]).sum()      * outer_n * SIGMA_M2)
            rate_outer_legacy = float((w[legacy_outer_mask] * v[legacy_outer_mask]).sum() * outer_n * SIGMA_M2)
            n_outer_active    = int(outer_active.sum())
            n_outer_legacy    = int(legacy_outer_mask.sum())
        else:
            in_outer_mask = np.zeros_like(r, dtype=bool)
            rate_outer = rate_outer_legacy = 0.0
            n_outer_active = n_outer_legacy = 0

        # Default in_rod_mask in case the rod block was skipped (zone
        # disabled).  Required for the base-region tally below.
        if not rod_enabled:
            in_rod_mask = np.zeros_like(r, dtype=bool)

        # Shared exterior-cut mask (used by base-zone tally below and by
        # the global-mode block further down). When R_exterior is +inf
        # (--exterior-margin-um 0) this is all-True and the cut is a no-op.
        not_in_exterior_zone = (r <= R_exterior)

        # ── BASE-REGION first-transit fusion contribution (v0.7 SR patch) ──
        # Captures particles that are NOT inside a placed zone, fusing
        # against the local base_n. This is what zone mode previously
        # dropped: reconnection-accelerated protons at the X-line annulus
        # (r ≈ ring_radius - spot_radius, well outside any placed rod) plus
        # everything in the inter-zone region, all sitting in the base
        # ch_bn at base_n and fusing against it.
        #
        # First-transit semantics for the base region: there is no zone
        # boundary to cross, so the natural extension is the same as
        # global mode's per-particle logic — count contribution while:
        #   (i)   above threshold,
        #   (ii)  not wrapped,
        #   (iii) inside the exterior cut (r <= R_exterior).
        # No t_cut termination in zone mode (would couple Paper 3 numbers
        # to a Paper 1 methodology choice).
        #
        # Edge cases:
        #   - For configurations where the reaction's target nucleus is
        #     absent from base fuel (e.g. p7li on a ch_bn base where
        #     base_n_li7=0), base_n=0 and the tally is identically zero —
        #     automatically correct, no further special-casing needed.
        #   - For ring-only runs (both zones disabled), in_base_mask is
        #     True everywhere and the tally covers the whole simulation
        #     volume. This is equivalent to global mode WITHOUT t_cut.
        in_base_mask = ~in_rod_mask & ~in_outer_mask
        base_active = (in_base_mask & above_thr
                       & not_in_exterior_zone & ~wrapped[rows])
        rate_base   = float((w[base_active] * v[base_active]).sum() * base_n * SIGMA_M2)
        n_base_active = int(base_active.sum())

        # Total SR rate: additive across all three first-transit regions.
        # This is the headline Paper 3 gain numerator (multiplied by
        # E_FUSION_J · dt_s and accumulated below).
        rate_total_SR     = rate_rod + rate_outer + rate_base
        n_total_SR_active = n_rod_active + n_outer_active + n_base_active

        # ── Global-mode accumulation (Paper 1 ring-only) ──
        # Three parallel rates per dump, all sharing the SAME exterior cut:
        #   rate_global_ft       = first-transit: above thr & not wrapped &
        #                           not in exterior & t <= t_cut
        #   rate_global_pretcut  = above thr & not in exterior & t <= t_cut
        #                           (ignores wrap; isolates wrap effect)
        #   rate_global_legacy   = above thr & not in exterior
        #                           (no temporal/wrap termination; matches
        #                           the zone-analysis-summed convention if
        #                           computed against the same bulk_n)
        #
        # The exterior cut (r > R_exterior, where R_exterior = box_half_width
        # - margin) is applied UNIFORMLY across all three to keep the
        # comparison apples-to-apples. The wrap-artifact correction
        # (FT vs pre-t_cut) then isolates ONLY the wrap effect, and the
        # t_cut-only correction (pre-t_cut vs legacy) isolates ONLY the
        # temporal effect — neither is conflated with the exterior choice.
        #
        # If --exterior-margin-um is 0, R_exterior is +inf so this mask is
        # all-True and the cut is effectively disabled (v0.6 legacy behaviour).
        #
        # IMPORTANT: past_tcut is a Python scalar bool. We must NOT do
        # `above_thr & ~past_tcut` because Python's `~` on a bool returns the
        # integer -2, and numpy then produces an int64 mask which, when used
        # as a fancy index, performs positional indexing instead of boolean
        # masking — corrupting the per-dump rate. Branch explicitly instead.
        if global_mode:
            past_tcut = bool(t_ps > t_cut_ps)
            # Reuse the shared not_in_exterior_zone mask computed above.
            leg_mask = above_thr & not_in_exterior_zone
            if past_tcut:
                ft_mask  = np.zeros_like(above_thr)
                pre_mask = np.zeros_like(above_thr)
            else:
                ft_mask  = above_thr & not_in_exterior_zone & ~wrapped[rows]
                pre_mask = above_thr & not_in_exterior_zone
            rate_global_ft      = float((w[ft_mask]  * v[ft_mask]).sum()  * bulk_n * SIGMA_M2)
            rate_global_pretcut = float((w[pre_mask] * v[pre_mask]).sum() * bulk_n * SIGMA_M2)
            rate_global_legacy  = float((w[leg_mask] * v[leg_mask]).sum() * bulk_n * SIGMA_M2)
            n_global_ft         = int(ft_mask.sum())
            n_global_pretcut    = int(pre_mask.sum())
            n_global_legacy     = int(leg_mask.sum())
            # Capture above-thr count at the dump where t_cut is first crossed
            if past_tcut and n_above_thr_at_tcut == 0:
                n_above_thr_at_tcut = int((above_thr & not_in_exterior_zone).sum())
        else:
            past_tcut = False  # so the CSV column reads 0 consistently in zone mode
            rate_global_ft = rate_global_pretcut = rate_global_legacy = 0.0
            n_global_ft = n_global_pretcut = n_global_legacy = 0

        # Accumulate cumulative fusion energy
        if dt_s > 0:
            cum_E_rod_J          += rate_rod          * E_FUSION_J * dt_s
            cum_E_outer_J        += rate_outer        * E_FUSION_J * dt_s
            cum_E_legacy_rod_J   += rate_rod_legacy   * E_FUSION_J * dt_s
            cum_E_legacy_outer_J += rate_outer_legacy * E_FUSION_J * dt_s
            cum_E_global_J         += rate_global_ft      * E_FUSION_J * dt_s
            cum_E_global_legacy_J  += rate_global_legacy  * E_FUSION_J * dt_s
            cum_E_global_pretcut_J += rate_global_pretcut * E_FUSION_J * dt_s
            # v0.7 SR patch: base-region cumulative
            cum_E_base_J         += rate_base         * E_FUSION_J * dt_s

        # SR total cumulative (rod + outer + base; headline Paper 3 number)
        cum_E_total_SR_J = cum_E_rod_J + cum_E_outer_J + cum_E_base_J

        ft_rows.append({
            'iter':                idx,
            't_ps':                t_ps,
            'dt_s':                dt_s,
            'reaction':            rxn_tag,
            'rate_rod_first_transit_per_s':   rate_rod,
            'rate_outer_first_transit_per_s': rate_outer,
            'rate_rod_legacy_per_s':          rate_rod_legacy,
            'rate_outer_legacy_per_s':        rate_outer_legacy,
            'cum_E_rod_first_transit_J':      cum_E_rod_J,
            'cum_E_outer_first_transit_J':    cum_E_outer_J,
            'cum_E_rod_legacy_J':             cum_E_legacy_rod_J,
            'cum_E_outer_legacy_J':           cum_E_legacy_outer_J,
            'N_active_rod':                   n_rod_active,
            'N_active_outer':                 n_outer_active,
            'N_legacy_rod':                   n_rod_legacy,
            'N_legacy_outer':                 n_outer_legacy,
            # Global-mode columns (zero in zone mode for column-stable output)
            'rate_global_first_transit_per_s': rate_global_ft,
            'rate_global_pretcut_per_s':       rate_global_pretcut,
            'rate_global_legacy_per_s':        rate_global_legacy,
            'cum_E_global_first_transit_J':    cum_E_global_J,
            'cum_E_global_pretcut_J':          cum_E_global_pretcut_J,
            'cum_E_global_legacy_J':           cum_E_global_legacy_J,
            'N_active_global_first_transit':   n_global_ft,
            'N_active_global_pretcut':         n_global_pretcut,
            'N_active_global_legacy':          n_global_legacy,
            'past_t_cut':                      int(past_tcut) if global_mode else 0,
            # v0.7 SR-patch columns (zone-mode base tally + SR total).
            # In global mode these are still populated for diagnostic
            # cross-reference (rate_base vs rate_global_ft, etc.) — base
            # in zone mode and global in global mode answer overlapping
            # but not identical questions, and having both side-by-side
            # is useful when validating the patch.
            'rate_base_first_transit_per_s':   rate_base,
            'cum_E_base_first_transit_J':      cum_E_base_J,
            'N_active_base':                   n_base_active,
            'rate_total_SR_first_transit_per_s': rate_total_SR,
            'cum_E_total_SR_first_transit_J':    cum_E_total_SR_J,
            'N_active_total_SR':                 n_total_SR_active,
        })

        n_wrapped_total = int(wrapped.sum())
        wa_rows.append({
            'iter':              idx,
            't_ps':              t_ps,
            'reaction':          rxn_tag,
            'N_total':           int(len(ids)),
            'N_wrapped_total':   n_wrapped_total,
            'frac_wrapped':      n_wrapped_total / max(1, len(ids)),
            'N_outer_legacy':    n_outer_legacy,
            'N_outer_first_transit':  n_outer_active,
            'outer_legacy_inflation': (n_outer_legacy / n_outer_active
                                       if n_outer_active > 0 else 0.0),
            'N_global_legacy':         n_global_legacy,
            'N_global_first_transit':  n_global_ft,
            'global_legacy_inflation': (n_global_legacy / n_global_ft
                                        if n_global_ft > 0 else 0.0),
        })

        # Update prev_r_state for next dump (vectorized).  Only the rows we
        # saw this dump get updated; rows for particles not in this dump
        # retain their previous prev_r value (those particles have left the
        # simulation domain or are masked out by --max-dumps).
        prev_r_state[rows] = r

        if k == 0 or (k + 1) % 10 == 0 or k == len(iters) - 1:
            if global_mode:
                tcut_marker = ' [past t_cut]' if t_ps > t_cut_ps else ''
                print(f'  dump {k+1:3d}/{len(iters):3d}  t={t_ps:7.1f} ps  '
                      f'global_ft={n_global_ft:>6d}  '
                      f'global_legacy={n_global_legacy:>6d}  '
                      f'wrapped={n_wrapped_total:>6d}{tcut_marker}')
            else:
                print(f'  dump {k+1:3d}/{len(iters):3d}  t={t_ps:7.1f} ps  '
                      f'rod={n_rod_active:>6d}  '
                      f'outer={n_outer_active:>6d}  '
                      f'base={n_base_active:>7d}  '
                      f'wrapped={n_wrapped_total:>6d}')

    # ── output CSVs and summary ───────────────────────────────────────────
    out_prefix = args.out_prefix or args.run_dir.name
    out_dir    = args.run_dir
    mode_infix = '_global' if global_mode else ''
    ft_path    = out_dir / f'{out_prefix}_fusion_rate_first_transit{mode_infix}_{rxn_tag}.csv'
    wa_path    = out_dir / f'{out_prefix}_wrap_artifact_metrics{mode_infix}_{rxn_tag}.csv'
    sm_path    = out_dir / f'{out_prefix}_first_transit_summary{mode_infix}_{rxn_tag}.txt'

    if ft_rows:
        with ft_path.open('w', newline='') as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(ft_rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(ft_rows)
        print(f'Wrote: {ft_path}')

    if wa_rows:
        with wa_path.open('w', newline='') as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(wa_rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(wa_rows)
        print(f'Wrote: {wa_path}')

    with sm_path.open('w') as fh:
        fh.write('=' * 92 + '\n')
        fh.write(f'  FIRST-TRANSIT FUSION ANALYSIS - mode={args.mode}  reaction={rxn_tag} ({rxn_label})\n')
        fh.write(f'  Run: {args.run_dir.name}\n')
        fh.write('=' * 92 + '\n\n')
        fh.write(f'  Reaction parameters:\n')
        fh.write(f'    sigma            = {SIGMA_M2:.3e} m^2\n')
        fh.write(f'    threshold        = {THRESHOLD:.1f} keV\n')
        fh.write(f'    E_fusion         = {E_FUSION_J / MEV:.2f} MeV\n')
        fh.write(f'    target nucleus   = {target_iso}\n\n')
        fh.write(f'  Geometry:\n')
        if rod_enabled:
            fh.write(f'    rod_radius_m       = {rod_radius_m:.3e}     (zone: r < {R_rod*1e6:.0f} um)   ENABLED\n')
        else:
            fh.write(f'    rod zone           DISABLED  (rod_radius_m={rod_radius_m}, rod_placed={rod_placed})\n')
        if outer_enabled:
            fh.write(f'    outer_radius_m     = {outer_radius_m:.3e}     (zone: {R_out_lo*1e6:.0f}-{R_out_hi*1e6:.0f} um)   ENABLED\n')
        else:
            fh.write(f'    outer zone         DISABLED  (outer_radius_m={outer_radius_m}, outer_placed={outer_placed})\n')
        fh.write(f'    box half-width     = {half_box_min*1e6:.0f} um\n')
        fh.write(f'    wrap |dr| trigger  = {wrap_dist_threshold*1e6:.0f} um\n\n')
        fh.write(f'  Target densities (per zone n_{target_iso}, additive model):\n')
        fh.write(f'    base region                  = {base_n:.2e}\n')
        fh.write(f'    rod   additive contribution  = {rod_n_addl:.2e}\n')
        fh.write(f'    rod   total (base + addl)    = {rod_n:.2e}     placed: {rod_placed}\n')
        fh.write(f'    outer additive contribution  = {outer_n_addl:.2e}\n')
        fh.write(f'    outer total (base + addl)    = {outer_n:.2e}     placed: {outer_placed}\n\n')
        fh.write(f'  Cumulative fusion energy (J) - ZONE first-transit method:\n')
        fh.write(f'    rod        = {cum_E_rod_J:.4e}\n')
        fh.write(f'    outer      = {cum_E_outer_J:.4e}\n')
        fh.write(f'    total      = {cum_E_rod_J + cum_E_outer_J:.4e}\n\n')
        fh.write(f'  Cumulative fusion energy (J) - ZONE legacy (instantaneous occupation) method:\n')
        fh.write(f'    rod        = {cum_E_legacy_rod_J:.4e}\n')
        fh.write(f'    outer      = {cum_E_legacy_outer_J:.4e}\n')
        fh.write(f'    total      = {cum_E_legacy_rod_J + cum_E_legacy_outer_J:.4e}\n\n')
        ratio_outer = (cum_E_outer_J / cum_E_legacy_outer_J) if cum_E_legacy_outer_J > 0 else 0.0
        ratio_rod   = (cum_E_rod_J   / cum_E_legacy_rod_J)   if cum_E_legacy_rod_J   > 0 else 0.0
        fh.write(f'  ZONE wrap-artifact correction factor (first-transit / legacy):\n')
        fh.write(f'    rod        = {ratio_rod:.3f}\n')
        fh.write(f'    outer      = {ratio_outer:.3f}\n')
        n_wrapped_final = int(wrapped.sum())
        fh.write(f'\n  Total wrapped particles by end of run: {n_wrapped_final}\n')
        fh.write(f'  (First-transit method excludes their post-wrap contributions.)\n')

        # ── ZONE SR (spatially-resolved) headline block (v0.7 SR patch) ──
        # base + rod + outer; this is the correct Paper 3 gain numerator.
        # Previously, zone mode dropped the base-region contribution entirely
        # — i.e. every reconnection-accelerated proton at the X-line annulus,
        # plus everything outside placed zones, fusing against base_n. This
        # block reports the correction explicitly.
        fh.write('\n' + '=' * 92 + '\n')
        fh.write('  ZONE SR (spatially-resolved) HEADLINE BLOCK (v0.7 SR patch)\n')
        fh.write('  Paper 3 gain numerator = total_SR (base + rod + outer)\n')
        fh.write('=' * 92 + '\n\n')
        fh.write(f'  Cumulative fusion energy (J) - SR first-transit method:\n')
        fh.write(f'    base       = {cum_E_base_J:.4e}\n')
        fh.write(f'    rod        = {cum_E_rod_J:.4e}\n')
        fh.write(f'    outer      = {cum_E_outer_J:.4e}\n')
        cum_E_total_SR_J_final = cum_E_rod_J + cum_E_outer_J + cum_E_base_J
        fh.write(f'    total_SR   = {cum_E_total_SR_J_final:.4e}     <-- HEADLINE\n\n')
        # Compare against the pre-patch rod+outer-only first-transit total to
        # quantify how much physics the old zone mode was dropping.
        cum_E_old_FT_total = cum_E_rod_J + cum_E_outer_J
        if cum_E_old_FT_total > 0:
            sr_uplift = cum_E_total_SR_J_final / cum_E_old_FT_total
            fh.write(f'  SR uplift over pre-patch zone first-transit '
                     f'(total_SR / (rod+outer)):  {sr_uplift:.3f}\n')
            fh.write(f'  (Value > 1 means the base region contains '
                     f'previously-dropped fusion contribution. '
                     f'A factor of ~1.0 indicates either base_n=0 for this '
                     f'reaction or no particles above threshold outside placed zones.)\n')
        elif cum_E_base_J > 0:
            fh.write(f'  SR uplift over pre-patch zone first-transit:  '
                     f'infinite (placed zones contributed nothing; '
                     f'base region was the entire fusion source)\n')
        else:
            fh.write(f'  SR uplift over pre-patch zone first-transit:  '
                     f'0 (no fusion contribution anywhere — likely '
                     f'base_n_{target_iso}=0 AND no zones placed)\n')
        if exterior_enabled:
            fh.write(f'\n  Exterior cut applied to base tally: '
                     f'r > {R_exterior*1e6:.0f} um excluded '
                     f'(box_half_width {min(lx_m,lz_m)/2.0*1e6:.0f}um '
                     f'- margin {args.exterior_margin_um:.0f}um)\n')
        else:
            fh.write(f'\n  Exterior cut DISABLED on base tally '
                     f'(--exterior-margin-um=0; legacy v0.6 behaviour)\n')
        if laser_energy_j and laser_energy_j > 0:
            G_SR     = cum_E_total_SR_J_final / laser_energy_j
            G_old_FT = cum_E_old_FT_total     / laser_energy_j
            fh.write(f'\n  Gain vs laser energy ({laser_energy_j:.3e} J):\n')
            fh.write(f'    G_SR (total_SR / E_laser, headline)            = {G_SR:.4f}\n')
            fh.write(f'    G_old_FT (pre-patch rod+outer / E_laser)       = {G_old_FT:.4f}\n')
        else:
            fh.write(f'\n  Gain vs laser energy: laser_energy_j unavailable '
                     f'from step_time_index.csv (cannot compute G_SR).\n')

        # ── Global-mode summary block (only meaningful when global_mode=True)
        if global_mode:
            fh.write('\n' + '=' * 92 + '\n')
            fh.write('  GLOBAL FIRST-TRANSIT BLOCK (Paper 1 v0.7 methodology)\n')
            fh.write('=' * 92 + '\n\n')
            if t_cut_ps == float('inf'):
                tcut_disp = 'inf (end-of-run; reconnection_rate_offline.csv unavailable)'
            else:
                tcut_disp = f'{t_cut_ps:.2f} ps'
            fh.write(f'  Termination cuts (applied uniformly across all three tallies):\n')
            fh.write(f'    t_cut_ps                       = {tcut_disp}\n')
            fh.write(f'    t_cut source                   = {t_cut_source}\n')
            if b_min_T is not None and b_initial_T is not None:
                fh.write(f'    B_initial (post-Biermann-ramp) = {b_initial_T:.2f} T  '
                         f'(reference, NOT seed B)\n')
                fh.write(f'    B_min (lobe-field minimum)     = {b_min_T:.2f} T  '
                         f'at t = {t_cut_ps:.2f} ps\n')
            if exterior_enabled:
                fh.write(f'    Exterior cut                   = r > {R_exterior*1e6:.0f} um  '
                         f'(box_half_width {min(lx_m,lz_m)/2.0*1e6:.0f}um '
                         f'- margin {args.exterior_margin_um:.0f}um)\n')
            else:
                fh.write(f'    Exterior cut                   = DISABLED  '
                         f'(legacy v0.6: corner particles included)\n')
            fh.write(f'    Bulk target density (n_{target_iso})       = {bulk_n:.3e} m^-3\n')
            fh.write(f'    Total particles wrapped          = {n_wrapped_final}\n')
            if n_above_thr_at_tcut > 0:
                fh.write(f'    N above-threshold at t_cut       = {n_above_thr_at_tcut}  '
                         f'(post-exterior-cut)\n')
            fh.write('\n')
            fh.write(f'  Cumulative fusion energy (J):\n')
            fh.write(f'    GLOBAL first-transit (wrap+t_cut excluded)   = {cum_E_global_J:.4e}\n')
            fh.write(f'    GLOBAL pre-t_cut    (wrap included, t<=t_cut) = {cum_E_global_pretcut_J:.4e}\n')
            fh.write(f'    GLOBAL legacy       (no termination)          = {cum_E_global_legacy_J:.4e}\n\n')
            r_wrap = (cum_E_global_J / cum_E_global_pretcut_J) if cum_E_global_pretcut_J > 0 else 0.0
            r_tcut = (cum_E_global_pretcut_J / cum_E_global_legacy_J) if cum_E_global_legacy_J > 0 else 0.0
            r_tot  = (cum_E_global_J / cum_E_global_legacy_J) if cum_E_global_legacy_J > 0 else 0.0
            fh.write(f'  Correction factors (first-transit / reference):\n')
            fh.write(f'    wrap-only (FT / pre-t_cut)      = {r_wrap:.4f}   (isolates wrap effect)\n')
            fh.write(f'    t_cut-only (pre-t_cut / legacy) = {r_tcut:.4f}   (isolates t_cut effect)\n')
            fh.write(f'    combined  (FT / legacy)         = {r_tot:.4f}   (Paper 1 headline correction)\n\n')
            if laser_energy_j and laser_energy_j > 0:
                G_ft     = cum_E_global_J        / laser_energy_j
                G_pre    = cum_E_global_pretcut_J/ laser_energy_j
                G_legacy = cum_E_global_legacy_J / laser_energy_j
                fh.write(f'  Gain vs laser energy ({laser_energy_j:.3e} J):\n')
                fh.write(f'    G_FT (first-transit, Paper 1 v0.7 headline)  = {G_ft:.3f}\n')
                fh.write(f'    G_pre-t_cut (wrap not excluded)              = {G_pre:.3f}\n')
                fh.write(f'    G_legacy (full simulation, exterior-cut on)  = {G_legacy:.3f}\n')
                fh.write(f'\n  Interpretation (Paper 1 v0.7 seed-vs-sustain framing):\n')
                fh.write(f'    G_FT     = reconnection-event-specific gain (seed)\n')
                fh.write(f'    G_legacy = total gain over simulation including thermal sustain\n')
                if G_legacy > 0:
                    sustain_frac = (G_legacy - G_ft) / G_legacy * 100.0
                    fh.write(f'    Sustain fraction (1 - G_FT/G_legacy) = {sustain_frac:.1f}%  '
                             f'of total is bulk-thermal post-reconnection contribution\n')
            else:
                fh.write(f'  Gain vs laser energy: laser_energy_j unavailable from '
                         f'step_time_index.csv (cannot compute G).\n')
                fh.write(f'  Workaround: provide laser energy via the gain = E_fus / E_laser\n')
                fh.write(f'  formula manually, using the cumulative_fusion_energy values above.\n')

        fh.write('=' * 92 + '\n')

    print(f'Wrote: {sm_path}')
    print()
    print(f'Done.  mode={args.mode}.  reaction={rxn_tag}.')
    if global_mode:
        print(f'Paper 1 headline: G_FT (first-transit gain) is the corrected '
              f'wrap+t_cut figure in the GLOBAL block of the summary.')
    else:
        print(f'Compare first-transit vs legacy in summary.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
