#!/usr/bin/env python3
"""
pb11_phase_analysis.py — phase-resolved analysis for ring-reconnection runs

Reads the cache .npz files produced by visualize_all.py and performs:
  1. Peak detection across zone E95 timeseries
  2. Phase classification (compression-pinch, X-line, inflow, sustained, equilibration)
  3. Time-resolved fusion rate integration (zone-resolved)
  4. Cumulative fusion energy and gain calculation
  5. Oscillation frequency analysis
  6. Reconnection-signature quantification

Outputs a structured text report with phase-by-phase breakdown that captures
the multi-peak reconnection dynamics that early-third/late-third averaging
misses.

USAGE:
  python pb11_phase_analysis.py --dir runs/p1_hd_512_1500_ultrafine
  python pb11_phase_analysis.py --dir runs/X --report custom_report.txt
  python pb11_phase_analysis.py --dir runs/X --laser-energy 5.0  # for gain calc

The script expects:
  <dir>/figures/cache/zone_timeseries.npz
  <dir>/figures/cache/snapshot_data.npz   (optional, for spectrum analysis)
  <dir>/figures/cache/field_timeseries.npz (optional, for B-field analysis)
  <dir>/run_meta.txt
  <dir>/fusion_accounting_notes.txt        (optional, for cross-section data)

If the cache files are missing, falls back to reading particle dumps directly
(slower but fully functional).
"""

import argparse
import os
import sys
import re
import numpy as np
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONSTANTS
# =============================================================================

# p-11B fusion physics (from fusion_accounting_notes.txt)
P11B_THRESHOLD_KEV = 500.0       # E_thr for cross-section integration
P11B_PEAK_KEV = 675.0            # Peak cross-section energy
P11B_SIGMA_M2 = 1.200e-31        # Peak cross-section [m^2]
P11B_ENERGY_MEV = 8.680          # Energy released per reaction
P11B_N_ALPHAS = 3                # Alpha particles per reaction

# Particle thermal velocity at threshold (used for fusion rate calc)
PROTON_MASS_KG = 1.6726e-27
EV_TO_J = 1.602e-19
V_THR_AT_500KEV = np.sqrt(2.0 * P11B_THRESHOLD_KEV * 1e3 * EV_TO_J / PROTON_MASS_KG)

# Peak detection thresholds
PEAK_RATIO_THRESHOLD = 1.05      # core/spot must exceed this for "core acceleration"
PEAK_RATIO_STRONG = 1.20         # for "strong acceleration"
PEAK_PROMINENCE_KEV = 30000.0    # min E95 prominence above local baseline


# =============================================================================
# I/O
# =============================================================================

def parse_zones_report(report_path):
    """
    Parse a zones report (output of pb11_zone_analysis.py) and extract
    timeseries data. Returns a dict matching the cache structure, or None
    if parsing fails.

    Used as a fallback when the cache file has corrupt/zero E95 values.
    """
    if not Path(report_path).exists():
        return None

    times = []
    n_core, n_inner, n_xline, n_spot = [], [], [], []
    e95_core, e95_inner, e95_xline, e95_spot = [], [], [], []

    with open(report_path) as f:
        in_table = False
        for line in f:
            line = line.strip()
            # Detect start of data table (header line)
            if 't(ps)' in line and 'N_core' in line and 'E95_core' in line:
                in_table = True
                continue
            # Detect end of table (TREND or blank or === )
            if in_table:
                if line.startswith('===') or line.startswith('TREND') or 'TREND' in line:
                    in_table = False
                    continue
                if not line or line.startswith('-'):
                    continue
                # Try to parse a data row
                # Format: t(ps) N_core N_inner N_xline N_spot | E95_core E95_inner E95_xline E95_spot | ratio xratio
                parts = line.replace('|', ' ').split()
                if len(parts) >= 11:
                    try:
                        t = float(parts[0])
                        nc = int(parts[1])
                        ni = int(parts[2])
                        nx = int(parts[3])
                        ns = int(parts[4])
                        ec = float(parts[5])
                        ei = float(parts[6])
                        ex = float(parts[7])
                        es = float(parts[8])
                        times.append(t)
                        n_core.append(nc)
                        n_inner.append(ni)
                        n_xline.append(nx)
                        n_spot.append(ns)
                        e95_core.append(ec)
                        e95_inner.append(ei)
                        e95_xline.append(ex)
                        e95_spot.append(es)
                    except (ValueError, IndexError):
                        continue

    if not times:
        return None

    return {
        'times_ps': np.array(times),
        'N_core': np.array(n_core, dtype=float),
        'N_inner': np.array(n_inner, dtype=float),
        'N_xline': np.array(n_xline, dtype=float),
        'N_spot': np.array(n_spot, dtype=float),
        'E95_core': np.array(e95_core),
        'E95_inner': np.array(e95_inner),
        'E95_xline': np.array(e95_xline),
        'E95_spot': np.array(e95_spot),
    }


def load_zone_cache(cache_dir):
    """Load zone_timeseries.npz from cache. Returns dict with keys."""
    path = cache_dir / 'zone_timeseries.npz'
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_field_cache(cache_dir):
    """Load field_timeseries.npz from cache. Returns dict or None."""
    path = cache_dir / 'field_timeseries.npz'
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_snapshot_cache(cache_dir):
    """Load snapshot_data.npz from cache. Returns dict or None."""
    path = cache_dir / 'snapshot_data.npz'
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def parse_run_meta(meta_path):
    """Parse run_meta.txt key=value file. Returns flat dict."""
    meta = {}
    if not meta_path.exists():
        return meta
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            if line.startswith('==') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            meta[k.strip()] = v.strip()
    return meta


def safe_float(v, default=0.0):
    """Parse a value to float, falling back if it has units suffixed."""
    try:
        return float(v)
    except (ValueError, TypeError):
        try:
            # strip trailing non-numeric (e.g., '85.0 T' -> '85.0')
            m = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', str(v))
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return default


# =============================================================================
# PEAK DETECTION
# =============================================================================

def find_peaks(t, y, prominence=None, min_distance=None):
    """
    Simple peak finder. Returns indices of local maxima where y[i] is greater
    than y[i-k] and y[i+k] for k in 1..min_distance, with optional prominence.

    Designed to find ALL peaks including small ones, which we filter later.
    """
    if len(y) < 3:
        return np.array([], dtype=int)

    # First derivative sign change: positive -> negative
    dy = np.diff(y)
    sign_changes = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1

    if len(sign_changes) == 0:
        return np.array([], dtype=int)

    if min_distance is not None and len(sign_changes) > 1:
        # Sort by y value descending and remove peaks too close to higher peaks
        sorted_idx = sign_changes[np.argsort(-y[sign_changes])]
        keep = []
        used = np.zeros(len(t), dtype=bool)
        for idx in sorted_idx:
            if not used[max(0, idx - min_distance):min(len(t), idx + min_distance + 1)].any():
                keep.append(idx)
                used[max(0, idx - min_distance):min(len(t), idx + min_distance + 1)] = True
        sign_changes = np.array(sorted(keep))

    if prominence is not None:
        # require peak height > local minimum + prominence
        kept = []
        for idx in sign_changes:
            # find local minima on either side
            left = y[max(0, idx - 5):idx]
            right = y[idx + 1:min(len(y), idx + 6)]
            local_min = min(left.min() if len(left) else y[idx],
                            right.min() if len(right) else y[idx])
            if y[idx] - local_min > prominence:
                kept.append(idx)
        sign_changes = np.array(kept)

    return sign_changes


def find_valleys(t, y):
    """Inverse of find_peaks."""
    return find_peaks(t, -y)


# =============================================================================
# PHASE CLASSIFICATION
# =============================================================================

def classify_phase(t_start, t_end, ratios, n_core, e95_core, e95_spot, e95_inner, e95_xline):
    """
    Given a time window with peak data, classify which phase it is.
    Returns a string label and a dict of properties.
    """
    n_core_avg = float(np.mean(n_core))
    n_core_max = float(np.max(n_core))
    n_core_initial = float(n_core[0])
    e95_core_max = float(np.max(e95_core))
    e95_xline_max = float(np.max(e95_xline))
    ratio_max = float(np.max(ratios))

    # Compression pinch: N_core spikes well above initial
    if n_core_max > 1.5 * n_core_initial and e95_core_max < 400000:
        return "COMPRESSION-PINCH", {
            "mechanism": "Field collapse driving radial inflow",
            "n_core_peak_factor": n_core_max / n_core_initial if n_core_initial else 0,
        }
    # X-line acceleration: ratio very high, both core and xline elevated
    if ratio_max > 1.5 and e95_core_max > 500000 and e95_xline_max > 350000:
        return "X-LINE ACCELERATION", {
            "mechanism": "Direct E-field acceleration at reconnection X-lines",
            "max_ratio": ratio_max,
        }
    # Inflow heating: n_core RISING during the window AND e95 elevated
    if len(n_core) > 1 and n_core[-1] > n_core[0] * 1.05 and e95_core_max > 400000:
        return "INFLOW HEATING", {
            "mechanism": "Hot plasma flowing into depleted core region",
            "n_core_growth": (n_core[-1] - n_core[0]) / n_core[0] if n_core[0] else 0,
        }
    # Sustained: ratio moderate, e95 sustained
    if ratio_max > 1.10:
        return "SUSTAINED RECONNECTION", {
            "mechanism": "Continuing reconnection-driven heating",
            "max_ratio": ratio_max,
        }
    # Equilibration
    return "EQUILIBRATION", {
        "mechanism": "Thermalized plasma, slow drift",
    }


def detect_acceleration_phases(t, e95_core, e95_inner, e95_xline, e95_spot, n_core,
                                ratio_threshold=1.05):
    """
    Detect distinct acceleration phases in the timeseries.

    Returns list of phase dicts: each with keys 'label', 't_start', 't_end',
    'peak_t', 'peak_e95_core', 'peak_ratio', 'mechanism', 'n_core_at_peak'.
    """
    ratio_cs = e95_core / np.maximum(e95_spot, 1.0)

    # Find peaks in core E95 (smooth a bit to suppress noise)
    e95_core_sm = e95_core.copy()
    if len(e95_core_sm) > 3:
        # 3-point boxcar smoothing
        e95_core_sm = np.convolve(e95_core_sm, np.ones(3) / 3.0, mode='same')

    peak_idx = find_peaks(t, e95_core_sm,
                          prominence=PEAK_PROMINENCE_KEV,
                          min_distance=2)

    phases = []
    for pk in peak_idx:
        # Find phase boundaries: walk back to local minimum, walk forward to local minimum
        left = pk
        while left > 0 and e95_core_sm[left - 1] < e95_core_sm[left]:
            left -= 1
        right = pk
        while right < len(t) - 1 and e95_core_sm[right + 1] < e95_core_sm[right]:
            right += 1

        # Skip if ratio at peak is below threshold (means it's not a core acceleration)
        # — except for compression pinches which can have ratio < 1
        if ratio_cs[pk] < ratio_threshold and n_core[pk] < 1.5 * n_core[0]:
            continue

        label, props = classify_phase(
            t[left], t[right],
            ratio_cs[left:right+1],
            n_core[left:right+1],
            e95_core[left:right+1],
            e95_spot[left:right+1],
            e95_inner[left:right+1],
            e95_xline[left:right+1],
        )

        phases.append({
            'label': label,
            't_start': float(t[left]),
            't_end': float(t[right]),
            'peak_t': float(t[pk]),
            'peak_e95_core': float(e95_core[pk]),
            'peak_e95_inner': float(e95_inner[pk]),
            'peak_e95_xline': float(e95_xline[pk]),
            'peak_e95_spot': float(e95_spot[pk]),
            'peak_ratio': float(ratio_cs[pk]),
            'n_core_at_peak': float(n_core[pk]),
            'mechanism': props.get('mechanism', ''),
            'props': props,
        })

    # Add equilibration phase if last detected peak is well before end
    if phases and phases[-1]['t_end'] < t[-1] - 20.0:
        eq_start = phases[-1]['t_end']
        eq_idx = np.where(t >= eq_start)[0]
        if len(eq_idx) > 0:
            i0 = eq_idx[0]
            phases.append({
                'label': 'EQUILIBRATION',
                't_start': float(t[i0]),
                't_end': float(t[-1]),
                'peak_t': float(t[-1]),
                'peak_e95_core': float(e95_core[-1]),
                'peak_e95_inner': float(e95_inner[-1]),
                'peak_e95_xline': float(e95_xline[-1]),
                'peak_e95_spot': float(e95_spot[-1]),
                'peak_ratio': float(ratio_cs[-1]),
                'n_core_at_peak': float(n_core[-1]),
                'mechanism': 'Thermal equilibrium, slow drift',
                'props': {},
            })

    return phases


# =============================================================================
# FUSION RATE INTEGRATION
# =============================================================================

def estimate_fast_fraction(e95_kev, threshold_kev=P11B_THRESHOLD_KEV):
    """
    Estimate the fraction of particles above the fusion threshold given E95.

    For a Maxwell-Boltzmann distribution, E95 (95th percentile of energy) relates
    to thermal temperature T by E95 ≈ 3.0 * kT (for 3D isotropic distribution).
    The fraction above E_thr for a Maxwellian is:
        ff(E_thr) = (2/sqrt(pi)) * Gamma_inc(3/2, E_thr/kT)

    For practical use we approximate:
        kT ≈ E95 / 3.0
        ff ≈ exp(-E_thr / kT)   for E_thr/kT > 1
        ff ≈ 1 - exp(...)        for E_thr/kT < 1

    More accurate model that handles both regimes:
        u = E_thr / kT
        ff(u) ≈ 2/sqrt(pi) * sqrt(u) * exp(-u) * (1 + 1/(2u) + ...)  [for u >> 1]
        ff(u) ≈ 1 - 4u^(3/2)/(3*sqrt(pi))  [for u << 1]

    Simplified Maxwell tail formula:
        ff ≈ (1 + 2*sqrt(u/pi)) * exp(-u)  is a reasonable approximation
    """
    e95 = np.maximum(e95_kev, 1.0)
    kT = e95 / 3.0  # approximate thermal temperature [keV]
    u = threshold_kev / kT  # dimensionless threshold

    # Maxwellian tail integral approximation
    # exp(-u) * (1 + 2*sqrt(u/pi))
    ff = np.exp(-u) * (1.0 + 2.0 * np.sqrt(u / np.pi))

    # Clamp to [0, 1]
    ff = np.clip(ff, 0.0, 1.0)
    return ff


def compute_fusion_yield(t_ps, e95_zones, n_zones, n_target=5e25, sigma=P11B_SIGMA_M2,
                        e_rxn_j=P11B_ENERGY_MEV * 1e6 * EV_TO_J,
                        threshold_kev=P11B_THRESHOLD_KEV,
                        fast_frac_override=None,
                        domain_volume_m3=None,
                        method='maxwellian'):
    """
    Estimate cumulative fusion yield from zone-resolved E95 and N data.

    Computes fusion rate as:
        R(t) = sum over zones of [n_p_zone(t) * fast_frac(t)] * n_target * sigma * v_thr * V_zone

    Where:
        n_p_zone = N_zone / V_zone  (proton density in zone)
        fast_frac = fraction of protons above E_thr
        v_thr = relative velocity at threshold energy
        V_zone = volume of the zone

    Three fast_frac methods:
        'maxwellian': proper Maxwellian tail integral from zone-averaged E95
                     (conservative; reports thermal fusion only)
        'optimistic': assume fast_frac = 1 when E95 > threshold
                     (upper bound; treats all particles as monoenergetic)
        'csv_match':  use 0.99 when zone_e95 > 0.5 * threshold
                     (matches inline CSV diagnostic, for cross-verification)

    Returns:
        t_ps: time array
        rate_per_s: fusion rate per second at each timepoint
        cum_alphas: cumulative alpha count
        cum_energy_j: cumulative fusion energy [J]
    """
    rate = np.zeros(len(t_ps))

    # Total macro-particle count (sum of all zones at t=0)
    n_macro_total = sum(zone_n[0] for zone_n in n_zones)

    if domain_volume_m3 is None:
        # 9.6 mm × 9.6 mm × unit thickness 1 m (for 2D rate per meter)
        domain_volume_m3 = 9.6e-3 * 9.6e-3 * 1.0

    n_phys_per_macro = n_target * domain_volume_m3 / max(n_macro_total, 1)

    for zone_e95, zone_n in zip(e95_zones, n_zones):
        if fast_frac_override is not None:
            ff = np.full_like(zone_e95, fast_frac_override)
        elif method == 'optimistic':
            ff = np.where(zone_e95 > threshold_kev, 1.0, 0.0)
        elif method == 'csv_match':
            ff = np.where(zone_e95 > 0.5 * threshold_kev, 0.99, 0.0)
        else:  # maxwellian (default)
            ff = estimate_fast_fraction(zone_e95, threshold_kev=threshold_kev)

        # Physical proton count in this zone at each timepoint
        n_phys_zone = zone_n * n_phys_per_macro

        # Reaction rate per second [1/s] from this zone
        rate += n_phys_zone * ff * n_target * sigma * V_THR_AT_500KEV

    # Integrate using trapezoidal rule
    if len(t_ps) > 1:
        dt_s = np.diff(t_ps) * 1e-12
        rate_avg = 0.5 * (rate[:-1] + rate[1:])
        counts_per_step = rate_avg * dt_s
        cum_counts = np.concatenate([[0.0], np.cumsum(counts_per_step)])
    else:
        cum_counts = np.zeros(len(t_ps))

    cum_alphas = cum_counts * P11B_N_ALPHAS
    cum_energy_j = cum_counts * e_rxn_j

    return t_ps, rate, cum_alphas, cum_energy_j


# =============================================================================
# OSCILLATION ANALYSIS
# =============================================================================

def analyze_oscillations(t, signal):
    """Compute peak-to-peak spacing in a signal, useful for natural-frequency."""
    pks = find_peaks(t, signal, prominence=10000.0, min_distance=1)
    if len(pks) < 2:
        return None

    spacings = np.diff(t[pks])  # [ps]
    avg_spacing_ps = float(np.mean(spacings))
    median_spacing_ps = float(np.median(spacings))
    natural_freq_ghz = 1000.0 / avg_spacing_ps if avg_spacing_ps > 0 else 0

    return {
        'n_peaks': len(pks),
        'peak_times': t[pks].tolist(),
        'peak_values': signal[pks].tolist(),
        'spacings_ps': spacings.tolist(),
        'avg_spacing_ps': avg_spacing_ps,
        'median_spacing_ps': median_spacing_ps,
        'natural_freq_ghz': natural_freq_ghz,
    }


# =============================================================================
# B-FIELD ANALYSIS
# =============================================================================

def analyze_field_evolution(field_data):
    """Extract B-field collapse / oscillation statistics from field_timeseries."""
    if field_data is None:
        return None

    # Find common keys
    t_field = None
    b_max = None
    b_mean = None
    j_max = None

    for key in field_data:
        k = key.lower()
        arr = field_data[key]
        if 't_ps' in k or 'time' in k:
            t_field = np.asarray(arr).flatten()
        elif 'b_max' in k or 'bmax' in k:
            b_max = np.asarray(arr).flatten()
        elif 'b_mean' in k or 'bmean' in k:
            b_mean = np.asarray(arr).flatten()
        elif 'j_max' in k or 'jmax' in k:
            j_max = np.asarray(arr).flatten()

    if t_field is None or b_max is None:
        return None

    b_initial = float(b_max[0]) if len(b_max) else 0.0
    b_min = float(np.min(b_max))
    b_final = float(b_max[-1])
    collapse_frac = (b_initial - b_min) / b_initial if b_initial else 0.0

    osc = analyze_oscillations(t_field, b_max)

    return {
        't_ps': t_field.tolist(),
        'b_max': b_max.tolist(),
        'b_mean': b_mean.tolist() if b_mean is not None else None,
        'j_max': j_max.tolist() if j_max is not None else None,
        'b_initial': b_initial,
        'b_min': b_min,
        'b_final': b_final,
        'collapse_fraction': collapse_frac,
        'oscillations': osc,
    }


# =============================================================================
# MAIN ANALYSIS DRIVER
# =============================================================================

def analyze_run(run_dir, laser_energy_j=5.0, b_threshold=500.0,
                zones_report_path=None, debug=False):
    """Top-level analysis. Returns dict of results."""
    run_dir = Path(run_dir)
    cache_dir = run_dir / 'figures' / 'cache'

    # Load metadata
    meta = parse_run_meta(run_dir / 'run_meta.txt')
    n_steps = int(safe_float(meta.get('n_steps', 0)))
    dt_s = safe_float(meta.get('time_step_s', 0))
    base_density = safe_float(meta.get('base_density', 5e25))
    base_n_b11 = safe_float(meta.get('base_n_b11', base_density))
    b_seed = safe_float(meta.get('b_seed_t', 0))
    plasma_beta = safe_float(meta.get('plasma_beta', 0))

    # If user provided explicit zones report, use it directly
    if zones_report_path:
        explicit_path = Path(zones_report_path)
        if explicit_path.exists():
            print(f"INFO: Using user-supplied zones report: {explicit_path}")
            zone_data = parse_zones_report(explicit_path)
            used_fallback = True
            if zone_data is None:
                print(f"ERROR: Could not parse zones report at {explicit_path}")
                sys.exit(1)
        else:
            print(f"ERROR: Zones report not found at {explicit_path}")
            sys.exit(1)
    else:
        # Load zone cache
        zone_data = load_zone_cache(cache_dir)
        used_fallback = False

    # === v12.18 FIX: Detect corrupt cache (E95 fields all zero) and fall back
    # to parsing the zones report. This handles the bug in older versions of
    # visualize_all.py where the gamma calculation produced zero KE values.
    cache_is_corrupt = False
    if zone_data is not None and not used_fallback:
        for key in ('E95_core', 'e95_core', 'E95_spot', 'e95_spot'):
            if key in zone_data:
                arr = np.asarray(zone_data[key])
                if arr.size > 0 and float(np.max(arr)) == 0.0:
                    cache_is_corrupt = True
                    break

    if (zone_data is None or cache_is_corrupt) and not used_fallback:
        # Try to find the zones report in multiple possible locations
        run_name = run_dir.name
        possible_reports = [
            # Inside the run directory
            run_dir / 'zone_report.txt',
            run_dir / f'{run_name}_zones.txt',
            run_dir / 'post_analysis_reports' / f'{run_name}_zones.txt',
            # Sibling post_analysis_reports/ (project structure: project/runs/X/, project/post_analysis_reports/)
            run_dir.parent / 'post_analysis_reports' / f'{run_name}_zones.txt',
            # Two levels up (project/post_analysis_reports/, runs/X/)
            run_dir.parent.parent / 'post_analysis_reports' / f'{run_name}_zones.txt',
            # Up three levels (occasionally needed)
            run_dir.parent.parent.parent / 'post_analysis_reports' / f'{run_name}_zones.txt',
            # Current working directory
            Path.cwd() / 'post_analysis_reports' / f'{run_name}_zones.txt',
        ]

        if debug:
            print(f"Cache is corrupt or missing. Searching for zones report fallback...")
            print(f"  Run name: {run_name}")
            for p in possible_reports:
                print(f"  Try: {p} exists={p.exists()}")

        report_data = None
        for report_path in possible_reports:
            if report_path.exists():
                if debug:
                    print(f"Trying fallback parser on: {report_path}")
                report_data = parse_zones_report(report_path)
                if report_data is not None:
                    if cache_is_corrupt:
                        print(f"WARNING: Cache file E95 values are zero (visualize_all bug).")
                        print(f"         Falling back to zones report: {report_path}")
                    else:
                        print(f"INFO: No cache file. Using zones report: {report_path}")
                    zone_data = report_data
                    used_fallback = True
                    break

        if zone_data is None or cache_is_corrupt:
            if zone_data is None:
                print(f"ERROR: No usable cache or zones report found.")
                print(f"  Cache dir: {cache_dir}")
                print(f"  Searched paths:")
                for p in possible_reports:
                    print(f"    {p}")
                print(f"")
                print(f"  Run pb11_zone_analysis.py first to generate the zones report,")
                print(f"  or fix visualize_all.py and regenerate the cache.")
                sys.exit(1)
            elif cache_is_corrupt and not used_fallback:
                print(f"WARNING: Cache is corrupt (E95=0) and no zones report found.")
                print(f"  Continuing with corrupt data — results will be wrong!")
                print(f"  To fix: regenerate cache via fixed visualize_all.py, OR")
                print(f"  generate zones report via pb11_zone_analysis.py")

    # Extract arrays from cache (handle different naming conventions)
    t_ps = None
    n_zones = {}
    e95_zones = {}

    for key in zone_data:
        k = key.lower()
        arr = zone_data[key]
        try:
            arr = np.asarray(arr).astype(float).flatten()
        except (ValueError, TypeError):
            continue

        # Strict key matching: exact name (lowercase) or prefix-based
        if k in ('t_ps', 'times_ps', 't', 'time_ps') or k.startswith('time'):
            t_ps = arr
        # Particle counts: must be exactly N_zone or n_zone or count_zone
        elif k == 'n_core' or k == 'count_core':
            n_zones['core'] = arr
        elif k == 'n_inner' or k == 'count_inner':
            n_zones['inner'] = arr
        elif k in ('n_xline', 'n_x_line') or k == 'count_xline':
            n_zones['xline'] = arr
        elif k == 'n_spot' or k == 'count_spot':
            n_zones['spot'] = arr
        # E95 (energy 95th percentile)
        elif k == 'e95_core' or k == 'energy95_core':
            e95_zones['core'] = arr
        elif k == 'e95_inner' or k == 'energy95_inner':
            e95_zones['inner'] = arr
        elif k in ('e95_xline', 'e95_x_line') or k == 'energy95_xline':
            e95_zones['xline'] = arr
        elif k == 'e95_spot' or k == 'energy95_spot':
            e95_zones['spot'] = arr
        # Skip Emean_*, fast_frac_*, iterations, etc. — not what we need

    if debug:
        print(f"Cache keys: {list(zone_data.keys())}")
        print(f"Loaded zones: n={list(n_zones.keys())}, e95={list(e95_zones.keys())}")

    if t_ps is None or len(n_zones) < 4 or len(e95_zones) < 4:
        print("ERROR: Could not extract all needed zone data from cache.")
        print(f"  Found keys: {list(zone_data.keys())}")
        sys.exit(1)

    # Detect phases
    phases = detect_acceleration_phases(
        t_ps,
        e95_zones['core'], e95_zones['inner'],
        e95_zones['xline'], e95_zones['spot'],
        n_zones['core']
    )

    # Compute fusion yield with 3 different methods for transparency
    e95_list = [e95_zones['core'], e95_zones['inner'],
                e95_zones['xline'], e95_zones['spot']]
    n_list = [n_zones['core'], n_zones['inner'],
              n_zones['xline'], n_zones['spot']]

    # Maxwellian (conservative): assumes thermal Maxwell distribution
    _, fusion_rate_maxw, cum_alphas_maxw, cum_e_maxw = compute_fusion_yield(
        t_ps, e95_list, n_list,
        n_target=base_n_b11, method='maxwellian',
    )

    # Optimistic: assumes monoenergetic acceleration
    _, fusion_rate_opt, cum_alphas_opt, cum_e_opt = compute_fusion_yield(
        t_ps, e95_list, n_list,
        n_target=base_n_b11, method='optimistic',
    )

    # CSV-match: matches what the inline diagnostic reports
    _, fusion_rate_csv, cum_alphas_csv, cum_e_csv = compute_fusion_yield(
        t_ps, e95_list, n_list,
        n_target=base_n_b11, method='csv_match',
    )

    # Use Maxwellian as the default for phase reporting
    fusion_rate = fusion_rate_maxw
    cum_alphas = cum_alphas_maxw
    cum_energy_j = cum_e_maxw

    # Per-phase fusion yield
    for ph in phases:
        i0 = np.searchsorted(t_ps, ph['t_start'], side='left')
        i1 = np.searchsorted(t_ps, ph['t_end'], side='right')
        i0 = max(0, i0)
        i1 = min(len(t_ps), i1)
        if i1 > i0:
            ph['fusion_energy_j'] = float(cum_energy_j[i1 - 1] - cum_energy_j[i0])
            ph['fusion_alphas'] = float(cum_alphas[i1 - 1] - cum_alphas[i0])
            ph['avg_fusion_rate'] = float(np.mean(fusion_rate[i0:i1]))
        else:
            ph['fusion_energy_j'] = 0.0
            ph['fusion_alphas'] = 0.0
            ph['avg_fusion_rate'] = 0.0

    # Field analysis
    field_data = load_field_cache(cache_dir)
    field_analysis = analyze_field_evolution(field_data) if field_data else None

    # Reconnection signature analysis
    ratio_cs = e95_zones['core'] / np.maximum(e95_zones['spot'], 1.0)
    time_above_1 = float(np.sum((ratio_cs > 1.0).astype(float)) * (t_ps[1] - t_ps[0])
                         if len(t_ps) > 1 else 0)
    time_above_strong = float(np.sum((ratio_cs > 1.5).astype(float)) * (t_ps[1] - t_ps[0])
                              if len(t_ps) > 1 else 0)
    avg_ratio_when_active = float(np.mean(ratio_cs[ratio_cs > 1.0])) if np.any(ratio_cs > 1.0) else 1.0

    # Oscillation analysis on core E95
    osc_e95 = analyze_oscillations(t_ps, e95_zones['core'])

    return {
        'run_dir': str(run_dir),
        'meta': meta,
        'plasma_beta': plasma_beta,
        'b_seed_t': b_seed,
        'n_steps': n_steps,
        't_total_ps': float(t_ps[-1]) if len(t_ps) else 0,
        'n_dumps': len(t_ps),
        'dump_cadence_ps': float(t_ps[1] - t_ps[0]) if len(t_ps) > 1 else 0,
        'phases': phases,
        'fusion': {
            'method': 'maxwellian',
            'rate_per_s': fusion_rate.tolist(),
            'cum_alphas': cum_alphas.tolist(),
            'cum_energy_j': cum_energy_j.tolist(),
            'final_energy_j': float(cum_energy_j[-1]),
            'final_alphas': float(cum_alphas[-1]),
            'final_rate_per_s': float(fusion_rate[-1]),
            'gain_vs_laser': float(cum_energy_j[-1]) / laser_energy_j,
            'laser_energy_j': laser_energy_j,
            # Comparison estimates
            'maxwellian': {
                'final_energy_j': float(cum_e_maxw[-1]),
                'final_alphas': float(cum_alphas_maxw[-1]),
                'gain': float(cum_e_maxw[-1]) / laser_energy_j,
            },
            'optimistic': {
                'final_energy_j': float(cum_e_opt[-1]),
                'final_alphas': float(cum_alphas_opt[-1]),
                'gain': float(cum_e_opt[-1]) / laser_energy_j,
            },
            'csv_match': {
                'final_energy_j': float(cum_e_csv[-1]),
                'final_alphas': float(cum_alphas_csv[-1]),
                'gain': float(cum_e_csv[-1]) / laser_energy_j,
            },
        },
        'reconnection': {
            'time_above_1_ps': time_above_1,
            'time_above_1_5_ps': time_above_strong,
            'avg_ratio_when_active': avg_ratio_when_active,
            'max_ratio': float(np.max(ratio_cs)),
            'max_ratio_t_ps': float(t_ps[np.argmax(ratio_cs)]),
        },
        'field': field_analysis,
        'oscillation': osc_e95,
        't_ps': t_ps.tolist(),
        'e95_zones': {k: v.tolist() for k, v in e95_zones.items()},
        'n_zones': {k: v.tolist() for k, v in n_zones.items()},
    }


# =============================================================================
# REPORT WRITING
# =============================================================================

def format_report(results, report_path=None):
    """Format the analysis results as a human-readable report."""
    lines = []
    L = lines.append

    L("=" * 92)
    L("  PHASE-RESOLVED ANALYSIS — " + results['run_dir'])
    L("=" * 92)
    L(f"  Generated:    {datetime.now().isoformat(timespec='seconds')}")
    L(f"  Run steps:    {results['n_steps']:,}")
    L(f"  Sim duration: {results['t_total_ps']:.2f} ps")
    L(f"  Dumps:        {results['n_dumps']}")
    L(f"  Cadence:      {results['dump_cadence_ps']:.2f} ps")
    L(f"  Plasma beta:  {results['plasma_beta']:.2f}")
    L(f"  B-seed:       {results['b_seed_t']:.1f} T")
    L("")

    # ---------- PHASES ----------
    L("=" * 92)
    L("  DETECTED ACCELERATION PHASES")
    L("=" * 92)

    if not results['phases']:
        L("  No distinct acceleration phases detected (uniform thermalization regime).")
    else:
        for i, ph in enumerate(results['phases'], 1):
            L(f"")
            L(f"  Phase {i}: {ph['label']}")
            L(f"    Window:           {ph['t_start']:7.2f} -> {ph['t_end']:7.2f} ps")
            L(f"    Mechanism:        {ph['mechanism']}")
            L(f"    Peak time:        t = {ph['peak_t']:.2f} ps")
            L(f"    Peak E95 core:    {ph['peak_e95_core']:>10,.0f} keV")
            L(f"    Peak E95 inner:   {ph['peak_e95_inner']:>10,.0f} keV")
            L(f"    Peak E95 xline:   {ph['peak_e95_xline']:>10,.0f} keV")
            L(f"    Peak E95 spot:    {ph['peak_e95_spot']:>10,.0f} keV")
            L(f"    Core/spot ratio:  {ph['peak_ratio']:.3f}")
            L(f"    N_core at peak:   {ph['n_core_at_peak']:>10,.0f}")
            L(f"    Avg fusion rate:  {ph['avg_fusion_rate']:.3e} /s")
            L(f"    Phase fusion E:   {ph['fusion_energy_j']:.3e} J")
            L(f"    Phase alphas:     {ph['fusion_alphas']:.3e}")

    L("")

    # ---------- FUSION ----------
    L("=" * 92)
    L("  CUMULATIVE FUSION YIELD")
    L("=" * 92)

    fus = results['fusion']
    L(f"  Final fusion rate (Maxwellian):  {fus['final_rate_per_s']:.3e} /s")
    L(f"")
    L(f"  COMPARISON OF FAST-FRACTION MODELS:")
    L(f"  {'Method':<15s} {'Final E (J)':>15s} {'Alphas':>15s} {'Gain':>10s}")
    L(f"  {'-'*60}")
    L(f"  {'Maxwellian':<15s} {fus['maxwellian']['final_energy_j']:>15.3e} "
      f"{fus['maxwellian']['final_alphas']:>15.3e} {fus['maxwellian']['gain']:>10.2f}")
    L(f"  {'Optimistic':<15s} {fus['optimistic']['final_energy_j']:>15.3e} "
      f"{fus['optimistic']['final_alphas']:>15.3e} {fus['optimistic']['gain']:>10.2f}")
    L(f"  {'CSV-matched':<15s} {fus['csv_match']['final_energy_j']:>15.3e} "
      f"{fus['csv_match']['final_alphas']:>15.3e} {fus['csv_match']['gain']:>10.2f}")
    L(f"")
    L(f"  Laser energy:                    {fus['laser_energy_j']:.1f} J")
    L(f"")
    L(f"  CAVEAT: Each method uses a different fast-fraction model:")
    L(f"   - Maxwellian: integrates Maxwell distribution tail above threshold")
    L(f"     (conservative; reflects bulk thermal fusion)")
    L(f"   - Optimistic: assumes 100% above threshold when E95 > threshold")
    L(f"     (upper bound; assumes monoenergetic acceleration)")
    L(f"   - CSV-matched: uses 99% above threshold when E95 > 0.5 * threshold")
    L(f"     (matches inline pb11 diagnostic; for cross-verification)")
    L(f"")
    L(f"  For paper claims, use the Maxwellian estimate as conservative.")
    L(f"  Accurate gain requires full energy distribution from raw particles.")
    L("")

    # ---------- RECONNECTION SIGNATURE ----------
    L("=" * 92)
    L("  RECONNECTION SIGNATURE")
    L("=" * 92)

    rec = results['reconnection']
    L(f"  Time core/spot > 1.0:    {rec['time_above_1_ps']:.1f} ps")
    L(f"  Time core/spot > 1.5:    {rec['time_above_1_5_ps']:.1f} ps")
    L(f"  Average ratio when >1:   {rec['avg_ratio_when_active']:.3f}")
    L(f"  Maximum ratio:           {rec['max_ratio']:.3f} at t = {rec['max_ratio_t_ps']:.2f} ps")
    L("")

    # ---------- FIELD ----------
    if results['field']:
        L("=" * 92)
        L("  B-FIELD EVOLUTION")
        L("=" * 92)

        f = results['field']
        L(f"  Initial |B|_max:        {f['b_initial']:.2f} T")
        L(f"  Minimum |B|_max:        {f['b_min']:.2f} T")
        L(f"  Final |B|_max:          {f['b_final']:.2f} T")
        L(f"  Collapse fraction:      {f['collapse_fraction']*100:.1f}%")

        if f['oscillations']:
            o = f['oscillations']
            L(f"  Field oscillations:     {o['n_peaks']} peaks detected")
            L(f"  Avg peak spacing:       {o['avg_spacing_ps']:.2f} ps")
            L(f"  Median peak spacing:    {o['median_spacing_ps']:.2f} ps")
            L(f"  Natural freq estimate:  {o['natural_freq_ghz']:.2f} GHz")
            # ── Cadence-aliasing warning ──
            cadence_ps = results.get('dump_cadence_ps', 0)
            if cadence_ps > 0:
                min_resolvable_period_ps = cadence_ps * 4  # Nyquist + safety margin
                if cadence_ps > 35.0 / 4:  # 35 ps = natural reconnection cycle
                    L("")
                    L(f"  ⚠ CADENCE WARNING: detected frequency may be aliased.")
                    L(f"    At {cadence_ps:.0f} ps cadence, can only resolve cycles > "
                      f"{min_resolvable_period_ps:.0f} ps (Nyquist).")
                    L(f"    True natural reconnection cycle is ~28 GHz (35 ps period); "
                      f"requires cadence ≤ 7 ps to resolve.")
                    L(f"    Detected {o['natural_freq_ghz']:.2f} GHz is likely a coarse "
                      f"envelope (e.g. ion cyclotron beat), not the fundamental cycle.")
        L("")

    # ---------- OSCILLATION (E95) ----------
    if results['oscillation']:
        L("=" * 92)
        L("  CORE E95 OSCILLATION ANALYSIS")
        L("=" * 92)

        o = results['oscillation']
        L(f"  Number of E95 peaks:    {o['n_peaks']}")
        if o['n_peaks'] > 0:
            L(f"  Peak times (ps):        {[f'{t:.1f}' for t in o['peak_times']]}")
            L(f"  Peak E95 (keV):         {[f'{v:.0f}' for v in o['peak_values']]}")
        if len(o['spacings_ps']) > 0:
            L(f"  Avg peak spacing:       {o['avg_spacing_ps']:.2f} ps")
            L(f"  Natural freq estimate:  {o['natural_freq_ghz']:.2f} GHz")

            # ── Cadence-aliasing warning ──
            cadence_ps = results.get('dump_cadence_ps', 0)
            aliased = (cadence_ps > 35.0 / 4)  # 35 ps = natural cycle period
            if cadence_ps > 0 and aliased:
                min_resolvable_period_ps = cadence_ps * 4
                L(f"")
                L(f"  ⚠ CADENCE WARNING: detected frequency may be aliased.")
                L(f"    At {cadence_ps:.0f} ps cadence, can only resolve cycles > "
                  f"{min_resolvable_period_ps:.0f} ps (Nyquist + safety margin).")
                L(f"    True natural reconnection cycle is ~28 GHz (35 ps period); "
                  f"requires cadence ≤ 7 ps to resolve.")
                L(f"    Detected {o['natural_freq_ghz']:.2f} GHz is likely a coarse "
                  f"envelope (e.g. ion cyclotron beat), not the fundamental")
                L(f"    reconnection cycle. For Paper 2 modulation studies, use ultrafine")
                L(f"    cadence (max-steps 4500, dump-period 50, 7.5 ps cadence).")
            else:
                L(f"")
                L(f"  IMPLICATION FOR PAPER 2:")
                L(f"  Multi-peak natural frequency ~{o['natural_freq_ghz']:.0f} GHz suggests modulation")
                L(f"  resonance window. Frequencies 1/4 to 4x of this may show enhanced")
                L(f"  or suppressed reconnection-driven heating.")
        L("")

    # ---------- TIMESERIES TABLE ----------
    L("=" * 92)
    L("  ZONE E95 TIMESERIES (selected)")
    L("=" * 92)
    L(f"    {'t(ps)':>8s}  {'E95_core':>10s}  {'E95_inner':>10s}  {'E95_xline':>10s}  {'E95_spot':>10s}  {'ratio':>6s}")
    L("    " + "-" * 70)

    t_arr = results['t_ps']
    e95 = results['e95_zones']
    # show every ~5th point or all if <30 points
    stride = max(1, len(t_arr) // 30)
    for i in range(0, len(t_arr), stride):
        ratio = e95['core'][i] / max(e95['spot'][i], 1.0)
        L(f"    {t_arr[i]:>8.2f}  {e95['core'][i]:>10,.0f}  {e95['inner'][i]:>10,.0f}  "
          f"{e95['xline'][i]:>10,.0f}  {e95['spot'][i]:>10,.0f}  {ratio:>6.3f}")
    L("")

    L("=" * 92)
    L("  END PHASE-RESOLVED ANALYSIS")
    L("=" * 92)

    text = "\n".join(lines)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(text)
    return text


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase-resolved analysis for ring-reconnection runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--dir', required=True,
                        help='Run directory (containing figures/cache/*.npz)')
    parser.add_argument('--report', default=None,
                        help='Output report path. Default: <dir>/phase_analysis_report.txt')
    parser.add_argument('--laser-energy', type=float, default=5.0,
                        help='Laser energy in J for gain calculation. Default: 5.0')
    parser.add_argument('--zones-report', default=None,
                        help='Explicit path to zones report (output of pb11_zone_analysis.py). '
                             'Used when cache file is missing or corrupt.')
    parser.add_argument('--debug', action='store_true', help='Verbose output')

    args = parser.parse_args()
    run_dir = Path(args.dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    report_path = args.report or (run_dir / 'phase_analysis_report.txt')

    print(f"Analyzing run: {run_dir}")
    print(f"Report will be written to: {report_path}")
    print("")

    results = analyze_run(run_dir, laser_energy_j=args.laser_energy,
                          zones_report_path=args.zones_report, debug=args.debug)
    text = format_report(results, report_path=report_path)
    print(text)


if __name__ == '__main__':
    main()
