#!/usr/bin/env python3
"""
fermi_bounce_check.py — Compare q3 baseline / q3 two-pair / q4 Fermi-bounce runs

Tests the Fermi-bounce hypothesis: does perturbative-mode (alternating polarity)
drive at 5 GHz rotation, modulated at 10 GHz schedule (matched to natural
plasma resonance), produce a non-thermal tail that bulk heating cannot?

Usage:
    cd ~/LaserFusionResearch/research/laser-plasma-research
    python tools/fermi_bounce_check.py

    # Or with explicit paths:
    python tools/fermi_bounce_check.py \
        --baseline runs/paper02/q3_28ghz_long_release_5x \
        --two-pair runs/paper02/q3_28ghz_long_release_5x_two_pair \
        --q4       runs/paper02/q4_fermi_bounce_10ghz_20x \
        --outdir   runs/paper02/_cross_run_reports/q4_fermi_bounce

Safe to run before q4 finishes — missing run directories are detected and
reported. You can dry-run it now against just the two q3 runs to verify
the script produces sensible output before pointing it at q4.

Outputs:
    <outdir>/q3_vs_q4_comparison.txt — headline table + verdict
    <outdir>/q3_vs_q4_panels.png     — three-panel diagnostic figure
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Matplotlib used only for the figure; import lazily so the table works
# even without matplotlib installed.
_HAVE_MPL = True
try:
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt
except ImportError:
    _HAVE_MPL = False


# ── Project layout defaults ──────────────────────────────────────────────────
DEFAULT_PROJECT_ROOT = Path.home() / 'LaserFusionResearch' / 'research' / 'laser-plasma-research'
DEFAULT_BASELINE_DIR = 'runs/paper02/q3_28ghz_long_release_5x'
DEFAULT_TWO_PAIR_DIR = 'runs/paper02/q3_28ghz_long_release_5x_two_pair'
DEFAULT_Q4_DIR       = 'runs/paper02/q4_fermi_bounce_10ghz_20x'
DEFAULT_OUTDIR       = 'runs/paper02/_cross_run_reports/q4_fermi_bounce'

# ── Verdict thresholds ───────────────────────────────────────────────────────
# Confirmation requires ANY of:
#   - gain_csv exceeds gain_maxwellian by more than this fraction (non-thermal tail)
#   - cumulative N_KE_above_5000_keV grows by more than this factor end vs start
#   - sustained-reconnection time exceeds this fraction of run duration
GAIN_DIVERGENCE_CONFIRM = 0.10       # 10% divergence
TAIL_GROWTH_CONFIRM     = 3.0        # 3x growth in 5 MeV count
SUSTAINED_FRAC_CONFIRM  = 0.30       # 30% of run

# Falsification requires ALL of:
#   - gain estimators agree within this fraction
#   - N_KE_above_5000_keV trajectory stays within this fraction of q3 baseline
GAIN_AGREEMENT_FALSIFY  = 0.02       # 2%
TAIL_DEVIATION_FALSIFY  = 0.30       # 30% of baseline


# ── Data containers ─────────────────────────────────────────────────────────
@dataclass
class RunData:
    """Holds extracted metrics from a single run directory."""
    label: str
    path: Path
    available: bool = False
    error: str = ''

    # From phase_analysis_report.txt
    gain_maxwellian: Optional[float] = None
    gain_optimistic: Optional[float] = None
    gain_csv:        Optional[float] = None
    fusion_energy_j: Optional[float] = None
    laser_energy_j:  Optional[float] = None
    final_b_max_t:   Optional[float] = None
    natural_freq_ghz: Optional[float] = None
    n_e95_peaks:     Optional[int] = None
    time_core_outer_gt_1_ps: Optional[float] = None
    max_core_spot_ratio: Optional[float] = None

    # Phase decomposition counts
    n_sustained_phases: int = 0
    n_equilibration_phases: int = 0

    # Time series from fusion_diagnostics.csv
    t_ps_diag:           list[float] = field(default_factory=list)
    n_above_5000_kev:    list[int]   = field(default_factory=list)
    n_above_1000_kev:    list[int]   = field(default_factory=list)
    mean_ke_kev:         list[float] = field(default_factory=list)

    # Time series from fusion_rate_power_by_iter.csv
    t_ps_iter:           list[float] = field(default_factory=list)
    cumulative_energy_j: list[float] = field(default_factory=list)
    centre_outer_ratio:  list[float] = field(default_factory=list)

    # Run duration
    run_duration_ps: Optional[float] = None


def _safe_float(s: str) -> Optional[float]:
    """Parse a float, returning None on failure."""
    try:
        return float(s.replace(',', '').strip())
    except (ValueError, AttributeError):
        return None


def _safe_int_from_phase_count(s: str) -> Optional[int]:
    """Parse an integer from N_core text like '20,324,693,374,431,109,120'."""
    try:
        return int(s.replace(',', '').strip())
    except (ValueError, AttributeError):
        return None


def parse_phase_report(path: Path, run: RunData) -> None:
    """Extract gain, B-field, frequency, and phase counts from phase_analysis_report.txt."""
    if not path.exists():
        return

    in_phase_block = False
    in_yield_table = False
    in_recon_signature = False
    in_b_evolution = False
    in_oscillation = False

    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            stripped = line.strip()

            # Section markers
            if 'CUMULATIVE FUSION YIELD' in stripped:
                in_phase_block = False
                in_yield_table = True
                continue
            if 'RECONNECTION SIGNATURE' in stripped:
                in_yield_table = False
                in_recon_signature = True
                continue
            if 'B-FIELD EVOLUTION' in stripped:
                in_recon_signature = False
                in_b_evolution = True
                continue
            if 'CORE E95 OSCILLATION ANALYSIS' in stripped:
                in_b_evolution = False
                in_oscillation = True
                continue
            if 'ZONE E95 TIMESERIES' in stripped or 'END PHASE-RESOLVED' in stripped:
                in_oscillation = False
                continue
            if 'DETECTED ACCELERATION PHASES' in stripped:
                in_phase_block = True
                continue

            # Phase count from headers like "Phase 4: SUSTAINED RECONNECTION"
            if in_phase_block and 'Phase' in line and ':' in line:
                if 'SUSTAINED RECONNECTION' in line:
                    run.n_sustained_phases += 1
                elif 'EQUILIBRATION' in line:
                    run.n_equilibration_phases += 1

            # Run duration from "Sim duration: 2694.79 ps"
            if 'Sim duration:' in stripped:
                parts = stripped.split(':', 1)[1].strip().split()
                if parts:
                    run.run_duration_ps = _safe_float(parts[0])

            # Gain table — three rows after the header
            if in_yield_table:
                if stripped.startswith('Maxwellian'):
                    parts = stripped.split()
                    if len(parts) >= 4:
                        run.fusion_energy_j = _safe_float(parts[1])
                        run.gain_maxwellian = _safe_float(parts[3])
                elif stripped.startswith('Optimistic'):
                    parts = stripped.split()
                    if len(parts) >= 4:
                        run.gain_optimistic = _safe_float(parts[3])
                elif stripped.startswith('CSV-matched'):
                    parts = stripped.split()
                    if len(parts) >= 4:
                        run.gain_csv = _safe_float(parts[3])
                elif 'Laser energy:' in stripped:
                    parts = stripped.split(':', 1)[1].strip().split()
                    if parts:
                        run.laser_energy_j = _safe_float(parts[0])

            # Reconnection signature
            if in_recon_signature:
                if 'Time core/spot > 1.0:' in stripped:
                    parts = stripped.split(':', 1)[1].strip().split()
                    if parts:
                        run.time_core_outer_gt_1_ps = _safe_float(parts[0])
                elif 'Maximum ratio:' in stripped:
                    parts = stripped.split(':', 1)[1].strip().split()
                    if parts:
                        run.max_core_spot_ratio = _safe_float(parts[0])

            # B-field evolution
            if in_b_evolution:
                if 'Final |B|_max:' in stripped:
                    parts = stripped.split(':', 1)[1].strip().split()
                    if parts:
                        run.final_b_max_t = _safe_float(parts[0])

            # Oscillation analysis
            if in_oscillation:
                if 'Number of E95 peaks:' in stripped:
                    parts = stripped.split(':', 1)[1].strip().split()
                    if parts:
                        run.n_e95_peaks = int(_safe_float(parts[0]) or 0) or None
                elif 'Natural freq estimate:' in stripped:
                    parts = stripped.split(':', 1)[1].strip().split()
                    if parts:
                        run.natural_freq_ghz = _safe_float(parts[0])


def parse_fusion_diagnostics(path: Path, run: RunData) -> None:
    """Extract time series from fusion_diagnostics.csv."""
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t_ps = _safe_float(row.get('t_ps', ''))
            if t_ps is None:
                continue
            run.t_ps_diag.append(t_ps)
            try:
                run.n_above_5000_kev.append(int(row['N_KE_above_5000_keV']))
            except (KeyError, ValueError):
                run.n_above_5000_kev.append(0)
            try:
                run.n_above_1000_kev.append(int(row['N_KE_above_1000_keV']))
            except (KeyError, ValueError):
                run.n_above_1000_kev.append(0)
            mke = _safe_float(row.get('mean_KE_keV', ''))
            run.mean_ke_kev.append(mke if mke is not None else 0.0)


def parse_fusion_rate_power(path: Path, run: RunData) -> None:
    """Extract time series from fusion_rate_power_by_iter.csv."""
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t_ps = _safe_float(row.get('time_ps', ''))
            if t_ps is None:
                continue
            run.t_ps_iter.append(t_ps)
            ce = _safe_float(row.get('cumulative_fusion_energy_j', ''))
            run.cumulative_energy_j.append(ce if ce is not None else 0.0)
            co = _safe_float(row.get('centre_outer_ratio', ''))
            run.centre_outer_ratio.append(co if co is not None else 0.0)


def load_run(label: str, run_dir: Path) -> RunData:
    """Load all relevant data for a single run."""
    run = RunData(label=label, path=run_dir)
    if not run_dir.is_dir():
        run.error = f'directory does not exist: {run_dir}'
        return run

    phase_path  = run_dir / 'phase_analysis_report.txt'
    diag_path   = run_dir / 'fusion_diagnostics.csv'
    iter_path   = run_dir / 'fusion_rate_power_by_iter.csv'

    if not phase_path.exists():
        run.error = f'missing: {phase_path.name}'
        return run

    parse_phase_report(phase_path, run)
    parse_fusion_diagnostics(diag_path, run)
    parse_fusion_rate_power(iter_path, run)
    run.available = True
    return run


# ── Verdict logic ───────────────────────────────────────────────────────────
def evaluate_fermi_bounce(q4: RunData, baseline: RunData) -> tuple[str, list[str]]:
    """
    Returns ('CONFIRMED' | 'FALSIFIED' | 'INCONCLUSIVE', [reasons]).
    """
    if not q4.available:
        return ('PENDING', [f'q4 run not yet available ({q4.error or "no data"})'])
    if not baseline.available:
        return ('INCONCLUSIVE', ['baseline run unavailable for comparison'])

    confirm_signals = []
    falsify_signals = []

    # Signal 1: gain estimator divergence
    if q4.gain_maxwellian and q4.gain_csv:
        rel_diff = (q4.gain_csv - q4.gain_maxwellian) / q4.gain_maxwellian
        if rel_diff > GAIN_DIVERGENCE_CONFIRM:
            confirm_signals.append(
                f'Gain estimators diverge: CSV={q4.gain_csv:.3f} > '
                f'Maxwellian={q4.gain_maxwellian:.3f} by {100*rel_diff:.1f}% '
                f'(threshold {100*GAIN_DIVERGENCE_CONFIRM:.0f}%) — '
                f'non-thermal tail present'
            )
        elif abs(rel_diff) < GAIN_AGREEMENT_FALSIFY:
            falsify_signals.append(
                f'Gain estimators agree: |CSV-Maxwellian|={100*abs(rel_diff):.1f}% '
                f'(< {100*GAIN_AGREEMENT_FALSIFY:.0f}% threshold) — '
                f'tail is purely thermal'
            )

    # Signal 2: tail growth
    if len(q4.n_above_5000_kev) >= 5:
        early = sum(q4.n_above_5000_kev[:3]) / 3.0
        late  = sum(q4.n_above_5000_kev[-3:]) / 3.0
        if early > 0:
            growth = late / early
            if growth > TAIL_GROWTH_CONFIRM:
                confirm_signals.append(
                    f'5 MeV count grew by {growth:.1f}x across run '
                    f'(threshold {TAIL_GROWTH_CONFIRM:.1f}x) — non-thermal accumulation'
                )

    # Signal 3: sustained reconnection time
    if q4.time_core_outer_gt_1_ps and q4.run_duration_ps:
        sustained_frac = q4.time_core_outer_gt_1_ps / q4.run_duration_ps
        baseline_frac = (baseline.time_core_outer_gt_1_ps / baseline.run_duration_ps
                         if baseline.time_core_outer_gt_1_ps and baseline.run_duration_ps
                         else 0.0)
        if sustained_frac > SUSTAINED_FRAC_CONFIRM and sustained_frac > 1.5 * baseline_frac:
            confirm_signals.append(
                f'Sustained reconnection {100*sustained_frac:.0f}% of run '
                f'(threshold {100*SUSTAINED_FRAC_CONFIRM:.0f}%; baseline {100*baseline_frac:.0f}%) — '
                f'reconnection regime, not bulk-heating'
            )

    # Signal 4: tail trajectory deviation from baseline
    if (len(q4.n_above_5000_kev) >= 5 and len(baseline.n_above_5000_kev) >= 5
            and sum(baseline.n_above_5000_kev) > 0):
        # Compare integrated counts above 5 MeV
        q4_int = sum(q4.n_above_5000_kev)
        bl_int = sum(baseline.n_above_5000_kev)
        if bl_int > 0:
            deviation = abs(q4_int - bl_int) / bl_int
            if deviation < TAIL_DEVIATION_FALSIFY:
                falsify_signals.append(
                    f'5 MeV count integral matches baseline within {100*deviation:.0f}% '
                    f'(< {100*TAIL_DEVIATION_FALSIFY:.0f}% threshold) — tail unchanged'
                )

    # Decide verdict
    if confirm_signals:
        return ('CONFIRMED', confirm_signals + ['---'] + falsify_signals)
    elif len(falsify_signals) >= 2:
        return ('FALSIFIED', falsify_signals)
    elif falsify_signals:
        return ('LIKELY FALSIFIED', falsify_signals + ['(awaiting more signals)'])
    else:
        return ('INCONCLUSIVE', ['no strong signals in either direction'])


# ── Report rendering ────────────────────────────────────────────────────────
def fmt(value, spec: str = '.3f', missing: str = '—') -> str:
    """Format a value with a fallback for None."""
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return missing


def write_text_report(out_path: Path, runs: list[RunData],
                      verdict: str, reasons: list[str]) -> None:
    """Write the comparison table and verdict to text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    add = lines.append

    add('=' * 92)
    add('  FERMI-BOUNCE HYPOTHESIS CHECK — q3 baseline vs q3 two-pair vs q4 perturbative')
    add('=' * 92)
    add('')
    add(f'  Verdict: {verdict}')
    add('')
    for reason in reasons:
        if reason == '---':
            add('')
        else:
            add(f'    - {reason}')
    add('')
    add('=' * 92)
    add('  HEADLINE COMPARISON')
    add('=' * 92)
    add('')

    headers = ['Metric'] + [r.label for r in runs]
    rows: list[tuple[str, list[str]]] = []

    rows.append(('Run available', [
        'YES' if r.available else f'NO ({r.error})' for r in runs
    ]))
    rows.append(('Run duration (ps)', [
        fmt(r.run_duration_ps, '.0f') for r in runs
    ]))
    rows.append(('Laser energy (J)', [
        fmt(r.laser_energy_j, '.1f') for r in runs
    ]))
    rows.append(('Fusion energy (J)', [
        fmt(r.fusion_energy_j, '.3f') for r in runs
    ]))
    rows.append(('Gain — Maxwellian', [
        fmt(r.gain_maxwellian, '.3f') for r in runs
    ]))
    rows.append(('Gain — Optimistic', [
        fmt(r.gain_optimistic, '.3f') for r in runs
    ]))
    rows.append(('Gain — CSV-matched', [
        fmt(r.gain_csv, '.3f') for r in runs
    ]))
    rows.append(('Gain divergence (CSV vs Max)', [
        fmt(100 * (r.gain_csv - r.gain_maxwellian) / r.gain_maxwellian, '.1f') + '%'
        if (r.gain_csv and r.gain_maxwellian) else '—'
        for r in runs
    ]))
    rows.append(('Final |B|_max (T)', [
        fmt(r.final_b_max_t, '.0f') for r in runs
    ]))
    rows.append(('Natural freq (GHz)', [
        fmt(r.natural_freq_ghz, '.2f') for r in runs
    ]))
    rows.append(('# E95 peaks detected', [
        fmt(r.n_e95_peaks, 'd') for r in runs
    ]))
    rows.append(('# SUSTAINED RECONNECTION phases', [
        fmt(r.n_sustained_phases, 'd') for r in runs
    ]))
    rows.append(('# EQUILIBRATION phases', [
        fmt(r.n_equilibration_phases, 'd') for r in runs
    ]))
    rows.append(('Time core/spot > 1.0 (ps)', [
        fmt(r.time_core_outer_gt_1_ps, '.0f') for r in runs
    ]))
    rows.append(('Max core/spot ratio', [
        fmt(r.max_core_spot_ratio, '.3f') for r in runs
    ]))

    # Tail population stats
    rows.append(('5 MeV count (early avg)', [
        fmt(sum(r.n_above_5000_kev[:3])/3 if len(r.n_above_5000_kev) >= 3 else None, '.1f')
        for r in runs
    ]))
    rows.append(('5 MeV count (late avg)', [
        fmt(sum(r.n_above_5000_kev[-3:])/3 if len(r.n_above_5000_kev) >= 3 else None, '.1f')
        for r in runs
    ]))
    rows.append(('5 MeV growth ratio (late/early)', [
        fmt((sum(r.n_above_5000_kev[-3:])/3) / (sum(r.n_above_5000_kev[:3])/3 + 1e-9)
            if len(r.n_above_5000_kev) >= 3 else None, '.2f')
        + 'x' if len(r.n_above_5000_kev) >= 3 else '—'
        for r in runs
    ]))

    # Compute column widths
    col_widths = [max(len(headers[0]), max(len(row[0]) for row in rows))]
    for i in range(len(runs)):
        col_widths.append(max(len(headers[i+1]), max(len(row[1][i]) for row in rows)))

    def line_fmt(cells: list[str]) -> str:
        return '  ' + '  |  '.join(c.ljust(w) for c, w in zip(cells, col_widths))

    add(line_fmt(headers))
    add('  ' + '-+-'.join('-' * w for w in col_widths))
    for label, cells in rows:
        add(line_fmt([label] + cells))

    add('')
    add('=' * 92)
    add('  INTERPRETATION GUIDE')
    add('=' * 92)
    add('')
    add('  Fermi-bounce hypothesis: perturbative-mode polarity reversal arriving during')
    add('  reconnection exhaust produces non-thermal tail acceleration that bulk heating')
    add('  cannot access.')
    add('')
    add('  Confirmation signatures (need ANY):')
    add(f'    - Gain estimator divergence > {100*GAIN_DIVERGENCE_CONFIRM:.0f}% '
        f'(CSV-matched > Maxwellian)')
    add(f'    - 5 MeV count grows > {TAIL_GROWTH_CONFIRM:.1f}x end vs start')
    add(f'    - Sustained-reconnection time > {100*SUSTAINED_FRAC_CONFIRM:.0f}% of run duration')
    add('')
    add('  Falsification signatures (need ALL):')
    add(f'    - Gain estimators agree within {100*GAIN_AGREEMENT_FALSIFY:.0f}%')
    add(f'    - 5 MeV count integral matches baseline within '
        f'{100*TAIL_DEVIATION_FALSIFY:.0f}%')
    add('')
    add('=' * 92)

    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_panels(out_path: Path, runs: list[RunData]) -> None:
    """Render three-panel diagnostic figure."""
    if not _HAVE_MPL:
        return

    available = [r for r in runs if r.available]
    if not available:
        return

    colors = {'q3 baseline': 'tab:blue',
              'q3 two-pair': 'tab:orange',
              'q4 perturbative': 'tab:red'}

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)

    # Panel 1: Cumulative fusion energy
    ax = axes[0]
    for r in available:
        if r.t_ps_iter and r.cumulative_energy_j:
            ax.plot(r.t_ps_iter, r.cumulative_energy_j,
                    label=r.label, color=colors.get(r.label, 'gray'), linewidth=2)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Cumulative fusion energy (J)')
    ax.set_title('Cumulative fusion energy — bulk-heating runs grow linearly; '
                 'Fermi-bounce shows super-linear growth or higher saturation')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: 5 MeV non-thermal tail
    ax = axes[1]
    for r in available:
        if r.t_ps_diag and r.n_above_5000_kev:
            ax.plot(r.t_ps_diag, r.n_above_5000_kev,
                    label=r.label, color=colors.get(r.label, 'gray'),
                    linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Particle count above 5 MeV')
    ax.set_yscale('log')
    ax.set_title('Non-thermal tail — Fermi-bounce signature is cycle-on-cycle '
                 'accumulation (climbing trace)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: Centre/outer ratio
    ax = axes[2]
    for r in available:
        if r.t_ps_iter and r.centre_outer_ratio:
            # Limit y range; ratio occasionally spikes near 0 in early steps
            ratios = [min(max(c, 0.0), 3.0) for c in r.centre_outer_ratio]
            ax.plot(r.t_ps_iter, ratios,
                    label=r.label, color=colors.get(r.label, 'gray'),
                    linewidth=1.5, alpha=0.8)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='ratio = 1')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Centre / outer E95 ratio')
    ax.set_title('Centre/outer ratio — sustained > 1 across cycles is the '
                 'reconnection-acceleration signature')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Compare q3 baseline / q3 two-pair / q4 Fermi-bounce runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--baseline', type=Path, default=None,
                        help=f'q3 baseline run dir (default: {DEFAULT_BASELINE_DIR})')
    parser.add_argument('--two-pair', type=Path, default=None,
                        help=f'q3 two-pair run dir (default: {DEFAULT_TWO_PAIR_DIR})')
    parser.add_argument('--q4', type=Path, default=None,
                        help=f'q4 Fermi-bounce run dir (default: {DEFAULT_Q4_DIR})')
    parser.add_argument('--outdir', type=Path, default=None,
                        help=f'output directory (default: {DEFAULT_OUTDIR})')
    parser.add_argument('--project-root', type=Path, default=None,
                        help='project root (for resolving relative defaults)')
    args = parser.parse_args()

    # Resolve paths
    project_root = args.project_root or DEFAULT_PROJECT_ROOT
    if not project_root.is_dir():
        # Fall back to cwd if the default isn't right for this machine
        project_root = Path.cwd()

    def _resolve(arg_val, default_rel):
        if arg_val is None:
            return project_root / default_rel
        return arg_val if arg_val.is_absolute() else project_root / arg_val

    baseline_dir = _resolve(args.baseline, DEFAULT_BASELINE_DIR)
    two_pair_dir = _resolve(args.two_pair, DEFAULT_TWO_PAIR_DIR)
    q4_dir       = _resolve(args.q4,       DEFAULT_Q4_DIR)
    outdir       = _resolve(args.outdir,   DEFAULT_OUTDIR)

    print(f'Project root: {project_root}')
    print(f'Baseline:     {baseline_dir}  ({"OK" if baseline_dir.is_dir() else "MISSING"})')
    print(f'Two-pair:     {two_pair_dir}  ({"OK" if two_pair_dir.is_dir() else "MISSING"})')
    print(f'Q4:           {q4_dir}  ({"OK" if q4_dir.is_dir() else "MISSING"})')
    print(f'Output:       {outdir}')
    print()

    # Load
    runs = [
        load_run('q3 baseline', baseline_dir),
        load_run('q3 two-pair', two_pair_dir),
        load_run('q4 perturbative', q4_dir),
    ]

    # Verdict
    q4_run = runs[2]
    baseline_run = runs[0]
    verdict, reasons = evaluate_fermi_bounce(q4_run, baseline_run)

    # Report
    txt_path = outdir / 'q3_vs_q4_comparison.txt'
    write_text_report(txt_path, runs, verdict, reasons)
    print(f'Wrote: {txt_path}')

    if _HAVE_MPL:
        png_path = outdir / 'q3_vs_q4_panels.png'
        write_panels(png_path, runs)
        print(f'Wrote: {png_path}')
    else:
        print('(matplotlib not available — skipping figure)')

    # Echo key numbers to stdout
    print()
    print('=' * 60)
    print(f'  VERDICT: {verdict}')
    print('=' * 60)
    for reason in reasons:
        if reason != '---':
            print(f'    - {reason}')
    print()
    for r in runs:
        if r.available:
            print(f'  {r.label:20s}  gain_max={fmt(r.gain_maxwellian, ".3f")}  '
                  f'gain_csv={fmt(r.gain_csv, ".3f")}  '
                  f'E_fusion={fmt(r.fusion_energy_j, ".2f")} J  '
                  f'B_max={fmt(r.final_b_max_t, ".0f")} T')
        else:
            print(f'  {r.label:20s}  {r.error}')


if __name__ == '__main__':
    main()
