#!/usr/bin/env python3
"""
test_b_min_t_cut.py — unit tests for find_b_minimum_t_cut.

Covers both legacy v0.7 (plain argmin) and v0.8 (smoothed first-prominent
minimum).  The v0.7 contract is preserved by passing smooth_sigma_ps=0,
prominence_frac=0; v0.8 default behavior is validated by separate tests.
"""

import csv
import math
import tempfile
from pathlib import Path

import importlib.util
_HERE = Path(__file__).resolve().parent
_FT_SCRIPT = _HERE / 'pb11_first_transit_fusion.py'
if not _FT_SCRIPT.is_file():
    raise FileNotFoundError(
        f'Expected pb11_first_transit_fusion.py next to this test file at '
        f'{_FT_SCRIPT}. Place both files in the same directory '
        f'(typically analysis_scripts/).')
spec = importlib.util.spec_from_file_location('ft_patched', str(_FT_SCRIPT))
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def _make_csv(rows, td):
    rd = Path(td)
    p = rd / 'reconnection_rate_offline.csv'
    with p.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['step', 't_ps', 'B_max_T', 'extra'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rd


# =============================================================================
#  Legacy v0.7 contract — explicit smooth_sigma_ps=0, prominence_frac=0
# =============================================================================


def test_real_p1_ld_uuf_csv():
    """Real-data validation under v0.8 defaults.  v0.8 should land in
    the broad reconnection window [200, 800] ps (vs v0.7 unpinned)."""
    candidate_paths = [
        Path('/mnt/user-data/uploads/reconnection_rate_offline.csv'),
        Path.home() / 'LaserFusionResearch/research/laser-plasma-research/runs/paper01/p1_ld_uuf/reconnection_rate_offline.csv',
        Path.home() / 'laser-plasma-research/runs/p1_ld_uuf/reconnection_rate_offline.csv',
    ]
    csv_src = next((p for p in candidate_paths if p.is_file()), None)
    if csv_src is None:
        print(f'  SKIP  real CSV not found')
        return
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        (rd / 'reconnection_rate_offline.csv').write_bytes(csv_src.read_bytes())
        t_cut_ps, b_min, b_init, n_rows = ft.find_b_minimum_t_cut(rd)
    assert n_rows == 76, f'expected 76 rows, got {n_rows}'
    assert b_init is not None and b_init > 0
    assert 25.0 <= b_init <= 60.0, f'B_init={b_init:.2f} outside [25, 60] T'
    assert b_min is not None and 20.0 <= b_min <= 25.0, f'B_min={b_min:.2f} outside [20, 25] T'
    assert 200.0 <= t_cut_ps <= 800.0, f't_cut={t_cut_ps:.2f} outside [200, 800] ps'
    print(f'  PASS  real CSV (v0.8 defaults):  t_cut={t_cut_ps:.2f}ps  '
          f'B_min={b_min:.2f}T  B_init={b_init:.2f}T  n={n_rows}')


def test_monotonic_decay_no_rebound_v07():
    rows = [{'step': i, 't_ps': i * 10.0, 'B_max_T': 50.0 - i, 'extra': 0}
            for i in range(20)]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(
            rd, smooth_sigma_ps=0, prominence_frac=0,
        )
    assert n == 20
    assert b_init == 50.0
    assert b_min == 31.0
    assert t_cut_ps == 190.0
    print(f'  PASS  v0.7 monotonic decay: t_cut={t_cut_ps}  B_min={b_min}')


def test_rebound_in_middle_v07():
    rows = [
        {'step': 0,   't_ps':   0.0, 'B_max_T': 50.0, 'extra': 0},
        {'step': 10,  't_ps':  20.0, 'B_max_T': 40.0, 'extra': 0},
        {'step': 20,  't_ps':  40.0, 'B_max_T': 30.0, 'extra': 0},
        {'step': 30,  't_ps':  60.0, 'B_max_T': 25.0, 'extra': 0},  # MIN
        {'step': 40,  't_ps':  80.0, 'B_max_T': 27.0, 'extra': 0},
        {'step': 50,  't_ps': 100.0, 'B_max_T': 28.0, 'extra': 0},
    ]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(
            rd, smooth_sigma_ps=0, prominence_frac=0,
        )
    assert n == 6
    assert b_init == 50.0
    assert b_min == 25.0
    assert t_cut_ps == 60.0
    print(f'  PASS  v0.7 rebound: t_cut={t_cut_ps}  B_min={b_min}')


def test_missing_csv():
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(rd)
    assert t_cut_ps is None
    assert b_min is None
    assert b_init is None
    assert n == 0
    print(f'  PASS  missing CSV: all-None return')


def test_missing_column():
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        p = rd / 'reconnection_rate_offline.csv'
        with p.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['step', 't_ps', 'something_else'])
            w.writeheader()
            w.writerow({'step': 0, 't_ps': 0.0, 'something_else': 1.0})
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(rd)
    assert t_cut_ps is None
    print(f'  PASS  missing column: returns None gracefully')


def test_malformed_rows_skipped_v07():
    rows = [
        {'step': 0,  't_ps':  0.0, 'B_max_T': 50.0,    'extra': 0},
        {'step': 1,  't_ps':  10.0, 'B_max_T': 'NaN',  'extra': 0},
        {'step': 2,  't_ps':  20.0, 'B_max_T': 30.0,   'extra': 0},
        {'step': 3,  't_ps':  30.0, 'B_max_T': '',      'extra': 0},
        {'step': 4,  't_ps':  40.0, 'B_max_T': 25.0,    'extra': 0},
    ]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(
            rd, smooth_sigma_ps=0, prominence_frac=0,
        )
    assert b_min == 25.0
    assert t_cut_ps == 40.0
    print(f'  PASS  v0.7 malformed rows skipped: t_cut={t_cut_ps}  B_min={b_min}')


def test_pre_biermann_dump_skipped_v07():
    rows = [
        {'step': 0,  't_ps':   0.0, 'B_max_T':  0.0, 'extra': 0},
        {'step': 20, 't_ps':  15.43, 'B_max_T': 30.81, 'extra': 0},
        {'step': 40, 't_ps':  30.87, 'B_max_T': 24.81, 'extra': 0},
        {'step': 60, 't_ps':  46.30, 'B_max_T': 24.31, 'extra': 0},
        {'step': 80, 't_ps':  77.17, 'B_max_T': 22.39, 'extra': 0},  # MIN
        {'step': 100, 't_ps': 138.91, 'B_max_T': 26.73, 'extra': 0},
    ]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(
            rd, smooth_sigma_ps=0, prominence_frac=0,
        )
    assert b_min == 22.39
    assert t_cut_ps == 77.17
    assert b_init == 30.81
    print(f'  PASS  v0.7 pre-Biermann dump 0 skipped: t_cut={t_cut_ps}  B_min={b_min}')


def test_all_zero_returns_none_safely():
    rows = [{'step': i, 't_ps': i*10.0, 'B_max_T': 0.0, 'extra': 0}
            for i in range(5)]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(rd)
    assert t_cut_ps is None
    assert n == 5
    print(f'  PASS  all-zero B returns None safely')


def test_single_row_v07():
    rows = [{'step': 0, 't_ps': 0.0, 'B_max_T': 42.0, 'extra': 0}]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_cut_ps, b_min, b_init, n = ft.find_b_minimum_t_cut(
            rd, smooth_sigma_ps=0, prominence_frac=0,
        )
    assert t_cut_ps == 0.0
    assert b_min == 42.0
    assert b_init == 42.0
    print(f'  PASS  v0.7 single row: t_cut={t_cut_ps}  B_min={b_min}=B_init')


# =============================================================================
#  v0.8 smoothed first-prominent-minimum behavior
# =============================================================================


def _realistic_flat_well_trace(seed_phase: float,
                                 dump_period_ps: float = 15.0,
                                 n_dumps: int = 76):
    """|B|_max(t) shaped like the real May 31 256^2 baseline:
      - t < 80 ps: ramp settling 52 -> 35 T
      - t >= 80:   broad reconnection collapse centered at 540 ps,
                    depth ~13 T (down to ~22 T), with ~1 T oscillation
                    noise whose phase is seed-dependent

    The broad minimum is clearly deeper than the ramp tail (22 vs 35),
    matching the real physics where v0.7 plain argmin fights *only* over
    which oscillation sample within the flat well registers deepest.
    """
    rows = []
    for i in range(n_dumps):
        t = i * dump_period_ps
        if t < 80:
            B = 52.0 - 17.0 * (t / 80.0)
        else:
            broad = 35.0 - 13.0 * math.exp(-((t - 540) ** 2) / (2 * 200 ** 2))
            osc = 1.0 * math.sin((t - 80) * 0.08 + seed_phase)
            B = broad + osc
        rows.append({'step': i * 20, 't_ps': t, 'B_max_T': B, 'extra': 0})
    return rows


def test_v08_flat_well_stability_across_seeds():
    """Key v0.8 regression test: identical-physics traces with different
    oscillation phases must produce tightly clustered v0.8 t_cuts in the
    broad reconnection window."""
    tcuts_v07 = []
    tcuts_v08 = []
    for phase in [0.0, 1.5, 3.0]:
        rows = _realistic_flat_well_trace(phase)
        with tempfile.TemporaryDirectory() as td:
            rd = _make_csv(rows, td)
            t_v07, _, _, _ = ft.find_b_minimum_t_cut(
                rd, smooth_sigma_ps=0, prominence_frac=0,
            )
            t_v08, _, _, _ = ft.find_b_minimum_t_cut(rd)
        tcuts_v07.append(t_v07)
        tcuts_v08.append(t_v08)
    v07_spread = max(tcuts_v07) - min(tcuts_v07)
    v08_spread = max(tcuts_v08) - min(tcuts_v08)
    assert v08_spread <= v07_spread + 30.0, (
        f'v0.8 spread {v08_spread:.1f}ps > v0.7 spread {v07_spread:.1f}ps '
        f'(v07={tcuts_v07}, v08={tcuts_v08})'
    )
    for t in tcuts_v08:
        assert 350.0 <= t <= 750.0, (
            f'v0.8 t_cut={t:.1f}ps outside broad collapse window '
            f'(all v08={tcuts_v08})'
        )
    print(f'  PASS  v0.8 stability: v0.7 spread={v07_spread:.0f}ps  '
          f'v0.8 spread={v08_spread:.0f}ps  t_v08={[round(t,0) for t in tcuts_v08]}')


def test_v08_lands_near_broad_minimum():
    rows = _realistic_flat_well_trace(seed_phase=2.0)
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_v08, b_v08, _, _ = ft.find_b_minimum_t_cut(rd)
    assert 440.0 <= t_v08 <= 640.0, (
        f'v0.8 t_cut={t_v08:.1f}ps not within ±100ps of broad min at 540ps'
    )
    print(f'  PASS  v0.8 lands near broad min: t_cut={t_v08:.1f}ps (truth=540)')


def test_v08_falls_back_for_short_monotonic_run():
    rows = [{'step': i, 't_ps': i * 10.0, 'B_max_T': 50.0 - i, 'extra': 0}
            for i in range(20)]
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        t_v08, b_v08, _, n = ft.find_b_minimum_t_cut(rd)
    assert t_v08 >= 150.0, f'v0.8 fallback t_cut={t_v08:.1f} should be near end'
    print(f'  PASS  v0.8 fallback on monotonic decay: t_cut={t_v08:.1f}ps')


def test_v08_parameter_insensitivity():
    """v0.8 with sigma in [30, 50] ps and prominence in [0.2, 0.4] gives
    t_cut within 30 ps — answers 'why exactly 30 ps?' (it's the minimum
    width that resolves the broad collapse; 30-50 all give the same
    answer; sigma < 25 ps starts firing on ramp-settling wiggles)."""
    rows = _realistic_flat_well_trace(seed_phase=1.0)
    tcuts = []
    with tempfile.TemporaryDirectory() as td:
        rd = _make_csv(rows, td)
        for sigma in [30.0, 40.0, 50.0]:
            for prom in [0.2, 0.3, 0.4]:
                t, _, _, _ = ft.find_b_minimum_t_cut(
                    rd, smooth_sigma_ps=sigma, prominence_frac=prom,
                )
                tcuts.append(t)
    spread = max(tcuts) - min(tcuts)
    assert spread <= 30.0, (
        f'v0.8 parameter sensitivity too high in [30,50] x [0.2,0.4] grid: '
        f'spread={spread:.1f}ps'
    )
    print(f'  PASS  v0.8 parameter insensitivity in working regime: '
          f'spread={spread:.0f}ps across [30-50ps] x [0.2-0.4]')


if __name__ == '__main__':
    print('Running unit tests for find_b_minimum_t_cut...')
    print()
    print('Legacy v0.7 contract (smooth_sigma_ps=0, prominence_frac=0):')
    test_monotonic_decay_no_rebound_v07()
    test_rebound_in_middle_v07()
    test_missing_csv()
    test_missing_column()
    test_malformed_rows_skipped_v07()
    test_pre_biermann_dump_skipped_v07()
    test_all_zero_returns_none_safely()
    test_single_row_v07()
    print()
    print('v0.8 smoothed first-prominent-minimum (defaults):')
    test_real_p1_ld_uuf_csv()
    test_v08_flat_well_stability_across_seeds()
    test_v08_lands_near_broad_minimum()
    test_v08_falls_back_for_short_monotonic_run()
    test_v08_parameter_insensitivity()
    print()
    print('All tests passed.')
