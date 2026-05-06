#!/usr/bin/env python3
"""
walk_loose.py — query results across an entire program of runs without
decompressing any tarballs.

After ./archive_runs.sh produces:

    runs/cloud_run_001/archives/
        loose/p01/p1_static_p11b/{run_meta.txt, fusion_rate_power_by_iter.csv, ...}
        loose/p02/p2_rotating_500MHz/{...}
        loose/p02/p2_freq_208MHz/{...}
        ...

this module gives you:

    - RunIndex          : enumerates every run, parses metadata
    - RunIndex.summary(): one-line summary per run
    - RunIndex.dataframe(): pandas DataFrame across all runs of a given CSV
    - RunIndex.collect_metric(): grab one column from every run, concatenate
    - CLI: `python walk_loose.py [archives_dir] [--paper p2] [--csv ...]`

Designed to be the fast-iteration tool for cross-run plots and tables. When
you need the openPMD diagnostic data inside a specific run, decompress that
one run with extract_run.sh -- this tool intentionally does not touch the
.tar.zst files.

Examples:

    # one-line summary of every run
    python walk_loose.py runs/cloud_run_001/archives

    # only Paper 2 runs
    python walk_loose.py runs/cloud_run_001/archives --paper p2

    # show what CSVs are available inside the loose archive
    python walk_loose.py runs/cloud_run_001/archives --list-csvs

    # peak fusion rate of every Paper 2 run, sorted by frequency
    python walk_loose.py runs/cloud_run_001/archives \\
        --paper p2 --metric fusion_rate_power_by_iter.csv:fusion_rate_p11b_s^-1:max

    # programmatic use:
    from walk_loose import RunIndex
    idx = RunIndex('runs/cloud_run_001/archives')
    for run in idx.runs:
        df = run.read_csv('centre_outer.csv')
        if df is not None:
            print(f'{run.tag}: peak ratio = {df["ratio"].max():.2f}')
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# pandas is a soft dependency — we import lazily so the index works without it.
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False


# ============================================================================
# Run record
# ============================================================================

@dataclass
class Run:
    """A single simulation run as represented by its loose archive directory."""
    tag: str                                # e.g. "p2_freq_208MHz"
    paper: str                              # e.g. "p02" (parent dir under loose/)
    path: Path                              # absolute path to the loose dir
    meta: dict = field(default_factory=dict)   # parsed run_meta.txt key/values
    accounting: dict = field(default_factory=dict)  # parsed fusion_accounting_notes.txt

    # ── File access ──────────────────────────────────────────────────────────

    def has(self, filename: str) -> bool:
        """True iff filename exists in the loose copy of this run."""
        return (self.path / filename).is_file()

    def files(self) -> list[str]:
        """List of filenames (not paths) available in the loose archive."""
        return sorted(p.name for p in self.path.iterdir() if p.is_file())

    def read_csv(self, filename: str, **read_csv_kwargs):
        """Read a CSV by filename; returns a pandas DataFrame or None.

        Returns None if pandas is unavailable, the file is missing, or the
        file is empty/malformed. Does not raise — designed for batch use
        across many runs where some may be missing files.
        """
        if not _HAS_PANDAS:
            return None
        p = self.path / filename
        if not p.is_file() or p.stat().st_size == 0:
            return None
        try:
            return pd.read_csv(p, **read_csv_kwargs)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return None

    def read_csv_dicts(self, filename: str) -> list[dict]:
        """Pandas-free fallback: read CSV as list of dicts."""
        p = self.path / filename
        if not p.is_file():
            return []
        with open(p, newline='') as f:
            return list(csv.DictReader(f))

    def read_text(self, filename: str) -> Optional[str]:
        """Return file contents as text, or None if absent."""
        p = self.path / filename
        return p.read_text() if p.is_file() else None

    # ── Computed properties from meta/accounting ─────────────────────────────

    @property
    def b_seed_T(self) -> Optional[float]:
        v = self.meta.get('b_seed_t') or self.meta.get('B_SEED_T')
        return float(v) if v is not None else None

    @property
    def n_steps(self) -> Optional[int]:
        v = self.meta.get('n_steps')
        return int(float(v)) if v is not None else None

    @property
    def dt_s(self) -> Optional[float]:
        v = self.meta.get('time_step_s') or self.meta.get('dt_s')
        return float(v) if v is not None else None

    @property
    def total_time_s(self) -> Optional[float]:
        v = self.meta.get('total_time_s')
        if v is not None:
            return float(v)
        # Compute if missing
        if self.dt_s is not None and self.n_steps is not None:
            return self.dt_s * self.n_steps
        return None

    @property
    def freq_Hz(self) -> Optional[float]:
        """Extract rotation frequency from tag (e.g. 'p2_freq_208MHz' → 2.08e8)."""
        m = re.search(r'(\d+(?:\.\d+)?)MHz', self.tag, re.IGNORECASE)
        return float(m.group(1)) * 1e6 if m else None

    @property
    def is_rotating(self) -> bool:
        return 'rotating' in self.tag.lower() or 'freq' in self.tag.lower()

    @property
    def base_fuel(self) -> Optional[str]:
        return self.meta.get('base_fuel')

    def __repr__(self):
        return f'Run({self.tag}, paper={self.paper})'


# ============================================================================
# Metadata parsing
# ============================================================================

# run_meta.txt is a flat key=value file with section headers in [brackets].
# We flatten everything into a single dict; section headers are stored under
# '_section_<name>' keys so they're addressable but don't collide with values.
_META_KV_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$')
_META_SECTION_RE = re.compile(r'^\[([A-Za-z_][A-Za-z0-9_]*)\]\s*$')


def parse_kv_file(path: Path) -> dict:
    """Parse a key=value text file (run_meta.txt, fusion_accounting_notes.txt)."""
    if not path.is_file():
        return {}
    out = {}
    section = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = _META_SECTION_RE.match(line)
        if m:
            section = m.group(1)
            continue
        m = _META_KV_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            # Strip trailing comments
            val = val.split('#', 1)[0].strip()
            # Section-prefixed key for collision avoidance
            if section is not None:
                qualified = f'{section}.{key}'
                out[qualified] = val
            # Also store unqualified for quick lookup (last writer wins)
            out[key] = val
    return out


# ============================================================================
# Index
# ============================================================================

class RunIndex:
    """Walks a loose/ archive and indexes every run."""

    def __init__(self, archives_dir: str | Path):
        self.archives_dir = Path(archives_dir).expanduser().resolve()
        if not self.archives_dir.is_dir():
            raise FileNotFoundError(f'{self.archives_dir} does not exist')

        # Locate the loose root. Two conventions accepted:
        #   <archives_dir>/loose/<paper>/<run>/...   (output of archive_runs.sh)
        #   <archives_dir>/<paper>/<run>/...         (raw runs dir, no archiving)
        loose_root = self.archives_dir / 'loose'
        self.loose_root = loose_root if loose_root.is_dir() else self.archives_dir

        self.runs: list[Run] = self._scan()

    def _scan(self) -> list[Run]:
        runs = []
        # A "run" is any directory that contains run_meta.txt
        for meta_path in sorted(self.loose_root.rglob('run_meta.txt')):
            run_dir = meta_path.parent
            tag = run_dir.name
            # Paper is the first directory component below loose_root
            try:
                rel = run_dir.relative_to(self.loose_root)
                paper = rel.parts[0] if len(rel.parts) > 1 else 'unknown'
            except ValueError:
                paper = 'unknown'

            run = Run(
                tag=tag,
                paper=paper,
                path=run_dir,
                meta=parse_kv_file(meta_path),
                accounting=parse_kv_file(run_dir / 'fusion_accounting_notes.txt'),
            )
            runs.append(run)
        return runs

    # ── Filtering ────────────────────────────────────────────────────────────

    def filter(self, paper: Optional[str] = None,
               tag_pattern: Optional[str] = None,
               rotating: Optional[bool] = None) -> list[Run]:
        """Return runs matching the given criteria."""
        out = self.runs
        if paper is not None:
            out = [r for r in out if r.paper.lower() == paper.lower()
                   or r.paper.lower() == f'p{paper.lstrip("p").zfill(2)}']
        if tag_pattern is not None:
            rx = re.compile(tag_pattern, re.IGNORECASE)
            out = [r for r in out if rx.search(r.tag)]
        if rotating is not None:
            out = [r for r in out if r.is_rotating == rotating]
        return out

    def by_tag(self, tag: str) -> Optional[Run]:
        for r in self.runs:
            if r.tag == tag:
                return r
        return None

    def papers(self) -> list[str]:
        """List of papers that have at least one run indexed."""
        return sorted({r.paper for r in self.runs})

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    # ── Cross-run aggregates ─────────────────────────────────────────────────

    def list_csvs(self) -> dict[str, int]:
        """Return {csv_filename: number_of_runs_containing_it}."""
        counts: dict[str, int] = {}
        for r in self.runs:
            for f in r.files():
                if f.endswith('.csv'):
                    counts[f] = counts.get(f, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def dataframe(self, csv_filename: str, runs: Optional[list[Run]] = None):
        """Concatenate the same CSV across all (filtered) runs into one DF.

        Adds 'tag' and 'paper' columns identifying which run each row came from.
        Returns None if pandas not available.
        """
        if not _HAS_PANDAS:
            return None
        if runs is None:
            runs = self.runs
        frames = []
        for r in runs:
            df = r.read_csv(csv_filename)
            if df is None or df.empty:
                continue
            df = df.copy()
            df['tag'] = r.tag
            df['paper'] = r.paper
            if r.freq_Hz is not None:
                df['freq_Hz'] = r.freq_Hz
            if r.b_seed_T is not None:
                df['b_seed_T'] = r.b_seed_T
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def collect_metric(self, csv_filename: str, column: str,
                       reduce: str = 'max',
                       runs: Optional[list[Run]] = None) -> dict[str, float]:
        """For each run, reduce one column of one CSV to a single number.

        reduce: 'max' | 'min' | 'mean' | 'last' | 'first' | 'sum'
        Returns {tag: reduced_value}, with NaN for missing.
        """
        if runs is None:
            runs = self.runs
        out = {}
        for r in runs:
            df = r.read_csv(csv_filename)
            if df is None or df.empty or column not in df.columns:
                out[r.tag] = float('nan')
                continue
            col = df[column].dropna() if _HAS_PANDAS else df[column]
            if len(col) == 0:
                out[r.tag] = float('nan')
                continue
            try:
                if reduce == 'max':       out[r.tag] = float(col.max())
                elif reduce == 'min':     out[r.tag] = float(col.min())
                elif reduce == 'mean':    out[r.tag] = float(col.mean())
                elif reduce == 'sum':     out[r.tag] = float(col.sum())
                elif reduce == 'last':    out[r.tag] = float(col.iloc[-1])
                elif reduce == 'first':   out[r.tag] = float(col.iloc[0])
                else:
                    raise ValueError(f'Unknown reduce: {reduce}')
            except (ValueError, TypeError):
                out[r.tag] = float('nan')
        return out

    # ── Reporting ────────────────────────────────────────────────────────────

    def summary_table(self, runs: Optional[list[Run]] = None) -> str:
        """Human-readable table; one row per run with the most useful fields."""
        if runs is None:
            runs = self.runs
        if not runs:
            return '  (no runs)'

        rows = []
        for r in runs:
            n_steps = r.n_steps if r.n_steps is not None else '?'
            t_ns = (r.total_time_s * 1e9) if r.total_time_s else None
            t_str = f'{t_ns:.2f}ns' if t_ns is not None else '?'
            b_str = f'{r.b_seed_T:.0f}T' if r.b_seed_T is not None else '?'
            f_str = f'{r.freq_Hz/1e6:.0f}MHz' if r.freq_Hz else 'static'
            rows.append((r.paper, r.tag, b_str, f_str, n_steps, t_str,
                         r.base_fuel or '?'))

        widths = [max(len(str(row[i])) for row in rows + [
            ('paper', 'tag', 'B', 'rot', 'steps', 'duration', 'fuel')
        ]) for i in range(7)]

        hdr = ('paper', 'tag', 'B', 'rot', 'steps', 'duration', 'fuel')
        lines = []
        lines.append('  ' + '  '.join(f'{h:<{widths[i]}}' for i, h in enumerate(hdr)))
        lines.append('  ' + '  '.join('-' * widths[i] for i in range(7)))
        for row in rows:
            lines.append('  ' + '  '.join(f'{str(v):<{widths[i]}}'
                                          for i, v in enumerate(row)))
        return '\n'.join(lines)


# ============================================================================
# CLI
# ============================================================================

def _parse_metric_spec(spec: str) -> tuple[str, str, str]:
    """Parse 'file.csv:column:reduce' into a 3-tuple."""
    parts = spec.split(':')
    if len(parts) == 2:
        f, col = parts
        return f, col, 'max'
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f'Bad metric spec {spec!r}; expected file.csv:column[:reduce]')


def main():
    ap = argparse.ArgumentParser(
        description='Walk a loose/ archive of runs without decompressing tarballs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s runs/cloud_run_001/archives
  %(prog)s runs/cloud_run_001/archives --paper p2
  %(prog)s runs/cloud_run_001/archives --list-csvs
  %(prog)s runs/cloud_run_001/archives --paper p2 \\
      --metric fusion_rate_power_by_iter.csv:fusion_rate_p11b_s^-1:max
""")
    ap.add_argument('archives_dir', nargs='?', default='runs/archives',
                    help='Path to <runs>/archives/ (containing loose/) '
                         'or directly to a runs directory')
    ap.add_argument('--paper', help='Filter to one paper (e.g. p2 or p02)')
    ap.add_argument('--tag', help='Regex to match run tags')
    ap.add_argument('--rotating', action='store_true', help='Only rotating runs')
    ap.add_argument('--static', action='store_true', help='Only static runs')
    ap.add_argument('--list-csvs', action='store_true',
                    help='List CSV files present across all runs and counts')
    ap.add_argument('--metric',
                    help='file.csv:column[:reduce] -- print one number per run. '
                         'reduce defaults to max; choices: max/min/mean/sum/last/first')
    ap.add_argument('--json', action='store_true',
                    help='Output machine-readable JSON instead of text')
    ap.add_argument('--files', action='store_true',
                    help='List every file in every run (verbose)')
    args = ap.parse_args()

    try:
        idx = RunIndex(args.archives_dir)
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1

    rotating: Optional[bool] = None
    if args.rotating: rotating = True
    if args.static:   rotating = False
    runs = idx.filter(paper=args.paper, tag_pattern=args.tag, rotating=rotating)

    if not runs:
        print('  (no runs match the filter)', file=sys.stderr)
        return 1

    # ── --list-csvs mode ─────────────────────────────────────────────────────
    if args.list_csvs:
        counts = idx.list_csvs()
        if args.json:
            print(json.dumps(counts, indent=2))
        else:
            print(f'  CSV files in archive (across {len(idx)} runs):')
            for fname, cnt in counts.items():
                print(f'    {cnt:>4d} runs  {fname}')
        return 0

    # ── --files mode ─────────────────────────────────────────────────────────
    if args.files:
        for r in runs:
            print(f'\n{r.tag}  ({r.path})')
            for f in r.files():
                size = (r.path / f).stat().st_size
                print(f'    {size:>10d}  {f}')
        return 0

    # ── --metric mode ────────────────────────────────────────────────────────
    if args.metric:
        try:
            csv_file, col, reduce_op = _parse_metric_spec(args.metric)
        except ValueError as e:
            print(f'ERROR: {e}', file=sys.stderr)
            return 1
        results = idx.collect_metric(csv_file, col, reduce=reduce_op, runs=runs)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f'  {reduce_op}({col}) from {csv_file}:')
            for tag, val in results.items():
                print(f'    {tag:<40s}  {val}')
        return 0

    # ── Default: summary table ───────────────────────────────────────────────
    if args.json:
        out = []
        for r in runs:
            out.append({
                'tag': r.tag, 'paper': r.paper, 'path': str(r.path),
                'b_seed_T': r.b_seed_T, 'freq_Hz': r.freq_Hz,
                'n_steps': r.n_steps, 'total_time_s': r.total_time_s,
                'base_fuel': r.base_fuel,
                'is_rotating': r.is_rotating,
                'files': r.files(),
            })
        print(json.dumps(out, indent=2))
    else:
        print(f'  Archive: {idx.archives_dir}')
        print(f'  Loose root: {idx.loose_root}')
        print(f'  {len(runs)} run(s) matching filter (out of {len(idx)} total)')
        print()
        print(idx.summary_table(runs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
