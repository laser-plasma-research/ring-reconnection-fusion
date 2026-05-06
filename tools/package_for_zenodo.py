#!/usr/bin/env python3
"""
package_for_zenodo.py — assemble a publication-ready tarball for Zenodo upload.

Takes a paper-id (matching program_config.yaml) and produces a
self-contained tarball with:

  - Source code snapshot (git archive of tracked files)
  - Configuration: program_config.yaml, the exact CLI invocation
  - Environment: environment.yml, conda list, pip freeze, system info
  - Run metadata: run.log, run_meta.txt, invocation
  - Selected diagnostics (Standard tier: CSVs + every 10th early, every late)
  - Whitelisted deliverables (manuscript, references, lay summary, press
    release, paper.jsonld) — IP-sensitive files NOT included by default
  - Auto-generated README and CITATION.cff
  - Manifest (paper id, version, git SHA, sizes, timestamps)
  - SHA-256 checksums of every file inside

Usage:
  python3 tools/package_for_zenodo.py --paper A1 --version v0.1 [--output DIR]
                                      [--build-pdf] [--include-uniqueness]
                                      [--dry-run]

Patent-sensitive files (potential_claims.md, uniqueness_review.md) are
excluded by default. --include-uniqueness allows uniqueness_review.md only
after attorney review (e.g. post-PPA filing). potential_claims.md is never
included; copy it manually if needed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Whitelist of deliverable files (default-deny everything else)
# ---------------------------------------------------------------------------

DELIVERABLE_WHITELIST = [
    'manuscript.tex',
    'references.bib',
    'lay_summary_en.md',
    'press_release_en.md',
    'paper.jsonld',
]

# Optional inclusions (require explicit flag)
DELIVERABLE_OPTIONAL = {
    '--include-uniqueness': 'uniqueness_review.md',
}

# IP-sensitive files that are NEVER included
DELIVERABLE_FORBIDDEN = [
    'potential_claims.md',
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROGRAM_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def run(cmd, cwd=None, capture=True, check=True):
    """Run a shell command and return its stdout (or empty string on capture=False)."""
    if isinstance(cmd, str):
        cmd_list = cmd.split()
    else:
        cmd_list = list(cmd)
    res = subprocess.run(cmd_list, cwd=cwd, capture_output=capture,
                         text=True, check=False)
    if check and res.returncode != 0:
        sys.stderr.write(f'  ERROR running {cmd_list}\n')
        sys.stderr.write(f'  stdout: {res.stdout}\n')
        sys.stderr.write(f'  stderr: {res.stderr}\n')
        raise subprocess.CalledProcessError(res.returncode, cmd_list,
                                            output=res.stdout, stderr=res.stderr)
    return res.stdout if capture else ''


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: Resolve paper config from YAML
# ---------------------------------------------------------------------------

def resolve_paper(yaml_path: Path, paper_id: str) -> dict:
    """Find the paper definition in YAML and return its config plus tag/outdir."""
    cfg = yaml.safe_load(yaml_path.read_text())
    papers = cfg.get('papers', [])
    for p in papers:
        if p.get('id') == paper_id:
            tag = p.get('tag', f'p{p.get("paper_num", "??")}_unknown')
            paper_num = p.get('paper_num', 0)
            return {
                'id': paper_id,
                'paper_num': paper_num,
                'tag': tag,
                'title': p.get('title', f'Paper {paper_id}'),
                'mode_test': f'runs/paper{paper_num:02d}/test/{tag}',
                'mode_prod': f'runs/paper{paper_num:02d}/baseline/{tag}',
                'paper_dir_test': f'papers/worth-paper-{paper_id}-{tag.replace("_", "-")}',
                'paper_dir_prod': f'papers/worth-paper-{paper_id}-{tag.replace("_", "-")}',
            }
    raise ValueError(f'Paper id {paper_id!r} not found in {yaml_path}')


# ---------------------------------------------------------------------------
# Step 2: Snapshot environment
# ---------------------------------------------------------------------------

def write_environment_files(env_dir: Path):
    env_dir.mkdir(parents=True, exist_ok=True)

    # Copy environment.yml if present
    env_yml = PROGRAM_ROOT / 'environment.yml'
    if env_yml.exists():
        shutil.copy2(env_yml, env_dir / 'environment.yml')

    # pip freeze
    try:
        pip_out = run(['pip', 'freeze'], cwd=PROGRAM_ROOT)
        (env_dir / 'pip_freeze.txt').write_text(pip_out)
    except Exception as e:
        (env_dir / 'pip_freeze.txt').write_text(f'pip freeze failed: {e}\n')

    # conda list (might not exist in non-conda env)
    try:
        conda_out = run(['conda', 'list', '--export'], cwd=PROGRAM_ROOT)
        (env_dir / 'conda_list.txt').write_text(conda_out)
    except Exception:
        # not running under conda or conda missing
        (env_dir / 'conda_list.txt').write_text('conda not available\n')

    # System info
    info_lines = [
        f'platform: {platform.platform()}',
        f'system:   {platform.system()}',
        f'release:  {platform.release()}',
        f'machine:  {platform.machine()}',
        f'processor: {platform.processor()}',
        f'python:   {sys.version}',
    ]

    for tool in ['mpirun', 'python3', 'gcc', 'cmake']:
        try:
            ver = run([tool, '--version'])
            ver_first_line = ver.split('\n')[0].strip()
            info_lines.append(f'{tool}: {ver_first_line}')
        except Exception:
            info_lines.append(f'{tool}: not available')

    # macOS specific
    if platform.system() == 'Darwin':
        try:
            sw = run(['sw_vers'])
            info_lines.append(f'sw_vers:\n{sw}')
        except Exception:
            pass

    (env_dir / 'system_info.txt').write_text('\n'.join(info_lines) + '\n')


# ---------------------------------------------------------------------------
# Step 3: Snapshot code via git archive
# ---------------------------------------------------------------------------

def write_code_snapshot(code_dir: Path) -> dict:
    """
    Use git archive to capture only tracked files (no __pycache__, no logs).
    Returns dict with commit SHA and tree state.
    """
    code_dir.mkdir(parents=True, exist_ok=True)

    # Check if we're in a git repo
    try:
        sha = run(['git', 'rev-parse', 'HEAD'], cwd=PROGRAM_ROOT).strip()
        branch = run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=PROGRAM_ROOT).strip()
        msg = run(['git', 'log', '-1', '--pretty=%s'], cwd=PROGRAM_ROOT).strip()
        date = run(['git', 'log', '-1', '--pretty=%ci'], cwd=PROGRAM_ROOT).strip()
        # check tree state (clean / dirty)
        status = run(['git', 'status', '--porcelain'], cwd=PROGRAM_ROOT).strip()
        clean = (status == '')
    except Exception as e:
        sha = 'unknown'
        branch = 'unknown'
        msg = f'(git command failed: {e})'
        date = 'unknown'
        clean = False
        status = '(git not available)'

    info = [
        f'commit: {sha}',
        f'branch: {branch}',
        f'date:   {date}',
        f'subject: {msg}',
        f'tree:   {"clean" if clean else "DIRTY (uncommitted changes present)"}',
    ]
    if not clean:
        info.append('')
        info.append('Uncommitted changes:')
        info.append(status)
    (code_dir / 'git_commit.txt').write_text('\n'.join(info) + '\n')

    # git archive — tracked files only, in a tarball inside the bundle
    if sha != 'unknown':
        archive_path = code_dir / 'source.tar.gz'
        try:
            run(['git', 'archive', '--format=tar.gz', f'--output={archive_path}',
                 'HEAD'], cwd=PROGRAM_ROOT, capture=False)
        except Exception as e:
            sys.stderr.write(f'  WARN: git archive failed: {e}\n')

    # Also include program_config.yaml separately for easy access
    yaml_src = PROGRAM_ROOT / 'program_config.yaml'
    if yaml_src.exists():
        shutil.copy2(yaml_src, code_dir / 'program_config.yaml')

    return {
        'commit': sha,
        'branch': branch,
        'date': date,
        'subject': msg,
        'clean': clean,
    }


# ---------------------------------------------------------------------------
# Step 4: Selectively copy diagnostics
# ---------------------------------------------------------------------------

def copy_diagnostics_standard(src_dir: Path, dst_dir: Path) -> dict:
    """
    Standard tier:
      - All CSVs and metadata files (small)
      - Every 10th early-diag snapshot (fields_early, particles_early)
      - Every late-diag snapshot (fields, particles) — already infrequent
      - run.log
    Returns stats: files included, files excluded, size info.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f'Simulation output dir does not exist: {src_dir}')

    dst_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        'csv_count': 0, 'csv_bytes': 0,
        'meta_count': 0, 'meta_bytes': 0,
        'log_bytes': 0,
        'early_total_snapshots': 0, 'early_included': 0, 'early_bytes': 0,
        'late_total_snapshots': 0, 'late_included': 0, 'late_bytes': 0,
    }

    # Top-level files: copy CSVs, txt, log
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        size = f.stat().st_size
        if f.suffix == '.csv':
            shutil.copy2(f, dst_dir / f.name)
            stats['csv_count'] += 1
            stats['csv_bytes'] += size
        elif f.name == 'run.log':
            shutil.copy2(f, dst_dir / 'run.log')
            stats['log_bytes'] = size
        elif f.suffix == '.txt':
            shutil.copy2(f, dst_dir / f.name)
            stats['meta_count'] += 1
            stats['meta_bytes'] += size

    # Diagnostic snapshots (subdirs named like fields_early, particles_early, fields, particles)
    for diag_name in ('fields_early', 'particles_early', 'fields', 'particles'):
        src = src_dir / diag_name
        if not src.is_dir():
            continue

        # openPMD writes one subdir per timestep (or one file). Sort and select.
        snapshots = sorted([s for s in src.iterdir()])
        is_early = 'early' in diag_name
        keep_every = 10 if is_early else 1   # late: keep all, early: every 10th

        if is_early:
            stats['early_total_snapshots'] += len(snapshots)
        else:
            stats['late_total_snapshots'] += len(snapshots)

        dst = dst_dir / diag_name
        dst.mkdir(exist_ok=True)

        for i, snap in enumerate(snapshots):
            if i % keep_every != 0:
                continue
            target = dst / snap.name
            try:
                if snap.is_dir():
                    shutil.copytree(snap, target)
                    size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
                else:
                    shutil.copy2(snap, target)
                    size = target.stat().st_size

                if is_early:
                    stats['early_included'] += 1
                    stats['early_bytes'] += size
                else:
                    stats['late_included'] += 1
                    stats['late_bytes'] += size
            except Exception as e:
                sys.stderr.write(f'  WARN: failed to copy {snap}: {e}\n')

    return stats


# ---------------------------------------------------------------------------
# Step 5: Copy whitelisted deliverables
# ---------------------------------------------------------------------------

def copy_deliverables(paper_dir: Path, dst_dir: Path,
                      include_uniqueness: bool = False) -> dict:
    """Copy ONLY whitelisted files. Returns stats."""
    if not paper_dir.exists():
        return {'included': [], 'missing': [], 'forbidden_skipped': []}

    dst_dir.mkdir(parents=True, exist_ok=True)
    stats = {'included': [], 'missing': [], 'forbidden_skipped': [],
             'optional_skipped': []}

    whitelist = list(DELIVERABLE_WHITELIST)
    if include_uniqueness:
        whitelist.append(DELIVERABLE_OPTIONAL['--include-uniqueness'])
        # uniqueness_review.md needs explicit flag
    else:
        stats['optional_skipped'].append(DELIVERABLE_OPTIONAL['--include-uniqueness'])

    for fname in whitelist:
        src = paper_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            stats['included'].append(fname)
        else:
            stats['missing'].append(fname)

    # Defensive: warn if forbidden files are present (we explicitly don't copy them)
    for fname in DELIVERABLE_FORBIDDEN:
        if (paper_dir / fname).exists():
            stats['forbidden_skipped'].append(fname)

    return stats


# ---------------------------------------------------------------------------
# Step 6: Optionally compile manuscript PDF
# ---------------------------------------------------------------------------

def compile_pdf(manuscript_dir: Path) -> bool:
    """Compile manuscript.tex via pdflatex+bibtex+pdflatex. Returns True on success."""
    tex = manuscript_dir / 'manuscript.tex'
    if not tex.exists():
        sys.stderr.write('  --build-pdf: manuscript.tex not in deliverables, skipping\n')
        return False

    # Need pdflatex available
    if shutil.which('pdflatex') is None:
        sys.stderr.write('  --build-pdf: pdflatex not in PATH, skipping\n')
        return False

    print('  Compiling manuscript.tex via pdflatex+bibtex+pdflatex+pdflatex...')
    try:
        # Pass 1
        run(['pdflatex', '-interaction=nonstopmode', 'manuscript.tex'],
            cwd=manuscript_dir)
        # bibtex (only if references.bib present)
        if (manuscript_dir / 'references.bib').exists():
            run(['bibtex', 'manuscript'], cwd=manuscript_dir, check=False)
        # Pass 2 + 3 to resolve refs
        run(['pdflatex', '-interaction=nonstopmode', 'manuscript.tex'],
            cwd=manuscript_dir, check=False)
        run(['pdflatex', '-interaction=nonstopmode', 'manuscript.tex'],
            cwd=manuscript_dir, check=False)
    except Exception as e:
        sys.stderr.write(f'  --build-pdf: compilation failed: {e}\n')
        return False

    pdf = manuscript_dir / 'manuscript.pdf'
    if pdf.exists():
        # Clean up intermediate files
        for ext in ['aux', 'bbl', 'blg', 'log', 'out', 'toc']:
            f = manuscript_dir / f'manuscript.{ext}'
            if f.exists():
                f.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Step 7: Generate manifest
# ---------------------------------------------------------------------------

def write_manifest(manifest_path: Path, paper_info: dict, version: str,
                   git_info: dict, mode: str, sim_outdir: Path,
                   data_stats: dict, deliv_stats: dict,
                   timestamp_iso: str):
    manifest = {
        'package_format_version': '1.0',
        'paper': {
            'id': paper_info['id'],
            'paper_num': paper_info['paper_num'],
            'tag': paper_info['tag'],
            'title': paper_info['title'],
        },
        'version': version,
        'mode': mode,
        'created': timestamp_iso,
        'git': git_info,
        'simulation_source_path': str(sim_outdir.relative_to(PROGRAM_ROOT)) if sim_outdir.is_relative_to(PROGRAM_ROOT) else str(sim_outdir),
        'data_tier': 'standard',
        'data_stats': data_stats,
        'deliverables': deliv_stats,
        'system': {
            'platform': platform.platform(),
            'python': sys.version.split()[0],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Step 8: Generate README inside the package
# ---------------------------------------------------------------------------

def write_readme(readme_path: Path, paper_info: dict, version: str,
                 git_info: dict, mode: str, data_stats: dict,
                 deliv_stats: dict, has_pdf: bool):
    early_total = data_stats.get('early_total_snapshots', 0)
    early_kept = data_stats.get('early_included', 0)
    late_total = data_stats.get('late_total_snapshots', 0)
    late_kept = data_stats.get('late_included', 0)

    deliv_in = deliv_stats.get('included', [])
    deliv_missing = deliv_stats.get('missing', [])
    deliv_forbidden = deliv_stats.get('forbidden_skipped', [])

    content = f"""# {paper_info['title']}

**Paper ID:** {paper_info['id']}
**Version:** {version}
**Run mode:** {mode}
**Code commit:** `{git_info['commit'][:12]}`{'  (DIRTY)' if not git_info['clean'] else ''}
**Created:** {datetime.datetime.utcnow().isoformat()}Z

## Contents

This Zenodo deposit contains the code, data, and manuscript artifacts for
"{paper_info['title']}".

```
.
├── README.md            ← this file
├── manifest.json        ← machine-readable metadata
├── checksums.sha256     ← SHA-256 of every file in this package
├── code/
│   ├── source.tar.gz    ← snapshot of all tracked source code at commit {git_info['commit'][:12]}
│   ├── program_config.yaml
│   └── git_commit.txt
├── environment/         ← conda env, pip freeze, system info
├── simulation/
│   ├── run.log
│   ├── *.csv            ← analysis time series
│   └── fields_early/, particles_early/, fields/, particles/
│                        ← openPMD diagnostics (Standard tier: every 10th early,
│                          every late)
└── manuscript/
    ├── manuscript.tex
    ├── references.bib
    ├── lay_summary_en.md
    ├── press_release_en.md
    └── paper.jsonld
{'    └── manuscript.pdf  ← compiled' if has_pdf else ''}
```

## Data tier: Standard

This deposit is a **Standard tier** package, designed for figure reproduction
and analysis verification. It includes:
- All CSV time series ({data_stats.get('csv_count', 0)} files, {human_size(data_stats.get('csv_bytes', 0))})
- Run log and metadata
- A subset of openPMD field/particle snapshots:
  - **Early diagnostics**: {early_kept}/{early_total} snapshots (every 10th)
  - **Late diagnostics**: {late_kept}/{late_total} snapshots (all)

For the full diagnostic record (often >100 GB), the simulation can be
re-run from the embedded source code with the configuration in
`code/program_config.yaml`.

## Reproduction

To reproduce the analysis:

```bash
# 1. Extract the source code
tar -xzf code/source.tar.gz -C /path/to/destination

# 2. Set up the conda environment
cd /path/to/destination
conda env create -f environment/environment.yml
conda activate plasma

# 3. Verify environment matches
diff <(pip freeze) environment/pip_freeze.txt

# 4. Re-run simulation OR use the included diagnostics in simulation/
```

## Deliverables

Included: {', '.join(deliv_in) if deliv_in else '(none)'}

Missing (not produced for this run): {', '.join(deliv_missing) if deliv_missing else '(none)'}

{'**Note:** Patent-sensitive files (' + ', '.join(deliv_forbidden) + ') are present in the local working directory but are EXCLUDED from this Zenodo deposit by policy.' if deliv_forbidden else ''}

## Citation

If you use this code or data, please cite:

> Worth, J.B. ({datetime.datetime.utcnow().year}). {paper_info['title']} (Version {version}) [Data set]. Zenodo. https://doi.org/[DOI assigned by Zenodo on publish]

A `CITATION.cff` is included for automated citation tools.

## License

- Code: see LICENSE in code/source.tar.gz (typically MIT)
- Data: CC-BY-4.0 (recommended for Zenodo data deposits)
- Manuscript: depends on journal — see preprint or final publication

## Contact

James B. Worth — bworth@substrate.ai
ORCID: 0009-0005-5000-9497
Substrate AI, Valencia, Spain
"""
    readme_path.write_text(content)


# ---------------------------------------------------------------------------
# Step 9: Generate CITATION.cff
# ---------------------------------------------------------------------------

def write_citation_cff(cff_path: Path, paper_info: dict, version: str,
                       git_info: dict):
    cff = {
        'cff-version': '1.2.0',
        'message': 'If you use this software or data, please cite as below.',
        'authors': [{
            'family-names': 'Worth',
            'given-names': 'James B.',
            'orcid': 'https://orcid.org/0009-0005-5000-9497',
            'affiliation': 'Substrate AI',
        }],
        'title': paper_info['title'],
        'version': version,
        'date-released': datetime.date.today().isoformat(),
        'type': 'dataset',
        'repository-code': 'https://github.com/[your-username]/[your-repo]',
        'identifiers': [
            {
                'type': 'other',
                'value': f'git-commit:{git_info["commit"]}',
                'description': 'Git commit hash',
            },
        ],
    }
    cff_path.write_text(yaml.dump(cff, default_flow_style=False, sort_keys=False))


# ---------------------------------------------------------------------------
# Step 10: SHA-256 manifest of the entire package
# ---------------------------------------------------------------------------

def write_checksums(staging_dir: Path):
    """Walk the staging dir, compute sha256 of every file, write checksums.sha256."""
    lines = []
    for f in sorted(staging_dir.rglob('*')):
        if not f.is_file() or f.name == 'checksums.sha256':
            continue
        rel = f.relative_to(staging_dir)
        h = sha256_file(f)
        lines.append(f'{h}  {rel}')
    (staging_dir / 'checksums.sha256').write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paper', required=True,
                    help='Paper ID (must match an entry in program_config.yaml)')
    ap.add_argument('--version', required=True,
                    help='Version string for this release (e.g. v0.1, v1.0-paper01)')
    ap.add_argument('--mode', default='test', choices=['test', 'baseline'],
                    help='Run mode: test (smaller grid) or baseline (production)')
    ap.add_argument('--output', default=None,
                    help='Output dir (default: ./releases/)')
    ap.add_argument('--build-pdf', action='store_true',
                    help='Compile manuscript.tex to PDF before packaging')
    ap.add_argument('--include-uniqueness', action='store_true',
                    help='Include uniqueness_review.md (only after attorney review)')
    ap.add_argument('--config', default='program_config.yaml',
                    help='Path to program config YAML')
    ap.add_argument('--dry-run', action='store_true',
                    help='Show what would be packaged without producing tarball')
    args = ap.parse_args()

    # Validate paper exists in YAML
    yaml_path = PROGRAM_ROOT / args.config
    if not yaml_path.exists():
        sys.exit(f'ERROR: {yaml_path} not found')

    paper_info = resolve_paper(yaml_path, args.paper)
    print(f'Paper: {paper_info["id"]} - {paper_info["title"]}')
    print(f'Tag:   {paper_info["tag"]}')
    print(f'Mode:  {args.mode}')

    # Locate simulation output
    if args.mode == 'test':
        sim_outdir = PROGRAM_ROOT / paper_info['mode_test']
    else:
        sim_outdir = PROGRAM_ROOT / paper_info['mode_prod']

    if not sim_outdir.exists():
        sys.exit(f'ERROR: no simulation output at {sim_outdir}\n'
                 f'  Run the simulation first.')

    # Locate paper deliverables dir
    paper_dir = PROGRAM_ROOT / paper_info['paper_dir_test']
    if not paper_dir.exists():
        print(f'  WARN: no deliverables dir at {paper_dir} (drafting may not have run)')

    # Output location
    output_dir = Path(args.output).resolve() if args.output else (PROGRAM_ROOT / 'releases')
    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = f'worth-paper-{args.paper}-{paper_info["tag"].replace("_", "-")}-{args.version}'
    final_tarball = output_dir / f'{package_name}.tar.gz'

    if args.dry_run:
        print()
        print('=== DRY RUN ===')
        print(f'Would create:  {final_tarball}')
        print(f'Source sim:    {sim_outdir}')
        print(f'Source paper:  {paper_dir}')
        print(f'Build PDF:     {args.build_pdf}')
        print(f'Include uniq:  {args.include_uniqueness}')
        # Show what diagnostics would be in standard tier
        print()
        print('Standard-tier diagnostic preview:')
        for d in ('fields_early', 'particles_early', 'fields', 'particles'):
            sub = sim_outdir / d
            if sub.is_dir():
                snaps = sorted(sub.iterdir())
                kept = 1 if 'early' not in d else 10
                inc = (len(snaps) + kept - 1) // kept
                print(f'  {d}: {inc}/{len(snaps)} snapshots would be included')
        sys.exit(0)

    # Build in temp staging dir, then tarball
    timestamp_iso = datetime.datetime.utcnow().isoformat() + 'Z'

    print()
    print('=== Packaging ===')
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / package_name
        staging.mkdir(parents=True)

        print('  [1/8] Snapshotting code (git archive)...')
        git_info = write_code_snapshot(staging / 'code')

        print('  [2/8] Capturing environment...')
        write_environment_files(staging / 'environment')

        print('  [3/8] Copying simulation diagnostics...')
        data_stats = copy_diagnostics_standard(sim_outdir, staging / 'simulation')
        print(f'       early: {data_stats["early_included"]}/{data_stats["early_total_snapshots"]} '
              f'({human_size(data_stats["early_bytes"])})')
        print(f'       late:  {data_stats["late_included"]}/{data_stats["late_total_snapshots"]} '
              f'({human_size(data_stats["late_bytes"])})')

        print('  [4/8] Copying deliverables (whitelist)...')
        deliv_stats = copy_deliverables(paper_dir, staging / 'manuscript',
                                        include_uniqueness=args.include_uniqueness)
        print(f'       included: {deliv_stats["included"]}')
        if deliv_stats['missing']:
            print(f'       missing:  {deliv_stats["missing"]}')
        if deliv_stats['forbidden_skipped']:
            print(f'       NOT included (patent-sensitive): {deliv_stats["forbidden_skipped"]}')

        has_pdf = False
        if args.build_pdf:
            print('  [5/8] Building PDF...')
            has_pdf = compile_pdf(staging / 'manuscript')
        else:
            print('  [5/8] Skipping PDF build (use --build-pdf to enable)')

        print('  [6/8] Writing manifest, README, CITATION.cff...')
        write_manifest(staging / 'manifest.json', paper_info, args.version,
                       git_info, args.mode, sim_outdir, data_stats, deliv_stats,
                       timestamp_iso)
        write_readme(staging / 'README.md', paper_info, args.version, git_info,
                     args.mode, data_stats, deliv_stats, has_pdf)
        write_citation_cff(staging / 'CITATION.cff', paper_info, args.version,
                           git_info)

        print('  [7/8] Computing SHA-256 checksums...')
        write_checksums(staging)

        print(f'  [8/8] Creating tarball: {final_tarball}')
        with tarfile.open(final_tarball, 'w:gz') as tf:
            tf.add(staging, arcname=package_name)

    # Final stats
    final_size = final_tarball.stat().st_size
    final_sha = sha256_file(final_tarball)

    print()
    print('=' * 70)
    print(f'  ✓ Created: {final_tarball}')
    print(f'    Size:    {human_size(final_size)}')
    print(f'    SHA-256: {final_sha}')
    print('=' * 70)
    print()
    print('Next steps:')
    print('  1. Review the tarball contents:  tar -tzf ' + str(final_tarball) + ' | head')
    print('  2. Upload to Zenodo:             https://zenodo.org/uploads/new')
    print(f'     Title: "Data and analysis for: {paper_info["title"]}"')
    print('     License: CC-BY-4.0')
    print('     Upload type: Dataset')
    print('  3. (Optional) Tag a code release on GitHub for code DOI')
    print('  4. After publish, link the two via Zenodo "Related identifiers"')
    print()


if __name__ == '__main__':
    main()
