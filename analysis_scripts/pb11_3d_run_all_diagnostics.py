#!/usr/bin/env python3
"""
pb11_3d_run_all_diagnostics.py — Run all three 3D-pilot diagnostics in sequence.

This is a convenience wrapper. Use it after a 3D-pilot run completes:

    python pb11_3d_run_all_diagnostics.py \\
        --run-dir runs/pb11_3d_pilot_<timestamp> \\
        --compare-2d-zone-report runs/p1_ld_512_4500_ultrafine/zone_report.txt

Produces in <run-dir>/3d_diag/:
    3d_kink_growth.{csv,png,txt}
    3d_outflow_tilt.{csv,png,txt}
    3d_central_convergence.{csv,png,txt}
    3d_summary.txt        (combined verdicts from all three)

Total runtime: ~5-15 minutes depending on number of dumps and particle count.
"""

import argparse
import subprocess
import sys
from pathlib import Path

DIAGS = [
    ('pb11_3d_kink_growth.py', '3d_kink_growth.txt'),
    ('pb11_3d_outflow_tilt.py', '3d_outflow_tilt.txt'),
    ('pb11_3d_central_convergence.py', '3d_central_convergence.txt'),
]

ANIMATIONS = [
    ('pb11_3d_anim_xz_slice.py', '3d_anim_xz_slice.mp4'),
    ('pb11_3d_anim_yz_xline.py', '3d_anim_yz_xline0.mp4'),
    ('pb11_3d_anim_isosurface.py', '3d_anim_isosurface.mp4'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--compare-2d-zone-report', default=None, type=Path)
    p.add_argument('--script-dir', default=Path(__file__).parent, type=Path,
                   help='Directory containing the diagnostic + animation scripts')
    p.add_argument('--skip-animations', action='store_true',
                   help='Skip the three animation scripts (faster, ~5 min total). '
                        'Default: run animations after diagnostics (~15-30 min total).')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.run_dir / '3d_diag'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Running 3D-pilot diagnostics on: {args.run_dir}')
    print(f'Output directory: {out_dir}')
    print(f'='*70)

    summaries = []
    for script_name, summary_file in DIAGS:
        script_path = args.script_dir / script_name
        if not script_path.exists():
            print(f'\nWARN: {script_name} not found at {script_path}, skipping')
            summaries.append((script_name, '(script not found)'))
            continue

        print(f'\n>> {script_name}')
        cmd = [sys.executable, str(script_path),
               '--run-dir', str(args.run_dir),
               '--out-dir', str(out_dir)]
        if (script_name == 'pb11_3d_central_convergence.py'
                and args.compare_2d_zone_report is not None):
            cmd += ['--compare-2d-zone-report', str(args.compare_2d_zone_report)]

        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f'  WARN: {script_name} exited with code {result.returncode}')
        except Exception as e:
            print(f'  ERROR running {script_name}: {e}')

        summary_path = out_dir / summary_file
        if summary_path.exists():
            content = summary_path.read_text()
            verdict_lines = [ln for ln in content.splitlines() if 'VERDICT' in ln]
            verdict = verdict_lines[0] if verdict_lines else '(no verdict line)'
            summaries.append((script_name, verdict))

    # Animations
    if not args.skip_animations:
        print(f'\n{"="*70}')
        print('ANIMATIONS (this may take 15-30 minutes)')
        print(f'{"="*70}')
        for script_name, output_name in ANIMATIONS:
            script_path = args.script_dir / script_name
            if not script_path.exists():
                print(f'\nWARN: {script_name} not found, skipping')
                continue
            print(f'\n>> {script_name}')
            cmd = [sys.executable, str(script_path),
                   '--run-dir', str(args.run_dir),
                   '--out-dir', str(out_dir)]
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f'  ERROR running {script_name}: {e}')

    # Combined summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    for script_name, verdict in summaries:
        print(f'{script_name}:')
        print(f'  {verdict}')

    summary_path = out_dir / '3d_summary.txt'
    with open(summary_path, 'w') as fh:
        fh.write(f'3D-pilot diagnostic summary\n')
        fh.write(f'{"="*70}\n')
        fh.write(f'run_dir = {args.run_dir}\n\n')
        for script_name, verdict in summaries:
            fh.write(f'{script_name}:\n  {verdict}\n\n')
    print(f'\nWrote {summary_path}')


if __name__ == '__main__':
    main()
