#!/usr/bin/env python3
"""
Run this script in the experiment directory to patch pb11_run_all_papers.py
in-place. It replaces the exit-code check so that exit code 1 with
SIMULATION COMPLETE in the log is treated as success (normal MPI cleanup).

Usage:
    cd ~/LaserFusionResearch/experiments/pb11_rot_reconnection_ring
    python3 patch_runner.py
"""
import re, ast, sys

FNAME = 'pb11_run_all_papers.py'

with open(FNAME) as f:
    src = f.read()

# ── Find the block to replace ────────────────────────────────────────────────
# We search for the distinctive string that starts the exit-code check block.
MARKER = 'ret = proc.returncode'
idx = src.find(MARKER)
if idx == -1:
    print("ERROR: Could not find 'ret = proc.returncode' in runner — wrong file?")
    sys.exit(1)

# Find the end of the block: '_unregister_proc(tag)' followed by the analyse block
END_MARKER = 'checks[c] = (\'SKIP\', \'skipped — run did not complete\')'
end_idx = src.find(END_MARKER, idx)
if end_idx == -1:
    print("ERROR: Could not find end marker of exit-code block.")
    sys.exit(1)
# Include the closing line + newline
end_idx = src.index('\n', end_idx) + 1

old_block = src[idx:end_idx]

NEW_BLOCK = '''\
ret = proc.returncode
                    # Read log to check for clean completion
                    log_content = ''
                    try:
                        with open(log_path) as _lf:
                            log_content = _lf.read()
                    except Exception:
                        pass
                    sim_completed = 'SIMULATION COMPLETE' in log_content

                    # Exit code 1 on macOS MPI (prterun) is normal when non-rank-0
                    # processes exit after rank 0 finishes — NOT a real failure.
                    # Real failures: negative exit (signal), or no SIMULATION COMPLETE.
                    real_failure = False
                    if 'timeout' not in checks:
                        if ret is not None and ret < 0:
                            checks['process_exit'] = ('FAIL',
                                f'Killed by signal {-ret} — check preflight/{tag}.log')
                            real_failure = True
                        elif not sim_completed and ret not in (0, None):
                            checks['process_exit'] = ('FAIL',
                                f'Exit code {ret} without SIMULATION COMPLETE — '
                                f'check preflight/{tag}.log')
                            real_failure = True

                    _unregister_proc(tag)
                    if not real_failure and 'timeout' not in checks:
                        # Simulation completed — run physics analysis
                        try:
                            more = analyse_preflight_job(
                                pdir, tag, base_density=5e24)
                            checks.update(more)
                        except Exception as ae:
                            checks['analysis_error'] = ('WARN',
                                f'Analysis failed: {ae}')
                    else:
                        # Run failed — skip remaining checks
                        for c in ['b_field_init', 'b_field_sustained',
                                  'no_spike', 'n_centre', 'fusion_csv', 'density_ratio']:
                            if c not in checks:
                                checks[c] = ('SKIP', 'skipped — run did not complete')
'''

if old_block == NEW_BLOCK:
    print("Already patched — nothing to do.")
    sys.exit(0)

src_new = src[:idx] + NEW_BLOCK + src[end_idx:]

# Verify syntax before writing
try:
    ast.parse(src_new)
except SyntaxError as e:
    print(f"ERROR: Syntax error in patched file: {e}")
    sys.exit(1)

# Verify key phrases
assert 'sim_completed' in src_new
assert 'SIMULATION COMPLETE' in src_new
assert 'real_failure' in src_new

with open(FNAME, 'w') as f:
    f.write(src_new)

print(f"PATCHED OK — {FNAME}")
print(f"  Old block: {len(old_block)} chars")
print(f"  New block: {len(NEW_BLOCK)} chars")
print(f"  File lines: {len(src_new.splitlines())}")
print()
print("Verify with:")
print("  grep 'sim_completed\\|real_failure\\|SIMULATION COMPLETE' pb11_run_all_papers.py | head -6")
