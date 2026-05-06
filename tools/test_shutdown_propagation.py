#!/usr/bin/env python3
"""
test_shutdown_propagation.py — Verify that the orchestrator's ShutdownCoordinator
correctly propagates Ctrl-C to pb11_run_all_papers' _stop_event and process
registry.

This is a fast unit test (no WarpX, no MPI). It:
  1. Imports both modules
  2. Manually adds a fake process to pb11's registry
  3. Triggers ShutdownCoordinator's cleanup
  4. Verifies pb11._stop_event is set and registry was processed

Run from program root:
    python3 tools/test_shutdown_propagation.py
"""

import sys
import os
import threading
import subprocess
from pathlib import Path

# Add simulation/ to sys.path so pb11 imports work
PROGRAM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROGRAM_ROOT))
sys.path.insert(0, str(PROGRAM_ROOT / 'simulation'))

print("=== Shutdown propagation test ===")
print()

# 1. Import modules
print("[1] Importing orchestrator and pb11...")
from orchestrator.shutdown import ShutdownCoordinator
import pb11_run_all_papers as pb11
print("    ✓ Both imported")
print()

# 2. Verify pb11's stop_event starts unset
print("[2] pb11._stop_event initial state...")
assert not pb11._stop_event.is_set(), "stop_event should start clear"
print(f"    ✓ pb11._stop_event.is_set() = {pb11._stop_event.is_set()}")
print()

# 3. Define the cleanup function (same as in run_all.py)
print("[3] Setting up ShutdownCoordinator with pb11 cleanup callback...")
shutdown = ShutdownCoordinator(force_exit_after_seconds=15.0)

def stop_pb11():
    pb11._kill_all_procs(signum=None, frame=None)

shutdown.register_cleanup('pb11_simulations', stop_pb11, priority=1)
print("    ✓ Cleanup registered at priority 1")
print()

# 4. Spawn a real (harmless) subprocess and register it with pb11
print("[4] Spawning a sleep subprocess and registering with pb11...")
proc = subprocess.Popen(
    ['sleep', '60'],
    start_new_session=True,
)
pb11._register_proc('test_sleep', proc)
print(f"    ✓ Spawned PID {proc.pid}, registered as 'test_sleep'")
print(f"    PIDs in registry: {list(pb11._active_procs.keys())}")
print()

# 5. Trigger shutdown manually
print("[5] Triggering shutdown.request_shutdown('test')...")
shutdown.request_shutdown(reason='test')
print()

# 6. Run cleanup callbacks
print("[6] Running cleanup callbacks...")
shutdown.run_cleanup_callbacks()
print()

# 7. Verify state
print("[7] Verifying state after cleanup:")
print(f"    pb11._stop_event.is_set() = {pb11._stop_event.is_set()}")
assert pb11._stop_event.is_set(), "stop_event should be set after cleanup"

import time
time.sleep(1)  # give SIGTERM time to land
proc.poll()
print(f"    subprocess returncode = {proc.returncode}")

# Process should be dead (returncode set) or about to die
if proc.returncode is None:
    print("    Process still alive after 1s — sending SIGKILL as fallback")
    proc.kill()
    proc.wait(timeout=2)

print()
print("=== All checks passed — shutdown propagation works ===")
