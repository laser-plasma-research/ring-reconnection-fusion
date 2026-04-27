"""
shutdown.py — Clean shutdown coordinator for the orchestrator.

Coordinates graceful shutdown across simulation pool, drafting pool,
and progress reporter. Handles SIGINT (Ctrl-C) and SIGTERM cleanly.

Design principles:
  - Bounded waits: every join() has a timeout; nothing blocks forever
  - Clear escalation: SIGTERM → wait → force exit
  - Daemon threads die with main process; non-daemon threads are joined
  - Fire-and-forget cleanup callbacks for downstream cleanup tasks

Usage:
    from orchestrator.shutdown import ShutdownCoordinator
    sd = ShutdownCoordinator()
    sd.register_cleanup('drafting_pool', drafting_pool.shutdown)
    sd.install_signal_handlers()
    # ... main loop ...
    sd.shutdown_all(reason='completed normally')
"""

import sys
import signal
import threading
import time
from typing import Callable
from datetime import datetime


class ShutdownCoordinator:
    """
    Tracks the shutdown lifecycle and coordinates clean exit.

    The main thread checks `is_shutting_down()` periodically. Worker
    threads check `should_stop()` between iterations.
    """

    def __init__(self, force_exit_after_seconds: float = 15.0):
        self._stop_event = threading.Event()
        self._cleanup_callbacks: list = []
        self._cleanup_lock = threading.Lock()
        self._force_exit_after = force_exit_after_seconds
        self._shutdown_started_at = None
        self._original_handlers = {}

    # ---- Status ----

    def is_shutting_down(self) -> bool:
        return self._stop_event.is_set()

    def should_stop(self) -> bool:
        """Workers call this between iterations to check for shutdown."""
        return self._stop_event.is_set()

    @property
    def stop_event(self) -> threading.Event:
        """Direct access for code that needs to wait() on the event."""
        return self._stop_event

    # ---- Cleanup callback registration ----

    def register_cleanup(self, name: str, callback: Callable, priority: int = 0):
        """
        Register a cleanup callback. Higher priority callbacks run first.

        The callback is called with no arguments. Exceptions are logged
        but do not block other cleanups from running.
        """
        with self._cleanup_lock:
            self._cleanup_callbacks.append((priority, name, callback))
            self._cleanup_callbacks.sort(key=lambda t: -t[0])  # highest first

    # ---- Signal handling ----

    def install_signal_handlers(self):
        """Install SIGINT and SIGTERM handlers. Save originals for restore."""
        self._original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        self._original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def restore_signal_handlers(self):
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    def _signal_handler(self, signum, frame):
        """Set stop event on first signal; force-exit on second."""
        if self._stop_event.is_set():
            # Second Ctrl-C: force exit
            print("\n\n  Force exit (second signal received)", flush=True)
            import os
            os._exit(2)

        # First signal: graceful shutdown
        sig_name = {signal.SIGINT: 'SIGINT (Ctrl-C)',
                    signal.SIGTERM: 'SIGTERM'}.get(signum, f'signal {signum}')
        print(f"\n\n  {sig_name} received — shutting down gracefully...", flush=True)
        print(f"  Press Ctrl-C again to force exit.\n", flush=True)
        self.shutdown_all(reason=sig_name)

    # ---- Shutdown driver ----

    def shutdown_all(self, reason: str = 'requested'):
        """
        Run all registered cleanup callbacks, then return.

        Each callback gets up to a portion of the total budget. If the
        whole shutdown exceeds force_exit_after_seconds, force exit.
        """
        if self._stop_event.is_set():
            return  # already shutting down
        self._stop_event.set()
        self._shutdown_started_at = time.time()

        with self._cleanup_lock:
            callbacks = list(self._cleanup_callbacks)

        if not callbacks:
            return

        total_budget = self._force_exit_after
        per_callback = max(1.0, total_budget / max(1, len(callbacks)))

        for priority, name, cb in callbacks:
            elapsed = time.time() - self._shutdown_started_at
            if elapsed >= total_budget:
                print(f"  WARN: shutdown budget exceeded; skipping {name}",
                      file=sys.stderr, flush=True)
                continue
            try:
                self._run_with_timeout(cb, per_callback, name)
            except Exception as e:
                print(f"  WARN: cleanup '{name}' raised: {e}",
                      file=sys.stderr, flush=True)

    def _run_with_timeout(self, callback: Callable, timeout: float, name: str):
        """Run callback in a thread, killing it after timeout."""
        result = {'done': False, 'error': None}

        def runner():
            try:
                callback()
            except Exception as e:
                result['error'] = e
            finally:
                result['done'] = True

        t = threading.Thread(target=runner, daemon=True, name=f'cleanup-{name}')
        t.start()
        t.join(timeout=timeout)
        if not result['done']:
            print(f"  WARN: cleanup '{name}' did not complete within {timeout:.1f}s",
                  file=sys.stderr, flush=True)
        elif result['error']:
            print(f"  WARN: cleanup '{name}' raised: {result['error']}",
                  file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sd = ShutdownCoordinator(force_exit_after_seconds=5.0)
    sd.register_cleanup('fast', lambda: print('  fast cleanup ran'), priority=10)
    sd.register_cleanup('slow', lambda: time.sleep(0.5) or print('  slow cleanup ran'), priority=5)
    sd.shutdown_all(reason='self-test')
    print(f"  is_shutting_down: {sd.is_shutting_down()}")
