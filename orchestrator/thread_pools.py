"""
thread_pools.py — Pool management for the orchestrator.

The orchestrator coordinates three concurrent workstreams:
  1. Simulation pool (managed by pb11_run_all_papers.py — we don't replace it)
  2. Drafting pool (managed here — semaphore + worker threads)
  3. Progress reporting (managed by progress.py)

This module owns the drafting pool. Simulation pool is left to the
existing pb11_run_all_papers infrastructure; we just hook into its
on_complete callback.

Design:
  - Semaphore caps concurrent API calls (default 4)
  - Worker threads are daemons so they die with the main process
  - Bounded join timeouts on shutdown (no infinite waits)
  - Submission queue allows bursting more than capacity tasks
  - All state changes report to ProgressDisplay
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Optional


class DraftingPool:
    """
    Bounded thread pool for AI drafting tasks.

    Submitted tasks queue and run concurrently up to max_workers limit.
    On shutdown, in-flight tasks are allowed to complete (with timeout);
    queued tasks are dropped.
    """

    def __init__(self, max_workers: int = 4,
                 shutdown_timeout: float = 10.0,
                 on_task_start: Optional[Callable] = None,
                 on_task_done: Optional[Callable] = None):
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='drafting'
        )
        self._in_flight = 0
        self._lock = threading.Lock()
        self._shutdown_timeout = shutdown_timeout
        self._on_task_start = on_task_start
        self._on_task_done = on_task_done
        self._futures = []   # track for shutdown

    @property
    def capacity(self) -> int:
        return self._max_workers

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def submit(self, name: str, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a drafting task. Returns a Future.

        The task wrapper notifies hooks before/after execution.
        """
        def wrapper():
            with self._lock:
                self._in_flight += 1
            if self._on_task_start:
                try:
                    self._on_task_start(name)
                except Exception:
                    pass
            error = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                with self._lock:
                    self._in_flight -= 1
                if self._on_task_done:
                    try:
                        self._on_task_done(name, error)
                    except Exception:
                        pass

        future = self._executor.submit(wrapper)
        self._futures.append(future)
        return future

    def shutdown(self, wait: bool = True):
        """
        Initiate shutdown. If wait, block (with timeout) for in-flight
        tasks to complete. Drops queued (not-yet-started) tasks.
        """
        # Cancel queued futures (Python 3.9+: cancel_futures kwarg)
        try:
            self._executor.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            # Older Python: no cancel_futures. Best-effort.
            self._executor.shutdown(wait=wait)

    def shutdown_with_timeout(self):
        """Bounded shutdown — don't block forever."""
        # First, try to cancel pending futures
        for f in self._futures:
            if not f.running() and not f.done():
                try:
                    f.cancel()
                except Exception:
                    pass

        # Then shutdown executor with timeout via separate thread
        done = threading.Event()

        def stopper():
            try:
                self._executor.shutdown(wait=True)
            finally:
                done.set()

        t = threading.Thread(target=stopper, daemon=True, name='executor-shutdown')
        t.start()
        if not done.wait(timeout=self._shutdown_timeout):
            import sys
            print(f"  WARN: drafting pool shutdown exceeded "
                  f"{self._shutdown_timeout}s; some tasks may not finish",
                  file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pool = DraftingPool(max_workers=2)

    def task(name: str, duration: float):
        print(f"  [{name}] starting ({duration}s)")
        time.sleep(duration)
        print(f"  [{name}] done")
        return name

    futures = [pool.submit(f'task-{i}', task, f'task-{i}', 1.0) for i in range(4)]
    print(f"  Capacity: {pool.capacity}, currently in-flight: {pool.in_flight}")
    time.sleep(0.3)
    print(f"  After 0.3s, in-flight: {pool.in_flight}")

    for f in futures:
        f.result()

    pool.shutdown_with_timeout()
    print('  self-test complete')
