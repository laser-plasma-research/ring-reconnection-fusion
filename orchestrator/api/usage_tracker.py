"""
usage_tracker.py — Persistent spend tracking against budget caps.

Tracks token usage and dollar cost per Claude API call. Budget enforcement
operates at three time scales:
  - per_run: cumulative spend in the current orchestrator invocation
  - per_day: spend in the current calendar day (UTC)
  - per_month: spend in the current calendar month

Usage state persists in shared/usage_log.json so daily/monthly totals
survive between invocations. The file is gitignored.

Usage:
    tracker = UsageTracker(
        budget={'per_run_usd': 5.0, 'per_day_usd': 10.0, 'per_month_usd': 50.0},
        log_path='shared/usage_log.json',
    )
    
    # Before making a call:
    tracker.check_can_spend(estimated_cost_usd=0.05)  # raises if would exceed
    
    # After the call:
    tracker.record(model='claude-opus-4-6',
                   input_tokens=12000, output_tokens=2500,
                   actual_cost_usd=0.18,
                   deliverable='manuscript', paper_id='A1')
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class BudgetExceededError(Exception):
    """Raised when a planned API call would exceed a budget cap."""
    def __init__(self, scope: str, current: float, would_become: float, cap: float):
        self.scope = scope
        self.current = current
        self.would_become = would_become
        self.cap = cap
        super().__init__(
            f"Budget cap exceeded ({scope}): "
            f"current ${current:.2f} + estimated would reach ${would_become:.2f}, "
            f"cap is ${cap:.2f}"
        )


# ---------------------------------------------------------------------------
# Approximate costs per model (USD per million tokens)
# Used for pre-call estimation. Actual costs come from API headers when available.
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # Claude 4.x family (approximate; update as Anthropic publishes pricing)
    'claude-opus-4-7':         {'input': 15.0,  'output': 75.0},
    'claude-opus-4-6':         {'input': 15.0,  'output': 75.0},
    'claude-opus-4-5':         {'input': 15.0,  'output': 75.0},
    'claude-sonnet-4-6':       {'input':  3.0,  'output': 15.0},
    'claude-sonnet-4-5':       {'input':  3.0,  'output': 15.0},
    'claude-haiku-4-5':        {'input':  1.0,  'output':  5.0},
    'claude-haiku-4-5-20251001': {'input': 1.0, 'output': 5.0},
    # Legacy fallback
    'default':                 {'input':  3.0,  'output': 15.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API call cost from model and token counts."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING['default'])
    return (input_tokens * pricing['input'] / 1_000_000 +
            output_tokens * pricing['output'] / 1_000_000)


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------

class UsageTracker:
    """
    Tracks API usage and enforces budget caps.

    Thread-safe: uses a lock around all state mutation.
    """

    def __init__(self,
                 budget: dict,
                 log_path: Optional[str] = None):
        """
        Args:
            budget: dict with keys per_run_usd, per_day_usd, per_month_usd,
                    hard_stop_at_pct (default 100), warn_at_pct (default 70)
            log_path: path to JSON log file. If None, no persistence.
        """
        self._budget = budget or {}
        self._log_path = Path(log_path).expanduser() if log_path else None
        # RLock allows re-entrant acquisition by the same thread
        self._lock = threading.RLock()

        self._run_total_usd = 0.0
        self._call_log = []  # in-memory log for this run

        # Load persisted history
        self._history = self._load_history()

    # ---- Persistence ----

    def _load_history(self) -> list:
        """Load past calls from log file. Returns [] if missing."""
        if not self._log_path or not self._log_path.exists():
            return []
        try:
            with open(self._log_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_history(self):
        """Write history to log file."""
        if not self._log_path:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, 'w') as f:
                json.dump(self._history, f, indent=2)
        except IOError:
            # Best-effort persistence
            pass

    # ---- Budget queries ----

    def get_spent(self, scope: str = 'run') -> float:
        """
        Return total USD spent in scope.
        
        scope: 'run', 'day', 'month', 'all_time'
        """
        with self._lock:
            if scope == 'run':
                return self._run_total_usd
            
            now = datetime.now(timezone.utc)
            if scope == 'day':
                today = now.date().isoformat()
                return sum(c['cost_usd'] for c in self._history
                          if c.get('timestamp', '').startswith(today))
            elif scope == 'month':
                month = now.strftime('%Y-%m')
                return sum(c['cost_usd'] for c in self._history
                          if c.get('timestamp', '').startswith(month))
            elif scope == 'all_time':
                return sum(c['cost_usd'] for c in self._history)
            else:
                raise ValueError(f"Unknown scope: {scope}")

    def get_call_count(self, scope: str = 'run') -> int:
        """Return number of calls in scope."""
        with self._lock:
            if scope == 'run':
                return len(self._call_log)
            now = datetime.now(timezone.utc)
            if scope == 'day':
                today = now.date().isoformat()
                return sum(1 for c in self._history
                          if c.get('timestamp', '').startswith(today))
            elif scope == 'month':
                month = now.strftime('%Y-%m')
                return sum(1 for c in self._history
                          if c.get('timestamp', '').startswith(month))
            elif scope == 'all_time':
                return len(self._history)
            else:
                raise ValueError(f"Unknown scope: {scope}")

    # ---- Budget enforcement ----

    def check_can_spend(self, estimated_cost_usd: float) -> None:
        """
        Verify that adding `estimated_cost_usd` to spend won't exceed any cap.
        Raises BudgetExceededError if it would.
        """
        with self._lock:
            scope_caps = {
                'per_run_usd': ('run', self.get_spent('run')),
                'per_day_usd': ('day', self.get_spent('day')),
                'per_month_usd': ('month', self.get_spent('month')),
            }
            
            for budget_key, (scope, current) in scope_caps.items():
                cap = self._budget.get(budget_key)
                if cap is None or cap <= 0:
                    continue
                hard_stop_pct = self._budget.get('hard_stop_at_pct', 100) / 100.0
                effective_cap = cap * hard_stop_pct
                would_become = current + estimated_cost_usd
                if would_become > effective_cap:
                    raise BudgetExceededError(scope, current, would_become, effective_cap)

    def warning_threshold_reached(self) -> Optional[str]:
        """
        Check if any scope has crossed warn_at_pct of cap.
        Returns scope name (e.g. 'day') if crossed, else None.
        """
        warn_pct = self._budget.get('warn_at_pct', 70) / 100.0
        # NOTE: don't acquire lock here — get_spent() acquires it
        for scope, budget_key in [('run', 'per_run_usd'),
                                   ('day', 'per_day_usd'),
                                   ('month', 'per_month_usd')]:
            cap = self._budget.get(budget_key)
            if cap is None or cap <= 0:
                continue
            spent = self.get_spent(scope)
            if spent >= cap * warn_pct:
                return scope
        return None

    # ---- Recording ----

    def record(self,
               model: str,
               input_tokens: int,
               output_tokens: int,
               actual_cost_usd: Optional[float] = None,
               deliverable: Optional[str] = None,
               paper_id: Optional[str] = None,
               extra: Optional[dict] = None) -> dict:
        """
        Record a completed API call. Returns the recorded entry.

        If actual_cost_usd is None, it's computed from MODEL_PRICING.
        """
        if actual_cost_usd is None:
            actual_cost_usd = estimate_cost(model, input_tokens, output_tokens)

        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': round(actual_cost_usd, 6),
            'deliverable': deliverable,
            'paper_id': paper_id,
        }
        if extra:
            entry.update(extra)

        with self._lock:
            self._run_total_usd += actual_cost_usd
            self._call_log.append(entry)
            self._history.append(entry)
            self._save_history()

        return entry

    # ---- Summary ----

    def summary(self) -> dict:
        """Return a snapshot of current usage state."""
        with self._lock:
            return {
                'this_run': {
                    'calls': len(self._call_log),
                    'cost_usd': round(self._run_total_usd, 4),
                    'cap_usd': self._budget.get('per_run_usd', 0),
                },
                'today': {
                    'calls': self.get_call_count('day'),
                    'cost_usd': round(self.get_spent('day'), 4),
                    'cap_usd': self._budget.get('per_day_usd', 0),
                },
                'this_month': {
                    'calls': self.get_call_count('month'),
                    'cost_usd': round(self.get_spent('month'), 4),
                    'cap_usd': self._budget.get('per_month_usd', 0),
                },
                'all_time': {
                    'calls': self.get_call_count('all_time'),
                    'cost_usd': round(self.get_spent('all_time'), 4),
                },
            }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        log_path = f.name

    tracker = UsageTracker(
        budget={'per_run_usd': 1.0, 'per_day_usd': 2.0, 'per_month_usd': 10.0,
                'hard_stop_at_pct': 100, 'warn_at_pct': 70},
        log_path=log_path,
    )
    
    print('=== usage_tracker self-test ===')
    
    # First call
    tracker.check_can_spend(0.10)
    tracker.record(model='claude-opus-4-6', input_tokens=5000, output_tokens=1000,
                   deliverable='manuscript', paper_id='A1')
    print(f"  Spent so far: ${tracker.get_spent('run'):.4f}")
    
    # Second call
    tracker.check_can_spend(0.15)
    tracker.record(model='claude-opus-4-6', input_tokens=8000, output_tokens=1500,
                   deliverable='press_release', paper_id='A1')
    print(f"  After 2 calls: ${tracker.get_spent('run'):.4f}")

    # Hit warning threshold
    tracker.record(model='claude-opus-4-6', input_tokens=20000, output_tokens=4000,
                   deliverable='references', paper_id='A1')
    print(f"  Warning threshold? {tracker.warning_threshold_reached()}")

    # Try to exceed budget
    try:
        tracker.check_can_spend(2.0)
    except BudgetExceededError as e:
        print(f"  ✓ Caught budget exceeded: {e}")

    # Summary
    summary = tracker.summary()
    print(f"  Summary: {json.dumps(summary, indent=2)}")
    
    Path(log_path).unlink()
    print('  ✓ self-test complete')
