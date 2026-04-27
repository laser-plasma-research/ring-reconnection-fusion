"""
progress.py — Unified progress display for the orchestrator.

Owns the terminal. Renders pipeline status across all papers showing:
  - Per-paper status: simulation / analysis / drafting columns
  - Active resource pools
  - API usage tracking
  - Recent events feed

Also writes a persistent progress_report.txt file that's tail-f-able.

Design:
  - Single rendering thread (daemon) refreshes display every 5s
  - State updates via thread-safe set methods
  - Box-drawing characters for clarity (matches existing pb11 report style)
"""

import threading
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STAGE_STATUS_PENDING = 'pending'
STAGE_STATUS_QUEUED = 'queued'
STAGE_STATUS_RUNNING = 'running'
STAGE_STATUS_DONE = 'done'
STAGE_STATUS_FAILED = 'failed'
STAGE_STATUS_SKIPPED = 'skipped'
STAGE_STATUS_NA = 'na'   # not applicable for this paper

STAGE_GLYPHS = {
    STAGE_STATUS_PENDING: '⏸',
    STAGE_STATUS_QUEUED:  '⏸',
    STAGE_STATUS_RUNNING: '⟳',
    STAGE_STATUS_DONE:    '✓',
    STAGE_STATUS_FAILED:  '✗',
    STAGE_STATUS_SKIPPED: '⊘',
    STAGE_STATUS_NA:      '–',
}


# ---------------------------------------------------------------------------
# Per-paper state
# ---------------------------------------------------------------------------

class PaperPipelineState:
    """Tracks a single paper's progression through the three stages."""

    def __init__(self, paper_id: str, paper_num: Optional[int],
                 tag: str, label: str,
                 has_simulation: bool, has_analysis: bool,
                 deliverable_names: list):
        self.paper_id = paper_id
        self.paper_num = paper_num
        self.tag = tag
        self.label = label

        self.sim_status = STAGE_STATUS_PENDING if has_simulation else STAGE_STATUS_NA
        self.ana_status = STAGE_STATUS_PENDING if has_analysis else STAGE_STATUS_NA

        # Drafting: dict of deliverable name -> status
        self.deliverable_states = {n: STAGE_STATUS_PENDING for n in deliverable_names}

        self.sim_start = None
        self.sim_end = None
        self.ana_start = None
        self.ana_end = None
        self.draft_start = None
        self.draft_end = None

        # Sub-job tracking for papers with multiple simulations
        self.sub_job_total = 0
        self.sub_job_done = 0
        self.sub_job_failed = 0

    def set_sim(self, status: str):
        self.sim_status = status
        if status == STAGE_STATUS_RUNNING and self.sim_start is None:
            self.sim_start = time.time()
        if status in (STAGE_STATUS_DONE, STAGE_STATUS_FAILED, STAGE_STATUS_SKIPPED):
            self.sim_end = time.time()

    def set_ana(self, status: str):
        self.ana_status = status
        if status == STAGE_STATUS_RUNNING and self.ana_start is None:
            self.ana_start = time.time()
        if status in (STAGE_STATUS_DONE, STAGE_STATUS_FAILED, STAGE_STATUS_SKIPPED):
            self.ana_end = time.time()

    def set_deliverable(self, name: str, status: str):
        self.deliverable_states[name] = status
        if status == STAGE_STATUS_RUNNING and self.draft_start is None:
            self.draft_start = time.time()
        # All deliverables done?
        if all(s in (STAGE_STATUS_DONE, STAGE_STATUS_FAILED, STAGE_STATUS_SKIPPED)
               for s in self.deliverable_states.values()):
            if self.draft_end is None:
                self.draft_end = time.time()

    @property
    def has_drafting(self) -> bool:
        return bool(self.deliverable_states)

    @property
    def deliverables_done(self) -> int:
        return sum(1 for s in self.deliverable_states.values()
                   if s in (STAGE_STATUS_DONE, STAGE_STATUS_FAILED, STAGE_STATUS_SKIPPED))

    @property
    def deliverables_total(self) -> int:
        return len(self.deliverable_states)

    @property
    def deliverables_running(self) -> list:
        return [n for n, s in self.deliverable_states.items()
                if s == STAGE_STATUS_RUNNING]

    def set_sub_job_progress(self, total: int, done: int, failed: int):
        self.sub_job_total = total
        self.sub_job_done = done
        self.sub_job_failed = failed

    # --- Render helpers ---

    def sim_summary(self) -> str:
        if self.sim_status == STAGE_STATUS_NA:
            return '–'
        glyph = STAGE_GLYPHS[self.sim_status]
        if self.sim_status == STAGE_STATUS_RUNNING and self.sim_start:
            elapsed = _format_elapsed(time.time() - self.sim_start)
            sub = ''
            if self.sub_job_total:
                sub = f' [{self.sub_job_done}/{self.sub_job_total}]'
            return f'{glyph} {elapsed}{sub}'
        if self.sim_status == STAGE_STATUS_DONE and self.sim_start and self.sim_end:
            return f'{glyph} {_format_elapsed(self.sim_end - self.sim_start)}'
        return f'{glyph} {self.sim_status}'

    def ana_summary(self) -> str:
        if self.ana_status == STAGE_STATUS_NA:
            return '–'
        glyph = STAGE_GLYPHS[self.ana_status]
        if self.ana_status == STAGE_STATUS_RUNNING and self.ana_start:
            return f'{glyph} {_format_elapsed(time.time() - self.ana_start)}'
        if self.ana_status == STAGE_STATUS_DONE and self.ana_start and self.ana_end:
            return f'{glyph} {_format_elapsed(self.ana_end - self.ana_start)}'
        return f'{glyph} {self.ana_status}'

    def draft_summary(self) -> str:
        if not self.has_drafting:
            return '–'
        done = self.deliverables_done
        total = self.deliverables_total
        running = self.deliverables_running
        if done == total and total > 0:
            return f'✓ {done}/{total}'
        if running:
            return f'⟳ {done}/{total}  {running[0]}' + (f' +{len(running)-1}' if len(running) > 1 else '')
        if done == 0:
            return f'⏸ 0/{total}'
        return f'⏸ {done}/{total}'


def _format_elapsed(seconds: float) -> str:
    """Return short human elapsed string e.g. '2h 14m', '34m', '12s'."""
    if seconds < 60:
        return f'{int(seconds)}s'
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f'{m}m {s:02d}s'
    h, rest = divmod(int(seconds), 3600)
    m = rest // 60
    return f'{h}h {m:02d}m'


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

class ProgressDisplay:
    """
    Owns the terminal output. Threadsafe state updates from worker threads;
    rendering on a single dedicated thread.
    """

    def __init__(self, papers: list, report_path: Optional[Path] = None,
                 refresh_interval: float = 5.0, quiet: bool = False):
        """
        papers: iterable of dicts with keys: paper_id, paper_num, tag, label,
                has_simulation, has_analysis, deliverable_names
        report_path: optional file to write tail-f-able progress
        quiet: if True, no terminal rendering (only file)
        """
        self._lock = threading.Lock()
        self._states = {}
        for p in papers:
            self._states[p['paper_id']] = PaperPipelineState(
                paper_id=p['paper_id'],
                paper_num=p.get('paper_num'),
                tag=p['tag'],
                label=p.get('label', ''),
                has_simulation=p.get('has_simulation', True),
                has_analysis=p.get('has_analysis', True),
                deliverable_names=p.get('deliverable_names', []),
            )
        self._events = deque(maxlen=8)
        self._start = time.time()
        self._mode_label = 'PRODUCTION'
        self._sim_pool_capacity = 0
        self._sim_pool_in_use = 0
        self._draft_pool_capacity = 0
        self._draft_pool_in_use = 0
        self._api_calls = 0
        self._api_spent_usd = 0.0
        self._api_budget_usd = 0.0
        self._refresh_interval = refresh_interval
        self._report_path = Path(report_path) if report_path else None
        self._quiet = quiet
        self._stop_event = threading.Event()
        self._thread = None
        self._first_render = True

    # ---- State updates (called from any thread) ----

    def set_mode(self, label: str):
        with self._lock:
            self._mode_label = label

    def set_pool_capacity(self, sim: int, draft: int):
        with self._lock:
            self._sim_pool_capacity = sim
            self._draft_pool_capacity = draft

    def set_pool_usage(self, sim_in_use: int, draft_in_use: int):
        with self._lock:
            self._sim_pool_in_use = sim_in_use
            self._draft_pool_in_use = draft_in_use

    def set_api_usage(self, calls: int, spent_usd: float, budget_usd: float):
        with self._lock:
            self._api_calls = calls
            self._api_spent_usd = spent_usd
            self._api_budget_usd = budget_usd

    def update_sim(self, paper_id: str, status: str):
        with self._lock:
            st = self._states.get(paper_id)
            if st:
                st.set_sim(status)

    def update_ana(self, paper_id: str, status: str):
        with self._lock:
            st = self._states.get(paper_id)
            if st:
                st.set_ana(status)

    def update_deliverable(self, paper_id: str, name: str, status: str):
        with self._lock:
            st = self._states.get(paper_id)
            if st:
                st.set_deliverable(name, status)

    def update_sub_jobs(self, paper_id: str, total: int, done: int, failed: int):
        with self._lock:
            st = self._states.get(paper_id)
            if st:
                st.set_sub_job_progress(total, done, failed)

    def add_event(self, msg: str):
        with self._lock:
            self._events.append((datetime.now().strftime('%H:%M:%S'), msg))

    # ---- Rendering ----

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name='ProgressDisplay')
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        # Final render to capture end state
        try:
            self._render_once()
        except Exception:
            pass

    def _render_loop(self):
        while not self._stop_event.is_set():
            try:
                self._render_once()
            except Exception as e:
                print(f"  WARN: render error: {e}", file=sys.stderr)
            self._stop_event.wait(timeout=self._refresh_interval)

    def _render_once(self):
        with self._lock:
            text = self._build_text()

        if self._report_path:
            try:
                self._report_path.parent.mkdir(parents=True, exist_ok=True)
                self._report_path.write_text(text)
            except Exception:
                pass

        if not self._quiet:
            # Clear-screen redraw
            if self._first_render:
                self._first_render = False
            else:
                # ANSI clear screen + cursor home
                sys.stdout.write('\x1b[2J\x1b[H')
            sys.stdout.write(text)
            sys.stdout.flush()

    def _build_text(self) -> str:
        elapsed_s = time.time() - self._start
        elapsed = _format_elapsed(elapsed_s)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = []
        BAR = '═' * 92

        lines.append(f'╔{BAR}╗')
        title = '  Ring-Reconnection Plasma Research Programme — Live Status  '
        lines.append(f'║{title:<{len(BAR)}}║')
        lines.append(f'║  Mode: {self._mode_label}'
                     f'    Elapsed: {elapsed:<10s}'
                     f'    Updated: {ts}'
                     f'{" " * 92}'[:92] + '║')
        lines.append(f'╠{BAR}╣')
        lines.append(f'║  Pipeline:                                                                        {" " * 9}║')

        header = f'  {"Paper":<6}│ {"Sim":<22}│ {"Analysis":<14}│ {"Deliverables":<35}'
        lines.append(f'║{header:<{len(BAR)}}║')
        lines.append(f'║  {"─"*6}─┼─{"─"*22}┼─{"─"*14}┼─{"─"*35}{" "*8}║')

        # Sort papers by paper_num then by id
        sorted_states = sorted(
            self._states.values(),
            key=lambda s: (s.paper_num if s.paper_num is not None else 999, s.paper_id))

        for st in sorted_states:
            row = f'  {st.paper_id:<6}│ {st.sim_summary():<22}│ {st.ana_summary():<14}│ {st.draft_summary():<35}'
            lines.append(f'║{row:<{len(BAR)}}║')

        lines.append(f'║{" " * len(BAR)}║')
        lines.append(f'╠{BAR}╣')

        # Resource pools
        lines.append(f'║  Resources:{" " * (len(BAR) - 13)}║')
        sim_use = f'{self._sim_pool_in_use}/{self._sim_pool_capacity}' if self._sim_pool_capacity else 'n/a'
        draft_use = f'{self._draft_pool_in_use}/{self._draft_pool_capacity}' if self._draft_pool_capacity else 'n/a'
        line = f'    Simulation pool: {sim_use:<8s}    Drafting pool: {draft_use:<8s}'
        lines.append(f'║{line:<{len(BAR)}}║')

        if self._api_budget_usd > 0:
            lines.append(f'║{" " * len(BAR)}║')
            api_line = (f'    API: {self._api_calls} calls, '
                        f'${self._api_spent_usd:.2f} spent / ${self._api_budget_usd:.2f} budget')
            lines.append(f'║{api_line:<{len(BAR)}}║')

        # Recent events
        if self._events:
            lines.append(f'╠{BAR}╣')
            lines.append(f'║  Recent events:{" " * (len(BAR) - 17)}║')
            for ts_str, msg in list(self._events)[-6:]:
                event_line = f'    [{ts_str}] {msg}'
                # Truncate long events
                if len(event_line) > len(BAR) - 2:
                    event_line = event_line[:len(BAR) - 5] + '...'
                lines.append(f'║{event_line:<{len(BAR)}}║')

        lines.append(f'╠{BAR}╣')
        lines.append(f'║  Press Ctrl-C for graceful shutdown (saves state for resume){" " * (len(BAR) - 65)}║')
        lines.append(f'╚{BAR}╝')

        return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    papers = [
        {'paper_id': 'A1', 'paper_num': 1, 'tag': 'p1_static', 'label': 'Static',
         'has_simulation': True, 'has_analysis': True,
         'deliverable_names': ['manuscript', 'press_release', 'lay_summary']},
        {'paper_id': 'A2', 'paper_num': 2, 'tag': 'p2_rotating', 'label': 'Rotating',
         'has_simulation': True, 'has_analysis': True,
         'deliverable_names': ['manuscript', 'press_release']},
    ]
    pd = ProgressDisplay(papers, refresh_interval=1.0)
    pd.set_mode('TEST DRY-RUN')
    pd.set_pool_capacity(sim=2, draft=4)

    pd.update_sim('A1', STAGE_STATUS_RUNNING)
    pd.set_pool_usage(sim_in_use=1, draft_in_use=0)
    pd.add_event('A1 simulation started')

    pd.start()
    time.sleep(2)

    pd.update_sim('A1', STAGE_STATUS_DONE)
    pd.update_ana('A1', STAGE_STATUS_RUNNING)
    pd.add_event('A1 simulation done; analysis started')

    time.sleep(2)
    pd.update_ana('A1', STAGE_STATUS_DONE)
    pd.update_deliverable('A1', 'manuscript', STAGE_STATUS_RUNNING)
    pd.add_event('A1 manuscript drafting started')

    time.sleep(2)
    pd.stop()
    print('\n  self-test complete')
