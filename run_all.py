#!/usr/bin/env python3
"""
run_all.py — Orchestrator entry point for the Ring-Reconnection Plasma
Research programme.

Reads program_config.yaml. For each paper, runs the configured stages
(simulation, analysis, deliverables) with proper concurrency and clean
shutdown.

Direct integration with simulation/pb11_run_all_papers.py: imports its
execution functions and registers a callback to trigger drafting when
each paper completes.

Usage:
    # Run everything for all papers (default)
    python3 run_all.py

    # Specific paper
    python3 run_all.py --paper A11

    # Multiple papers
    python3 run_all.py --papers A1,A2,A11

    # Stage selection
    python3 run_all.py --paper A11 --stages simulation,analysis
    python3 run_all.py --paper A11 --stages deliverables   # uses existing data

    # Skip drafting layer
    python3 run_all.py --paper A11 --no-drafting

    # Specific deliverables only
    python3 run_all.py --paper A11 --stages deliverables --deliverables manuscript,references

    # Test/dry-run/preflight
    python3 run_all.py --paper A11 --test --dry-run
    python3 run_all.py --paper A11 --preflight   # cheap AI validation
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROGRAM_ROOT = Path(__file__).parent.resolve()
SIMULATION_DIR = PROGRAM_ROOT / 'simulation'
ORCHESTRATOR_DIR = PROGRAM_ROOT / 'orchestrator'

# Add simulation/ to sys.path so we can import pb11_run_all_papers
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

# Add program root to sys.path so we can import orchestrator/
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))


# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from orchestrator.config import load_config, ProgramConfig, PaperConfig, ConfigError
from orchestrator.progress import (
    ProgressDisplay,
    STAGE_STATUS_PENDING, STAGE_STATUS_QUEUED, STAGE_STATUS_RUNNING,
    STAGE_STATUS_DONE, STAGE_STATUS_FAILED, STAGE_STATUS_SKIPPED,
)
from orchestrator.shutdown import ShutdownCoordinator
from orchestrator.thread_pools import DraftingPool
from orchestrator.deliverable_handlers import StubHandler, get_handler
from orchestrator.visibility import write_gitignore

# Step 4 additions: API integration
from orchestrator.api.env_loader import load_dotenv
from orchestrator.api.usage_tracker import UsageTracker, BudgetExceededError
from orchestrator.api.claude_client import ClaudeClient, ApiError
from orchestrator.api.crossref_client import CrossrefClient
from orchestrator.api.openalex_client import OpenAlexClient
from orchestrator.citation_engine import CitationEngine


# ---------------------------------------------------------------------------
# Environment cleanup (--clean flag)
# ---------------------------------------------------------------------------

# Directories that --clean wipes. Defensive: only these specific names.
# Each must be relative to PROGRAM_ROOT.
CLEAN_TARGETS = ['runs', 'papers', 'simulation/runs']


def _handle_clean(args):
    """
    Wipe the directories listed in CLEAN_TARGETS before the orchestrator
    starts. Refuses to delete anything outside PROGRAM_ROOT. Prompts for
    confirmation unless --yes was given.

    Called from main() when --clean is on the CLI.
    """
    import shutil

    # Find what would be deleted and how much disk it consumes
    candidates = []
    for rel in CLEAN_TARGETS:
        path = (PROGRAM_ROOT / rel).resolve()
        # Defensive check: must still be under PROGRAM_ROOT after resolve
        try:
            path.relative_to(PROGRAM_ROOT.resolve())
        except ValueError:
            print(f"  REFUSING to delete {path} (outside program root)",
                  file=sys.stderr)
            continue
        if path.exists():
            try:
                size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_str = _human_size(size_bytes)
            except Exception:
                size_str = '?'
            candidates.append((rel, path, size_str))

    if not candidates:
        print("--clean: nothing to remove (all target directories already absent)")
        return

    # Show what will be deleted
    print()
    print("=" * 70)
    print("  --clean will REMOVE the following directories:")
    print("=" * 70)
    total_disk = 0
    for rel, path, size_str in candidates:
        print(f"    {rel:25s}  {size_str:>10s}    {path}")
    print("=" * 70)

    # Confirm unless --yes
    if not args.yes:
        try:
            answer = input("  Type 'yes' to confirm: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Aborted.", file=sys.stderr)
            sys.exit(1)
        if answer != 'yes':
            print("  Aborted.", file=sys.stderr)
            sys.exit(1)

    # Delete
    print()
    for rel, path, _ in candidates:
        try:
            shutil.rmtree(path)
            print(f"    ✓ removed {rel}")
        except Exception as e:
            print(f"    ✗ failed to remove {rel}: {e}", file=sys.stderr)
            sys.exit(1)
    print()


def _human_size(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_simulation_stage(
    cfg: ProgramConfig,
    selected_papers: list,
    args,
    display: ProgressDisplay,
    shutdown: ShutdownCoordinator,
    on_paper_complete=None,
) -> dict:
    """
    Drive simulation+analysis through pb11_run_all_papers.

    Imports the existing runner directly, sets quiet mode so it doesn't
    print to terminal, and registers a callback to fire when each
    paper's simulation+analysis completes.

    Returns dict mapping paper_id -> success bool.
    """
    import pb11_run_all_papers as pb11

    # Build job list from YAML
    use_test = args.test
    mpi_ranks = args.mpi_ranks or cfg.default_mpi_ranks
    runs_root = args.runs_dir or cfg.runs_root

    Path(runs_root).mkdir(parents=True, exist_ok=True)

    # The orchestrator drives simulation through pb11's existing main()
    # logic, but we can't easily call it directly because main() does a
    # lot of CLI parsing. Instead we replicate the essential pieces.

    all_jobs = pb11.make_jobs_from_yaml(
        config_path=str(cfg.config_path),
        runs=runs_root,
        mpi_ranks=mpi_ranks,
        use_test=use_test,
    )

    # Filter to only papers we're running
    selected_ids = {p.id for p in selected_papers}
    all_jobs = [j for j in all_jobs if j.get('paper_id') in selected_ids]

    if args.dry_run:
        display.add_event(f'Dry-run mode — would execute {len(all_jobs)} jobs')
        for j in all_jobs:
            paper_id = j.get('paper_id', 'unknown')
            tag = j.get('tag', 'unknown')
            display.update_sim(paper_id, STAGE_STATUS_DONE)
            display.update_ana(paper_id, STAGE_STATUS_DONE)
            display.add_event(f'  [DRY] {paper_id}/{tag}')
            if on_paper_complete:
                on_paper_complete(j)
        return {p.id: True for p in selected_papers}

    # Quiet mode — pb11 stops printing to terminal
    pb11.set_quiet_mode(True)

    # Initialise GPU pool inside pb11 if --gpus-per-host > 0. Each parallel
    # simulation job will be pinned to a distinct GPU via CUDA_VISIBLE_DEVICES.
    gpus = getattr(args, 'gpus_per_host', 0) or 0
    if gpus > 0:
        pb11._gpu_pool = pb11._GPUPool(gpus)
        display.add_event(f'GPU pool: {gpus} GPUs (CUDA_VISIBLE_DEVICES pinning active)')

    # Register callback for paper completion
    if on_paper_complete:
        pb11.set_on_complete_callback(on_paper_complete)

    # Build tracker and progress report (from pb11 module).
    #
    # PROBLEM: pb11.ProgressTracker auto-paints a fullscreen dashboard
    # every 5 seconds via os.system('clear') + print, which clobbers the
    # orchestrator's own dashboard. The orchestrator owns the display.
    #
    # SOLUTION: Use a no-op tracker shim with the same interface that
    # pb11.run_job calls (start, stop, update, _render). This keeps state
    # tracking working while suppressing the screen-clearing dashboard.
    ordered_jobs = pb11.resolve_order(all_jobs)

    class _QuietTracker:
        """Drop-in replacement for pb11.ProgressTracker that doesn't render."""
        def __init__(self, jobs):
            self.jobs = jobs
            self.lock = threading.Lock()
            self.start_t = time.time()
        def start(self):
            pass   # no-op — orchestrator owns the display
        def stop(self):
            pass
        def update(self, tag, **kwargs):
            with self.lock:
                for j in self.jobs:
                    if j['tag'] == tag:
                        j.update(kwargs)
                        break
        def _render(self, final=False):
            pass

    import threading
    tracker = _QuietTracker(ordered_jobs)
    tracker.start()   # no-op, but keeps interface symmetric

    # Wire pb11's progress to our display by registering a callback...
    # but pb11 doesn't have such a hook. We'll let pb11 write to its
    # own report file under runs/preflight/ or runs/, and we update
    # our display from the on_complete callback only.

    # Run jobs. pb11.run_job is per-job; we need to drive concurrency.
    # The simplest path: import the run_job_pool inline-defined in
    # pb11.main(). But that's not exposed. Re-implement here.

    semaphore = threading.Semaphore(args.max_parallel or cfg.default_max_parallel)
    threads = {}
    completed = set()
    failed_set = set()

    def make_runner(job):
        def run():
            try:
                ok = pb11.run_job(
                    job=job,
                    semaphore=semaphore,
                    tracker=tracker,
                    runs_root=runs_root,
                    skip_analysis=args.skip_analysis,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    memory_cooldown=args.memory_cooldown,
                    progress_report=None,  # we own the display
                )
                if ok:
                    completed.add(job['tag'])
                else:
                    failed_set.add(job['tag'])
            except Exception as e:
                failed_set.add(job['tag'])
                display.add_event(f'  EXCEPTION in {job["tag"]}: {e}')

        return threading.Thread(target=run, daemon=True,
                                name=f'sim-{job["tag"]}')

    # Track per-paper sub-job progress
    paper_sub_counts = {}
    for j in ordered_jobs:
        pid = j.get('paper_id', 'unknown')
        paper_sub_counts.setdefault(pid, {'total': 0, 'done': 0, 'failed': 0})
        paper_sub_counts[pid]['total'] += 1

    for pid, counts in paper_sub_counts.items():
        display.update_sub_jobs(pid, counts['total'], 0, 0)
        display.update_sim(pid, STAGE_STATUS_QUEUED)

    # Update display state when sub-jobs complete
    completed_count = 0

    def update_paper_status():
        nonlocal completed_count
        for pid in paper_sub_counts:
            paper_jobs = [j for j in ordered_jobs if j.get('paper_id') == pid]
            done_now = sum(1 for j in paper_jobs if j['tag'] in completed)
            failed_now = sum(1 for j in paper_jobs if j['tag'] in failed_set)
            paper_sub_counts[pid]['done'] = done_now
            paper_sub_counts[pid]['failed'] = failed_now
            display.update_sub_jobs(pid, paper_sub_counts[pid]['total'],
                                    done_now, failed_now)

            if done_now + failed_now == paper_sub_counts[pid]['total']:
                if failed_now == 0:
                    display.update_sim(pid, STAGE_STATUS_DONE)
                    display.update_ana(pid, STAGE_STATUS_DONE)
                else:
                    display.update_sim(pid, STAGE_STATUS_FAILED)
            elif done_now > 0 or failed_now > 0:
                display.update_sim(pid, STAGE_STATUS_RUNNING)

    # Launch and monitor
    pending = list(ordered_jobs)
    running = []
    n_active_max = args.max_parallel or cfg.default_max_parallel

    while (pending or running or threads) and not shutdown.should_stop():
        # Resolve dependencies and launch eligible jobs
        for job in list(pending):
            if shutdown.should_stop():
                break
            if job.get('analysis_only'):
                # No simulation; mark its prerequisites met if deps complete
                deps_ok = all(d in completed for d in job.get('depends_on', []))
                if deps_ok:
                    completed.add(job['tag'])
                    pending.remove(job)
                    if on_paper_complete:
                        try:
                            on_paper_complete(job)
                        except Exception:
                            pass
                continue

            deps_ok = all(d in completed for d in job.get('depends_on', []))
            if not deps_ok:
                continue

            n_running = sum(1 for t in threads.values() if t.is_alive())
            if n_running >= n_active_max:
                break

            t = make_runner(job)
            threads[job['tag']] = t
            t.start()
            pending.remove(job)
            display.update_sim(job.get('paper_id', 'unknown'),
                               STAGE_STATUS_RUNNING)
            display.add_event(f'started {job["tag"]}')

        # Reap finished threads
        for tag in list(threads.keys()):
            if not threads[tag].is_alive():
                threads[tag].join(timeout=2.0)
                del threads[tag]
                update_paper_status()
                display.set_pool_usage(
                    sim_in_use=sum(1 for t in threads.values() if t.is_alive()),
                    draft_in_use=display._draft_pool_in_use,
                )

        time.sleep(1.0)

    # Final reaping
    for tag, t in list(threads.items()):
        t.join(timeout=5.0)

    update_paper_status()
    tracker.stop()

    return {p.id: (p.id not in failed_set) for p in selected_papers}


def run_deliverables_stage(
    cfg: ProgramConfig,
    paper: PaperConfig,
    args,
    display: ProgressDisplay,
    shutdown: ShutdownCoordinator,
    drafting_pool: DraftingPool,
    paper_repo_root: Path,
    claude_client=None,
    citation_engine=None,
    use_stubs: bool = False,
):
    """
    Submit drafting tasks for one paper to the drafting pool.

    Tasks are sequenced in two waves:
      Wave 1: manuscript (and any independent deliverables like metadata)
      Wave 2: deliverables that depend on manuscript content
              (references, lay_summary, press_release, IP analyses, etc.)

    The orchestrator polls for wave 1 completion before submitting wave 2.
    Tasks within a wave run concurrently up to the drafting pool capacity.
    """
    if not paper.has_deliverables:
        display.add_event(f'{paper.id}: no deliverables configured')
        return

    requested = args.deliverables.split(',') if args.deliverables else None
    runs_root = args.runs_dir or cfg.runs_root
    attorney_review = paper.requires_attorney_review

    # Classify each deliverable into its wave.
    # Wave 1 = independent (does not consume manuscript content).
    # Wave 2 = consumes manuscript content (must run after manuscript completes).
    WAVE_1_INDEPENDENT = {'manuscript', 'metadata'}
    # Everything else benefits from having a finished manuscript

    wave_1, wave_2 = [], []
    for name in paper.deliverable_names:
        if requested and name not in requested:
            continue
        if name in WAVE_1_INDEPENDENT:
            wave_1.append(name)
        else:
            wave_2.append(name)

    # Mark all queued
    for name in wave_1 + wave_2:
        display.update_deliverable(paper.id, name, STAGE_STATUS_QUEUED)

    # ---- Submit wave 1 ----
    wave_1_futures = []
    for name in wave_1:
        if shutdown.should_stop():
            break
        future = _submit_deliverable_task(
            paper, name, attorney_review, args, cfg, runs_root,
            paper_repo_root, claude_client, citation_engine, use_stubs,
            display, shutdown, drafting_pool,
        )
        wave_1_futures.append((name, future))

    # ---- Wait for wave 1 to complete (with shutdown awareness) ----
    if wave_2:   # Only wait if we have wave 2 work to do
        for name, future in wave_1_futures:
            if shutdown.should_stop():
                return
            try:
                future.result(timeout=300)   # 5-minute cap per deliverable
            except Exception as e:
                display.add_event(f'{paper.id}: wave 1 ({name}) error: {e}')

    # ---- Submit wave 2 ----
    for name in wave_2:
        if shutdown.should_stop():
            break
        _submit_deliverable_task(
            paper, name, attorney_review, args, cfg, runs_root,
            paper_repo_root, claude_client, citation_engine, use_stubs,
            display, shutdown, drafting_pool,
        )


def _submit_deliverable_task(paper, name, attorney_required, args, cfg, runs_root,
                              paper_repo_root, claude_client, citation_engine,
                              use_stubs, display, shutdown, drafting_pool):
    """Build a closure that runs one deliverable and submit to the pool."""
    def task():
        if shutdown.should_stop():
            return
        display.update_deliverable(paper.id, name, STAGE_STATUS_RUNNING)
        display.add_event(f'{paper.id}: drafting {name}')
        try:
            if use_stubs:
                HandlerCls = StubHandler
            else:
                HandlerCls = get_handler(name)

            handler = HandlerCls(
                paper=paper,
                deliverable_name=name,
                program_config=cfg,
                runs_root=runs_root,
                paper_repo_root=paper_repo_root,
                preflight=args.preflight,
                claude_client=claude_client,
                citation_engine=citation_engine,
                attorney_review_required=attorney_required,
            )
            output_path = handler.run()
            display.update_deliverable(paper.id, name, STAGE_STATUS_DONE)
            display.add_event(f'{paper.id}: {name} → {output_path.name}')
        except BudgetExceededError as e:
            display.update_deliverable(paper.id, name, STAGE_STATUS_FAILED)
            display.add_event(f'{paper.id}: {name} BUDGET EXCEEDED — {e}')
        except Exception as e:
            display.update_deliverable(paper.id, name, STAGE_STATUS_FAILED)
            display.add_event(f'{paper.id}: {name} FAILED — '
                               f'{type(e).__name__}: {e}')

    return drafting_pool.submit(f'{paper.id}/{name}', task)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Orchestrator for the Ring-Reconnection Plasma Research programme'
    )
    parser.add_argument('--config', type=str, default='program_config.yaml',
                        help='Path to program_config.yaml')
    parser.add_argument('--paper', type=str, default=None,
                        help='Single paper id (e.g. A11)')
    parser.add_argument('--papers', type=str, default=None,
                        help='Comma-separated paper ids (e.g. A1,A2,A11)')
    parser.add_argument('--all', action='store_true',
                        help='Process all papers in config')
    parser.add_argument('--stages', type=str, default='simulation,analysis,deliverables',
                        help='Comma-separated stages (simulation,analysis,deliverables). Default: all')
    parser.add_argument('--deliverables', type=str, default=None,
                        help='Comma-separated deliverable names (default: all configured for paper)')
    parser.add_argument('--no-drafting', action='store_true',
                        help='Skip the deliverables stage entirely')
    parser.add_argument('--preflight', action='store_true',
                        help='Use cheap model and short outputs (validate pipeline)')
    parser.add_argument('--test', action='store_true',
                        help='Run simulations in --test mode (short)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without executing')
    parser.add_argument('--resume', action='store_true',
                        help='Skip jobs whose outputs already exist')
    parser.add_argument('--max-parallel', type=int, default=None,
                        help='Max simultaneous simulations (default from config)')
    parser.add_argument('--gpus-per-host', type=int, default=0,
                        help='Number of GPUs available on this host (default: 0 = '
                             'no CUDA_VISIBLE_DEVICES pinning). Set to 2 on a 2-GPU '
                             'instance to pin each parallel job to a distinct GPU.')
    parser.add_argument('--mpi-ranks', type=int, default=None,
                        help='MPI ranks per simulation job (default from config)')
    parser.add_argument('--runs-dir', type=str, default=None,
                        help='Override runs root directory')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip analysis after simulation')
    parser.add_argument('--memory-cooldown', type=int, default=30,
                        help='Seconds between sim jobs (default: 30)')
    parser.add_argument('--draft-parallel', type=int, default=4,
                        help='Max concurrent drafting tasks (default: 4)')
    parser.add_argument('--quiet', action='store_true',
                        help='No terminal display (only writes to report file)')
    parser.add_argument('--use-stubs', action='store_true',
                        help='Force stub handlers (no API calls; for testing)')
    parser.add_argument('--env-file', type=str, default=None,
                        help='Path to .env file (default: shared/.env)')
    parser.add_argument('--clean', action='store_true',
                        help='Delete runs/, papers/, simulation/runs/ before '
                             'starting. Asks for confirmation unless --yes is given.')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Auto-confirm prompts (e.g. for --clean)')

    args = parser.parse_args()

    # ---- Handle --clean BEFORE loading config or doing anything else ----
    if args.clean:
        _handle_clean(args)

    # ---- Load config ----
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"ERROR loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Select papers ----
    if args.paper:
        if ',' in args.paper:
            paper_ids = [p.strip() for p in args.paper.split(',') if p.strip()]
        else:
            paper_ids = [args.paper]
    elif args.papers:
        paper_ids = [p.strip() for p in args.papers.split(',') if p.strip()]
    elif args.all:
        paper_ids = [p.id for p in cfg.papers]
    else:
        # Default behavior: all papers
        paper_ids = [p.id for p in cfg.papers]

    selected = []
    for pid in paper_ids:
        p = cfg.get_paper(pid)
        if p is None:
            print(f"ERROR: paper id {pid!r} not found in config", file=sys.stderr)
            sys.exit(1)
        selected.append(p)

    # ---- Parse stages ----
    requested_stages = set(s.strip() for s in args.stages.split(',') if s.strip())
    if args.no_drafting:
        requested_stages.discard('deliverables')

    valid_stages = {'simulation', 'analysis', 'deliverables'}
    invalid = requested_stages - valid_stages
    if invalid:
        print(f"ERROR: unknown stages: {invalid}. Valid: {valid_stages}",
              file=sys.stderr)
        sys.exit(1)

    # ---- Set up display ----
    runs_root = args.runs_dir or cfg.runs_root
    Path(runs_root).mkdir(parents=True, exist_ok=True)
    report_path = Path(runs_root) / 'orchestrator_progress.txt'

    paper_specs = []
    for p in selected:
        paper_specs.append({
            'paper_id': p.id,
            'paper_num': p.paper_num,
            'tag': p.tag,
            'label': p.label,
            'has_simulation': not p.is_analysis_only and 'simulation' in requested_stages,
            'has_analysis': 'analysis' in requested_stages,
            'deliverable_names': p.deliverable_names if 'deliverables' in requested_stages else [],
        })

    mode_label = []
    if args.test:
        mode_label.append('TEST')
    if args.preflight:
        mode_label.append('PREFLIGHT')
    if args.dry_run:
        mode_label.append('DRY-RUN')
    if args.no_drafting:
        mode_label.append('NO-DRAFTING')
    mode_str = ' / '.join(mode_label) or 'PRODUCTION'

    display = ProgressDisplay(
        papers=paper_specs,
        report_path=report_path,
        refresh_interval=5.0,
        quiet=args.quiet,
    )
    display.set_mode(mode_str)
    display.set_pool_capacity(
        sim=args.max_parallel or cfg.default_max_parallel,
        draft=args.draft_parallel,
    )
    display.add_event(f'Orchestrator starting: {len(selected)} paper(s), '
                      f'stages={sorted(requested_stages)}')

    # ---- Set up shutdown coordinator ----
    shutdown = ShutdownCoordinator(force_exit_after_seconds=15.0)
    shutdown.install_signal_handlers()

    # ---- Register pb11 simulation cleanup (PRIORITY 1 — runs first) ----
    # The orchestrator's signal handler overrides pb11_run_all_papers.py's own
    # signal handler. Without this cleanup callback, pb11's _stop_event never
    # gets set when Ctrl-C is pressed, so its polling loops keep running and
    # mpirun + 8 worker processes survive.
    #
    # This function calls into pb11's already-correct _kill_all_procs(), which:
    #   1. Sets _stop_event to unblock pb11's polling loops
    #   2. Sends SIGTERM to each tracked process group (kills mpirun + workers)
    #   3. Falls back to pkill by name for any orphans
    #   4. Sends SIGKILL after a grace period to anything still alive
    #
    # Priority 1 ensures this runs BEFORE drafting_pool (priority 8) and
    # progress_display (priority 10) so MPI dies before the rest of cleanup.
    def _stop_pb11_simulations():
        try:
            import pb11_run_all_papers as pb11
            pb11._kill_all_procs(signum=None, frame=None)
        except ImportError:
            pass
        except Exception as e:
            print(f"\n  WARN: pb11 cleanup encountered error: {e}",
                  file=sys.stderr, flush=True)

    shutdown.register_cleanup('pb11_simulations', _stop_pb11_simulations,
                              priority=1)

    # ---- Set up API integration (only if drafting is happening) ----
    claude_client = None
    citation_engine = None
    usage_tracker = None

    needs_api = ('deliverables' in requested_stages and not args.use_stubs
                 and not args.dry_run)
    if needs_api:
        # Load .env
        env_path = Path(args.env_file) if args.env_file else (cfg.shared_dir / '.env')
        if env_path.exists():
            try:
                load_dotenv(env_path, required_keys=['ANTHROPIC_API_KEY'])
                display.add_event(f'Loaded .env from {env_path}')
            except (FileNotFoundError, KeyError) as e:
                print(f"\nERROR loading .env: {e}", file=sys.stderr)
                print(f"  Either populate {env_path} with ANTHROPIC_API_KEY=...,", file=sys.stderr)
                print(f"  or use --use-stubs to skip API calls.\n", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"\nERROR: .env file not found at {env_path}", file=sys.stderr)
            print(f"  Use --use-stubs to skip API calls, or", file=sys.stderr)
            print(f"  create {env_path} with: ANTHROPIC_API_KEY=sk-ant-...\n", file=sys.stderr)
            sys.exit(1)

        # Set up usage tracker with budget caps
        budget_cfg = cfg.drafting.get('budget', {})
        usage_log_path = cfg.shared_dir / 'usage_log.json'
        usage_tracker = UsageTracker(budget=budget_cfg, log_path=str(usage_log_path))

        # Set up Claude client
        try:
            claude_client = ClaudeClient(
                usage_tracker=usage_tracker,
                default_model=cfg.default_model,
            )
        except ApiError as e:
            print(f"\nERROR: Cannot initialize Claude client: {e}", file=sys.stderr)
            sys.exit(1)

        # Set up citation engine
        contact_email = cfg.program.get('contact_email', 'no-contact@example.com')
        crossref_client = CrossrefClient(contact_email=contact_email)
        openalex_client = OpenAlexClient(contact_email=contact_email)
        citation_engine = CitationEngine(
            openalex_client=openalex_client,
            crossref_client=crossref_client,
        )

        # Show initial budget state
        summary = usage_tracker.summary()
        display.set_api_usage(
            calls=summary['this_run']['calls'],
            spent_usd=summary['this_run']['cost_usd'],
            budget_usd=summary['this_run']['cap_usd'],
        )
        display.add_event(
            f"API ready. Today: ${summary['today']['cost_usd']:.2f}/"
            f"${summary['today']['cap_usd']:.2f}, "
            f"Month: ${summary['this_month']['cost_usd']:.2f}/"
            f"${summary['this_month']['cap_usd']:.2f}"
        )

    # ---- Set up drafting pool ----
    drafting_pool = DraftingPool(
        max_workers=args.draft_parallel,
        on_task_start=lambda n: display.set_pool_usage(
            display._sim_pool_in_use, display._draft_pool_in_use + 1),
        on_task_done=lambda n, err: display.set_pool_usage(
            display._sim_pool_in_use,
            max(0, display._draft_pool_in_use - 1)),
    )
    shutdown.register_cleanup('drafting_pool',
                              drafting_pool.shutdown_with_timeout, priority=8)
    shutdown.register_cleanup('progress_display', display.stop, priority=10)

    # ---- Define paper completion callback ----
    def on_paper_complete(job):
        """Called by pb11 when a paper's simulation+analysis completes."""
        paper_id = job.get('paper_id')
        if not paper_id:
            return
        if 'deliverables' not in requested_stages:
            return
        paper = cfg.get_paper(paper_id)
        if paper is None or not paper.has_deliverables:
            return

        # Resolve paper repo root
        program_root = cfg.program_root if cfg.program_root.exists() else PROGRAM_ROOT
        paper_repo_root = program_root / 'papers' / paper.repo_name

        display.add_event(f'{paper_id}: simulation+analysis complete; '
                          f'queueing {len(paper.deliverable_names)} deliverables')

        run_deliverables_stage(
            cfg=cfg,
            paper=paper,
            args=args,
            display=display,
            shutdown=shutdown,
            drafting_pool=drafting_pool,
            paper_repo_root=paper_repo_root,
            claude_client=claude_client,
            citation_engine=citation_engine,
            use_stubs=args.use_stubs or args.dry_run,
        )

    # ---- Start display ----
    display.start()

    # ---- Run stages ----
    try:
        # Simulation+analysis stage
        if 'simulation' in requested_stages or 'analysis' in requested_stages:
            display.add_event('Stage: simulation/analysis')
            results = run_simulation_stage(
                cfg=cfg,
                selected_papers=selected,
                args=args,
                display=display,
                shutdown=shutdown,
                on_paper_complete=on_paper_complete,
            )
        else:
            # Deliverables-only path: trigger callback for each paper directly
            display.add_event('Stage: deliverables only (using existing simulation outputs)')
            for paper in selected:
                if shutdown.should_stop():
                    break
                # Build a synthetic "job" so the callback works
                fake_job = {'paper_id': paper.id, 'tag': paper.tag,
                            'paper_def': paper.raw}
                on_paper_complete(fake_job)

        # Wait for drafting tasks to complete (with shutdown awareness)
        if 'deliverables' in requested_stages and not shutdown.should_stop():
            display.add_event('Waiting for deliverables to complete...')
            while drafting_pool.in_flight > 0 and not shutdown.should_stop():
                time.sleep(2.0)
                # Update usage display
                if usage_tracker:
                    summary = usage_tracker.summary()
                    display.set_api_usage(
                        calls=summary['this_run']['calls'],
                        spent_usd=summary['this_run']['cost_usd'],
                        budget_usd=summary['this_run']['cap_usd'],
                    )
                    # Warn if budget threshold crossed
                    warn_scope = usage_tracker.warning_threshold_reached()
                    if warn_scope:
                        display.add_event(
                            f'WARN: API budget at warning threshold ({warn_scope})')

        # ---- Final summary ----
        display.add_event('All stages complete')
        time.sleep(1.0)   # let final render happen

    finally:
        # Clean shutdown
        if not shutdown.is_shutting_down():
            shutdown.shutdown_all(reason='completed')

        # Final render
        try:
            display.stop()
        except Exception:
            pass

        # Tally results
        n_sim_done = sum(1 for p in selected
                         if display._states[p.id].sim_status == STAGE_STATUS_DONE)
        n_sim_failed = sum(1 for p in selected
                           if display._states[p.id].sim_status == STAGE_STATUS_FAILED)
        n_draft_done = sum(display._states[p.id].deliverables_done
                           for p in selected
                           if display._states[p.id].has_drafting)
        n_draft_total = sum(display._states[p.id].deliverables_total
                            for p in selected
                            if display._states[p.id].has_drafting)

        print()
        print('=' * 70)
        print(f'  ORCHESTRATOR FINAL — {mode_str}')
        print(f'  Simulation: {n_sim_done} done, {n_sim_failed} failed '
              f'(of {len(selected)})')
        if n_draft_total:
            print(f'  Drafting:   {n_draft_done}/{n_draft_total} deliverables')
        if usage_tracker:
            summary = usage_tracker.summary()
            print(f'  API usage:')
            print(f'    This run:  ${summary["this_run"]["cost_usd"]:.4f} '
                  f'({summary["this_run"]["calls"]} calls)')
            print(f'    Today:     ${summary["today"]["cost_usd"]:.4f} / '
                  f'${summary["today"]["cap_usd"]:.2f} budget')
            print(f'    Month:     ${summary["this_month"]["cost_usd"]:.4f} / '
                  f'${summary["this_month"]["cap_usd"]:.2f} budget')
        print(f'  Report:     {report_path}')
        print('=' * 70)

        sys.exit(1 if n_sim_failed > 0 else 0)


if __name__ == '__main__':
    main()
