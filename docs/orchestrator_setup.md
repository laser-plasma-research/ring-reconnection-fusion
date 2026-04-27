# Orchestrator Setup Guide — Step 3

This bundle delivers the orchestrator skeleton for the Ring-Reconnection
Plasma Research programme. Step 3 ships the orchestration layer with stub
deliverable handlers (no API calls yet); Step 4 replaces stubs with real
Claude API integration.

## What's in the bundle

```
run_all.py                              # Main orchestrator entry point
orchestrator/
├── __init__.py
├── config.py                            # YAML loading and typed access
├── visibility.py                        # gitignore generation, path validation
├── shutdown.py                          # Clean shutdown coordinator
├── progress.py                          # Unified terminal display
├── thread_pools.py                      # Drafting pool (semaphore-bounded)
└── deliverable_handlers/
    ├── __init__.py
    └── base.py                          # Base class + StubHandler
```

Total: ~2400 lines across 9 files.

## Installation

Extract the tarball into your program root:

```bash
cd ~/LaserFusionResearch/research/laser-plasma-research
tar -xzf ~/Downloads/orchestrator_step3.tar.gz

# Verify structure
ls run_all.py orchestrator/
```

The orchestrator imports from `simulation/pb11_run_all_papers.py`, which
should already be in place from the migration. No additional dependencies
required (uses what's already in your `plasma` conda env plus `anthropic`).

## Smoke test

Verify the orchestrator imports cleanly and runs against your config:

```bash
# Dry-run a single paper (no actual simulations launched)
python3 run_all.py --paper A1 --no-drafting --test --dry-run

# Multiple papers, dry-run
python3 run_all.py --papers A1,A2 --no-drafting --test --dry-run

# Run only the deliverables stage (uses stub handler — produces placeholder content)
python3 run_all.py --paper A1 --stages deliverables
```

After the deliverables run, you should see output files at:

```
papers/worth-paper-A1-static-pb11/paper/manuscript.tex          # stub
papers/worth-paper-A1-static-pb11/paper/references.bib          # stub
papers/worth-paper-A1-static-pb11/companion/public/press_release_en.md
papers/worth-paper-A1-static-pb11/ip_analysis/potential_claims.md
... etc.
```

Each file is clearly marked as a stub. Step 4 replaces these with real
AI-drafted content.

## CLI reference

```
Usage: python3 run_all.py [OPTIONS]

Paper selection (mutually exclusive):
  --paper ID                Single paper id (e.g. A11)
  --papers ID1,ID2,...      Comma-separated paper ids
  --all                     All papers in config (default)

Stage selection:
  --stages STAGES           Comma-separated: simulation,analysis,deliverables
                            Default: all three
  --no-drafting             Skip the deliverables stage
  --deliverables NAMES      Specific deliverable names only (when running deliverables)

Mode flags:
  --test                    Run simulations in --test mode (short)
  --dry-run                 Show what would happen, don't execute
  --preflight               Use cheap model + short outputs (Step 4)
  --resume                  Skip already-complete jobs

Resource controls:
  --max-parallel N          Max simultaneous simulations
  --mpi-ranks N             MPI ranks per simulation
  --draft-parallel N        Max concurrent drafting tasks (default 4)
  --memory-cooldown SEC     Wait between sim jobs (default 30)

Other:
  --config PATH             program_config.yaml path (default: ./program_config.yaml)
  --runs-dir PATH           Override runs root
  --skip-analysis           Skip analysis after simulation
  --quiet                   No terminal display (only writes report file)
```

## Common workflows

**Iterate on stub deliverables for one paper:**

```bash
python3 run_all.py --paper A1 --stages deliverables
```

**Run simulation pipeline only (no AI drafting):**

```bash
python3 run_all.py --paper A11 --no-drafting --test
```

**Full programme dry-run:**

```bash
python3 run_all.py --all --no-drafting --test --dry-run
```

**Resume an interrupted run:**

```bash
python3 run_all.py --resume
```

## What's working in Step 3

- ✓ YAML config loading and validation
- ✓ Paper selection by id, list, or --all
- ✓ Stage selection (simulation, analysis, deliverables)
- ✓ Direct integration with pb11_run_all_papers.py (imports, no subprocess)
- ✓ Callback hook fires when papers complete simulation+analysis
- ✓ Drafting pool with bounded concurrency (semaphore)
- ✓ Unified progress display (terminal + tail-f-able report file)
- ✓ Clean shutdown on Ctrl-C with bounded join timeouts
- ✓ Visibility classification and gitignore generation
- ✓ Stub handler produces placeholder outputs to all 12 deliverable types

## What's NOT yet working (Step 4)

- ✗ Real Claude API calls (StubHandler only)
- ✗ Citation discovery and verification (Crossref, OpenAlex)
- ✗ Per-deliverable content validation (forbidden/required) is heuristic only
- ✗ Budget tracking against API spend
- ✗ ORCID work record propagation

These are all intentionally deferred. Step 3 establishes the orchestration
layer; Step 4 plugs in the API integration and content validation.

## Progress report file

While running, the orchestrator writes a tail-f-able status file:

```bash
tail -f runs/orchestrator_progress.txt
```

This shows the same information as the terminal display, refreshed every
5 seconds. Useful when running long simulations in the background.

## Troubleshooting

**ImportError: No module named 'pb11_run_all_papers'**

The orchestrator expects `simulation/pb11_run_all_papers.py` relative to
`run_all.py`. Verify you're running from the program root, not from
`simulation/`.

**ImportError: No module named 'anthropic'**

Install: `pip install anthropic`. (Required for Step 4; not strictly
needed for Step 3 since stubs don't call the API.)

**Output goes to wrong location**

The orchestrator writes deliverables to
`papers/{repo_name}/{output_path}` per the YAML config. Make sure
`paths.program_root` in YAML matches your actual program root.

**Ctrl-C doesn't terminate cleanly**

First Ctrl-C triggers graceful shutdown (up to 15 seconds). Second Ctrl-C
forces immediate exit. If the orchestrator hangs on shutdown, that's a
bug — report it.

## Next steps

After verifying Step 3 works end-to-end, we proceed to Step 4 which adds:

1. `orchestrator/api/claude_client.py` — Anthropic SDK wrapper with retries and budget tracking
2. `orchestrator/api/crossref_client.py` — Citation verification
3. `orchestrator/api/openalex_client.py` — Citation discovery
4. `orchestrator/citation_engine.py` — Strategic citation selection
5. `orchestrator/content_validator.py` — Refined forbidden/required validation
6. Real implementations of all 12 deliverable handlers (replacing StubHandler)

Step 4 is approximately 5-7 days of focused work.
