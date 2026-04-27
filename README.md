# Ring-Reconnection Plasma Research

Hybrid particle-in-cell (PIC) simulations of laser-driven plasma physics with a focus on ring-geometry magnetic reconnection. This repository contains the simulation pipeline, analysis tools, and orchestration infrastructure for an independent multi-paper research programme.

> **Status:** Active development. Simulation pipeline operational; AI-assisted drafting layer under construction (Step 3 complete, Step 4 in progress). No peer-reviewed papers published yet from this repo. See [Project status](#project-status) below for a realistic assessment.

## Research focus

The simulations explore a class of laser-target geometries in which a small number of laser spots in a ring arrangement create plasma plumes whose Biermann-battery magnetic fields form alternating-polarity X-lines between adjacent spots. Magnetic reconnection at these X-lines accelerates ions through the reconnection electric field. The research questions concern:

- Whether reconnection acceleration produces a non-thermal ion tail with energies relevant to fusion cross-sections
- How the geometry, drive parameters, and plasma conditions affect reconnection dynamics
- Whether continuous rotational modulation of the drive provides a measurable enhancement compared with static drive
- The applicability of these geometries to neutron sources (D-D), aneutronic fusion fuels (p-¹¹B), and adjacent applications

The fuels under study include both deuterium (D-D, neutron-producing) and proton-boron-11 (p-¹¹B, aneutronic). The aneutronic case is well known to face severe bremsstrahlung losses in any thermal plasma; the question this work investigates is whether non-thermal acceleration mechanisms can sidestep that limit at relevant scales. The answer remains open and the simulations alone cannot settle it.

## Repository contents

```
.
├── simulation/                  # PIC simulation pipeline (WarpX hybrid-PIC)
│   ├── pb11_ring_reconnection_v12_fuel_center_outer.py    Main simulation script
│   ├── pb11_run_all_papers.py                              Multi-job runner
│   ├── analyse_pb11.py                                      Image-based analysis
│   ├── pb11_analyse_rotating.py                             Rotating-case analysis
│   └── pb11_text_analysis.py                                Text diagnostic output
│
├── orchestrator/                # Programme orchestration (Python package)
│   ├── config.py                YAML loading and validation
│   ├── progress.py              Unified terminal display
│   ├── shutdown.py              Clean shutdown coordinator
│   ├── thread_pools.py          Bounded concurrency for drafting tasks
│   ├── visibility.py            Path classification and gitignore generation
│   └── deliverable_handlers/    Per-deliverable handlers (Step 3 stubs; Step 4 real)
│
├── tools/                       # Developer utilities
│   ├── pb11_debug.py            Short diagnostic runs
│   ├── find_bfield_method.py    pywarpx API discovery
│   └── patch_runner.py          macOS MPI exit-code patch
│
├── docs/                        # Project documentation
│   └── orchestrator_setup.md    Orchestrator developer guide
│
├── shared/                      # Local shared resources (gitignored)
│   └── .env                     API credentials (NEVER committed)
│
├── runs/                        # Simulation outputs (gitignored, large)
├── papers/                      # Per-paper output directories (gitignored or selectively tracked)
│
├── run_all.py                   # Orchestrator entry point
├── program_config.yaml          # Programme configuration (papers, deliverables, journals)
└── environment.yml              # Conda environment specification
```

## Computational environment

Simulations target the [WarpX](https://github.com/ECP-WarpX/WarpX) hybrid-PIC solver via its `pywarpx` PICMI Python interface. Development and testing is on Apple M2 Max (8 performance cores, 96 GB unified memory). Production runs scale to GPU-equipped cloud nodes when warranted.

The conda environment specification (`environment.yml`) pins all dependencies including WarpX (AMReX 26.04), Open MPI, and the analysis stack (numpy, scipy, openpmd-viewer, h5py).

## Quick start

Recreate the environment and run a short test:

```bash
# Create the conda environment
conda env create -f environment.yml
conda activate plasma

# Run a smoke-test simulation in dry-run mode (no actual MPI execution)
python3 simulation/pb11_run_all_papers.py --papers 1 --test --dry-run
```

For the full programme orchestrator (currently produces stub deliverables):

```bash
# Single paper, simulation pipeline only
python3 run_all.py --paper A1 --no-drafting --test

# Stub deliverables for one paper (no API calls)
python3 run_all.py --paper A1 --stages deliverables
```

See `docs/orchestrator_setup.md` for the full orchestrator reference.

## Project status

This is honest accounting of where the work currently stands:

**What works**
- Hybrid-PIC simulation pipeline runs end-to-end on a single workstation
- Ring-geometry magnetic reconnection signatures observed in 2D simulations
- Multi-paper job orchestration (sub-job dependencies, resume, parallel execution)
- Visibility-aware output handling separating public from internal content

**What is preliminary**
- 2D simulations only; 3D scaling not yet validated
- Reconnection-driven non-thermal ion tail observed but not yet at full target energies in production runs
- Rotational drive implementation is currently approximated by static initial conditions; full dynamic injection in development
- No experimental validation exists for the specific geometries simulated

**What is not yet done**
- Peer-reviewed publication
- Independent reproduction of the simulations
- Cross-validation against established reconnection codes
- Cloud-GPU runs at higher resolution

The intent of this repository is to make the work reproducible and open as it progresses, not to advance claims that are not yet substantiated.

## Author and affiliation

**James B. Worth** — Founder & CTO, [Substrate AI](https://substrate.ai), Valencia, Spain.
ORCID: [0009-0005-5000-9497](https://orcid.org/0009-0005-5000-9497).
Contact: bworth@substrate.ai

This research is conducted personally and independently of Substrate AI's commercial activities. The intellectual property and any resulting publications are individually held. Patent applications related to specific geometries and methods are pending.

## Licence

Code in this repository is released under the [Apache License 2.0](LICENSE). See `LICENSE` for full terms.

Simulation output data and any documentation released alongside papers is available under [Creative Commons Attribution 4.0 International (CC-BY 4.0)](DATA_LICENSE). See `DATA_LICENSE` for full terms.

For citation guidance, see [CITATION.cff](CITATION.cff).

## AI assistance disclosure

In the interest of full transparency, this repository was developed with substantial AI assistance:

- **Code authoring**: The orchestrator infrastructure (`orchestrator/` package, `run_all.py`) and modifications to the simulation runner (`simulation/pb11_run_all_papers.py`) were produced collaboratively with [Claude](https://claude.ai) (Anthropic). The author specified architecture, validated outputs, debugged integration issues, and made all design decisions; Claude generated the implementation code under direction.
- **Future drafting layer**: The `orchestrator/deliverable_handlers/` package (currently containing only stub handlers) is designed to use Claude to draft research artefacts (manuscripts, lay summaries, conference abstracts, IP analyses) from simulation outputs. Step 4 of the development plan will replace stubs with real Claude API calls. Any AI-drafted research content will be clearly marked and human-reviewed before any publication.
- **Simulation code**: The core PIC simulation script (`pb11_ring_reconnection_v12_fuel_center_outer.py`) was developed iteratively with AI assistance for debugging API quirks (pywarpx, AMReX, MPI on macOS) and physical-parameter validation. The physics specifications, parameter choices, and interpretations of results are the author's own.
- **README and documentation**: This README and the orchestrator documentation were drafted with AI assistance.

The author takes full responsibility for the scientific content, claims, and interpretations regardless of AI involvement in the underlying tools or text.

## Pending items

This repository is a work in progress. The following are recognised gaps to be addressed before this is a fit-for-purpose research artefact:

- README expansion once Step 4 (real AI drafting) lands
- Pre-commit hook validating no private content is staged for commit
- Linked paper preprints once available
- 3D simulation scaling validation
- Cross-validation against established reconnection codes

## Contact

For questions about the research or repository: bworth@substrate.ai. Issues and pull requests on this repository are welcome but please note that this is a single-researcher project; response times are best-effort.
