# =============================================================================
# Ring Reconnection Fusion — reproducible runtime container (v3.1)
#
# v3.1 CHANGE (vs v3): The smoke test no longer imports pandas or yt. These
# packages are not pinned in environment_cuda_build.yml, so they aren't
# guaranteed present in the conda-pack tarball. They were also not part of
# the v1.0.0 failure mode this smoke test is meant to catch. The test now
# imports only the libraries explicitly pinned in the env file: numpy,
# scipy, matplotlib, h5py, mpi4py. Plus amrex.space2d, amrex.space3d, and
# pywarpx, which are the actual v1.0.0-defect detectors.
#
# v3 CHANGE (vs v2): The smoke test no longer imports amrex.space2d and
# amrex.space3d in the same Python process. pybind11 registers an "AMReX"
# type for each module at import time, and importing both in one process
# triggers an ImportError: generic_type: type "AMReX" is already registered.
# This was discovered during the v1.1.0 rebuild — the build itself works
# fine, only the test methodology was wrong. We now run each amrex.space*
# import in its own python process, which mirrors how real WarpX simulations
# load AMReX (one dimensionality per process).
#
# v2 FIXES (post-mortem of v1.0.0):
#   - Smoke test now imports amrex.space2d and amrex.space3d directly, which
#     forces the AMReX core .so to load. The v1 test used `import pywarpx`,
#     which loaded the Python layer but never touched the AMReX libraries —
#     so a build that produced bindings without libraries (the actual v1.0.0
#     failure) passed v1's test. The v2 test fails at docker-build time if
#     libamrex_*.so are missing.
#
#   - Base image left at ubuntu:22.04 because the rebuilt WarpX is now
#     produced with conda-forge's cross-compiler (glibc 2.17 sysroot), so
#     the resulting binaries are portable to 22.04's glibc 2.35. The v1
#     image required GLIBC_2.38 which 22.04 doesn't have; v2's binaries
#     require only GLIBC_2.17, which every supported Ubuntu provides.
#
# Author: James B. Worth
# License: Apache-2.0
# =============================================================================

FROM ubuntu:22.04

# -----------------------------------------------------------------------------
# Minimal host packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Unpack the pre-built conda env (plasma_env.tar.gz produced by conda-pack)
# -----------------------------------------------------------------------------
RUN mkdir -p /opt/conda/envs/plasma

COPY plasma_env.tar.gz /tmp/plasma_env.tar.gz

RUN tar -xzf /tmp/plasma_env.tar.gz -C /opt/conda/envs/plasma \
    && rm /tmp/plasma_env.tar.gz \
    && /opt/conda/envs/plasma/bin/python /opt/conda/envs/plasma/bin/conda-unpack

ENV PATH=/opt/conda/envs/plasma/bin:$PATH
ENV CONDA_DEFAULT_ENV=plasma
ENV CONDA_PREFIX=/opt/conda/envs/plasma

# -----------------------------------------------------------------------------
# v3 SMOKE TEST: import the AMReX core libraries directly, BUT in separate
# Python processes for 2D and 3D.
#
# This is the critical fix from v1. The v1 test was `import pywarpx`, which
# loads only the Python wrapper and DOES NOT cause the AMReX shared library
# to be dlopen()ed. The v1.0.0 broken image passed that test despite missing
# libamrex_2d.so entirely. By importing amrex.space2d/3d here, we force the
# dynamic linker to resolve libamrex_*.so right at docker-build time. If
# they're missing, the build fails here, not at first GPU run.
#
# v3 refinement: 2D and 3D imports MUST be in separate python processes
# because pybind11 registers a type named "AMReX" per module at import time.
# Loading both in one process collides:
#   ImportError: generic_type: type "AMReX" is already registered!
# This is not a build defect — real WarpX simulations only load one
# dimensionality per process. The smoke test now matches real usage.
# -----------------------------------------------------------------------------
RUN python -c "import sys, amrex.space2d as amr2; print('  Python:        ', sys.version.split()[0]); print('  amrex.space2d: ', amr2.__file__)" \
 && python -c "import amrex.space3d as amr3; print('  amrex.space3d: ', amr3.__file__)" \
 && python -c "import pywarpx; from pywarpx import picmi; print('  pywarpx:       ', getattr(pywarpx, '__version__', 'imported'))" \
 && python -c "import numpy, scipy, matplotlib, h5py, mpi4py; print('  numpy/scipy:   ', numpy.__version__, '/', scipy.__version__); print('  smoke test PASSED (AMReX cores loaded — v1.0.0 defect would fail here)')" \
 && python -c "import glob; libs = sorted(glob.glob('/opt/conda/envs/plasma/lib/libamrex*.so*')); print('  libamrex core libraries in env:'); [print('   ', l) for l in libs] if libs else (print('  ERROR: no libamrex*.so found') or exit(1))"

# -----------------------------------------------------------------------------
# Copy simulation code into the container
# -----------------------------------------------------------------------------
WORKDIR /work

RUN mkdir -p /work/simulation /work/analysis_scripts

COPY simulation/pb11_ring_reconnection_v15_pulsed.py /work/simulation/
COPY simulation/run_with_analysis.sh /work/simulation/
COPY analysis_scripts/ /work/analysis_scripts/

COPY program_config.yaml /work/
COPY program_config_paper1.yaml /work/

COPY setup_cloud.sh /work/
COPY build_warpx_cuda.sh /work/

COPY LICENSE NOTICE CITATION.cff README.md /work/

RUN mkdir -p /work/runs

CMD ["/bin/bash"]
