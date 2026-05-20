# =============================================================================
# Ring Reconnection Fusion — reproducible runtime container
#
# Uses conda-pack to capture the exact production conda env from the H100 cloud,
# bypassing the conda solve / source-build chain entirely. The env tarball is
# byte-identical to the validated cloud env, including CUDA-enabled pywarpx.
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

# Make the env's bin the default PATH; activate the env for all subsequent steps
ENV PATH=/opt/conda/envs/plasma/bin:$PATH
ENV CONDA_DEFAULT_ENV=plasma
ENV CONDA_PREFIX=/opt/conda/envs/plasma

# -----------------------------------------------------------------------------
# Smoke test: ensure the conda env unpacked cleanly and pywarpx imports
# -----------------------------------------------------------------------------
RUN python -c "import sys; \
import numpy, scipy, matplotlib, pandas, yt, h5py, mpi4py, pywarpx; \
print('  Python:', sys.version.split()[0]); \
print('  pywarpx:', getattr(pywarpx, '__version__', 'imported')); \
print('  numpy:', numpy.__version__); \
print('  scipy:', scipy.__version__); \
print('  smoke test PASSED')"

# -----------------------------------------------------------------------------
# Copy simulation code into the container
# -----------------------------------------------------------------------------
WORKDIR /work

RUN mkdir -p /work/simulation /work/analysis_scripts

# Simulation core
COPY simulation/pb11_ring_reconnection_v15_pulsed.py /work/simulation/
COPY simulation/run_with_analysis.sh /work/simulation/
COPY analysis_scripts/ /work/analysis_scripts/

# Program configs
COPY program_config.yaml /work/
COPY program_config_paper1.yaml /work/

# Helper scripts
COPY setup_cloud.sh /work/
COPY build_warpx_cuda.sh /work/

# Legal / metadata
COPY LICENSE NOTICE CITATION.cff README.md /work/

# Mountpoint for runs/ — host can bind-mount over this
RUN mkdir -p /work/runs

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
CMD ["/bin/bash"]
