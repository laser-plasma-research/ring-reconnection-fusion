#!/usr/bin/env bash
# ============================================================================
# build_warpx_cuda.sh — compile WarpX from source with CUDA support
#
# Prerequisites:
#   - Ubuntu 22.04+ with NVIDIA driver supporting CUDA 12+
#   - 'plasma' conda env created from environment_cuda_build.yml
#   - At least 16 GB RAM and 16 GB free disk
#
# Usage:
#   conda activate plasma
#   bash build_warpx_cuda.sh
#
# Approximate runtime on 1× H100 instance (16 vCPU): 30-60 minutes.
# Most of the time is spent compiling AMReX + WarpX kernels for CUDA.
# ============================================================================

set -euo pipefail

# ── Where to clone and build ──────────────────────────────────────────────────
WARPX_REPO="${WARPX_REPO:-https://github.com/BLAST-WarpX/warpx.git}"
WARPX_REF="${WARPX_REF:-development}"     # or pin to a tag like '26.04'
WARPX_SRC="${WARPX_SRC:-$HOME/warpx-src}"
WARPX_BUILD="${WARPX_BUILD:-$WARPX_SRC/build}"

# Build options — can override via env vars
WARPX_DIMS="${WARPX_DIMS:-2;3}"            # build 2D and 3D variants
WARPX_PARALLEL_JOBS="${WARPX_PARALLEL_JOBS:-$(nproc)}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════════"
echo "  WarpX-CUDA build"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Repo:      $WARPX_REPO"
echo "  Ref:       $WARPX_REF"
echo "  Source:    $WARPX_SRC"
echo "  Build:     $WARPX_BUILD"
echo "  DIMS:      $WARPX_DIMS"
echo "  Jobs:      $WARPX_PARALLEL_JOBS"
echo

# Conda env active?
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "plasma" ]]; then
    echo "ERROR: Conda env 'plasma' is not active." >&2
    echo "       Run: conda activate plasma" >&2
    exit 1
fi
echo "  Active conda env: $CONDA_DEFAULT_ENV"

# Required tools
for tool in nvcc cmake make git mpicc python3; do
    if ! command -v "$tool" &>/dev/null; then
        echo "  ERROR: '$tool' not found in PATH" >&2
        echo "  Make sure environment_cuda_build.yml was installed correctly." >&2
        exit 1
    fi
done
echo "  Build tools: ✓"

# CUDA toolkit version
NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
echo "  nvcc:      $NVCC_VERSION"

# GPU detection (warn if missing)
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "  WARNING: no nvidia-smi — building on a host without GPU. Build will work, runs won't." >&2
fi

# ── Clone WarpX ────────────────────────────────────────────────────────────────
echo
echo "── Step 1: Clone WarpX source ────────────────────────────────────────────"
if [[ -d "$WARPX_SRC/.git" ]]; then
    echo "  Source dir exists, fetching latest..."
    cd "$WARPX_SRC"
    git fetch --all --quiet
    git checkout "$WARPX_REF" --quiet
    git pull --quiet || true
else
    echo "  Cloning $WARPX_REPO @ $WARPX_REF..."
    git clone --depth 1 --branch "$WARPX_REF" "$WARPX_REPO" "$WARPX_SRC" 2>/dev/null || \
        git clone "$WARPX_REPO" "$WARPX_SRC"
    cd "$WARPX_SRC"
    git checkout "$WARPX_REF" --quiet
fi
COMMIT=$(git rev-parse --short HEAD)
echo "  Checked out $COMMIT"

# ── Configure the build ───────────────────────────────────────────────────────
echo
echo "── Step 2: CMake configure (~1-2 min) ────────────────────────────────────"

# Wipe any prior build to avoid cached config conflicts
rm -rf "$WARPX_BUILD"

# Tell CMake where conda's CUDA lives so nvcc finds the libs
export CUDACXX=$(which nvcc)
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH:-}"

cmake -S "$WARPX_SRC" -B "$WARPX_BUILD" \
    -DWarpX_DIMS="$WARPX_DIMS" \
    -DWarpX_COMPUTE=CUDA \
    -DWarpX_PYTHON=ON \
    -DWarpX_LIB=ON \
    -DWarpX_MPI=ON \
    -DWarpX_OPENPMD=ON \
    -DWarpX_FFT=ON \
    -DWarpX_QED=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMReX_CUDA_ARCH=90 \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"

# AMReX_CUDA_ARCH=90 is H100 (Hopper). For other GPUs:
#   80 = A100 (Ampere)
#   86 = A40, RTX 3090 (Ampere)
#   89 = L40, L40S, RTX 4090 (Ada)
#   90 = H100, H200 (Hopper)

# ── Build ─────────────────────────────────────────────────────────────────────
echo
echo "── Step 3: Compile (30-60 min on $WARPX_PARALLEL_JOBS cores) ─────────────"
echo "  Output is verbose; expect long quiet stretches between progress lines."
echo

# Use pip_install target — this builds and installs pywarpx into conda env
cmake --build "$WARPX_BUILD" --target pip_install -j "$WARPX_PARALLEL_JOBS"

# ── Verify the install ────────────────────────────────────────────────────────
echo
echo "── Step 4: Verify pywarpx installed and CUDA-enabled ─────────────────────"

python -c "import pywarpx; print(f'  pywarpx imported OK')" || {
    echo "  ERROR: pywarpx import failed" >&2
    exit 1
}

# Check that the .so files have CUDA in their names
PYWARPX_DIR=$(python -c "import pywarpx, os; print(os.path.dirname(pywarpx.__file__))")
echo "  pywarpx location: $PYWARPX_DIR"
SO_FILES=$(ls "$PYWARPX_DIR"/*.so 2>/dev/null || true)
if [[ -z "$SO_FILES" ]]; then
    echo "  WARNING: no .so files found in $PYWARPX_DIR" >&2
fi
echo "  Library files:"
for f in $SO_FILES; do
    echo "    $(basename $f) ($(du -h "$f" | awk '{print $1}'))"
done

# Sanity check: try a tiny PICMI grid creation
python <<'PYEOF'
from pywarpx import picmi
g = picmi.Cartesian2DGrid(
    number_of_cells=[16, 16],
    lower_bound=[-1e-3, -1e-3],
    upper_bound=[ 1e-3,  1e-3],
    lower_boundary_conditions=['periodic']*2,
    upper_boundary_conditions=['periodic']*2,
    lower_boundary_conditions_particles=['periodic']*2,
    upper_boundary_conditions_particles=['periodic']*2,
)
print("  PICMI grid construction: ✓")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════════════"
echo "  WARPX-CUDA BUILD COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo "  Source:  $WARPX_SRC ($COMMIT)"
echo "  Build:   $WARPX_BUILD"
echo "  Install: $CONDA_PREFIX (pywarpx in plasma env)"
echo
echo "  Next: run a 1-step probe to confirm GPU execution"
echo "    cd ~/laser-plasma-research"
echo "    mpirun -n 1 python simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \\"
echo "        --base-fuel p11b --b-seed 85 --test --max-steps 1 \\"
echo "        --outdir runs/cloud_gpu_probe 2>&1 | tee gpu_probe.log"
echo
echo "  Watch for: 'particle access probe ... OK' line"
echo "  GPU usage check (in another shell): watch -n 1 nvidia-smi"
echo
