#!/usr/bin/env bash
# =============================================================================
# build_container.sh — Build the Ring Reconnection Fusion runtime container
#
# Uses conda-pack to capture the live cloud plasma env, then builds a Docker
# image with that env preinstalled.
#
# Usage:
#   bash build_container.sh <version>      e.g. bash build_container.sh v1.0.0
# =============================================================================

set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>     e.g. $0 v1.0.0"
    exit 1
fi

IMAGE_NAME="ring-reconnection-fusion"
TAG="${VERSION}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_PATH="${PROJECT_ROOT}/plasma_env.tar.gz"

echo "=========================================================="
echo "  Building ${IMAGE_NAME}:${TAG}"
echo "=========================================================="
echo ""

cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Step 1: conda-pack the live plasma env (if not already done or stale)
# -----------------------------------------------------------------------------

# Compare modification time of the pack against the cloud's conda env directory
NEEDS_PACK=0
if [ ! -f "${PACK_PATH}" ]; then
    echo "→ No plasma_env.tar.gz found, packing..."
    NEEDS_PACK=1
else
    PACK_AGE_S=$(( $(date +%s) - $(stat -c %Y "${PACK_PATH}") ))
    PACK_AGE_H=$(( PACK_AGE_S / 3600 ))
    echo "→ plasma_env.tar.gz exists, age: ${PACK_AGE_H} hours"
    if [ "${PACK_AGE_H}" -gt 24 ]; then
        echo "  Re-packing (stale, >24h old)..."
        NEEDS_PACK=1
    else
        echo "  Reusing existing pack"
    fi
fi

if [ "${NEEDS_PACK}" -eq 1 ]; then
    if ! command -v conda-pack >/dev/null 2>&1; then
        # Find conda-pack via miniforge
        if [ -x "${HOME}/miniforge3/envs/plasma/bin/conda-pack" ]; then
            CONDA_PACK="${HOME}/miniforge3/envs/plasma/bin/conda-pack"
        else
            echo "ERROR: conda-pack not found. Install with: conda install -n plasma conda-pack"
            exit 1
        fi
    else
        CONDA_PACK=$(command -v conda-pack)
    fi
    
    echo "  Using: ${CONDA_PACK}"
    rm -f "${PACK_PATH}"
    "${CONDA_PACK}" -n plasma -o "${PACK_PATH}"
fi

PACK_SIZE=$(du -h "${PACK_PATH}" | cut -f1)
echo "  Pack size: ${PACK_SIZE}"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Verify required files exist
# -----------------------------------------------------------------------------
echo "→ Verifying required files..."
REQUIRED_FILES=(
    Dockerfile
    .dockerignore
    plasma_env.tar.gz
    simulation/pb11_ring_reconnection_v15_pulsed.py
    simulation/run_with_analysis.sh
    program_config.yaml
    program_config_paper1.yaml
    program_config_paper3.yaml
    setup_cloud.sh
    build_warpx_cuda.sh
    LICENSE
    NOTICE
    CITATION.cff
    README.md
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$f" ]; then
        echo "  MISSING: $f"
        MISSING=1
    fi
done
if [ "${MISSING}" -eq 1 ]; then
    echo "ERROR: Missing required files (see above). Cannot proceed."
    exit 1
fi
echo "  All required files present"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Docker build
# -----------------------------------------------------------------------------
echo "→ Build starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)..."

docker build \
    --tag "${IMAGE_NAME}:${TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    .

echo ""
echo "=========================================================="
echo "  Build complete"
echo "=========================================================="

# -----------------------------------------------------------------------------
# Step 4: Report image info
# -----------------------------------------------------------------------------
echo ""
echo "→ Image info:"
docker images "${IMAGE_NAME}:${TAG}" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}"

echo ""
echo "→ Image SHA256:"
docker inspect "${IMAGE_NAME}:${TAG}" --format "  {{.Id}}"

# -----------------------------------------------------------------------------
# Step 5: Optional GPU access test (only if --nvidia runtime available)
# -----------------------------------------------------------------------------
echo ""
echo "→ Running GPU access test..."
if docker run --rm --gpus all "${IMAGE_NAME}:${TAG}" \
        python -c "import pywarpx; print('  pywarpx imports under GPU runtime: OK')" 2>&1; then
    echo "  GPU access test PASSED"
else
    echo "  WARNING: GPU access test failed (may be normal if no NVIDIA runtime here)"
fi

echo ""
echo "Container ready: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Run with:"
echo "  docker run --rm --gpus all -v \$PWD/runs:/work/runs ${IMAGE_NAME}:${TAG}"
