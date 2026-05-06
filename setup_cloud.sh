#!/usr/bin/env bash
# ============================================================================
# setup_cloud.sh — cloud-instance bootstrap for laser-plasma-research
#
# Run this AFTER extracting the tarball on a fresh Ubuntu cloud instance.
# It does:
#   1. Verify required system packages are installed
#   2. Install miniforge (mamba) if not already present
#   3. Create the 'plasma' conda env from environment.yml
#   4. Verify WarpX is GPU-enabled
#   5. Verify the simulation script parses
#   6. Run the particle-access probe (1-step run) to confirm physics pipeline
#
# Designed to fail loudly with actionable messages, not silently.
# ============================================================================

set -euo pipefail

PROJECT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT"

echo "════════════════════════════════════════════════════════════════════════"
echo "  Cloud setup — laser-plasma-research"
echo "  Project:  $PROJECT"
echo "  Hostname: $(hostname)"
echo "  Date:     $(date -u)"
echo "════════════════════════════════════════════════════════════════════════"

# ── Step 1: System checks ─────────────────────────────────────────────────────
echo
echo "── Step 1: System checks ─────────────────────────────────────────────────"

# OS check
if [[ -f /etc/os-release ]]; then
    OS_NAME=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2 | tr -d '"')
    echo "  OS:        $OS_NAME"
    if ! grep -q "Ubuntu 22\." /etc/os-release && ! grep -q "Ubuntu 24\." /etc/os-release; then
        echo "  WARNING: This script is tested on Ubuntu 22.04/24.04. Other distros may need adjustment." >&2
    fi
fi

# nvidia-smi check
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU detection:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
    echo "  GPU count: $GPU_COUNT"
else
    echo "  ERROR: nvidia-smi not found. CUDA driver not installed?" >&2
    echo "         WarpX will fall back to CPU — runs will be 50× slower." >&2
    GPU_COUNT=0
fi

# CUDA version
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    echo "  CUDA:      $CUDA_VER"
fi

# Required system tools
for tool in gcc make tar git; do
    if ! command -v "$tool" &>/dev/null; then
        echo "  ERROR: required system tool '$tool' not found" >&2
        echo "         Install with: sudo apt-get install build-essential git" >&2
        exit 1
    fi
done
echo "  Build tools: ✓"

# ── Step 2: Install miniforge if needed ───────────────────────────────────────
echo
echo "── Step 2: Conda/mamba environment ───────────────────────────────────────"

if ! command -v mamba &>/dev/null && ! command -v conda &>/dev/null; then
    echo "  Installing miniforge (mamba)..."
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    wget -q "$MINIFORGE_URL" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge.sh
    
    # Add to current shell
    export PATH="$HOME/miniforge3/bin:$PATH"
    
    # Add to .bashrc for future sessions
    if ! grep -q "miniforge3/bin" ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Initialise conda for bash
    "$HOME/miniforge3/bin/conda" init bash >/dev/null 2>&1
    
    echo "  Miniforge installed at $HOME/miniforge3"
fi

# Source conda so 'conda activate' works in this script
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

if command -v mamba &>/dev/null; then
    PKG_MGR=mamba
else
    PKG_MGR=conda
fi
echo "  Package manager: $PKG_MGR ($(${PKG_MGR} --version | head -1))"

# ── Step 3: Create env from environment.yml ───────────────────────────────────
echo
echo "── Step 3: Create plasma env ─────────────────────────────────────────────"

if conda env list | grep -q '^plasma '; then
    echo "  Env 'plasma' already exists. Skipping creation."
    echo "  (To recreate: conda env remove -n plasma && rerun this script)"
else
    if [[ ! -f environment.yml ]]; then
        echo "  ERROR: environment.yml not found in $PROJECT" >&2
        exit 1
    fi
    echo "  Creating env from environment.yml (this takes 5–15 minutes)..."
    $PKG_MGR env create -f environment.yml
fi

conda activate plasma
echo "  Active env: $CONDA_DEFAULT_ENV"

# ── Step 4: Verify WarpX is GPU-enabled ───────────────────────────────────────
echo
echo "── Step 4: WarpX verification ────────────────────────────────────────────"

python -c "import pywarpx; print(f'  pywarpx imported OK')" || {
    echo "  ERROR: pywarpx import failed" >&2
    exit 1
}

# Check WarpX dimension support and GPU build
WARPX_INFO=$(python -c "
import pywarpx
print('amrex version:', getattr(pywarpx, '__version__', 'unknown'))
# Try to detect GPU build by checking for CUDA symbols
try:
    from pywarpx import _libwarpx
    # Look for any CUDA-related attribute or function
    has_cuda = any('cuda' in attr.lower() or 'gpu' in attr.lower()
                   for attr in dir(_libwarpx))
    print('CUDA symbols in _libwarpx:', has_cuda)
except Exception as e:
    print('Could not introspect _libwarpx:', e)
")
echo "$WARPX_INFO" | sed 's/^/  /'

# A more direct test: try creating a minimal grid and see if it complains about GPU
python <<'PYEOF'
from pywarpx import picmi
import numpy as np
g = picmi.Cartesian2DGrid(
    number_of_cells=[16, 16],
    lower_bound=[-1e-3, -1e-3],
    upper_bound=[ 1e-3,  1e-3],
    lower_boundary_conditions=['periodic']*2,
    upper_boundary_conditions=['periodic']*2,
    lower_boundary_conditions_particles=['periodic']*2,
    upper_boundary_conditions_particles=['periodic']*2,
)
print("  Minimal PICMI grid OK")
PYEOF
if [[ $? -ne 0 ]]; then
    echo "  WARNING: Minimal pywarpx import test failed — env may have issues" >&2
fi

# ── Step 5: Syntax-check simulation script ────────────────────────────────────
echo
echo "── Step 5: Simulation script syntax check ───────────────────────────────"
SIM_SCRIPT="simulation/pb11_ring_reconnection_v12_fuel_center_outer.py"
if [[ ! -f "$SIM_SCRIPT" ]]; then
    echo "  ERROR: $SIM_SCRIPT not found" >&2
    exit 1
fi
python -c "import ast; ast.parse(open('$SIM_SCRIPT').read()); print('  Simulation script parses ✓')"

# Also check the orchestrator
python -c "import ast; ast.parse(open('simulation/pb11_run_all_papers.py').read()); print('  Orchestrator parses ✓')"
python -c "import ast; ast.parse(open('run_all.py').read()); print('  run_all.py parses ✓')"

# ── Step 6: Probe run ─────────────────────────────────────────────────────────
echo
echo "── Step 6: 1-step probe run (verifies particle access on this hardware) ─"

PROBE_DIR="$PROJECT/runs/cloud_probe_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$PROBE_DIR"

echo "  Probe output: $PROBE_DIR"
echo "  Running mpirun -n 1 with 1 GPU..."
echo "  This should complete in <1 minute and produce a probe diagnostic line."
echo

# Capture the first ~20 lines of output and check for the probe success
mpirun -n 1 python "$SIM_SCRIPT" \
    --base-fuel p11b --b-seed 300 --test \
    --outdir "$PROBE_DIR" 2>&1 | tee "$PROBE_DIR/probe.log" | \
    grep -E "particle access probe|FAILED|ERROR" | head -20 &

PROBE_PID=$!

# Let it run for ~60 seconds — long enough to do init + step 1 + probe
sleep 60

# Kill the simulation (we only wanted the probe result)
if kill -0 $PROBE_PID 2>/dev/null; then
    pkill -P $PROBE_PID 2>/dev/null || true
    kill $PROBE_PID 2>/dev/null || true
fi
sleep 2

# Check what the probe said
if grep -q "particle access probe.*OK" "$PROBE_DIR/probe.log" 2>/dev/null; then
    echo
    echo "  ✓ Particle access probe SUCCEEDED"
    grep "particle access probe" "$PROBE_DIR/probe.log" | sed 's/^/    /'
elif grep -q "particle access probe.*FAILED" "$PROBE_DIR/probe.log" 2>/dev/null; then
    echo
    echo "  ✗ Particle access probe FAILED" >&2
    grep -A 3 "particle access probe" "$PROBE_DIR/probe.log" | sed 's/^/    /' >&2
    exit 1
else
    echo
    echo "  WARNING: probe didn't complete in 60s — may need longer init" >&2
    echo "  Last 20 lines of $PROBE_DIR/probe.log:" >&2
    tail -20 "$PROBE_DIR/probe.log" | sed 's/^/    /' >&2
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo
echo "  Next steps:"
echo
echo "  1. Set up your API key (if using orchestrator with deliverables):"
echo "       echo 'ANTHROPIC_API_KEY=sk-ant-...' > shared/.env"
echo "       chmod 600 shared/.env"
echo
echo "  2. Run a validation sub-job (vortex test + no-B baseline):"
echo "       mpirun -n 1 python simulation/pb11_ring_reconnection_v12_fuel_center_outer.py \\"
echo "           --base-fuel p11b --b-seed 300 --test --outdir runs/validation_with_b 2>&1 | tee val.log"
echo
echo "  3. Once validated, run full program in parallel pairs:"
echo "       python3 run_all.py --gpus-per-host $GPU_COUNT --max-parallel $GPU_COUNT \\"
echo "           --mpi-ranks 1 --runs-dir runs/cloud_run_001"
echo
echo "  4. Monitor:"
echo "       tail -f runs/cloud_run_001/progress_report.txt"
echo
