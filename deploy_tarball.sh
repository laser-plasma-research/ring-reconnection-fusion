#!/usr/bin/env bash
# ============================================================================
# deploy_tarball.sh — package laser-plasma-research for cloud deployment
#
# Creates a tarball suitable for transfer to a fresh cloud instance,
# excluding outputs, secrets, caches, and local-only state.
#
# Usage:
#   ./deploy_tarball.sh                       # writes ./laser-plasma-research_<timestamp>.tar.gz
#   ./deploy_tarball.sh /path/to/output.tar.gz   # custom output path
#
# After upload to cloud, on the cloud instance:
#   tar -xzf laser-plasma-research_*.tar.gz
#   cd laser-plasma-research
#   mamba env create -f environment.yml
#   mamba activate plasma
#   # Set up your API key:
#   echo "ANTHROPIC_API_KEY=sk-..." > shared/.env
#   # Verify simulation script works:
#   mpirun -n 1 python simulation/pb11_ring_reconnection_v12_fuel_center_outer.py --test &
#   # Watch for the [particle access probe] OK line then Ctrl-C
# ============================================================================

set -euo pipefail

# ── Locate project root ────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Try common locations for the project root. Override with $PROJECT_ROOT env var.
if [[ -n "${PROJECT_ROOT:-}" ]]; then
    PROJECT="$PROJECT_ROOT"
elif [[ -f "$SCRIPT_DIR/program_config.yaml" ]]; then
    PROJECT="$SCRIPT_DIR"
elif [[ -f "$HOME/LaserFusionResearch/research/laser-plasma-research/program_config.yaml" ]]; then
    PROJECT="$HOME/LaserFusionResearch/research/laser-plasma-research"
elif [[ -f "$HOME/laser-plasma-research/program_config.yaml" ]]; then
    PROJECT="$HOME/laser-plasma-research"
else
    echo "ERROR: Could not find project root. Set PROJECT_ROOT env var or" >&2
    echo "       run this script from inside the project directory." >&2
    exit 1
fi

cd "$PROJECT"
echo "  Project root: $PROJECT"

# ── Output path ───────────────────────────────────────────────────────────────
TS="$(date -u +%Y%m%dT%H%M%SZ)"
# Default: write tarball to parent dir of project so tar doesn't try to add
# the in-progress archive to itself ("Can't add archive to itself" warning).
# Override with first positional arg.
OUT="${1:-../laser-plasma-research_${TS}.tar.gz}"

# Make output path absolute if not already
if [[ "$OUT" != /* ]]; then
    OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
fi

# Don't overwrite existing tarballs without warning
if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists. Refusing to overwrite." >&2
    exit 1
fi

echo "  Output:       $OUT"

# ── Sanity checks before packaging ────────────────────────────────────────────
echo
echo "── Pre-package checks ─────────────────────────────────────────────────────"

# 1. Check no .env files will be included
ENV_FILES=$(find . -name '.env' -not -path './.git/*' 2>/dev/null || true)
if [[ -n "$ENV_FILES" ]]; then
    echo "  Found .env files (will be EXCLUDED):"
    echo "$ENV_FILES" | sed 's/^/    /'
fi

# 2. Check for the simulation script
if [[ ! -f simulation/pb11_ring_reconnection_v12_fuel_center_outer.py ]]; then
    echo "  WARNING: simulation/pb11_ring_reconnection_v12_fuel_center_outer.py not found" >&2
fi

# 3. Check for orchestrator entry point
if [[ ! -f run_all.py ]]; then
    echo "  WARNING: run_all.py not found" >&2
fi

# 4. Check for environment.yml
if [[ ! -f environment.yml ]]; then
    echo "  ERROR: environment.yml is required for cloud deployment" >&2
    exit 1
fi

# 5. Check for any obvious hardcoded user paths in tracked files
HARDCODED=$(grep -rEn "/Users/|/home/[a-z]+/" \
    --include="*.py" --include="*.yaml" --include="*.sh" \
    --exclude-dir=runs --exclude-dir=__pycache__ --exclude-dir=.git \
    --exclude-dir=papers \
    . 2>/dev/null | grep -v "Path(__file__)" | grep -v "expanduser" || true)
if [[ -n "$HARDCODED" ]]; then
    echo "  WARNING: hardcoded user paths found (will not work on cloud):"
    echo "$HARDCODED" | head -10 | sed 's/^/    /'
    echo "  (showing first 10; check these before deployment)"
fi

# 6. Estimate size — BSD du (macOS) and GNU du have different exclude flags.
# Detect which we have; fall back to plain du if exclusions aren't supported.
if du --version 2>/dev/null | grep -q "GNU"; then
    # GNU du — supports --exclude
    TOTAL_SIZE=$(du -sh --exclude=runs --exclude=papers/*/paper/*.pdf \
        --exclude=__pycache__ --exclude=.git --exclude=.DS_Store \
        --exclude=*.pyc --exclude=.ipynb_checkpoints \
        . 2>/dev/null | awk '{print $1}')
else
    # BSD du (macOS) — use du -sk -I patterns or just measure everything
    # Note: BSD du -I requires -d 0 which conflicts with -s. Easiest path is
    # to measure with -sh on the whole tree and accept the slight overestimate.
    TOTAL_SIZE=$(du -sh -I .git -I runs -I __pycache__ -I .DS_Store . 2>/dev/null | \
                 awk '{print $1}')
    if [[ -z "$TOTAL_SIZE" ]]; then
        # Fall back to no exclusions if even -I isn't supported
        TOTAL_SIZE=$(du -sh . 2>/dev/null | awk '{print $1}')
    fi
fi
echo "  Est. unpacked size (after exclusions): ${TOTAL_SIZE:-unknown}"

echo

# ── Build the tarball ─────────────────────────────────────────────────────────
echo "── Packaging ──────────────────────────────────────────────────────────────"

# Determine tar version (BSD on macOS doesn't support --exclude-vcs in same way)
if tar --version 2>&1 | grep -q "GNU tar"; then
    TAR_VCS_FLAG="--exclude-vcs"
else
    TAR_VCS_FLAG=""  # BSD tar — handle .git via explicit excludes below
fi

# Project root is the parent of the project directory, so the tarball
# contains "laser-plasma-research/..." as the top-level path.
PROJECT_NAME="$(basename "$PROJECT")"
PARENT="$(dirname "$PROJECT")"

cd "$PARENT"
tar -czf "$OUT" \
    --exclude="${PROJECT_NAME}/runs" \
    --exclude="${PROJECT_NAME}/.git" \
    --exclude="${PROJECT_NAME}/.DS_Store" \
    --exclude="${PROJECT_NAME}/**/.DS_Store" \
    --exclude="${PROJECT_NAME}/**/__pycache__" \
    --exclude="${PROJECT_NAME}/**/*.pyc" \
    --exclude="${PROJECT_NAME}/**/.ipynb_checkpoints" \
    --exclude="${PROJECT_NAME}/shared/.env" \
    --exclude="${PROJECT_NAME}/shared/usage_log.json" \
    --exclude="${PROJECT_NAME}/papers/*/paper/*.pdf" \
    --exclude="${PROJECT_NAME}/papers/*/paper/*.aux" \
    --exclude="${PROJECT_NAME}/papers/*/paper/*.log" \
    --exclude="${PROJECT_NAME}/papers/*/paper/*.bbl" \
    --exclude="${PROJECT_NAME}/papers/*/paper/*.blg" \
    --exclude="${PROJECT_NAME}/papers/*/paper/*.out" \
    --exclude="${PROJECT_NAME}/papers/*/companion/private" \
    --exclude="${PROJECT_NAME}/papers/*/ip_analysis" \
    --exclude="${PROJECT_NAME}/**/*.bp" \
    --exclude="${PROJECT_NAME}/**/*.h5" \
    --exclude="${PROJECT_NAME}/**/diags" \
    --exclude="${PROJECT_NAME}/**/pb11_diags*" \
    $TAR_VCS_FLAG \
    "$PROJECT_NAME"

# ── Verify the result ─────────────────────────────────────────────────────────
echo
echo "── Verification ───────────────────────────────────────────────────────────"

PACKED_SIZE=$(du -h "$OUT" | awk '{print $1}')
FILE_COUNT=$(tar -tzf "$OUT" | wc -l | tr -d ' ')
echo "  Tarball:      $OUT"
echo "  Packed size:  $PACKED_SIZE"
echo "  File count:   $FILE_COUNT"

# Confirm no .env is inside
ENV_IN_TARBALL=$(tar -tzf "$OUT" | grep '\.env$' || true)
if [[ -n "$ENV_IN_TARBALL" ]]; then
    echo "  CRITICAL: .env file(s) found INSIDE tarball — DO NOT UPLOAD:"
    echo "$ENV_IN_TARBALL" | sed 's/^/    /'
    rm "$OUT"
    echo "  Tarball deleted. Fix exclusions and retry."
    exit 1
fi
echo "  .env check:   no .env files in tarball ✓"

# Confirm no api keys in tarball — extract entire tarball to temp dir and grep.
# The original version used tar -xzOf in a pipe which was fragile on BSD tar.
KEY_TMPDIR=$(mktemp -d)
tar -xzf "$OUT" -C "$KEY_TMPDIR" 2>/dev/null
# Use a key-format-specific regex (40+ char key body) so the script doesn't
# match its own detection pattern. Real keys are sk-ant-api03-<long random>.
KEY_HITS=$(grep -rcE 'sk-ant-api[0-9]+-[A-Za-z0-9_-]{20,}' "$KEY_TMPDIR" 2>/dev/null | awk -F: '{sum+=$NF} END {print sum+0}')
rm -rf "$KEY_TMPDIR"
if [[ "$KEY_HITS" -gt 0 ]]; then
    echo "  CRITICAL: detected ${KEY_HITS} occurrences of API key pattern in tarball" >&2
    rm "$OUT"
    exit 1
fi
echo "  API key check: no API keys in tarball ✓"

# Confirm key files made it in
EXPECT=(
    "${PROJECT_NAME}/run_all.py"
    "${PROJECT_NAME}/program_config.yaml"
    "${PROJECT_NAME}/environment.yml"
    "${PROJECT_NAME}/simulation/pb11_ring_reconnection_v12_fuel_center_outer.py"
    "${PROJECT_NAME}/simulation/pb11_run_all_papers.py"
    "${PROJECT_NAME}/orchestrator/__init__.py"
)
MISSING=()
for f in "${EXPECT[@]}"; do
    if ! tar -tzf "$OUT" | grep -q "^${f}$"; then
        MISSING+=("$f")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "  WARNING: expected files missing from tarball:" >&2
    for f in "${MISSING[@]}"; do echo "    $f"; done
fi

echo
echo "── Done ───────────────────────────────────────────────────────────────────"
echo "  Tarball ready: $OUT"
echo
echo "  Upload to cloud (replace HOST and KEY):"
echo "    scp -i ~/.ssh/laser-fusion-research_private.pem \\"
echo "        $OUT \\"
echo "        ubuntu@HOST:~/"
echo
echo "  On the cloud instance:"
echo "    tar -xzf $(basename $OUT)"
echo "    cd $PROJECT_NAME"
echo "    bash setup_cloud.sh    # (next step — env + WarpX + verify)"
echo
