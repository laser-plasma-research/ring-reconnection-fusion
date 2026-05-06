#!/usr/bin/env bash
# ============================================================================
# archive_runs.sh — compress completed simulation runs into per-run tarballs
#
# Run on the CLOUD instance after all simulations complete and after summary
# CSVs have been pulled to the workstation. Compresses the heavy openPMD
# diagnostic data (fields/, particles/, etc.) into per-run .tar.zst archives.
#
# Per-run tarballs let you download individual runs on demand without
# pulling the entire 2 TB dataset.
#
# Usage:
#   ./archive_runs.sh runs/cloud_run_001                  # compress in-place
#   ./archive_runs.sh runs/cloud_run_001 /mnt/archive/    # output dir
#
# After this completes:
#   - Original run directories preserved (NOT deleted automatically)
#   - One .tar.zst per sub-job in <output>/archives/
#   - manifest.txt listing each archive's size + sha256 for download verification
# ============================================================================

set -euo pipefail

# ── Cross-platform helpers (Linux + macOS) ────────────────────────────────────
# macOS BSD coreutils differ from GNU in flags that this script uses. These
# wrappers detect which is available and dispatch accordingly.

_du_bytes() {
    # Print byte count of a file or directory. GNU: -sb; BSD: -sk × 1024.
    if du -sb "$1" >/dev/null 2>&1; then
        du -sb "$1" | awk '{print $1}'
    else
        du -sk "$1" | awk '{print $1 * 1024}'
    fi
}

_relpath() {
    # _relpath <target> <base> -> path of <target> relative to <base>.
    # GNU realpath has --relative-to; BSD doesn't. Fall back to python3.
    if realpath --relative-to=/ / >/dev/null 2>&1; then
        realpath --relative-to="$2" "$1"
    else
        python3 -c "import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$1" "$2"
    fi
}

_stat_size() {
    # Byte size of a regular file. GNU: stat -c%s; BSD/macOS: stat -f%z.
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

_sha256() {
    # SHA256 hex digest of a file. Try sha256sum first, then shasum.
    # Use a functional-test (not just command -v) since 'command -v' returns
    # success even for broken shim files on PATH.
    local h
    if h=$(sha256sum "$1" 2>/dev/null) && [[ -n "$h" ]]; then
        echo "$h" | awk '{print $1}'
        return 0
    fi
    if h=$(shasum -a 256 "$1" 2>/dev/null) && [[ -n "$h" ]]; then
        echo "$h" | awk '{print $1}'
        return 0
    fi
    echo "ERROR: neither sha256sum nor shasum produced output for $1" >&2
    return 1
}

_with_lock() {
    # Run "$@" while holding an exclusive lock on $1. flock on Linux,
    # mkdir-based spinlock on macOS / where flock is missing/broken.
    local lockfile="$1"; shift

    # Test flock by running it on a throwaway fd to see if it actually works
    # (not just whether the binary is on PATH — broken shims can lie).
    local flock_works=0
    if flock --version >/dev/null 2>&1; then
        flock_works=1
    fi

    if [[ $flock_works -eq 1 ]]; then
        (
            flock 200
            "$@"
        ) 200>"$lockfile"
    else
        local lockdir="${lockfile}.d"
        local tries=0
        while ! mkdir "$lockdir" 2>/dev/null; do
            sleep 0.05
            tries=$((tries+1))
            if [[ $tries -gt 1000 ]]; then
                echo "WARNING: lock $lockdir held >50s; bypassing" >&2
                break
            fi
        done
        "$@"
        rmdir "$lockdir" 2>/dev/null || true
    fi
}

export -f _du_bytes _relpath _stat_size _sha256 _with_lock

# ── Args ──────────────────────────────────────────────────────────────────────
RUNS_DIR="${1:?Usage: $0 <runs_dir> [output_dir]}"
OUT_DIR="${2:-${RUNS_DIR}/archives}"
COMPRESS_LEVEL="${ZSTD_LEVEL:-19}"   # 19 is the sweet spot; 22 with --long for max
JOBS="${ARCHIVE_JOBS:-4}"            # parallel compression workers

# ── Sanity ────────────────────────────────────────────────────────────────────
if [[ ! -d "$RUNS_DIR" ]]; then
    echo "ERROR: $RUNS_DIR does not exist" >&2
    exit 1
fi

if ! command -v zstd &>/dev/null; then
    echo "ERROR: zstd not installed. Install with:" >&2
    echo "  Ubuntu/Debian: sudo apt-get install -y zstd" >&2
    echo "  macOS:         brew install zstd" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "  Source:    $RUNS_DIR"
echo "  Output:    $OUT_DIR"
echo "  zstd level: $COMPRESS_LEVEL"
echo "  Parallel:  $JOBS workers"

# ── Find run directories ──────────────────────────────────────────────────────
# A run dir is one containing run_meta.txt.
# Use a while-read loop instead of mapfile (mapfile is bash 4+; macOS ships 3.2).
RUN_DIRS=()
while IFS= read -r line; do
    RUN_DIRS+=("$line")
done < <(find "$RUNS_DIR" -maxdepth 4 -name 'run_meta.txt' \
              -exec dirname {} \; | sort)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no run directories found under $RUNS_DIR" >&2
    echo "       (looking for any subdirectory containing run_meta.txt)" >&2
    exit 1
fi

echo "  Found ${#RUN_DIRS[@]} run(s) to archive"
echo

# ── Estimate total size ───────────────────────────────────────────────────────
TOTAL_BYTES=0
for d in "${RUN_DIRS[@]}"; do
    BYTES=$(_du_bytes "$d")
    TOTAL_BYTES=$((TOTAL_BYTES + BYTES))
done
TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_BYTES / 1024 / 1024 / 1024}")
echo "  Total source size: ${TOTAL_GB} GB"
echo "  Expected compressed (3-4x typical): $(awk "BEGIN {printf \"%.0f\", $TOTAL_GB * 0.3}") GB"
echo

# ── Manifest header ───────────────────────────────────────────────────────────
MANIFEST="$OUT_DIR/manifest.txt"
{
    echo "# Archive manifest — generated $(date -u)"
    echo "# Hostname: $(hostname)"
    echo "# Source:   $RUNS_DIR"
    echo "# zstd level: $COMPRESS_LEVEL"
    echo "#"
    echo "# Each run is archived in two ways:"
    echo "#   1. Complete tarball (.tar.zst)  -- everything, including big openPMD bp5 dirs"
    echo "#   2. Loose copies in loose/<rel>/ -- small text/image files for direct access"
    echo "#      (CSV, TXT, LOG, JSON, PNG, PDF, SVG, MD, warpx_used_inputs)"
    echo "# The loose copies duplicate files already in the tarball; safe to delete"
    echo "# loose/ if disk space matters and re-extract from .tar.zst when needed."
    echo "#"
    echo "# Format: size_bytes  sha256  archive_path  source_path  (source_size, ratio, loose_count)"
    echo "#"
} > "$MANIFEST"

# ── Small files worth keeping loose for direct access ────────────────────────
# These are extracted alongside the tarball into <out>/loose/<rel_path>/ so you
# can grep/plot/diff without decompressing. The tarball contains the same files
# plus all the heavy diagnostic data.
LOOSE_PATTERNS=(
    "*.csv"
    "*.txt"
    "*.log"
    "*.json"
    "*.png"
    "*.pdf"
    "*.svg"
    "*.md"
    "warpx_used_inputs"
)

# ── Compress one run ──────────────────────────────────────────────────────────
compress_one() {
    local src_dir="$1"
    local rel
    rel="$(_relpath "$src_dir" "$RUNS_DIR")"
    # archive name: replace / with _ for flat output
    local arc_name="${rel//\//_}.tar.zst"
    local arc_path="$OUT_DIR/$arc_name"

    if [[ -f "$arc_path" ]]; then
        echo "  [skip]  $rel  (already archived)"
        return 0
    fi

    # Reconstruct LOOSE_PATTERNS array from exported string (arrays don't
    # cleanly cross subprocesses via export -f, so we pass as space-sep string).
    local -a loose_patterns
    read -ra loose_patterns <<< "${LOOSE_PATTERNS_STR}"

    local src_bytes
    src_bytes=$(_du_bytes "$src_dir")
    local src_gb
    src_gb=$(awk "BEGIN {printf \"%.2f\", $src_bytes / 1024 / 1024 / 1024}")

    echo "  [pack]  $rel  (${src_gb} GB) → $arc_name"

    # Tar from parent of source so the archive contains a single top-level dir
    # matching the run's relative path. Pipe through zstd with multi-threading.
    local parent base
    parent="$(dirname "$src_dir")"
    base="$(basename "$src_dir")"

    tar -C "$parent" -cf - "$base" \
        | zstd -${COMPRESS_LEVEL} -T0 --long=27 -q -o "$arc_path"

    # ── Publish loose copies of small files for direct access ─────────────────
    # Mirrors the run directory structure under <out>/loose/<rel>/ but only
    # for files matching the LOOSE_PATTERNS (typically ~50 KB per run).
    local loose_dir="$OUT_DIR/loose/$rel"
    mkdir -p "$loose_dir"
    local loose_count=0
    for pattern in "${loose_patterns[@]}"; do
        # Use find so we only match regular files and preserve subdir structure
        while IFS= read -r -d '' f; do
            local rel_to_src
            rel_to_src="$(_relpath "$f" "$src_dir")"
            local dest="$loose_dir/$rel_to_src"
            mkdir -p "$(dirname "$dest")"
            cp -p "$f" "$dest"
            loose_count=$((loose_count + 1))
        done < <(find "$src_dir" -type f -name "$pattern" -print0 2>/dev/null)
    done

    # Compute hash and size for manifest
    local arc_bytes hash
    arc_bytes=$(_stat_size "$arc_path")
    hash=$(_sha256 "$arc_path")
    local ratio
    ratio=$(awk "BEGIN {printf \"%.2fx\", $src_bytes / $arc_bytes}")

    # Append to manifest (atomic write — buffer first then append)
    printf "%s  %s  %s  %s  (%s, %s, %d loose)\n" \
        "$arc_bytes" "$hash" "$arc_name" "$rel" "$src_gb GB" "$ratio" "$loose_count" \
        >> "$MANIFEST.tmp.$$"

    # Atomic append using a cross-platform lock (flock on Linux, mkdir on macOS)
    _with_lock "$MANIFEST.lock" bash -c "cat '$MANIFEST.tmp.$$' >> '$MANIFEST' && rm -f '$MANIFEST.tmp.$$'"
}

export -f compress_one
export RUNS_DIR OUT_DIR COMPRESS_LEVEL MANIFEST
export LOOSE_PATTERNS_STR="${LOOSE_PATTERNS[*]}"

# ── Run in parallel ───────────────────────────────────────────────────────────
echo "── Compressing ${#RUN_DIRS[@]} runs in parallel (${JOBS} at a time) ──"
echo

if command -v parallel &>/dev/null; then
    printf '%s\n' "${RUN_DIRS[@]}" | parallel -j"$JOBS" --bar compress_one
else
    # Fallback: xargs
    printf '%s\n' "${RUN_DIRS[@]}" | xargs -I{} -P"$JOBS" bash -c 'compress_one "$@"' _ {}
fi

# Clean up lock file
# Clean up lock files (both flock and mkdir-style) and any leftover tmps
rm -f "$MANIFEST.lock" "$MANIFEST.tmp"
rmdir "$MANIFEST.lock.d" 2>/dev/null || true
rm -f "$MANIFEST.tmp."*

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "── Archive complete ──"

# Total of compressed tarballs only — sum byte sizes via portable helper
ARCHIVES_BYTES=0
while IFS= read -r -d '' f; do
    ARCHIVES_BYTES=$((ARCHIVES_BYTES + $(_stat_size "$f")))
done < <(find "$OUT_DIR" -maxdepth 1 -name '*.tar.zst' -print0)
ARCHIVES_GB=$(awk "BEGIN {printf \"%.1f\", ${ARCHIVES_BYTES:-0} / 1024 / 1024 / 1024}")

# Total of loose files
if [[ -d "$OUT_DIR/loose" ]]; then
    LOOSE_BYTES=$(_du_bytes "$OUT_DIR/loose")
    LOOSE_MB=$(awk "BEGIN {printf \"%.1f\", ${LOOSE_BYTES:-0} / 1024 / 1024}")
    LOOSE_COUNT=$(find "$OUT_DIR/loose" -type f 2>/dev/null | wc -l | tr -d ' ')
else
    LOOSE_MB=0; LOOSE_COUNT=0
fi

RATIO=$(awk "BEGIN {printf \"%.2fx\", $TOTAL_BYTES / ${ARCHIVES_BYTES:-1}}")
echo "  Original:    ${TOTAL_GB} GB"
echo "  Archives:    ${ARCHIVES_GB} GB  (${RATIO} compression)"
echo "  Loose files: ${LOOSE_MB} MB  (${LOOSE_COUNT} files)"
echo "  Manifest:    $MANIFEST"
echo
echo "  Layout:"
echo "    $OUT_DIR/"
echo "    ├── manifest.txt"
echo "    ├── loose/                       <-- direct access to small files"
echo "    │   └── <run_path>/<files>"
echo "    └── *.tar.zst                    <-- complete tarballs (everything)"
echo
echo "  Direct access to small files (no decompression needed):"
echo "    cat $OUT_DIR/loose/p01/p1_static_p11b/fusion_rate_power_by_iter.csv"
echo
echo "  Full extraction when you need the openPMD diagnostics:"
echo "    ./extract_run.sh $OUT_DIR/p01_p1_static_p11b.tar.zst"
echo
echo "  Original run directories were NOT deleted. To free space after"
echo "  verifying the archives are good:"
echo "    rm -rf $RUNS_DIR/p*/             # delete originals (only after verify!)"
