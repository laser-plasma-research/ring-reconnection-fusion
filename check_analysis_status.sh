#!/bin/bash
# check_analysis_status.sh — sanity check & cleanup utility for laser-plasma research
# Usage: bash check_analysis_status.sh [--clean]
#
# Without --clean: just shows status (running processes, locks, log tails)
# With --clean:    kills any stale processes and removes lock files

set -u

CLEAN_MODE=false
if [ "${1:-}" = "--clean" ]; then
    CLEAN_MODE=true
fi

echo "============================================================"
echo "  ANALYSIS PIPELINE STATUS CHECK"
echo "  Time: $(date)"
echo "============================================================"
echo

# 1. Running processes
echo "--- Running Python/analysis processes ---"
PROCS=$(pgrep -af "python.*analysis_scripts|python.*pb11_|python.*visualize_all|run_full_analysis|run_pipeline" | grep -v grep || true)
if [ -z "$PROCS" ]; then
    echo "  (none running)"
else
    echo "$PROCS"
fi
echo

# 2. Running simulation processes (DON'T kill these)
echo "--- Running simulation processes (preserved) ---"
SIMS=$(pgrep -af "ring_reconnection" | grep -v grep || true)
if [ -z "$SIMS" ]; then
    echo "  (none running)"
else
    echo "$SIMS"
    echo "  WARNING: simulation(s) still running - will NOT be killed"
fi
echo

# 3. Lock / temp files
echo "--- Lock / temp files in runs/ ---"
LOCKS=$(find ~/laser-plasma-research/runs -maxdepth 3 -name "*.lock" -o -name ".running" -o -name ".analysis_pid" 2>/dev/null || true)
if [ -z "$LOCKS" ]; then
    echo "  (none found)"
else
    echo "$LOCKS"
fi
echo

# 4. Recent analysis logs
echo "--- Recent analysis logs ---"
LOGS=$(ls -t ~/laser-plasma-research/*analysis*.log 2>/dev/null | head -3 || true)
if [ -z "$LOGS" ]; then
    echo "  (no analysis logs found)"
else
    for log in $LOGS; do
        echo "  $log ($(stat -c %y $log 2>/dev/null || stat -f '%Sm' $log))"
    done
fi
echo

# 5. Free disk
echo "--- Free disk space ---"
df -h ~/laser-plasma-research | head -2
echo

# 6. Cleanup mode
if $CLEAN_MODE; then
    echo "============================================================"
    echo "  CLEANUP MODE — killing stale analysis processes"
    echo "============================================================"
    
    if [ -n "$PROCS" ]; then
        echo "$PROCS" | awk '{print $1}' | while read pid; do
            echo "  Killing PID $pid..."
            kill -9 $pid 2>/dev/null && echo "    OK" || echo "    (already gone)"
        done
    else
        echo "  No analysis processes to kill"
    fi
    
    if [ -n "$LOCKS" ]; then
        echo
        echo "  Removing lock files..."
        echo "$LOCKS" | while read lock; do
            rm -f "$lock" && echo "    Removed $lock"
        done
    fi
    
    echo
    echo "  Cleanup complete. State is now clean."
else
    if [ -n "$PROCS" ] || [ -n "$LOCKS" ]; then
        echo "  Run with --clean to kill processes / remove locks:"
        echo "    bash $0 --clean"
    else
        echo "  STATE IS CLEAN — safe to launch new analysis"
    fi
fi

echo
echo "============================================================"
