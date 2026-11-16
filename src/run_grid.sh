#!/usr/bin/env bash
# Run every command line of a grid file in parallel; one log per line in logs/.
# Usage (project root): bash src/run_grid.sh src/grid_v1.txt 7
set -u
cd "$(dirname "$0")/.."
GRID=$1
P=${2:-7}
mkdir -p logs runs
grep -vE '^\s*(#|$)' "$GRID" | xargs -d '\n' -P "$P" -I{} bash -c '{}'
echo "grid done: $GRID"
