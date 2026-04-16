#!/usr/bin/env bash
set -euo pipefail

# Move slurm-*.out files to a destination directory, recursively.
# Defaults: start at current directory, move to ./slurm_out
#
# Usage:
#   bash scripts/move-slurm-logs.sh [SRC_ROOT] [DEST_DIR] [PATTERN]
#
# Examples:
#   bash scripts/move-slurm-logs.sh
#   bash scripts/move-slurm-logs.sh . ./slurm_out "slurm-*.out"
#   bash scripts/move-slurm-logs.sh /data /archive/slurm "slurm-*.out"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/move-slurm-logs.sh [SRC_ROOT] [DEST_DIR] [PATTERN]

Arguments:
  SRC_ROOT  Root directory to search (default: .)
  DEST_DIR  Destination directory (default: ./slurm_out)
  PATTERN   Filename pattern (default: slurm-*.out)

Examples:
  bash scripts/move-slurm-logs.sh
  bash scripts/move-slurm-logs.sh . ./slurm_out "slurm-*.out"
  bash scripts/move-slurm-logs.sh /data /archive/slurm "slurm-*.out"
EOF
  exit 0
fi

SRC_ROOT="${1:-.}"
DEST_DIR="${2:-./slurm_out}"
PATTERN="${3:-slurm-*.out}"

mkdir -p "$DEST_DIR"

# Find matching files and move them, preserving only filenames.
# If you want to preserve directory structure, use the alternate command below.
find "$SRC_ROOT" -type f -name "$PATTERN" -print0 | while IFS= read -r -d '' file; do
  mv "$file" "$DEST_DIR"/
done

# Alternate (preserve structure):
# find "$SRC_ROOT" -type f -name "$PATTERN" -print0 | \
#   rsync -0 -a --remove-source-files --files-from=- "$SRC_ROOT"/ "$DEST_DIR"/
