#!/usr/bin/env bash
# Copy the canonical v10 pipeline and its deidentify support files
# from the repo root into src/python/ so they ship in the Electron bundle.
# Run manually via `npm run sync-pipeline`, or automatically via prebuild/prepackage.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT_DIR"
DEST="$ROOT_DIR/src/python"

FILES=(
  "video_transcription_pipeline_v10.py"
  "deidentify_names.py"
  "pseudonym_pool.json"
)

missing=()
for f in "${FILES[@]}"; do
  if [[ ! -f "$SRC/$f" ]]; then
    missing+=("$f")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "sync-pipeline: missing required files at repo root:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  cp "$SRC/$f" "$DEST/$f"
  echo "sync-pipeline: $f"
done

echo "sync-pipeline: done ($DEST)"
