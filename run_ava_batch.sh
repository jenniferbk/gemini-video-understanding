#!/bin/bash
# Batch process Ava 4S Science SG videos with V10
# Runs 4 in parallel to stay within API rate limits

export GOOGLE_API_KEY="${GOOGLE_API_KEY:?Set GOOGLE_API_KEY env var before running}"
PYTHON="/Users/jenniferkleiman/Documents/COMS/venv/bin/python3"
SCRIPT="/Users/jenniferkleiman/Documents/COMS/video_transcription_pipeline_v10.py"
OUTDIR="/Users/jenniferkleiman/Documents/COMS/batch_ava_4s_output"
VIDDIR="/Users/jenniferkleiman/Documents/COMS/batch_ava_4s"

mkdir -p "$OUTDIR"

process_video() {
  local vid="$1"
  local name=$(basename "$vid" .mp4)
  local outdir="$OUTDIR/$name"
  echo "[$(date +%H:%M:%S)] START: $name"
  "$PYTHON" "$SCRIPT" process "$vid" \
    -o "$outdir" \
    --no-confirm \
    --chunk-minutes 1.0 \
    --overlap 15 \
    --resolution HIGH \
    --fps 2 \
    -m gemini-3-flash-preview \
    --thinking-budget 4096 \
    > "$OUTDIR/${name}.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] DONE: $name (exit=$rc)"
  return $rc
}

echo "=== Ava 4S Science Batch - $(date) ==="
echo "Processing 8 videos, 4 at a time"
echo ""

# Wave 1: first 4 videos
echo "--- Wave 1 ---"
process_video "$VIDDIR/Y2_4S_Ava_Day1_SG1.mp4" &
process_video "$VIDDIR/Y2_4S_Ava_Day1_SG2.mp4" &
process_video "$VIDDIR/Y2_4S_Ava_Day2_SG1.mp4" &
process_video "$VIDDIR/Y2_4S_Ava_Day2_SG2.mp4" &
wait
echo ""

# Wave 2: PAUSED per user request
echo "Wave 1 complete. Wave 2 paused."
echo "=== Wave 1 done - $(date) ==="
