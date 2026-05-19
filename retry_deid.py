#!/usr/bin/env python3
"""Retry de-identification on an existing DEIDENTIFICATION_FAILED transcript.

Usage:
    python3 retry_deid.py <output_dir> <video_stem>
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from video_transcription_pipeline_v10 import (
    GeminiClient,
    TranscriptionConfigV10,
    SubtitleExporter,
    VideoTranscriptionPipelineV10,
)
from deidentify_names import deidentify_transcript

if len(sys.argv) not in (3, 4):
    print("Usage: retry_deid.py <output_dir> <video_stem> [model_name]", file=sys.stderr)
    sys.exit(1)

OUT_DIR = Path(sys.argv[1])
STEM = sys.argv[2]
OVERRIDE_MODEL = sys.argv[3] if len(sys.argv) == 4 else None
FAILED_TRANSCRIPT = OUT_DIR / f"{STEM}_DEIDENTIFICATION_FAILED_transcript.txt"
FAILED_TRANSANA = OUT_DIR / f"{STEM}_DEIDENTIFICATION_FAILED_transana.txt"
FAILED_SRT = OUT_DIR / f"{STEM}_DEIDENTIFICATION_FAILED.srt"

if not FAILED_TRANSCRIPT.exists():
    print(f"ERROR: not found: {FAILED_TRANSCRIPT}", file=sys.stderr)
    sys.exit(1)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
    sys.exit(1)

config = TranscriptionConfigV10()
if OVERRIDE_MODEL:
    config.model_name = OVERRIDE_MODEL
    config.thinking_budget = 0
    print(f"Overriding model: {config.model_name}, thinking_budget=0")
client = GeminiClient(api_key, config)

text = FAILED_TRANSCRIPT.read_text(encoding="utf-8")
pool_path = str(Path(__file__).parent / "pseudonym_pool.json")

print(f"Running de-id over {len(text):,} chars from {FAILED_TRANSCRIPT.name}...")
deidentified, name_map = deidentify_transcript(text, client, pool_path)

name_map_path = OUT_DIR / "transcript_name_map.json"
with open(name_map_path, "w", encoding="utf-8") as f:
    json.dump(name_map.to_dict(), f, indent=2, ensure_ascii=False)
try:
    os.chmod(name_map_path, 0o600)
except OSError:
    pass
print(f"  wrote {name_map_path}")

new_transcript = OUT_DIR / f"{STEM}_transcript.txt"
new_transcript.write_text(deidentified, encoding="utf-8")
print(f"  wrote {new_transcript}")

pipeline = VideoTranscriptionPipelineV10(api_key, config)
clean = pipeline._create_clean_transcript(deidentified)
new_transana = OUT_DIR / f"{STEM}_transana.txt"
new_transana.write_text(clean, encoding="utf-8")
print(f"  wrote {new_transana}")

new_srt = OUT_DIR / f"{STEM}.srt"
SubtitleExporter.to_srt(clean, str(new_srt))
print(f"  wrote {new_srt}")

for old in (FAILED_TRANSCRIPT, FAILED_TRANSANA, FAILED_SRT):
    if old.exists():
        old.unlink()
        print(f"  removed {old.name}")

print(f"\nDone. Speakers de-identified:")
print(f"  students: {len(name_map.students)}")
print(f"  adults:   {len(name_map.adults)}")
