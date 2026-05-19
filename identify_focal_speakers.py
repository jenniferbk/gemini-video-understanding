#!/usr/bin/env python3
"""Frame-extract + Gemini structured-JSON speaker identification for small-group videos.

Better than v10's `identify` for small-group recordings because v10 only samples
the first 2 chunks (~1:45) and misses speakers that arrive later or change
visibility. This script samples frames spread across the whole video.

Usage:
    python3 identify_focal_speakers.py path/to/video.mp4
    python3 identify_focal_speakers.py path/to/video.mp4 --start 360  # skip first 6 min (e.g. for hybrid videos)
    python3 identify_focal_speakers.py path/to/video.mp4 --timepoints 60,300,900,1500,2400,3300

Writes <video_dir>/<video_stem>_speakers.json in v10-compatible format
(list of {label, description, type}).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Literal, Optional

from google.genai import types
from pydantic import BaseModel


class _Speaker(BaseModel):
    label: str
    description: str
    type: Literal["teacher", "student", "researcher"]


class _SpeakerList(BaseModel):
    speakers: List[_Speaker]


_PROMPT = """These frames are sampled from a single small-group classroom recording. \
The camera is focused on ONE focal table/group within a larger classroom. \
Identify every distinct person who is part of the FOCAL GROUP or interacts with it \
(students at the focal table, the teacher when she visits the group, and any researcher / \
adult observer who addresses the group).

For each speaker, provide:
1. A `label` that UNIQUELY identifies this person within the focal group. Use a feature \
that no other focal-group person shares (hair color/length, specific clothing, seating \
position). Avoid generic labels like "Student1". Examples: "Girl-BlondeBraids", \
"Boy-RedHoodie", "Teacher", "Researcher-TanCoat".

2. A `description` covering hair color/style, clothing specifics, position at the table, \
and any distinguishing accessories. Be specific — this label is used consistently across \
a long video.

3. A `type`: exactly one of "teacher", "student", "researcher".

EXCLUDE students at OTHER tables visible in the background — only label people in the \
focal group / interacting with it. If you see a transient person (someone who walks past \
once and never speaks), skip them.
"""


def _probe_duration_seconds(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _default_timepoints(duration_s: float, start_s: float) -> List[float]:
    """Six frames spread from start_s to end, biased toward the early/middle portion."""
    span = duration_s - start_s
    # 60s after start, then 10/25/40/60/80% of the remaining span
    return [
        start_s + 60,
        start_s + span * 0.10,
        start_s + span * 0.25,
        start_s + span * 0.40,
        start_s + span * 0.60,
        start_s + span * 0.80,
    ]


def _extract_frame(video_path: Path, t_seconds: float, out_path: Path) -> None:
    """Extract a single JPEG frame at t_seconds using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t_seconds:.2f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "3",  # high quality JPEG
            str(out_path),
        ],
        check=True,
    )


def _call_gemini(api_key: str, frame_paths: List[Path], model: str) -> List[dict]:
    """Send frames + prompt to Gemini, parse structured JSON response."""
    from google import genai
    client = genai.Client(api_key=api_key)

    # Upload frames as inline parts (small JPEGs, no need for files API)
    parts: List = []
    for fp in frame_paths:
        with open(fp, "rb") as f:
            data = f.read()
        parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
    parts.append(_PROMPT)

    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=_SpeakerList,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
    )

    resp = client.models.generate_content(model=model, contents=parts, config=cfg)
    parsed = _SpeakerList.model_validate_json(resp.text)
    return [s.model_dump() for s in parsed.speakers]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="Path to mp4 video")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Seconds to skip before sampling (e.g. for hybrid videos, set to small-group start time)")
    ap.add_argument("--timepoints", type=str, default=None,
                    help="Comma-separated frame timestamps in seconds (overrides default spread)")
    ap.add_argument("--model", default="gemini-3-flash-preview",
                    help="Gemini model for speaker identification")
    ap.add_argument("--keep-frames", action="store_true",
                    help="Don't delete extracted frames after success (helpful for debugging)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output manifest path (default: <video_dir>/<video_stem>_speakers.json)")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    duration = _probe_duration_seconds(args.video)
    print(f"Video duration: {duration / 60:.1f} min")

    if args.timepoints:
        timepoints = [float(t) for t in args.timepoints.split(",")]
    else:
        timepoints = _default_timepoints(duration, args.start)
    timepoints = [t for t in timepoints if t < duration]
    print(f"Sampling {len(timepoints)} frames at: " +
          ", ".join(f"{t/60:.1f}min" for t in timepoints))

    out_path = args.output or args.video.with_name(f"{args.video.stem}_speakers.json")

    with tempfile.TemporaryDirectory(prefix="identify_frames_") as tmp:
        tmp_dir = Path(tmp)
        frame_paths = []
        for i, t in enumerate(timepoints):
            fp = tmp_dir / f"frame_{i:02d}_t{int(t):05d}.jpg"
            _extract_frame(args.video, t, fp)
            print(f"  extracted {fp.name}")
            frame_paths.append(fp)

        if args.keep_frames:
            keep_dir = args.video.with_name(f"{args.video.stem}_frames")
            keep_dir.mkdir(exist_ok=True)
            for fp in frame_paths:
                (keep_dir / fp.name).write_bytes(fp.read_bytes())
            print(f"  kept frames at {keep_dir}")

        print(f"\nCalling Gemini ({args.model}) with {len(frame_paths)} frames...")
        speakers = _call_gemini(api_key, frame_paths, args.model)

    if not speakers:
        print("WARNING: Gemini returned no speakers — manifest will be empty", file=sys.stderr)

    print(f"\nDetected {len(speakers)} speaker(s):")
    for s in speakers:
        print(f"  [{s['type']}] {s['label']}: {s['description'][:80]}{'...' if len(s['description']) > 80 else ''}")

    # v10 manifest format: list of {label, description, type}
    out_path.write_text(json.dumps(speakers, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {out_path}")


if __name__ == "__main__":
    main()
