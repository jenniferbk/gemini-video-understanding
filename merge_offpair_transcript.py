#!/usr/bin/env python3
"""Merge an off-pair audio transcript into a video transcript on the video timeline.

The off-pair recorder is a stereo desk mic on two focal students ("Pair 2"); it hears
them close/loud but also picks up the rest of the room as faint bleed, which Gemini
tends to hallucinate. So the off-pair is used conservatively: energy-gate to keep only
close speech, then fill ONLY the gaps the video missed. The video stays authoritative.

See docs/superpowers/specs/2026-05-29-offpair-merge-design.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from dataclasses import dataclass, field  # field used by later tasks  # noqa: F401
from typing import List, Literal, Optional, Tuple  # Tuple used by later tasks  # noqa: F401

_TS_RE = re.compile(r"^(\d{1,3}):(\d{2})\s+(.*)$")


@dataclass
class Entry:
    time_s: float
    speaker: Optional[str]
    text: str
    kind: Literal["speech", "visual"]
    source: Literal["video", "offpair"]


@dataclass
class TimeMap:
    a: float
    b: float
    windows: list = field(default_factory=list)
    residual: float = 0.0
    confidence: float = 0.0

    def map(self, mp3_t: float) -> float:
        return self.a * mp3_t + self.b


@dataclass
class PairMap:
    mapping: dict = field(default_factory=dict)
    confidence: float = 0.0

    def label_for(self, speaker: str) -> str:
        return self.mapping.get(speaker, speaker)


def parse_transcript_text(text: str, source: str) -> List[Entry]:
    """Parse a v10 / off-pair transcript into timestamped entries.

    Lines look like 'MM:SS Speaker: words' (speech) or 'MM:SS [action]' (visual).
    Chunk headers ('--- CHUNK ... ---'), file headers, and blank lines are skipped.
    Minutes may exceed 59 (e.g. '75:10').
    """
    entries: List[Entry] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TS_RE.match(line)
        if not m:
            continue  # header / banner / non-timestamped line
        mm, ss, rest = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        if ss >= 60:
            continue  # malformed seconds field (e.g. '40:60')
        if not rest:
            continue  # timestamp with no body (e.g. '41:00 ')
        t = mm * 60 + ss
        if rest.startswith("[") and rest.endswith("]"):
            entries.append(Entry(t, None, rest, "visual", source))
            continue
        # speech: 'Speaker: text' — speaker labels never contain a colon
        sm = re.match(r"^([^:\[]+):\s*(.*)$", rest)
        if sm:
            entries.append(Entry(t, sm.group(1).strip(), sm.group(2).strip(), "speech", source))
        else:
            entries.append(Entry(t, None, rest, "visual", source))
    return entries


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return set(_WORD_RE.findall(s.lower()))


def text_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity in [0, 1]; 0 if either side has no word tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cross_correlate_offset(ref: "np.ndarray", sig: "np.ndarray", sr: int) -> Tuple[float, float]:
    """Find the offset (seconds from ref start) where the shorter `sig` best aligns
    inside `ref`. Returns (offset_seconds, strength) where strength is the normalized
    correlation peak in roughly [0, 1] (1.0 = perfect match)."""
    ref = np.asarray(ref, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    ref = ref - ref.mean()
    sig = sig - sig.mean()
    nfft = 1 << int(np.ceil(np.log2(len(ref) + len(sig))))
    corr = np.fft.irfft(np.fft.rfft(ref, nfft) * np.conj(np.fft.rfft(sig, nfft)), nfft)
    valid = corr[: len(ref) - len(sig) + 1]
    idx = int(np.argmax(valid))
    window = ref[idx: idx + len(sig)]
    denom = np.linalg.norm(window) * np.linalg.norm(sig) + 1e-9
    return idx / sr, float(valid[idx] / denom)


def rms_envelope(samples: "np.ndarray", sr: int, hop_s: float = 0.5) -> "np.ndarray":
    """RMS per non-overlapping hop window. Returns a 1-D array (one value per hop)."""
    samples = np.asarray(samples, dtype=np.float64)
    hop = max(1, int(round(hop_s * sr)))
    n = len(samples) // hop
    if n == 0:
        return np.array([np.sqrt(np.mean(samples ** 2))]) if len(samples) else np.array([0.0])
    trimmed = samples[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def choose_threshold(env: "np.ndarray", k: float = 1.0) -> float:
    """Adaptive close-speech threshold: median + k * MAD of the envelope."""
    env = np.asarray(env, dtype=np.float64)
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    return med + k * mad


def is_close(env: "np.ndarray", hop_s: float, t: float, threshold: float) -> bool:
    """True if the off-pair energy at time t (seconds) is at/above threshold (close speech)."""
    idx = int(t // hop_s)
    if idx < 0 or idx >= len(env):
        return False
    return bool(env[idx] >= threshold)


def fit_time_map(pairs: List[Tuple[float, float]]) -> TimeMap:
    """Least-squares fit video_t = a*mp3_t + b over (mp3_t, video_t) pairs.

    With a single pair, assume no drift (a=1). Residual is RMS error in seconds.
    """
    if not pairs:
        raise ValueError("fit_time_map needs at least one (mp3_t, video_t) pair")
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)
    if len(pairs) == 1:
        a, b = 1.0, float(ys[0] - xs[0])
    else:
        a, b = np.polyfit(xs, ys, 1)
    resid = float(np.sqrt(np.mean((ys - (a * xs + b)) ** 2)))
    return TimeMap(a=float(a), b=float(b), residual=resid)


def is_student_speaker(speaker: Optional[str],
                       teacher_markers=("Teacher", "Ms.", "Mr.", "Mrs.")) -> bool:
    """True if the label looks like a student (not the teacher / not a visual line)."""
    if not speaker:
        return False
    return not any(marker in speaker for marker in teacher_markers)


def detect_pair2(video_entries: List[Entry],
                 offpair_close_overlap: List[Entry],
                 time_map: TimeMap,
                 window: float = 8.0) -> PairMap:
    """Learn off-pair Speaker-A/B -> student identity from off-pair lines that overlap a
    video student line (bleed excluded by the caller via energy gating). For each off-pair
    speaker, accumulate text-similarity mass per candidate video student; assign the best.
    Confidence = winner mass margin over runner-up, averaged across assigned speakers."""
    from collections import defaultdict
    vid = [e for e in video_entries if e.kind == "speech" and is_student_speaker(e.speaker)]
    mass: dict = defaultdict(lambda: defaultdict(float))
    for off in offpair_close_overlap:
        if off.kind != "speech":
            continue
        vt = time_map.map(off.time_s)
        for ve in vid:
            if abs(ve.time_s - vt) <= window:
                sim = text_similarity(off.text, ve.text)
                if sim > 0:
                    mass[off.speaker][ve.speaker] += sim
    mapping, margins = {}, []
    for off_spk, cands in mass.items():
        ranked = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            continue
        mapping[off_spk] = ranked[0][0]
        total = sum(v for _, v in ranked) or 1.0
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        margins.append((ranked[0][1] - runner) / total)
    confidence = float(sum(margins) / len(margins)) if margins else 0.0
    return PairMap(mapping=mapping, confidence=confidence)


def video_has_coverage(video_entries: List[Entry], vt: float, window: float,
                       teacher_markers=("Teacher", "Ms.", "Mr.", "Mrs.")) -> bool:
    """True if the video already has RELIABLE student speech near vt (so an off-pair line
    there is redundant/bleed). Teacher-only moments and [inaudible] do NOT count as
    coverage — those are gaps the off-pair may fill."""
    for ve in video_entries:
        if ve.kind != "speech" or not is_student_speaker(ve.speaker, teacher_markers):
            continue
        if "[inaudible]" in ve.text.lower() or not ve.text.strip():
            continue
        if abs(ve.time_s - vt) <= window:
            return True
    return False


def merge(video_entries: List[Entry],
          offpair_entries: List[Entry],
          time_map: TimeMap,
          env: "np.ndarray",
          hop_s: float,
          threshold: float,
          pair_map: PairMap,
          window: float = 8.0) -> List[Entry]:
    """Video is the spine (all entries kept). Add an off-pair speech line only if it is
    close (energy >= threshold) AND the video has no reliable student coverage at the
    mapped time. Inserted lines are relabeled via pair_map and tagged source='offpair'."""
    merged: List[Entry] = list(video_entries)
    for off in offpair_entries:
        if off.kind != "speech":
            continue
        if not is_close(env, hop_s, off.time_s, threshold):
            continue  # faint bleed -> drop
        vt = time_map.map(off.time_s)
        if video_has_coverage(video_entries, vt, window):
            continue  # redundant / captured talk -> drop
        merged.append(Entry(vt, pair_map.label_for(off.speaker), off.text, "speech", "offpair"))
    merged.sort(key=lambda e: e.time_s)
    return merged


def _fmt_ts(t: float) -> str:
    total = int(round(t))
    return f"{total // 60:02d}:{total % 60:02d}"


def format_transcript(entries: List[Entry], header_lines: List[str]) -> str:
    lines = list(header_lines) + [""]
    for e in entries:
        if e.kind == "visual":
            lines.append(f"{_fmt_ts(e.time_s)} {e.text}")
        else:
            lines.append(f"{_fmt_ts(e.time_s)} {e.speaker}: {e.text}")
    return "\n".join(lines) + "\n"


def build_audit(time_map: TimeMap, threshold: float, close_count: int, faint_count: int,
                pair_map: PairMap, inserted: int, discarded: int, warnings: list) -> dict:
    return {
        "time_map": {"a": time_map.a, "b": time_map.b,
                     "residual": time_map.residual, "confidence": time_map.confidence,
                     "windows": time_map.windows},
        "energy": {"threshold": threshold, "close_count": close_count,
                   "faint_count": faint_count},
        "pair2": {"mapping": pair_map.mapping, "confidence": pair_map.confidence},
        "counts": {"inserted": inserted, "discarded": discarded},
        "warnings": warnings,
    }


def extract_audio(media_path: str) -> Tuple["np.ndarray", int]:
    """Decode any media to mono 16 kHz float32 in [-1,1] via ffmpeg -> temp WAV -> numpy."""
    sr = 16000
    with tempfile.TemporaryDirectory(prefix="merge_audio_") as tmp:
        wav_path = Path(tmp) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(media_path), "-ac", "1", "-ar", str(sr),
             "-c:a", "pcm_s16le", str(wav_path)],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sr


def compute_time_map(video_media: str, offpair_mp3: str,
                     n_windows: int = 4, win_s: float = 60.0) -> TimeMap:
    """Estimate video_t = a*mp3_t + b by cross-correlating n_windows off-pair slices
    against the full video audio. Drops windows whose correlation strength < 0.15."""
    vid, sr = extract_audio(video_media)
    mp3, _ = extract_audio(offpair_mp3)
    mp3_dur = len(mp3) / sr
    win = int(win_s * sr)
    centers = np.linspace(0.15, 0.85, n_windows) * mp3_dur
    pairs, windows = [], []
    for c in centers:
        start = int(max(0, c * sr - win // 2))
        sig = mp3[start:start + win]
        if len(sig) < win // 2:
            continue
        offset, strength = cross_correlate_offset(vid, sig, sr)
        mp3_t = start / sr
        windows.append({"mp3_t": mp3_t, "video_t": offset, "strength": strength})
        if strength >= 0.15:
            pairs.append((mp3_t, offset))
    if not pairs:
        raise RuntimeError("time-map: no window correlated above 0.15 — audio may not overlap")
    tm = fit_time_map(pairs)
    tm.windows = windows
    tm.confidence = min(w["strength"] for w in windows if w["strength"] >= 0.15)
    return tm


def offpair_energy(offpair_mp3: str, hop_s: float = 0.5, k: float = 1.0):
    samples, sr = extract_audio(offpair_mp3)
    env = rms_envelope(samples, sr, hop_s=hop_s)
    return env, hop_s, choose_threshold(env, k=k)


def main():
    ap = argparse.ArgumentParser(description="Merge off-pair transcript into video transcript")
    ap.add_argument("--video-transcript", required=True)
    ap.add_argument("--offpair-transcript", required=True)
    ap.add_argument("--video-media", required=True)
    ap.add_argument("--offpair-mp3", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--energy-k", type=float, default=1.0, help="close-speech threshold = median + k*MAD")
    ap.add_argument("--window", type=float, default=8.0, help="alignment match window (s)")
    ap.add_argument("--min-pair2-confidence", type=float, default=0.15,
                    help="below this, keep Speaker-A/B labels instead of student names")
    args = ap.parse_args()

    video_entries = parse_transcript_text(Path(args.video_transcript).read_text("utf-8"), "video")
    offpair_entries = parse_transcript_text(Path(args.offpair_transcript).read_text("utf-8"), "offpair")

    warnings: list = []
    tm = compute_time_map(args.video_media, args.offpair_mp3)
    if tm.residual > 5.0:
        warnings.append(f"time-map residual {tm.residual:.1f}s — alignment may be poor")
    env, hop_s, threshold = offpair_energy(args.offpair_mp3, k=args.energy_k)

    close_overlap = [e for e in offpair_entries
                     if e.kind == "speech" and is_close(env, hop_s, e.time_s, threshold)
                     and video_has_coverage(video_entries, tm.map(e.time_s), args.window)]
    pair_map = detect_pair2(video_entries, close_overlap, tm, window=args.window)
    if pair_map.confidence < args.min_pair2_confidence:
        warnings.append(f"low Pair-2 confidence {pair_map.confidence:.2f} — keeping Speaker-A/B labels")
        pair_map = PairMap(mapping={}, confidence=pair_map.confidence)

    close_count = sum(1 for e in offpair_entries
                      if e.kind == "speech" and is_close(env, hop_s, e.time_s, threshold))
    faint_count = sum(1 for e in offpair_entries if e.kind == "speech") - close_count

    merged = merge(video_entries, offpair_entries, tm, env, hop_s, threshold, pair_map,
                   window=args.window)
    inserted = sum(1 for e in merged if e.source == "offpair")
    discarded = sum(1 for e in offpair_entries if e.kind == "speech") - inserted

    header = [
        "=" * 80,
        "UNIFIED TRANSCRIPT (video + off-pair gap-fill, video timeline)",
        "=" * 80,
        f"Video transcript: {Path(args.video_transcript).name}",
        f"Off-pair transcript: {Path(args.offpair_transcript).name}",
        f"Time map: video_t = {tm.a:.5f}*mp3_t + {tm.b:.1f}  (residual {tm.residual:.2f}s)",
        f"Off-pair lines inserted: {inserted}  |  discarded (faint/redundant): {discarded}",
        "NOTE: off-pair fills only gaps the video missed; video is authoritative elsewhere.",
        "=" * 80,
    ]
    Path(args.output).write_text(format_transcript(merged, header), encoding="utf-8")
    audit = build_audit(tm, threshold, close_count, faint_count, pair_map,
                        inserted, discarded, warnings)
    audit_path = str(Path(args.output).with_name(Path(args.output).stem + "_merge_audit.json"))
    Path(audit_path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {audit_path}")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
