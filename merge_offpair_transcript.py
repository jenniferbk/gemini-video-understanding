#!/usr/bin/env python3
"""Merge an off-pair audio transcript into a video transcript on the video timeline.

The off-pair recorder is a stereo desk mic on two focal students ("Pair 2"); it hears
them close/loud but also picks up the rest of the room as faint bleed, which Gemini
tends to hallucinate. So the off-pair is used conservatively: energy-gate to keep only
close speech, then fill ONLY the gaps the video missed. The video stays authoritative.

See docs/superpowers/specs/2026-05-29-offpair-merge-design.md
"""
from __future__ import annotations

import re

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
