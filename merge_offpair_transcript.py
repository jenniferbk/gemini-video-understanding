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
