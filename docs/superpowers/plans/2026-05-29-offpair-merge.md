# Off-Pair Transcript Merge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable CLI tool that merges an off-pair audio transcript into a video transcript on the video timeline, energy-gating the off-pair so it only fills gaps the video missed (defending against bleed/hallucination).

**Architecture:** Single script `merge_offpair_transcript.py` of small pure functions (parsing, text similarity, FFT cross-correlation, linear time-map fit, RMS energy gating, Pair-2 detection, gap-fill merge, formatting) plus thin ffmpeg/audio-IO wrappers and a `main()` CLI. Pure functions are unit-tested with synthetic data; audio wrappers are validated end-to-end on SG2.

**Tech Stack:** Python 3, numpy (FFT + RMS), ffmpeg (audio extract), stdlib `wave`. No scipy. Tests via pytest (`tests/test_merge_offpair.py`).

**Spec:** `docs/superpowers/specs/2026-05-29-offpair-merge-design.md`

---

## File Structure

- Create: `merge_offpair_transcript.py` (repo root) — the whole tool.
- Create: `tests/test_merge_offpair.py` — unit tests for the pure functions.

All public types/functions defined once, used consistently:

```python
@dataclass
class Entry:
    time_s: float
    speaker: Optional[str]   # None for visual/non-speech lines
    text: str
    kind: str                # "speech" | "visual"
    source: str              # "video" | "offpair"

@dataclass
class TimeMap:
    a: float                 # drift (≈1.0); video_t = a*mp3_t + b
    b: float                 # base offset (seconds)
    windows: list            # [{"mp3_t":float,"video_t":float,"strength":float}]
    residual: float          # RMS fit residual (seconds)
    confidence: float        # min window strength used
    def map(self, mp3_t: float) -> float: ...

@dataclass
class PairMap:
    mapping: dict            # {"Speaker-A":"Student-Maya","Speaker-B":"Student-Omar"}
    confidence: float        # margin of winner over runner-up (0..1)
    def label_for(self, speaker: str) -> str: ...
```

Function inventory (signatures are contracts; later tasks must match exactly):

- `parse_transcript_text(text: str, source: str) -> list[Entry]`
- `text_similarity(a: str, b: str) -> float`  (0..1 token Jaccard)
- `cross_correlate_offset(ref: np.ndarray, sig: np.ndarray, sr: int) -> tuple[float, float]`  → `(offset_seconds, strength)`; position in `ref` where shorter `sig` best aligns.
- `fit_time_map(pairs: list[tuple[float, float]]) -> TimeMap`  (pairs are `(mp3_t, video_t)`)
- `rms_envelope(samples: np.ndarray, sr: int, hop_s: float = 0.5) -> np.ndarray`
- `choose_threshold(env: np.ndarray, k: float = 1.0) -> float`  (median + k·MAD)
- `is_close(env: np.ndarray, hop_s: float, t: float, threshold: float) -> bool`
- `is_student_speaker(speaker: Optional[str], teacher_markers=("Teacher","Ms.","Mr.","Mrs.")) -> bool`
- `video_has_coverage(video_entries: list[Entry], vt: float, window: float, teacher_markers) -> bool`
- `detect_pair2(video_entries, offpair_close_overlap, time_map, window=8.0) -> PairMap`
- `merge(video_entries, offpair_entries, time_map, env, hop_s, threshold, pair_map, window=8.0) -> list[Entry]`
- `format_transcript(entries: list[Entry], header_lines: list[str]) -> str`
- `build_audit(time_map, threshold, close_count, faint_count, pair_map, inserted, discarded, warnings) -> dict`
- `extract_audio(media_path: str) -> tuple[np.ndarray, int]`  (mono float32, 16000 Hz)
- `compute_time_map(video_media: str, offpair_mp3: str, n_windows: int = 4, win_s: float = 60.0) -> TimeMap`
- `offpair_energy(offpair_mp3: str, hop_s: float = 0.5, k: float = 1.0) -> tuple[np.ndarray, float, float]`  → `(env, hop_s, threshold)`
- `main()`

---

## Task 1: Scaffolding + transcript parsing

**Files:**
- Create: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge_offpair.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from merge_offpair_transcript import Entry, parse_transcript_text


def test_parse_speech_visual_and_skips_headers():
    text = (
        "================================================================\n"
        "COMPLETE TRANSCRIPT - V10\n"
        "Speakers: Student-Maya, Teacher-Lee\n"
        "================================================================\n"
        "\n"
        "--- CHUNK 54 (39:45 - 40:45) ---\n"
        "40:00 [Student-Omar types \"45\" into the turn block]\n"
        "40:03 Student-Maya: Spin it again.\n"
        "75:10 Teacher-Lee: Eyes up here, please.\n"
    )
    entries = parse_transcript_text(text, source="video")
    assert len(entries) == 3
    visual, maya, teacher = entries
    assert visual.kind == "visual" and visual.speaker is None and visual.time_s == 40 * 60
    assert maya.kind == "speech" and maya.speaker == "Student-Maya"
    assert maya.text == "Spin it again." and maya.time_s == 40 * 60 + 3
    assert teacher.time_s == 75 * 60 + 10  # minutes > 59 supported
    assert all(e.source == "video" for e in entries)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_parse_speech_visual_and_skips_headers -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'merge_offpair_transcript'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_TS_RE = re.compile(r"^(\d{1,3}):(\d{2})\s+(.*)$")


@dataclass
class Entry:
    time_s: float
    speaker: Optional[str]
    text: str
    kind: str          # "speech" | "visual"
    source: str        # "video" | "offpair"


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_parse_speech_visual_and_skips_headers -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): transcript parsing for off-pair merge tool"
```

---

## Task 2: Text similarity

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_text_similarity():
    from merge_offpair_transcript import text_similarity
    assert text_similarity("Spin it again!", "spin it again") == 1.0
    assert text_similarity("rotate it sideways", "no way") == 0.0
    mid = text_similarity("we counted to forty", "we counted to fifty")
    assert 0.4 < mid < 0.9
    assert text_similarity("", "anything") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_text_similarity -v`
Expected: FAIL — `ImportError: cannot import name 'text_similarity'`.

- [ ] **Step 3: Write minimal implementation**

Add to `merge_offpair_transcript.py`:

```python
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return set(_WORD_RE.findall(s.lower()))


def text_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity in [0, 1]; 0 if either side has no word tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_text_similarity -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): token Jaccard text similarity"
```

---

## Task 3: FFT cross-correlation offset

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cross_correlate_offset_finds_excerpt():
    import numpy as np
    from merge_offpair_transcript import cross_correlate_offset
    rng = np.random.default_rng(0)
    sr = 16000
    ref = rng.standard_normal(sr * 2).astype(np.float32)  # 2 s reference
    start = int(0.4 * sr)
    sig = ref[start:start + sr // 2].copy()                # 0.5 s excerpt at t=0.4s
    offset, strength = cross_correlate_offset(ref, sig, sr)
    assert abs(offset - 0.4) < 0.01
    assert strength > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_cross_correlate_offset_finds_excerpt -v`
Expected: FAIL — `ImportError: cannot import name 'cross_correlate_offset'`.

- [ ] **Step 3: Write minimal implementation**

Add `import numpy as np` to the top imports, then add:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_cross_correlate_offset_finds_excerpt -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): FFT cross-correlation offset finder"
```

---

## Task 4: Linear time-map fit

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fit_time_map_recovers_offset_and_drift():
    from merge_offpair_transcript import fit_time_map
    # true map: video_t = 1.002*mp3_t + 480   (drift + 8 min offset)
    pairs = [(t, 1.002 * t + 480.0) for t in (60.0, 1200.0, 2400.0, 3600.0)]
    tm = fit_time_map(pairs)
    assert abs(tm.a - 1.002) < 1e-4
    assert abs(tm.b - 480.0) < 0.5
    assert abs(tm.map(1800.0) - (1.002 * 1800.0 + 480.0)) < 0.5
    assert tm.residual < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_fit_time_map_recovers_offset_and_drift -v`
Expected: FAIL — `ImportError: cannot import name 'fit_time_map'`.

- [ ] **Step 3: Write minimal implementation**

Add the `TimeMap` dataclass and `fit_time_map`:

```python
@dataclass
class TimeMap:
    a: float
    b: float
    windows: list = field(default_factory=list)
    residual: float = 0.0
    confidence: float = 0.0

    def map(self, mp3_t: float) -> float:
        return self.a * mp3_t + self.b


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_fit_time_map_recovers_offset_and_drift -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): linear time-map fit (offset + drift)"
```

---

## Task 5: RMS energy envelope + close-speech gating

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_energy_envelope_and_gating():
    import numpy as np
    from merge_offpair_transcript import rms_envelope, choose_threshold, is_close
    sr = 16000
    quiet = 0.02 * np.random.default_rng(1).standard_normal(sr * 4).astype(np.float32)
    loud = quiet.copy()
    loud[sr * 1: sr * 2] += 0.8 * np.random.default_rng(2).standard_normal(sr).astype(np.float32)
    env = rms_envelope(loud, sr, hop_s=0.5)
    assert len(env) == 8  # 4 s / 0.5 s
    thr = choose_threshold(env, k=1.0)
    assert is_close(env, 0.5, 1.4, thr) is True     # inside the loud second
    assert is_close(env, 0.5, 3.2, thr) is False    # quiet region
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_energy_envelope_and_gating -v`
Expected: FAIL — `ImportError: cannot import name 'rms_envelope'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_energy_envelope_and_gating -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): RMS energy envelope + adaptive close-speech gating"
```

---

## Task 6: Pair-2 detection by content match

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_detect_pair2_maps_speakers_to_best_match():
    from merge_offpair_transcript import Entry, TimeMap, detect_pair2
    tm = TimeMap(a=1.0, b=0.0)
    video = [
        Entry(100.0, "Student-Maya", "spin it again you got this", "speech", "video"),
        Entry(130.0, "Student-Omar", "try the other arrow key", "speech", "video"),
        Entry(160.0, "Teacher-Lee", "eyes up here everyone", "speech", "video"),
    ]
    offpair_close_overlap = [
        Entry(101.0, "Speaker-A", "spin it again you got this", "speech", "offpair"),
        Entry(131.0, "Speaker-B", "try the other arrow key", "speech", "offpair"),
    ]
    pm = detect_pair2(video, offpair_close_overlap, tm, window=8.0)
    assert pm.mapping["Speaker-A"] == "Student-Maya"
    assert pm.mapping["Speaker-B"] == "Student-Omar"
    assert pm.label_for("Speaker-A") == "Student-Maya"
    assert pm.label_for("Speaker-Z") == "Speaker-Z"  # unknown passes through
    assert pm.confidence > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_detect_pair2_maps_speakers_to_best_match -v`
Expected: FAIL — `ImportError: cannot import name 'detect_pair2'`.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class PairMap:
    mapping: dict = field(default_factory=dict)
    confidence: float = 0.0

    def label_for(self, speaker: str) -> str:
        return self.mapping.get(speaker, speaker)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_detect_pair2_maps_speakers_to_best_match -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): Pair-2 detection via overlap content matching"
```

---

## Task 7: Gated gap-fill merge

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_merge_gated_gapfill():
    import numpy as np
    from merge_offpair_transcript import Entry, TimeMap, PairMap, merge
    tm = TimeMap(a=1.0, b=0.0)
    video = [
        Entry(100.0, "Student-Maya", "spin it again", "speech", "video"),      # reliable
        Entry(200.0, "Student-Maya", "[inaudible]", "speech", "video"),        # gap (inaudible)
        Entry(300.0, "Teacher-Lee", "eyes up here", "speech", "video"),            # teacher only -> gap for students
    ]
    offpair = [
        Entry(100.0, "Speaker-B", "spin it again", "speech", "offpair"),        # redundant -> drop
        Entry(200.0, "Speaker-B", "maybe its a hexagon", "speech", "offpair"),  # fills inaudible -> insert
        Entry(300.0, "Speaker-B", "try the green block", "speech", "offpair"),  # student gap under teacher -> insert
        Entry(400.0, "Speaker-B", "faint bleed words", "speech", "offpair"),   # faint -> drop
    ]
    # energy: close at 100/200/300, faint at 400 (hop_s=0.5 -> index = t/0.5)
    env = np.zeros(900)
    for t in (100.0, 200.0, 300.0):
        env[int(t / 0.5)] = 1.0
    threshold = 0.5
    pm = PairMap(mapping={"Speaker-B": "Student-Omar"}, confidence=0.9)
    merged = merge(video, offpair, tm, env, 0.5, threshold, pm, window=8.0)

    texts = [(e.time_s, e.speaker, e.text, e.source) for e in merged]
    # all video lines preserved
    assert (100.0, "Student-Maya", "spin it again", "video") in texts
    assert (300.0, "Teacher-Lee", "eyes up here", "video") in texts
    # inaudible-gap filled, relabeled, sourced offpair
    assert (200.0, "Student-Omar", "maybe its a hexagon", "offpair") in texts
    # student gap under teacher filled
    assert (300.0, "Student-Omar", "try the green block", "offpair") in texts
    # redundant + faint NOT inserted
    assert all(e.text != "spin it again" or e.source == "video" for e in merged)
    assert all(e.text != "faint bleed words" for e in merged)
    # sorted by time
    assert [e.time_s for e in merged] == sorted(e.time_s for e in merged)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_merge_gated_gapfill -v`
Expected: FAIL — `ImportError: cannot import name 'merge'` (and `video_has_coverage`).

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_merge_gated_gapfill -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): gated gap-fill merge engine"
```

---

## Task 8: Output formatting + audit

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test**

```python
def test_format_transcript_and_audit():
    from merge_offpair_transcript import Entry, TimeMap, PairMap, format_transcript, build_audit
    entries = [
        Entry(63.0, "Student-Maya", "spin it again", "speech", "video"),
        Entry(75.0, None, "[Student-Omar points at screen]", "visual", "video"),
        Entry(200.0, "Student-Omar", "maybe its a hexagon", "speech", "offpair"),
    ]
    out = format_transcript(entries, header_lines=["Unified transcript", "Source: SG2"])
    assert "Unified transcript" in out
    assert "01:03 Student-Maya: spin it again" in out
    assert "01:15 [Student-Omar points at screen]" in out
    assert "03:20 Student-Omar: maybe its a hexagon" in out

    tm = TimeMap(a=1.0, b=480.0, residual=0.2, confidence=0.8)
    pm = PairMap(mapping={"Speaker-B": "Student-Omar"}, confidence=0.9)
    audit = build_audit(tm, threshold=0.5, close_count=120, faint_count=300,
                        pair_map=pm, inserted=2, discarded=5, warnings=["low overlap"])
    assert audit["time_map"]["b"] == 480.0
    assert audit["energy"]["close_count"] == 120
    assert audit["pair2"]["mapping"]["Speaker-B"] == "Student-Omar"
    assert audit["counts"] == {"inserted": 2, "discarded": 5}
    assert audit["warnings"] == ["low overlap"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_format_transcript_and_audit -v`
Expected: FAIL — `ImportError: cannot import name 'format_transcript'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_format_transcript_and_audit -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): unified transcript formatting + audit builder"
```

---

## Task 9: Audio I/O, glue, CLI, and SG2 end-to-end validation

**Files:**
- Modify: `merge_offpair_transcript.py`
- Test: `tests/test_merge_offpair.py`

- [ ] **Step 1: Write the failing test** (round-trip the WAV reader; ffmpeg-dependent glue is validated manually in Step 6)

```python
def test_extract_audio_reads_wav(tmp_path):
    import wave
    import numpy as np
    from merge_offpair_transcript import extract_audio
    sr = 16000
    sig = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    pcm = (sig * 32767).astype("<i2")
    wav_path = tmp_path / "tone.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    samples, got_sr = extract_audio(str(wav_path))
    assert got_sr == sr
    assert len(samples) == sr
    assert float(np.max(np.abs(samples))) <= 1.0
    assert abs(float(np.max(np.abs(samples))) - 0.5) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_extract_audio_reads_wav -v`
Expected: FAIL — `ImportError: cannot import name 'extract_audio'`.

- [ ] **Step 3: Write minimal implementation** (audio wrappers + glue + CLI)

Add imports `import argparse, json, subprocess, sys, tempfile, wave` and `from pathlib import Path`, then:

```python
def extract_audio(media_path: str) -> Tuple["np.ndarray", int]:
    """Decode any media to mono 16 kHz float32 in [-1,1] via ffmpeg -> temp WAV -> numpy.
    If the input is already a WAV it is read directly when mono/16k, else re-encoded."""
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

    # close off-pair lines that overlap a reliable video student line -> learn identities
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_offpair.py::test_extract_audio_reads_wav -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `python3 -m pytest tests/test_merge_offpair.py -v`
Expected: all tests PASS.

- [ ] **Step 6: SG2 end-to-end validation (manual check, not a unit test)**

Run:
```bash
cd ~/Documents/COMS
python3 merge_offpair_transcript.py \
  --video-transcript batch_y2_cora_day2_sg/SG2/Y2_4M_Cora_Day2_SG2_transcript.txt \
  --offpair-transcript batch_y2_cora_day2_sg/Y2_4M_Cora_Day2_SG2_offpair_transcript.txt \
  --video-media "$HOME/Downloads/Y2_4M_Cora_Day2_SG2.mp4" \
  --offpair-mp3 "$HOME/Downloads/Y2_4M_Cora_Day2_SG2.MP3(2nd pair).MP3" \
  -o batch_y2_cora_day2_sg/SG2/Y2_4M_Cora_Day2_SG2_unified.txt
```
Then verify in the audit JSON + transcript:
- `time_map.b` ≈ −480..−540 s region and `a` ≈ 1.0 (off-pair clock behind video by ~8 min); residual small.
- The "landmark" off-pair segment is NOT inserted (it is redundant with the video's focal-student lines and/or faint) — `grep -c "<landmark line>" ...unified.txt` should match the video's count, not exceed it.
- Some genuine gap-fills appear where the video had `[inaudible]`.
Report the audit summary to Jennifer; if alignment or gating looks off, tune `--energy-k` / `--window`.

- [ ] **Step 7: Commit**

```bash
git add merge_offpair_transcript.py tests/test_merge_offpair.py
git commit -m "feat(merge): audio IO, time-map/energy glue, CLI, SG2 validation"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** sync (T3/T4/T9), parsing (T1), energy gating (T5), Pair-2 detection from high-energy overlaps (T6), gated gap-fill with no-override (T7), clean output + audit sidecar (T8), CLI + SG2 validation incl. the "landmark not inserted" bleed test (T9). Conservative-merge blast-radius note: even a mis-detected Pair-2 label only mislabels an inserted gap-fill line; it never overrides video content.
- **Placeholders:** none — every step has runnable code/commands.
- **Type consistency:** `Entry`, `TimeMap`, `PairMap` and all signatures match across tasks; `video_has_coverage`/`is_student_speaker`/`is_close` reused with identical signatures.
- **Out of scope (per spec):** SG1 deid fix, cross-video pseudonym normalization, channel separation.
