# Off-Pair Transcript Merge — Design

**Date:** 2026-05-29
**Status:** Approved (pending spec review)
**Tool:** `merge_offpair_transcript.py` (new, reusable)

## Problem

Year-2 small-group recordings use individual microphone pairs. The focal group is
four students split into two pairs:

- **Pair 1** — well captured in the video's own embedded audio.
- **Pair 2 ("off-pair")** — captured on a separate MP3; quieter / less reliable in the
  video's audio.

We already produce two independent transcripts per session:

- **Video transcript** (v10): video timeline, appearance/pseudonym speaker labels,
  visual brackets, teacher, both pairs (Pair 2 weakly).
- **Off-pair transcript** (`transcribe_offpair_audio.py`): the MP3's *own* timeline,
  generic `Speaker-A`/`Speaker-B` labels, no visuals.

The two clocks are **not aligned** — the recorders and camera were started/stopped
independently (observed offsets: SG1 ≈ +3 min, SG2 ≈ −7 min on the MP3 vs video) and
may drift over 80–90 min. Confirmed by a content landmark: the SG2 off-pair "landmark
phrase" exchange at off-pair ~31:23 corresponds to video ~40:00.

**Goal:** one unified transcript on the **video timeline** in which the off-pair is
authoritative for Pair 2's speech, and the video remains authoritative for Pair 1, the
teacher, and all visual brackets.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| What is the off-pair mic? | Pair 2 — a *different* two students than the video favors. |
| Merge philosophy | Off-pair is **authoritative for its pair**; video authoritative for everything else. |
| Identify Pair 2 / map A,B | **Auto-detect by content match** after time-alignment. |
| Scope | **Reusable** tool for all Y2 paired-mic recordings. |
| Sync method | **Audio cross-correlation**, piecewise for drift; teacher-line matches as cross-check. |
| Provenance | **Clean transcript text + sidecar `_merge_audit.json`** (no inline tags). |
| SG1 ordering | Build/validate on **SG2 first**; run SG1 only after its manual deid (labels currently garbled). |

## Inputs / Outputs

**Inputs:** video transcript `.txt`, off-pair transcript `.txt`, video media file,
off-pair MP3, output path.

**Outputs:**
- Unified transcript `.txt` in v10 format (`MM:SS Speaker: text` + visual brackets),
  on the video timeline, Transana-compatible. Clean text, no inline provenance.
- Sidecar `<output>_merge_audit.json`: time map (`a`, `b`, per-window residuals/strength),
  Pair-2 detection (A→student, B→student, confidence margins), counts (video lines kept,
  off-pair lines inserted, Pair-2 video lines superseded), and any warnings.

## Components (isolated units)

### 1. `compute_time_map(video_media, offpair_mp3) -> TimeMap`
- ffmpeg-extract both audios to mono 16 kHz WAV (temp files).
- Piecewise FFT cross-correlation in 3–5 windows spread across the overlap region: for
  each window take a slice of the off-pair signal and cross-correlate against a search
  region of the video signal to find the local offset and its peak strength.
- Least-squares fit `video_t = a * mp3_t + b` over the window-center pairs; `a` captures
  drift (≈1.0 expected), `b` the base offset. Drop windows with weak/ambiguous peaks.
- **Cross-check (approach B):** match a few shared teacher whole-class utterances between
  the two transcripts; compare predicted vs actual mapping; emit a warning if divergent.
- Returns `TimeMap{a, b, windows[], residual, confidence}`.
- Uses numpy FFT only (no scipy dependency).

### 2. `parse_transcript(path, source) -> list[Entry]`
- Entry: `{time_s: float, speaker: str|None, text: str, kind: "speech"|"visual", source}`.
- Handles `MM:SS` timestamps including minutes > 59; treats `[...]`-only lines as
  `visual` (no speaker); skips `--- CHUNK n ---` headers and file headers.

### 3. `detect_pair2(video_entries, offpair_entries, time_map) -> PairMap`
- Map each off-pair entry's time to video time via `time_map`.
- For each off-pair speech entry, gather video speech entries within ±~8 s and score text
  similarity (normalized token-overlap / fuzzy ratio).
- Accumulate similarity mass per (offpair_speaker → video_speaker). Assign `Speaker-A`
  and `Speaker-B` to their highest-mass video speakers; those two = Pair 2.
- Report confidence as the margin over the next-best candidate. If A and B collapse to
  one speaker, or confidence is low, **keep A/B labels** and warn.

### 4. `merge(video_entries, offpair_entries, time_map, pair_map) -> list[Entry]`
- Spine = video entries on the video timeline.
- Re-timestamp off-pair speech entries to video time via `time_map`; relabel A/B →
  detected student identities (or leave A/B if detection was low-confidence).
- **"Authoritative for its pair" = off-pair wins on overlap, no data loss:**
  - A video *speech* line whose speaker ∈ Pair 2 is **dropped (superseded)** only if an
    aligned off-pair line covers the same moment (within ±~8 s).
  - A Pair-2 video line with **no** aligned off-pair counterpart is **kept** (the off-pair
    mic may have dropped out / the student turned away — the video is then the only record).
  - This avoids discarding real utterances while still giving the cleaner off-pair priority
    wherever both captured the same talk. (Chosen over strict full-replacement, which would
    lose Pair-2 utterances the off-pair missed.)
- Keep all video visuals, teacher, and Pair 1 lines untouched.
- Merge-sort both streams by timestamp; guard against duplicate lines.

### 5. `write_outputs(merged, audit, out_path)`
- Write the unified `.txt` (v10 format) and the `_merge_audit.json` sidecar.

## Edge Cases & Risks

- **Weak audio correlation** (mics share little signal): low confidence → fall back to
  landmark-utterance map (B), else prompt for a manual offset. Never silently emit a
  bad map.
- **Non-linear drift:** the linear fit's residuals flag it; warn if large.
- **Ambiguous Pair-2 detection:** keep `Speaker-A`/`Speaker-B` and warn rather than
  guess identities.
- **SG1 deid is garbled:** content-matching needs reliable video labels, so SG1 runs
  only after manual deid. SG2 is mergeable immediately.
- **Approximate Gemini timestamps:** real sync comes from the audio map; per-line
  placement uses each transcript's own (mapped) timestamps — adequate for a readable,
  analyzable transcript, not frame-accurate captioning.

## Testing / Validation

- **Sync:** on SG2, the map must place off-pair ~31:23 at video ~40:00 (the "landmark"
  landmark); drift `a` ≈ 1.0; residuals small.
- **Pair-2 detection:** returns two distinct students with a clear confidence margin on SG2.
- **Merge:** spot-check output around the landmark moment — Pair-2 lines present, labeled
  with detected identities, video visuals/teacher/Pair-1 intact, no duplicates.
- **Audit JSON:** counts reconcile (video kept + off-pair inserted; Pair-2 video lines
  superseded equals expected).

## Tooling / Dependencies

- New script `merge_offpair_transcript.py` at repo root (mirrors `transcribe_offpair_audio.py`).
- Dependencies: `numpy` (FFT cross-correlation), `ffmpeg` (already used). No scipy.

## Out of Scope

- Fixing SG1's deid (handled separately/manually).
- Cross-video pseudonym normalization (known limitation; not addressed here).
- Re-running transcription; this tool only merges existing transcripts.
