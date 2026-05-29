# Off-Pair Transcript Merge — Design

**Date:** 2026-05-29
**Status:** Approved (pending spec review)
**Tool:** `merge_offpair_transcript.py` (new, reusable)

## Problem

Year-2 small-group recordings use a second audio recorder. The focal group is four
students; the **video's own audio** captures the group (Pair 1 clearly), and a separate
**off-pair MP3** comes from a stereo desk recorder sitting with the other two students
("Pair 2").

We already produce two independent transcripts per session:

- **Video transcript** (v10): video timeline, pseudonym/appearance labels, visual
  brackets, teacher, both pairs (Pair 2 weakly). **Reliable** — this is the spine.
- **Off-pair transcript** (`transcribe_offpair_audio.py`): the MP3's *own* timeline,
  `Speaker-A`/`Speaker-B` labels, no visuals.

**Clocks are not aligned.** Recorders/camera started independently (observed offsets:
SG1 ≈ +3 min, SG2 ≈ −7 min) and may drift over 80–90 min. Landmark: the SG2 off-pair
"landmark phrase" exchange at off-pair ~31:23 ↔ video ~40:00.

## Recording reality (drives the design)

Channel analysis of the off-pair MP3 (SG2): `L−R` is ~9–10 dB **below** `L+R`, so the
two channels are largely the **same acoustic scene** (closely-spaced stereo mics) — **not**
one student per channel. Implications:

- A mono down-mix is fine; channel-splitting would not cleanly separate the two students.
- The desk recorder hears **Pair 2 loud and close**, but also picks up **Pair 1 / teacher
  / room as faint bleed**. Gemini transcribing that faint bleed is the **hallucination
  source** (user's concern, confirmed).
- The "landmark" content found in *both* transcripts is almost certainly **loud Pair-1
  bleed** into the Pair-2 recorder — i.e., naive content-matching would mis-assign a Pair-1
  student as Pair 2 and overwrite good video lines with hallucinated bleed.

**Core principle:** separate **close speech (reliable)** from **faint bleed
(hallucination-prone)** via **audio energy**, and let the off-pair only *fill gaps* the
video missed — never override reliable video content.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| What is the off-pair recorder? | Stereo desk recorder on Pair 2 (same scene both channels), not per-student lavaliers. |
| Merge philosophy | **Conservative gated gap-fill** — off-pair adds a line only where it is high-energy/close AND the video has a gap or `[inaudible]` for a focal student. |
| Bleed handling | **Audio-energy gating**: trust only high-energy (close) off-pair segments; faint bleed excluded. |
| Identify Pair 2 / map A,B | Content-match **only on high-energy off-pair lines that overlap a video focal-student line**; fall back to `Speaker-A`/`B` if low confidence. |
| Scope | **Reusable** tool for all Y2 paired-mic recordings. |
| Sync method | **Audio cross-correlation**, piecewise for drift; teacher-line matches as cross-check. |
| Provenance | **Clean transcript text + sidecar `_merge_audit.json`** (no inline tags). |
| SG1 ordering | Build/validate on **SG2 first**; run SG1 only after its manual deid. |

## Inputs / Outputs

**Inputs:** video transcript `.txt`, off-pair transcript `.txt`, video media file,
off-pair MP3, output path.

**Outputs:**
- Unified transcript `.txt` in v10 format on the video timeline, Transana-compatible.
  Clean text; off-pair gap-fills carry detected student labels (or `Speaker-A/B`).
- Sidecar `<output>_merge_audit.json`: time map (`a`, `b`, residuals/strength); energy
  threshold + count of off-pair lines classed close vs faint; Pair-2 detection +
  confidence; gap-fills inserted; off-pair lines discarded (faint bleed / redundant);
  warnings.

## Components (isolated units)

### 1. `compute_time_map(video_media, offpair_mp3) -> TimeMap`
- ffmpeg-extract both audios to mono 16 kHz WAV (temp).
- Piecewise FFT cross-correlation in 3–5 windows across the overlap; least-squares fit
  `video_t = a*mp3_t + b` (`a`≈1 = drift, `b` = offset). Drop weak-peak windows.
- Cross-check vs shared teacher whole-class utterances; warn on divergence.
- numpy FFT only (no scipy). Returns `TimeMap{a, b, windows[], residual, confidence}`.

### 2. `parse_transcript(path, source) -> list[Entry]`
- Entry: `{time_s, speaker|None, text, kind: speech|visual, source}`. Handles `MM:SS`
  (minutes > 59); `[...]`-only lines are `visual`; skips chunk/file headers.

### 3. `offpair_energy(offpair_mp3) -> EnergyEnvelope`
- Compute a short-window RMS envelope (e.g., 0.5 s hop) over the off-pair audio.
- Derive a **close-speech threshold** adaptively (e.g., median + k·MAD, or a percentile),
  configurable via CLI; record the chosen value + distribution in the audit.
- `is_close(t)` → True if energy at off-pair time `t` is above threshold.
- Each off-pair line is classified **close** (trust) or **faint** (bleed → drop).

### 4. `detect_pair2(video_entries, offpair_entries_close, time_map) -> PairMap`
- Use **only close off-pair lines that DO overlap** a video focal-student speech line
  (within ±~8 s after mapping) — these are genuine shared moments, bleed excluded.
- Score text similarity; tally per (offpair_speaker → video_speaker); assign A,B to their
  best video speakers (= Pair 2), with a confidence margin.
- Low confidence / A==B → keep `Speaker-A/B` labels on inserted lines and warn.

### 5. `merge(video_entries, offpair_entries, time_map, energy, pair_map) -> list[Entry]`
- Spine = **all** video entries on the video timeline (never overridden).
- For each off-pair speech line:
  - Drop if **faint** (energy below threshold → bleed/hallucination-prone).
  - If **close**: map to video time; insert **only if the video has no reliable
    focal-student speech at that moment** — i.e., a gap, or only `[inaudible]`/missing for
    a focal student within ±~8 s. Otherwise discard (redundant or bleed of captured talk).
  - Label inserted lines via `pair_map` (or `Speaker-A/B` on low confidence).
- Merge-sort by timestamp; guard against duplicates.

### 6. `write_outputs(merged, audit, out_path)`
- Unified `.txt` (v10 format) + `_merge_audit.json` sidecar.

## Edge Cases & Risks

- **Energy threshold sensitivity:** configurable; audit reports the distribution and
  close/faint counts so the threshold can be tuned. If gating removes almost everything,
  or almost nothing, warn (recorder gain may make bleed loud too).
- **Weak audio correlation:** low sync confidence → fall back to landmark/manual offset
  with a warning; never emit a silent bad map.
- **Non-linear drift:** linear-fit residuals flag it.
- **Pair-2 detection ambiguous:** keep `Speaker-A/B`, warn — do not guess identities.
- **SG1 deid garbled:** content-matching needs reliable video labels → SG1 runs only
  after manual deid. SG2 mergeable now.
- **Approximate Gemini timestamps:** real sync is the audio map; per-line placement uses
  mapped transcript timestamps — adequate for a readable, analyzable transcript.

## Testing / Validation

- **Sync:** SG2 map places off-pair ~31:23 at video ~40:00; `a` ≈ 1.0; small residuals.
- **Bleed handling (key):** the "landmark" off-pair segment (~31:23) is classified
  **faint** and/or **redundant** (video already has the focal student's version) and is **NOT inserted**.
- **Gap-fill:** genuine quiet Pair-2 talk during a video `[inaudible]`/gap **is** inserted.
- **Audit JSON:** counts reconcile (close vs faint; inserted vs discarded).

## Tooling / Dependencies

- New script `merge_offpair_transcript.py` at repo root (mirrors `transcribe_offpair_audio.py`).
- Dependencies: `numpy` (FFT xcorr + RMS envelope), `ffmpeg`. No scipy.

## Out of Scope

- Fixing SG1's deid (handled manually).
- Cross-video pseudonym normalization (known limitation).
- Re-running transcription; this tool only merges existing transcripts.
- Per-student channel separation (channels are the same scene — not feasible here).
