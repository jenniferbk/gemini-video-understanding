# TIMSS US1 benchmark — system comparison

All systems scored against the reviewer-corrected gold transcript (8 corrections applied: 3 inaudible over-transcriptions removed, 5 text corrections for transcription mistakes in the published TIMSS gold).

Primary metric is **content-equivalence WER** — a time-windowed word-set membership score that tolerates arbitrary turn division, ±2s timestamp jitter, punctuation/contraction normalization, hyphen compounds, number/word interchange, and inaudible-marker invisibility. Rules are identical to the human review tool's highlighting logic, so the metric and the visual review can never drift apart.

Strict WER (last column) is the raw jiwer number on normalized text, included for calibration against published ASR literature.

Speaker role accuracy (role acc) uses a **content-aware** alignment: for each gold turn, find the pred turn with the highest F1 of word-set overlap within ±12 seconds, with role-aware tiebreaking (when multiple pred turns have near-tied F1, prefer the one whose role matches gold — which correctly credits systems for classroom call-and-response where teacher and student say the same short phrase). Unmatched gold turns (no pred turn reached 0.5 F1) are **excluded** from the speaker accuracy denominator — they're a content-WER issue, not a speaker-ID issue. Nearest-timestamp alignment is much less informative here because classroom transcripts are teacher-dominated: a brief student utterance's nearest pred is almost always an adjacent teacher turn, producing spurious student→teacher errors.


## Headline comparison

| system | recall | precision | F1 | content WER | role acc | T acc | S acc | strict WER |
|---|---|---|---|---|---|---|---|---|
| Whisper large-v3 (alone) | 96.1% | 97.6% | 96.8% | 6.3% | 62.6% | 100.0% | 0.0% | 30.7% |
| Whisper + pyannote 3.1 | 96.1% | 97.6% | 96.8% | 6.3% | 75.4% | 84.0% | 61.1% | 30.7% |
| v10 (this work) — run 1 | 95.6% | 98.1% | 96.8% | 6.1% | 92.0% | 97.5% | 83.3% | 22.1% |
| v10 (this work) — run 2 (replicate) | 95.5% | 97.9% | 96.7% | 6.3% | 93.1% | 98.0% | 84.8% | 22.9% |

**Reference:** 7622 gold content words (936 gold turns after correction).

## v10 replicate variance

Across 2 independent runs of the same config on the same video (temperature 0.2, so some stochasticity is expected):

| metric | mean | range (max − min) |
|---|---|---|
| recall | 95.56% | 0.04% |
| precision | 97.97% | 0.23% |
| F1 | 96.75% | 0.13% |
| content WER | 6.21% | 0.22% |
| speaker accuracy | 92.52% | 1.10% |

**Takeaway:** content metrics are stable across replicates (<1 pp spread). Speaker accuracy is the noisier axis — Gemini's visual-feature diarization is deterministic in principle but the same video can produce somewhat different speaker clusterings across runs because individual-student labels depend on which visual features the model latches onto.

## Key findings

1. **On the speech content axis, all three systems tie.** Whisper large-v3, Whisper+pyannote, and v10 all hit F1 = 96.7-96.8% on content-equivalence. v10 is not better at speech recognition per se — it matches the state of the art. The value proposition is elsewhere.

2. **v10 dominates on speaker attribution: 92-93% role accuracy vs Whisper+pyannote's 75%.** Scored by content-aware alignment (F1 match of gold turn → nearest pred turn by word-set similarity, with role-aware tiebreaking for classroom call-and-response), v10 achieves **97-98% accuracy on teacher turns and 83-85% on student turns**. Whisper+pyannote hits 84% teacher / 61% student even with an oracle SPEAKER_00→teacher mapping. pyannote detects only **2 distinct speakers** across a classroom with ~15-20 students because it clusters by voice characteristics; v10 uses **visual features** (clothing, position, hair, facial features) to distinguish individuals. The qualitative win is even larger: v10 produces per-student labels like `S-Jenna`, `S-Boy-Afro`, `S-Girl-StripedShirt` that let researchers identify who said what without manual cluster labeling or re-watching the video.

3. **Visual descriptions are uncontested.** No baseline produces interleaved visual descriptions of classroom activity (gestures, whiteboard content, gaze, shared materials). v10 produces these in the same single API call as the speech transcription, at no additional cost and no additional pipeline complexity.

4. **Human review confirms the content-equivalence number is an accurate reflection of pipeline quality.** Reviewer-validated audits of US1 rows (first 5 minutes and flagged rows across the full 44 minutes) confirmed that the content-equivalence metric's flagged misses correspond to the reviewer's ground-truth verdicts. The remaining gold-unmatched words are overwhelmingly: (a) legitimate pipeline misses on short student backchannels, (b) additional TIMSS gold errors, or (c) short inaudible passages the reviewer couldn't verify either way. Raw (strict) WER overestimates pipeline error by 3-4× on this corpus.

5. **TIMSS gold transcripts contain errors and over-transcription of inaudible audio.** We documented 8 corrections in the first 5 minutes alone (3 removes of inaudible passages, 5 text edits for transcription mistakes like 'Sarah' miscoded as 'sir' and 'Y intercept and X intercept' where no 'X intercept' was actually said). The corrected gold and correction log are shipped as paper artifacts.

## Practical comparison

| aspect | Whisper alone | Whisper + pyannote | v10 (this work) |
|---|---|---|---|
| Speech transcription | ✓ | ✓ | ✓ |
| Speaker diarization | ✗ | anonymous clusters only | visual-feature per-student |
| Visual descriptions | ✗ | ✗ | ✓ interleaved |
| Classroom activity context | ✗ | ✗ | ✓ (whiteboard, gestures, materials) |
| API calls per video | 0 (local) | 0 (local) + HF model download | ~60 chunks/hr |
| Cost per hour of video | ~free (local CPU/GPU) | ~free (local) | ~$0.19 |
| Setup complexity | low | medium (HF auth + gated models) | low (one API key) |
| End-to-end wall time (44-min video, our machine) | ~2+ hours (large-v3 CPU) | +~10 min pyannote | ~42 min (API-bound) |

