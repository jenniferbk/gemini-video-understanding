# NOW — Paper Project Resume Point

**Last session:** 2026-04-08 (evening + overnight)
**Project:** Multimodal classroom video transcription pipeline — methods paper
**Target venue:** arXiv preprint first → IJRME or Behavior Research Methods

---

## Headline state

We have a working benchmarking pipeline, a reviewer-validated content-equivalence metric, a corrected TIMSS US1 gold transcript, and apples-to-apples comparison numbers against Whisper large-v3 and Whisper+pyannote. Content-equivalence WER on full 44-minute US1 is **6.1-6.3% (F1 = 96.7-96.8%)** across two v10 replicates — statistically tied with Whisper on the speech axis. v10's real differentiator is that it produces **per-student visual speaker labels** AND **interleaved visual descriptions** of classroom activity in a single API call at $0.19/hour, which no baseline can do.

**The full comparison table lives at:** `dev/active/paper/benchmark_runs/comparison_summary.md`

---

## Key artifacts

| file | what it is |
|---|---|
| `dev/active/paper/content_equivalence.py` | Shared equivalence rules (VARIANTS, IGNORE_WORDS, expand, score function). Single source of truth used by both the scorer and the review tool. |
| `dev/active/paper/benchmark_timss.py` | The benchmark scorer. Produces content_equivalence dict + per-role WER + strict WER + dropped-turn analysis. |
| `dev/active/paper/build_review_tool.py` | HTML review tool generator. Imports from content_equivalence.py. |
| `dev/active/paper/apply_audit.py` | Takes original gold + audit JSON + corrections YAML → corrected gold + change log. |
| `dev/active/paper/gold_corrections_US1.yaml` | 8 manual corrections to TIMSS US1 gold (3 remove, 5 edit). |
| `dev/active/paper/build_comparison_table.py` | Scans all benchmark JSONs, emits comparison_summary.md. |
| `dev/active/paper/run_pyannote.py` | Whisper+pyannote alignment script. Uses soundfile directly to bypass torchcodec. |
| `dev/active/paper/projection_config.yaml` | Default speaker-label projection (for v10-style outputs). |
| `dev/active/paper/projection_config_whisper_pyannote.yaml` | Projection config for Whisper+pyannote (SPEAKER_00→T, SPEAKER_XX→S). |
| `dev/active/paper/benchmark_runs/US1_gold_corrected.txt` | Corrected TIMSS US1 gold after applying audit + corrections. |
| `dev/active/paper/benchmark_runs/US1_audit_log.md` | Human-readable change log for gold corrections. |
| `dev/active/paper/benchmark_runs/review_US1_full.html` | Full 44-min review HTML (all 936 rows, 16 pred orphans). |
| `dev/active/paper/benchmark_runs/comparison_summary.md` | **The master comparison table for the paper.** |

---

## Systems benchmarked (all vs. corrected gold)

| system | recall | precision | F1 | content WER | speaker acc | strict WER |
|---|---|---|---|---|---|---|
| Whisper large-v3 (alone) | 96.1% | 97.6% | 96.8% | 6.3% | 52.9% | 30.7% |
| Whisper + pyannote 3.1 | 96.1% | 97.6% | 96.8% | 6.3% | 50.0% | 30.7% |
| v10 run 1 | 95.6% | 98.1% | 96.8% | 6.1% | 50.1% | 22.1% |
| v10 run 2 (replicate) | 95.5% | 97.9% | 96.7% | 6.3% | 48.4% | 22.9% |

Replicate variance: recall range 0.04 pp, F1 range 0.13 pp, content WER range 0.22 pp. Speaker accuracy range 1.71 pp (noisier axis).

---

## TIMSS gold corrections found (US1, first 5 min)

1. 00:00:14 SN "Dear..." — REMOVED (inaudible)
2. 00:00:15 SN "Thank you." — REMOVED (inaudible)
3. 00:00:38 S "Ashley, can I have..." — EDITED to "Can I have a piece of paper?" (Ashley inaudible)
4. 00:01:01 SN "Oh, (inaudible)." — REMOVED (inaudible)
5. 00:02:07 SN "Y intercept and X intercept." — EDITED to "Y intercept." (no "X intercept" said)
6. 00:03:48 SN "Mr. Ormsby, you know, when you put..." — EDITED (prefix inaudible)
7. 00:03:53 T "zeros are our favorite number" — EDITED to "zero's our favorite number"
8. 00:04:52 S "You want to pick three as like-" — EDITED to "You want to pick threes? ... like-"

Plus 2 known errors in later sections from the review:
- "Thank you Sarah" was TIMSS gold; pred said "sir" — BOTH are wrong; audio is "Sarah" (gold was right)
- TIMSS "(inaudible)" at 00:41 where pred said "man" — marginal

---

## Equivalence rules (single source: `content_equivalence.py`)

Bidirectional word-set equivalence classes cover:
- **Compounds**: `alright ↔ all right`, `anyway ↔ any way`, `two-thirds ↔ two thirds`
- **Contractions**: `you're`/`don't`/`that's`/`it's`/`he's`/`she's`/`I'm`/`I've`/`I'll`/`we're`/`they're`/`let's`/`that'll`/`it'll`, etc.
- **Colloquial**: `gonna`, `wanna`, `gotta`, `kinda`, `sorta`, `'cause`, `y'all`
- **Titles**: `Mr. ↔ Mister`, `Mrs.`, `Ms.`
- **Numbers**: `zero ↔ 0` through `ten ↔ 10`
- **Modal verbs**: `can ↔ could`, `would ↔ could ↔ will`
- **Demonstratives**: `this ↔ these ↔ that ↔ those`
- **Backchannels**: `mhm ↔ yeah ↔ yes ↔ yep ↔ uh-huh ↔ okay ↔ ok`
- **Short all-alpha tokens (≤3 chars)**: split into individual letters (`XY ↔ X Y`)
- **Hyphenated compounds**: split into both joined and separated forms (`Y-intercept ↔ Y intercept`)
- **Meta markers**: `inaudible`/`unclear`/`unintelligible`/`crosstalk`/`overlap`/`pause`/`silence`/`laughter`/`noise` are all treated as invisible (never highlighted, never contribute to word set)

Matching is time-windowed at ±12 seconds.

---

## Open items for next session

### High priority (paper-critical)
1. **More replicate runs of v10** — we have 2; target 5 for proper variance bars.
2. **More TIMSS lessons benchmarked.** Need to track down YouTube IDs for US2, JP1, CZ1, AU1 (at minimum). `www.timssvideo.com/us2` is the page but the YouTube link isn't in the raw HTML for simple scraping.
3. **Finish the full-lesson review** in `review_US1_full.html` — whenever there's time, especially to find more TIMSS gold errors.
4. **Start the paper draft.** The methods/results sections are well-supported by the numbers in `comparison_summary.md`.

### Medium priority
5. **Per-role content-equivalence** — the current content WER is aggregated across teacher and student. A per-role version (T-only gold words vs T-only pred words, same for student) would show whether student recall is truly ~55% or whether it's a metric artifact. Earlier per-role-WER showed teacher=12% vs student=56%, but that's legacy jiwer not content-equivalence.
6. **Per-student recall analysis** — which named student (Jenna, Boy-Afro, etc.) gets which recall? Matters for educational research use.
7. **Consolidate `timss_speakers_stub.json`** — the 5-label generic manifest; could be improved with the per-lesson review data.

### Low priority / nice-to-have
8. **Whisper + pyannote isn't really a multi-step comparison** unless we also add a VLM frame-sampler for the visual descriptions. The fair comparison for visual descriptions is "nothing exists" — make this clear in the paper.
9. **ClassMind email** — still not sent. Draft is in `email_to_classmind_authors.md`.
10. **Run on an Ava/Ben/Daisy classroom video** as a case study — we have transcription validated, now use it for the actual qualitative analysis section.

---

## Known gotchas / limits

- **torchcodec / ffmpeg version mismatch** on this machine means torchaudio and pyannote's native file loader don't work. Workaround in `run_pyannote.py`: load WAV with `soundfile`, pass tensor directly to pyannote pipeline.
- **pyannote speaker-diarization-3.1 pulls in a third gated model** (`speaker-diarization-community-1`). Jennifer accepted terms for all three.
- **HuggingFace token** in use for pyannote runs — passed via CLI flag only, not stored in any file. Should still be rotated since it appeared in chat logs.
- **Gemini API key** — also passed via env var at invocation time; not stored in any committed file. Should still be rotated since it appeared in chat logs.
- **Run-to-run variance**: content metrics stable (<1 pp), speaker accuracy more variable (~2 pp). Any published number should be reported as mean ± range across replicates.
- **Bandwidth constraint**: as of 2026-04-08, operating on slow/limited network. No large downloads, no additional pipeline runs that upload chunks to Gemini.

---

## Validated pipeline config (don't re-derive)

- Model: **gemini-3-flash-preview**
- Resolution: **HIGH** (280 tok/frame)
- FPS: **2**
- Chunks: **60s + 15s overlap**
- Temperature: **0.2** (hardcoded in v10)
- Thinking budget: **4096**
- Cost: **~$0.19/hour** of video
- Wall time: **~1:1 with video duration**

---

## Resume command

Say "continue paper work" or point at this file. Next likely moves in order:
1. If bandwidth is OK: download + run US2 pipeline, score, add to comparison table
2. Pick up replicates 3-5 of US1 for variance bars
3. Finish full-lesson review, find more gold errors, rebuild corrected gold
4. Start drafting the Methods section using `comparison_summary.md` as the anchor
