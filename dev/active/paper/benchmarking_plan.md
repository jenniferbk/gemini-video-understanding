# Benchmarking Plan

## Strategy: three-layer evaluation

### Layer 1 — Speech accuracy (WER)
- **Dataset:** NCTE Transcripts (Demszky & Hill, 2023). 1,660 4th/5th grade math classroom transcripts, 45-60 min each, 317 teachers. Same grade band as our Ava corpus. Access via form at https://github.com/ddemszky/classroom-transcript-analysis
- **Caveat:** NCTE provides transcripts, not video. We need either (a) audio/video files paired with the transcripts (some classroom recordings may be in ICPSR), or (b) treat NCTE as ground-truth-style transcripts and evaluate against a public classroom video dataset that has its own transcripts.
- **Metric:** Word Error Rate against human transcripts.
- **Baselines:** Whisper-large-v3 audio-only, AssemblyAI commercial, raw Gemini single-call.

### Layer 2 — Visual description quality (the novel contribution)
- **Method:** Adapt ARGUS (Rawal et al., ICCV 2025) for classroom video. ARGUS evaluates open-ended dense captions using `ArgusCost-H` (hallucination) and `ArgusCost-O` (omission), with human-annotated 477-word dense captions per video.
- **Procedure:**
  1. Pick 5–10 short clips (1–2 min each) from ATLAS free cases + Databrary (when access lands).
  2. Have 2–3 human annotators produce dense visual captions following ARGUS protocol, adapted with educational research framing (gestures, board content, materials, collaboration).
  3. Score our pipeline output, raw Gemini long-context, and ClassMind-style split-pipeline output against the human captions using ArgusCost-H and ArgusCost-O.
  4. Report inter-annotator agreement.
- **Why this matters:** This adapts an existing video-LLM evaluation framework to the classroom domain — that itself is a methodological contribution beyond just our pipeline.

### Layer 3 — Speaker diarization
- **Method:** Borrow AVUT's "visually specified speaker" framing (EMNLP 2025) — transcribe speech of a speaker identified by visual features. This conceptually matches our visual-feature labeling approach better than traditional DER.
- **Metric:** Label consistency across long videos (does "Boy-RedHoodie" remain the same person across 60+ minutes?), plus turn-attribution accuracy on a labeled subset.
- **Baselines:** Pyannote 3.1 (audio-only DER), ClassMind's binary teacher/student.

## Datasets to commit to

| Dataset | Purpose | Status |
|---|---|---|
| NCTE Transcripts | Speech ground truth (4-5 grade math) | Need to apply |
| ATLAS free cases (29 available) | ClassMind comparability + visual rating | Free, ready to download |
| Databrary subset | Real, unconstrained classroom conditions | Application submitted |
| Ava corpus | Case study examples (IRB-permitted excerpts) | In hand |

## Baselines to run

| Baseline | What it tests |
|---|---|
| Raw Gemini single long-context call (no chunking, no prompting) | Ablates chunking + prompt engineering |
| Whisper-large-v3 audio-only | Baseline for audio quality |
| Whisper + Pyannote 3.1 | Audio-only with diarization |
| AssemblyAI commercial | Commercial audio benchmark |
| ClassMind reproduction (if code available) | Direct architectural comparison |
| Our pipeline w/o anti-hallucination block | Ablation: how much does the prompt matter? |
| Our pipeline w/o overlap | Ablation: how much does overlap matter? |
| Our pipeline at MEDIUM/LOW resolution | Ablation: resolution sensitivity |
| Our pipeline at FPS=1, FPS=3 | Ablation: frame rate sensitivity |
| Our pipeline with Gemini 2.5 Pro | Ablation: model choice |

## Tables we'll produce

- **Table 1.** Headline comparison: our pipeline vs. all baselines on WER, ArgusCost-H, ArgusCost-O, diarization consistency
- **Table 2.** Ablation study: each pipeline component disabled, scored on the same metrics
- **Table 3.** Cost comparison ($/hour of video) and processing-time-to-video ratio
- **Table 4.** Per-dataset breakdown (ATLAS vs. Databrary vs. Ava case studies)

## Open questions / risks

1. **NCTE access timing** — application form, no published timeline. May need to start without it.
2. **Annotator recruitment** — Layer 2 needs 2–3 raters. Grad students? Undergrad RAs? Pay rate?
3. **Inter-annotator agreement** for visual descriptions is an open methodological question — ARGUS uses dense captions but doesn't deeply discuss reliability for the educational domain.
4. **ClassMind reproduction** — they call it "open-source" but no repo URL was found in our research. If unavailable, we describe their approach textually and compare against the metrics they reported.
5. **IRB constraints on Ava corpus** — confirm we can publish de-identified excerpts for case studies.

## Next concrete actions

- [ ] Apply for NCTE Transcripts access
- [ ] Download ATLAS 29 free cases (when subscription confirmed not needed)
- [ ] Read ARGUS paper in detail; extract their annotation protocol and adapt
- [ ] Wait for Databrary access decision
- [ ] Draft annotator instructions (visual description guidelines for classroom video)
- [ ] Decide on rater recruitment plan
- [ ] Find/check if ClassMind code is publicly released
