# Evaluation Protocol — ARGUS-Adapted for Classroom Video

This document specifies the evaluation methodology for the multimodal classroom video transcription pipeline. It adapts ARGUS (Rawal et al., ICCV 2025) for classroom contexts and adds layers for speech accuracy and speaker diarization.

## 1. Three-layer evaluation summary

| Layer | What it measures | Source dataset | Metric |
|---|---|---|---|
| 1. Speech accuracy | How well the pipeline transcribes spoken dialogue | NCTE Transcripts (4-5 grade math) + ATLAS subset | WER |
| 2. Visual description quality | How well the pipeline describes what is visually happening | ATLAS free cases + Databrary subset, with classroom-domain dense captions we annotate ourselves | ArgusCost-H, ArgusCost-O (adapted) |
| 3. Speaker diarization | Whether speaker labels stay consistent across long videos | ATLAS subset + Ava case study clips | Label consistency rate, turn-attribution accuracy |

## 2. Layer 1 — Speech accuracy

### Datasets
- **NCTE Transcripts** — 1,660 4th/5th grade math classroom transcripts. We use these as ground-truth references against any matched audio/video. (Pending access form approval.)
- **ATLAS free cases** — 29 freely available videos from National Board Certified Teachers; we transcribe the audio ourselves using a careful manual protocol on a 5-clip subset.

### Procedure
1. Pick 5–10 clips totaling ~30 minutes of classroom audio.
2. Produce ground-truth transcripts via two-pass human transcription with disagreement reconciliation.
3. Run all baselines and our pipeline on the same audio.
4. Compute Word Error Rate using the standard `jiwer` library, with normalized text (lowercase, punctuation stripped, contractions expanded).
5. Report WER overall and broken down by speaker type (teacher vs. student) — student speech is harder and is where most baselines fail.

### Baselines
- Whisper-large-v3 (audio-only)
- Whisper + Pyannote 3.1 (audio-only with diarization)
- AssemblyAI Universal-2 (commercial)
- Raw Gemini-3-Flash single long-context call
- Our pipeline (full)

## 3. Layer 2 — Visual description quality (ARGUS adapted)

This is the novel methodological contribution. We adapt ARGUS for classroom video.

### Annotation procedure

**Annotators:** 3 raters (target: 2 graduate students in education + 1 undergraduate research assistant). Pay rate: TBD per UGA standard. Each rater annotates all clips independently.

**Training:** 1-hour group session walking through:
- The three sentence types (Summary, Visual Description, Dynamic Action)
- Educational research focus (gestures, board content, materials, collaboration)
- The "describe what helps a researcher who wasn't in the room understand the activity" framing
- A worked example using a calibration clip not in the evaluation set

**Annotation format:** Each rater produces a dense caption per clip in the format:
```
[type: SUM] High-level summary sentence.
[type: VD] Visual scene detail.
[type: DA, t=00:08] Temporally ordered action.
[type: DA, t=00:12] Next temporally ordered action.
```

Target ~25 sentences per minute of video, following ARGUS density (~24.4 words/sec).

**Inter-annotator agreement (IAA) — required for the paper:**
- Compute pairwise sentence-level agreement on type labels (Cohen's kappa, target ≥ 0.7)
- Compute Krippendorff's alpha on entailment judgments for a subset (treating each rater's caption as both source and target for the others)
- Report IAA in the paper — this is something ARGUS itself does NOT do, so it strengthens our methodological contribution

**Reconciliation:** For cases where raters strongly disagree (kappa < 0.5 on a clip), a fourth senior rater adjudicates.

### Scoring procedure

We follow ARGUS's Eq. 1–3 directly:

1. **Sentence-level entailment** via LLM-as-judge:
   - Default judge: GPT-5 (or GPT-4o for direct ARGUS comparability)
   - Sensitivity check: also run with Claude Sonnet 4.6 and an open-source DeBERTa NLI model
   - Each model-generated sentence is classified as Entailed / Contradictory / Undetermined against the human reference
   - Sentence type (SUM / VD / DA) is also classified

2. **Dynamic programming alignment** with temporal ordering penalty:
   - Use ARGUS's recurrence (Eq. 3) directly
   - **λ value:** ARGUS uses λ=0.1 with no justification. We will run a sensitivity sweep over λ ∈ {0.05, 0.1, 0.2, 0.5} on a development subset and report results at the value with highest correlation to direct human ratings of "this caption is faithful to the video"
   - Justify our chosen λ in the paper as a methodological refinement

3. **Compute ArgusCost-H and ArgusCost-O** per clip and report:
   - Mean ± std across clips
   - Per-baseline results with 95% confidence intervals (bootstrap)
   - Decomposed by sentence type — does our pipeline help most on DA (dynamic actions, where hallucination is highest)?

### What we additionally report beyond ARGUS

- **Inter-annotator agreement** (ARGUS does not disclose this)
- **Annotation guidelines** as a supplement (ARGUS does not publish theirs)
- **λ sensitivity analysis** (ARGUS picks λ=0.1 without justification)
- **Domain-specific sentence type breakdown** — VD sub-categories: gesture, board content, materials, collaboration, classroom layout

### Baselines for Layer 2
- Raw Gemini-3-Flash single long-context call (the headline ablation)
- Our pipeline w/o anti-hallucination block
- Our pipeline w/o overlap (60s chunks, no overlap)
- Our pipeline w/o speaker registry (no Phase 2)
- Our pipeline at MEDIUM resolution (vs. HIGH)
- Our pipeline at FPS=1 and FPS=3 (vs. FPS=2)
- Our pipeline with Gemini 2.5 Pro (vs. Gemini 3 Flash)
- ClassMind-style split pipeline (Whisper + Pyannote + Gemini-Flash captions merged after) — we build this ourselves if their code isn't released
- Our pipeline (full)

## 4. Layer 3 — Speaker diarization

### Why we don't use traditional DER
Diarization Error Rate assumes speaker identities are arbitrary indices and scores correctness up to permutation. For our pipeline, the *meaning* of the labels matters — "Girl-BlueTieDyeHoodie" is a verifiable visual claim, not a permutation of "Speaker 1." We need a metric that rewards consistency and visual verifiability.

### Procedure
1. **Label consistency rate (LCR):** For each labeled speaker in our output, what fraction of their attributed turns visually correspond to the same physical person across the full video? Verified by a human rater spot-checking 10 turns per labeled speaker per video.
2. **Turn attribution accuracy (TAA):** On a labeled-ground-truth subset (we manually label every turn for ~5 clips), what fraction of turns are attributed to the correct speaker?
3. **Label informativeness:** Survey-based — show 3 raters our labels vs. ClassMind-style binary labels vs. Pyannote numeric labels and rate which is most useful for finding a specific student in the video. Likert 1-5.

### Baselines
- Pyannote 3.1 (audio-only, indexed labels)
- ClassMind binary teacher/student labels
- Our pipeline (visual feature labels)

## 5. Cost and runtime evaluation

For each baseline, report:
- API cost per video-hour (USD)
- Wall-clock processing time per video-hour
- Hardware requirements (GPU, CPU, none)
- Whether the system is freely runnable by an individual researcher

Table format (extending ClassMind's Table 3 style):

| System | Cost ($/hr) | Time (min/hr-video) | GPU? | WER | ArgusCost-H | ArgusCost-O | LCR |
|---|---|---|---|---|---|---|---|

## 6. Statistical reporting

- Report mean ± standard deviation across clips
- 95% bootstrap confidence intervals on all main metrics
- Paired comparisons (our pipeline vs. each baseline) with Wilcoxon signed-rank test
- Multiple-comparison correction (Holm-Bonferroni) where applicable

## 7. Open questions to resolve before benchmarking begins

1. **NCTE access timing** — apply now, proceed with ATLAS subset if delayed
2. **Annotator recruitment** — confirm UGA pay rate, post for grad students
3. **IRB scope** — confirm we can publish de-identified excerpts from the Ava corpus for case studies
4. **ClassMind code** — wait ~10 days after the email; build self-implementation if no response
5. **λ sensitivity sweep** — needs a development set of ~3 clips with paired human "faithfulness" ratings

## 8. Concrete next steps

- [ ] Apply for NCTE Transcripts access
- [ ] Confirm ATLAS free-case download mechanism
- [ ] Send email to ClassMind authors (draft ready)
- [ ] Recruit 3 annotators for Layer 2
- [ ] Draft annotation guidelines (educational research framing of ARGUS sentence types)
- [ ] Build the ClassMind-style split-pipeline baseline
- [ ] Build the ablation runner script (toggle each pipeline component independently)
- [ ] Set up the ARGUS-style scoring infrastructure (LLM judge + DP alignment)
