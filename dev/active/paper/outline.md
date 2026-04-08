# Paper Outline — Multimodal Classroom Transcription Pipeline

**Working title:** *Multimodal Transcription of Classroom Video for Qualitative Educational Research: A Single-Model Approach Using Gemini*

**Target:** arXiv preprint first → IJRME or Behavior Research Methods
**Authors:** Jennifer Kleiman (UGA COMS) + collaborators TBD

---

## 1. Introduction (~1-1.5 pp)
- The problem: qualitative educational researchers who study classroom learning need transcripts that capture *what students are doing*, not just *what they are saying*. Gestures, board content, manipulation of materials, and collaboration patterns are central to analyzing learning — but existing tools can't capture them.
- Current options and their gaps:
  - Commercial transcription (Otter, Rev, Trint, AssemblyAI): audio-only diarization, no visual layer
  - Manual video coding (Transana, NVivo): accurate but prohibitively slow
  - Recent multimodal AI work (ClassMind, Laurent Picard): focused on teacher evaluation or general demo, not research transcription
- Contribution: a practical, GPU-free, single-model pipeline that produces research-grade multimodal transcripts — speech and visual description interleaved — suitable for qualitative analysis of classroom learning.
- Key claims:
  1. A single multimodal model (Gemini) can produce high-quality interleaved speech+visual transcripts in one pass, preserving gesture-speech coupling lost by split pipelines
  2. Targeted prompt engineering and chunking controls hallucination effectively, without requiring separate audio pipelines or validation agents
  3. The approach scales to real hour-long classroom recordings on commodity hardware, with no GPU

## 2. Related Work (~1.5 pp)
- Automated speech transcription and diarization (Whisper, Pyannote, AssemblyAI) — strong audio, no visual
- Manual video coding tools (Transana, ELAN, NVivo) — accurate, human-scale, slow
- Classroom observation frameworks (COPUS, Danielson, CLASS) — what researchers *want* to code
- Multimodal LLMs for video (Gemini, VideoMind, MMCTAgent) — general-purpose, not education-specific
- **ClassMind (Sun et al., 2025)** — closest related work; contrast architecture (split Whisper/Pyannote + Gemini captions, merged post-hoc, validation agents) vs. our unified multimodal pass; contrast purpose (teacher feedback/COPUS scoring vs. research transcription)
- Laurent Picard's Gemini multimodal transcription series — general demo, no education framing, no chunking for long videos

## 3. System Architecture (~2 pp)
- High-level diagram: video → chunking → speaker identification → chunk processing with overlap → assembly → dual output (research + Transana)
- **Chunking strategy:** 60-second chunks with 15-second overlap; rationale (Gemini context limits, speaker consistency across chunks, cost)
- **Speaker identification phase:** visual-feature labels (Girl-BlueTieDyeHoodie, Boy-GreyTShirtFrontLeft), not binary teacher/student; speaker registry shared across chunks for consistency
- **Per-chunk prompt construction:**
  - Base prompt (wholeclass / smallgroup / etc.)
  - Speaker registry
  - Anti-hallucination system block (educational research framing, visual specificity guidelines, format rules, example GOOD/BAD outputs)
  - Overlap handling for continuity
- **Model configuration:** Gemini 3 Flash Preview at HIGH resolution (280 tok/frame), FPS=2, temperature=0.2, thinking budget=4096
- **Assembly:** overlap resolution, dual output format (annotated research transcript with visual descriptions; clean Transana-compatible transcript for legacy tools)
- **Cost:** ~$0.19 per hour of video

## 4. Prompt Engineering and Hallucination Control (~1.5 pp)
**This is probably the most novel/useful section for other researchers.**
- The core insight: "describe everything" prompts produce hallucinated dialogue and fake gestures. "What are kids working on / what would a researcher need to see" produces grounded, relevant descriptions.
- Anti-hallucination techniques used:
  1. Prefer [inaudible] over filled-in speech
  2. "Write what you hear, not what makes sense"
  3. Example GOOD/BAD outputs in the prompt
  4. Hallucination-check patterns (textbook-perfect dialogue, synonym rephrasing, inferred vs. observed actions)
  5. Overlap windows let later chunks correct earlier speaker-label drift
- **A/B findings to report:**
  - Gemini 2.5 Pro: better audio, worse visual (fabrication); 3 Flash + HIGH is the right tradeoff
  - FPS: 2 vs 3 — 3 is overkill, 2 matches Google's own recommendation
  - Resolution: HIGH critical for Gemini 3 (280 tok/frame vs 70 default), not for 2.5
  - Temperature: 0.0 killed visual richness; 0.2 is sweet spot
  - Prompt framing: educational research framing reduced hallucination AND improved relevance
  - Context caching: works for Flash, often fails for 2.5 Pro

## 5. Evaluation (~2-3 pp) **[NEEDS BENCHMARKING WORK]**
- **Datasets:** ATLAS (free cases, shared with ClassMind for comparability) + Databrary subset (pending access approval, for realistic hour-long recordings)
- **Ground truth:** human-verified transcripts for N clips (TBD — probably 5-10 per dataset)
- **Metrics:**
  - Speech accuracy: WER against human-verified transcript
  - Diarization: DER and label consistency across long videos
  - Visual description quality: human-rated rubric (accuracy, relevance, specificity) scored by N raters
  - Hallucination rate: % of visual descriptions that are fabricated vs. verifiable against video
  - Coverage: amount of visual information captured per minute
- **Baselines:**
  - Raw Gemini (single long-context call, no chunking, no prompt engineering)
  - Whisper + Pyannote (audio-only baseline)
  - ClassMind (if reproducible from their paper/code)
  - Commercial (Otter or AssemblyAI) on audio track
- **Ablations:** contribution of each component (chunking, overlap, speaker registry, anti-hallucination prompt, resolution, FPS)

## 6. Case Studies (~1-1.5 pp)
- 1-2 qualitative examples from Ava 4th grade STEM corpus (or public-dataset equivalents if IRB issues)
- Show: teacher whiteboard content being captured, student gesture-speech coupling, choral response handling, inaudible segments appropriately marked
- Compare same segment processed by our pipeline vs. audio-only Whisper vs. raw Gemini to make the value visible

## 7. Discussion (~1 pp)
- Implications for qualitative classroom research: scale of analysis becomes tractable
- Limitations:
  - Ceiling dependent on Gemini quality (not our model)
  - Visual descriptions still miss subtle cognitive moves
  - Cost per video small but nonzero; privacy of sending video to commercial API
  - No longitudinal analysis yet
- Ethical considerations: student privacy, de-identification, IRB implications of cloud-based processing
- When to use this vs. manual coding vs. ClassMind-style teacher evaluation

## 8. Conclusion (~0.5 pp)

---

## Figures / Tables to prepare
- **Fig 1:** System architecture diagram
- **Fig 2:** Example interleaved transcript excerpt (annotated)
- **Fig 3:** Side-by-side comparison — our pipeline vs. audio-only vs. raw Gemini
- **Table 1:** Ablation study results
- **Table 2:** Comparison to baselines (WER, DER, visual quality rubric)
- **Table 3:** Cost comparison across approaches
- **Table 4:** Config comparison (models, resolutions, FPS findings)

---

## TODOs

### Writing (can do now)
- [ ] Draft Methods section (3 + 4) — pipeline is already built, just describe it
- [ ] Draft Related Work — research ClassMind details, Picard series, commercial landscape
- [ ] Draft Introduction (final pass after methods)
- [ ] Build system architecture figure

### Benchmarking (needs infrastructure)
- [ ] Get Databrary access (applied — pending)
- [ ] Identify 5-10 ATLAS free-case videos for benchmarking
- [ ] Create ground-truth transcripts for benchmark subset (human-verified)
- [ ] Design visual description quality rubric + find 2-3 raters
- [ ] Run baselines: Whisper+Pyannote, raw Gemini, Otter/AssemblyAI
- [ ] Run ablations (disable each component, compare output)
- [ ] Score all outputs against ground truth + rubric
- [ ] Produce comparison tables

### Practical
- [ ] Clean up pipeline code for public release
- [ ] Decide license (MIT? Apache 2.0?)
- [ ] Write README + install instructions
- [ ] Prepare de-identification protocol for any example excerpts
- [ ] Confirm IRB coverage for publication
