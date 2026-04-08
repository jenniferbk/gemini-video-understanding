# Related Work Research: Multimodal Classroom Video Transcription Pipeline

Compiled 2026-04-06. Research conducted via arXiv, Google Scholar/web, Semantic Scholar surface results, and direct fetches of HTML papers. Forward-citation searching for ClassMind (2509.18020) was limited because the paper is recent (Sept 2025) and Google Scholar's "cited by" graph is not directly queryable from this environment — the formal forward-citation list is incomplete and flagged below.

---

## 1. Direct Competitors (tools/papers doing similar things)

### ClassMind (Sun et al., Sept 2025) — arXiv 2509.18020
**Closest competitor.** Stanford SCALE Initiative. AI-driven classroom observation system with the AVA-Align agent framework. Pipeline:
- **ASR/diarization:** Whisper-Large-v3 + Pyannote 3.1 (audio-only diarization, ~30s per 30-min video on A6000)
- **Visual captioning:** Gemini-2.5-Flash on 2-minute video segments (transcript and visual caption produced separately and *merged after*, not interleaved at the model level)
- **Activity coding:** video-language model aligned to COPUS codes
- **Agent layer:** AVA-Align finds rubric "hotspots," aligns to Danielson Framework, validates against video evidence
- **Aimed at teacher feedback / instructional coaching**, not student learning research
- Three-phase HCI study: co-design, build, field test with teachers
- **Differentiation from our work:** (a) audio diarization via Pyannote, not visual-feature diarization; (b) separate audio + visual passes merged after, not interleaved single-pass multimodal transcription; (c) goal is teacher evaluation/coaching with rubrics, not researcher-facing interleaved transcripts for student-learning analysis; (d) much heavier agent stack on top of the transcription primitive.

### Bueno, Hou, Bühler et al. (Dec 2025) — arXiv 2512.00087
"Exploring Automated Recognition of Instructional Activity and Discourse from Multimodal Classroom Data." TUM + Tübingen + Virginia + SUNY Albany + Gordon. **164 hours of densely annotated classroom video, 68 transcripts.** Parallel pipelines:
- Video: Qwen2.5-VL (zero-shot), X-CLIP, V-JEPA 2 — 24 activity labels, macro-F1 0.577
- Text: Llama3 Instruct + DeBERTaV3 — 19 discourse labels, macro-F1 0.460
- 2-second video clips, multi-label classification, dynamic per-label thresholding
- **Relevance:** Same "multimodal classroom" niche but framed as classification benchmark, not as a transcription tool. Useful for citing the dataset and as evidence that fine-tuning beats zero-shot for activity recognition. **Worth reading in full.**

### Shen, He, Liu et al. (2026) — arXiv 2602.18466 — SciIBI
"Can Multimodal LLMs 'See' Science Instruction? Benchmarking Pedagogical Reasoning in K–12 Classroom Videos." Drexel/WSU/BNU/UNC/CityU HK. **First video benchmark for K-12 science classroom discourse coding.** 113 NGSS-aligned clips, Whisper transcripts + Core Instructional Practice labels. Benchmarks 8 models (GPT-4o, Claude Sonnet 4.5, Gemini-2.5-Pro, Qwen3-VL-235B, Mistral, GPT-OSS, Llama-3.3, InternVL3). Key findings:
- Zero-shot accuracy 39–54%, well below math benchmarks (~79%)
- Visual input helps inconsistently; gains concentrated in "artifact-mediated" clips
- Models rely on linguistic surface features over genuine pedagogical reasoning
- **Relevance:** Validates the premise that current VLMs struggle to reason about classroom video and that visuals add real signal — supports our motivation for visual-aware transcription. **Worth reading in full.**

### TeachFX (commercial)
Audio-only. Differentiates teacher vs. student speech, dashboards on talk-time, wait time, questioning patterns, equity of participation. ~$10K–$30K/year per school. Demszky et al. (2023) RCT showed 20% increase in "focusing questions" among teachers receiving TeachFX feedback. **No visual analysis. No researcher-facing transcripts.** Aimed at teacher coaching.

### Edthena (commercial) — Observation Copilot, VC3, AI Coach
Video coaching platform. Recently added "Observation Copilot" that turns low-inference teacher notes into Danielson-aligned feedback. AI Coach is non-directive, question-based reflection. **No automated multimodal transcription** — humans still watch and write notes. Edthena positions itself as reflection-first vs. TeachFX's data-first.

### Laurent Picard's Gemini multimodal video transcription series (Google Cloud Medium / Codelabs, 2024–2025)
8-part demo series. General-purpose, single-prompt multimodal video transcription with Gemini. **No education framing, no diarization beyond what Gemini does inline, no chunking/overlap engineering, no anti-hallucination prompt work, no positioning for research use.** Useful as the prior-art "proof of concept" we extend.

---

## 2. Related Multimodal Video LLM Research

### Foundation models
- **Gemini 1.5 Technical Report** (Reid et al., 2024) — arXiv 2403.05530. Long-context multimodal model, hours of video in context. Foundational citation for why Gemini works for our use case.
- **Gemini original report** (Anil et al., 2023) — arXiv 2312.11805.
- **Gemini 2.5 Tech Report** (DeepMind, 2025). Native audio understanding, long context.
- **Whisper** (Radford et al., 2022) — "Robust Speech Recognition via Large-Scale Weak Supervision." The standard for ASR baselines.
- **Pyannote.audio** (Bredin et al., 2020) — neural end-to-end speaker diarization. The standard audio-diarization baseline.

### Video LLM hallucination — directly relevant to our anti-hallucination prompt engineering
- **VidHalluc** (Li et al., 2024) — arXiv 2412.03735, CVPR 2025. 5,002-video benchmark for action / temporal-sequence / scene-transition hallucinations in video MLLMs.
- **VideoHallu** (OpenReview 2025). 3,000 synthetic videos with counterintuitive QA pairs.
- **ARGUS** (ICCV 2025). Hallucination + omission evaluation in video LLMs.
- **MASH-VLM** (CVPR 2025). Mitigating action-scene hallucination via disentangled spatial-temporal representations.
- **DINO-HEAL** — training-free hallucination reduction via spatial saliency reweighting (~3% improvement).
- **Video-MME** (Fu et al., 2024) — arXiv 2405.21075. First comprehensive video-LLM evaluation benchmark.
- **VIDHALLUC PMC review** (PMC12408113). Useful summary of the hallucination-evaluation landscape.

### Long-video understanding & chunking strategies
- **OneClip-RAG** (arXiv 2512.08410). Instruction-aware clip chunking + retrieval for long-video MLLMs.
- **FlexMem / Visual Memory Mechanism** (arXiv 2603.29252). Continual-watch memory recall for arbitrary-length video.
- **Awesome-LLMs-for-Video-Understanding** (Tang et al., IEEE TCSVT survey, github yunlong10/Awesome-LLMs-for-Video-Understanding). Living survey, useful one-stop bibliography.
- **Video Understanding with LLMs: A Survey** — arXiv 2312.17432.

### Multimodal reasoning across audio + vision + video
- Chen et al. (2025), Han et al. (2024), Du et al. (2025), Comanici et al. (2025) — all cited by ClassMind as multimodal-LLM background. None offer a transcription-tool framing.

---

## 3. Classroom Observation Research Tradition

### Frameworks our work should explicitly position against / cite
- **COPUS** (Smith et al., 2013) — Classroom Observation Protocol for Undergraduate STEM. 25 codes in two categories (student doing / instructor doing). Designed for low-training (1.5 hours) human observers. *CBE—Life Sciences Education*.
- **Danielson Framework for Teaching** (Danielson Group). 4 domains, 22 sub-domains. Most widely deployed K–12 teacher-evaluation rubric in the US.
- **CLASS** (Pianta, La Paro, Hamre — Classroom Assessment Scoring System). Used heavily in early-childhood research. Cited via Martínez et al. (2016) in ClassMind's bibliography.
- **Bloom's Taxonomy** (Bloom 1956; Anderson & Krathwohl 2001). Used by ClassMind for question classification.
- **Hattie & Timperley (2007)** "The power of feedback," *Review of Educational Research* 77(1):81–112. Foundational citation for any feedback-loop framing.

### Critiques of human classroom observation that motivate AI tooling
- Hill & Grossman (2013); Ho & Kane (2013); Cantrell & Kane (2013); Kraft et al. (2018); Knight & Skrtic (2020) — all cited by ClassMind. Establish that human observation is expensive, low-reliability, and doesn't scale.

### NLP / ML for classroom transcripts
- **Demszky & Hill (2023)** "The NCTE Transcripts: A Dataset of Elementary Math Classroom Transcripts." arXiv 2211.11772, BEA 2023. **1,660 transcripts of 4th/5th-grade math, 317 teachers, turn-level discourse-move annotations.** Largest publicly available classroom-transcript dataset; the obvious citation for "NLP on classroom dialogue."
- **Demszky et al. (2024)** RCT of NLP feedback on instructor discourse (TeachFX-related).
- **Jensen et al. (2020)** ML for scoring instructional quality from transcripts.
- **Alic et al. (2022)** NLP for teacher question extraction.
- **Wang et al. (2024)** AI for automated classroom interaction analysis.
- **Yun et al. (2025)** Teachers' use of automated indicators (wait time, uptake, talk balance).

### Speaker diarization specifically for classrooms
- **"Optimizing Speaker Diarization for the Classroom"** — *Journal of Educational Data Mining*. ECAPA-TDNN + Whisper VAD hybrid; ~17% DER for teacher vs. student.
- **"Speaker Diarization in the Classroom: How Much Does Each Student Speak in Group Discussions?"** — EDM 2024 short paper.
- **"Multi-Stage Speaker Diarization for Noisy Classrooms"** — arXiv 2505.10879.
- **Dubey et al. (2022)** "Speaker Diarization and Identification from Single-Channel Classroom Audio Using Virtual Microphones," IEEE — arXiv 2207.00660.
- **"Integrating Audio, Visual, and Semantic Information for Enhanced Multimodal Speaker Diarization"** — arXiv 2408.12102. Audio-visual pipeline using YuNet face detection + AdaFace embeddings. **Closest prior art for the "visual speaker identification" angle**, but not classroom-specific and not LLM-based. **Worth reading.**
- **Eurasip Journal** lightweight real-time audio→audio-visual diarization (2024).

---

## 4. Manual Video Coding Tools (positioning targets)

- **Transana** (transana.com). Long-running qualitative video analysis tool, big in education and CA research. As of 2023 added automated transcription via cutting-edge ASR, and now integrates ChatGPT or local Ollama for exploration. Supports up to 5 parallel transcripts of the same video. **Our output should be Transana-importable (RTF) — already in MVP success criteria.**
- **ELAN** (Max Planck Institute for Psycholinguistics). Multi-tier annotation, extensive segmentation, automatic VAD/turn segmentation. Standard in linguistics, gesture, and sign-language research.
- **NVivo** (Lumivero). General qualitative analysis; weak on video relative to Transana/ELAN but dominant in ed research broadly.
- **CLAN** (CHILDES project). Conversation-analytic transcription; classic in child language research.
- **MAXQDA** — comparable to NVivo, used in mixed-methods ed research.

**Positioning angle:** these tools all assume a human writes the transcript and codes the video. None produce interleaved speech+visual descriptions automatically. Our pipeline produces the *first draft* that gets imported into one of these tools.

---

## 5. Audio-only Transcription Landscape

- **Whisper** (OpenAI) — Radford et al. 2022. The de-facto open-source ASR baseline.
- **WhisperX** (Bain et al. 2023). Whisper + forced alignment + Pyannote diarization. Probably the closest open-source pipeline to "Whisper + Pyannote" as used by ClassMind.
- **Pyannote.audio 3.1** (Bredin et al.) — speaker diarization SOTA for open-source.
- **Sortformer** (NVIDIA) — newer end-to-end diarization architecture; comparison vs. Pyannote in vast.ai blog.
- **AssemblyAI, Rev.ai, Otter.ai, Deepgram** — commercial ASR-with-diarization services. None do visual description.
- **Descript** — commercial transcription with editing UI; no visual.

---

## 6. Key Papers — Forward / Backward Citation Notes

### ClassMind (2509.18020) — backward references most relevant to us
Pulled from the HTML version of the paper. The full bibliography has ~133 entries; the ones most relevant:

| Citation | Year | Topic | Why relevant |
|---|---|---|---|
| Radford et al. (Whisper) | 2022 | ASR | Core ASR baseline; we should cite even though we don't use it |
| Bredin et al. (Pyannote.audio) | 2020 | Diarization | Audio-diarization SOTA we're departing from |
| Bain et al. (WhisperX) | 2023 | ASR+diarization | Closest open-source competitor stack |
| Smith et al. (COPUS) | 2013 | Observation framework | The undergrad-STEM observation rubric |
| Danielson Group | n/a | Observation framework | K-12 evaluation rubric |
| Martínez et al. | 2016 | CLASS + Danielson | Reliability/validity background |
| Hill & Grossman | 2013 | Observation critique | Why human observation doesn't scale |
| Kraft, Blazar, Hogan | 2018 | Coaching meta-analysis | Cost/scaling problem |
| Demszky & Liu | 2023 | NLP feedback | Classroom NLP precedent |
| Demszky et al. | 2024 | RCT NLP feedback | TeachFX-style RCT |
| Alic et al. | 2022 | Question extraction | Discourse-NLP precedent |
| Jensen et al. | 2020 | ML quality scoring | Transcript-based quality scoring |
| Hattie & Timperley | 2007 | Feedback theory | Foundational feedback framework |
| Bloom 1956 / Anderson & Krathwohl 2001 | | Bloom's taxonomy | Question classification |
| Comanici et al. | 2025 | GPT-4o multimodal | Multimodal-LLM background |
| Han et al. | 2024 | Multimodal LLMs | Multimodal-LLM background |
| Du et al. | 2025 | Multimodal LLMs | Multimodal-LLM background |
| Cai et al. | 2024 | Temporal video limits | Video-LLM limitations |
| Li et al. | 2025 | Timestamp precision | Temporal grounding problem |
| Shu et al. | 2025 | Long-sequence hallucinations | Hallucination in long video |
| Wang et al. | 2025 | Temporal grounding | Video-LLM limits |
| Pereira & Hone | 2021 | Multimodal teacher feedback | Pre-LLM multimodal observation |
| Arakawa & Yakura | 2019 | Real-time behavioral anomaly | Pre-LLM real-time coaching |
| Chandler et al. | 2024 | Peer-behavior learning analytics | LA-tradition precedent |

### ClassMind forward citations
**Direct forward-citation lookup is incomplete.** Google Scholar's "cited by" graph isn't queryable here, and Semantic Scholar surface results didn't return distinct citing papers as of April 2026. The paper was posted Sept 22, 2025, so the citing literature is still thin. Two adjacent late-2025/early-2026 papers in the same niche that are NOT explicit citations but are concurrent work:
- Bueno et al. (2512.00087) — Dec 2025
- Shen et al. SciIBI (2602.18466) — 2026

**Recommendation:** Manually run a Google Scholar "Cited by" check on ClassMind before submission, and rerun semanticscholar.org/paper/... lookup closer to the deadline. Flag this as a known gap.

---

## 7. Gaps in the Literature — what our paper fills

1. **Interleaved speech + visual description as a single-pass output.** Every prior pipeline (ClassMind, WhisperX, TeachFX, Bueno et al.) produces audio transcript and visual labels as **separate streams merged after**. No one ships a transcript where the visual description sits inline with the dialogue at the moment it occurred.
2. **Visual-feature speaker diarization** ("Girl-BlueTieDyeHoodie") in lieu of audio-only diarization. The closest prior work is the audio-visual diarization paper (arXiv 2408.12102) which uses face embeddings, not natural-language visual descriptors, and is not classroom-targeted. ClassMind explicitly uses Pyannote (audio-only).
3. **Researcher-facing tooling rather than teacher-evaluation tooling.** ClassMind, TeachFX, Edthena, Observation Copilot all target the teacher-coaching market. The classroom-NLP literature (Demszky lineage) targets discourse measurement. **Nothing currently targets the qualitative-research workflow** (code-and-retrieve, Transana/ELAN import, student-learning focus rather than teacher rating).
4. **Anti-hallucination prompt engineering for educational video specifically.** The hallucination literature (VidHalluc, ARGUS, MASH-VLM, DINO-HEAL) exists at the benchmark level but doesn't translate into deployable prompt patterns for domain-specific transcription. Our `[inaudible]` policy and "what are kids working on" framing are concrete contributions.
5. **Chunking-with-overlap as anti-hallucination strategy.** Long-video chunking (OneClip-RAG, FlexMem, ClassMind's 2-min chunks) is treated as a context-window engineering problem, not as a hallucination-mitigation strategy. Our 60s + 15s overlap design is a different framing.
6. **Desktop application for non-technical research users.** All academic prior work is research code; commercial tools (TeachFX/Edthena) are SaaS aimed at districts. The "8 colleagues at UGA, Mac, drag-and-drop" user model is essentially absent from the literature.
7. **Gesture and nonverbal content as first-class transcript citizens.** The gesture-transcription tradition (McNeill 1992; Mondada; Goodwin) is entirely manual. No automated tool produces gesture-aware transcripts. We don't claim full gesture coding, but the visual descriptions implicitly capture pointing, tracing, counting — something the manual tradition would recognize as valuable.

---

## Papers Recommended for Full-Text Reading

1. **ClassMind** (2509.18020) — full PDF, especially the AVA-Align section and the limitations discussion. This is your single most important comparison point.
2. **Bueno et al.** (2512.00087) — for the dataset (164 hrs annotated) and the Qwen2.5-VL zero-shot vs. fine-tuned comparison.
3. **SciIBI / Shen et al.** (2602.18466) — for the "current MLLMs can't really reason about classroom video" framing, which justifies why your prompt engineering matters.
4. **Demszky & Hill NCTE Transcripts** (2211.11772) — for the dataset and the discourse-move annotation scheme; this is the canonical citation for "NLP on classroom transcripts."
5. **arXiv 2408.12102** "Integrating Audio, Visual, and Semantic Information for Enhanced Multimodal Speaker Diarization" — closest prior art for the audio-visual diarization angle.
6. **VidHalluc** (2412.03735) and **ARGUS** (ICCV 2025) — to ground the anti-hallucination contribution.
7. **WhisperX** (Bain et al. 2023) — to position cleanly against the obvious open-source baseline pipeline.

---

## Known Limitations of This Search

- No direct access to Google Scholar's "cited by" graph; forward-citation list for ClassMind is best-effort.
- Some arXiv IDs (e.g., 2602.18466, 2603.29252, 2512.08410) have unusual prefixes — these appear to be valid newer-format IDs but should be double-checked when retrieving full PDFs.
- Did not access Semantic Scholar API directly; only surface web results.
- Did not search ACM DL, IEEE Xplore, or AERA proceedings (which would surface education-research-specific papers not indexed on arXiv).
- Did not search for non-English literature.
- The Picard Medium series was not enumerated part-by-part; treated as a single body of work.

---

## Sources

- [ClassMind on arXiv (abs)](https://arxiv.org/abs/2509.18020)
- [ClassMind HTML](https://arxiv.org/html/2509.18020v1)
- [ClassMind on Stanford SCALE](https://scale.stanford.edu/ai/repository/classmind-scaling-classroom-observation-and-instructional-feedback-multimodal-ai)
- [Bueno et al. — Multimodal Classroom Data](https://arxiv.org/html/2512.00087v1)
- [SciIBI — Can Multimodal LLMs See Science Instruction?](https://arxiv.org/html/2602.18466)
- [NCTE Transcripts — Demszky & Hill](https://arxiv.org/abs/2211.11772)
- [NCTE Transcripts on ACL Anthology](https://aclanthology.org/2023.bea-1.44/)
- [Gemini 1.5 Tech Report](https://arxiv.org/abs/2403.05530)
- [Gemini original paper](https://arxiv.org/abs/2312.11805)
- [VidHalluc](https://arxiv.org/abs/2412.03735)
- [ARGUS ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Rawal_ARGUS_Hallucination_and_Omission_Evaluation_in_Video-LLMs_ICCV_2025_paper.pdf)
- [MASH-VLM CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Bae_MASH-VLM_Mitigating_Action-Scene_Hallucination_in_Video-LLMs_through_Disentangled_Spatial-Temporal_Representations_CVPR_2025_paper.pdf)
- [Awesome-LLMs-for-Video-Understanding survey repo](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)
- [Video-MME](https://arxiv.org/abs/2405.21075)
- [Optimizing Speaker Diarization for the Classroom — JEDM](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/841)
- [Speaker Diarization in the Classroom — EDM 2024](https://educationaldatamining.org/edm2024/proceedings/2024.EDM-short-papers.33/index.html)
- [Multi-Stage Speaker Diarization for Noisy Classrooms](https://arxiv.org/html/2505.10879v1)
- [Single-Channel Classroom Diarization w/ Virtual Microphones](https://arxiv.org/abs/2207.00660)
- [Audio-Visual-Semantic Multimodal Diarization](https://arxiv.org/html/2408.12102v1)
- [COPUS original paper — CBE LSE](https://www.lifescied.org/doi/10.1187/cbe.13-08-0154)
- [Edthena Observation Copilot + Danielson](https://www.edthena.com/edthena-danielson-framework-for-teaching-observation-copilot/)
- [TeachFX](https://teachfx.com/)
- [Education Next on AI classroom observation](https://www.educationnext.org/next-gen-classroom-observations-powered-by-ai/)
- [Transana automated transcription](https://www.transana.com/blog/2023/03/25/automated_transcription/)
- [Transana](https://www.transana.com/)
- [Picard — Unlocking Multimodal Video Transcription with Gemini Part 1](https://medium.com/google-cloud/unlocking-multimodal-video-transcription-with-gemini-part1-02dc32118f41)
- [Google Codelabs — Gemini multimodal video transcription](https://codelabs.developers.google.com/gemini-multimodal-video-transcription-notebook)
- [McNeill-tradition gesture transcription review](https://www.researchgate.net/publication/273319186_Transcribing_gaze_and_gesture)
