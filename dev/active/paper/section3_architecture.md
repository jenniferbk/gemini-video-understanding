# Section 3: System Architecture

## 3.1 Overview

The pipeline takes a single classroom video file and produces a multimodal transcript: a time-aligned record of speech and visually observable activity, with speakers identified by visual features and visual descriptions interleaved with dialogue. It is implemented as a single Python script (~2,100 lines) that calls the Gemini API and requires no GPU, no local models, and no specialized infrastructure beyond a standard Python environment and an API key.

The pipeline runs in six phases: (1) chunking, (2) speaker identification, (3) chunk upload, (4) per-chunk transcription, (5) overlap resolution and assembly, and (6) dual-format output. Figure 1 [TO ADD] shows the high-level flow.

A central design choice distinguishes this approach from prior multimodal classroom video systems (e.g., Sun et al., 2025): rather than running separate audio (Whisper, Pyannote) and visual (Gemini captioning) pipelines and merging their outputs after the fact, we issue a single multimodal call per chunk that asks Gemini to produce *interleaved* speech and visual description in one pass. This preserves the temporal coupling between what a student says and what they are doing — a coupling that is central to qualitative analysis of classroom learning and that is degraded or lost when audio and visual layers are produced independently and stitched together.

## 3.2 Chunking with Overlap

Long classroom videos cannot be sent to a multimodal LLM in a single call. Real classroom recordings in our corpus range from 30 to 80 minutes; with frame sampling at 2 frames per second and HIGH media resolution (280 tokens per frame), an hour-long video would consume well over a million tokens of video alone, before any audio or prompt content. Chunking is therefore a practical necessity, but it also introduces two problems: (a) speaker identity may drift between chunks if the model relabels the same person differently, and (b) information at chunk boundaries can be lost or duplicated.

We address both with **60-second chunks and a 15-second sliding overlap.** The chunk size was selected empirically. Shorter chunks (30 seconds) increased per-chunk overhead and produced terse, context-poor descriptions; longer chunks (2–5 minutes, comparable to Sun et al., 2025) improved per-chunk context but reduced the granularity of overlap-based correction and made it harder for the model to maintain timestamp precision toward the end of each chunk. 60 seconds is short enough that the model reliably attends to the entire window and long enough to contain meaningful instructional units (a teacher question and student response, a brief gesture sequence, a board update).

The 15-second overlap serves three functions:

1. **Speaker label stabilization.** A student first seen at the end of chunk *N* is also visible at the start of chunk *N+1*, giving the model a second chance to attach a consistent label. The persistent speaker registry (Section 3.3) reinforces this.
2. **Boundary continuity.** Speech that crosses a 60-second boundary is captured in both chunks; the assembler (Section 3.5) deduplicates by timestamp.
3. **Local error correction.** If a chunk fails validation (Section 3.6) or returns a degenerate transcript, the overlapping content from the adjacent chunk recovers most of the missing material.

For a 60-minute video this produces approximately 80 chunks; for a 70-minute video, approximately 95.

## 3.3 Speaker Identification and the Visual Speaker Registry

Before transcription begins, the pipeline performs a dedicated speaker identification pass using the first two chunks of the video (configurable via `speaker_id_chunks`). These two chunks are uploaded to Gemini and processed with a prompt asking the model to identify all distinct visible speakers and return a JSON array of speaker records, each containing:

- A **unique visual label** (e.g., `Girl-BlondeHair`, `Boy-RedHoodie`, `Teacher-PinkPants`) constructed from the speaker's most distinguishing visual feature
- A **detailed physical description** (hair, clothing, position, height, accessories) intended to support consistent re-identification across long videos
- A **role** (teacher, student, researcher)

The speaker identification prompt explicitly warns against label collisions ("if two girls both wear grey shirts, do NOT use 'GreyShirt' for either") and demands a feature unique to each individual. This produces a far more useful registry than the binary teacher/student labels used by prior systems (e.g., Sun et al., 2025) or the gendered-numeric labels (`Male Student 1`, `Female Student 2`) that early commercial diarization tools default to. For qualitative researchers, knowing that "Boy-GreyTShirtFrontLeft" said something is operationally useful in a way that "Male Student 1" is not — the label itself encodes verifiable visual evidence the researcher can return to in the video.

When the teacher names a student during the lesson (e.g., "Josh, give me an answer"), the model is instructed to switch from visual label to given name from that point forward, producing transcripts that mix visual labels and proper names as the lesson reveals identity.

The resulting speaker registry is reused as part of the prompt for every subsequent chunk, anchoring labels across the full video.

## 3.4 Per-Chunk Multimodal Transcription

Each 60-second chunk is uploaded to the Gemini File API and then submitted with a composed prompt containing:

1. **The base transcription prompt** — selected by `--prompt` flag from a library of context-specific prompts (e.g., `wholeclass`, `smallgroup`, `smallgroup_ben_day1`). Each base prompt provides the lesson context (subject, grade, group composition) without overspecifying.
2. **The speaker registry** from Phase 2.
3. **A standing anti-hallucination block** (Section 4) that defines format rules, the educational research framing, speech transcription guidelines, visual description guidelines, and explicit examples of desired and undesired output.
4. **Chunk-specific framing** — chunk number, total chunks, exact duration, and instructions to transcribe the *entire* clip without stopping early or generating timestamps beyond the clip length.
5. **Continuity context** from the previous chunk's last 8 lines (configurable), so the model has dialogue and visual state heading into the new chunk.

The model returns interleaved timestamped lines in a strict format: `MM:SS Speaker: dialogue` for speech and `MM:SS [bracketed description]` for visual events. A typical chunk produces 25–60 such lines.

**Model configuration** (selected through systematic A/B testing — see Section 4):

| Parameter | Value | Rationale |
|---|---|---|
| Model | `gemini-3-flash-preview` | Best visual detail at HIGH resolution; 2.5 Pro had better audio but fabricated visual actions |
| Media resolution | HIGH | 280 tokens/frame for Gemini 3 (vs. 70 default); critical for whiteboard reading |
| Frame rate | 2 FPS | Matches Google's published recommendation; 3 FPS adds cost without quality gain |
| Temperature | 0.2 | 0.0 collapsed visual richness; 0.2 balances determinism and descriptive variety |
| Thinking budget | 4096 tokens | Sufficient for chunk-level reasoning |

Per-chunk processing time is dominated by the Gemini call itself; uploads run in parallel (default 3 concurrent). End-to-end pipeline throughput is approximately 1:1 with video duration (a 70-minute video processes in ~70 minutes), and total cost runs approximately **\$0.19 per video-hour** at current Gemini pricing.

## 3.5 Assembly and Output

After all chunks are transcribed, the assembler:

1. Sorts chunks by start time.
2. Resolves overlap by deduplicating timestamped lines from the 15-second overlap window. Lines from the later chunk are preferred when they refine speaker labels (e.g., visual label → proper name) or correct timing.
3. Renumbers timestamps from chunk-relative to video-relative.
4. Produces two output files:
   - **Research transcript** (`*_transcript.txt`): the full annotated multimodal transcript with visual descriptions, intended for qualitative coding and analysis.
   - **Transana transcript** (`*_transana.txt`): a clean, dialogue-only version stripped of bracketed visual descriptions, formatted for import into Transana, the dominant manual video-coding tool in qualitative classroom research.
5. Optionally exports an **SRT file** for use with standard subtitle workflows.

The dual-output design is deliberate. The research transcript is the novel artifact this pipeline produces, but Transana remains the de facto tool for qualitative classroom video coding, and many researchers will want to import the dialogue layer into their existing Transana workflows while consulting the multimodal version separately.

## 3.6 Validation and Retry

Each chunk's transcript passes through a `TranscriptValidator` that checks for:

- **Minimum length** (50 characters) — rejects empty or near-empty outputs
- **Error markers** — rejects transcripts beginning with bracketed error tokens
- **Excessive repetition** — detects the runaway-loop hallucination pattern in which a model repeats the same line many times
- **Timestamp coverage** — requires that at least 30% of non-empty lines start with a `MM:SS` timestamp, catching outputs where the model lapsed into prose

Failed chunks are retried up to three times with exponential backoff. Persistent failures produce a `[CHUNK FAILED]` placeholder in the assembled transcript rather than aborting the entire run; the overlap mechanism typically recovers most of the missing content from neighboring chunks.

This validation layer is lightweight but essential. In our 40+ video corpus, fewer than 1 in 100 chunks ultimately fails after retries, and those failures are almost entirely concentrated in the very last chunk of a video (where the clip may be only a few seconds of the trailing chunk window).

## 3.7 Cost Model

For a typical 60-minute classroom video at our default settings:

- **~80 chunks** × 60 seconds each (with 15s overlap)
- **Input tokens:** ~80 × (2 FPS × 60s × 280 tok/frame + audio tokens + ~800 prompt tokens) ≈ 3.0M input tokens
- **Output tokens:** ~80 × ~3,000 ≈ 240K output tokens
- **Cost:** ~\$0.19 per video-hour

This is well within reach for individual researchers working on modest grants. A 40-video corpus — the size of the dataset that motivated this work — costs roughly **\$7.60 to transcribe end to end**. The dominant cost is researcher time spent reviewing the output, not API spend.

---

*[Figure 1 — System architecture diagram — TO CREATE]*
*[Figure 2 — Annotated example of interleaved transcript output — TO CREATE]*
