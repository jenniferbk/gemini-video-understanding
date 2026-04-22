# Colleague-Engagement Package — Design Spec

**Date:** 2026-04-22
**Purpose:** Design the invitation letter and materials package for Anna Bloodworth and Uyi Ugiagbe to serve as independent raters and co-authors on the visual-description validation study.
**Paper:** "The Prompt as Apparatus: Multimodal AI Transcription for Classroom Research" (`paper.tex`)
**Unblocks:** Item 1 (visual-validation results) — the only submission blocker.

---

## Context

- **Audience:** Anna Bloodworth and Uyi Ugiagbe. Math ed grad students at UGA, peers, first-name basis. Both worked with Jennifer on an earlier messier version of the pipeline paper that didn't publish. Both know Barad and the apparatus framing.
- **Ask:** Co-authorship in exchange for ~5 hours of rating work (calibration + 45-event independent rating + disagreement resolution).
- **They are not just raters.** They are co-authors shaping the method. The package explicitly invites their input on the protocol and rubric before rating begins.
- **Data constraint:** Only TIMSS 1999 videos can appear in the paper or the Drive folder. COMS videos (Ava, Ben, Daisy, Kelly) are excluded. The 3 rating lessons are US1, US2, AU3 — all TIMSS.

## Delivery mechanics

- **Email** carrying the cover letter, pointing to a shared Google Drive folder.
- **Drive folder** holds all study materials, organized by numbered prefix.
- **Protocol and rubric** as Google Docs (not PDFs) so Anna and Uyi can inline-comment during the pre-calibration feedback window.
- **Rating spreadsheet** as Google Sheets — one shared file, separate rater columns.
- **Videos** in the Drive folder as MP4s (TIMSS permissions confirmed OK for UGA-internal sharing).

## Cover email

**Target length:** ~220 words. Collegial, peer-to-peer, unvarnished. No hedging, no performed casualness, no over-apology.

**Paragraph structure:**

1. **Warm open** (1–2 sentences). "Hope you're well, wanted to tell you where the pipeline paper landed."
2. **Acknowledgment of the earlier attempt** (2–3 sentences). Content (Jennifer's direction): "I'm sorry we didn't get that turned around — but the experience informed my approach this time. Now I have a specific plan, and I've already drafted most of the paper, so we can actually execute and submit." Owns the prior failure without dwelling; pivots to what's different now (specific plan, draft mostly done).
3. **What the paper became** (2–3 sentences). Barad / apparatus framing as the thing that cracked it open. 8 TIMSS lessons benchmarked. v10 beats Whisper+pyannote on content and substantially on speaker attribution. Names the real intellectual stakes.
4. **The ask** (3–4 sentences). Visual descriptions need human validation (no ground truth, no baseline produces them). Two independent raters. 45 events × 4 dimensions, ~5 hours total including calibration session + independent rating + disagreement resolution. **Co-authorship on the paper.**
5. **The input ask — pivot sentence.** Direct quote of Jennifer's phrasing: *"I'd like your input on the protocol and rubric before we start rating. Specifically I made some choices on the rubric and methodology — the README in the folder walks through them and where I'd most like your read."*
6. **Timeline** (1 sentence). "Hoping to do protocol review late April, calibration early May, independent rating by mid-May, so I can submit end of May / early June."
7. **Pointer to the Drive folder** (1 sentence). "Everything's in the folder linked below — README tells you where to start."
8. **Close** (1 sentence). "No pressure if the timing doesn't work — but I'd love to have you both on this."

## Drive folder structure

```
visual-validation-study/
├── README.md                          ← start here
├── 01_protocol.gdoc                   ← Google Doc, comment-enabled
├── 02_rubric_quickref.gdoc            ← 1-page rubric, comment-enabled
├── 03_calibration/
│   ├── calibration_events.pdf         ← 5 pre-rated events from US3 with Jennifer's ratings + reasoning
│   └── calibration_spreadsheet.xlsx   ← blank version for Anna/Uyi to rate first
├── 04_rating_spreadsheet.xlsx         ← 45 events, pre-populated, one rater column per person
├── 05_videos/
│   ├── US1_lesson.mp4
│   ├── US2_lesson.mp4
│   └── AU3_lesson.mp4
├── 06_FAQ.md                          ← edge cases + gotchas
└── 07_paper_draft.pdf                 ← optional: current paper.tex rendered
```

**Design decisions:**

- **Numbered prefixes** so the folder sorts intuitively — "start with 01, end with 07."
- **Calibration in its own subfolder** because norming is unmistakably the first rating step after protocol review.
- **One shared rating spreadsheet with separate rater columns** makes disagreement resolution trivial (same row, two columns side-by-side). Calibration stays separate so it doesn't contaminate the 45-event analysis.
- **Protocol + rubric as Google Docs** so comments can be anchored to the exact text they concern. Eliminates need for a separate suggestions channel.
- **Paper draft included but optional.** Framed as "read if curious." They'll want to see where their work is landing.
- **No stats / IRR files** in the folder. Jennifer computes κ, Gwet's AC2, percent agreement after submissions; Anna/Uyi don't need those materials.

## README contents

**Target length:** ~350 words. Scannable, not instructional-ish.

**Section order:**

1. **What this is** (1 paragraph).
2. **This is a draft. Push back on it.** Explicit framing — protocol, rubric, sampling, spreadsheet are Jennifer's current best guess; Anna and Uyi should flag anything that feels wrong, missing, or over-engineered. Revisions welcome before calibration, or during calibration if an issue only emerges in practice.
3. **Four specific places Jennifer wants their read** (bullets):
   - **The four dimensions** — factual accuracy, temporal precision, research relevance, level of detail. Is "level of detail" really separable from "factual accuracy"? Should there be a fifth dimension (e.g., speaker-attribution correctness as its own axis)?
   - **The 0/1/2 scale** — chosen for reliability over 5/7-point per Gwet. Open to revisiting if 3-point flattens too much signal.
   - **Sampling strata** — 5 teacher / 5 student / 5 scene per lesson. Should "scene" split into board content vs. material artifacts vs. spatial movement?
   - **Calibration size** — 5 events. Bump to 8–10 if they think it matters.
4. **Your role** (numbered list):
   1. Read the protocol and rubric (01, 02). Comment inline with any pushback.
   2. Meet with Jennifer (30 min, Zoom) to reconcile comments and revise if needed.
   3. Rate the 5 calibration events in `calibration_spreadsheet`. Work independently.
   4. Meet with Jennifer (30 min, Zoom) to compare calibration ratings and norm the scale.
   5. Rate the 45 events independently in `04_rating_spreadsheet`. ~3–4 hours.
   6. Meet with Jennifer (1 hr, Zoom) to resolve disagreements and write case notes together.
5. **How to rate** (short). Navigate to timestamp. Watch ±15s window. Rate 0/1/2 on each of the four dimensions. Leave a note if anything feels off. Don't talk to each other about ratings until both are done.
6. **Four dimensions in one line each** (quick reference).
7. **When to leave a note** — disagreements with the description, ambiguous cases, any "wait, what about…" thought.
8. **Ping Jennifer** — email/Slack line.
9. **Deadlines** — specific dates for each meeting + rating completion.

## FAQ contents

**Target length:** ~350 words. Direct, anticipates real moments of confusion.

Entries:

1. *The event is described but I can't find it in the video.* → Rate 0 on factual accuracy, 0 on temporal precision, note it.
2. *The description mentions a student I can't identify.* → Rate on whether the action happened, not on whether the label is right (speaker-ID is handled elsewhere in the paper).
3. *The description reads text from the board but I can't tell if the text is really there.* → Pause + zoom if possible; if unreadable, rate 1 on factual accuracy and note "unverifiable."
4. *Two descriptions seem to describe the same physical event.* → The sampler should have filtered near-duplicates; if one slipped through, rate both independently and flag in notes.
5. *I want to give a 1.5.* → Round down. Note the ambivalence.
6. *Video playback is laggy / timestamps drift by a second or two.* → Expected. The ±5s / ±15s rubric windows are designed for this.
7. *The event is tagged as teacher-attributed but the actor looks like a student.* → Rate factual accuracy 1 (action right, agent wrong), note it — this is the "agent confusion" failure mode from the protocol.
8. *How long should this really take?* → Plan 3–4 hours for the 45. Some events take 30 seconds, a few take 5 minutes. Normal.

## Timeline (updated)

| Step | Who | Duration | Target date |
|---|---|---|---|
| Send email + Drive link | Jennifer | — | 2026-04-23 |
| Protocol comment + suggestions | Anna, Uyi | 1 week (async) | 2026-04-24 → 2026-05-01 |
| Reconciliation meeting (revise protocol) | All 3 | 30 min (Zoom) | 2026-05-02 |
| Calibration rating (5 events) | Anna, Uyi | 30 min (async) | 2026-05-03 → 2026-05-04 |
| Norming meeting | All 3 | 30 min (Zoom) | 2026-05-05 |
| Independent rating (45 events) | Anna, Uyi | 3–4 hrs (async) | 2026-05-06 → 2026-05-15 |
| Disagreement resolution meeting | All 3 | 1 hr (Zoom) | 2026-05-16 |
| Compile ratings, compute IRR | Jennifer | 1 hr | 2026-05-17 |
| Write 1–2 paragraph results narrative | Jennifer (with rater input) | — | 2026-05-18 |
| Paper submission window | Jennifer | — | 2026-05-25 → 2026-06-05 |

The pre-calibration feedback window (1 week async + 30-min meeting) is new — it makes the "your suggestions" ask substantive rather than performative.

## Deliverables this spec produces

When this design is implemented, the following artifacts exist:

1. **`colleague_email_draft.md`** — the cover email text, ready for Jennifer to review and send.
2. **Drive folder contents** (the files listed above, excluding videos which Jennifer uploads separately).
3. **`README.md`** — inside the Drive folder.
4. **`FAQ.md`** — inside the Drive folder.
5. **`calibration_events.pdf`** — 5 pre-rated US3 events with Jennifer's ratings + brief reasoning (Jennifer generates this; implementation plan can include a script or template).
6. **`calibration_spreadsheet.xlsx`** — blank version of the calibration events for Anna/Uyi.
7. **`04_rating_spreadsheet.xlsx`** — 45-event spreadsheet, pre-populated from `sample_visual_events.py` output, with rater columns added.

## Out of scope for this spec

- Actually creating the Drive folder (Jennifer does this manually with her own credentials).
- Uploading videos (Jennifer does this manually).
- Scheduling the Zoom meetings (Jennifer does this after Anna/Uyi accept).
- Item 10 from the paper punch list (Rymes apparatus-demo excerpt). That requires re-running the Rymes prompt against a TIMSS video with a generalized prompt — separate work stream.

## Open questions for Jennifer's review

None flagged — all design decisions above have been confirmed in the brainstorming dialog.
