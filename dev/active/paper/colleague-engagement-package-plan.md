# Colleague-Engagement Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce all text and data artifacts for the colleague-engagement package so Jennifer can assemble the Drive folder and send the invitation email to Anna Bloodworth and Uyi Ugiagbe.

**Architecture:** Content-generation plan. Each deliverable is a separate file in `dev/active/paper/colleague_package/`. Drafts are written in Jennifer's collegial peer-to-peer voice, then run through the `humanize` skill as a final scrub. Jennifer uploads the folder to Google Drive manually (videos, final conversions, and sharing permissions are outside the automation scope).

**Tech Stack:** Markdown (email, README, FAQ, scaffold), Python + openpyxl (spreadsheets), reuse of existing `visual_validation_sample.csv`.

**Reference spec:** `dev/active/paper/colleague-engagement-package-design.md`

---

## File Structure

All outputs live in `dev/active/paper/colleague_package/`:

```
colleague_package/
├── colleague_email_draft.md           ← Task 1
├── README.md                          ← Task 2
├── FAQ.md                             ← Task 3
├── build_rating_spreadsheet.py        ← Task 4 (script)
├── 04_rating_spreadsheet.xlsx         ← Task 4 (output)
├── build_calibration_spreadsheet.py   ← Task 5 (script)
├── calibration_spreadsheet.xlsx       ← Task 5 (output)
└── calibration_events_scaffold.md     ← Task 6
```

**Input data** (already exists, do not regenerate):
- `dev/active/paper/visual_validation_sample.csv` — 45 events, 12 columns. Verified: 15 events per lesson (US1/US2/AU3), 15 events per attribution stratum (teacher/student/scene).

**Out of scope** (Jennifer does manually):
- Drive folder creation, video upload, sharing permissions.
- Converting `01_protocol.tex` to Google Doc format.
- Rendering `paper.tex` to `07_paper_draft.pdf`.
- Actually filling in `calibration_events_scaffold.md` with her ratings + reasoning.
- Sending the email.

---

## Task 1: Draft cover email

**Files:**
- Create: `dev/active/paper/colleague_package/colleague_email_draft.md`

**Target:** ~220–260 words. Collegial, peer-to-peer. No hedging, no performed casualness, no over-apology. The acknowledgment of the prior paper is 2–3 sentences per Jennifer's explicit direction.

- [ ] **Step 1: Write the email draft**

Create `dev/active/paper/colleague_package/colleague_email_draft.md` with exactly this structure. Write the prose in each section; do not leave headers or bracketed placeholders.

```markdown
# Cover Email — Visual Validation Invitation

**To:** Anna Bloodworth, Uyi Ugiagbe
**From:** Jennifer Kleiman
**Subject:** [DRAFT — revise before sending] The pipeline paper — want to be on this with me?

---

Hi Anna, hi Uyi —

[Paragraph 1 — warm open, 1–2 sentences: "Hope you're both well" / "Wanted to tell you where the pipeline paper landed."]

[Paragraph 2 — acknowledgment of prior paper, 2–3 sentences using Jennifer's exact phrasing: "I'm sorry we didn't get that turned around — but the experience informed my approach this time. Now I have a specific plan, and I've already drafted most of the paper, so we can actually execute and submit."]

[Paragraph 3 — what the paper became, 2–3 sentences: the Barad/apparatus framing as the thing that cracked it open; 8 TIMSS lessons benchmarked; v10 beats Whisper+pyannote on content (mean F1 0.926) and substantially on speaker attribution (+12.5 pp role accuracy, +24.6 pp student accuracy). Target venue JLS or C&I.]

[Paragraph 4 — the ask, 3–4 sentences: visual descriptions are the part that needs human validation (no baseline produces them, no ground truth exists); 45 events × 4 dimensions; ~5 hours total (calibration + independent rating + disagreement resolution); co-authorship on the paper.]

[Paragraph 5 — input request, verbatim Jennifer phrasing: "I'd like your input on the protocol and rubric before we start rating. Specifically I made some choices on the rubric and methodology — the README in the folder walks through them and where I'd most like your read."]

[Paragraph 6 — timeline, 1 sentence: "Hoping to do protocol review late April, calibration early May, independent rating by mid-May, so I can submit end of May / early June."]

[Paragraph 7 — Drive folder pointer, 1 sentence: "Everything's in the folder [LINK] — README tells you where to start."]

[Paragraph 8 — close, 1 sentence: "No pressure if the timing doesn't work — but I'd love to have you both on this."]

— Jennifer
```

After writing the structure, fill in each `[bracketed guide]` with actual prose. The final file should have no remaining brackets except `[LINK]` (Jennifer inserts the Drive URL) and `[DRAFT — revise before sending]` in the subject.

- [ ] **Step 2: Verify length and check against spec**

Run: `wc -w dev/active/paper/colleague_package/colleague_email_draft.md`
Expected: 220–280 words (excluding header and horizontal rule).

Grep for accidental AI phrases that need to go:
Run: `grep -iE "I hope this (email|message) finds|delve|navigate|leverage|synergy|robust|comprehensive|furthermore|moreover|in conclusion" dev/active/paper/colleague_package/colleague_email_draft.md`
Expected: no matches.

- [ ] **Step 3: Run humanize scrub**

Invoke the `humanize` skill on `colleague_email_draft.md`. Apply its voice-authenticity suggestions inline. Do not dilute the content — only remove AI-shaped phrasing.

- [ ] **Step 4: Commit**

```bash
git add dev/active/paper/colleague_package/colleague_email_draft.md
git commit -m "feat(paper): draft colleague invitation email for validation study"
```

---

## Task 2: Draft README.md (Drive folder orientation)

**Files:**
- Create: `dev/active/paper/colleague_package/README.md`

**Target:** ~300–400 words. Scannable, not instructional-ish. Per spec §README contents.

- [ ] **Step 1: Write the README**

Create with this exact section structure. Each section is prose, not bullets-only, except where specified.

```markdown
# Visual Validation Study — Start Here

## What this is

[1 paragraph: this folder contains materials for the visual-description validation study for the paper "The Prompt as Apparatus: Multimodal AI Transcription for Classroom Research." You're rating 45 visual events sampled from 3 TIMSS lessons (US1, US2, AU3). Your ratings validate the one thing no baseline system produces — interleaved visual descriptions of classroom activity.]

## This is a draft. Push back on it.

[1 paragraph: the protocol, rubric, sampling strategy, and rating spreadsheet are Jennifer's current best guess. You know this kind of work as well as she does. If something feels wrong, missing, or over-engineered, flag it. Revisions welcome before calibration, or during calibration if the issue only shows up once you start rating.]

Four places Jennifer specifically wants your read:

- **The four dimensions** — factual accuracy, temporal precision, research relevance, level of detail. Is "level of detail" really separable from "factual accuracy"? Should there be a fifth dimension (e.g., speaker-attribution correctness as its own axis)?
- **The 0/1/2 scale** — chosen for reliability over 5/7-point per Gwet. Open to revisiting if a 3-point scale flattens too much signal.
- **Sampling strata** — 5 teacher / 5 student / 5 scene per lesson. Should "scene" split into board content vs. material artifacts vs. spatial movement?
- **Calibration size** — 5 events. Bump to 8–10 if you think it matters.

Leave comments directly on `01_protocol` or `02_rubric_quickref` (Google Doc comments anchored to text work best).

## Your role

1. **Read** `01_protocol` and `02_rubric_quickref`. Comment inline with any pushback. (1–2 hours, async)
2. **Meet** with Jennifer (30 min, Zoom) to reconcile comments and revise protocol if needed.
3. **Rate** the 5 calibration events in `03_calibration/calibration_spreadsheet`. Work independently.
4. **Meet** with Jennifer (30 min, Zoom) to compare calibration ratings and norm the scale.
5. **Rate** the 45 events independently in `04_rating_spreadsheet`. ~3–4 hours, async.
6. **Meet** with Jennifer (1 hr, Zoom) to resolve disagreements and write case notes together.

## How to rate

- Navigate to the event's timestamp. Watch a 30-second window centered on it (±15s).
- Rate 0/1/2 on each of the four dimensions.
- Leave a note in the `notes` column if anything feels off — ambiguous cases, disagreements with the description, any "wait, what about…" thought.
- **Don't talk to each other about ratings until both of you are done.** Independence is the whole point.

## The four dimensions in one line each

- **Factual Accuracy** — Is the described event actually in the video?
- **Temporal Precision** — Is it at the timestamp where the transcript placed it?
- **Research Relevance** — Does it provide information useful for understanding student learning?
- **Level of Detail** — Is the specificity appropriate for what's observable?

See `02_rubric_quickref` for scale anchors.

## When to leave a note

- You disagree with the description in any substantive way.
- The event is ambiguous and your rating feels like a coin flip.
- Something about the event surprised you.
- You spot a pattern (e.g., "the transcript keeps calling this student by the wrong label").

## Questions

Ping Jennifer: jennifer.kleiman@uga.edu or Slack.

## Deadlines

- Protocol comments: by **2026-05-01**.
- Protocol reconciliation meeting: **2026-05-02** (Jennifer will schedule).
- Calibration rating: by **2026-05-04**.
- Norming meeting: **2026-05-05**.
- Independent rating (45 events): by **2026-05-15**.
- Disagreement resolution meeting: **2026-05-16**.
```

- [ ] **Step 2: Verify length**

Run: `wc -w dev/active/paper/colleague_package/README.md`
Expected: 300–450 words.

- [ ] **Step 3: Grep check**

Run: `grep -iE "delve|navigate the|leverage|synergy|robust|comprehensive|furthermore|moreover|in conclusion|as an AI" dev/active/paper/colleague_package/README.md`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add dev/active/paper/colleague_package/README.md
git commit -m "feat(paper): add README for validation-study Drive folder"
```

---

## Task 3: Draft FAQ.md

**Files:**
- Create: `dev/active/paper/colleague_package/FAQ.md`

**Target:** ~300–400 words. Direct, anticipates the actual moments of confusion during rating.

- [ ] **Step 1: Write the FAQ**

Create with exactly these 8 entries:

```markdown
# FAQ — Visual Validation Rating

## 1. The event is described but I can't find it in the video.

Rate **0 on factual accuracy** and **0 on temporal precision**. Leave a note describing what you looked for and where. This is a hallucination — important to catch and document.

## 2. The description mentions a student I can't identify.

Rate on whether the action happened, not on whether the label is right. Speaker-ID accuracy is handled elsewhere in the paper; your job here is the visual event itself.

Example: "S-BoyRed points at the worksheet" — if *a* student pointed at the worksheet near that timestamp, that's factually accurate even if you can't confirm which specific student.

## 3. The description reads text from the board but I can't tell if the text is really there.

Pause and zoom if your player allows. If the text is genuinely unreadable from video quality, rate **1 on factual accuracy** (partially accurate — we can't verify) and note "unverifiable from video."

## 4. Two descriptions seem to describe the same physical event.

The sampler should have filtered near-duplicates within 5 seconds. If one slipped through, rate both independently and flag in the notes column. Don't try to "pick the better one" — that's not your call, and the data point matters.

## 5. I want to give a 1.5.

Round down. Note the ambivalence in the notes column. The 3-point scale is intentional (Gwet's argument about reliability at small N), but your narrative comments are where nuance lives.

## 6. Video playback is laggy / timestamps drift by a second or two.

Expected. The ±5s (precise) / ±15s (proximate) windows in the rubric are designed for this. Don't penalize small drift that's within the rubric tolerances.

## 7. The event is tagged as teacher-attributed but the actor looks like a student to me.

Rate **factual accuracy 1** (action right, agent wrong). Note it — this is the "agent confusion" failure mode from the protocol, and it's something the paper specifically reports on.

## 8. How long should this really take?

Plan 3–4 hours for the 45 events. Some take 30 seconds to rate. A few will take 5 minutes because you'll want to rewatch. That's normal and fine.

If you find yourself consistently at 10+ minutes per event, stop and ping Jennifer — something about the protocol isn't working.
```

- [ ] **Step 2: Verify length**

Run: `wc -w dev/active/paper/colleague_package/FAQ.md`
Expected: 300–450 words.

- [ ] **Step 3: Commit**

```bash
git add dev/active/paper/colleague_package/FAQ.md
git commit -m "feat(paper): add FAQ for validation-study rater edge cases"
```

---

## Task 4: Build rating spreadsheet

**Files:**
- Create: `dev/active/paper/colleague_package/build_rating_spreadsheet.py`
- Create: `dev/active/paper/colleague_package/04_rating_spreadsheet.xlsx`

**Input:** `dev/active/paper/visual_validation_sample.csv` (45 rows, 12 columns).

**Output design:** Each of the 45 events becomes one row. The rating columns (factual_accuracy, temporal_precision, research_relevance, level_of_detail, notes) in the source CSV are single-rater; we need to fan them out to two raters. Final schema:

| event_id | lesson | timestamp | attribution | speaker_label | event_text | context | AB_factual | AB_temporal | AB_relevance | AB_detail | AB_notes | UU_factual | UU_temporal | UU_relevance | UU_detail | UU_notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

AB = Anna Bloodworth, UU = Uyi Ugiagbe. Each rater pair of 5 columns is visually grouped by cell fill color (light blue for AB, light green for UU) so the rater knows which block to fill.

- [ ] **Step 1: Install openpyxl in the project venv**

Run: `/Users/jenniferkleiman/Documents/COMS/src/python/venv/bin/pip install openpyxl`
Expected: "Successfully installed openpyxl-X.Y.Z"

- [ ] **Step 2: Write the build script**

Create `dev/active/paper/colleague_package/build_rating_spreadsheet.py`:

```python
#!/usr/bin/env python3
"""Build 04_rating_spreadsheet.xlsx from visual_validation_sample.csv.

Fans out the single rater-column block into two (AB + UU), adds header styling,
freezes panes, sets column widths, and applies fill colors per rater block.
"""
from __future__ import annotations
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent
CSV_PATH = HERE.parent / "visual_validation_sample.csv"
OUT_PATH = HERE / "04_rating_spreadsheet.xlsx"

META_COLS = ["event_id", "lesson", "timestamp", "attribution", "speaker_label", "event_text", "context"]
RATER_COLS = ["factual", "temporal", "relevance", "detail", "notes"]
RATERS = [
    ("AB", "DCE7F1"),   # Anna Bloodworth — light blue
    ("UU", "D5E8D4"),   # Uyi Ugiagbe — light green
]

HEADER_FILL = PatternFill("solid", fgColor="4A4A4A")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def build_headers() -> tuple[list[str], dict[int, str]]:
    """Return (headers, col_index_to_fill_color)."""
    headers = list(META_COLS)
    fills: dict[int, str] = {}
    for rater, color in RATERS:
        for field in RATER_COLS:
            headers.append(f"{rater}_{field}")
            fills[len(headers)] = color  # 1-indexed
    return headers, fills


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Input CSV missing: {CSV_PATH}")

    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 45:
        print(f"WARNING: expected 45 rows, found {len(rows)}")

    headers, fills = build_headers()

    wb = Workbook()
    ws = wb.active
    ws.title = "Ratings"

    for col_idx, name in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row_idx, r in enumerate(rows, start=2):
        for col_idx, field in enumerate(META_COLS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=r.get(field, ""))
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        # Rater columns left blank; apply fill for visibility.
        for col_idx, color in fills.items():
            ws.cell(row=row_idx, column=col_idx).fill = PatternFill("solid", fgColor=color)

    # Column widths: meta readable, context wide, rater numeric narrow, notes wide.
    widths = {"event_id": 12, "lesson": 8, "timestamp": 10, "attribution": 12,
              "speaker_label": 16, "event_text": 40, "context": 50}
    for idx, name in enumerate(headers, start=1):
        base = name.split("_", 1)[-1] if "_" in name and name[:2] in {"AB", "UU"} else name
        if name in widths:
            w = widths[name]
        elif base == "notes":
            w = 30
        else:
            w = 11
        ws.column_dimensions[get_column_letter(idx)].width = w

    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "A2"

    wb.save(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({len(rows)} events, {len(headers)} columns)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script**

Run: `/Users/jenniferkleiman/Documents/COMS/src/python/venv/bin/python3 dev/active/paper/colleague_package/build_rating_spreadsheet.py`
Expected: `Wrote .../04_rating_spreadsheet.xlsx (45 events, 17 columns)`

- [ ] **Step 4: Verify output**

Run: `/Users/jenniferkleiman/Documents/COMS/src/python/venv/bin/python3 -c "from openpyxl import load_workbook; wb = load_workbook('/Users/jenniferkleiman/Documents/COMS/dev/active/paper/colleague_package/04_rating_spreadsheet.xlsx'); ws = wb.active; print(f'Rows: {ws.max_row}, Cols: {ws.max_column}'); print('Headers:', [ws.cell(1, c).value for c in range(1, ws.max_column + 1)])"`
Expected: `Rows: 46, Cols: 17` (header + 45 events) with the AB_/UU_ prefixed rater columns at the end.

- [ ] **Step 5: Commit**

```bash
git add dev/active/paper/colleague_package/build_rating_spreadsheet.py dev/active/paper/colleague_package/04_rating_spreadsheet.xlsx
git commit -m "feat(paper): build rating spreadsheet with two-rater columns"
```

---

## Task 5: Build calibration spreadsheet

**Files:**
- Create: `dev/active/paper/colleague_package/build_calibration_spreadsheet.py`
- Create: `dev/active/paper/colleague_package/calibration_spreadsheet.xlsx`

**Input:** manually specified — 5 events pulled from US3 (a lesson **not** in the rating sample, per protocol §Calibration). The calibration events must be selected by Jennifer; this task produces the spreadsheet skeleton that she can populate.

**Decision:** Since Jennifer must choose which US3 events to use for calibration (based on her judgment of what spans the rating scale well), this script generates a skeleton with **5 blank event rows** plus rater columns. Jennifer will fill in event_id/timestamp/event_text/context manually by referencing her US3 v10 transcript.

**Alternative considered:** Auto-sample 5 events from US3 using the same stratification logic. Rejected because calibration events should be deliberately chosen to span the scale (e.g., one clear hallucination, one clear success, one borderline case), not randomly sampled.

- [ ] **Step 1: Write the build script**

Create `dev/active/paper/colleague_package/build_calibration_spreadsheet.py`:

```python
#!/usr/bin/env python3
"""Build calibration_spreadsheet.xlsx — blank template for 5 calibration events.

Jennifer fills in event_id/lesson/timestamp/event_text/context manually by
pulling from the US3 v10 transcript. Anna and Uyi rate; a third rater column
(JK) holds Jennifer's pre-rated reference ratings revealed during the norming
meeting.
"""
from __future__ import annotations
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).parent
OUT_PATH = HERE / "calibration_spreadsheet.xlsx"

META_COLS = ["event_id", "lesson", "timestamp", "attribution", "speaker_label", "event_text", "context"]
RATER_COLS = ["factual", "temporal", "relevance", "detail", "notes"]
RATERS = [
    ("AB", "DCE7F1"),   # Anna — light blue
    ("UU", "D5E8D4"),   # Uyi — light green
    ("JK", "FFF2CC"),   # Jennifer (reference; hidden until norming meeting) — light yellow
]

HEADER_FILL = PatternFill("solid", fgColor="4A4A4A")
HEADER_FONT = Font(bold=True, color="FFFFFF")
N_CALIBRATION_EVENTS = 5


def main() -> None:
    headers = list(META_COLS)
    fills: dict[int, str] = {}
    for rater, color in RATERS:
        for field in RATER_COLS:
            headers.append(f"{rater}_{field}")
            fills[len(headers)] = color

    wb = Workbook()
    ws = wb.active
    ws.title = "Calibration"

    for col_idx, name in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for row_idx in range(2, 2 + N_CALIBRATION_EVENTS):
        # Empty meta cells — Jennifer fills in.
        for col_idx in range(1, len(META_COLS) + 1):
            ws.cell(row=row_idx, column=col_idx).alignment = Alignment(vertical="top", wrap_text=True)
        # Rater fill colors.
        for col_idx, color in fills.items():
            ws.cell(row=row_idx, column=col_idx).fill = PatternFill("solid", fgColor=color)

    widths = {"event_id": 14, "lesson": 8, "timestamp": 10, "attribution": 12,
              "speaker_label": 16, "event_text": 40, "context": 50}
    for idx, name in enumerate(headers, start=1):
        if name in widths:
            w = widths[name]
        elif name.endswith("_notes"):
            w = 30
        else:
            w = 11
        ws.column_dimensions[get_column_letter(idx)].width = w

    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "A2"

    wb.save(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({N_CALIBRATION_EVENTS} blank event rows, {len(headers)} columns)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `/Users/jenniferkleiman/Documents/COMS/src/python/venv/bin/python3 dev/active/paper/colleague_package/build_calibration_spreadsheet.py`
Expected: `Wrote .../calibration_spreadsheet.xlsx (5 blank event rows, 22 columns)`

- [ ] **Step 3: Verify**

Run: `/Users/jenniferkleiman/Documents/COMS/src/python/venv/bin/python3 -c "from openpyxl import load_workbook; wb = load_workbook('/Users/jenniferkleiman/Documents/COMS/dev/active/paper/colleague_package/calibration_spreadsheet.xlsx'); ws = wb.active; print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')"`
Expected: `Rows: 6, Cols: 22` (header + 5 calibration rows; 7 meta + 3×5 rater blocks).

- [ ] **Step 4: Commit**

```bash
git add dev/active/paper/colleague_package/build_calibration_spreadsheet.py dev/active/paper/colleague_package/calibration_spreadsheet.xlsx
git commit -m "feat(paper): add calibration spreadsheet template"
```

---

## Task 6: Create calibration events scaffold

**Files:**
- Create: `dev/active/paper/colleague_package/calibration_events_scaffold.md`

**Purpose:** Structured markdown for Jennifer to fill in with her 5 chosen US3 calibration events, her ratings on each of the 4 dimensions, and her reasoning. Anna and Uyi use this (revealed at the norming meeting) to calibrate their interpretation of the scale anchors.

**Why scaffold not content:** The actual calibration ratings + reasoning are Jennifer's norming anchors — they must reflect her considered judgment, not a draft from Claude. The scaffold enforces structure so she's not staring at a blank page.

- [ ] **Step 1: Write the scaffold**

Create `dev/active/paper/colleague_package/calibration_events_scaffold.md`:

```markdown
# Calibration Events — Jennifer's Reference Ratings

**Lesson:** US3 (deliberately outside the rating sample of US1/US2/AU3)
**Instructions for Jennifer:** Select 5 events from the US3 v10 transcript that span the rating scale — ideally one clear 2/2/2/2, one clear 0 on factual accuracy (confirmed hallucination if one exists; otherwise a 0 on research relevance), one 1 on at least one dimension (partial or ambiguous), and two more that cover underused corners of the rubric (e.g., a temporal-precision 1, a level-of-detail mismatch). Anna and Uyi rate blind first, then this document is revealed at the norming meeting.

---

## Event 1

- **Event ID:** `US3_CAL_01`
- **Timestamp:** `[MM:SS]`
- **Attribution:** `[teacher | student | scene]`
- **Speaker label:** `[label or <visual>]`
- **Event text:** `[the v10 description, verbatim from transcript]`
- **Context (±15s):** `[surrounding transcript lines]`

**Jennifer's ratings:**

| Dimension | Rating (0/1/2) | Reasoning |
|---|---|---|
| Factual Accuracy | | |
| Temporal Precision | | |
| Research Relevance | | |
| Level of Detail | | |

**Why this event for calibration:** [1–2 sentences: what this event teaches raters about the rubric.]

---

## Event 2

- **Event ID:** `US3_CAL_02`
- **Timestamp:** `[MM:SS]`
- **Attribution:** `[teacher | student | scene]`
- **Speaker label:** `[label or <visual>]`
- **Event text:** `[the v10 description, verbatim from transcript]`
- **Context (±15s):** `[surrounding transcript lines]`

**Jennifer's ratings:**

| Dimension | Rating (0/1/2) | Reasoning |
|---|---|---|
| Factual Accuracy | | |
| Temporal Precision | | |
| Research Relevance | | |
| Level of Detail | | |

**Why this event for calibration:** [1–2 sentences.]

---

## Event 3

- **Event ID:** `US3_CAL_03`
- **Timestamp:** `[MM:SS]`
- **Attribution:** `[teacher | student | scene]`
- **Speaker label:** `[label or <visual>]`
- **Event text:** `[the v10 description, verbatim from transcript]`
- **Context (±15s):** `[surrounding transcript lines]`

**Jennifer's ratings:**

| Dimension | Rating (0/1/2) | Reasoning |
|---|---|---|
| Factual Accuracy | | |
| Temporal Precision | | |
| Research Relevance | | |
| Level of Detail | | |

**Why this event for calibration:** [1–2 sentences.]

---

## Event 4

- **Event ID:** `US3_CAL_04`
- **Timestamp:** `[MM:SS]`
- **Attribution:** `[teacher | student | scene]`
- **Speaker label:** `[label or <visual>]`
- **Event text:** `[the v10 description, verbatim from transcript]`
- **Context (±15s):** `[surrounding transcript lines]`

**Jennifer's ratings:**

| Dimension | Rating (0/1/2) | Reasoning |
|---|---|---|
| Factual Accuracy | | |
| Temporal Precision | | |
| Research Relevance | | |
| Level of Detail | | |

**Why this event for calibration:** [1–2 sentences.]

---

## Event 5

- **Event ID:** `US3_CAL_05`
- **Timestamp:** `[MM:SS]`
- **Attribution:** `[teacher | student | scene]`
- **Speaker label:** `[label or <visual>]`
- **Event text:** `[the v10 description, verbatim from transcript]`
- **Context (±15s):** `[surrounding transcript lines]`

**Jennifer's ratings:**

| Dimension | Rating (0/1/2) | Reasoning |
|---|---|---|
| Factual Accuracy | | |
| Temporal Precision | | |
| Research Relevance | | |
| Level of Detail | | |

**Why this event for calibration:** [1–2 sentences.]

---

## Norming discussion prompts (for the meeting)

After Anna and Uyi reveal their calibration ratings, discuss:

1. Where did ratings diverge by ≥1 point? What made the event look different from different vantages?
2. For each of the 4 dimensions, is the 0/1/2 anchor language in the rubric capturing the right cut?
3. Is there a failure mode we've seen in calibration that the rubric doesn't have a home for? (If yes, amend the rubric before independent rating begins.)
```

- [ ] **Step 2: Verify structure**

Run: `grep -c "^## Event" dev/active/paper/colleague_package/calibration_events_scaffold.md`
Expected: `5`

Run: `grep -c "Jennifer's ratings:" dev/active/paper/colleague_package/calibration_events_scaffold.md`
Expected: `5`

- [ ] **Step 3: Commit**

```bash
git add dev/active/paper/colleague_package/calibration_events_scaffold.md
git commit -m "feat(paper): scaffold calibration events doc for Jennifer to populate"
```

---

## Task 7: Final package index

**Files:**
- Create: `dev/active/paper/colleague_package/PACKAGE_INDEX.md` (index of what's in the folder, for Jennifer's reference before she uploads to Drive)

**Purpose:** A local-only crib sheet so Jennifer knows, when she gets to the Drive upload step, exactly which files map to which Drive folder slots (01/02/03/04/05/06/07 per the spec's folder structure). The `colleague_package` directory contents don't use numeric prefixes locally because git doesn't need them; Drive ordering matters only on Drive.

- [ ] **Step 1: Write the index**

Create `dev/active/paper/colleague_package/PACKAGE_INDEX.md`:

```markdown
# Local Package Index → Drive Folder Mapping

This directory holds the assets for the visual-validation colleague package.
When Jennifer uploads to Drive, rename per the numbered scheme below.

| Local file | Drive destination | Format at upload | Notes |
|---|---|---|---|
| `colleague_email_draft.md` | (not uploaded — email body) | — | Paste into Gmail; add Drive link before sending. |
| `README.md` | `visual-validation-study/README.md` | Markdown or Google Doc | Keep as .md; Drive renders markdown. |
| `FAQ.md` | `visual-validation-study/06_FAQ.md` | Markdown | Rename with `06_` prefix on upload. |
| (from `../visual_validation_protocol.tex`) | `visual-validation-study/01_protocol.gdoc` | Google Doc | Render tex → PDF or Doc; enable comments. |
| (new, 1-page summary) | `visual-validation-study/02_rubric_quickref.gdoc` | Google Doc | Extract from protocol §Rating Rubric; enable comments. |
| `calibration_events_scaffold.md` (after Jennifer fills in) | `visual-validation-study/03_calibration/calibration_events.pdf` | PDF (export from Doc) | Hidden until norming meeting. |
| `calibration_spreadsheet.xlsx` | `visual-validation-study/03_calibration/calibration_spreadsheet.xlsx` | Google Sheets | Auto-convert on upload. |
| `04_rating_spreadsheet.xlsx` | `visual-validation-study/04_rating_spreadsheet.xlsx` | Google Sheets | Auto-convert on upload. |
| (Jennifer's video files) | `visual-validation-study/05_videos/` | MP4 | US1, US2, AU3 lesson videos. |
| (from `../paper.tex`) | `visual-validation-study/07_paper_draft.pdf` | PDF | Optional; label "draft — for context." |

## Upload checklist

- [ ] Create top-level Drive folder `visual-validation-study/`.
- [ ] Set folder sharing: Anna Bloodworth + Uyi Ugiagbe as commenters (not editors, so spreadsheet rater columns don't get accidentally restructured).
- [ ] Upload all files per the table.
- [ ] For `01_protocol` and `02_rubric_quickref`: enable "Anyone with link can comment."
- [ ] For `04_rating_spreadsheet` and `calibration_spreadsheet`: give Anna and Uyi EDITOR access (they need to fill in their rater columns).
- [ ] Test the Drive link by opening in an incognito browser to confirm permissions.
- [ ] Copy the folder URL into the `[LINK]` placeholder in `colleague_email_draft.md`.
- [ ] Send.
```

- [ ] **Step 2: Commit**

```bash
git add dev/active/paper/colleague_package/PACKAGE_INDEX.md
git commit -m "docs(paper): add local-to-Drive mapping for colleague package"
```

---

## Self-review checklist

Once all tasks are complete, verify:

- [ ] All 6 deliverables exist and are non-empty:
  ```bash
  ls -la /Users/jenniferkleiman/Documents/COMS/dev/active/paper/colleague_package/
  ```
  Expected: `colleague_email_draft.md`, `README.md`, `FAQ.md`, `build_rating_spreadsheet.py`, `04_rating_spreadsheet.xlsx`, `build_calibration_spreadsheet.py`, `calibration_spreadsheet.xlsx`, `calibration_events_scaffold.md`, `PACKAGE_INDEX.md`.

- [ ] No AI-shaped phrasing remains in the email or README:
  ```bash
  grep -riE "delve|navigate (the|through)|leverage|synergy|robust|comprehensive|furthermore|moreover|in conclusion|as an AI|i hope this" \
    /Users/jenniferkleiman/Documents/COMS/dev/active/paper/colleague_package/*.md
  ```
  Expected: no matches.

- [ ] The rating spreadsheet has 45 event rows with AB_ and UU_ rater columns:
  ```bash
  /Users/jenniferkleiman/Documents/COMS/src/python/venv/bin/python3 -c "
  from openpyxl import load_workbook
  wb = load_workbook('/Users/jenniferkleiman/Documents/COMS/dev/active/paper/colleague_package/04_rating_spreadsheet.xlsx')
  ws = wb.active
  headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
  assert ws.max_row == 46, f'want 46 rows, got {ws.max_row}'
  assert any(h.startswith('AB_') for h in headers), 'missing AB columns'
  assert any(h.startswith('UU_') for h in headers), 'missing UU columns'
  print('OK')
  "
  ```
  Expected: `OK`

- [ ] Offer Jennifer a final read of the email draft before she sends.

## Execution notes

- **Sequence:** Tasks 1–3 (text drafts) can be done in any order. Tasks 4–5 (spreadsheets) depend on openpyxl being installed in the venv (Task 4 Step 1). Task 6 (calibration scaffold) has no dependencies. Task 7 (index) depends on all prior deliverables existing.
- **If inline execution:** Do Tasks 1–3 first, then install openpyxl once, then 4–5, then 6–7.
- **If subagent-driven:** Tasks 1, 2, 3, 6 can all run in parallel (no shared state). Tasks 4 + 5 can run in parallel after Task 4's Step 1 pip install completes (or dispatch a one-off subagent for the install as a prerequisite). Task 7 serializes last.
- **Commit discipline:** one commit per task. No squashing — the granular history helps Jennifer review before sending.
