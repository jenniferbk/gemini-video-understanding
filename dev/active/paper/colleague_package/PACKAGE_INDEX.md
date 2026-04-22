# Local Package Index → Drive Folder Mapping

This directory holds the assets for the visual-validation colleague package.
When uploading to Drive, rename per the numbered scheme below.

| Local file | Drive destination | Format at upload | Notes |
|---|---|---|---|
| `colleague_email_draft.md` | (not uploaded; email body) | — | Paste into Gmail; add Drive link before sending. |
| `README.md` | `visual-validation-study/README.md` | Markdown or Google Doc | Drive renders markdown. |
| `FAQ.md` | `visual-validation-study/06_FAQ.md` | Markdown | Rename with `06_` prefix on upload. |
| (from `../visual_validation_protocol.tex`) | `visual-validation-study/01_protocol.gdoc` | Google Doc | Render tex → PDF or Doc; enable comments. |
| (new, 1-page summary) | `visual-validation-study/02_rubric_quickref.gdoc` | Google Doc | Extract from protocol §Rating Rubric; enable comments. |
| `calibration_events_scaffold.md` (after filling in) | `visual-validation-study/03_calibration/calibration_events.pdf` | PDF (export from Doc) | Hidden until norming meeting. |
| `calibration_spreadsheet.xlsx` | `visual-validation-study/03_calibration/calibration_spreadsheet.xlsx` | Google Sheets | Auto-convert on upload. |
| `04_rating_spreadsheet.xlsx` | `visual-validation-study/04_rating_spreadsheet.xlsx` | Google Sheets | Auto-convert on upload. |
| (video files) | `visual-validation-study/05_videos/` | MP4 | US1, US2, AU3 lesson videos. |
| (from `../paper.tex`) | `visual-validation-study/07_paper_draft.pdf` | PDF | Optional; label "draft, for context." |

## Upload checklist

- [ ] Create top-level Drive folder `visual-validation-study/`.
- [ ] Set folder sharing: Anna Bloodworth + Uyi Ugiagbe as commenters (not editors, so spreadsheet rater columns don't get accidentally restructured).
- [ ] Upload all files per the table.
- [ ] For `01_protocol` and `02_rubric_quickref`: enable "Anyone with link can comment."
- [ ] For `04_rating_spreadsheet` and `calibration_spreadsheet`: give Anna and Uyi EDITOR access (they need to fill in their rater columns).
- [ ] Test the Drive link by opening in an incognito browser to confirm permissions.
- [ ] Copy the folder URL into the `[LINK]` placeholder in `colleague_email_draft.md`.
- [ ] Send.

## Out of scope for automation

- Rendering `visual_validation_protocol.tex` to PDF or Google Doc (run `pdflatex`, or export through Overleaf).
- Extracting the 1-page rubric quick-reference from the protocol (copy-paste §Rating Rubric).
- Rendering `paper.tex` to `07_paper_draft.pdf` (run `pdflatex`).
- Uploading videos to Drive.
- Scheduling the Zoom meetings.
- The actual colleague email (paste from `colleague_email_draft.md` after Jennifer's voice pass).
