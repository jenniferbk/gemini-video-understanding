# COMS Directory Reorganization — Design

**Date:** 2026-04-20
**Author:** Jennifer + Claude (brainstorming session)
**Status:** Approved verbally; awaiting written-spec review before writing implementation plan

## Problem

`/Users/jenniferkleiman/Documents/COMS/` has grown to 317 top-level entries and ~34 GB. Root causes:

- **91 transcription run folders** (`*_transcription_YYYYMMDD_*/`) from iterative pipeline development (v04 → v08 → v10). Many are duplicate runs of the same source video.
- **~100 loose files at top level:** pre-v10 status docs (`PHASE1/2/3_COMPLETE.md`, `V04_*`), old pipeline versions, fix/patch scripts, build logs, stale requirements files, chunk-transcript fragments.
- **Mixed concerns:** raw media (COMS + TIMSS + GENIUS-project interviews), sensitive data (IRB forms, student audio), active Electron app code, and active paper work — all commingled at top level.
- **Chunks not cleaned up:** pipeline leaves chunk videos after successful runs, consuming space and (when paired with `--deidentify-names`) leaking PII to disk.

## Goals

All four of:

- **A. Visual clutter** — cut top-level entries from ~317 to a manageable number
- **B. Disk space** — reclaim ~20-25 GB
- **C. PII containment** — sensitive content (IRB, name-maps) into a locked-down folder; GENIUS-project content staged for relocation; raw COMS media deleted (backed up in Teams)
- **D. Git hygiene** — no broken moves; git history preserved via `git mv`; leave Jennifer's in-progress paper work untouched

## Non-goals

- **No restructure of the Electron app.** `src/`, `package.json`, `electron-builder.json`, `resources/`, `database/`, `scripts/`, `test_fixtures/` stay where they are — build paths depend on them.
- **No touching `dev/active/paper/`.** Active paper work (`paper.tex`, benchmark scripts, argumentation diagrams, modified `NOW.md`). Stays as-is.
- **No deletion of TIMSS videos.** Benchmarking work is ongoing.
- **No deletion of transcripts unless a newer complete run exists for the same source video.** Jennifer has not yet uploaded all transcripts to Teams.

## Constraints

- **Teams storage as backup:** All COMS raw media (`.mp4`, `.mov`, `.m4a`, `.MP3`) is in the lab's Teams group. Safe to delete locally.
- **IRB folder is active:** amendment in progress, must remain accessible (moved to `sensitive/`, not deleted).
- **Clarke Middle School interviews:** belong to a different project (GENIUS), not COMS. Keep but stage for relocation.
- **TIMSS videos:** identified by matching against benchmark-run gold filenames (`US*_gold.txt`, `AU*`, `CZ*`, `JP*` in `dev/active/paper/benchmark_runs/`).
- **Destructive actions gated:** every deletion batch requires explicit approval with the file list shown.

## Target directory structure

```
COMS/
├── src/                     # [UNCHANGED] Electron renderer/main
├── scripts/                 # [UNCHANGED]
├── resources/               # [UNCHANGED]
├── database/                # [UNCHANGED]
├── test_fixtures/           # [UNCHANGED]
├── dev/                     # [UNCHANGED] active paper work stays put
├── analysis/                # [UNCHANGED] active
├── benchmark_runs/          # [UNCHANGED] active (top-level, distinct from dev/active/paper/benchmark_runs/)
├── node_modules/, venv/     # [UNCHANGED]
├── release/, dist/, binaries/  # [UNCHANGED] build outputs
│
├── transcription_runs/      # [NEW] keeper runs only (newest v10 per video)
├── archive/                 # [NEW]
│   ├── docs/                #   stale pre-v10 .md files
│   ├── scripts/             #   old pipeline versions, fix/patch scripts
│   └── logs/                #   *.log, build-log*.txt
├── sensitive/               # [NEW] chmod 700, gitignored
│   ├── IRB/                 #   moved from top level
│   └── name_maps/           #   any transcript_name_map.json extracted
├── timss_videos/            # [NEW] TIMSS benchmarking videos grouped
├── _to_relocate/            # [NEW] staging for Clarke/GENIUS content
├── misc/                    # [NEW] low-priority keep-for-now items
│
├── video_transcription_pipeline_v10.py  # [STAY]
├── deidentify_names.py                  # [STAY]
├── quality_checker.py                   # [STAY]
├── clean_transcript_for_transana.py     # [STAY]
├── test_deidentify_names.py             # [STAY]
├── prompts.json, pseudonym_pool.json, Smallgroup_ben.json  # [STAY]
├── Math+transcripts+as+txt+files+(1)/   # [STAY]
├── Science+transcripts+as+txt+files+(1).zip  # [STAY]
├── package.json, package-lock.json, tsconfig.json, electron-builder.json  # [STAY]
├── requirements.txt, requirements_v10.txt  # [STAY]
└── CLAUDE.md, README.md, PROJECT_KNOWLEDGE.md, TROUBLESHOOTING.md, INSTALLATION.md  # [STAY]
```

## Categorization rules

### → `archive/docs/` (historical, pre-v10)
- `PHASE1_COMPLETE.md`, `PHASE2_COMPLETE.md`, `PHASE3_PROGRESS.md`
- `V04_IMPLEMENTATION_SUMMARY.md`, `V04_QUICK_FIX_INSTRUCTIONS.md`, `V03_to_V04_Migration_Guide.md`
- `FIX_SUMMARY.md`, `PROMPT_VALIDATION_SUMMARY.md`, `PYTHON_SCRIPT_FIXES.md`
- `BETA_TESTING_GUIDE.md`, `BUNDLING_GUIDE.md`, `TESTING_CHECKLIST.md`, `TODO_MVP.md`
- `QUICK_FIX_FOR_USERS.md`, `README_V04.md`

### → `archive/scripts/` (superseded / one-off)
- `batch_process_v04.py`, `compare_v03_v04.py`, `consensus_analysis_script.js`, `consensus_tester.py`
- `fix_float_issue.py`, `fix_thinking_parameter.py`, `fix_v04_setup.sh`, `patch_v04.py`
- `reprocess_chunks.py`, `v04_config_generator.py`, `v04_migration_tool.py`
- `test_imports.py`, `test_rtf_output.py`, `test_v04_installation.py`
- `setup_v04.sh`, `run_ava_batch.sh`, `test-fresh-install.sh`, `quick-test-uuid-fix.sh`, `test-prompt-uuid-fix.sh`
- `requirements_v04.txt`, `requirements_v04_fixed.txt`, `requirements_v04_simple.txt`

### → `archive/logs/`
- `build-log.txt`, `build-log-v1.1.4.txt`, `build-log-v1.1.5.txt`, `build-log-v1.1.6.txt`
- `ben_day2_log.txt`, `ben_day3_log.txt`, `debug_response_object.log`, `transcription_debug.log`
- `fixed_chunking_test.log`, `fixed_test.log`, `ultra_fixed_test.log`, `mystery_solved.log`

### → `sensitive/`
- `IRB/` → `sensitive/IRB/`
- Any `transcript_name_map.json` found in keeper runs → `sensitive/name_maps/`
- `chmod 700` the folder; add to `.gitignore`

### → `_to_relocate/` (GENIUS project, not COMS)
- `Clarke Middle School ArguAgent (1st period) Student Interview_03232026 phone.m4a`
- `Clarke Middle School ModelAgent (1st period) Student Interview_03232026 phone.m4a`
- `Clarke Middle School ModelAgent (1st period) Student Interview_03232026.MP3`
- `Clarke Middle School ArguAgent (1st period) Student Interview_03232026 phone_v10_20260323_143122/`
- `Clarke Middle School ModelAgent (1st period) Student Interview_03232026 phone_v10_20260323_143649/`
- `Clarke Middle School ModelAgent (1st period) Student Interview_03232026_v10_20260323_144315/`
- `Clarke_MS_Argumentation_HaydenLane_Mar19.mp4` + `Clarke_MS_Argumentation_HaydenLane_Mar19_v10_20260319_180217/`
- `chmod 700`

### → `misc/`
- `transcripts_annaB.xlsx`, `transcripts_combined.xlsx`
- `3Math_Ben_Day4_SG2_Arg2.drawing`
- `Y2_whole_class_episodes.html`, `whole_class_episodes.html`
- `v04_processing_summary.json`

### → `timss_videos/`
Identified during inventory by matching against `dev/active/paper/benchmark_runs/*_gold.txt` references.

### → `transcription_runs/` (keepers only)
For each unique source video (base name before `_transcription_` / `_v04_transcription_` / `_v08_transcription_` / `_v09_transcription_` / `_v10_`):
1. Prefer newest `v10` run
2. Fallback: newest run overall
3. Skip empty or obviously-failed folders (no transcript output)
Keeper goes to `transcription_runs/`; non-keepers marked for deletion.

### → Delete (approval-gated batches)
- **Batch A:** Non-keeper transcription folders (~70 folders)
- **Batch B:** All `chunks/` subdirectories inside keeper runs + top-level `video_chunks/` (287 MB)
- **Batch C:** Top-level raw COMS videos — NOT TIMSS, NOT Clarke: `Y2_4S_Ava_Day*.mp4`, `250415_4Math_Ava_Day1.mp4`, `250416_4Math_Ava_Day2.mp4`
- **Batch D:** Loose top-level `chunk_01_v04_transcript.txt` through `chunk_XX_v04_transcript.txt`

### → Inventory-decide (do NOT pre-commit to deletion)
Top-level loose complete-transcript `.txt` files — preserve; determine during inventory whether these are duplicates of a keeper run's transcript (then move to keeper-run folder or delete duplicate) or unique copies (then move to `transcription_runs/` as standalone files):
- `240920_4Science_Ava_Post_0509_v04_complete_transcript.txt`
- `250124_3Science_Faith_Day4_0408_v04_complete_transcript.txt`

Expected reclaim: **~20-25 GB**

## Pipeline code change

Edit `video_transcription_pipeline_v10.py`:
- On successful completion of a run, delete the run's `chunks/` subdirectory unless `--keep-chunks` explicitly set
- Do NOT delete on failure (debugging needs the chunks)
- Verify current `--keep-chunks` flag behavior; the change may be small (flipping default, or ensuring cleanup reliably runs)

Motivation: project memory flags that `--keep-chunks` combined with `--deidentify-names` leaks PII to disk via un-redacted chunk transcripts.

## Execution sequence

1. **Inventory (read-only)** — generate `REORG_INVENTORY.md` with keeper selection, TIMSS identification, name-map locations. Jennifer reviews.
2. **Create empty dirs + gitignore updates** — no file moves yet.
3. **Moves** — gitignored items first (majority), then `git mv` for tracked files. `git status` checkpoint.
4. **Deletions (approval-gated per batch A/B/C/D)** — file list shown before each batch.
5. **Pipeline code change** — diff reviewed before commit.
6. **Commits** — three commits: structure/gitignore, moves, pipeline change.

## Safety rails

- Every destructive step requires explicit per-batch approval with file list
- All moves logged to `REORG_INVENTORY.md` for reversibility
- Git-tracked content committed before any `rm`, so `git restore` is always an option
- `dev/active/paper/*` and all currently-modified files left completely untouched
- Electron build paths (`src/`, `resources/`, `scripts/`, `database/`, `test_fixtures/`) unchanged

## Out of scope

- Audit of `dev/` folder internals (she actively uses it)
- Any restructure of `src/` or Electron-app internals
- Reorganizing transcription folder contents (only moving/deleting whole folders)
- GENIUS-project organization (content goes to `_to_relocate/` for her to handle)
- Uploading transcripts to Teams (separate process, Jennifer's task)

## Open questions / items deferred to inventory phase

- Final list of TIMSS videos (determined by gold-file cross-reference)
- Exact count of keeper vs non-keeper runs (determined by grouping by base-video name)
- Whether any `transcript_name_map.json` files exist in keeper runs (determined by scan)
