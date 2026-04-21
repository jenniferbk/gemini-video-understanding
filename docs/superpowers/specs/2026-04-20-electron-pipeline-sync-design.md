# Electron App — Pipeline Sync and New-Flag Integration

**Date:** 2026-04-20
**Status:** Design approved, ready for implementation plan
**Scope:** Sync the Electron app's bundled Python pipeline with the canonical root-level v10 pipeline, and expose the two new user-facing flags (`--burn-timestamps`, `--deidentify-names`).

## Background

The Electron desktop app (`GeminiVideoUnderstanding`) ships a bundled copy of the v10 transcription pipeline at `src/python/video_transcription_pipeline_v10.py`. That copy is dated March 20 and has fallen behind the canonical v10 at the repo root, which now includes three material changes:

1. `a9894ed` — `--burn-timestamps` (ffmpeg clock overlay + per-chunk resume)
2. `c8fe460` — `--deidentify-names` (second Gemini pass, writes `transcript_name_map.json`)
3. `b6b75ba` — cleanup also removes per-chunk `.txt` files (closes a PII-on-disk gap)

The app's `pythonRunner.ts` exposes no `burnTimestamps`, `deidentifyNames`, or `keepChunks` fields on `TranscriptionConfig`. The supporting Python module `deidentify_names.py` and its data file `pseudonym_pool.json` live only at the repo root — they are not in `src/python/` at all, so even if the flag were wired, the app would crash on `--deidentify-names`.

## Goals

- The Electron app runs the same v10 behavior researchers use at the CLI.
- Deidentification is available to the app's 8 non-technical users through a single Config-screen toggle.
- Timestamp burning is silently on by default — a correctness fix, not a user-facing choice.
- The pipeline has **one** source of truth (repo root); the `src/python/` copy is generated.

## Non-goals

- No new CLI flags in the Python pipeline. All pipeline work is already committed.
- No changes to speaker-detection behavior (`detectSpeakers`).
- No UI for `--keep-chunks`. It remains CLI-only, available to the researcher but not surfaced in the GUI. Rationale: combined with `--deidentify-names` it leaves PII in per-chunk `.txt` files, which is a failure mode the GUI should not expose.
- No persistence of the deidentify toggle between sessions. Off by default every time, per the cost/privacy profile.
- No preset-level wiring of deidentification. Presets stay focused on model/resolution/fps/thinking-budget.

## Design

### 1. Pipeline sync strategy

**Source of truth:** repo root. Research and benchmarking work continues there; the Electron app never reads directly from root.

**Sync mechanism:** `scripts/sync-pipeline.sh` (a few `cp` lines) wired into `package.json`:

- `prebuild` — runs before `npm run build`
- `prepackage` — runs before `npm run package`
- `sync-pipeline` — standalone, for manual use during dev

**Files synced from repo root → `src/python/`:**

- `video_transcription_pipeline_v10.py`
- `deidentify_names.py`
- `pseudonym_pool.json`

**Stale cleanup (one-time, as part of this work):**

- Delete `src/python/video_transcription_pipeline_v03.py`
- Delete `src/python/video_transcription_pipeline_v04.py`
- Update the stale reference to `v04.py` in `CLAUDE.md` → `v10.py`

**What this does not change:**

- `src/python/requirements.txt` stays minimal (`google-genai>=1.0.0`). `deidentify_names.py` is pure stdlib, so no new Python dependencies.
- No electron-builder config changes — `src/python/` is already bundled into resources.

### 2. `pythonRunner.ts` — flag wiring

Add one field to `TranscriptionConfig`:

```ts
export interface TranscriptionConfig {
  // ...existing fields...
  deidentifyNames?: boolean;
}
```

In `start()`, after the existing arg-building block:

```ts
args.push('--burn-timestamps');
if (config.deidentifyNames) {
  args.push('--deidentify-names');
}
```

`--burn-timestamps` is hardcoded, not configurable through the app. It is a correctness fix (closes up-to-14s intra-chunk clock drift) and the research case for disabling it is not relevant to the app's users.

`detectSpeakers()` is unchanged — the speaker-detection pass does not use either flag.

### 3. `ConfigScreen.tsx` — UI addition

One new control in the existing **Advanced** section (gated by `showAdvanced`):

- **Label:** "De-identify student and adult names"
- **Control:** toggle, default `false`
- **Helper text:** "Runs a second Gemini pass to replace real names with realistic pseudonyms (e.g., Student-Hannah, Ms. Kelly). Writes an audit file (`transcript_name_map.json`) next to the transcript — store this file under separate access control. Adds processing time and API cost."
- State: `const [deidentifyNames, setDeidentifyNames] = useState(false);`
- Passed through to `TranscriptionConfig` in the existing start-transcription handler.

### 4. IPC layer (`src/main/ipc/transcription.ts`)

The `transcription:start` handler (line 99) receives a full `TranscriptionConfig` object and forwards it to `pythonRunner.start()`. Adding `deidentifyNames` to the `TranscriptionConfig` interface is sufficient — no handler code changes needed.

### 5. `ResultsScreen.tsx` — audit-file note

The pipeline writes the audit file as `transcript_name_map.json` in the same directory as the transcript, with mode `0600` (owner-only read/write). On Results-screen load, check whether `transcript_name_map.json` exists in `path.dirname(job.output_path)`. If it does, render a small note under the stats block:

> **Name audit file:** `transcript_name_map.json` was saved alongside this transcript. It contains the real-name↔pseudonym mapping and should be stored under separate access control from the transcript itself.

No new database column. The existence check is sufficient and stays correct for jobs run before this feature ships. No "Open audit file" button — clicking would render PII inline, which is exactly what the user may not want on a shared screen. "Open Folder" (already present) is the right affordance.

Caveat: multiple transcripts written to the same directory will share one `transcript_name_map.json` (the most recent run overwrites earlier audit files). This is pipeline behavior, not an app concern — the app faithfully surfaces whatever the pipeline produced. Flag for follow-up if it becomes a real problem in practice.

## Data flow summary

```
ConfigScreen (deidentifyNames toggle)
  → TranscriptionConfig (deidentifyNames: boolean)
  → IPC transcription.start
  → pythonRunner.start()
      → args.push('--burn-timestamps') [always]
      → args.push('--deidentify-names') [if config.deidentifyNames]
  → spawn Python (bundled v10 in src/python/)
      → pipeline writes transcript + optionally transcript_name_map.json
  → ResultsScreen (detects _name_map.json, shows privacy note)
```

## Error handling and edge cases

- **Sync script missing files at root** — `sync-pipeline.sh` exits non-zero; `prebuild`/`prepackage` fail loudly. Better than silently shipping a stale bundle.
- **Deidentification-pass failure mid-run** — already handled in-pipeline by `74e5ac1` (robust failure handling). App surfaces the pipeline's error via the existing `GVU_ERROR:` path.
- **User toggles deidentification but API key has insufficient quota for the second pass** — surfaced by pipeline as a normal error. No app-specific handling needed.
- **Audit file detection misses** — if a transcript was renamed after the pipeline wrote the map file, the Results-screen note won't appear. Acceptable; "Open Folder" still shows the file.

## Testing

- Manual: dev build with `npm run dev`, run a short video with and without the deidentify toggle, confirm `transcript_name_map.json` appears (when toggled on) and the Results note renders.
- Manual: `npm run package` produces a .dmg whose bundled `src/python/` contains all three synced files.
- No new unit tests proposed for the Electron side (consistent with the existing codebase's testing profile). Python-side tests for deidentification already exist (`test_deidentify_names.py`, 38 passing).

## Migration / rollout

- None required for users. The first app build after this work ships the new pipeline automatically; deidentification is opt-in via the Advanced toggle.

## Open questions

None remaining after brainstorming. All design questions resolved 2026-04-20.
