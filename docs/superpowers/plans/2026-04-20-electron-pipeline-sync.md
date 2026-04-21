# Electron Pipeline Sync + New-Flag Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sync the Electron app's bundled Python pipeline with the canonical root-level v10, hardcode `--burn-timestamps`, expose `--deidentify-names` as an Advanced-section toggle, and surface the `transcript_name_map.json` audit file on the Results screen.

**Architecture:** Root is the source of truth for the pipeline; `scripts/sync-pipeline.sh` copies `video_transcription_pipeline_v10.py`, `deidentify_names.py`, and `pseudonym_pool.json` from root into `src/python/` before build and package. Two new fields are threaded through the four `*Config` interfaces (two renderer, two main/preload), one new IPC handler checks for the audit file, and one toggle is added to ConfigScreen's existing Advanced section.

**Tech Stack:** Electron 38, React 19, TypeScript 5.9, Node 24, Python 3.13 (bundled via venv), bash (sync script), webpack (build).

**Reference spec:** `docs/superpowers/specs/2026-04-20-electron-pipeline-sync-design.md`

---

## Pre-flight — repo state assumptions

- Working on branch `master` (or a feature branch off master).
- Repo root contains current `video_transcription_pipeline_v10.py`, `deidentify_names.py`, `pseudonym_pool.json`.
- `src/python/` contains stale `video_transcription_pipeline_v10.py` (dated ~Mar 20), plus `video_transcription_pipeline_v03.py` and `video_transcription_pipeline_v04.py` (both stale, app only uses v10).
- The Electron codebase has no JS test framework (`package.json` has `"test": "echo ... && exit 1"`). Verification steps in this plan are manual (typecheck + run the app) rather than unit-test-driven. This matches the spec's "No new unit tests proposed for the Electron side."

---

## Task 1: Pipeline sync script

**Files:**
- Create: `scripts/sync-pipeline.sh`

- [ ] **Step 1: Create the sync script**

Write `scripts/sync-pipeline.sh`:

```bash
#!/usr/bin/env bash
# Copy the canonical v10 pipeline and its deidentify support files
# from the repo root into src/python/ so they ship in the Electron bundle.
# Run manually via `npm run sync-pipeline`, or automatically via prebuild/prepackage.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT_DIR"
DEST="$ROOT_DIR/src/python"

FILES=(
  "video_transcription_pipeline_v10.py"
  "deidentify_names.py"
  "pseudonym_pool.json"
)

missing=()
for f in "${FILES[@]}"; do
  if [[ ! -f "$SRC/$f" ]]; then
    missing+=("$f")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "sync-pipeline: missing required files at repo root:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  cp "$SRC/$f" "$DEST/$f"
  echo "sync-pipeline: $f"
done

echo "sync-pipeline: done ($DEST)"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/sync-pipeline.sh`

- [ ] **Step 3: Run it and verify output**

Run: `bash scripts/sync-pipeline.sh`

Expected output (three lines):
```
sync-pipeline: video_transcription_pipeline_v10.py
sync-pipeline: deidentify_names.py
sync-pipeline: pseudonym_pool.json
sync-pipeline: done (/Users/jenniferkleiman/Documents/COMS/src/python)
```

- [ ] **Step 4: Verify file contents match**

Run: `diff -q video_transcription_pipeline_v10.py src/python/video_transcription_pipeline_v10.py && diff -q deidentify_names.py src/python/deidentify_names.py && diff -q pseudonym_pool.json src/python/pseudonym_pool.json && echo ALL MATCH`

Expected: `ALL MATCH` (no diff output before the echo).

- [ ] **Step 5: Test the missing-file failure mode**

Run:
```bash
mv deidentify_names.py deidentify_names.py.bak
bash scripts/sync-pipeline.sh; echo "exit=$?"
mv deidentify_names.py.bak deidentify_names.py
```

Expected:
```
sync-pipeline: missing required files at repo root:
  - deidentify_names.py
exit=1
```

- [ ] **Step 6: Commit**

```bash
git add scripts/sync-pipeline.sh
git commit -m "$(cat <<'EOF'
feat(build): add scripts/sync-pipeline.sh

Copies canonical v10 pipeline + deidentify support files from repo root
into src/python/ so the Electron bundle stays in sync with the research
pipeline. Fails loudly if any source file is missing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire sync into npm scripts

**Files:**
- Modify: `package.json` (scripts block)

- [ ] **Step 1: Read current scripts block**

Read `package.json` and confirm the `scripts` block currently looks like:

```json
"scripts": {
  "build": "webpack",
  "build:python": "bash scripts/build-python.sh",
  "dev": "npm run build && electron .",
  "start": "electron .",
  "package": "npm run build && npm run build:python && electron-builder --config electron-builder.json",
  "test": "echo \"Error: no test specified\" && exit 1"
}
```

- [ ] **Step 2: Add three script entries**

Replace the scripts block with:

```json
"scripts": {
  "sync-pipeline": "bash scripts/sync-pipeline.sh",
  "prebuild": "npm run sync-pipeline",
  "build": "webpack",
  "build:python": "bash scripts/build-python.sh",
  "dev": "npm run build && electron .",
  "start": "electron .",
  "prepackage": "npm run sync-pipeline",
  "package": "npm run build && npm run build:python && electron-builder --config electron-builder.json",
  "test": "echo \"Error: no test specified\" && exit 1"
}
```

npm automatically runs `prebuild` before `build` and `prepackage` before `package` when the scripts exist — no other wiring needed.

- [ ] **Step 3: Verify prebuild fires**

Run: `npm run build 2>&1 | head -20`

Expected: output includes the three `sync-pipeline:` lines from Task 1 BEFORE webpack output.

- [ ] **Step 4: Verify standalone script works**

Run: `npm run sync-pipeline`

Expected: same three `sync-pipeline:` lines.

- [ ] **Step 5: Commit**

```bash
git add package.json
git commit -m "$(cat <<'EOF'
feat(build): wire sync-pipeline into prebuild and prepackage

npm automatically runs prebuild before build and prepackage before package,
so every build and every dmg build now pulls the latest v10 pipeline and
deidentify files from the repo root into src/python/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: One-time stale-file cleanup

**Files:**
- Delete: `src/python/video_transcription_pipeline_v03.py`
- Delete: `src/python/video_transcription_pipeline_v04.py`
- Modify: `CLAUDE.md` (one line)

- [ ] **Step 1: Confirm nothing references v03/v04 in source**

Run: `grep -rn "video_transcription_pipeline_v0[34]" src/ scripts/ package.json electron-builder.json 2>/dev/null || echo "No references found."`

Expected: "No references found." (or output is limited to CLAUDE.md which we handle in step 3).

If any real references turn up in `src/`, `scripts/`, or build configs: STOP and flag to Jennifer. Do not delete until resolved.

- [ ] **Step 2: Delete the stale files**

Run: `git rm src/python/video_transcription_pipeline_v03.py src/python/video_transcription_pipeline_v04.py`

- [ ] **Step 3: Fix CLAUDE.md stale v04 reference**

In `CLAUDE.md`, find the lines in the "Python Integration" section:

```
**Script Location:**
- Development: `src/python/video_transcription_pipeline_v04.py`
- Production: App resources bundle
```

Replace with:

```
**Script Location:**
- Development: `src/python/video_transcription_pipeline_v10.py` (synced from repo root via `npm run sync-pipeline`)
- Production: App resources bundle
```

- [ ] **Step 4: Verify build still succeeds**

Run: `npm run build 2>&1 | tail -10`

Expected: webpack completes without error. Any "module not found" referring to v03/v04 means step 1's grep missed something — revert and investigate.

- [ ] **Step 5: Commit**

```bash
git add -u src/python/ CLAUDE.md
git commit -m "$(cat <<'EOF'
chore(electron): remove stale v03/v04 pipeline copies from src/python/

The Electron app only runs v10. The v03/v04 copies were dead weight that
were still being bundled into the .dmg. Also fix the stale v04 reference
in CLAUDE.md to point at v10 and document the sync flow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire flags into `pythonRunner.ts`

**Files:**
- Modify: `src/main/python/pythonRunner.ts` (`TranscriptionConfig` interface ~line 14, `start()` method ~line 283)

- [ ] **Step 1: Add `deidentifyNames` field to TranscriptionConfig**

In `src/main/python/pythonRunner.ts`, find the `TranscriptionConfig` interface:

```ts
export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  thinkingBudget: number;
  outputPath: string;
  apiKey: string;
  speakersManifestPath?: string;
  audioOnly?: boolean;
}
```

Add one optional field before the closing brace:

```ts
export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  thinkingBudget: number;
  outputPath: string;
  apiKey: string;
  speakersManifestPath?: string;
  audioOnly?: boolean;
  deidentifyNames?: boolean;
}
```

- [ ] **Step 2: Always append `--burn-timestamps` in `start()`**

In the `start()` method of `PythonTranscriptionRunner`, find the existing arg-building block that ends around the `audioOnly` handling:

```ts
    // Audio-only mode
    if (config.audioOnly) {
      args.push('--audio-only');
    }
```

Immediately after that block, add:

```ts
    // Always burn timestamps (ffmpeg clock overlay + per-chunk resume). Closes
    // up-to-14s intra-chunk clock drift; not a user-facing choice.
    args.push('--burn-timestamps');

    // Optional second Gemini pass that replaces real names with pseudonyms
    // and writes transcript_name_map.json. Off by default; toggled from the
    // ConfigScreen Advanced section.
    if (config.deidentifyNames) {
      args.push('--deidentify-names');
    }
```

- [ ] **Step 3: Confirm detectSpeakers is untouched**

Verify that `detectSpeakers()` (higher in the same file) does NOT push `--burn-timestamps` or `--deidentify-names`. Speaker detection is a separate subcommand that doesn't use either flag.

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit -p tsconfig.json 2>&1 | head -20`

Expected: no errors, or only pre-existing errors unrelated to this task. If the config interface now has a field that's not forwarded from `preload.ts` or `App.tsx`, that's fine at this stage — we wire those in Task 5.

- [ ] **Step 5: Commit**

```bash
git add src/main/python/pythonRunner.ts
git commit -m "$(cat <<'EOF'
feat(pipeline): always burn timestamps, optional deidentify-names

--burn-timestamps is hardcoded on (it is a correctness fix for up-to-14s
intra-chunk clock drift, not a user-facing choice). --deidentify-names is
gated by the new optional TranscriptionConfig.deidentifyNames field; UI
wiring lands in a follow-up commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Thread `deidentifyNames` through the renderer/preload types

**Files:**
- Modify: `src/main/preload.ts` (`TranscriptionConfig` interface ~line 190)
- Modify: `src/renderer/App.tsx` (`V10Config` interface ~line 17, startTranscription call ~line 125)
- Modify: `src/renderer/components/ConfigScreen/ConfigScreen.tsx` (`V10Config` interface ~line 22, config builder in `handleStart` ~line 223)

The codebase duplicates the config shape across four files. This plan follows the existing pattern rather than unifying them — unifying is out of scope.

- [ ] **Step 1: Add field to preload.ts TranscriptionConfig**

In `src/main/preload.ts`, find `export interface TranscriptionConfig`. Add `deidentifyNames?: boolean;` as the last field before the closing brace, mirroring the pythonRunner.ts change from Task 4.

Resulting interface:

```ts
export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  thinkingBudget: number;
  outputPath: string;
  apiKey: string;
  speakersManifestPath?: string;
  audioOnly?: boolean;
  deidentifyNames?: boolean;
}
```

- [ ] **Step 2: Add field to App.tsx V10Config**

In `src/renderer/App.tsx`, find `interface V10Config`. Add `deidentifyNames?: boolean;` as the last field before the closing brace:

```ts
interface V10Config {
  videoPath: string;
  prompt: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  thinkingBudget: number;
  audioOnly?: boolean;
  deidentifyNames?: boolean;
}
```

- [ ] **Step 3: Forward the field in the startTranscription call**

In `src/renderer/App.tsx`, find the `window.electronAPI.startTranscription({...})` call (around line 125). It currently ends with `audioOnly: v10Config.audioOnly,`. Add one line after:

```ts
      const result = await window.electronAPI.startTranscription({
        videoPath: v10Config.videoPath,
        prompt: v10Config.prompt,
        model: v10Config.model,
        resolution: v10Config.resolution,
        fps: v10Config.fps,
        chunkMinutes: v10Config.chunkMinutes,
        overlapSeconds: v10Config.overlapSeconds,
        thinkingBudget: v10Config.thinkingBudget,
        outputPath,
        apiKey,
        speakersManifestPath: manifestResult.path,
        audioOnly: v10Config.audioOnly,
        deidentifyNames: v10Config.deidentifyNames,
      });
```

- [ ] **Step 4: Add field to ConfigScreen.tsx V10Config**

In `src/renderer/components/ConfigScreen/ConfigScreen.tsx`, find `interface V10Config` (near line 22). Add `deidentifyNames?: boolean;` as the last field before the closing brace, mirroring step 2.

- [ ] **Step 5: Typecheck**

Run: `npx tsc --noEmit -p tsconfig.json 2>&1 | head -20`

Expected: no new errors. The ConfigScreen's `handleStart` doesn't yet SET `deidentifyNames` on the config object (that lands in Task 6), but since it's optional the types still compile.

- [ ] **Step 6: Commit**

```bash
git add src/main/preload.ts src/renderer/App.tsx src/renderer/components/ConfigScreen/ConfigScreen.tsx
git commit -m "$(cat <<'EOF'
feat(types): thread deidentifyNames through V10Config and TranscriptionConfig

Four interfaces duplicate the config shape; adding the field to all of
them keeps the existing pattern consistent. App.tsx now forwards the new
field through to the main-process IPC call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ConfigScreen toggle UI

**Files:**
- Modify: `src/renderer/components/ConfigScreen/ConfigScreen.tsx` (state ~line 109, Advanced section JSX, `handleStart` config builder ~line 223)

- [ ] **Step 1: Add state for the toggle**

In `src/renderer/components/ConfigScreen/ConfigScreen.tsx`, find the state declarations (near the `audioOnly` state declaration at line 112):

```ts
  const [audioOnly, setAudioOnly] = useState(isAudioFile);
```

Add immediately after:

```ts
  // Off by default — privacy feature with cost implications. Not persisted
  // between sessions.
  const [deidentifyNames, setDeidentifyNames] = useState(false);
```

- [ ] **Step 2: Set the field on the config in handleStart**

In `handleStart` (around line 223), find the `const config: V10Config = { ... }` object. The last property is currently `audioOnly,`. Add `deidentifyNames,` on the next line:

```ts
    const config: V10Config = {
      videoPath: videoInfo.path,
      prompt: selectedPromptId,
      model,
      resolution,
      fps,
      chunkMinutes,
      overlapSeconds,
      thinkingBudget,
      audioOnly,
      deidentifyNames,
    };
```

- [ ] **Step 3: Add `deidentifyNames` to the useCallback dependency array**

In the same `handleStart`, find the useCallback dependency array (around line 236). Add `deidentifyNames` before `onStart`:

```ts
  }, [
    hasApiKey,
    selectedPromptId,
    videoInfo.path,
    model,
    resolution,
    fps,
    chunkMinutes,
    overlapSeconds,
    thinkingBudget,
    audioOnly,
    deidentifyNames,
    onStart
  ]);
```

- [ ] **Step 4: Locate the Advanced section JSX**

Open `src/renderer/components/ConfigScreen/ConfigScreen.tsx` and find the Advanced section. Look for `{showAdvanced && (` around line 372. The block structure is:

```tsx
{showAdvanced && (
  <div className={styles.advanced}>
    <div className={styles.advancedRow}>...</div>
    <div className={styles.advancedRow}>...</div>
    ...
    <div className={styles.advancedRow}>   ← last advancedRow: Thinking Budget (around line 446)
      <label className={styles.advancedLabel}>
        Thinking Budget:
        ...
      </label>
    </div>
  </div>
)}
```

We'll insert the new control as a new sibling AFTER the last `advancedRow` (Thinking Budget) but still inside `<div className={styles.advanced}>`.

- [ ] **Step 5: Add the toggle inside the Advanced block**

The Advanced section's existing children use `styles.advancedRow` + `styles.advancedLabel` (designed for number/select inputs, not descriptive toggles). The ConfigScreen's best-fitting toggle pattern is the `audioOnlyToggle` style used outside the Advanced section (lines ~294–309). We'll reuse those existing classes — they're generic toggle-with-description styling despite their name. Adding new CSS classes is out of scope.

Insert this block immediately after the Thinking Budget `advancedRow`, before the closing `</div>` of `styles.advanced`:

```tsx
              <div className={styles.advancedRow}>
                <label className={styles.audioOnlyToggle}>
                  <input
                    type="checkbox"
                    checked={deidentifyNames}
                    onChange={(e) => setDeidentifyNames(e.target.checked)}
                  />
                  <div className={styles.audioOnlyContent}>
                    <span className={styles.audioOnlyLabel}>De-identify student and adult names</span>
                    <span className={styles.audioOnlyDesc}>
                      Runs a second Gemini pass to replace real names with realistic pseudonyms
                      (e.g., Student-Hannah, Ms. Kelly). Writes an audit file
                      (transcript_name_map.json) next to the transcript — store this file under
                      separate access control. Adds processing time and API cost.
                    </span>
                  </div>
                </label>
              </div>
```

Class reuse rationale: `audioOnlyToggle`/`audioOnlyContent`/`audioOnlyLabel`/`audioOnlyDesc` are already defined in the component's CSS module (see their use at lines 295, 301, 302, 303) and provide the checkbox-with-description layout we want. The outer `advancedRow` wrapper keeps the Advanced section's vertical rhythm consistent.

- [ ] **Step 6: Verify visually**

Run: `npm run dev`

In the app:
1. Upload a short test video to reach the Config screen.
2. Click "Advanced" to expand.
3. Confirm the "De-identify student and adult names" toggle is present.
4. Confirm it defaults to unchecked.
5. Click it; confirm it toggles visually.
6. Close the app.

- [ ] **Step 7: Commit**

```bash
git add src/renderer/components/ConfigScreen/ConfigScreen.tsx
git commit -m "$(cat <<'EOF'
feat(config-ui): add De-identify names toggle to Advanced section

Off by default. Helper text explains the second-pass cost and the audit
file's privacy implications. Not persisted between sessions — a fresh
default every time is the safer posture for a privacy flag with cost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: IPC handler + preload API for audit-file existence

**Files:**
- Modify: `src/main/ipc/transcription.ts` (register a new handler)
- Modify: `src/main/preload.ts` (expose the new API)

- [ ] **Step 1: Add IPC handler to transcription.ts**

In `src/main/ipc/transcription.ts`, inside the `setupTranscriptionHandlers` function, after the existing `transcription:readTranscript` handler (find it by searching for that string), add a new handler:

```ts
  // Check whether transcript_name_map.json exists in the same directory as
  // a given transcript. Narrow by design — we don't want to expose generic
  // fs.existsSync to the renderer, and this is the only file we need to probe.
  ipcMain.handle('transcription:hasAuditFile', async (_event, transcriptPath: string) => {
    try {
      if (!transcriptPath || typeof transcriptPath !== 'string') {
        return { exists: false };
      }
      const auditPath = path.join(path.dirname(transcriptPath), 'transcript_name_map.json');
      return { exists: fs.existsSync(auditPath), path: auditPath };
    } catch (error) {
      console.error('hasAuditFile check failed:', error);
      return { exists: false };
    }
  });
```

`path` and `fs` are already imported at the top of this file (confirmed at lines 4–5 of the file as of 2026-04-20).

- [ ] **Step 2: Expose via preload**

In `src/main/preload.ts`, find the transcription section of `contextBridge.exposeInMainWorld('electronAPI', {...})`. Near `readTranscript:` (line 31), add:

```ts
  hasAuditFile: (transcriptPath: string) =>
    ipcRenderer.invoke('transcription:hasAuditFile', transcriptPath),
```

- [ ] **Step 3: Add to ElectronAPI TypeScript interface**

In the same `preload.ts`, find `interface ElectronAPI`. Near the `readTranscript:` type declaration (around line 226), add:

```ts
  hasAuditFile: (transcriptPath: string) => Promise<{ exists: boolean; path?: string }>;
```

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit -p tsconfig.json 2>&1 | head -20`

Expected: no errors.

- [ ] **Step 5: Smoke-test the handler**

Run: `npm run dev`

In the app:
1. Open DevTools (Cmd+Opt+I on Mac).
2. In the Console, run:
   ```js
   await window.electronAPI.hasAuditFile('/tmp/nonexistent.txt')
   ```
   Expected: `{ exists: false, path: '/tmp/transcript_name_map.json' }`
3. Close the app.

- [ ] **Step 6: Commit**

```bash
git add src/main/ipc/transcription.ts src/main/preload.ts
git commit -m "$(cat <<'EOF'
feat(ipc): narrow handler for checking audit-file presence

transcription:hasAuditFile probes for transcript_name_map.json in the
transcript's directory. Intentionally narrow — does not expose generic
fs access to the renderer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: ResultsScreen audit-file note

**Files:**
- Modify: `src/renderer/components/ResultsScreen/ResultsScreen.tsx`

- [ ] **Step 1: Add state for the audit-file flag**

In `src/renderer/components/ResultsScreen/ResultsScreen.tsx`, near the existing `useState` declarations at the top of the component body (around line 27), add:

```tsx
  const [hasAuditFile, setHasAuditFile] = useState(false);
```

- [ ] **Step 2: Probe for the audit file during result load**

In the existing `loadResults` function (inside the `useEffect` near line 32), after the line that sets the transcript content (`transcript = transcriptData.content;` around line 47) and before the `setResult({...})` call, add:

```tsx
        // Check for the deidentification audit file next to the transcript.
        try {
          const audit = await window.electronAPI.hasAuditFile(job.output_path);
          setHasAuditFile(audit.exists);
        } catch (error) {
          console.error('Failed to check for audit file:', error);
          setHasAuditFile(false);
        }
```

- [ ] **Step 3: Render the note when the audit file exists**

In the return block (around line 112), the existing structure is:

```tsx
<div className={styles.content}>
  <div className={styles.header}>...</div>
  <div className={styles.statsSection}>...</div>     ← ends around line 147
  <div className={styles.transcriptSection}>...</div>
  <div className={styles.outputSection}>...</div>
  ...
</div>
```

Insert the audit note BETWEEN the `statsSection` closing `</div>` and the `transcriptSection` opening `<div>`. The privacy note should be visible prominently near the stats, not buried below the transcript.

Use the existing `styles.outputSection` class as the container — it provides a labeled-panel layout that matches this note's shape (see lines 162–165 for how the "Saved to:" output path uses the same class). Reusing it avoids adding CSS. Inline overrides are fine if needed to distinguish visually (e.g., a subtle warning color), but adding new CSS module classes is out of scope.

```tsx
{hasAuditFile && (
  <div className={styles.outputSection}>
    <div className={styles.outputLabel}>Name audit file:</div>
    <div>
      <code>transcript_name_map.json</code> was saved alongside this transcript.
      It contains the real-name↔pseudonym mapping and should be stored under
      separate access control from the transcript itself.
    </div>
  </div>
)}
```

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit -p tsconfig.json 2>&1 | head -20`

Expected: no errors.

- [ ] **Step 5: Verify with a fake audit file**

Run: `npm run dev`

In the app:
1. Navigate to any prior completed transcription's Results screen.
2. Confirm the note does NOT appear (no audit file present for old jobs).
3. In a separate terminal, copy or create a file at `<transcript_dir>/transcript_name_map.json` (pick any completed job's output directory; the file can contain `{}`).
4. Re-open the Results screen for that job (e.g., by navigating away and back).
5. Confirm the note NOW appears.
6. Delete the test file.
7. Close the app.

- [ ] **Step 6: Commit**

```bash
git add src/renderer/components/ResultsScreen/ResultsScreen.tsx
git commit -m "$(cat <<'EOF'
feat(results-ui): surface transcript_name_map.json when present

When deidentification was used, the Results screen now shows a short
note reminding the user the audit file exists and should be stored
under separate access control. Existence check; no new DB column.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: End-to-end verification

**Files:** none (verification only)

This task does no coding. It confirms the full flow works before we call the feature done.

- [ ] **Step 1: Build and launch dev**

Run: `npm run dev`

Expected: build logs include the three `sync-pipeline:` lines, then webpack, then Electron launches.

- [ ] **Step 2: Short transcription without deidentification**

In the app:
1. Upload the test video: `250415_4Math_Daisy_Day1_SG2_01050109_v04_transcription_20250719_090912/chunks/250415_4Math_Daisy_Day1_SG2_01050109_chunk_01.mp4` (a 4-minute clip — from project MEMORY.md).
2. On Config, leave defaults, do NOT enable "De-identify student and adult names."
3. Start transcription.
4. Wait for completion (~3 min given the clip length).
5. Confirm: transcript file exists at the output path, Results screen shows stats, Results screen does NOT show the audit-file note.
6. Check the transcript folder: should NOT contain `transcript_name_map.json`.

- [ ] **Step 3: Short transcription WITH deidentification**

In the app:
1. New transcription with the same video.
2. On Config, open Advanced, enable "De-identify student and adult names."
3. Start transcription.
4. Wait for completion (slightly longer than step 2 due to the second pass).
5. Confirm: transcript has pseudonymized names (spot-check by opening it), Results screen DOES show the audit-file note.
6. Check the transcript folder: `transcript_name_map.json` is present, file permissions are `-rw-------` (0600). Verify with `ls -l <transcript_dir>/transcript_name_map.json`.

- [ ] **Step 4: Confirm packaging sync**

Run: `npm run package 2>&1 | grep -E "sync-pipeline|electron-builder" | head -20`

Expected: `sync-pipeline:` lines appear before any `electron-builder` output.

After packaging completes, verify the bundled pipeline was updated. The DMG's `.app/Contents/Resources/python/scripts/video_transcription_pipeline_v10.py` (or equivalent path per `electron-builder.json`) should have the same sha256 as the repo-root file:

```bash
shasum -a 256 video_transcription_pipeline_v10.py
# compare to bundled copy (path depends on electron-builder config)
```

- [ ] **Step 5: No commit**

This task produces no code changes. Do not commit.

If any step above failed, STOP and report the failure to Jennifer with the exact command output and the step number. Do not attempt speculative fixes — the earlier tasks have clear boundaries and most failures point to a specific upstream task.

---

## Spec coverage check

| Spec section | Implemented in task |
|---|---|
| §1 Pipeline sync: source of truth at root | Task 1 |
| §1 Pipeline sync: prebuild/prepackage wiring | Task 2 |
| §1 Pipeline sync: stale cleanup + CLAUDE.md fix | Task 3 |
| §2 `--burn-timestamps` always on | Task 4 step 2 |
| §2 `--deidentify-names` optional | Tasks 4 step 2, 5, 6 |
| §2 `--keep-chunks` not exposed | (deliberately nothing to implement) |
| §2 detectSpeakers unchanged | Task 4 step 3 (verification) |
| §3 Advanced-section toggle | Task 6 |
| §4 IPC pass-through | Task 5 step 3 (via App.tsx forwarding) |
| §5 Results-screen audit note | Tasks 7, 8 |
| §5 No "Open audit file" button | (deliberately nothing to implement) |
