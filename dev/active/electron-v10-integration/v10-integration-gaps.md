# Electron ↔ v10 Integration Gaps

Two pre-existing bugs in `src/main/python/pythonRunner.ts` that were dormant while the app bundled the March copy of v10. As of 2026-04-21 the app syncs to canonical root v10 on every build, which makes both bugs live.

Both were caught by the final code review of `feature/electron-pipeline-sync` (session 2026-04-21). Neither is a regression from that branch — they were on master before it. The branch just exposes them.

---

## Gap 1 — Speaker detection is broken against v10 (RESOLVED 2026-04-21)

**Resolution:** Option A — `identify --json-speakers` headless mode added to v10. `pythonRunner.detectSpeakers()` now calls `identify --json-speakers` (dropping the stale `--chunk-minutes`/`--overlap`/`--audio-only` args). The GVU_SPEAKERS stdout contract is covered by `tests/test_speakers_json_emission.py`. `emit_speakers_json_line` helper at v10:2083.

Additional finding during the verification pass: `p_identify` doesn't attach `add_chunk_args`, so `--chunk-minutes` and `--overlap` were never going to work on this subcommand even after the rename. Identification uses the internal `speaker_id_chunks=2` default, independent of the transcription chunk grid. Runner args simplified accordingly.

### Original analysis (pre-resolution)

### Symptom

Clicking "Detect Speakers & Start" in the app spawns v10 with subcommand `detect-speakers`. v10 rejects the subcommand at argparse and exits before any work happens.

### Call site

`src/main/python/pythonRunner.ts:205` (inside `detectSpeakers()`):

```ts
const args = [
  this.scriptPath,
  'detect-speakers',   // ← subcommand does not exist in v10
  config.videoPath,
  '--api-key', config.apiKey,
  ...
];
```

### Root cause

`video_transcription_pipeline_v10.py` exposes four subcommands: `identify`, `process`, `batch`, `estimate`. There is no `detect-speakers`. The closest match is `identify` (`cmd_identify` at line 2083), which calls `registry.interactive_edit(speakers, path.name)` — an interactive stdin-based editor. Electron spawns the process with `stdio: ['ignore', 'pipe', 'pipe']`, so stdin is closed and `interactive_edit` will block or EOF.

Even if the subcommand string is changed from `detect-speakers` to `identify`, the pipeline still calls the interactive editor. The runner's `handleStdout` expects a `GVU_SPEAKERS:` JSON line, which `cmd_identify` does not emit.

### Fix directions (pick one)

**A. Add a headless mode to v10's `identify` subcommand.** New flag `--json-speakers` (or equivalent) that:
- Skips `registry.interactive_edit(...)`.
- Emits the speaker list as `GVU_SPEAKERS: {...}` on stdout.
- Also writes the `*_speakers.json` manifest to disk so the rest of the pipeline's existing flow still works.

Then update `pythonRunner.detectSpeakers()` to spawn `identify --json-speakers ...` and drop the old arg shape that relied on `detect-speakers`-specific flags (`--chunk-minutes`, `--overlap`).

Recommended. This is the cleanest long-term path because it lets the GUI reuse the same speaker-identification logic researchers use at the CLI.

**B. Skip speaker detection in the app; call `process` directly.** The `process` subcommand accepts `--speakers <manifest>` but also works without one. Remove the detection phase from the app's flow and have users either run `identify` manually at CLI first, or proceed without a speaker manifest (v10 will generate description-based labels on the fly). This reduces app capability but requires no pipeline changes.

**C. Keep a pinned older pipeline copy in `src/python/` only for identification.** Hacky — means the app runs two different pipeline versions for Phase 1 vs Phase 2. Not recommended.

### Open questions

- Does `cmd_identify`'s current speaker-registry output format match what the renderer's `SpeakerReview` component expects? Worth checking `SpeakerReview.tsx` against `SpeakerRegistry.save_manifest` before settling on a JSON shape for the new headless mode.
- What should the headless mode do when the registry can't resolve ambiguous speakers that `interactive_edit` would normally prompt about? Default: accept the registry's best guesses and emit with a confidence field so the UI can surface low-confidence matches.

---

## Gap 2 — `--prompts-file` flag doesn't exist in v10 (IMPORTANT)

**Interim state (2026-04-21):** the `--prompts-file` push at `pythonRunner.ts:322-324` was removed as a Task 9 pre-flight so the runner never sends a flag v10 rejects. The ConfigScreen custom-prompt UI, `convertAndWritePrompts()`, and `tempPromptsFile` state are all still in place — the UI silently no-ops for custom prompts. Users who pick a bundled-key prompt are unaffected. Final fix still pending the A/B/C decision below.

### Symptom

When a user has saved custom prompts in the app, `pythonRunner.start()` writes them to a temp JSON file and pushes `--prompts-file <tempfile>`. v10 rejects the flag at argparse and the process exits.

### Call site

`src/main/python/pythonRunner.ts:326`:

```ts
if (this.tempPromptsFile) {
  args.push('--prompts-file', this.tempPromptsFile);
}
```

Plus the `convertAndWritePrompts()` method (lines ~139–176) that generates the tempfile.

### Root cause

v10 exposes only `-p`/`--prompt <key>` — a lookup key into whatever prompt dictionary the pipeline has baked in. There is no `--prompts-file` flag. Grep confirms: zero matches in `video_transcription_pipeline_v10.py`.

The custom-prompt flow assumed a pipeline feature that doesn't exist (or existed in the old March bundle and was removed).

### Fix directions (pick one)

**A. Add `--prompts-file` to v10.** Accepts a JSON file of `{key: {name, description, prompt}}` entries that overlay or replace v10's built-in prompt dictionary. The app already writes exactly that shape in `convertAndWritePrompts` (see `pythonRunner.ts:155–165`). Minimal v10 change.

Recommended if the user-facing "custom prompts" feature is to be preserved.

**B. Remove the `--prompts-file` push.** The app currently lets users save custom prompts with a UUID. If v10 has no way to consume them, the feature is a lie. Either:
- Remove the ConfigScreen's custom-prompt UI entirely, OR
- Keep the UI but document that custom prompts only work if their `id` matches a baked-in key.

**C. Inline the prompt text instead.** Push `--prompt-text "<the actual prompt string>"` instead of a key lookup. Requires a new v10 flag but is simpler than JSON-file parsing. Marginally worse for maintaining the bundled prompt library.

### Open question

- How often do the 8 colleagues actually author custom prompts vs. pick from the bundled library? If the answer is "rarely," option B might be the right call. If "regularly," option A is needed.

---

## Relationship to the merged branch

The `feature/electron-pipeline-sync` branch is correct and complete for its defined scope (`docs/superpowers/specs/2026-04-20-electron-pipeline-sync-design.md`). Speaker detection and custom prompts were explicitly not in scope. These gaps are tracked here so they don't fall through the cracks, not because the branch failed to handle them.
