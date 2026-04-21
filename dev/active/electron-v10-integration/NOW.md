# Electron ↔ v10 Integration — NOW

**Last updated:** 2026-04-21

## 2026-04-21 (evening) — Task 9 GREEN

Manual end-to-end on a 4-min classroom clip (`AvaDay4SG_5v10_chunk_01.mp4`). Both passes completed successfully:

- **Run #1 (deidentify OFF):** speaker detection → review → 4/4 chunks transcribed → transcript on disk ✓. UI hit "Transcription results not found" — tracked down to two stacked bugs (below), fixed, verified on Run #2.
- **Run #2 (deidentify ON):** speaker detection → review → 4/4 chunks in 213s → ResultsScreen rendered correctly with transcript + privacy note. `transcript_name_map.json` audit file written ("Audrey" → "Student-Hannah"; visual label `Girl-BlackHoodieBandage` retired). Pseudonymization end-to-end works.

### Bugs fixed mid-session

1. **Broken dev venv.** `pythonRunner.getPythonPath()` pointed at `<project>/venv/bin/python3`, which was built with `--copies` against Python 3.13.7 (now 3.13.13 after Homebrew upgrade). Python crashed with SIGABRT on dyld framework load. Repointed to `src/python/venv/bin/python3` (symlink-based, per CLAUDE.md dev convention).
2. **`output_file` snake/camel + directory-vs-file mismatch.** Python emitted `output_file=str(output_dir)` (directory, snake_case); `transcription.ts:157` read `completion.outputFile` (camelCase). DB stored empty `output_path`, ResultsScreen hit the "not found" branch. Fixed v10 to send the actual transcript file path (`main_transcript_path = output_dir / f"{stem}_transcript.txt"`); fixed TS to accept `output_file || outputFile`.

### Task 9 pre-flight (done earlier 2026-04-21)

Removed the `--prompts-file` and `--audio-only` pushes from `pythonRunner.start()` (both flags are grep-zero in v10). UI state and tempfile generation for custom prompts are still in place — just not passed through. See `v10-integration-gaps.md` Gap #2 interim note.

### Silent-failure UX wart (unfixed)

`detectSpeakers()` exit handler at `pythonRunner.ts:256` uses `code !== 0 && code !== null` — so signal-killed processes (SIGABRT, SIGKILL) are treated as success and no error event fires. UI sits on "Preparing speaker detection…" forever. Didn't bite this session because the fix above stopped the SIGABRT, but the handler is still wrong. Small fix for a future session.

## Next session — chunk-1 timestamp offset

Both Task 9 runs reproduced a 5–6s timestamp offset in chunk 1 only (chunks 2+ are correct). Working hypothesis: the source video has an audio-only preamble where the video stream starts several seconds after audio. ffmpeg's `-ss 0 -i <file>` + drawtext `%{pts\:hms\:0}` burns the clock starting from the first visible video frame (actual time ~5s), so Gemini reads 00:00 where real time is 00:05 — systematic 5s-early offset for all chunk-1 utterances. Chunks 2+ start mid-video where streams are in sync, no offset.

Secondary symptom: the first ~4–5s of audio is partially dropped from the transcript (varies run-to-run; Gemini's handling of the black-screen preamble is nondeterministic, but the clock-offset is consistent).

Reproduction: Run #2 shows `00:04 Teacher-PatternedTop: Okay. So, can you run it for me so I can see what you got?` — Jennifer confirmed the real time of that utterance is ~00:10.

Proposed approach for next session:
1. Pre-probe source video with `ffprobe -show_streams` to detect audio/video `start_time` mismatch.
2. Compensate: either pad video at the front with a black frame aligned to audio start, OR adjust the drawtext offset to reflect the true audio-anchored wall clock, OR `-itsoffset` the audio/video to force sync before chunking.
3. Spot-check: re-run AU4 (existing benchmark outlier) with the fix to see if it changes the numbers.

## Backlog

- **Real test suite.** The single `tests/test_speakers_json_emission.py` file uses stdlib `unittest` only because the repo had no test infra at all. Proper future setup: pytest + a renderer-side jest/vitest, fixtures for short video clips, mocked GeminiClient for pipeline-level integration tests. The narrow-scope unittest is a stopgap, not a pattern to scale.
- **Custom-prompt UI is a no-op as of the Task 9 pre-flight.** Either remove the ConfigScreen custom-prompt UI + `convertAndWritePrompts()` entirely, or add `--prompts-file` to v10 so the UI works. Decision still pending (Gap #2 A/B/C).
- **`--audio-only` UI is a no-op as of the Task 9 pre-flight.** Users who drop an audio file into the app will see the toggle auto-on but v10 won't receive the flag. Either add `--audio-only` to v10 or remove the toggle. Follows the Gap #2 decision.
- **Silent-failure in `detectSpeakers()` exit handler** (see above). One-line fix: change `code !== 0 && code !== null` to `(code !== null && code !== 0) || signal !== null`.
- **`~/Documents/VideoTranscripts` default output path isn't used.** Run #1/#2 output landed directly in the COMS project root. Where `--output` gets its value is worth tracing — default Settings value is `~/Documents/VideoTranscripts` but clearly something else is passing through.

## Backlog

- **Real test suite.** The single `tests/test_speakers_json_emission.py` file uses stdlib `unittest` only because the repo had no test infra at all. Proper future setup: pytest + a renderer-side jest/vitest, fixtures for short video clips, mocked GeminiClient for pipeline-level integration tests. The narrow-scope unittest is a stopgap, not a pattern to scale.
- **Custom-prompt UI is a no-op as of the Task 9 pre-flight.** Either remove the ConfigScreen custom-prompt UI + `convertAndWritePrompts()` entirely, or add `--prompts-file` to v10 so the UI works. Decision still pending (Gap #2 A/B/C).
- **`--audio-only` UI is a no-op as of the Task 9 pre-flight.** Users who drop an audio file into the app will see the toggle auto-on but v10 won't receive the flag. Either add `--audio-only` to v10 or remove the toggle. Follows the Gap #2 decision.

## Shipped to master on 2026-04-21

Branch `feature/electron-pipeline-sync` merged fast-forward. 9 commits, tip `39378d1`. Master has no upstream yet — not pushed.

- `scripts/sync-pipeline.sh` + `prebuild`/`prepackage` npm wiring (root pipeline → `src/python/` on every build/package)
- Stale `src/python/v03` and `v04` removed; `CLAUDE.md` v04 reference fixed
- `--burn-timestamps` hardcoded on in `pythonRunner.ts`
- `--deidentify-names` wired end-to-end: ConfigScreen Advanced toggle → V10Config → TranscriptionConfig → IPC → CLI arg
- `transcription:hasAuditFile` narrow IPC handler
- ResultsScreen shows a privacy note when `transcript_name_map.json` is present next to the transcript

Spec: `docs/superpowers/specs/2026-04-20-electron-pipeline-sync-design.md`
Plan: `docs/superpowers/plans/2026-04-20-electron-pipeline-sync.md`

## Blocking end-to-end verification

Gap #1 (speaker detection) is now resolved via `identify --json-speakers`. Gap #2 (`--prompts-file`) is still open but does NOT block Task 9 — it only affects users who author custom prompts. See `v10-integration-gaps.md` for the Gap #2 decision tree.

## Next session — likely entry points

- Task 9 end-to-end test: short clip, both with and without the deidentify toggle.
- Gap #2 decision: add `--prompts-file` to v10, remove the custom-prompts UI, or inline prompt text. Call depends on how often the 8 colleagues actually author custom prompts.
- `--audio-only` follow-up (see Backlog).

## Decisions already made

- Root is the pipeline source of truth; `src/python/` is generated by the prebuild script (not hand-edited).
- `--burn-timestamps` is hardcoded on (correctness fix, not a user choice).
- `--deidentify-names` is off by default, not persisted between sessions, visible only under Advanced.
- `--keep-chunks` intentionally not exposed in the GUI (PII failure mode when combined with deidentify).
