# Gemini Video Understanding

Transcription of classroom videos using Google Gemini AI with speaker diarization. Built at the University of Georgia's Department of Mathematics and Science Education.

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![License](https://img.shields.io/badge/license-ISC-green)

## Overview

This project has two ways in:

1. **Desktop app** (macOS) — an Electron wrapper around the pipeline for non-technical research-team use. Drag-drop a video, review detected speakers, get a transcript.
2. **Python CLI** — the pipeline itself (`video_transcription_pipeline_v10.py`). If you're comfortable installing Python packages, this is the most direct way to use it, and the only way to access batch processing.

### Key Features

- **Multimodal transcription** using Gemini 3 Flash (`gemini-3-flash-preview` by default) — the model watches the video, so transcripts include gesture, written work, and screen content, not just speech
- **Two-pass speaker identification** — the pipeline auto-detects speakers with visual descriptions, then a human confirms/corrects before transcription
- **Speaker registry** passed as structured context to every chunk for consistent labels across a full class period
- **Chunked processing** (60s chunks, 15s overlap) for speaker continuity on long recordings
- **Optional name de-identification** — a second Gemini pass replaces real names with realistic pseudonyms, with an audit file for re-identification under controlled access
- **Burned-in timestamps** option that eliminates model clock drift
- **Dual output** — annotated research transcript + clean Transana-compatible transcript + SRT subtitles
- **Batch processing** with saved speaker manifests and parallel workers
- **Cost estimation** before you commit — roughly **$0.19 per hour of video** at default settings

## Option 1: Desktop App (macOS)

Download the `.dmg` from the [Releases page](https://github.com/jenniferbk/gemini-video-understanding/releases) and see **[INSTALL.md](INSTALL.md)** for step-by-step setup (including the right-click → Open step needed because the app is not code-signed).

App workflow: drag-drop a video → configure (or keep defaults) → **Detect Speakers & Start** → review the detected speakers → transcription runs with live progress. Transcripts land in `~/Documents/VideoTranscripts/` by default.

## Option 2: Python CLI

### Requirements

- **Python 3.13** (what the project is developed and tested on)
- **ffmpeg** and **ffprobe** on your `PATH` (used for chunking; `brew install ffmpeg`)
- For `--burn-timestamps` only: an ffmpeg build with `drawtext` support (`brew install ffmpeg-full`; Homebrew's default ffmpeg lacks it). Point the pipeline at it with `--drawtext-ffmpeg` if it isn't at the default Homebrew path.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Setup

```bash
git clone https://github.com/jenniferbk/gemini-video-understanding.git
cd gemini-video-understanding
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-here"   # or pass --api-key per run
```

### Usage

```bash
# Estimate cost first (no API charges for the estimate itself)
python video_transcription_pipeline_v10.py estimate video.mp4

# Process a single video (interactive speaker ID, then transcription)
python video_transcription_pipeline_v10.py process video.mp4

# Pick a prompt style from prompts.json (e.g. wholeclass, smallgroup)
python video_transcription_pipeline_v10.py process video.mp4 -p smallgroup

# Identify speakers for a folder of videos (saves manifests for later batch run)
python video_transcription_pipeline_v10.py identify ./videos/ --prompt smallgroup

# Batch process unattended using saved manifests
python video_transcription_pipeline_v10.py batch ./videos/ --workers 5
```

Useful `process` flags (see `--help` for the full list):

| Flag | What it does |
|---|---|
| `-p, --prompt` | Prompt key from `prompts.json` (whole-class vs. small-group framing, etc.) |
| `--deidentify-names` | Second Gemini pass replacing real names with pseudonyms (see below) |
| `--burn-timestamps` | Burn a timer into each chunk and have the model read it — eliminates timestamp drift |
| `-o, --output` | Output directory |
| `--single-output` | One annotated transcript instead of the dual research + Transana outputs |
| `--no-confirm` | Skip the interactive confirmation prompt |

### Output

By default each video produces three files:

- `<video>_transcript.txt` — annotated research transcript (timestamps, speaker labels, bracketed activity descriptions)
- `<video>_transana.txt` — clean transcript for import into Transana
- `<video>.srt` — subtitles

Transcript lines look like:

```
12:40 [Student-Maya points at the protractor on the shared worksheet]
12:43 Student-Maya: I think we turn it ninety degrees.
12:47 Teacher-Lee: What makes you say ninety?
```

(Speaker names shown here are pseudonyms; see de-identification below.)

## Name De-identification

`--deidentify-names` runs a second Gemini pass after transcription to detect real names (students and adults) and substitute realistic pseudonyms (`Student-Hannah`, `Ms. Kelly`, …). It writes `transcript_name_map.json` next to the transcript as an audit trail.

Two things to know if you work under an IRB:

- The audit file contains the real-name ↔ pseudonym mapping. Store it under separate access control from the de-identified transcript.
- Don't combine `--deidentify-names` with `--keep-chunks` when privacy matters — per-chunk files are written *before* the de-identification pass and retain real names.

## Supported Formats

MP4, MOV, AVI, MKV, WebM (anything ffmpeg can read, in practice). An audio-only mode is also available for recordings without video.

## Development (Electron App)

### Prerequisites

- Node.js 16+
- npm
- macOS development environment

### Setup

```bash
git clone https://github.com/jenniferbk/gemini-video-understanding.git
cd gemini-video-understanding
npm install
npm run dev      # run in development mode
npm run build    # build for production
npm run package  # package as DMG
```

### Project Structure

```
gemini-video-understanding/
├── video_transcription_pipeline_v10.py   # the pipeline (canonical copy)
├── deidentify_names.py                   # name de-identification pass
├── prompts.json                          # classroom prompt library
├── src/
│   ├── main/          # Electron main process
│   ├── renderer/      # React frontend
│   └── python/        # bundled pipeline (synced from repo root at build)
├── package.json
└── electron-builder.json
```

## Technology Stack

- **Google Gemini API** (`google-genai` SDK) — multimodal AI transcription
- **Python** — transcription pipeline
- **Electron + React + TypeScript** — desktop app
- **SQLite** — app-local data storage
- **ffmpeg** — video chunking and timestamp burn-in

## Project Team

**Lead Developer:** Jennifer Kleiman  
**Project PIs:** AnnaMarie Conner, Xiaoming Zhai  
**Institution:** University of Georgia, Department of Mathematics and Science Education  
**Funding:** National Science Foundation (C4OMS project, AI4STEM Center)

## License

ISC License

## Acknowledgments

Developed as part of the C4OMS project and AI4STEM Center at the University of Georgia, funded by the National Science Foundation.

## Support

For issues, questions, or feature requests, please [open an issue](https://github.com/jenniferbk/gemini-video-understanding/issues).

## Version History

### v1.2.0 (April 2026)
- Optional name de-identification (`--deidentify-names`) with pseudonym substitution and audit file
- Burned-in timestamp support; the desktop app now always burns timestamps
- Fixed chunk-1 timestamp offset on videos where audio starts before the first video frame
- Audio-only transcription mode
- Pipeline now synced from repo root into the app at build time (single canonical copy)

### v1.1.x (October–November 2025)
- Migrated from deprecated `google-generativeai` to the `google-genai` SDK
- Fixed ffmpeg/ffprobe bundling in the packaged app
- Prompt fixes: custom prompts file path, prompt editing, UUID matching between Electron and Python
- Added prompt validation to prevent wasted API spend

### v1.0.2 (October 2025)
- Fixed prompt selection dropdown to dynamically load and update prompts
- Added info button (ℹ️) to all screens for easy access to About information
- Moved "Manage Prompts" button to prompt selection area for better UX
- Fixed Python bundling with portable distribution for reliable installation

### v1.0.0 (January 2025)
- Initial production release
- Multimodal AI transcription with Google Gemini
- Automatic speaker diarization
- Context-aware classroom prompts
- Transana-compatible output format
- macOS native application
