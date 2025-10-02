# Gemini Video Understanding

Desktop application for transcribing classroom videos using Google Gemini AI with speaker diarization.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![License](https://img.shields.io/badge/license-ISC-green)

## Overview

**Gemini Video Understanding** is an Electron-based desktop application designed for educational researchers to transcribe classroom videos with accurate speaker identification. Built at the University of Georgia's Department of Mathematics and Science Education.

### Key Features

- 🎬 **Drag-and-drop video upload** - Simple interface for video selection
- 🤖 **Google Gemini AI integration** - Advanced multimodal transcription
- 👥 **Speaker diarization** - Automatic speaker identification and labeling
- 🎯 **Custom prompts** - Tailored transcription for different classroom contexts
- ⚡ **Quality presets** - Quick, Standard, and High Quality options
- 📊 **Real-time progress tracking** - Monitor transcription status
- 🎨 **UGA branding** - Official University of Georgia styling
- 📝 **Transana-compatible output** - Ready for qualitative analysis

## Installation

### Download

Download the latest release from the [Releases page](https://github.com/jenniferbk/gemini-video-understanding/releases).

### System Requirements

- **macOS:** 11.0 (Big Sur) or later
- **Mac Type:** Apple Silicon (M1, M2, M3) or Intel
- **Disk Space:** ~200 MB for the application
- **Internet:** Required for video transcription

### Setup Instructions

1. **Download** the `.dmg` file from releases
2. **Open** the DMG and drag the app to Applications
3. **Right-click** the app and select "Open" (first launch only)
4. **Get API key** from [Google AI Studio](https://aistudio.google.com/apikey)
5. **Enter API key** in the app settings

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

## Quick Start

1. Launch Gemini Video Understanding
2. Drag and drop a classroom video onto the upload area
3. Select a prompt from the dropdown (e.g., "Small Group Discussion")
4. Choose quality preset:
   - **Quick:** ~30 minutes processing (1 consensus run)
   - **Standard:** ~90 minutes processing (3 consensus runs) - Recommended
   - **High Quality:** ~150 minutes processing (5 consensus runs)
5. Click "Start Transcription"
6. Monitor progress and view results when complete

## Supported Formats

- MP4
- MOV
- AVI
- MKV
- WebM

## Output

Transcripts are saved to `~/Documents/VideoTranscripts/` by default. Each video gets its own folder with timestamped results in plain text format, compatible with Transana.

Example output:
```
00:00 Teacher: Let's begin class today.
00:05 Student1: Can I ask a question? [verify: spkr:65]
00:12 Teacher: Yes, go ahead.
00:15 Student2: What about the homework? [verify: spkr:58 text:72]
```

## Development

### Prerequisites

- Node.js 16+
- npm or yarn
- macOS development environment

### Setup

```bash
# Clone repository
git clone https://github.com/jenniferbk/gemini-video-understanding.git
cd gemini-video-understanding

# Install dependencies
npm install

# Run in development mode
npm run dev
```

### Build

```bash
# Build for production
npm run build

# Package as DMG
npm run package
```

### Project Structure

```
gemini-video-understanding/
├── src/
│   ├── main/          # Electron main process
│   ├── renderer/      # React frontend
│   └── python/        # Python transcription pipeline
├── package.json
├── electron-builder.json
├── tsconfig.json
└── webpack.config.js
```

## Technology Stack

- **Electron** - Desktop application framework
- **React** - UI library
- **TypeScript** - Type-safe development
- **Google Gemini API** - AI transcription
- **SQLite** - Local data storage
- **Python** - Video processing pipeline

## Cost

Each transcription costs approximately **$2-4** depending on video length. The app shows an estimate before starting.

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

### v1.0.0 (2025-01-XX)
- Initial production release
- Desktop application with drag-and-drop interface
- Google Gemini integration with speaker diarization
- Quality presets and custom prompts
- UGA branding
- macOS packaging

---

**Made with ❤️ at the University of Georgia**
