# GeminiVideoUnderstanding - Technical Architecture

Comprehensive architectural documentation for the GeminiVideoUnderstanding Electron desktop application. This document provides detailed technical specifications, system design, and implementation patterns.

**Last Updated:** November 4, 2025

---

## Table of Contents

1. [Technical Architecture](#technical-architecture)
2. [File Structure](#file-structure)
3. [Core Functionality Flows](#core-functionality-flows)
4. [Python Integration Architecture](#python-integration-architecture)
5. [Database Schema](#database-schema)
6. [Security Architecture](#security-architecture)
7. [UI/UX Design System](#uiux-design-system)
8. [Auto-Update System](#auto-update-system)
9. [Cost Estimation System](#cost-estimation-system)
10. [Build & Deployment](#build--deployment)
11. [Performance Architecture](#performance-architecture)
12. [Future Roadmap](#future-roadmap)

---

## Technical Architecture

### Tech Stack

**Frontend:**
- **Electron** - Desktop application framework
- **React 18+** with TypeScript - UI components
- **CSS Modules** or **Tailwind CSS** - Styling
- **React Router** - Navigation (if multi-window needed)

**Backend:**
- **Node.js** - Electron main process
- **Python 3.11+** - Bundled with app, includes complete venv
- **SQLite3** - Local job history and settings storage
- **IPC (Inter-Process Communication)** - Secure bridge between renderer and main process

**Python Dependencies (bundled in venv):**
- `google-generativeai` - Gemini API
- `librosa`, `soundfile`, `noisereduce` - Audio processing
- `whisper`, `transformers`, `torch` - VAD and ASR
- `sentence-transformers`, `scikit-learn` - Consensus analysis
- All requirements from `requirements_v04.txt`

**Build & Distribution:**
- **electron-builder** - Package for macOS (.dmg)
- **electron-updater** - Auto-update functionality
- **GitHub Releases** - Distribution mechanism

### Application Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Electron Main Process               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │ IPC Handlers │  │Python Process│  │  SQLite    ││
│  │              │  │  Manager     │  │  Database  ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
                          ↕ IPC
┌─────────────────────────────────────────────────────┐
│              Electron Renderer Process               │
│  ┌──────────────────────────────────────────────┐  │
│  │              React Application                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │  │
│  │  │  Upload  │ │  Config  │ │  Progress   │  │  │
│  │  │  Screen  │ │  Screen  │ │   Screen    │  │  │
│  │  └──────────┘ └──────────┘ └─────────────┘  │  │
│  │  ┌──────────┐ ┌──────────────────────────┐  │  │
│  │  │ Results  │ │   Prompt Manager         │  │  │
│  │  │  Screen  │ │                          │  │  │
│  │  └──────────┘ └──────────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│         Python Video Transcription Pipeline         │
│              (Bundled with Application)              │
│                                                      │
│  video_transcription_pipeline_v04.py                │
│    ↓ stdout (JSON progress updates)                 │
│    ↓ Creates transcript files                       │
│    ↓ Returns completion status                      │
└─────────────────────────────────────────────────────┘
```

---

## File Structure

```
GeminiVideoUnderstanding/
├── package.json                      # Electron + dependencies
├── tsconfig.json                     # TypeScript configuration
├── electron-builder.json             # Build configuration
├── .gitignore
├── README.md
├── CLAUDE.md                         # Project instructions
├── PROJECT_KNOWLEDGE.md              # This file
├── TROUBLESHOOTING.md                # Debugging guide
├── TODO_MVP.md                       # Task breakdown
│
├── .claude/                          # Claude Code configuration
│   ├── skills/                       # Reusable coding patterns
│   │   ├── electron-dev-guidelines.md
│   │   ├── python-integration.md
│   │   ├── react-typescript-patterns.md
│   │   └── database-operations.md
│   └── hooks/                        # Claude Code hooks
│       └── skill-rules.json
│
├── src/
│   ├── main/                         # Electron main process
│   │   ├── main.ts                   # Entry point
│   │   ├── preload.ts                # Context bridge (security)
│   │   ├── ipc/                      # IPC handlers
│   │   │   ├── transcription.ts      # Video processing handlers
│   │   │   ├── prompts.ts            # Prompt CRUD handlers
│   │   │   ├── settings.ts           # Settings handlers
│   │   │   └── jobs.ts               # Job history handlers
│   │   ├── python/                   # Python process management
│   │   │   ├── pythonRunner.ts       # Spawn and manage Python processes
│   │   │   └── progressParser.ts     # Parse stdout progress updates
│   │   └── database/                 # SQLite management
│   │       ├── schema.sql            # Database schema
│   │       └── database.ts           # Database operations
│   │
│   ├── renderer/                     # React frontend
│   │   ├── index.html                # Entry HTML
│   │   ├── index.tsx                 # React entry point
│   │   ├── App.tsx                   # Root component
│   │   ├── types/                    # TypeScript types
│   │   │   ├── transcription.ts      # Job, Config, Progress types
│   │   │   ├── prompt.ts             # Prompt types
│   │   │   └── api.ts                # IPC API types
│   │   ├── components/               # React components
│   │   │   ├── VideoUpload/
│   │   │   │   ├── VideoUpload.tsx
│   │   │   │   └── VideoUpload.module.css
│   │   │   ├── ConfigScreen/
│   │   │   │   ├── ConfigScreen.tsx
│   │   │   │   ├── PromptSelector.tsx
│   │   │   │   ├── QualityPresets.tsx
│   │   │   │   └── AdvancedSettings.tsx
│   │   │   ├── ProgressScreen/
│   │   │   │   ├── ProgressScreen.tsx
│   │   │   │   ├── ProgressBar.tsx
│   │   │   │   └── DetailedLog.tsx
│   │   │   ├── ResultsScreen/
│   │   │   │   ├── ResultsScreen.tsx
│   │   │   │   ├── TranscriptPreview.tsx
│   │   │   │   └── Statistics.tsx
│   │   │   ├── PromptManager/
│   │   │   │   ├── PromptManager.tsx
│   │   │   │   ├── PromptList.tsx
│   │   │   │   ├── PromptEditor.tsx
│   │   │   │   └── PromptImportExport.tsx
│   │   │   ├── Settings/
│   │   │   │   ├── Settings.tsx
│   │   │   │   ├── ApiKeyInput.tsx
│   │   │   │   └── PathSettings.tsx
│   │   │   └── shared/
│   │   │       ├── Button.tsx
│   │   │       ├── Input.tsx
│   │   │       ├── Select.tsx
│   │   │       └── Modal.tsx
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useTranscription.ts   # Transcription state management
│   │   │   ├── usePrompts.ts         # Prompt library management
│   │   │   └── useSettings.ts        # Settings management
│   │   ├── utils/                    # Utility functions
│   │   │   ├── formatting.ts         # Time/size formatting
│   │   │   ├── validation.ts         # Input validation
│   │   │   └── constants.ts          # App constants
│   │   └── styles/                   # Global styles
│   │       └── global.css
│   │
│   └── python/                       # Python pipeline (bundled)
│       ├── video_transcription_pipeline_v04.py
│       ├── prompts.json              # Default prompt library
│       ├── requirements.txt          # Python dependencies
│       └── venv/                     # Virtual environment (created during build)
│
├── database/                         # SQLite database (runtime)
│   └── gvu.db                        # Created on first run
│
├── resources/                        # App resources
│   ├── icon.icns                     # macOS app icon
│   └── installer-background.png      # DMG background
│
└── dist/                             # Build output (gitignored)
    └── GeminiVideoUnderstanding.dmg
```

---

## Core Functionality Flows

### 1. Video Upload Flow

**User Action:** Drag video file or click "Browse"

**System Behavior:**
1. Validate file (check extension: .mp4, .mov, .avi)
2. Extract metadata (duration, size, resolution)
3. Calculate cost estimate based on duration
4. Store file reference (don't copy - work with original location)
5. Navigate to Config Screen with video info pre-filled

**Key Files:**
- `src/renderer/components/VideoUpload/VideoUpload.tsx`
- `src/main/ipc/transcription.ts` (handler: `video:validate`)

**Implementation Notes:**
- Don't copy large video files to app directory
- Validate format quickly without loading entire file
- Cost estimation uses rough formula: chunks × consensus runs × $0.05

### 2. Configuration Screen

**User Action:** Select prompt, quality preset, configure settings

**System Behavior:**
1. Load available prompts from `prompts.json` (user's library)
2. Display quality presets:
   - **Quick:** 1 consensus run, 3-min chunks, VAD disabled (~30 min)
   - **Standard:** 3 consensus runs, 2-min chunks, VAD enabled (~90 min)
   - **High Quality:** 5 consensus runs, 2-min chunks, all features (~150 min)
3. Advanced settings toggle reveals:
   - Chunk duration (1-5 min slider)
   - VAD preprocessing (checkbox)
   - Denoising (checkbox)
   - Consensus runs (number input 1-10)
   - Model selection (dropdown)
4. Real-time cost recalculation on parameter changes
5. Validate Gemini API key is configured (redirect to settings if not)

**Key Files:**
- `src/renderer/components/ConfigScreen/ConfigScreen.tsx`
- `src/renderer/components/ConfigScreen/QualityPresets.tsx`
- `src/renderer/components/ConfigScreen/AdvancedSettings.tsx`
- `src/renderer/hooks/useSettings.ts`

### 3. Transcription Processing

**User Action:** Click "Start Transcription"

**Phase A: Job Initialization**
1. Create job record in SQLite:
```sql
INSERT INTO jobs (video_path, prompt_name, config_json, status, created_at)
VALUES (?, ?, ?, 'queued', CURRENT_TIMESTAMP);
```
2. Navigate to Progress Screen
3. Spawn Python child process with arguments:
```bash
python3 video_transcription_pipeline_v04.py \
  --video-path "/path/to/video.mp4" \
  --prompt "smallgroup_ava" \
  --consensus-runs 3 \
  --chunk-minutes 2 \
  --output "/path/to/output/folder" \
  --api-key "AIzaSy..." \
  --json-progress
```

**Phase B: Progress Monitoring**
Python script outputs JSON progress on stdout:
```json
{"type": "progress", "chunk": 5, "total": 16, "percent": 31, "status": "processing"}
{"type": "log", "level": "info", "message": "Chunk 5: Transcription complete"}
{"type": "error", "message": "Failed to process chunk 7", "retrying": true}
```

Node.js parses these and:
- Updates progress bar UI via IPC to renderer
- Logs to collapsible detail view
- Updates job status in database

**Phase C: Completion**
Python script outputs:
```json
{
  "type": "complete",
  "output_file": "/path/to/transcript.txt",
  "stats": {
    "chunks": 16,
    "lines": 342,
    "auto_accept": 287,
    "review_needed": 55,
    "processing_time_minutes": 87
  }
}
```

Main process:
1. Updates job status to 'complete' in database
2. Stores output path and stats
3. Sends completion event to renderer
4. Renderer navigates to Results Screen

**Error Handling:**
- Python process exits non-zero → mark job 'failed', show error
- Process killed by user → mark job 'cancelled'
- Retry logic for transient errors (handled by Python script)

**Key Files:**
- `src/renderer/components/ProgressScreen/ProgressScreen.tsx`
- `src/main/python/pythonRunner.ts`
- `src/main/python/progressParser.ts`
- `src/main/ipc/transcription.ts` (handlers: `transcription:start`, `transcription:cancel`)

### 4. Results & Output

**User Action:** View completed transcription

**System Behavior:**
1. Load transcript file from output path
2. Display first 100 lines in preview pane (with scroll)
3. Show statistics summary
4. Auto-save already completed by Python script to:
   - Default: `~/Documents/VideoTranscripts/[video-name]/`
   - Configurable in Settings
5. Provide actions:
   - **Open Folder:** Reveal in Finder
   - **Copy to Clipboard:** Copy full transcript text
   - **New Transcription:** Return to home screen

**Output Format (RTF-compatible):**
```
00:00 Teacher: Let's begin class today.
00:05 Student1: Can I ask a question? [verify: spkr:65]
00:12 Teacher: Yes, go ahead.
00:15 Student2: What about the homework? [verify: spkr:58 text:72]
```

**Key Files:**
- `src/renderer/components/ResultsScreen/ResultsScreen.tsx`
- `src/renderer/components/ResultsScreen/TranscriptPreview.tsx`
- `src/main/ipc/transcription.ts` (handler: `transcription:openFolder`)

### 5. Prompt Management

**User Action:** Open Prompt Manager from menu/home

**Prompt Storage:**
- Each user has local `prompts.json` in app data directory
- Default prompts bundled with app (copied on first launch)
- Format:
```json
{
  "prompts": [
    {
      "id": "uuid-here",
      "name": "smallgroup_ava",
      "description": "Small group science discussions",
      "prompt_text": "Please transcribe this classroom video...",
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-20T14:22:00Z"
    }
  ]
}
```

**Operations:**
- **Create New:** Opens editor with template
- **Edit:** Opens editor with existing prompt text
- **Duplicate:** Creates copy with " (copy)" suffix
- **Delete:** Confirms, then removes from library
- **Import:** File picker → select `.json` file → add to library
- **Export:** Save selected prompt as `.json` file for sharing

**Sharing Workflow:**
1. User A creates/edits prompt
2. User A clicks "Export" → saves `my-prompt.json`
3. User A shares file via Slack/email
4. User B opens Prompt Manager → "Import" → selects `my-prompt.json`
5. Prompt added to User B's library

**Key Files:**
- `src/renderer/components/PromptManager/PromptManager.tsx`
- `src/renderer/components/PromptManager/PromptEditor.tsx`
- `src/renderer/hooks/usePrompts.ts`
- `src/main/ipc/prompts.ts` (handlers: `prompts:list`, `prompts:save`, `prompts:delete`, `prompts:import`, `prompts:export`)

---

## Python Integration Architecture

### Build Process

1. During `electron-builder` packaging:
   ```bash
   # Create venv
   python3 -m venv src/python/venv

   # Install dependencies
   source src/python/venv/bin/activate
   pip install -r src/python/requirements.txt
   deactivate
   ```

2. `electron-builder.json` includes:
   ```json
   {
     "extraResources": [
       {
         "from": "src/python/venv",
         "to": "python",
         "filter": ["**/*"]
       },
       {
         "from": "src/python/*.py",
         "to": "python"
       }
     ]
   }
   ```

3. At runtime, Python location:
   - Development: `src/python/venv/bin/python3`
   - Production: `app.asar.unpacked/resources/python/bin/python3`

### Python Script Communication Protocol

The Python script (`video_transcription_pipeline_v04.py`) outputs structured JSON to stdout with prefixes for easy parsing:

**Progress Updates:**
```python
progress = {
    "type": "progress",
    "chunk": chunk_num,
    "total": total_chunks,
    "percent": int((chunk_num / total_chunks) * 100),
    "status": status,
    "timestamp": datetime.now().isoformat()
}
print(f"GVU_PROGRESS:{json.dumps(progress)}", flush=True)
```

**Completion:**
```python
completion = {
    "type": "complete",
    "output_file": str(final_file),
    "stats": {
        "chunks": len(all_transcripts),
        "lines": total_lines,
        "auto_accept": auto_accept_count,
        "review_needed": review_count,
        "processing_time_minutes": elapsed_minutes
    }
}
print(f"GVU_COMPLETE:{json.dumps(completion)}", flush=True)
```

**Errors:**
```python
error = {
    "type": "error",
    "message": str(e),
    "chunk": chunk_number if 'chunk_number' in locals() else None,
    "fatal": True
}
print(f"GVU_ERROR:{json.dumps(error)}", flush=True)
```

**Critical Requirements:**
- Always use `flush=True` to ensure immediate output
- Use `PYTHONUNBUFFERED: '1'` environment variable when spawning process
- Prefix all structured output for easy line parsing

See `.claude/skills/python-integration.md` for complete implementation patterns.

---

## Database Schema

### SQLite Database: `gvu.db`

**Location:**
- macOS: `~/Library/Application Support/GeminiVideoUnderstanding/gvu.db`

**Tables:**

```sql
-- Job history
CREATE TABLE jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_path TEXT NOT NULL,
  video_filename TEXT NOT NULL,
  video_duration_minutes REAL,
  prompt_name TEXT NOT NULL,
  config_json TEXT NOT NULL,  -- JSON serialized config
  status TEXT NOT NULL,  -- 'queued', 'processing', 'complete', 'failed', 'cancelled'
  output_path TEXT,
  stats_json TEXT,  -- JSON serialized stats from completion
  error_message TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME,
  completed_at DATETIME
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);

-- Settings/preferences
CREATE TABLE settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Initial settings
INSERT INTO settings (key, value) VALUES
  ('api_key_encrypted', ''),  -- Encrypted Gemini API key
  ('default_output_path', '~/Documents/VideoTranscripts'),
  ('auto_update_enabled', 'true'),
  ('theme', 'light');
```

See `.claude/skills/database-operations.md` for complete CRUD operations and patterns.

---

## Security Architecture

### Context Isolation

**Always enable in `main.ts`:**
```typescript
const mainWindow = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,        // NEVER enable
    contextIsolation: true,         // ALWAYS enable
    preload: path.join(__dirname, 'preload.js')
  }
});
```

This prevents the renderer process from directly accessing Node.js APIs, protecting against XSS attacks.

### API Key Storage

**Encryption using macOS Keychain:**

```typescript
// src/main/utils/keychain.ts
import keytar from 'keytar';

const SERVICE_NAME = 'GeminiVideoUnderstanding';
const ACCOUNT_NAME = 'gemini-api-key';

export async function saveApiKey(apiKey: string): Promise<void> {
  await keytar.setPassword(SERVICE_NAME, ACCOUNT_NAME, apiKey);
}

export async function getApiKey(): Promise<string | null> {
  return await keytar.getPassword(SERVICE_NAME, ACCOUNT_NAME);
}

export async function deleteApiKey(): Promise<boolean> {
  return await keytar.deletePassword(SERVICE_NAME, ACCOUNT_NAME);
}
```

**First-Launch Flow:**
1. App checks keychain for API key
2. If not found, show "Welcome" dialog with API key input
3. User enters key → saved to keychain
4. App validates key by making test API call
5. On success, proceed to main app

See `.claude/skills/electron-dev-guidelines.md` for complete security patterns.

---

## UI/UX Design System

### Color Palette

```css
:root {
  /* Primary - Blue (trustworthy, academic) */
  --primary-50: #eff6ff;
  --primary-500: #3b82f6;
  --primary-600: #2563eb;
  --primary-700: #1d4ed8;

  /* Success - Green */
  --success-500: #10b981;

  /* Warning - Yellow */
  --warning-500: #f59e0b;

  /* Error - Red */
  --error-500: #ef4444;

  /* Neutral - Gray */
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-200: #e5e7eb;
  --gray-300: #d1d5db;
  --gray-500: #6b7280;
  --gray-700: #374151;
  --gray-900: #111827;

  /* Semantic */
  --bg-primary: #ffffff;
  --bg-secondary: var(--gray-50);
  --text-primary: var(--gray-900);
  --text-secondary: var(--gray-700);
  --text-tertiary: var(--gray-500);
  --border: var(--gray-200);
}
```

### Typography

```css
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  --font-mono: 'SF Mono', Monaco, 'Cascadia Code', monospace;

  --text-xs: 0.75rem;    /* 12px */
  --text-sm: 0.875rem;   /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg: 1.125rem;   /* 18px */
  --text-xl: 1.25rem;    /* 20px */
  --text-2xl: 1.5rem;    /* 24px */
  --text-3xl: 1.875rem;  /* 30px */
}
```

### Spacing & Layout

```css
:root {
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-6: 1.5rem;    /* 24px */
  --space-8: 2rem;      /* 32px */
  --space-12: 3rem;     /* 48px */
  --space-16: 4rem;     /* 64px */

  --radius-sm: 0.25rem;  /* 4px */
  --radius-md: 0.375rem; /* 6px */
  --radius-lg: 0.5rem;   /* 8px */
  --radius-xl: 0.75rem;  /* 12px */
  --radius-2xl: 1rem;    /* 16px */
}
```

### Window Specifications

**Main Window:**
- Width: 800px
- Height: 600px
- Min width: 600px
- Min height: 500px
- Resizable: Yes
- Title bar: Standard macOS (with app name)

**Prompt Editor Modal:**
- Width: 700px
- Height: 500px
- Modal: Yes (blocks main window)
- Centered on screen

See `.claude/skills/react-typescript-patterns.md` for complete component patterns.

---

## Auto-Update System

### Using electron-updater

**Configuration:**
```json
// package.json
{
  "build": {
    "publish": {
      "provider": "github",
      "owner": "your-username",
      "repo": "GeminiVideoUnderstanding"
    }
  }
}
```

**Implementation:**
```typescript
// src/main/updater.ts
import { autoUpdater } from 'electron-updater';
import { BrowserWindow } from 'electron';

export function initializeAutoUpdater(mainWindow: BrowserWindow): void {
  // Check for updates on launch (silent)
  autoUpdater.checkForUpdatesAndNotify();

  // Check every 4 hours
  setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify();
  }, 4 * 60 * 60 * 1000);

  autoUpdater.on('update-available', (info) => {
    mainWindow.webContents.send('update:available', {
      version: info.version,
      releaseNotes: info.releaseNotes
    });
  });

  autoUpdater.on('update-downloaded', (info) => {
    mainWindow.webContents.send('update:downloaded', {
      version: info.version
    });
  });
}
```

**Distribution:**
1. Create GitHub Release with version tag (e.g., `v1.0.0`)
2. Upload `.dmg` file as release asset
3. electron-updater automatically detects new versions
4. User gets notification in app
5. One-click update and restart

---

## Cost Estimation System

### Gemini API Pricing

**Model:** `gemini-2.5-pro-preview-05-06`

Approximate costs based on typical classroom video (45 minutes):
- Video frames: ~2700 frames (1 fps)
- Text generation: ~4000 tokens per chunk
- 16 chunks × 3 consensus runs = 48 API calls

**Estimated cost:** $2.00 - $3.50 per video (depending on video length and settings)

**Cost Calculation:**
```typescript
function estimateCost(
  durationMinutes: number,
  consensusRuns: number,
  chunkMinutes: number,
  fps: number
): number {
  const numChunks = Math.ceil(durationMinutes / chunkMinutes);
  const framesPerChunk = chunkMinutes * 60 * fps;

  // Rough estimate: $0.05 per chunk (includes video + text generation)
  const costPerChunk = 0.05;
  const totalCost = numChunks * consensusRuns * costPerChunk;

  return totalCost;
}
```

---

## Build & Deployment

### Production Build Process

```bash
# 1. Create Python venv and install dependencies
python3 -m venv src/python/venv
source src/python/venv/bin/activate
pip install -r src/python/requirements.txt
deactivate

# 2. Build Electron app
npm run build  # Compiles TypeScript
npm run package  # Creates .dmg for macOS

# Output: dist/GeminiVideoUnderstanding.dmg
```

### electron-builder Configuration

```json
{
  "build": {
    "appId": "edu.uga.gvu",
    "productName": "GeminiVideoUnderstanding",
    "mac": {
      "category": "public.app-category.education",
      "target": ["dmg"],
      "icon": "resources/icon.icns",
      "minimumSystemVersion": "11.0",
      "hardenedRuntime": true,
      "gatekeeperAssess": false
    },
    "extraResources": [
      {
        "from": "src/python/venv",
        "to": "python",
        "filter": ["**/*", "!**/*.pyc", "!**/__pycache__"]
      },
      {
        "from": "src/python",
        "to": "python/scripts",
        "filter": ["*.py", "prompts.json", "requirements.txt"]
      }
    ]
  }
}
```

### Distribution Process

1. **Create GitHub Release:**
   - Tag version (e.g., `v1.0.0`)
   - Upload `GeminiVideoUnderstanding.dmg` as release asset
   - electron-updater detects this for auto-updates

2. **User Installation:**
   - Download `.dmg` from GitHub Releases
   - Drag app to Applications folder
   - First launch: macOS Gatekeeper prompt (if not code-signed)

3. **Code Signing (Optional but Recommended):**
   - Requires Apple Developer account ($99/year)
   - Sign with Developer ID certificate
   - Users won't see Gatekeeper warning

---

## Performance Architecture

### Video File Handling
- **Don't copy videos:** Work with original file location (users' videos are large)
- **Validate on upload:** Check format quickly without loading entire file
- **Chunking handled by Python:** Electron just passes file path

### UI Responsiveness
- **Progress updates:** Max 2 updates/second (throttle if Python outputs faster)
- **Transcript preview:** Load first 100 lines only, virtualize for scrolling
- **Database queries:** Index on `created_at` for recent jobs query

### Memory Management
- **Python process isolation:** Each transcription runs in separate process
- **Clean up completed jobs:** Archive or delete old transcripts
- **Target memory usage:** < 500MB idle, < 2GB during processing

---

## Future Roadmap

### MVP Limitations
1. **Single video at a time:** No batch queue
2. **No resume capability:** Failed jobs must restart from beginning
3. **No video preview:** Can't see clip before processing
4. **Manual prompt sharing:** No centralized prompt library
5. **Basic error messages:** Doesn't guide users on fixing common errors

### Phase 2 Enhancements
1. **Batch processing queue:** Add multiple videos, process sequentially
2. **Job resume:** Save checkpoints, restart from last completed chunk
3. **Cost tracking dashboard:** Show cumulative API spending
4. **Prompt templates:** Curated library with import
5. **Export to Transana directly:** Generate proper RTF with formatting
6. **Video trimming:** Select time range to transcribe
7. **Speaker labeling improvement:** AI-suggested speaker names based on voice
8. **Multi-language support:** Interface localization
9. **Cloud sync:** Optional sync of prompts/settings across devices
10. **Team collaboration:** Share jobs and results with colleagues

---

## Additional Resources

**Documentation:**
- Electron: https://www.electronjs.org/docs
- React: https://react.dev/
- TypeScript: https://www.typescriptlang.org/docs/
- electron-builder: https://www.electron.build/
- SQLite: https://www.sqlite.org/docs.html

**API Documentation:**
- Google Gemini: https://ai.google.dev/docs
- Gemini API Pricing: https://ai.google.dev/pricing

**Project Files:**
- CLAUDE.md - Project instructions and workflows
- TROUBLESHOOTING.md - Debugging guide
- .claude/skills/ - Reusable coding patterns

---

**Last Updated:** November 4, 2025
