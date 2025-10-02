# GeminiVideoUnderstanding - Project Context for Claude Code

## Project Overview

**Name:** GeminiVideoUnderstanding (GVU)  
**Purpose:** Desktop application for educational researchers to transcribe classroom videos using Google Gemini's multimodal AI with speaker diarization  
**Target Users:** 8 research colleagues at University of Georgia (non-technical, Mac users)  
**Current State:** Working Python pipeline (`video_transcription_pipeline_v04.py`) that needs user-friendly desktop interface  

### Problem Statement
Researchers need to transcribe hours of classroom video with accurate speaker identification (teacher + multiple students). Current solution requires command-line expertise:
```bash
python3 video_transcription_pipeline_v04.py --chunk-minutes 2 --prompt smallgroup_ava --no-vad video.mp4
```

This is error-prone and intimidating for non-technical users who need to:
- Select appropriate prompts for different video contexts
- Configure processing parameters
- Monitor long-running jobs (45+ minutes typical)
- Get clean output for import into Transana (qualitative analysis software)

### Solution
Native macOS Electron desktop application with:
- Drag-and-drop video upload
- Visual prompt selection and management
- Real-time progress tracking
- Clean RTF-compatible transcript output
- Zero Python knowledge required

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

### File Structure

```
GeminiVideoUnderstanding/
├── package.json                      # Electron + dependencies
├── tsconfig.json                     # TypeScript configuration
├── electron-builder.json             # Build configuration
├── .gitignore
├── README.md
├── CLAUDE.md                         # This file
├── TODO_MVP.md                       # Task breakdown
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

## Core Functionality

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

**System Behavior:**

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

**System Behavior:**

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

## Python Integration Details

### Bundling Python with Electron

**Build Process:**
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

**Python Path Resolution:**
```typescript
// src/main/python/pythonRunner.ts
import { app } from 'electron';
import path from 'path';

function getPythonPath(): string {
  if (app.isPackaged) {
    // Production
    return path.join(process.resourcesPath, 'python', 'bin', 'python3');
  } else {
    // Development
    return path.join(__dirname, '..', '..', 'src', 'python', 'venv', 'bin', 'python3');
  }
}
```

### Modified Python Script Requirements

**Add JSON Progress Output:**

The existing `video_transcription_pipeline_v04.py` needs modifications to output JSON progress. Add new CLI flag `--json-progress` and modify progress reporting:

```python
# Add to argument parser
parser.add_argument('--json-progress', action='store_true',
                   help='Output progress as JSON for Electron app')

# Progress reporting function
def report_progress(chunk_num, total_chunks, status, args):
    """Output progress for consumption by Electron app"""
    if args.json_progress:
        import json
        progress = {
            "type": "progress",
            "chunk": chunk_num,
            "total": total_chunks,
            "percent": int((chunk_num / total_chunks) * 100),
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        print(f"GVU_PROGRESS:{json.dumps(progress)}", flush=True)
    else:
        # Original human-readable output
        print(f"Processing chunk {chunk_num}/{total_chunks}...")

# Use throughout pipeline
report_progress(5, 16, "transcribing", args)
```

**Completion Output:**
```python
if args.json_progress:
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

**Error Output:**
```python
if args.json_progress:
    error = {
        "type": "error",
        "message": str(e),
        "chunk": chunk_number if 'chunk_number' in locals() else None,
        "fatal": True
    }
    print(f"GVU_ERROR:{json.dumps(error)}", flush=True)
```

### Node.js Python Process Manager

```typescript
// src/main/python/pythonRunner.ts
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import { EventEmitter } from 'events';

export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  consensusRuns: number;
  chunkMinutes: number;
  vadEnabled: boolean;
  denoisingEnabled: boolean;
  outputPath: string;
  apiKey: string;
}

export interface ProgressUpdate {
  type: 'progress' | 'log' | 'error' | 'complete';
  chunk?: number;
  total?: number;
  percent?: number;
  status?: string;
  message?: string;
  level?: 'info' | 'warning' | 'error';
  outputFile?: string;
  stats?: any;
}

export class PythonTranscriptionRunner extends EventEmitter {
  private process: ChildProcess | null = null;
  
  constructor(private pythonPath: string, private scriptPath: string) {
    super();
  }
  
  start(config: TranscriptionConfig): void {
    const args = [
      this.scriptPath,
      config.videoPath,
      '--prompt', config.prompt,
      '--consensus-runs', config.consensusRuns.toString(),
      '--chunk-minutes', config.chunkMinutes.toString(),
      '--output', config.outputPath,
      '--api-key', config.apiKey,
      '--json-progress',
    ];
    
    if (!config.vadEnabled) args.push('--no-vad');
    if (!config.denoisingEnabled) args.push('--no-denoise');
    
    this.process = spawn(this.pythonPath, args, {
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });
    
    this.process.stdout?.on('data', (data) => {
      this.handleStdout(data.toString());
    });
    
    this.process.stderr?.on('data', (data) => {
      this.emit('log', { type: 'log', level: 'error', message: data.toString() });
    });
    
    this.process.on('exit', (code) => {
      if (code !== 0 && code !== null) {
        this.emit('error', { type: 'error', message: `Process exited with code ${code}` });
      }
      this.process = null;
    });
  }
  
  private handleStdout(data: string): void {
    const lines = data.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('GVU_PROGRESS:')) {
        const json = line.substring(13);
        try {
          const progress: ProgressUpdate = JSON.parse(json);
          this.emit('progress', progress);
        } catch (e) {
          console.error('Failed to parse progress JSON:', e);
        }
      } else if (line.startsWith('GVU_COMPLETE:')) {
        const json = line.substring(13);
        try {
          const completion: ProgressUpdate = JSON.parse(json);
          this.emit('complete', completion);
        } catch (e) {
          console.error('Failed to parse completion JSON:', e);
        }
      } else if (line.startsWith('GVU_ERROR:')) {
        const json = line.substring(10);
        try {
          const error: ProgressUpdate = JSON.parse(json);
          this.emit('error', error);
        } catch (e) {
          console.error('Failed to parse error JSON:', e);
        }
      } else if (line.trim()) {
        // Regular log output
        this.emit('log', { type: 'log', level: 'info', message: line });
      }
    }
  }
  
  cancel(): void {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
  }
  
  isRunning(): boolean {
    return this.process !== null;
  }
}
```

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

**Database Operations:**

```typescript
// src/main/database/database.ts
import sqlite3 from 'sqlite3';
import { app } from 'electron';
import path from 'path';

export class Database {
  private db: sqlite3.Database;
  
  constructor() {
    const userDataPath = app.getPath('userData');
    const dbPath = path.join(userDataPath, 'gvu.db');
    this.db = new sqlite3.Database(dbPath);
    this.initialize();
  }
  
  private initialize(): void {
    // Create tables if not exist (use schema.sql content)
    this.db.exec(schemaSQL);
  }
  
  createJob(job: NewJob): Promise<number> {
    return new Promise((resolve, reject) => {
      const sql = `
        INSERT INTO jobs (video_path, video_filename, video_duration_minutes, 
                         prompt_name, config_json, status)
        VALUES (?, ?, ?, ?, ?, 'queued')
      `;
      this.db.run(sql, [job.videoPath, job.videoFilename, job.videoDuration,
                       job.promptName, JSON.stringify(job.config)],
        function(err) {
          if (err) reject(err);
          else resolve(this.lastID);
        }
      );
    });
  }
  
  updateJobStatus(id: number, status: string, error?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      let sql = 'UPDATE jobs SET status = ?';
      const params: any[] = [status];
      
      if (status === 'processing' && !error) {
        sql += ', started_at = CURRENT_TIMESTAMP';
      } else if (status === 'complete' || status === 'failed' || status === 'cancelled') {
        sql += ', completed_at = CURRENT_TIMESTAMP';
      }
      
      if (error) {
        sql += ', error_message = ?';
        params.push(error);
      }
      
      sql += ' WHERE id = ?';
      params.push(id);
      
      this.db.run(sql, params, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
  
  getRecentJobs(limit: number = 10): Promise<Job[]> {
    return new Promise((resolve, reject) => {
      const sql = `
        SELECT * FROM jobs 
        ORDER BY created_at DESC 
        LIMIT ?
      `;
      this.db.all(sql, [limit], (err, rows) => {
        if (err) reject(err);
        else resolve(rows as Job[]);
      });
    });
  }
  
  getSetting(key: string): Promise<string | null> {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT value FROM settings WHERE key = ?', [key],
        (err, row: any) => {
          if (err) reject(err);
          else resolve(row ? row.value : null);
        }
      );
    });
  }
  
  setSetting(key: string, value: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const sql = `
        INSERT INTO settings (key, value, updated_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET 
          value = excluded.value,
          updated_at = CURRENT_TIMESTAMP
      `;
      this.db.run(sql, [key, value], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
}
```

---

## Security & API Key Management

### Security Best Practices

**Context Isolation:**
Enable in `main.ts`:
```typescript
const mainWindow = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,
    contextIsolation: true,
    preload: path.join(__dirname, 'preload.js')
  }
});
```

**Preload Script (Bridge):**
```typescript
// src/main/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

// Expose safe, limited API to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  // Transcription
  startTranscription: (config: any) => ipcRenderer.invoke('transcription:start', config),
  cancelTranscription: () => ipcRenderer.invoke('transcription:cancel'),
  onProgress: (callback: (progress: any) => void) => {
    ipcRenderer.on('transcription:progress', (_event, progress) => callback(progress));
  },
  
  // Prompts
  getPrompts: () => ipcRenderer.invoke('prompts:list'),
  savePrompt: (prompt: any) => ipcRenderer.invoke('prompts:save', prompt),
  deletePrompt: (id: string) => ipcRenderer.invoke('prompts:delete', id),
  
  // Settings
  getSetting: (key: string) => ipcRenderer.invoke('settings:get', key),
  setSetting: (key: string, value: string) => ipcRenderer.invoke('settings:set', key, value),
  
  // System
  openFolder: (path: string) => ipcRenderer.invoke('system:openFolder', path),
});
```

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

**Usage in IPC handlers:**
```typescript
// src/main/ipc/settings.ts
import { ipcMain } from 'electron';
import { saveApiKey, getApiKey } from '../utils/keychain';

ipcMain.handle('settings:saveApiKey', async (_event, apiKey: string) => {
  await saveApiKey(apiKey);
  return { success: true };
});

ipcMain.handle('settings:getApiKey', async () => {
  const apiKey = await getApiKey();
  return { apiKey };
});
```

**First-Launch Flow:**
1. App checks keychain for API key
2. If not found, show "Welcome" dialog with API key input
3. User enters key → saved to keychain
4. App validates key by making test API call
5. On success, proceed to main app

---

## UI/UX Design Specifications

### Design System

**Colors:**
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

**Typography:**
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

**Spacing:**
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
}
```

**Border Radius:**
```css
:root {
  --radius-sm: 0.25rem;  /* 4px */
  --radius-md: 0.375rem; /* 6px */
  --radius-lg: 0.5rem;   /* 8px */
  --radius-xl: 0.75rem;  /* 12px */
  --radius-2xl: 1rem;    /* 16px */
}
```

### Component Specifications

**Button Variants:**
```typescript
// Primary button (main actions)
<button className="btn-primary">
  Start Transcription
</button>

// Secondary button (cancel, back)
<button className="btn-secondary">
  Cancel
</button>

// Ghost button (tertiary actions)
<button className="btn-ghost">
  Show Advanced
</button>
```

**CSS:**
```css
.btn-primary {
  background-color: var(--primary-600);
  color: white;
  padding: var(--space-2) var(--space-4);
  border-radius: var(--radius-md);
  font-weight: 500;
  transition: background-color 0.2s;
}

.btn-primary:hover {
  background-color: var(--primary-700);
}

.btn-primary:disabled {
  background-color: var(--gray-300);
  cursor: not-allowed;
}
```

**Input Fields:**
```css
.input {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  font-size: var(--text-base);
  transition: border-color 0.2s;
}

.input:focus {
  outline: none;
  border-color: var(--primary-500);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
```

**Progress Bar:**
```css
.progress-container {
  width: 100%;
  height: 8px;
  background-color: var(--gray-200);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background-color: var(--primary-600);
  transition: width 0.3s ease;
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
  
  autoUpdater.on('error', (err) => {
    console.error('AutoUpdater error:', err);
  });
}
```

**Renderer Handling:**
```typescript
// Show notification banner when update available
useEffect(() => {
  window.electronAPI.onUpdateAvailable((info: any) => {
    setUpdateAvailable(info);
  });
  
  window.electronAPI.onUpdateDownloaded((info: any) => {
    // Show "Install Update" prompt
    setUpdateReady(info);
  });
}, []);
```

**Update Dialog:**
User clicks "Install Update" → IPC call → `autoUpdater.quitAndInstall()`

---

## Cost Estimation

### Gemini API Pricing (as of 2025)

**Model:** `gemini-2.5-pro-preview-05-06`

Approximate costs based on typical classroom video (45 minutes):
- Video frames: ~2700 frames (1 fps)
- Text generation: ~4000 tokens per chunk
- 16 chunks × 3 consensus runs = 48 API calls

**Estimated cost:** $2.00 - $3.50 per video (depending on video length and settings)

**Cost calculation in app:**
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

## Testing Strategy

### Unit Tests
- Utility functions (formatting, validation)
- Database operations
- Progress parser logic

**Framework:** Jest

```typescript
// Example: src/renderer/utils/formatting.test.ts
describe('formatDuration', () => {
  it('formats minutes correctly', () => {
    expect(formatDuration(125)).toBe('2 hours 5 minutes');
  });
  
  it('handles seconds', () => {
    expect(formatDuration(65)).toBe('1 minute 5 seconds');
  });
});
```

### Integration Tests
- Python process spawning
- IPC communication
- Database queries

### Manual Testing Checklist
- [ ] Video upload (various formats)
- [ ] Config screen validation
- [ ] Progress tracking accuracy
- [ ] Cancellation mid-process
- [ ] Error handling (invalid API key, network failure)
- [ ] Prompt import/export
- [ ] Settings persistence
- [ ] Auto-update flow

---

## Deployment & Distribution

### Building the Application

**Development:**
```bash
npm install
npm run dev  # Starts Electron with hot reload
```

**Production Build:**
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

**electron-builder Configuration:**
```json
{
  "build": {
    "appId": "edu.uga.gvu",
    "productName": "GeminiVideoUnderstanding",
    "files": [
      "dist/**/*",
      "node_modules/**/*",
      "package.json"
    ],
    "mac": {
      "category": "public.app-category.education",
      "target": ["dmg"],
      "icon": "resources/icon.icns",
      "minimumSystemVersion": "11.0",
      "hardenedRuntime": true,
      "gatekeeperAssess": false,
      "entitlements": "entitlements.mac.plist",
      "entitlementsInherit": "entitlements.mac.plist"
    },
    "dmg": {
      "background": "resources/installer-background.png",
      "iconSize": 100,
      "contents": [
        {
          "x": 380,
          "y": 180,
          "type": "link",
          "path": "/Applications"
        },
        {
          "x": 110,
          "y": 180,
          "type": "file"
        }
      ]
    },
    "extraResources": [
      {
        "from": "src/python/venv",
        "to": "python",
        "filter": ["**/*"]
      },
      {
        "from": "src/python",
        "to": "python",
        "filter": ["*.py", "*.json", "requirements.txt"]
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
   - First launch: macOS Gatekeeper prompt (developer not verified)
   - User: System Preferences → Security → "Open Anyway"

3. **Code Signing (Optional but Recommended):**
   - Requires Apple Developer account ($99/year)
   - Sign with Developer ID certificate
   - Users won't see Gatekeeper warning
   - Instructions: https://www.electron.build/code-signing

---

## Known Limitations & Future Enhancements

### MVP Limitations
1. **Single video at a time:** No batch queue
2. **No resume capability:** Failed jobs must restart from beginning
3. **No video preview:** Can't see clip before processing
4. **Manual prompt sharing:** No centralized prompt library
5. **Basic error messages:** Doesn't guide users on fixing common errors

### Future Enhancements (Phase 2)
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

## Development Guidelines

### Code Style
- **TypeScript:** Strict mode enabled
- **ESLint:** Airbnb style guide
- **Prettier:** Automatic formatting
- **Naming:**
  - Components: PascalCase (`VideoUpload.tsx`)
  - Functions/variables: camelCase (`handleSubmit`)
  - Constants: UPPER_SNAKE_CASE (`DEFAULT_CHUNK_MINUTES`)
  - CSS Modules: camelCase (`styles.uploadContainer`)

### Git Workflow
- **Main branch:** Stable releases only
- **Develop branch:** Active development
- **Feature branches:** `feature/prompt-manager`, `feature/progress-tracking`
- **Commit messages:** Conventional commits format
  ```
  feat: add prompt import/export functionality
  fix: correct progress bar percentage calculation
  docs: update README with installation instructions
  ```

### Component Structure
```typescript
// src/renderer/components/VideoUpload/VideoUpload.tsx
import React, { useState, useCallback } from 'react';
import styles from './VideoUpload.module.css';

interface VideoUploadProps {
  onVideoSelected: (file: File) => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoSelected }) => {
  const [dragActive, setDragActive] = useState(false);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (file && isValidVideoFile(file)) {
      onVideoSelected(file);
    }
  }, [onVideoSelected]);
  
  return (
    <div 
      className={`${styles.dropzone} ${dragActive ? styles.active : ''}`}
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
    >
      <p>Drag video here or click to browse</p>
    </div>
  );
};

function isValidVideoFile(file: File): boolean {
  const validExtensions = ['.mp4', '.mov', '.avi'];
  return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
}
```

### Error Handling Patterns
```typescript
// Renderer (React)
try {
  const result = await window.electronAPI.startTranscription(config);
  setJobId(result.jobId);
  navigate('/progress');
} catch (error) {
  if (error instanceof ApiKeyError) {
    showError('Invalid API key. Please check your settings.');
  } else if (error instanceof NetworkError) {
    showError('Network error. Check your internet connection.');
  } else {
    showError('An unexpected error occurred. Please try again.');
    console.error(error);
  }
}

// Main (IPC Handler)
ipcMain.handle('transcription:start', async (_event, config) => {
  try {
    // Validate API key
    const apiKey = await getApiKey();
    if (!apiKey) {
      throw new ApiKeyError('No API key configured');
    }
    
    // Create job
    const jobId = await db.createJob({...config, apiKey});
    
    // Start Python process
    pythonRunner.start({...config, apiKey});
    
    return { success: true, jobId };
  } catch (error) {
    console.error('Transcription start error:', error);
    throw error;  // Propagates to renderer
  }
});
```

---

## Environment Variables

### Development
```bash
# .env.development
NODE_ENV=development
PYTHON_PATH=src/python/venv/bin/python3
```

### Production
```bash
# .env.production
NODE_ENV=production
# Python path resolved at runtime from app resources
```

---

## Performance Considerations

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

---

## Debugging & Troubleshooting

### Development Tools
- **Chrome DevTools:** Open with `Cmd+Option+I` in dev mode
- **Main process debugging:** VS Code with launch configuration
- **Python debugging:** Add `import pdb; pdb.set_trace()` in script

### Common Issues

**Issue:** Python process not found in production  
**Solution:** Check `getPythonPath()` resolves correctly for packaged app

**Issue:** API key not persisting  
**Solution:** Check keychain access permissions, fallback to encrypted file storage

**Issue:** Progress updates not showing  
**Solution:** Verify stdout flushing in Python (`flush=True`), check IPC listeners

**Issue:** Video won't upload  
**Solution:** Check file size limits, validate MIME type

---

## Resources & References

### Documentation
- **Electron:** https://www.electronjs.org/docs
- **React:** https://react.dev/
- **TypeScript:** https://www.typescriptlang.org/docs/
- **electron-builder:** https://www.electron.build/
- **SQLite:** https://www.sqlite.org/docs.html

### Existing Pipeline
- Current implementation: `video_transcription_pipeline_v04.py`
- Prompts library: `prompts.json`
- Requirements: `requirements_v04.txt`

### API Documentation
- **Google Gemini:** https://ai.google.dev/docs
- **Gemini API Pricing:** https://ai.google.dev/pricing

---

## Success Criteria

### MVP Complete When:
1. ✅ User can install app with zero configuration (except API key)
2. ✅ User can drag-drop video and complete transcription with 3 clicks
3. ✅ Progress bar accurately reflects processing status
4. ✅ Output transcript is RTF-compatible for Transana import
5. ✅ Prompt library supports create/edit/import/export
6. ✅ App auto-updates when new version available
7. ✅ 8 colleagues successfully use app for 2 weeks without technical support

### Quality Benchmarks:
- **Startup time:** < 3 seconds
- **Processing overhead:** < 5% slower than direct Python script
- **Memory usage:** < 500MB idle, < 2GB during processing
- **Crash rate:** < 1% of transcription jobs
- **User satisfaction:** 4/5 stars from colleague feedback

---

## Contact & Support

**Primary Developer:** [Your Name]  
**Project Lead:** Jennifer (UGA COMS Research)  
**Repository:** https://github.com/[username]/GeminiVideoUnderstanding  
**Issues:** GitHub Issues for bug reports and feature requests  

---

## Changelog

### v1.0.0 (Initial MVP)
- Video upload with drag-and-drop
- Prompt selection and management
- Real-time progress tracking
- Results preview and export
- Auto-update system
- API key management with keychain integration

---

**END OF CONTEXT DOCUMENT**

This document provides complete context for Claude Code to implement GeminiVideoUnderstanding. Refer to `TODO_MVP.md` for structured task breakdown.
