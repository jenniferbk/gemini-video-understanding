# GeminiVideoUnderstanding - Troubleshooting Guide

Comprehensive debugging and issue resolution guide for common problems encountered during development and production use.

**Last Updated:** November 4, 2025

---

## Table of Contents

1. [CRITICAL: Missing Dependencies in Packaged App](#critical-missing-dependencies-in-packaged-app)
2. [Development Environment Issues](#development-environment-issues)
3. [Python Integration Issues](#python-integration-issues)
4. [Electron Build & Packaging Issues](#electron-build--packaging-issues)
5. [Database Issues](#database-issues)
6. [IPC Communication Issues](#ipc-communication-issues)
7. [UI/UX Issues](#uiux-issues)
8. [Runtime Errors](#runtime-errors)
9. [Performance Issues](#performance-issues)
10. [macOS-Specific Issues](#macos-specific-issues)
11. [Debugging Tools & Techniques](#debugging-tools--techniques)

---

## CRITICAL: Missing Dependencies in Packaged App

### Issue: "No such file or directory: 'ffmpeg'" / "KeyError: 'basic'"

**Symptoms:**
```
[Errno 2] No such file or directory: 'ffmpeg'
[Errno 2] No such file or directory: 'ffprobe'
Warning: Prompt 'enhanced_vad' not found. Using 'basic' instead.
KeyError: 'basic'
ValueError: Could not determine video duration
```

**Cause:** The packaged app is missing:
1. `ffmpeg` and `ffprobe` binaries (required for video processing)
2. `prompts.json` file or it contains incorrect prompts

**Impact:** App cannot process videos at all - complete failure

### Solution: Rebuild with All Dependencies

**Step 1: Use the automated build script**
```bash
./scripts/build-with-deps.sh
```

This script automatically:
- Downloads ffmpeg/ffprobe static binaries
- Verifies prompts.json exists and is valid
- Builds Python venv with all dependencies
- Packages everything correctly
- Verifies the final bundle

**Step 2: Manual verification**

If you prefer to build manually, follow these steps:

**1. Download ffmpeg/ffprobe:**
```bash
mkdir -p resources/bin
cd resources/bin

# macOS - download static binaries
curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ffmpeg.zip
curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ffprobe.zip
unzip ffmpeg.zip
unzip ffprobe.zip
rm *.zip
chmod +x ffmpeg ffprobe

# Test they work
./ffmpeg -version
./ffprobe -version
```

**2. Verify prompts.json:**
```bash
# Check it exists
ls -la src/python/prompts.json

# Validate JSON
python3 -m json.tool src/python/prompts.json

# Ensure it has required prompts (at minimum, "basic")
cat src/python/prompts.json | grep -E '"basic"|"enhanced_vad"'
```

**3. Update electron-builder.json:**

Ensure your `electron-builder.json` includes:
```json
{
  "extraResources": [
    {
      "from": "resources/bin",
      "to": "bin",
      "filter": ["ffmpeg", "ffprobe"]
    }
  ]
}
```

**4. Apply Python script fixes:**

See `PYTHON_SCRIPT_FIXES.md` for detailed instructions.

Quick version - add to top of `video_transcription_pipeline_v04.py`:
```python
from bundled_resource_paths import setup_ffmpeg_environment

FFMPEG_PATH, FFPROBE_PATH = setup_ffmpeg_environment()

# Then use FFMPEG_PATH and FFPROBE_PATH in all subprocess calls
subprocess.run([FFMPEG_PATH, '-i', video_path, ...])
subprocess.run([FFPROBE_PATH, '-v', 'error', ...])
```

**5. Rebuild:**
```bash
npm run build
npm run package
```

### Quick Fix for Already-Distributed App

If users already have the broken app, they can fix it without rebuilding:

**Option 1: Install ffmpeg system-wide (Easiest)**
```bash
brew install ffmpeg
```

Then relaunch the app - it will find ffmpeg/ffprobe in system PATH.

**Option 2: Manually patch the app bundle**
```bash
# 1. Download ffmpeg/ffprobe
curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ~/Downloads/ffmpeg.zip
curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ~/Downloads/ffprobe.zip
cd ~/Downloads
unzip ffmpeg.zip
unzip ffprobe.zip

# 2. Create bin directory in app
sudo mkdir -p "/Applications/Gemini Video Understanding.app/Contents/Resources/bin"

# 3. Copy binaries
sudo cp ffmpeg "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
sudo cp ffprobe "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
sudo chmod +x "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"*

# 4. Fix prompts.json (if needed)
# Create basic prompts.json
cat > /tmp/prompts.json << 'EOF'
{
  "basic": {
    "name": "Basic Transcription",
    "prompt": "Please transcribe this classroom video accurately, identifying each speaker (Teacher, Student1, Student2, etc.) with timestamps in MM:SS format."
  }
}
EOF

# Copy to app
sudo cp /tmp/prompts.json "/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/"

# 5. Relaunch app
```

### Prevention: Pre-Release Checklist

Before distributing the app to users, verify:

```bash
# 1. Mount the DMG
hdiutil attach dist/GeminiVideoUnderstanding-*.dmg

# 2. Check bundled resources
APP="/Volumes/Gemini Video Understanding/Gemini Video Understanding.app"

echo "Checking ffmpeg..."
ls -lh "$APP/Contents/Resources/bin/ffmpeg"

echo "Checking ffprobe..."
ls -lh "$APP/Contents/Resources/bin/ffprobe"

echo "Checking prompts.json..."
ls -lh "$APP/Contents/Resources/python/scripts/prompts.json"
cat "$APP/Contents/Resources/python/scripts/prompts.json" | head -20

echo "Checking Python..."
ls -lh "$APP/Contents/Resources/python/bin/python3"

# 3. Test the app
# - Open it
# - Try transcribing a SHORT test video (30 seconds)
# - Check Console.app for "✅ Found ffmpeg at:" messages
# - Verify it completes successfully

# 4. Unmount
hdiutil detach "/Volumes/Gemini Video Understanding"
```

### Related Files

- **BUNDLING_GUIDE.md** - Complete guide to bundling dependencies
- **PYTHON_SCRIPT_FIXES.md** - Detailed Python code changes
- **scripts/build-with-deps.sh** - Automated build script
- **src/python/bundled_resource_paths.py** - Path resolution module

---

## Development Environment Issues

### Issue: `npm install` fails with Python dependencies

**Symptoms:**
```
gyp ERR! stack Error: Python executable "python" is v2.7.16
```

**Cause:** Node native modules require Python 3

**Solution:**
```bash
# Set Python 3 as default for npm
npm config set python python3

# Or specify during install
npm install --python=python3
```

### Issue: TypeScript compilation errors on fresh clone

**Symptoms:**
```
error TS2307: Cannot find module 'electron'
```

**Cause:** Missing type definitions

**Solution:**
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Install Electron types explicitly if needed
npm install --save-dev @types/node @types/react @types/react-dom
```

### Issue: Hot reload not working in development

**Symptoms:** Changes to React components don't reflect without full restart

**Cause:** Webpack dev server configuration issue

**Solution:**
```javascript
// Check webpack.config.js has:
devServer: {
  hot: true,
  liveReload: true,
}

// Ensure React Fast Refresh is enabled
plugins: [
  new ReactRefreshWebpackPlugin()
]
```

---

## Python Integration Issues

### Issue: Python process not found in production

**Symptoms:**
```
Error: spawn ENOENT
Failed to start Python process
```

**Cause:** Python path resolution incorrect for packaged app

**Solution:**
```typescript
// src/main/python/pythonRunner.ts
import { app } from 'electron';
import path from 'path';

function getPythonPath(): string {
  if (app.isPackaged) {
    // Production - CORRECT path
    return path.join(process.resourcesPath, 'python', 'bin', 'python3');
  } else {
    // Development
    return path.join(__dirname, '..', '..', 'src', 'python', 'venv', 'bin', 'python3');
  }
}

// Debug: Log the resolved path
console.log('Python path:', getPythonPath());
```

**Verify Python was bundled:**
```bash
# After building, check contents
ls -la dist/mac/GeminiVideoUnderstanding.app/Contents/Resources/python/bin/
```

### Issue: Python dependencies missing in production

**Symptoms:**
```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Cause:** Virtual environment not properly bundled or dependencies not installed

**Solution:**
```bash
# Rebuild venv completely
rm -rf src/python/venv
python3 -m venv src/python/venv
source src/python/venv/bin/activate
pip install -r src/python/requirements.txt
deactivate

# Verify installation
src/python/venv/bin/python3 -c "import google.generativeai; print('OK')"

# Then rebuild app
npm run build
npm run package
```

### Issue: Progress updates not showing

**Symptoms:** Progress bar stuck at 0%, but Python script is running

**Cause:** Python stdout buffering or missing `flush=True`

**Solution:**

**Python script:**
```python
# ALWAYS include flush=True
print(f"GVU_PROGRESS:{json.dumps(progress)}", flush=True)
```

**Environment variable:**
```typescript
// src/main/python/pythonRunner.ts
this.process = spawn(pythonPath, args, {
  env: {
    ...process.env,
    PYTHONUNBUFFERED: '1'  // CRITICAL - disables buffering
  }
});
```

**Debug stdout:**
```typescript
this.process.stdout?.on('data', (data: Buffer) => {
  const output = data.toString();
  console.log('RAW STDOUT:', output);  // Add this to debug
  this.handleStdout(output);
});
```

### Issue: Python process becomes zombie

**Symptoms:** Process shows as running but not responding, CPU at 0%

**Cause:** Process didn't exit cleanly, parent didn't reap child

**Solution:**
```typescript
// src/main/python/pythonRunner.ts
cancel(): void {
  if (this.process && !this.process.killed) {
    // First try graceful termination
    this.process.kill('SIGTERM');

    // Force kill after 5 seconds if still running
    setTimeout(() => {
      if (this.process && !this.process.killed) {
        console.warn('Force killing Python process');
        this.process.kill('SIGKILL');
      }
    }, 5000);
  }
}

// Also handle process exit
this.process.on('exit', (code, signal) => {
  console.log(`Process exited: code=${code}, signal=${signal}`);
  this.process = null;  // Important: clear reference
});
```

---

## Electron Build & Packaging Issues

### Issue: `electron-builder` fails with "No code signature"

**Symptoms:**
```
Error: Command failed: codesign --sign ...
```

**Cause:** App not code-signed (required for distribution, optional for development)

**Solution (Development):**
```json
// electron-builder.json
{
  "mac": {
    "identity": null  // Skip code signing
  }
}
```

**Solution (Production):**
1. Get Apple Developer account ($99/year)
2. Create Developer ID certificate
3. Install certificate in Keychain
4. electron-builder will auto-sign

### Issue: Build succeeds but app won't launch

**Symptoms:** App icon bounces once then disappears, no window appears

**Cause:** Main process crash before window creation

**Solution:**
```bash
# Run app from Terminal to see errors
/Applications/GeminiVideoUnderstanding.app/Contents/MacOS/GeminiVideoUnderstanding

# Check Console.app for crash logs
# Go to Console.app → User Reports → Look for crash reports
```

**Common causes:**
- Missing required files (database schema, Python scripts)
- Path resolution errors
- Uncaught exceptions in main.ts

### Issue: App size is huge (>500MB)

**Symptoms:** .dmg file is extremely large

**Cause:** Bundling unnecessary files

**Solution:**
```json
// electron-builder.json
{
  "extraResources": [
    {
      "from": "src/python/venv",
      "to": "python",
      "filter": [
        "**/*",
        "!**/*.pyc",           // Exclude compiled Python
        "!**/__pycache__",     // Exclude cache
        "!**/test/**",         // Exclude tests
        "!**/tests/**",
        "!**/*.dist-info",     // Exclude pip metadata
        "!**/pip*",
        "!**/setuptools*"
      ]
    }
  ]
}
```

---

## Database Issues

### Issue: Database file not found

**Symptoms:**
```
Error: SQLITE_CANTOPEN: unable to open database file
```

**Cause:** Database path doesn't exist or no write permissions

**Solution:**
```typescript
// src/main/database/database.ts
import { app } from 'electron';
import path from 'path';
import fs from 'fs';

constructor() {
  const userDataPath = app.getPath('userData');

  // Ensure directory exists
  if (!fs.existsSync(userDataPath)) {
    fs.mkdirSync(userDataPath, { recursive: true });
  }

  const dbPath = path.join(userDataPath, 'gvu.db');
  console.log('Database path:', dbPath);

  this.db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
      console.error('Database open error:', err);
      throw err;
    }
  });
}
```

### Issue: Database locked error

**Symptoms:**
```
Error: SQLITE_BUSY: database is locked
```

**Cause:** Multiple connections trying to write simultaneously

**Solution:**
```typescript
// Enable WAL mode for better concurrency
this.db.run('PRAGMA journal_mode=WAL');

// Use transactions for multiple writes
async updateMultiple(updates: Update[]): Promise<void> {
  return new Promise((resolve, reject) => {
    this.db.serialize(() => {
      this.db.run('BEGIN TRANSACTION');

      updates.forEach(update => {
        this.db.run(update.sql, update.params);
      });

      this.db.run('COMMIT', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  });
}
```

### Issue: Database schema migrations fail

**Symptoms:** Old database version incompatible with new schema

**Solution:**
```typescript
// src/main/database/database.ts
private async migrate(): Promise<void> {
  // Get current version
  const version = await this.getVersion();

  if (version < 2) {
    // Migration to v2
    await this.db.exec(`
      ALTER TABLE jobs ADD COLUMN retry_count INTEGER DEFAULT 0;
    `);
    await this.setVersion(2);
  }

  if (version < 3) {
    // Migration to v3
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT NOT NULL,
        data TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `);
    await this.setVersion(3);
  }
}

private async getVersion(): Promise<number> {
  const row = await this.db.get('PRAGMA user_version');
  return row.user_version || 0;
}

private async setVersion(version: number): Promise<void> {
  await this.db.run(`PRAGMA user_version = ${version}`);
}
```

---

## IPC Communication Issues

### Issue: IPC handler not receiving messages

**Symptoms:** Renderer calls `window.electronAPI.someMethod()` but nothing happens

**Cause:** Handler not registered or channel name mismatch

**Solution:**
```typescript
// Check main process registration
// src/main/main.ts
import { registerTranscriptionHandlers } from './ipc/transcription';

app.on('ready', () => {
  const mainWindow = createMainWindow();

  // MUST register handlers BEFORE window loads
  registerTranscriptionHandlers(mainWindow);
});

// Check channel names match exactly
// src/main/preload.ts
contextBridge.exposeInMainWorld('electronAPI', {
  startTranscription: (config) =>
    ipcRenderer.invoke('transcription:start', config),  // Channel name
});

// src/main/ipc/transcription.ts
ipcMain.handle('transcription:start', async (_event, config) => {
  // Must match exactly ^^^
});
```

### Issue: Preload script not loading

**Symptoms:**
```
Uncaught ReferenceError: electronAPI is not defined
```

**Cause:** Preload path incorrect or security settings wrong

**Solution:**
```typescript
// src/main/main.ts
import path from 'path';
import { app } from 'electron';

const preloadPath = app.isPackaged
  ? path.join(__dirname, 'preload.js')
  : path.join(__dirname, '..', 'preload.js');

console.log('Preload path:', preloadPath);

const mainWindow = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,
    contextIsolation: true,
    preload: preloadPath
  }
});
```

---

## UI/UX Issues

### Issue: React components not re-rendering

**Symptoms:** State updates but UI doesn't change

**Cause:** Mutation of state instead of creating new objects

**Solution:**
```typescript
// BAD - mutates array
const handleAdd = () => {
  items.push(newItem);  // ❌
  setItems(items);
};

// GOOD - creates new array
const handleAdd = () => {
  setItems([...items, newItem]);  // ✅
};

// BAD - mutates object
const handleUpdate = () => {
  config.prompt = 'new value';  // ❌
  setConfig(config);
};

// GOOD - creates new object
const handleUpdate = () => {
  setConfig({ ...config, prompt: 'new value' });  // ✅
};
```

### Issue: Memory leak from event listeners

**Symptoms:** App becomes slow over time, memory usage increases

**Cause:** Event listeners not cleaned up

**Solution:**
```typescript
// src/renderer/hooks/useTranscription.ts
useEffect(() => {
  const cleanup = window.electronAPI.onProgress((update) => {
    setProgress(update);
  });

  // MUST return cleanup function
  return cleanup;
}, []);

// src/main/preload.ts
onProgress: (callback: (progress: ProgressUpdate) => void) => {
  const subscription = (_event: any, progress: ProgressUpdate) => {
    callback(progress);
  };

  ipcRenderer.on('transcription:progress', subscription);

  // Return cleanup function
  return () => {
    ipcRenderer.removeListener('transcription:progress', subscription);
  };
}
```

---

## Runtime Errors

### Issue: API key validation fails

**Symptoms:**
```
Error: Invalid API key
```

**Debug steps:**
```typescript
// Check keychain access
const apiKey = await getApiKey();
console.log('API key retrieved:', apiKey ? 'YES' : 'NO');
console.log('API key length:', apiKey?.length);
console.log('API key prefix:', apiKey?.substring(0, 10));

// Test API call
try {
  const response = await fetch('https://generativelanguage.googleapis.com/v1/models', {
    headers: { 'x-goog-api-key': apiKey }
  });
  console.log('API test status:', response.status);
} catch (err) {
  console.error('API test failed:', err);
}
```

### Issue: Video file path contains spaces

**Symptoms:** Python script fails with "file not found"

**Cause:** Path not properly escaped

**Solution:**
```typescript
// src/main/python/pythonRunner.ts
start(config: TranscriptionConfig): void {
  const args = [
    this.scriptPath,
    config.videoPath,  // Will be automatically escaped by spawn()
    '--prompt', config.prompt,
    // ...
  ];

  // Don't manually quote - spawn() handles it
  this.process = spawn(pythonPath, args);
}

// Python script should receive correct path
# video_transcription_pipeline_v04.py
import sys
video_path = sys.argv[1]  # Already unescaped
print(f"Processing: {video_path}")
```

---

## Performance Issues

### Issue: App startup is slow (>5 seconds)

**Cause:** Loading too much data at startup

**Solution:**
```typescript
// Lazy load job history
useEffect(() => {
  // Load only recent jobs initially
  const loadJobs = async () => {
    const recent = await window.electronAPI.getRecentJobs(10);
    setJobs(recent);
  };

  loadJobs();
}, []);

// Load full history on demand
const loadAllJobs = async () => {
  const all = await window.electronAPI.getAllJobs();
  setJobs(all);
};
```

### Issue: Progress updates cause UI lag

**Cause:** Too many re-renders from frequent updates

**Solution:**
```typescript
// Throttle progress updates
const [progress, setProgress] = useState(0);
const lastUpdateRef = useRef(0);

useEffect(() => {
  const cleanup = window.electronAPI.onProgress((update) => {
    const now = Date.now();

    // Update UI max 2x per second
    if (now - lastUpdateRef.current > 500) {
      setProgress(update.percent);
      lastUpdateRef.current = now;
    }
  });

  return cleanup;
}, []);
```

---

## macOS-Specific Issues

### Issue: Gatekeeper blocks app launch

**Symptoms:**
```
"GeminiVideoUnderstanding" can't be opened because it is from an unidentified developer.
```

**User Solution:**
1. Right-click app → Open
2. Click "Open" in dialog
3. Or: System Preferences → Security & Privacy → "Open Anyway"

**Developer Solution:**
Sign the app with Developer ID certificate (requires Apple Developer account)

### Issue: Keychain access denied

**Symptoms:**
```
Error: User denied access to keychain
```

**Solution:**
```typescript
// Add fallback to encrypted file storage
async function saveApiKey(apiKey: string): Promise<void> {
  try {
    await keytar.setPassword(SERVICE_NAME, ACCOUNT_NAME, apiKey);
  } catch (err) {
    console.warn('Keychain access denied, using encrypted file storage');
    await saveToEncryptedFile(apiKey);
  }
}
```

---

## Debugging Tools & Techniques

### Chrome DevTools (Renderer Process)

**Open DevTools:**
- Development: Automatically opens
- Production: Cmd+Option+I (if enabled)

```typescript
// src/main/main.ts
if (process.env.NODE_ENV === 'development') {
  mainWindow.webContents.openDevTools();
}
```

**Useful Console Commands:**
```javascript
// Check IPC API availability
console.log(window.electronAPI);

// Test IPC call
window.electronAPI.getSettings('api_key').then(console.log);

// Monitor state updates
console.log('Jobs:', jobs);
```

### Main Process Debugging

**VS Code Launch Configuration:**
```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Main Process",
      "type": "node",
      "request": "launch",
      "cwd": "${workspaceFolder}",
      "runtimeExecutable": "${workspaceFolder}/node_modules/.bin/electron",
      "args": ["."],
      "outputCapture": "std"
    }
  ]
}
```

**Console Logging:**
```typescript
// src/main/main.ts
console.log('Main process started');
console.log('App path:', app.getPath('userData'));
console.log('Is packaged:', app.isPackaged);
```

### Python Script Debugging

**Add debug logging:**
```python
# video_transcription_pipeline_v04.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/gvu-python-debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug(f"Starting transcription: {video_path}")
```

**Interactive debugging:**
```python
# Add breakpoint
import pdb
pdb.set_trace()

# Or use ipdb for better experience
import ipdb
ipdb.set_trace()
```

### Network Debugging

**Monitor API calls:**
```typescript
// Intercept fetch requests
const originalFetch = window.fetch;
window.fetch = async (...args) => {
  console.log('Fetch:', args[0]);
  const response = await originalFetch(...args);
  console.log('Response:', response.status);
  return response;
};
```

### Database Debugging

**Inspect database:**
```bash
# Open database with sqlite3 CLI
sqlite3 ~/Library/Application\ Support/GeminiVideoUnderstanding/gvu.db

# Useful queries
.tables
.schema jobs
SELECT * FROM jobs ORDER BY created_at DESC LIMIT 5;
SELECT key, value FROM settings;
```

**Enable SQL logging:**
```typescript
// src/main/database/database.ts
this.db.on('trace', (sql) => {
  console.log('SQL:', sql);
});
```

---

## Getting Help

If issues persist after trying these solutions:

1. **Check logs:**
   - Main process: Terminal output when running `npm run dev`
   - Renderer process: Chrome DevTools Console
   - Python script: `/tmp/gvu-python-debug.log`
   - macOS Console.app: Crash reports and system logs

2. **Enable verbose logging:**
   ```bash
   # Run with debug flag
   DEBUG=* npm run dev
   ```

3. **Create minimal reproduction:**
   - Isolate the issue to smallest possible code
   - Test with fresh database
   - Test with sample video file

4. **Report issue:**
   - Include error messages
   - Include relevant log excerpts
   - Include steps to reproduce
   - Include environment (macOS version, Node version, etc.)

---

**Last Updated:** November 4, 2025
