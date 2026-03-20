# Python Integration with Electron

Patterns and best practices for bundling Python applications with Electron, managing Python processes, and handling inter-process communication between Node.js and Python.

## When to use this skill

- Bundling Python dependencies with Electron app
- Spawning and managing Python child processes
- Parsing Python stdout for progress updates
- Setting up Python virtual environments
- Configuring electron-builder for Python resources
- Working with `src/python/` or `src/main/python/` directories

## Core Concept

Electron apps can bundle a complete Python environment (interpreter + dependencies) and spawn Python processes to handle computational tasks. The pattern:

1. **Bundle** - Include Python venv in app resources
2. **Spawn** - Launch Python scripts as child processes from main process
3. **Communicate** - Use stdout/stderr for structured data exchange
4. **Monitor** - Track progress and handle errors

## Python Virtual Environment Setup

### Development Setup

```bash
# Create venv in project
cd src/python
python3 -m venv venv

# Activate venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python --version
pip list

# Deactivate when done
deactivate
```

### Production - Bundling with electron-builder

```json
// electron-builder.json or package.json build section
{
  "build": {
    "extraResources": [
      {
        "from": "src/python/venv",
        "to": "python",
        "filter": [
          "**/*",
          "!**/*.pyc",           // Exclude compiled Python
          "!**/__pycache__",     // Exclude cache directories
          "!**/test/**",         // Exclude tests
          "!**/tests/**"
        ]
      },
      {
        "from": "src/python",
        "to": "python/scripts",
        "filter": [
          "*.py",
          "*.json",
          "requirements.txt"
        ]
      }
    ]
  }
}
```

**Build Process:**
1. Create and populate venv **before** running `electron-builder`
2. electron-builder copies venv to app resources
3. App accesses Python via resource path at runtime

### Path Resolution

```typescript
// src/main/python/paths.ts
import { app } from 'electron';
import path from 'path';

export function getPythonPath(): string {
  if (app.isPackaged) {
    // Production: Python bundled in resources
    return path.join(process.resourcesPath, 'python', 'bin', 'python3');
  } else {
    // Development: Use local venv
    return path.join(__dirname, '..', '..', 'src', 'python', 'venv', 'bin', 'python3');
  }
}

export function getScriptPath(scriptName: string): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'python', 'scripts', scriptName);
  } else {
    return path.join(__dirname, '..', '..', 'src', 'python', scriptName);
  }
}

// Verify Python is available
export async function verifyPythonInstallation(): Promise<boolean> {
  const pythonPath = getPythonPath();
  const fs = require('fs/promises');

  try {
    await fs.access(pythonPath);
    return true;
  } catch {
    return false;
  }
}
```

## Python Script Communication Protocol

### JSON Progress Output Pattern

**Python Side:**

```python
# video_transcription_pipeline_v04.py
import json
import sys
from datetime import datetime

def report_progress(chunk_num, total_chunks, status, json_mode=True):
    """Output progress for consumption by Electron app"""
    if json_mode:
        progress = {
            "type": "progress",
            "chunk": chunk_num,
            "total": total_chunks,
            "percent": int((chunk_num / total_chunks) * 100),
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        # Prefix for easy parsing
        print(f"GVU_PROGRESS:{json.dumps(progress)}", flush=True)
    else:
        # Human-readable for CLI use
        print(f"Processing chunk {chunk_num}/{total_chunks}... {status}")

def report_completion(output_file, stats, json_mode=True):
    """Report completion with results"""
    if json_mode:
        completion = {
            "type": "complete",
            "output_file": str(output_file),
            "stats": stats
        }
        print(f"GVU_COMPLETE:{json.dumps(completion)}", flush=True)

def report_error(message, chunk=None, fatal=True, json_mode=True):
    """Report error"""
    if json_mode:
        error = {
            "type": "error",
            "message": str(message),
            "chunk": chunk,
            "fatal": fatal
        }
        print(f"GVU_ERROR:{json.dumps(error)}", flush=True)
    else:
        print(f"ERROR: {message}", file=sys.stderr)

# Usage in script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-progress', action='store_true',
                       help='Output JSON for Electron app')
    args = parser.parse_args()

    try:
        # Process chunks
        for i in range(total_chunks):
            report_progress(i + 1, total_chunks, "processing", args.json_progress)
            # ... do work ...

        # Report completion
        report_completion(output_file, stats, args.json_progress)
    except Exception as e:
        report_error(str(e), fatal=True, json_mode=args.json_progress)
        sys.exit(1)
```

**Key Patterns:**
- Use **prefixes** (`GVU_PROGRESS:`, etc.) for easy line parsing
- Always `flush=True` to ensure immediate output
- Support both JSON and human-readable modes
- Include timestamps for debugging
- Use structured error reporting

## Python Process Manager (Node.js)

```typescript
// src/main/python/pythonRunner.ts
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { getPythonPath, getScriptPath } from './paths';

export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  consensusRuns: number;
  chunkMinutes: number;
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
  outputFile?: string;
  stats?: Record<string, any>;
}

export class PythonTranscriptionRunner extends EventEmitter {
  private process: ChildProcess | null = null;
  private processId: number | undefined;

  start(config: TranscriptionConfig): void {
    const pythonPath = getPythonPath();
    const scriptPath = getScriptPath('video_transcription_pipeline_v04.py');

    const args = [
      scriptPath,
      config.videoPath,
      '--prompt', config.prompt,
      '--consensus-runs', String(config.consensusRuns),
      '--chunk-minutes', String(config.chunkMinutes),
      '--output', config.outputPath,
      '--api-key', config.apiKey,
      '--json-progress', // Enable JSON output
    ];

    // Spawn process
    this.process = spawn(pythonPath, args, {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',  // CRITICAL: Disable buffering
        // Pass additional env vars if needed
        GOOGLE_API_KEY: config.apiKey,
      },
      // Set working directory
      cwd: path.dirname(scriptPath),
    });

    this.processId = this.process.pid;

    // Handle stdout (progress updates)
    this.process.stdout?.on('data', (data: Buffer) => {
      this.handleStdout(data.toString());
    });

    // Handle stderr (errors, warnings)
    this.process.stderr?.on('data', (data: Buffer) => {
      this.emit('log', {
        type: 'log',
        level: 'error',
        message: data.toString()
      });
    });

    // Handle process exit
    this.process.on('exit', (code, signal) => {
      if (code !== 0 && code !== null) {
        this.emit('error', {
          type: 'error',
          message: `Process exited with code ${code}`,
          fatal: true
        });
      }
      if (signal) {
        this.emit('error', {
          type: 'error',
          message: `Process killed with signal ${signal}`,
          fatal: true
        });
      }
      this.process = null;
      this.processId = undefined;
    });

    // Handle spawn errors
    this.process.on('error', (error) => {
      this.emit('error', {
        type: 'error',
        message: `Failed to spawn process: ${error.message}`,
        fatal: true
      });
    });
  }

  private handleStdout(data: string): void {
    // Split by lines (handle partial lines)
    const lines = data.split('\n');

    for (const line of lines) {
      if (!line.trim()) continue;

      // Parse structured output
      if (line.startsWith('GVU_PROGRESS:')) {
        this.parseJson(line.substring(13), 'progress');
      } else if (line.startsWith('GVU_COMPLETE:')) {
        this.parseJson(line.substring(13), 'complete');
      } else if (line.startsWith('GVU_ERROR:')) {
        this.parseJson(line.substring(10), 'error');
      } else {
        // Regular log output
        this.emit('log', {
          type: 'log',
          level: 'info',
          message: line
        });
      }
    }
  }

  private parseJson(jsonStr: string, type: string): void {
    try {
      const parsed = JSON.parse(jsonStr);
      this.emit(type, parsed);
    } catch (error) {
      console.error(`Failed to parse ${type} JSON:`, error);
      this.emit('log', {
        type: 'log',
        level: 'error',
        message: `Invalid JSON in ${type}: ${jsonStr}`
      });
    }
  }

  cancel(): void {
    if (this.process) {
      // Graceful termination
      this.process.kill('SIGTERM');

      // Force kill after timeout
      setTimeout(() => {
        if (this.process) {
          this.process.kill('SIGKILL');
        }
      }, 5000);
    }
  }

  isRunning(): boolean {
    return this.process !== null && !this.process.killed;
  }

  getProcessId(): number | undefined {
    return this.processId;
  }
}
```

**Critical Details:**
- **`PYTHONUNBUFFERED: '1'`** - Must be set for real-time output
- Use **EventEmitter** for async communication
- Provide **graceful cancellation** with fallback to force kill
- Parse stdout **line-by-line** to handle partial chunks
- Emit structured events (`progress`, `complete`, `error`, `log`)

## Integration with IPC

```typescript
// src/main/ipc/transcription.ts
import { ipcMain, BrowserWindow } from 'electron';
import { PythonTranscriptionRunner } from '../python/pythonRunner';

let transcriptionRunner: PythonTranscriptionRunner | null = null;

export function registerTranscriptionHandlers(mainWindow: BrowserWindow) {
  // Start transcription
  ipcMain.handle('transcription:start', async (_event, config) => {
    try {
      // Create new runner
      transcriptionRunner = new PythonTranscriptionRunner();

      // Forward events to renderer
      transcriptionRunner.on('progress', (update) => {
        mainWindow.webContents.send('transcription:progress', update);
      });

      transcriptionRunner.on('complete', (result) => {
        mainWindow.webContents.send('transcription:complete', result);
      });

      transcriptionRunner.on('error', (error) => {
        mainWindow.webContents.send('transcription:error', error);
      });

      transcriptionRunner.on('log', (log) => {
        console.log('Python:', log.message);
      });

      // Start process
      transcriptionRunner.start(config);

      return { success: true };
    } catch (error) {
      console.error('Failed to start transcription:', error);
      throw error;
    }
  });

  // Cancel transcription
  ipcMain.handle('transcription:cancel', async () => {
    if (transcriptionRunner) {
      transcriptionRunner.cancel();
      transcriptionRunner = null;
    }
    return { success: true };
  });

  // Get status
  ipcMain.handle('transcription:status', async () => {
    return {
      isRunning: transcriptionRunner?.isRunning() ?? false,
      processId: transcriptionRunner?.getProcessId()
    };
  });
}
```

## Testing Python Integration

### Test Script

```typescript
// scripts/test-python-integration.ts
import { PythonTranscriptionRunner } from '../src/main/python/pythonRunner';

async function test() {
  const runner = new PythonTranscriptionRunner();

  runner.on('progress', (update) => {
    console.log('Progress:', update);
  });

  runner.on('complete', (result) => {
    console.log('Complete:', result);
    process.exit(0);
  });

  runner.on('error', (error) => {
    console.error('Error:', error);
    process.exit(1);
  });

  runner.start({
    videoPath: '/path/to/test.mp4',
    prompt: 'test_prompt',
    consensusRuns: 1,
    chunkMinutes: 1,
    outputPath: '/tmp/test-output',
    apiKey: process.env.GOOGLE_API_KEY || ''
  });
}

test();
```

### Debugging Python Issues

**Check Python availability:**
```typescript
import { verifyPythonInstallation } from './paths';

const hasP ython = await verifyPythonInstallation();
if (!hasPython) {
  throw new Error('Python not found at expected path');
}
```

**Log Python path:**
```typescript
const pythonPath = getPythonPath();
console.log('Using Python:', pythonPath);
```

**Test Python directly:**
```bash
# In development
./src/python/venv/bin/python3 --version

# Test script
./src/python/venv/bin/python3 src/python/video_transcription_pipeline_v04.py \
  --help
```

## Common Patterns

### Handling Large Output

If Python outputs large amounts of data, consider:

1. **Stream to file** instead of memory
2. **Throttle** progress updates
3. **Batch** log messages

```typescript
private lastProgressTime = 0;
private readonly PROGRESS_THROTTLE_MS = 100;

private handleProgress(update: ProgressUpdate): void {
  const now = Date.now();
  if (now - this.lastProgressTime > this.PROGRESS_THROTTLE_MS) {
    this.emit('progress', update);
    this.lastProgressTime = now;
  }
}
```

### Multiple Python Processes

```typescript
class PythonProcessManager {
  private processes = new Map<string, PythonTranscriptionRunner>();

  start(id: string, config: TranscriptionConfig): void {
    const runner = new PythonTranscriptionRunner();
    this.processes.set(id, runner);
    runner.start(config);
  }

  cancel(id: string): void {
    const runner = this.processes.get(id);
    if (runner) {
      runner.cancel();
      this.processes.delete(id);
    }
  }

  cancelAll(): void {
    for (const runner of this.processes.values()) {
      runner.cancel();
    }
    this.processes.clear();
  }
}
```

### Error Recovery

```python
# Python: Retry logic
import time

MAX_RETRIES = 3
RETRY_DELAY = 2

for attempt in range(MAX_RETRIES):
    try:
        result = risky_operation()
        break
    except TransientError as e:
        if attempt < MAX_RETRIES - 1:
            report_error(
                f"Attempt {attempt + 1} failed, retrying...",
                fatal=False,
                json_mode=json_mode
            )
            time.sleep(RETRY_DELAY)
        else:
            raise
```

## Dependency Management

### requirements.txt Best Practices

```txt
# Pin major versions for stability
google-generativeai>=0.3.0,<0.4.0
librosa>=0.10.0,<0.11.0
soundfile>=0.12.0,<0.13.0

# Exact versions for critical dependencies
numpy==1.24.3
torch==2.0.1

# Platform-specific dependencies
torch==2.0.1; sys_platform == 'darwin'  # macOS
torch==2.0.1; sys_platform == 'linux'   # Linux
```

### Verify Dependencies

```python
# Add to script startup
import sys

REQUIRED_PACKAGES = [
    'google.generativeai',
    'librosa',
    'soundfile',
]

def check_dependencies():
    """Verify all required packages are installed"""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(json.dumps({
            "type": "error",
            "message": f"Missing packages: {', '.join(missing)}",
            "fatal": True
        }), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
    # ... rest of script
```

## Additional Resources

- `PROJECT_KNOWLEDGE.md` - Application architecture
- `TROUBLESHOOTING.md` - Python-specific issues
- Electron with Python: https://github.com/fyears/electron-python-example
