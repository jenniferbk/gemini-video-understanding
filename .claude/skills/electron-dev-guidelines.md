# Electron Development Guidelines

Comprehensive patterns and best practices for Electron desktop application development with React and TypeScript. This skill covers IPC handlers, security, preload scripts, and window management.

## When to use this skill

- Working on files in `src/main/` or `src/renderer/`
- Implementing IPC (Inter-Process Communication) handlers
- Setting up preload scripts and context bridges
- Configuring Electron security settings
- Managing application windows and lifecycle
- Building and packaging Electron apps

## Core Architecture Pattern

### Security-First Approach

**ALWAYS** enable these security settings in main window creation:

```typescript
// src/main/main.ts
const mainWindow = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,        // NEVER enable
    contextIsolation: true,         // ALWAYS enable
    preload: path.join(__dirname, 'preload.js')
  }
});
```

**Why:** This prevents the renderer process from directly accessing Node.js APIs, protecting against XSS attacks.

## IPC Communication Patterns

### 1. Preload Script (Context Bridge)

The preload script is your **secure bridge** between main and renderer processes.

```typescript
// src/main/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

// Expose ONLY specific, safe methods to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  // Two-way communication (invoke/handle)
  startTranscription: (config: TranscriptionConfig) =>
    ipcRenderer.invoke('transcription:start', config),

  getSettings: (key: string) =>
    ipcRenderer.invoke('settings:get', key),

  // One-way communication (send)
  logEvent: (event: string, data: any) =>
    ipcRenderer.send('analytics:log', event, data),

  // Event listeners (on)
  onProgress: (callback: (progress: ProgressUpdate) => void) => {
    const subscription = (_event: any, progress: ProgressUpdate) =>
      callback(progress);
    ipcRenderer.on('transcription:progress', subscription);

    // Return cleanup function
    return () => {
      ipcRenderer.removeListener('transcription:progress', subscription);
    };
  },
});

// TypeScript declarations for renderer
declare global {
  interface Window {
    electronAPI: {
      startTranscription: (config: TranscriptionConfig) => Promise<{ jobId: number }>;
      getSettings: (key: string) => Promise<string | null>;
      logEvent: (event: string, data: any) => void;
      onProgress: (callback: (progress: ProgressUpdate) => void) => () => void;
    };
  }
}
```

**Key Patterns:**
- Use `invoke/handle` for request-response (async)
- Use `send/on` for one-way messages
- Always return cleanup functions for event listeners
- Never expose raw `ipcRenderer` to renderer

### 2. IPC Handlers (Main Process)

Organize handlers by domain in separate files:

```typescript
// src/main/ipc/transcription.ts
import { ipcMain } from 'electron';
import { PythonTranscriptionRunner } from '../python/pythonRunner';

export function registerTranscriptionHandlers(
  pythonRunner: PythonTranscriptionRunner
) {
  // Two-way handler
  ipcMain.handle('transcription:start', async (_event, config: TranscriptionConfig) => {
    try {
      // Validate input
      if (!config.videoPath || !config.prompt) {
        throw new Error('Missing required fields');
      }

      // Perform operation
      const jobId = await database.createJob(config);
      pythonRunner.start(config);

      return { success: true, jobId };
    } catch (error) {
      console.error('Transcription start error:', error);
      throw error; // Propagates to renderer
    }
  });

  // One-way handler
  ipcMain.on('analytics:log', (_event, eventName: string, data: any) => {
    // Handle event (no response expected)
    logger.info(eventName, data);
  });
}
```

**Error Handling:**
- Always wrap handlers in try-catch
- Log errors in main process
- Throw errors to propagate to renderer
- Use typed error classes for different error types

### 3. Renderer Usage (React)

```typescript
// src/renderer/hooks/useTranscription.ts
import { useState, useEffect, useCallback } from 'react';

export function useTranscription() {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Subscribe to progress events
  useEffect(() => {
    const cleanup = window.electronAPI.onProgress((update) => {
      setProgress(update);
    });

    return cleanup; // Cleanup on unmount
  }, []);

  // Start transcription
  const startTranscription = useCallback(async (config: TranscriptionConfig) => {
    try {
      setError(null);
      const result = await window.electronAPI.startTranscription(config);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    }
  }, []);

  return { progress, error, startTranscription };
}
```

## Window Management

### Creating Windows

```typescript
// src/main/windows/createMainWindow.ts
import { BrowserWindow } from 'electron';
import path from 'path';

export function createMainWindow(): BrowserWindow {
  const window = new BrowserWindow({
    width: 800,
    height: 600,
    minWidth: 600,
    minHeight: 500,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, '../preload.js'),
      // For production
      sandbox: true,
      // Disable remote module
      enableRemoteModule: false,
    },
    // macOS specific
    titleBarStyle: 'hiddenInset',
    show: false, // Show only when ready
  });

  // Load content
  if (process.env.NODE_ENV === 'development') {
    window.loadURL('http://localhost:3000');
    window.webContents.openDevTools();
  } else {
    window.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  // Show when ready to avoid flicker
  window.once('ready-to-show', () => {
    window.show();
  });

  return window;
}
```

### Window Lifecycle Management

```typescript
// src/main/main.ts
import { app, BrowserWindow } from 'electron';

let mainWindow: BrowserWindow | null = null;

app.on('ready', () => {
  mainWindow = createMainWindow();

  // Additional setup
  registerIpcHandlers();
  setupAutoUpdater();
});

// macOS: Keep app running when all windows closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// macOS: Recreate window when dock icon clicked
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    mainWindow = createMainWindow();
  }
});

// Clean up before quit
app.on('before-quit', async () => {
  // Clean up resources
  await database.close();
  pythonRunner.killAll();
});
```

## Python Process Management

### Spawning Child Processes

```typescript
// src/main/python/pythonRunner.ts
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';
import { app } from 'electron';

export class PythonTranscriptionRunner extends EventEmitter {
  private process: ChildProcess | null = null;

  private getPythonPath(): string {
    if (app.isPackaged) {
      // Production: bundled Python
      return path.join(process.resourcesPath, 'python', 'bin', 'python3');
    } else {
      // Development: venv
      return path.join(__dirname, '..', '..', 'src', 'python', 'venv', 'bin', 'python3');
    }
  }

  start(config: TranscriptionConfig): void {
    const pythonPath = this.getPythonPath();
    const scriptPath = this.getScriptPath();

    const args = [
      scriptPath,
      config.videoPath,
      '--prompt', config.prompt,
      '--output', config.outputPath,
      '--json-progress', // Important for IPC
    ];

    this.process = spawn(pythonPath, args, {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1' // Critical for real-time output
      }
    });

    this.process.stdout?.on('data', (data) => {
      this.handleStdout(data.toString());
    });

    this.process.stderr?.on('data', (data) => {
      this.emit('error', data.toString());
    });

    this.process.on('exit', (code) => {
      if (code !== 0) {
        this.emit('error', `Process exited with code ${code}`);
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
          const progress = JSON.parse(json);
          this.emit('progress', progress);
        } catch (e) {
          console.error('Failed to parse progress:', e);
        }
      }
    }
  }

  cancel(): void {
    if (this.process) {
      this.process.kill('SIGTERM');
    }
  }
}
```

**Key Patterns:**
- Always set `PYTHONUNBUFFERED: '1'` for real-time output
- Parse stdout line-by-line for structured data
- Use EventEmitter pattern for async communication
- Provide cleanup methods (`cancel()`)

## File System Operations

### Safe Path Resolution

```typescript
import { app } from 'electron';
import path from 'path';
import fs from 'fs/promises';

// Get user data directory (cross-platform)
export function getUserDataPath(): string {
  return app.getPath('userData');
  // macOS: ~/Library/Application Support/YourApp
  // Windows: %APPDATA%/YourApp
  // Linux: ~/.config/YourApp
}

// Get resource path (bundled files)
export function getResourcePath(filename: string): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, filename);
  } else {
    return path.join(__dirname, '..', '..', filename);
  }
}

// Safe file operations
export async function safeReadFile(filePath: string): Promise<string> {
  try {
    return await fs.readFile(filePath, 'utf-8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new Error(`File not found: ${filePath}`);
    }
    throw error;
  }
}
```

## Auto-Update Implementation

```typescript
// src/main/updater.ts
import { autoUpdater } from 'electron-updater';
import { BrowserWindow } from 'electron';

export function setupAutoUpdater(mainWindow: BrowserWindow) {
  // Configure
  autoUpdater.autoDownload = false;
  autoUpdater.autoInstallOnAppQuit = true;

  // Check on startup
  autoUpdater.checkForUpdatesAndNotify();

  // Check periodically
  setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify();
  }, 4 * 60 * 60 * 1000); // Every 4 hours

  // Events
  autoUpdater.on('update-available', (info) => {
    mainWindow.webContents.send('update:available', {
      version: info.version,
      releaseNotes: info.releaseNotes
    });
  });

  autoUpdater.on('update-downloaded', (info) => {
    mainWindow.webContents.send('update:ready', {
      version: info.version
    });
  });

  autoUpdater.on('error', (err) => {
    console.error('AutoUpdater error:', err);
  });
}

// In IPC handlers
ipcMain.handle('update:install', () => {
  autoUpdater.quitAndInstall();
});
```

## Build Configuration

```json
{
  "build": {
    "appId": "edu.uga.gemini-video-understanding",
    "productName": "Gemini Video Understanding",
    "files": [
      "dist/**/*",
      "!dist/**/*.map"
    ],
    "mac": {
      "category": "public.app-category.education",
      "target": ["dmg"],
      "icon": "resources/icon.icns",
      "hardenedRuntime": true,
      "gatekeeperAssess": false,
      "entitlements": "entitlements.mac.plist"
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

## Common Patterns

### Environment Detection

```typescript
export const isDevelopment = process.env.NODE_ENV === 'development';
export const isProduction = process.env.NODE_ENV === 'production';
export const isPackaged = app.isPackaged;
```

### Graceful Shutdown

```typescript
async function gracefulShutdown() {
  console.log('Shutting down...');

  // Close database connections
  await database.close();

  // Kill Python processes
  pythonRunner.killAll();

  // Wait for cleanup
  await new Promise(resolve => setTimeout(resolve, 100));

  app.quit();
}

app.on('before-quit', gracefulShutdown);
```

### Error Recovery

```typescript
// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  // Log to file
  logger.error('Uncaught exception', error);
  // Optionally show dialog to user
  dialog.showErrorBox('Unexpected Error', error.message);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled rejection at:', promise, 'reason:', reason);
  logger.error('Unhandled rejection', { reason });
});
```

## Additional Resources

For detailed examples and resource files, see:
- `PROJECT_KNOWLEDGE.md` - Application architecture details
- `TROUBLESHOOTING.md` - Common Electron issues and solutions
- Official Electron Security Guide: https://www.electronjs.org/docs/latest/tutorial/security
