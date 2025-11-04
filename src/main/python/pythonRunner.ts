import { spawn, ChildProcess } from 'child_process';
import { app } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
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
  timestamp?: string;
}

export class PythonTranscriptionRunner extends EventEmitter {
  private process: ChildProcess | null = null;
  private pythonPath: string;
  private scriptPath: string;
  private tempPromptsFile: string | null = null;

  constructor() {
    super();
    this.pythonPath = this.getPythonPath();
    this.scriptPath = this.getScriptPath();
  }

  /**
   * Get path to Python executable
   * In development: uses system Python or venv
   * In production: uses bundled Python
   */
  private getPythonPath(): string {
    if (app.isPackaged) {
      // Production: bundled Python in resources
      return path.join(process.resourcesPath, 'python', 'bin', 'python3');
    } else {
      // Development: use project venv
      return path.join(app.getAppPath(), 'venv', 'bin', 'python3');
    }
  }

  /**
   * Get path to Python transcription script
   */
  private getScriptPath(): string {
    if (app.isPackaged) {
      // Production: scripts are in python/scripts subdirectory
      return path.join(
        process.resourcesPath,
        'python',
        'scripts',
        'video_transcription_pipeline_v04.py'
      );
    } else {
      // Development
      return path.join(
        app.getAppPath(),
        'src',
        'python',
        'video_transcription_pipeline_v04.py'
      );
    }
  }

  /**
   * Get path to bundled FFmpeg binary
   */
  private getFFmpegPath(): string {
    if (app.isPackaged) {
      // Production: bundled in resources/bin
      return path.join(process.resourcesPath, 'bin', 'ffmpeg');
    } else {
      // Development: use system ffmpeg or bundled binaries
      const devBundledPath = path.join(app.getAppPath(), 'binaries', 'macos-arm64', 'ffmpeg');
      return devBundledPath;
    }
  }

  /**
   * Get path to bundled FFprobe binary
   */
  private getFFprobePath(): string {
    if (app.isPackaged) {
      // Production: bundled in resources/bin
      return path.join(process.resourcesPath, 'bin', 'ffprobe');
    } else {
      // Development: use system ffprobe or bundled binaries
      const devBundledPath = path.join(app.getAppPath(), 'binaries', 'macos-arm64', 'ffprobe');
      return devBundledPath;
    }
  }

  /**
   * Get path to user's prompts file
   */
  private getUserPromptsPath(): string {
    const userDataPath = app.getPath('userData');
    return path.join(userDataPath, 'prompts.json');
  }

  /**
   * Convert Electron prompts format to Python format and write to temp file
   * Electron format: {"prompts": [{"id": "...", "name": "...", "prompt_text": "..."}]}
   * Python format: {"name": {"name": "...", "description": "...", "prompt": "..."}}
   */
  private convertAndWritePrompts(): string {
    const userPromptsPath = this.getUserPromptsPath();

    // Check if user prompts file exists
    if (!fs.existsSync(userPromptsPath)) {
      console.log('⚠️  User prompts file not found, Python will use bundled prompts');
      return '';
    }

    try {
      // Read user prompts
      const electronPrompts = JSON.parse(fs.readFileSync(userPromptsPath, 'utf-8'));

      if (!electronPrompts.prompts || !Array.isArray(electronPrompts.prompts)) {
        console.log('⚠️  Invalid prompts format, Python will use bundled prompts');
        return '';
      }

      // Convert to Python format
      const pythonPrompts: any = {};
      for (const prompt of electronPrompts.prompts) {
        const key = prompt.name.toLowerCase().replace(/\s+/g, '_');
        pythonPrompts[key] = {
          id: prompt.id,  // Include UUID for Python's UUID matching
          uuid: prompt.id,  // Alternate field name for compatibility
          name: prompt.name,
          description: prompt.description || '',
          prompt: prompt.prompt_text
        };
      }

      // Write to temp file
      const tempFile = path.join(os.tmpdir(), `gvu-prompts-${Date.now()}.json`);
      fs.writeFileSync(tempFile, JSON.stringify(pythonPrompts, null, 2));

      console.log(`✅ Converted prompts written to: ${tempFile}`);
      return tempFile;
    } catch (error) {
      console.error('Failed to convert prompts:', error);
      return '';
    }
  }

  /**
   * Clean up temporary prompts file
   */
  private cleanupTempPromptsFile(): void {
    if (this.tempPromptsFile && fs.existsSync(this.tempPromptsFile)) {
      try {
        fs.unlinkSync(this.tempPromptsFile);
        console.log(`🗑️  Cleaned up temp prompts file: ${this.tempPromptsFile}`);
      } catch (error) {
        console.error('Failed to delete temp prompts file:', error);
      }
      this.tempPromptsFile = null;
    }
  }

  /**
   * Start transcription process
   */
  start(config: TranscriptionConfig): void {
    if (this.process) {
      throw new Error('Transcription already running');
    }

    // Convert and write user prompts to temp file
    this.tempPromptsFile = this.convertAndWritePrompts();

    // Build command arguments
    const args = [
      this.scriptPath,
      config.videoPath,
      '--prompt',
      config.prompt,
      '--consensus-runs',
      config.consensusRuns.toString(),
      '--chunk-minutes',
      config.chunkMinutes.toString(),
      '--output',
      config.outputPath,
      '--api-key',
      config.apiKey,
      '--json-progress' // Enable JSON output for Electron
    ];

    // Add prompts file if successfully created
    if (this.tempPromptsFile) {
      args.push('--prompts-file', this.tempPromptsFile);
    }

    // Add optional flags
    if (!config.vadEnabled) {
      args.push('--no-vad');
    }
    if (!config.denoisingEnabled) {
      args.push('--no-denoise');
    }

    console.log('🚀 Starting Python transcription:', {
      python: this.pythonPath,
      script: this.scriptPath,
      video: config.videoPath
    });

    // Spawn Python process with FFmpeg paths in environment
    this.process = spawn(this.pythonPath, args, {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        FFMPEG_PATH: this.getFFmpegPath(),
        FFPROBE_PATH: this.getFFprobePath()
      },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    // Handle stdout (JSON progress)
    this.process.stdout?.on('data', (data) => {
      this.handleStdout(data.toString());
    });

    // Handle stderr (errors and warnings)
    this.process.stderr?.on('data', (data) => {
      const message = data.toString().trim();
      if (message) {
        this.emit('log', {
          type: 'log',
          level: 'error',
          message: message
        });
      }
    });

    // Handle process exit
    this.process.on('exit', (code, signal) => {
      console.log(`Python process exited: code=${code}, signal=${signal}`);

      if (code !== 0 && code !== null) {
        this.emit('error', {
          type: 'error',
          message: `Process exited with code ${code}`,
          fatal: true
        });
      }

      // Clean up temp prompts file
      this.cleanupTempPromptsFile();

      this.process = null;
    });

    // Handle process errors
    this.process.on('error', (err) => {
      console.error('Python process error:', err);
      this.emit('error', {
        type: 'error',
        message: `Failed to start Python process: ${err.message}`,
        fatal: true
      });
      this.process = null;
    });
  }

  /**
   * Parse stdout for JSON progress updates
   */
  private handleStdout(data: string): void {
    const lines = data.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Parse our JSON protocol
      if (trimmed.startsWith('GVU_PROGRESS:')) {
        const json = trimmed.substring(13);
        try {
          const progress: ProgressUpdate = JSON.parse(json);
          this.emit('progress', progress);
        } catch (e) {
          console.error('Failed to parse progress JSON:', json, e);
        }
      } else if (trimmed.startsWith('GVU_COMPLETE:')) {
        const json = trimmed.substring(13);
        try {
          const completion: ProgressUpdate = JSON.parse(json);
          this.emit('complete', completion);
        } catch (e) {
          console.error('Failed to parse completion JSON:', json, e);
        }
      } else if (trimmed.startsWith('GVU_ERROR:')) {
        const json = trimmed.substring(10);
        try {
          const error: ProgressUpdate = JSON.parse(json);
          this.emit('error', error);
        } catch (e) {
          console.error('Failed to parse error JSON:', json, e);
        }
      } else if (trimmed.startsWith('GVU_LOG:')) {
        const json = trimmed.substring(8);
        try {
          const log: ProgressUpdate = JSON.parse(json);
          this.emit('log', log);
        } catch (e) {
          console.error('Failed to parse log JSON:', json, e);
        }
      } else {
        // Regular stdout (not JSON)
        // Emit as log for debugging
        this.emit('log', {
          type: 'log',
          level: 'info',
          message: trimmed
        });
      }
    }
  }

  /**
   * Cancel running transcription
   */
  cancel(): void {
    if (this.process) {
      console.log('Cancelling transcription process...');
      this.process.kill('SIGTERM');

      // Force kill after 5 seconds if needed
      setTimeout(() => {
        if (this.process) {
          console.log('Force killing transcription process');
          this.process.kill('SIGKILL');
        }
      }, 5000);

      this.process = null;
    }

    // Clean up temp prompts file
    this.cleanupTempPromptsFile();
  }

  /**
   * Check if transcription is running
   */
  isRunning(): boolean {
    return this.process !== null && !this.process.killed;
  }

  /**
   * Get current process ID (for debugging)
   */
  getProcessId(): number | undefined {
    return this.process?.pid;
  }
}
