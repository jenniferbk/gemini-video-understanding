import { spawn, ChildProcess } from 'child_process';
import { app } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { EventEmitter } from 'events';

export interface SpeakerInfo {
  label: string;
  description: string;
  type: 'teacher' | 'student' | 'researcher';
}

export interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  thinkingBudget: number;
  outputPath: string;
  apiKey: string;
  speakersManifestPath?: string;
  audioOnly?: boolean;
  deidentifyNames?: boolean;
}

export interface SpeakerDetectionConfig {
  videoPath: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  apiKey: string;
  audioOnly?: boolean;
}

export interface ProgressUpdate {
  type: 'progress' | 'log' | 'error' | 'complete' | 'speakers';
  chunk?: number;
  total?: number;
  percent?: number;
  status?: string;
  message?: string;
  level?: 'info' | 'warning' | 'error';
  outputFile?: string;
  // Python emits snake_case (`output_file` in GVU_COMPLETE); kept alongside
  // `outputFile` for compatibility with any code that already reads the latter.
  output_file?: string;
  stats?: any;
  timestamp?: string;
  speakers?: SpeakerInfo[];
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
      // Development: project venv lives at src/python/venv per CLAUDE.md
      return path.join(app.getAppPath(), 'src', 'python', 'venv', 'bin', 'python3');
    }
  }

  /**
   * Get path to Python transcription script (V10)
   */
  private getScriptPath(): string {
    if (app.isPackaged) {
      return path.join(
        process.resourcesPath,
        'python',
        'scripts',
        'video_transcription_pipeline_v10.py'
      );
    } else {
      return path.join(
        app.getAppPath(),
        'src',
        'python',
        'video_transcription_pipeline_v10.py'
      );
    }
  }

  /**
   * Get path to bundled FFmpeg binary
   */
  private getFFmpegPath(): string {
    if (app.isPackaged) {
      return path.join(process.resourcesPath, 'bin', 'ffmpeg');
    } else {
      const devBundledPath = path.join(app.getAppPath(), 'binaries', 'macos-arm64', 'ffmpeg');
      return devBundledPath;
    }
  }

  /**
   * Get path to bundled FFprobe binary
   */
  private getFFprobePath(): string {
    if (app.isPackaged) {
      return path.join(process.resourcesPath, 'bin', 'ffprobe');
    } else {
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

    if (!fs.existsSync(userPromptsPath)) {
      console.log('Warning: User prompts file not found, Python will use bundled prompts');
      return '';
    }

    try {
      const electronPrompts = JSON.parse(fs.readFileSync(userPromptsPath, 'utf-8'));

      if (!electronPrompts.prompts || !Array.isArray(electronPrompts.prompts)) {
        console.log('Warning: Invalid prompts format, Python will use bundled prompts');
        return '';
      }

      const pythonPrompts: any = {};
      for (const prompt of electronPrompts.prompts) {
        const key = prompt.name.toLowerCase().replace(/\s+/g, '_');
        pythonPrompts[key] = {
          id: prompt.id,
          uuid: prompt.id,
          name: prompt.name,
          description: prompt.description || '',
          prompt: prompt.prompt_text
        };
      }

      const tempFile = path.join(os.tmpdir(), `gvu-prompts-${Date.now()}.json`);
      fs.writeFileSync(tempFile, JSON.stringify(pythonPrompts, null, 2));

      console.log(`Converted prompts written to: ${tempFile}`);
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
        console.log(`Cleaned up temp prompts file: ${this.tempPromptsFile}`);
      } catch (error) {
        console.error('Failed to delete temp prompts file:', error);
      }
      this.tempPromptsFile = null;
    }
  }

  /**
   * Detect speakers from video (Phase 1).
   * Spawns v10 `identify --json-speakers` — headless mode that skips the
   * interactive terminal editor and emits one `GVU_SPEAKERS:` line on stdout
   * for handleStdout to parse. Chunk-grid args are intentionally omitted:
   * identification uses v10's internal speaker_id_chunks constant, not the
   * transcription chunk grid.
   */
  detectSpeakers(config: SpeakerDetectionConfig): void {
    if (this.process) {
      throw new Error('A process is already running');
    }

    const args = [
      this.scriptPath,
      'identify',
      config.videoPath,
      '--json-speakers',
      '--api-key',
      config.apiKey,
      '-m',
      config.model,
      '--resolution',
      config.resolution,
      '--fps',
      config.fps.toString(),
    ];

    console.log('Starting speaker detection:', {
      python: this.pythonPath,
      script: this.scriptPath,
      video: config.videoPath
    });

    this.process = spawn(this.pythonPath, args, {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        FFMPEG_PATH: this.getFFmpegPath(),
        FFPROBE_PATH: this.getFFprobePath()
      },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    this.process.stdout?.on('data', (data) => {
      this.handleStdout(data.toString());
    });

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

    this.process.on('exit', (code, signal) => {
      console.log(`Speaker detection process exited: code=${code}, signal=${signal}`);

      if ((code !== null && code !== 0) || signal !== null) {
        this.emit('error', {
          type: 'error',
          message: signal !== null
            ? `Speaker detection terminated by signal ${signal}`
            : `Speaker detection exited with code ${code}`,
          fatal: true
        });
      }

      this.process = null;
    });

    this.process.on('error', (err) => {
      console.error('Speaker detection process error:', err);
      this.emit('error', {
        type: 'error',
        message: `Failed to start speaker detection: ${err.message}`,
        fatal: true
      });
      this.process = null;
    });
  }

  /**
   * Start transcription process (Phase 2 - V10 process subcommand)
   */
  start(config: TranscriptionConfig): void {
    if (this.process) {
      throw new Error('Transcription already running');
    }

    // Convert and write user prompts to temp file
    this.tempPromptsFile = this.convertAndWritePrompts();

    // Build V10 process subcommand arguments
    const args = [
      this.scriptPath,
      'process',
      config.videoPath,
      '--prompt',
      config.prompt,
      '-m',
      config.model,
      '--resolution',
      config.resolution,
      '--fps',
      config.fps.toString(),
      '--chunk-minutes',
      config.chunkMinutes.toString(),
      '--overlap',
      config.overlapSeconds.toString(),
      '--thinking-budget',
      config.thinkingBudget.toString(),
      '--output',
      config.outputPath,
      '--api-key',
      config.apiKey,
      '--json-progress',
      '--no-confirm',
    ];

    // Add speakers manifest if provided
    if (config.speakersManifestPath) {
      args.push('--speakers', config.speakersManifestPath);
    }

    // `--prompts-file` and `--audio-only` pushes intentionally removed:
    // neither flag exists in v10. See dev/active/electron-v10-integration/
    // v10-integration-gaps.md (Gap #2 + its --audio-only cousin). The UI
    // state and tempfile generation are left in place pending the Gap #2
    // decision (add flag to v10, remove the UI, or inline prompt text).

    // Always burn timestamps (ffmpeg clock overlay + per-chunk resume). Closes
    // up-to-14s intra-chunk clock drift; not a user-facing choice.
    args.push('--burn-timestamps');

    // Optional second Gemini pass that replaces real names with pseudonyms
    // and writes transcript_name_map.json. Off by default; toggled from the
    // ConfigScreen Advanced section.
    if (config.deidentifyNames) {
      args.push('--deidentify-names');
    }

    console.log('Starting V10 transcription:', {
      python: this.pythonPath,
      script: this.scriptPath,
      video: config.videoPath,
      model: config.model,
      resolution: config.resolution,
      fps: config.fps
    });

    // Spawn Python process
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

      if ((code !== null && code !== 0) || signal !== null) {
        this.emit('error', {
          type: 'error',
          message: signal !== null
            ? `Process terminated by signal ${signal}`
            : `Process exited with code ${code}`,
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
      } else if (trimmed.startsWith('GVU_SPEAKERS:')) {
        const json = trimmed.substring(13);
        try {
          const speakerData = JSON.parse(json);
          this.emit('speakers', {
            type: 'speakers',
            speakers: speakerData.speakers
          });
        } catch (e) {
          console.error('Failed to parse speakers JSON:', json, e);
        }
      } else {
        // Regular stdout - emit as log for debugging
        this.emit('log', {
          type: 'log',
          level: 'info',
          message: trimmed
        });
      }
    }
  }

  /**
   * Cancel running process (works for both detection and transcription)
   */
  cancel(): void {
    if (this.process) {
      console.log('Cancelling process...');
      this.process.kill('SIGTERM');

      // Force kill after 5 seconds if needed
      setTimeout(() => {
        if (this.process) {
          console.log('Force killing process');
          this.process.kill('SIGKILL');
        }
      }, 5000);

      this.process = null;
    }

    // Clean up temp prompts file
    this.cleanupTempPromptsFile();
  }

  /**
   * Check if a process is running
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
