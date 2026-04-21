import { ipcMain, BrowserWindow, shell, dialog } from 'electron';
import { PythonTranscriptionRunner, TranscriptionConfig, SpeakerDetectionConfig, SpeakerInfo } from '../python/pythonRunner';
import { Database } from '../database/database';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

let pythonRunner: PythonTranscriptionRunner | null = null;
let currentJobId: number | null = null;
let speakerManifestPath: string | null = null;

/**
 * Initialize transcription IPC handlers
 */
export function setupTranscriptionHandlers(mainWindow: BrowserWindow, db: Database): void {

  // Detect speakers (Phase 1)
  ipcMain.handle('transcription:detectSpeakers', async (_event, config: SpeakerDetectionConfig) => {
    try {
      console.log('Starting speaker detection:', config.videoPath);

      if (!fs.existsSync(config.videoPath)) {
        throw new Error(`Video file not found: ${config.videoPath}`);
      }

      if (!config.apiKey || config.apiKey.trim() === '') {
        throw new Error('API key is required');
      }

      // Create runner if needed
      if (pythonRunner) {
        pythonRunner.removeAllListeners();
      }
      pythonRunner = new PythonTranscriptionRunner();

      return new Promise<{ speakers: SpeakerInfo[] }>((resolve, reject) => {
        let resolved = false;

        pythonRunner!.on('speakers', (data) => {
          if (!resolved) {
            resolved = true;
            resolve({ speakers: data.speakers });
          }
        });

        pythonRunner!.on('progress', (progress) => {
          mainWindow.webContents.send('transcription:speakerProgress', progress);
        });

        pythonRunner!.on('log', (log) => {
          mainWindow.webContents.send('transcription:log', log);
        });

        pythonRunner!.on('error', (error) => {
          console.error('Speaker detection error:', error);
          if (!resolved) {
            resolved = true;
            reject(new Error(error.message || 'Speaker detection failed'));
          }
        });

        pythonRunner!.on('complete', () => {
          // If we haven't received speakers by complete, something went wrong
          if (!resolved) {
            resolved = true;
            reject(new Error('Speaker detection completed without returning speakers'));
          }
        });

        pythonRunner!.detectSpeakers(config);
      });

    } catch (error: any) {
      console.error('Failed to detect speakers:', error);
      throw error;
    }
  });

  // Save speaker manifest (between Phase 1 and Phase 2)
  ipcMain.handle('transcription:saveSpeakerManifest', async (_event, speakers: SpeakerInfo[]) => {
    try {
      const tempFile = path.join(os.tmpdir(), `gvu-speakers-${Date.now()}.json`);
      const manifestData = speakers.map(s => ({
        label: s.label,
        description: s.description,
        type: s.type
      }));
      fs.writeFileSync(tempFile, JSON.stringify(manifestData, null, 2));
      speakerManifestPath = tempFile;
      console.log(`Speaker manifest saved: ${tempFile}`);
      return { success: true, path: tempFile };
    } catch (error: any) {
      console.error('Failed to save speaker manifest:', error);
      throw error;
    }
  });

  // Start transcription (Phase 2)
  ipcMain.handle('transcription:start', async (_event, config: TranscriptionConfig) => {
    try {
      console.log('Starting transcription:', config.videoPath);

      // Validate video file exists
      if (!fs.existsSync(config.videoPath)) {
        throw new Error(`Video file not found: ${config.videoPath}`);
      }

      // Validate API key
      if (!config.apiKey || config.apiKey.trim() === '') {
        throw new Error('API key is required');
      }

      // Ensure output directory exists
      if (!fs.existsSync(config.outputPath)) {
        fs.mkdirSync(config.outputPath, { recursive: true });
      }

      // Create job in database
      const jobId = await db.createJob({
        videoPath: config.videoPath,
        videoFilename: path.basename(config.videoPath),
        promptName: config.prompt,
        config: config
      });

      console.log(`Created job ${jobId} in database`);
      currentJobId = jobId;

      // Update status to processing
      await db.updateJobStatus(jobId, 'processing');

      // Create Python runner if needed
      if (pythonRunner) {
        pythonRunner.removeAllListeners();
      }
      pythonRunner = new PythonTranscriptionRunner();

      // Set up event listeners
      pythonRunner.on('progress', (progress) => {
        mainWindow.webContents.send('transcription:progress', {
          jobId,
          ...progress
        });
      });

      pythonRunner.on('log', (log) => {
        mainWindow.webContents.send('transcription:log', {
          jobId,
          ...log
        });
      });

      pythonRunner.on('complete', async (completion) => {
        console.log('Transcription complete:', completion);

        // Update database
        // Python sends snake_case `output_file`; accept either for safety.
        const outputFile = completion.output_file || completion.outputFile || '';
        await db.updateJobOutput(jobId, outputFile, completion.stats || {});

        // Notify renderer
        mainWindow.webContents.send('transcription:complete', {
          jobId,
          ...completion
        });

        // Clean up speaker manifest
        cleanupSpeakerManifest();
        currentJobId = null;
      });

      pythonRunner.on('error', async (error) => {
        console.error('Transcription error:', error);

        // Update database
        await db.updateJobStatus(jobId, 'failed', error.message);

        // Notify renderer
        mainWindow.webContents.send('transcription:error', {
          jobId,
          ...error
        });

        // Clean up speaker manifest
        cleanupSpeakerManifest();
        currentJobId = null;
      });

      // Start Python process
      pythonRunner.start(config);

      return { success: true, jobId };

    } catch (error: any) {
      console.error('Failed to start transcription:', error);

      // Update job status if we created one
      if (currentJobId) {
        await db.updateJobStatus(currentJobId, 'failed', error.message);
        currentJobId = null;
      }

      throw error;
    }
  });

  // Cancel transcription or speaker detection
  ipcMain.handle('transcription:cancel', async () => {
    try {
      if (!pythonRunner || !pythonRunner.isRunning()) {
        throw new Error('No process is running');
      }

      console.log('Cancelling process...');

      pythonRunner.cancel();

      // Update database
      if (currentJobId) {
        await db.updateJobStatus(currentJobId, 'cancelled');
        currentJobId = null;
      }

      // Clean up speaker manifest
      cleanupSpeakerManifest();

      return { success: true };

    } catch (error: any) {
      console.error('Failed to cancel:', error);
      throw error;
    }
  });

  // Get transcription status
  ipcMain.handle('transcription:status', async () => {
    return {
      isRunning: pythonRunner?.isRunning() || false,
      jobId: currentJobId,
      processId: pythonRunner?.getProcessId()
    };
  });

  // Get job by ID
  ipcMain.handle('transcription:getJob', async (_event, jobId: number) => {
    try {
      const job = await db.getJob(jobId);
      return job;
    } catch (error: any) {
      console.error('Failed to get job:', error);
      throw error;
    }
  });

  // Get recent jobs
  ipcMain.handle('transcription:getRecentJobs', async (_event, limit: number = 10) => {
    try {
      const jobs = await db.getRecentJobs(limit);
      return jobs;
    } catch (error: any) {
      console.error('Failed to get recent jobs:', error);
      throw error;
    }
  });

  // Open folder in Finder
  ipcMain.handle('transcription:openFolder', async (_event, folderPath: string) => {
    try {
      if (!fs.existsSync(folderPath)) {
        throw new Error(`Folder not found: ${folderPath}`);
      }

      await shell.openPath(folderPath);

      return { success: true };
    } catch (error: any) {
      console.error('Failed to open folder:', error);
      throw error;
    }
  });

  // Select video/audio file using native dialog
  ipcMain.handle('transcription:selectVideo', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Select Media File',
        properties: ['openFile'],
        filters: [
          { name: 'Media Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'] }
        ]
      });

      if (result.canceled || !result.filePaths[0]) {
        return { success: false, cancelled: true };
      }

      const videoPath = result.filePaths[0];
      const stats = fs.statSync(videoPath);
      const sizeInMB = stats.size / (1024 * 1024);

      const estimatedDurationMinutes = sizeInMB / 15;

      return {
        success: true,
        path: videoPath,
        filename: path.basename(videoPath),
        sizeInMB: Math.round(sizeInMB * 100) / 100,
        durationMinutes: Math.round(estimatedDurationMinutes * 10) / 10
      };
    } catch (error: any) {
      console.error('Failed to select video:', error);
      return { success: false, error: error.message };
    }
  });

  // Validate video/audio file
  ipcMain.handle('transcription:validateVideo', async (_event, videoPath: string) => {
    try {
      if (!fs.existsSync(videoPath)) {
        return { valid: false, error: 'File not found' };
      }

      const ext = path.extname(videoPath).toLowerCase();
      const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'];

      if (!validExtensions.includes(ext)) {
        return {
          valid: false,
          error: `Invalid file type. Supported: ${validExtensions.join(', ')}`
        };
      }

      const stats = fs.statSync(videoPath);
      const sizeInMB = stats.size / (1024 * 1024);

      return {
        valid: true,
        filename: path.basename(videoPath),
        sizeInMB: Math.round(sizeInMB * 100) / 100,
        extension: ext
      };

    } catch (error: any) {
      console.error('Failed to validate video:', error);
      return { valid: false, error: error.message };
    }
  });

  // Read transcript file
  ipcMain.handle('transcription:readTranscript', async (_event, filePath: string) => {
    try {
      if (!fs.existsSync(filePath)) {
        throw new Error(`Transcript file not found: ${filePath}`);
      }

      const content = fs.readFileSync(filePath, 'utf-8');

      const lines = content.split('\n');
      const preview = lines.slice(0, 100).join('\n');

      return {
        success: true,
        content: preview,
        totalLines: lines.length,
        truncated: lines.length > 100
      };
    } catch (error: any) {
      console.error('Failed to read transcript:', error);
      throw error;
    }
  });

  // Check whether transcript_name_map.json exists in the same directory as
  // a given transcript. Narrow by design — we don't want to expose generic
  // fs.existsSync to the renderer, and this is the only file we need to probe.
  ipcMain.handle('transcription:hasAuditFile', async (_event, transcriptPath: string) => {
    try {
      if (!transcriptPath || typeof transcriptPath !== 'string') {
        return { exists: false };
      }
      const auditPath = path.join(path.dirname(transcriptPath), 'transcript_name_map.json');
      return { exists: fs.existsSync(auditPath), path: auditPath };
    } catch (error) {
      console.error('hasAuditFile check failed:', error);
      return { exists: false };
    }
  });

  console.log('Transcription IPC handlers registered');
}

/**
 * Clean up temporary speaker manifest file
 */
function cleanupSpeakerManifest(): void {
  if (speakerManifestPath && fs.existsSync(speakerManifestPath)) {
    try {
      fs.unlinkSync(speakerManifestPath);
      console.log(`Cleaned up speaker manifest: ${speakerManifestPath}`);
    } catch (error) {
      console.error('Failed to clean up speaker manifest:', error);
    }
    speakerManifestPath = null;
  }
}
