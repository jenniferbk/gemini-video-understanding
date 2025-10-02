import { ipcMain, BrowserWindow, shell, dialog } from 'electron';
import { PythonTranscriptionRunner, TranscriptionConfig } from '../python/pythonRunner';
import { Database } from '../database/database';
import * as path from 'path';
import * as fs from 'fs';

let pythonRunner: PythonTranscriptionRunner | null = null;
let currentJobId: number | null = null;

/**
 * Initialize transcription IPC handlers
 */
export function setupTranscriptionHandlers(mainWindow: BrowserWindow, db: Database): void {

  // Start transcription
  ipcMain.handle('transcription:start', async (_event, config: TranscriptionConfig) => {
    try {
      console.log('📹 Starting transcription:', config.videoPath);

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

      console.log(`✅ Created job ${jobId} in database`);
      currentJobId = jobId;

      // Update status to processing
      await db.updateJobStatus(jobId, 'processing');

      // Create Python runner if needed
      if (!pythonRunner) {
        pythonRunner = new PythonTranscriptionRunner();
      }

      // Set up event listeners
      pythonRunner.on('progress', (progress) => {
        // Forward progress to renderer
        mainWindow.webContents.send('transcription:progress', {
          jobId,
          ...progress
        });
      });

      pythonRunner.on('log', (log) => {
        // Forward log to renderer
        mainWindow.webContents.send('transcription:log', {
          jobId,
          ...log
        });
      });

      pythonRunner.on('complete', async (completion) => {
        console.log('✅ Transcription complete:', completion);

        // Update database
        await db.updateJobOutput(jobId, completion.outputFile || '', completion.stats || {});

        // Notify renderer
        mainWindow.webContents.send('transcription:complete', {
          jobId,
          ...completion
        });

        currentJobId = null;
      });

      pythonRunner.on('error', async (error) => {
        console.error('❌ Transcription error:', error);

        // Update database
        await db.updateJobStatus(jobId, 'failed', error.message);

        // Notify renderer
        mainWindow.webContents.send('transcription:error', {
          jobId,
          ...error
        });

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

  // Cancel transcription
  ipcMain.handle('transcription:cancel', async () => {
    try {
      if (!pythonRunner || !pythonRunner.isRunning()) {
        throw new Error('No transcription is running');
      }

      console.log('🛑 Cancelling transcription...');

      // Cancel Python process
      pythonRunner.cancel();

      // Update database
      if (currentJobId) {
        await db.updateJobStatus(currentJobId, 'cancelled');
        currentJobId = null;
      }

      return { success: true };

    } catch (error: any) {
      console.error('Failed to cancel transcription:', error);
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

      // Open folder in Finder (macOS) or File Explorer (Windows/Linux)
      await shell.openPath(folderPath);

      return { success: true };
    } catch (error: any) {
      console.error('Failed to open folder:', error);
      throw error;
    }
  });

  // Select video file using native dialog
  ipcMain.handle('transcription:selectVideo', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Select Video File',
        properties: ['openFile'],
        filters: [
          { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] }
        ]
      });

      if (result.canceled || !result.filePaths[0]) {
        return { success: false, cancelled: true };
      }

      const videoPath = result.filePaths[0];
      const stats = fs.statSync(videoPath);
      const sizeInMB = stats.size / (1024 * 1024);

      // Rough estimate: assuming ~1MB per minute of video at standard quality
      // This is a placeholder until we implement proper ffprobe duration detection
      const estimatedDurationMinutes = sizeInMB / 15; // ~15MB per minute is more realistic for classroom videos

      return {
        success: true,
        path: videoPath,
        filename: path.basename(videoPath),
        sizeInMB: Math.round(sizeInMB * 100) / 100,
        durationMinutes: Math.round(estimatedDurationMinutes * 10) / 10 // Round to 1 decimal
      };
    } catch (error: any) {
      console.error('Failed to select video:', error);
      return { success: false, error: error.message };
    }
  });

  // Validate video file
  ipcMain.handle('transcription:validateVideo', async (_event, videoPath: string) => {
    try {
      // Check if file exists
      if (!fs.existsSync(videoPath)) {
        return { valid: false, error: 'File not found' };
      }

      // Check file extension
      const ext = path.extname(videoPath).toLowerCase();
      const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];

      if (!validExtensions.includes(ext)) {
        return {
          valid: false,
          error: `Invalid file type. Supported: ${validExtensions.join(', ')}`
        };
      }

      // Get file size
      const stats = fs.statSync(videoPath);
      const sizeInMB = stats.size / (1024 * 1024);

      // TODO: Get video duration using ffprobe
      // For now, just return basic info

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

      // Limit to first 100 lines for preview
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

  console.log('✅ Transcription IPC handlers registered');
}
