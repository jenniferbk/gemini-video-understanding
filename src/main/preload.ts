import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';

// Expose safe API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // ===== TRANSCRIPTION =====

  startTranscription: (config: any) =>
    ipcRenderer.invoke('transcription:start', config),

  cancelTranscription: () =>
    ipcRenderer.invoke('transcription:cancel'),

  getTranscriptionStatus: () =>
    ipcRenderer.invoke('transcription:status'),

  getJob: (jobId: number) =>
    ipcRenderer.invoke('transcription:getJob', jobId),

  getRecentJobs: (limit?: number) =>
    ipcRenderer.invoke('transcription:getRecentJobs', limit),

  openFolder: (path: string) =>
    ipcRenderer.invoke('transcription:openFolder', path),

  validateVideo: (videoPath: string) =>
    ipcRenderer.invoke('transcription:validateVideo', videoPath),

  selectVideo: () =>
    ipcRenderer.invoke('transcription:selectVideo'),

  readTranscript: (filePath: string) =>
    ipcRenderer.invoke('transcription:readTranscript', filePath),

  // Get file path from dropped file (for drag-and-drop)
  getDroppedFilePath: (file: File) => {
    // In Electron, File objects from drag-drop events have a path property
    return (file as any).path || null;
  },

  // Transcription events (listen to progress updates)
  onProgress: (callback: (data: any) => void) => {
    const listener = (_event: IpcRendererEvent, data: any) => callback(data);
    ipcRenderer.on('transcription:progress', listener);
    return () => ipcRenderer.removeListener('transcription:progress', listener);
  },

  onLog: (callback: (data: any) => void) => {
    const listener = (_event: IpcRendererEvent, data: any) => callback(data);
    ipcRenderer.on('transcription:log', listener);
    return () => ipcRenderer.removeListener('transcription:log', listener);
  },

  onComplete: (callback: (data: any) => void) => {
    const listener = (_event: IpcRendererEvent, data: any) => callback(data);
    ipcRenderer.on('transcription:complete', listener);
    return () => ipcRenderer.removeListener('transcription:complete', listener);
  },

  onError: (callback: (data: any) => void) => {
    const listener = (_event: IpcRendererEvent, data: any) => callback(data);
    ipcRenderer.on('transcription:error', listener);
    return () => ipcRenderer.removeListener('transcription:error', listener);
  },

  // Combined listener for all transcription updates
  onTranscriptionProgress: (callback: (data: any) => void) => {
    const progressListener = (_event: IpcRendererEvent, data: any) => callback({...data, type: 'progress'});
    const logListener = (_event: IpcRendererEvent, data: any) => callback({...data, type: 'log'});
    const completeListener = (_event: IpcRendererEvent, data: any) => callback({...data, type: 'complete'});
    const errorListener = (_event: IpcRendererEvent, data: any) => callback({...data, type: 'error'});

    ipcRenderer.on('transcription:progress', progressListener);
    ipcRenderer.on('transcription:log', logListener);
    ipcRenderer.on('transcription:complete', completeListener);
    ipcRenderer.on('transcription:error', errorListener);

    return () => {
      ipcRenderer.removeListener('transcription:progress', progressListener);
      ipcRenderer.removeListener('transcription:log', logListener);
      ipcRenderer.removeListener('transcription:complete', completeListener);
      ipcRenderer.removeListener('transcription:error', errorListener);
    };
  },

  // ===== PROMPTS =====

  getPrompts: () =>
    ipcRenderer.invoke('prompts:list'),

  getPrompt: (id: string) =>
    ipcRenderer.invoke('prompts:get', id),

  savePrompt: (prompt: any) =>
    ipcRenderer.invoke('prompts:save', prompt),

  deletePrompt: (id: string) =>
    ipcRenderer.invoke('prompts:delete', id),

  duplicatePrompt: (id: string) =>
    ipcRenderer.invoke('prompts:duplicate', id),

  importPrompt: () =>
    ipcRenderer.invoke('prompts:import'),

  exportPrompt: (id: string) =>
    ipcRenderer.invoke('prompts:export', id),

  // ===== SETTINGS =====

  getApiKey: () =>
    ipcRenderer.invoke('settings:getApiKey'),

  saveApiKey: (apiKey: string) =>
    ipcRenderer.invoke('settings:saveApiKey', apiKey),

  deleteApiKey: () =>
    ipcRenderer.invoke('settings:deleteApiKey'),

  hasApiKey: () =>
    ipcRenderer.invoke('settings:hasApiKey'),

  getSetting: (key: string) =>
    ipcRenderer.invoke('settings:get', key),

  setSetting: (key: string, value: string) =>
    ipcRenderer.invoke('settings:set', key, value),

  getAllSettings: () =>
    ipcRenderer.invoke('settings:getAll'),

  selectOutputDirectory: () =>
    ipcRenderer.invoke('settings:selectOutputDirectory'),

  getOutputPath: () =>
    ipcRenderer.invoke('settings:getOutputPath'),

  // ===== UTILITY =====

  getVersion: () => process.versions.electron,

  getPlatform: () => process.platform
});

// Type definitions for TypeScript
export interface Prompt {
  id: string;
  name: string;
  description: string;
  prompt_text: string;
  created_at?: string;
  updated_at?: string;
}

export interface Job {
  id: number;
  video_path: string;
  video_filename: string;
  video_duration_minutes?: number;
  prompt_name: string;
  config_json: string;
  status: 'queued' | 'processing' | 'complete' | 'failed' | 'cancelled';
  output_path?: string;
  stats_json?: string;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

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

export interface ElectronAPI {
  // Transcription
  startTranscription: (config: TranscriptionConfig) => Promise<{success: boolean; jobId: number}>;
  cancelTranscription: () => Promise<{success: boolean}>;
  getTranscriptionStatus: () => Promise<{isRunning: boolean; jobId?: number; processId?: number}>;
  getJob: (jobId: number) => Promise<Job | null>;
  getRecentJobs: (limit?: number) => Promise<Job[]>;
  openFolder: (path: string) => Promise<{success: boolean}>;
  validateVideo: (videoPath: string) => Promise<{valid: boolean; filename?: string; sizeInMB?: number; error?: string}>;
  selectVideo: () => Promise<{success: boolean; path?: string; filename?: string; sizeInMB?: number; durationMinutes?: number; cancelled?: boolean; error?: string}>;
  readTranscript: (filePath: string) => Promise<{success: boolean; content: string; totalLines: number; truncated: boolean}>;
  getDroppedFilePath: (file: File) => string | null;
  onProgress: (callback: (data: any) => void) => () => void;
  onLog: (callback: (data: any) => void) => () => void;
  onComplete: (callback: (data: any) => void) => () => void;
  onError: (callback: (data: any) => void) => () => void;
  onTranscriptionProgress: (callback: (data: any) => void) => () => void;

  // Prompts
  getPrompts: () => Promise<Prompt[]>;
  getPrompt: (id: string) => Promise<Prompt>;
  savePrompt: (prompt: Prompt) => Promise<Prompt>;
  deletePrompt: (id: string) => Promise<{success: boolean}>;
  duplicatePrompt: (id: string) => Promise<Prompt>;
  importPrompt: () => Promise<{success: boolean; prompt?: Prompt; cancelled?: boolean}>;
  exportPrompt: (id: string) => Promise<{success: boolean; filePath?: string; cancelled?: boolean}>;

  // Settings
  getApiKey: () => Promise<{apiKey: string | null}>;
  saveApiKey: (apiKey: string) => Promise<{success: boolean}>;
  deleteApiKey: () => Promise<{success: boolean}>;
  hasApiKey: () => Promise<{exists: boolean}>;
  getSetting: (key: string) => Promise<{value: string | null}>;
  setSetting: (key: string, value: string) => Promise<{success: boolean}>;
  getAllSettings: () => Promise<Record<string, string>>;
  selectOutputDirectory: () => Promise<{success: boolean; path?: string; cancelled?: boolean}>;
  getOutputPath: () => Promise<{path: string}>;

  // Utility
  getVersion: () => string;
  getPlatform: () => string;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
