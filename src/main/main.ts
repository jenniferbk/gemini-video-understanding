import { app, BrowserWindow } from 'electron';
import * as path from 'path';
import { Database } from './database/database';
import { setupTranscriptionHandlers } from './ipc/transcription';
import { setupPromptsHandlers } from './ipc/prompts';
import { setupSettingsHandlers } from './ipc/settings';

let mainWindow: BrowserWindow | null = null;
let database: Database | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    minWidth: 600,
    minHeight: 500,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    title: 'Gemini Video Understanding'
  });

  // Load the index.html from the dist directory
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // Open DevTools in development mode
  if (process.env.NODE_ENV !== 'production') {
    mainWindow.webContents.openDevTools();
  }

  // Handle file drop on the entire window
  mainWindow.webContents.on('will-navigate', (event, url) => {
    event.preventDefault();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  console.log('🚀 Electron app starting...');

  // Initialize database
  database = new Database();

  // Setup IPC handlers
  createWindow();

  if (mainWindow && database) {
    setupTranscriptionHandlers(mainWindow, database);
    setupPromptsHandlers();
    setupSettingsHandlers(database);
  }

  console.log('✅ App initialized');

  app.on('activate', () => {
    // On macOS, re-create window when dock icon is clicked and no windows are open
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Cleanup on quit
app.on('before-quit', async () => {
  console.log('🛑 App quitting, cleaning up...');
  if (database) {
    await database.close();
  }
});
