import { ipcMain, dialog, app } from 'electron';
import { Database } from '../database/database';
import * as path from 'path';
import * as fs from 'fs';

/**
 * API key storage interface
 * For macOS, we'll use a simple encrypted storage for now
 * TODO: Integrate with macOS Keychain using 'keytar' package
 */
class ApiKeyStorage {
  private keyPath: string;

  constructor() {
    const userDataPath = app.getPath('userData');
    this.keyPath = path.join(userDataPath, '.apikey');
  }

  /**
   * Save API key (basic encryption - will upgrade to Keychain)
   */
  async save(apiKey: string): Promise<void> {
    try {
      // Basic obfuscation (not secure - placeholder for Keychain)
      const encoded = Buffer.from(apiKey).toString('base64');
      fs.writeFileSync(this.keyPath, encoded, { mode: 0o600 });
    } catch (error) {
      console.error('Failed to save API key:', error);
      throw error;
    }
  }

  /**
   * Get API key
   */
  async get(): Promise<string | null> {
    try {
      if (!fs.existsSync(this.keyPath)) {
        return null;
      }

      const encoded = fs.readFileSync(this.keyPath, 'utf-8');
      const decoded = Buffer.from(encoded, 'base64').toString('utf-8');
      return decoded;
    } catch (error) {
      console.error('Failed to get API key:', error);
      return null;
    }
  }

  /**
   * Delete API key
   */
  async delete(): Promise<void> {
    try {
      if (fs.existsSync(this.keyPath)) {
        fs.unlinkSync(this.keyPath);
      }
    } catch (error) {
      console.error('Failed to delete API key:', error);
      throw error;
    }
  }

  /**
   * Check if API key exists
   */
  async exists(): Promise<boolean> {
    return fs.existsSync(this.keyPath);
  }
}

const apiKeyStorage = new ApiKeyStorage();

/**
 * Setup settings IPC handlers
 */
export function setupSettingsHandlers(db: Database): void {

  // Get API key
  ipcMain.handle('settings:getApiKey', async () => {
    try {
      const apiKey = await apiKeyStorage.get();
      return { apiKey };
    } catch (error: any) {
      console.error('Failed to get API key:', error);
      throw error;
    }
  });

  // Save API key
  ipcMain.handle('settings:saveApiKey', async (_event, apiKey: string) => {
    try {
      if (!apiKey || apiKey.trim() === '') {
        throw new Error('API key cannot be empty');
      }

      await apiKeyStorage.save(apiKey.trim());
      return { success: true };
    } catch (error: any) {
      console.error('Failed to save API key:', error);
      throw error;
    }
  });

  // Delete API key
  ipcMain.handle('settings:deleteApiKey', async () => {
    try {
      await apiKeyStorage.delete();
      return { success: true };
    } catch (error: any) {
      console.error('Failed to delete API key:', error);
      throw error;
    }
  });

  // Check if API key exists
  ipcMain.handle('settings:hasApiKey', async () => {
    try {
      const exists = await apiKeyStorage.exists();
      return { exists };
    } catch (error: any) {
      console.error('Failed to check API key:', error);
      return { exists: false };
    }
  });

  // Get setting
  ipcMain.handle('settings:get', async (_event, key: string) => {
    try {
      const value = await db.getSetting(key);
      return { value };
    } catch (error: any) {
      console.error('Failed to get setting:', error);
      throw error;
    }
  });

  // Set setting
  ipcMain.handle('settings:set', async (_event, key: string, value: string) => {
    try {
      await db.setSetting(key, value);
      return { success: true };
    } catch (error: any) {
      console.error('Failed to set setting:', error);
      throw error;
    }
  });

  // Get all settings
  ipcMain.handle('settings:getAll', async () => {
    try {
      const settings = await db.getAllSettings();
      return settings;
    } catch (error: any) {
      console.error('Failed to get all settings:', error);
      throw error;
    }
  });

  // Select output directory
  ipcMain.handle('settings:selectOutputDirectory', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Select Output Directory',
        properties: ['openDirectory', 'createDirectory']
      });

      if (result.canceled || !result.filePaths[0]) {
        return { success: false, cancelled: true };
      }

      const selectedPath = result.filePaths[0];

      // Save to settings
      await db.setSetting('default_output_path', selectedPath);

      return { success: true, path: selectedPath };
    } catch (error: any) {
      console.error('Failed to select output directory:', error);
      throw error;
    }
  });

  // Get default output path (expand ~ to home directory)
  ipcMain.handle('settings:getOutputPath', async () => {
    try {
      let outputPath = await db.getSetting('default_output_path');

      if (!outputPath) {
        outputPath = '~/Documents/VideoTranscripts';
      }

      // Expand ~ to home directory
      if (outputPath.startsWith('~')) {
        const homeDir = app.getPath('home');
        outputPath = path.join(homeDir, outputPath.slice(1));
      }

      // Ensure directory exists
      if (!fs.existsSync(outputPath)) {
        fs.mkdirSync(outputPath, { recursive: true });
      }

      return { path: outputPath };
    } catch (error: any) {
      console.error('Failed to get output path:', error);
      throw error;
    }
  });

  console.log('✅ Settings IPC handlers registered');
}
