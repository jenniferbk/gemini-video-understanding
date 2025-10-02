import { ipcMain, dialog } from 'electron';
import { app } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import { v4 as uuidv4 } from 'uuid';

export interface Prompt {
  id: string;
  name: string;
  description: string;
  prompt_text: string;
  created_at: string;
  updated_at: string;
}

interface PromptsData {
  prompts: Prompt[];
}

/**
 * Get path to user's prompts file
 */
function getPromptsFilePath(): string {
  const userDataPath = app.getPath('userData');
  return path.join(userDataPath, 'prompts.json');
}

/**
 * Get path to default prompts (bundled with app)
 */
function getDefaultPromptsPath(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'python', 'prompts.json');
  } else {
    return path.join(app.getAppPath(), 'src', 'python', 'prompts.json');
  }
}

/**
 * Initialize prompts file (copy defaults if needed)
 */
function initializePromptsFile(): void {
  const promptsPath = getPromptsFilePath();

  if (!fs.existsSync(promptsPath)) {
    console.log('📝 Initializing prompts file from defaults...');

    try {
      const defaultPath = getDefaultPromptsPath();
      const defaultPrompts = JSON.parse(fs.readFileSync(defaultPath, 'utf-8'));

      // Convert old format to new format if needed
      if (defaultPrompts.prompts && Array.isArray(defaultPrompts.prompts)) {
        // Add IDs and timestamps to prompts that don't have them
        defaultPrompts.prompts = defaultPrompts.prompts.map((p: any) => ({
          id: p.id || uuidv4(),
          name: p.name,
          description: p.description || '',
          prompt_text: p.prompt_text || p.text || '',
          created_at: p.created_at || new Date().toISOString(),
          updated_at: p.updated_at || new Date().toISOString()
        }));
      }

      fs.writeFileSync(promptsPath, JSON.stringify(defaultPrompts, null, 2));
      console.log('✅ Prompts file initialized');
    } catch (error) {
      console.error('Failed to initialize prompts file:', error);
    }
  }
}

/**
 * Load prompts from file
 */
function loadPrompts(): PromptsData {
  const promptsPath = getPromptsFilePath();

  try {
    const data = fs.readFileSync(promptsPath, 'utf-8');
    const parsed = JSON.parse(data);

    // Handle old format: { "promptName": { name, description, prompt } }
    if (!parsed.prompts && typeof parsed === 'object') {
      console.log('📦 Converting old prompts format to new format...');
      const convertedPrompts: Prompt[] = Object.entries(parsed).map(([key, value]: [string, any]) => ({
        id: uuidv4(),
        name: value.name || key,
        description: value.description || '',
        prompt_text: value.prompt || value.prompt_text || '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }));

      const newFormat = { prompts: convertedPrompts };
      savePrompts(newFormat); // Save in new format
      return newFormat;
    }

    return parsed;
  } catch (error) {
    console.error('Failed to load prompts:', error);
    return { prompts: [] };
  }
}

/**
 * Save prompts to file
 */
function savePrompts(data: PromptsData): void {
  const promptsPath = getPromptsFilePath();

  try {
    fs.writeFileSync(promptsPath, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error('Failed to save prompts:', error);
    throw error;
  }
}

/**
 * Setup prompts IPC handlers
 */
export function setupPromptsHandlers(): void {
  // Initialize prompts file on first run
  initializePromptsFile();

  // List all prompts
  ipcMain.handle('prompts:list', async () => {
    try {
      const data = loadPrompts();
      return data.prompts;
    } catch (error: any) {
      console.error('Failed to list prompts:', error);
      throw error;
    }
  });

  // Get single prompt by ID
  ipcMain.handle('prompts:get', async (_event, id: string) => {
    try {
      const data = loadPrompts();
      const prompt = data.prompts.find(p => p.id === id);

      if (!prompt) {
        throw new Error(`Prompt not found: ${id}`);
      }

      return prompt;
    } catch (error: any) {
      console.error('Failed to get prompt:', error);
      throw error;
    }
  });

  // Save prompt (create or update)
  ipcMain.handle('prompts:save', async (_event, prompt: Prompt) => {
    try {
      const data = loadPrompts();
      const existingIndex = data.prompts.findIndex(p => p.id === prompt.id);

      // Update timestamp
      prompt.updated_at = new Date().toISOString();

      if (existingIndex >= 0) {
        // Update existing
        data.prompts[existingIndex] = prompt;
      } else {
        // Create new
        if (!prompt.id) {
          prompt.id = uuidv4();
        }
        if (!prompt.created_at) {
          prompt.created_at = new Date().toISOString();
        }
        data.prompts.push(prompt);
      }

      savePrompts(data);
      return prompt;
    } catch (error: any) {
      console.error('Failed to save prompt:', error);
      throw error;
    }
  });

  // Delete prompt
  ipcMain.handle('prompts:delete', async (_event, id: string) => {
    try {
      const data = loadPrompts();
      const initialLength = data.prompts.length;

      data.prompts = data.prompts.filter(p => p.id !== id);

      if (data.prompts.length === initialLength) {
        throw new Error(`Prompt not found: ${id}`);
      }

      savePrompts(data);
      return { success: true };
    } catch (error: any) {
      console.error('Failed to delete prompt:', error);
      throw error;
    }
  });

  // Duplicate prompt
  ipcMain.handle('prompts:duplicate', async (_event, id: string) => {
    try {
      const data = loadPrompts();
      const original = data.prompts.find(p => p.id === id);

      if (!original) {
        throw new Error(`Prompt not found: ${id}`);
      }

      const duplicate: Prompt = {
        ...original,
        id: uuidv4(),
        name: `${original.name} (copy)`,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };

      data.prompts.push(duplicate);
      savePrompts(data);

      return duplicate;
    } catch (error: any) {
      console.error('Failed to duplicate prompt:', error);
      throw error;
    }
  });

  // Import prompt from file
  ipcMain.handle('prompts:import', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Import Prompt',
        filters: [
          { name: 'JSON Files', extensions: ['json'] }
        ],
        properties: ['openFile']
      });

      if (result.canceled || !result.filePaths[0]) {
        return { success: false, cancelled: true };
      }

      const filePath = result.filePaths[0];
      const fileContent = fs.readFileSync(filePath, 'utf-8');
      const importedPrompt = JSON.parse(fileContent);

      // Validate prompt structure
      if (!importedPrompt.name || !importedPrompt.prompt_text) {
        throw new Error('Invalid prompt file format');
      }

      // Add to library
      const data = loadPrompts();
      const newPrompt: Prompt = {
        id: uuidv4(),
        name: importedPrompt.name,
        description: importedPrompt.description || '',
        prompt_text: importedPrompt.prompt_text,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };

      data.prompts.push(newPrompt);
      savePrompts(data);

      return { success: true, prompt: newPrompt };
    } catch (error: any) {
      console.error('Failed to import prompt:', error);
      throw error;
    }
  });

  // Export prompt to file
  ipcMain.handle('prompts:export', async (_event, id: string) => {
    try {
      const data = loadPrompts();
      const prompt = data.prompts.find(p => p.id === id);

      if (!prompt) {
        throw new Error(`Prompt not found: ${id}`);
      }

      const result = await dialog.showSaveDialog({
        title: 'Export Prompt',
        defaultPath: `${prompt.name}.json`,
        filters: [
          { name: 'JSON Files', extensions: ['json'] }
        ]
      });

      if (result.canceled || !result.filePath) {
        return { success: false, cancelled: true };
      }

      // Export without internal IDs/timestamps for portability
      const exportData = {
        name: prompt.name,
        description: prompt.description,
        prompt_text: prompt.prompt_text
      };

      fs.writeFileSync(result.filePath, JSON.stringify(exportData, null, 2));

      return { success: true, filePath: result.filePath };
    } catch (error: any) {
      console.error('Failed to export prompt:', error);
      throw error;
    }
  });

  console.log('✅ Prompts IPC handlers registered');
}
