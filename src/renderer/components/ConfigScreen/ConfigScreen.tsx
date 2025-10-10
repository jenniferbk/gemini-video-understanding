import React, { useState, useEffect, useCallback } from 'react';
import styles from './ConfigScreen.module.css';
import { Button } from '../shared/Button';
import { Select } from '../shared/Select';
import { PromptManager } from '../PromptManager/PromptManager';
import { About } from '../About/About';

interface Prompt {
  id: string;
  name: string;
  description: string;
  prompt_text: string;
}

interface VideoInfo {
  path: string;
  filename: string;
  sizeInMB: number;
  durationMinutes?: number;
}

interface TranscriptionConfig {
  videoPath: string;
  prompt: string;
  consensusRuns: number;
  chunkMinutes: number;
  vadEnabled: boolean;
  denoisingEnabled: boolean;
  outputPath: string;
  apiKey: string;
}

interface ConfigScreenProps {
  videoInfo: VideoInfo;
  onBack: () => void;
  onStart: (config: TranscriptionConfig) => void;
  onOpenSettings: () => void;
}

type QualityPreset = 'quick' | 'standard' | 'high';

const QUALITY_PRESETS = {
  quick: {
    consensusRuns: 1,
    chunkMinutes: 3,
    vadEnabled: false,
    denoisingEnabled: false,
    estimatedMinutes: 30
  },
  standard: {
    consensusRuns: 3,
    chunkMinutes: 2,
    vadEnabled: true,
    denoisingEnabled: true,
    estimatedMinutes: 90
  },
  high: {
    consensusRuns: 5,
    chunkMinutes: 2,
    vadEnabled: true,
    denoisingEnabled: true,
    estimatedMinutes: 150
  }
};

export const ConfigScreen: React.FC<ConfigScreenProps> = ({
  videoInfo,
  onBack,
  onStart,
  onOpenSettings
}) => {
  // Load prompts dynamically from API
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState<string>('');
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>('standard');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showPromptManager, setShowPromptManager] = useState(false);
  const [showAbout, setShowAbout] = useState(false);

  // Advanced settings
  const [consensusRuns, setConsensusRuns] = useState(3);
  const [chunkMinutes, setChunkMinutes] = useState(2);
  const [vadEnabled, setVadEnabled] = useState(true);
  const [denoisingEnabled, setDenoisingEnabled] = useState(true);

  // Load prompts and check API key on mount
  useEffect(() => {
    async function loadData() {
      try {
        // Load prompts
        const loadedPrompts = await window.electronAPI.getPrompts();
        setPrompts(loadedPrompts);

        // Set default prompt if available
        if (loadedPrompts.length > 0 && !selectedPromptId) {
          // Try to find "smallgroup_jake" or just use first prompt
          const defaultPrompt = loadedPrompts.find(p => p.id === 'smallgroup_jake') || loadedPrompts[0];
          setSelectedPromptId(defaultPrompt.id);
        }

        // Check API key
        const { exists } = await window.electronAPI.hasApiKey();
        setHasApiKey(exists);
      } catch (error) {
        console.error('Failed to load data:', error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Update settings when preset changes
  useEffect(() => {
    const preset = QUALITY_PRESETS[qualityPreset];
    setConsensusRuns(preset.consensusRuns);
    setChunkMinutes(preset.chunkMinutes);
    setVadEnabled(preset.vadEnabled);
    setDenoisingEnabled(preset.denoisingEnabled);
  }, [qualityPreset]);

  // Calculate estimated cost and time
  const estimatedCost = ((videoInfo.sizeInMB / 1024) * consensusRuns * 0.15).toFixed(2);

  // Calculate dynamic processing time based on actual video length and settings
  const calculateProcessingTime = () => {
    if (!videoInfo.durationMinutes) {
      // Fallback to preset if duration not available
      return QUALITY_PRESETS[qualityPreset].estimatedMinutes;
    }

    const numChunks = Math.ceil(videoInfo.durationMinutes / chunkMinutes);

    // Time estimates per chunk (in minutes):
    // - Chunking overhead: ~0.5 min per chunk
    // - VAD processing: ~1 min per chunk (if enabled)
    // - Denoising: ~0.5 min per chunk (if enabled)
    // - Gemini API call: ~2 min per chunk per consensus run
    // - Consensus analysis: ~0.5 min per chunk (if multiple runs)

    let timePerChunk = 0.5; // Base chunking overhead
    if (vadEnabled) timePerChunk += 1;
    if (denoisingEnabled) timePerChunk += 0.5;
    timePerChunk += (2 * consensusRuns); // API calls
    if (consensusRuns > 1) timePerChunk += 0.5; // Consensus analysis

    return Math.round(numChunks * timePerChunk);
  };

  const estimatedMinutes = calculateProcessingTime();

  // Function to reload prompts (called when PromptManager closes)
  const reloadPrompts = useCallback(async () => {
    try {
      const loadedPrompts = await window.electronAPI.getPrompts();
      setPrompts(loadedPrompts);

      // If current selection is no longer valid, select first prompt
      if (selectedPromptId && !loadedPrompts.find(p => p.id === selectedPromptId)) {
        if (loadedPrompts.length > 0) {
          setSelectedPromptId(loadedPrompts[0].id);
        } else {
          setSelectedPromptId('');
        }
      }
    } catch (error) {
      console.error('Failed to reload prompts:', error);
    }
  }, [selectedPromptId]);

  const handleStart = useCallback(async () => {
    if (!hasApiKey) {
      alert('Please configure your Gemini API key in Settings first.');
      return;
    }

    if (!selectedPromptId) {
      alert('Please select a prompt.');
      return;
    }

    try {
      const { apiKey } = await window.electronAPI.getApiKey();
      if (!apiKey) {
        alert('Failed to retrieve API key. Please check Settings.');
        return;
      }

      const { path: outputPath } = await window.electronAPI.getOutputPath();

      const config: TranscriptionConfig = {
        videoPath: videoInfo.path,
        prompt: selectedPromptId,
        consensusRuns,
        chunkMinutes,
        vadEnabled,
        denoisingEnabled,
        outputPath,
        apiKey
      };

      onStart(config);
    } catch (error: any) {
      console.error('Failed to start transcription:', error);
      alert('Failed to start transcription: ' + error.message);
    }
  }, [
    hasApiKey,
    selectedPromptId,
    videoInfo.path,
    consensusRuns,
    chunkMinutes,
    vadEnabled,
    denoisingEnabled,
    onStart
  ]);

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading configuration...</div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>Configure Transcription</h1>
        <div className={styles.headerButtons}>
          <button className={styles.settingsButton} onClick={() => setShowAbout(true)} title="About">
            ℹ️
          </button>
          <button className={styles.settingsButton} onClick={onOpenSettings} title="Settings">
            ⚙️
          </button>
        </div>
      </div>

      <div className={styles.content}>
        {/* Video Info */}
        <div className={styles.videoInfo}>
          <div className={styles.infoRow}>
            <span className={styles.label}>Video:</span>
            <span className={styles.value}>{videoInfo.filename}</span>
          </div>
          <div className={styles.infoRow}>
            <span className={styles.label}>Size:</span>
            <span className={styles.value}>{videoInfo.sizeInMB.toLocaleString()} MB</span>
          </div>
          {videoInfo.durationMinutes && (
            <div className={styles.infoRow}>
              <span className={styles.label}>Duration:</span>
              <span className={styles.value}>
                {Math.floor(videoInfo.durationMinutes)} min {Math.round((videoInfo.durationMinutes % 1) * 60)} sec
              </span>
            </div>
          )}
        </div>

        {/* API Key Warning */}
        {!hasApiKey && (
          <div className={styles.warning} onClick={onOpenSettings} style={{ cursor: 'pointer' }}>
            <span className={styles.warningIcon}>⚠️</span>
            <span>No API key configured. Click here to add your Gemini API key in Settings.</span>
          </div>
        )}

        {/* Prompt Selection */}
        <div className={styles.section}>
          <div className={styles.promptHeader}>
            <label className={styles.sectionLabel}>Select Prompt</label>
            <button className={styles.managePromptsButton} onClick={() => setShowPromptManager(true)} title="Manage Prompts">
              📝 Manage Prompts
            </button>
          </div>
          <Select
            value={selectedPromptId}
            onChange={(value) => setSelectedPromptId(value)}
            options={prompts.map(p => ({ value: p.id, label: p.name }))}
            placeholder="Choose a prompt..."
          />
          {selectedPromptId && (
            <p className={styles.promptDescription}>
              {prompts.find(p => p.id === selectedPromptId)?.description}
            </p>
          )}
        </div>

        {/* Quality Presets */}
        <div className={styles.section}>
          <label className={styles.sectionLabel}>Quality Preset</label>
          <div className={styles.presets}>
            <label className={styles.preset}>
              <input
                type="radio"
                name="preset"
                value="quick"
                checked={qualityPreset === 'quick'}
                onChange={() => setQualityPreset('quick')}
              />
              <div className={styles.presetContent}>
                <span className={styles.presetName}>Quick</span>
                <span className={styles.presetTime}>~30 min</span>
              </div>
            </label>
            <label className={styles.preset}>
              <input
                type="radio"
                name="preset"
                value="standard"
                checked={qualityPreset === 'standard'}
                onChange={() => setQualityPreset('standard')}
              />
              <div className={styles.presetContent}>
                <span className={styles.presetName}>Standard</span>
                <span className={styles.presetTime}>~90 min</span>
              </div>
            </label>
            <label className={styles.preset}>
              <input
                type="radio"
                name="preset"
                value="high"
                checked={qualityPreset === 'high'}
                onChange={() => setQualityPreset('high')}
              />
              <div className={styles.presetContent}>
                <span className={styles.presetName}>High Quality</span>
                <span className={styles.presetTime}>~150 min</span>
              </div>
            </label>
          </div>
        </div>

        {/* Advanced Settings */}
        <div className={styles.section}>
          <button
            className={styles.advancedToggle}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '▼' : '▶'} Advanced Settings
          </button>

          {showAdvanced && (
            <div className={styles.advanced}>
              <div className={styles.advancedRow}>
                <label className={styles.advancedLabel}>
                  Chunk Duration (minutes):
                  <input
                    type="number"
                    min="1"
                    max="5"
                    value={chunkMinutes}
                    onChange={(e) => setChunkMinutes(parseInt(e.target.value))}
                    className={styles.numberInput}
                  />
                </label>
              </div>
              <div className={styles.advancedRow}>
                <label className={styles.advancedLabel}>
                  Consensus Runs:
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={consensusRuns}
                    onChange={(e) => setConsensusRuns(parseInt(e.target.value))}
                    className={styles.numberInput}
                  />
                </label>
              </div>
              <div className={styles.advancedRow}>
                <label className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={vadEnabled}
                    onChange={(e) => setVadEnabled(e.target.checked)}
                  />
                  VAD Preprocessing
                </label>
              </div>
              <div className={styles.advancedRow}>
                <label className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={denoisingEnabled}
                    onChange={(e) => setDenoisingEnabled(e.target.checked)}
                  />
                  Audio Denoising
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Estimates */}
        <div className={styles.estimates}>
          <div className={styles.estimate}>
            <span className={styles.estimateLabel}>Estimated Cost:</span>
            <span className={styles.estimateValue}>${estimatedCost}</span>
          </div>
          <div className={styles.estimate}>
            <span className={styles.estimateLabel}>Processing Time:</span>
            <span className={styles.estimateValue}>~{estimatedMinutes} minutes</span>
          </div>
        </div>

        {/* Actions */}
        <div className={styles.actions}>
          <Button variant="secondary" onClick={onBack}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleStart} disabled={!hasApiKey || !selectedPromptId}>
            Start Transcription
          </Button>
        </div>
      </div>

      {/* Prompt Manager Modal */}
      {showPromptManager && (
        <PromptManager onClose={() => {
          setShowPromptManager(false);
          reloadPrompts(); // Reload prompts when manager closes
        }} />
      )}

      {/* About Modal */}
      {showAbout && <About onClose={() => setShowAbout(false)} />}
    </div>
  );
};
