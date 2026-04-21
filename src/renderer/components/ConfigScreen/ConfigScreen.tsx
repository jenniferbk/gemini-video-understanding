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

interface V10Config {
  videoPath: string;
  prompt: string;
  model: string;
  resolution: 'LOW' | 'MEDIUM' | 'HIGH';
  fps: number;
  chunkMinutes: number;
  overlapSeconds: number;
  thinkingBudget: number;
  audioOnly?: boolean;
  deidentifyNames?: boolean;
}

interface ConfigScreenProps {
  videoInfo: VideoInfo;
  onBack: () => void;
  onStart: (config: V10Config) => void;
  onOpenSettings: () => void;
}

type QualityPreset = 'fast' | 'standard' | 'detailed';

const QUALITY_PRESETS = {
  fast: {
    model: 'gemini-2.0-flash',
    resolution: 'MEDIUM' as const,
    fps: 1,
    chunkMinutes: 1.0,
    overlapSeconds: 15,
    thinkingBudget: 2048,
    label: 'Fast',
    description: 'Lower cost, good for clear audio'
  },
  standard: {
    model: 'gemini-3-flash-preview',
    resolution: 'HIGH' as const,
    fps: 2,
    chunkMinutes: 1.0,
    overlapSeconds: 15,
    thinkingBudget: 4096,
    label: 'Standard',
    description: 'Best balance of quality & cost'
  },
  detailed: {
    model: 'gemini-3-flash-preview',
    resolution: 'HIGH' as const,
    fps: 4,
    chunkMinutes: 1.0,
    overlapSeconds: 15,
    thinkingBudget: 4096,
    label: 'Detailed',
    description: 'Maximum detail'
  }
};

const MODEL_OPTIONS = [
  { value: 'gemini-3-flash-preview', label: 'Gemini 3 Flash (recommended)' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gemini-2.5-pro-preview-05-06', label: 'Gemini 2.5 Pro' },
];

const RESOLUTION_OPTIONS = [
  { value: 'LOW', label: 'Low' },
  { value: 'MEDIUM', label: 'Medium' },
  { value: 'HIGH', label: 'High (recommended)' },
];

export const ConfigScreen: React.FC<ConfigScreenProps> = ({
  videoInfo,
  onBack,
  onStart,
  onOpenSettings
}) => {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [selectedPromptId, setSelectedPromptId] = useState<string>('');
  const [qualityPreset, setQualityPreset] = useState<QualityPreset>('standard');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showPromptManager, setShowPromptManager] = useState(false);
  const [showAbout, setShowAbout] = useState(false);

  // V10 advanced settings
  const [model, setModel] = useState('gemini-3-flash-preview');
  const [resolution, setResolution] = useState<'LOW' | 'MEDIUM' | 'HIGH'>('HIGH');
  const [fps, setFps] = useState(2);
  const [chunkMinutes, setChunkMinutes] = useState(1.0);
  const [overlapSeconds, setOverlapSeconds] = useState(15);
  const [thinkingBudget, setThinkingBudget] = useState(4096);
  // Auto-detect audio-only mode from file extension
  const isAudioFile = /\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(videoInfo.filename);
  const [audioOnly, setAudioOnly] = useState(isAudioFile);
  // Off by default — privacy feature with cost implications. Not persisted
  // between sessions.
  const [deidentifyNames, setDeidentifyNames] = useState(false);

  // Load prompts and check API key on mount
  useEffect(() => {
    async function loadData() {
      try {
        const loadedPrompts = await window.electronAPI.getPrompts();
        setPrompts(loadedPrompts);

        if (loadedPrompts.length > 0 && !selectedPromptId) {
          const defaultPrompt = loadedPrompts.find(p => p.id === 'smallgroup_jake') || loadedPrompts[0];
          setSelectedPromptId(defaultPrompt.id);
        }

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
    setModel(preset.model);
    setResolution(preset.resolution);
    setFps(preset.fps);
    setChunkMinutes(preset.chunkMinutes);
    setOverlapSeconds(preset.overlapSeconds);
    setThinkingBudget(preset.thinkingBudget);
  }, [qualityPreset]);

  // Cost estimate: ~$0.19/hr at standard settings
  const calculateCostEstimate = () => {
    const durationMinutes = videoInfo.durationMinutes || (videoInfo.sizeInMB / 15);
    const durationHours = durationMinutes / 60;

    // Token-based estimate
    const tokensPerFrame: Record<string, Record<string, number>> = {
      'gemini-3-flash-preview': { LOW: 70, MEDIUM: 70, HIGH: 280 },
      'gemini-2.0-flash': { LOW: 70, MEDIUM: 70, HIGH: 70 },
      'gemini-2.5-pro-preview-05-06': { LOW: 256, MEDIUM: 256, HIGH: 256 },
    };

    const tpf = tokensPerFrame[model]?.[resolution] ?? 70;
    const framesPerChunk = audioOnly ? 0 : fps * chunkMinutes * 60;
    const videoTokensPerChunk = framesPerChunk * tpf;
    // Audio tokens: ~32 tokens/sec for audio
    const audioTokensPerChunk = chunkMinutes * 60 * 32;
    const numChunks = Math.ceil((durationMinutes * 60) / (chunkMinutes * 60 - overlapSeconds));
    const totalInputTokens = numChunks * (videoTokensPerChunk + audioTokensPerChunk + 2000); // +2000 for prompt
    const totalOutputTokens = numChunks * 4000; // ~4000 output tokens per chunk

    // Pricing (approximate, per million tokens)
    const inputPricePerMillion = model.includes('2.5-pro') ? 1.25 : 0.10;
    const outputPricePerMillion = model.includes('2.5-pro') ? 10.0 : 0.40;

    const inputCost = (totalInputTokens / 1_000_000) * inputPricePerMillion;
    const outputCost = (totalOutputTokens / 1_000_000) * outputPricePerMillion;

    return (inputCost + outputCost).toFixed(2);
  };

  const estimatedCost = calculateCostEstimate();

  // Processing time estimate
  const calculateProcessingTime = () => {
    const durationMinutes = videoInfo.durationMinutes || (videoInfo.sizeInMB / 15);
    const numChunks = Math.ceil((durationMinutes * 60) / (chunkMinutes * 60 - overlapSeconds));

    // ~2 min per chunk for API call + overhead
    const timePerChunk = 2.5;
    const speakerDetectionTime = 3; // ~3 min for speaker detection

    return Math.round(speakerDetectionTime + (numChunks * timePerChunk));
  };

  const estimatedMinutes = calculateProcessingTime();

  const reloadPrompts = useCallback(async () => {
    try {
      const loadedPrompts = await window.electronAPI.getPrompts();
      setPrompts(loadedPrompts);

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

    const config: V10Config = {
      videoPath: videoInfo.path,
      prompt: selectedPromptId,
      model,
      resolution,
      fps,
      chunkMinutes,
      overlapSeconds,
      thinkingBudget,
      audioOnly,
      deidentifyNames,
    };

    onStart(config);
  }, [
    hasApiKey,
    selectedPromptId,
    videoInfo.path,
    model,
    resolution,
    fps,
    chunkMinutes,
    overlapSeconds,
    thinkingBudget,
    audioOnly,
    deidentifyNames,
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
            <span className={styles.label}>File:</span>
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

        {/* Audio-Only Mode Toggle */}
        <div className={styles.section}>
          <label className={styles.audioOnlyToggle}>
            <input
              type="checkbox"
              checked={audioOnly}
              onChange={(e) => setAudioOnly(e.target.checked)}
            />
            <div className={styles.audioOnlyContent}>
              <span className={styles.audioOnlyLabel}>Audio-Only Mode</span>
              <span className={styles.audioOnlyDesc}>
                {audioOnly
                  ? 'Extracts audio only — speakers identified by voice characteristics, no visual descriptions'
                  : 'Uses both video frames and audio for transcription with visual descriptions'}
              </span>
            </div>
          </label>
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
            {(Object.keys(QUALITY_PRESETS) as QualityPreset[]).map((key) => (
              <label key={key} className={styles.preset}>
                <input
                  type="radio"
                  name="preset"
                  value={key}
                  checked={qualityPreset === key}
                  onChange={() => setQualityPreset(key)}
                />
                <div className={styles.presetContent}>
                  <span className={styles.presetName}>{QUALITY_PRESETS[key].label}</span>
                  <span className={styles.presetDesc}>{QUALITY_PRESETS[key].description}</span>
                </div>
              </label>
            ))}
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
                  Model:
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className={styles.selectInput}
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </label>
              </div>
              {!audioOnly && (
                <>
                  <div className={styles.advancedRow}>
                    <label className={styles.advancedLabel}>
                      Resolution:
                      <select
                        value={resolution}
                        onChange={(e) => setResolution(e.target.value as 'LOW' | 'MEDIUM' | 'HIGH')}
                        className={styles.selectInput}
                      >
                        {RESOLUTION_OPTIONS.map(opt => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className={styles.advancedRow}>
                    <label className={styles.advancedLabel}>
                      FPS (frames/sec):
                      <input
                        type="number"
                        min="1"
                        max="4"
                        value={fps}
                        onChange={(e) => setFps(parseInt(e.target.value) || 1)}
                        className={styles.numberInput}
                      />
                    </label>
                  </div>
                </>
              )}
              <div className={styles.advancedRow}>
                <label className={styles.advancedLabel}>
                  Chunk Duration (min):
                  <input
                    type="number"
                    min="0.5"
                    max="5"
                    step="0.5"
                    value={chunkMinutes}
                    onChange={(e) => setChunkMinutes(parseFloat(e.target.value) || 1.0)}
                    className={styles.numberInput}
                  />
                </label>
              </div>
              <div className={styles.advancedRow}>
                <label className={styles.advancedLabel}>
                  Overlap (sec):
                  <input
                    type="number"
                    min="0"
                    max="30"
                    value={overlapSeconds}
                    onChange={(e) => setOverlapSeconds(parseInt(e.target.value) || 0)}
                    className={styles.numberInput}
                  />
                </label>
              </div>
              <div className={styles.advancedRow}>
                <label className={styles.advancedLabel}>
                  Thinking Budget:
                  <input
                    type="number"
                    min="1024"
                    max="16384"
                    step="1024"
                    value={thinkingBudget}
                    onChange={(e) => setThinkingBudget(parseInt(e.target.value) || 4096)}
                    className={styles.numberInput}
                  />
                </label>
              </div>
              <div className={styles.advancedRow}>
                <label className={styles.audioOnlyToggle}>
                  <input
                    type="checkbox"
                    checked={deidentifyNames}
                    onChange={(e) => setDeidentifyNames(e.target.checked)}
                  />
                  <div className={styles.audioOnlyContent}>
                    <span className={styles.audioOnlyLabel}>De-identify student and adult names</span>
                    <span className={styles.audioOnlyDesc}>
                      Runs a second Gemini pass to replace real names with realistic pseudonyms
                      (e.g., Student-Hannah, Ms. Kelly). Writes an audit file
                      (transcript_name_map.json) next to the transcript — store this file under
                      separate access control. Adds processing time and API cost.
                    </span>
                  </div>
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
            Detect Speakers & Start
          </Button>
        </div>
      </div>

      {/* Prompt Manager Modal */}
      {showPromptManager && (
        <PromptManager onClose={() => {
          setShowPromptManager(false);
          reloadPrompts();
        }} />
      )}

      {/* About Modal */}
      {showAbout && <About onClose={() => setShowAbout(false)} />}
    </div>
  );
};
