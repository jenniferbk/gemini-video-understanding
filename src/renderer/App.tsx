import React, { useState, useEffect } from 'react';
import { VideoUpload } from './components/VideoUpload/VideoUpload';
import { ConfigScreen } from './components/ConfigScreen/ConfigScreen';
import { SpeakerReview, SpeakerInfo } from './components/SpeakerReview/SpeakerReview';
import { Settings } from './components/Settings/Settings';
import { ProgressScreen } from './components/ProgressScreen/ProgressScreen';
import { ResultsScreen } from './components/ResultsScreen/ResultsScreen';

type Screen = 'upload' | 'config' | 'speakerDetection' | 'speakerReview' | 'progress' | 'results' | 'settings';

interface VideoInfo {
  path: string;
  filename: string;
  sizeInMB: number;
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
}

const App: React.FC = () => {
  const [currentScreen, setCurrentScreen] = useState<Screen>('upload');
  const [selectedVideo, setSelectedVideo] = useState<VideoInfo | null>(null);
  const [jobId, setJobId] = useState<number | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [configKey, setConfigKey] = useState(0);

  // V10 state
  const [v10Config, setV10Config] = useState<V10Config | null>(null);
  const [detectedSpeakers, setDetectedSpeakers] = useState<SpeakerInfo[]>([]);
  const [speakerDetectionStatus, setSpeakerDetectionStatus] = useState<string>('Detecting speakers...');
  const [speakerDetectionPercent, setSpeakerDetectionPercent] = useState(0);
  const [speakerDetectionError, setSpeakerDetectionError] = useState<string | null>(null);

  // Listen for speaker detection progress
  useEffect(() => {
    if (currentScreen !== 'speakerDetection') return;

    const unsubscribe = window.electronAPI.onSpeakerProgress((data: any) => {
      if (data.status) {
        setSpeakerDetectionStatus(data.status);
      }
      if (data.percent !== undefined) {
        setSpeakerDetectionPercent(data.percent);
      }
    });

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [currentScreen]);

  const handleVideoSelected = (videoInfo: VideoInfo) => {
    setSelectedVideo(videoInfo);
    setCurrentScreen('config');
  };

  const handleConfigBack = () => {
    setCurrentScreen('upload');
  };

  const handleConfigStart = async (config: V10Config) => {
    setV10Config(config);
    setSpeakerDetectionError(null);
    setSpeakerDetectionStatus('Preparing speaker detection...');
    setSpeakerDetectionPercent(0);
    setCurrentScreen('speakerDetection');

    try {
      // Get API key
      const { apiKey } = await window.electronAPI.getApiKey();
      if (!apiKey) {
        throw new Error('API key not found. Please check Settings.');
      }

      // Phase 1: Detect speakers
      const result = await window.electronAPI.detectSpeakers({
        videoPath: config.videoPath,
        model: config.model,
        resolution: config.resolution,
        fps: config.fps,
        chunkMinutes: config.chunkMinutes,
        overlapSeconds: config.overlapSeconds,
        apiKey,
        audioOnly: config.audioOnly,
      });

      setDetectedSpeakers(result.speakers);
      setCurrentScreen('speakerReview');
    } catch (error: any) {
      console.error('Speaker detection failed:', error);
      setSpeakerDetectionError(error.message || 'Speaker detection failed');
    }
  };

  const handleSpeakerReviewBack = () => {
    setCurrentScreen('config');
  };

  const handleSpeakerReviewContinue = async (speakers: SpeakerInfo[]) => {
    if (!v10Config) return;

    try {
      // Get API key
      const { apiKey } = await window.electronAPI.getApiKey();
      if (!apiKey) {
        throw new Error('API key not found. Please check Settings.');
      }

      // Save speaker manifest
      const manifestResult = await window.electronAPI.saveSpeakerManifest(speakers);

      // Get output path
      const { path: outputPath } = await window.electronAPI.getOutputPath();

      // Phase 2: Start transcription with speaker manifest
      const result = await window.electronAPI.startTranscription({
        videoPath: v10Config.videoPath,
        prompt: v10Config.prompt,
        model: v10Config.model,
        resolution: v10Config.resolution,
        fps: v10Config.fps,
        chunkMinutes: v10Config.chunkMinutes,
        overlapSeconds: v10Config.overlapSeconds,
        thinkingBudget: v10Config.thinkingBudget,
        outputPath,
        apiKey,
        speakersManifestPath: manifestResult.path,
        audioOnly: v10Config.audioOnly,
      });

      if (result.success) {
        setJobId(result.jobId);
        setCurrentScreen('progress');
      }
    } catch (error: any) {
      console.error('Failed to start transcription:', error);
      alert('Failed to start transcription: ' + error.message);
    }
  };

  const handleCancelSpeakerDetection = async () => {
    try {
      await window.electronAPI.cancelTranscription();
    } catch (error) {
      // Ignore cancel errors
    }
    setCurrentScreen('config');
  };

  const handleOpenSettings = () => {
    setShowSettings(true);
  };

  const handleCloseSettings = () => {
    setShowSettings(false);
    setConfigKey(prev => prev + 1);
  };

  return (
    <div>
      {currentScreen === 'upload' && (
        <VideoUpload onVideoSelected={handleVideoSelected} />
      )}
      {currentScreen === 'config' && selectedVideo && (
        <ConfigScreen
          key={configKey}
          videoInfo={selectedVideo}
          onBack={handleConfigBack}
          onStart={handleConfigStart}
          onOpenSettings={handleOpenSettings}
        />
      )}
      {currentScreen === 'speakerDetection' && (
        <SpeakerDetectionScreen
          status={speakerDetectionStatus}
          percent={speakerDetectionPercent}
          error={speakerDetectionError}
          onCancel={handleCancelSpeakerDetection}
          onRetry={() => v10Config && handleConfigStart(v10Config)}
        />
      )}
      {currentScreen === 'speakerReview' && selectedVideo && (
        <SpeakerReview
          speakers={detectedSpeakers}
          videoFilename={selectedVideo.filename}
          onBack={handleSpeakerReviewBack}
          onContinue={handleSpeakerReviewContinue}
        />
      )}
      {currentScreen === 'progress' && jobId && (
        <ProgressScreen
          jobId={jobId}
          onComplete={() => setCurrentScreen('results')}
          onCancel={() => setCurrentScreen('upload')}
        />
      )}
      {currentScreen === 'results' && jobId && (
        <ResultsScreen
          jobId={jobId}
          onNewTranscription={() => {
            setJobId(null);
            setSelectedVideo(null);
            setV10Config(null);
            setDetectedSpeakers([]);
            setCurrentScreen('upload');
          }}
        />
      )}

      {showSettings && <Settings onClose={handleCloseSettings} />}
    </div>
  );
};

// Inline loading screen for speaker detection
const SpeakerDetectionScreen: React.FC<{
  status: string;
  percent: number;
  error: string | null;
  onCancel: () => void;
  onRetry: () => void;
}> = ({ status, percent, error, onCancel, onRetry }) => {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '2rem',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      <div style={{
        width: '100%',
        maxWidth: '500px',
        background: 'white',
        borderRadius: '12px',
        padding: '2.5rem',
        boxShadow: '0 10px 40px rgba(0, 0, 0, 0.2)',
        textAlign: 'center',
      }}>
        <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.5rem', color: '#111827' }}>
          Detecting Speakers
        </h2>
        <p style={{ margin: '0 0 2rem 0', fontSize: '0.9rem', color: '#6b7280' }}>
          Analyzing media to identify speakers...
        </p>

        {error ? (
          <>
            <div style={{
              background: '#fef2f2',
              border: '1px solid #ef4444',
              borderRadius: '8px',
              padding: '1rem',
              marginBottom: '1.5rem',
              color: '#991b1b',
              fontSize: '0.9rem',
            }}>
              {error}
            </div>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button
                onClick={onCancel}
                style={{
                  padding: '0.6rem 1.5rem',
                  borderRadius: '8px',
                  border: '1px solid #d1d5db',
                  background: 'white',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                }}
              >
                Back
              </button>
              <button
                onClick={onRetry}
                style={{
                  padding: '0.6rem 1.5rem',
                  borderRadius: '8px',
                  border: 'none',
                  background: '#667eea',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                }}
              >
                Retry
              </button>
            </div>
          </>
        ) : (
          <>
            {/* Progress bar */}
            <div style={{
              width: '100%',
              height: '8px',
              background: '#e5e7eb',
              borderRadius: '999px',
              overflow: 'hidden',
              marginBottom: '1rem',
            }}>
              <div style={{
                height: '100%',
                background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                transition: 'width 0.3s ease',
                borderRadius: '999px',
                width: `${percent}%`,
              }} />
            </div>

            <p style={{ margin: '0 0 1.5rem 0', fontSize: '0.85rem', color: '#6b7280' }}>
              {status}
            </p>

            <button
              onClick={onCancel}
              style={{
                padding: '0.5rem 1.5rem',
                borderRadius: '8px',
                border: '1px solid #d1d5db',
                background: 'white',
                cursor: 'pointer',
                fontSize: '0.9rem',
                color: '#374151',
              }}
            >
              Cancel
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default App;
