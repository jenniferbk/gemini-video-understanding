import React, { useState } from 'react';
import { VideoUpload } from './components/VideoUpload/VideoUpload';
import { ConfigScreen } from './components/ConfigScreen/ConfigScreen';
import { Settings } from './components/Settings/Settings';
import { ProgressScreen } from './components/ProgressScreen/ProgressScreen';
import { ResultsScreen } from './components/ResultsScreen/ResultsScreen';

type Screen = 'upload' | 'config' | 'progress' | 'results' | 'settings';

interface VideoInfo {
  path: string;
  filename: string;
  sizeInMB: number;
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

const App: React.FC = () => {
  const [currentScreen, setCurrentScreen] = useState<Screen>('upload');
  const [selectedVideo, setSelectedVideo] = useState<VideoInfo | null>(null);
  const [jobId, setJobId] = useState<number | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [configKey, setConfigKey] = useState(0);

  const handleVideoSelected = (videoInfo: VideoInfo) => {
    setSelectedVideo(videoInfo);
    setCurrentScreen('config');
  };

  const handleConfigBack = () => {
    setCurrentScreen('upload');
  };

  const handleConfigStart = async (config: TranscriptionConfig) => {
    try {
      console.log('Starting transcription with config:', config);
      const result = await window.electronAPI.startTranscription(config);

      if (result.success) {
        setJobId(result.jobId);
        setCurrentScreen('progress');
        console.log('Transcription started, job ID:', result.jobId);
      }
    } catch (error: any) {
      console.error('Failed to start transcription:', error);
      alert('Failed to start transcription: ' + error.message);
    }
  };

  const handleOpenSettings = () => {
    setShowSettings(true);
  };

  const handleCloseSettings = () => {
    setShowSettings(false);
    // Force ConfigScreen to re-check API key
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
            setCurrentScreen('upload');
          }}
        />
      )}

      {showSettings && <Settings onClose={handleCloseSettings} />}
    </div>
  );
};

export default App;
