import React, { useState, useCallback } from 'react';
import styles from './VideoUpload.module.css';
import { Button } from '../shared/Button';
import { About } from '../About/About';

interface VideoInfo {
  path: string;
  filename: string;
  sizeInMB: number;
  durationMinutes?: number;
}

interface VideoUploadProps {
  onVideoSelected: (videoInfo: VideoInfo) => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoSelected }) => {
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);

  const handleBrowseClick = useCallback(async () => {
    try {
      setError(null);
      const result = await window.electronAPI.selectVideo();
      console.log('Select video result:', result);

      if (result.success && result.path) {
        onVideoSelected({
          path: result.path,
          filename: result.filename!,
          sizeInMB: result.sizeInMB!,
          durationMinutes: result.durationMinutes
        });
      }
    } catch (err: any) {
      console.error('Browse error:', err);
      setError(err.message || 'Failed to select video file');
    }
  }, [onVideoSelected]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h1>Gemini Video Understanding</h1>
          <p>Upload a classroom video to transcribe with AI-powered speaker diarization</p>
        </div>
        <button className={styles.aboutButton} onClick={() => setShowAbout(true)} title="About">
          ℹ️
        </button>
      </div>

      <div className={`${styles.dropzone} ${error ? styles.error : ''}`}>
        {validating ? (
          <div className={styles.validating}>
            <div className={styles.spinner}></div>
            <p>Validating video file...</p>
          </div>
        ) : (
          <>
            <svg className={styles.icon} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <h2 className={styles.selectTitle}>Select Video File</h2>
            <p className={styles.supportedFormats}>
              Supported formats: MP4, MOV, AVI, MKV, WebM
            </p>
            <Button variant="primary" size="large" onClick={handleBrowseClick}>
              Choose Video
            </Button>
          </>
        )}
      </div>

      {error && (
        <div className={styles.errorMessage}>
          <span className={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}

      <div className={styles.recentJobs}>
        <h3>Recent Jobs</h3>
        <p className={styles.emptyState}>No recent transcriptions</p>
      </div>

      {showAbout && <About onClose={() => setShowAbout(false)} />}
    </div>
  );
};
