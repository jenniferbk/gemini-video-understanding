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
      setError(err.message || 'Failed to select file');
    }
  }, [onVideoSelected]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h1>Gemini Video Understanding</h1>
          <p>Upload a classroom video or audio file to transcribe with AI-powered speaker diarization</p>
        </div>
        <button className={styles.aboutButton} onClick={() => setShowAbout(true)} title="About">
          ℹ️
        </button>
      </div>

      <div className={`${styles.dropzone} ${error ? styles.error : ''}`}>
        {validating ? (
          <div className={styles.validating}>
            <div className={styles.spinner}></div>
            <p>Validating file...</p>
          </div>
        ) : (
          <>
            <svg className={styles.icon} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
            </svg>
            <h2 className={styles.selectTitle}>Select Media File</h2>
            <p className={styles.supportedFormats}>
              Video: MP4, MOV, AVI, MKV, WebM &nbsp;|&nbsp; Audio: MP3, WAV, M4A, AAC
            </p>
            <Button variant="primary" size="large" onClick={handleBrowseClick}>
              Choose File
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
