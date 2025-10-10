import React, { useState, useEffect } from 'react';
import styles from './ProgressScreen.module.css';
import { Button } from '../shared/Button';
import { About } from '../About/About';

interface ProgressScreenProps {
  jobId: number;
  onComplete: () => void;
  onCancel: () => void;
}

interface ProgressUpdate {
  percent: number;
  currentStep: string;
  chunksProcessed: number;
  totalChunks: number;
  estimatedTimeRemaining?: string;
}

export const ProgressScreen: React.FC<ProgressScreenProps> = ({
  jobId,
  onComplete,
  onCancel
}) => {
  const [progress, setProgress] = useState<ProgressUpdate>({
    percent: 0,
    currentStep: 'Initializing...',
    chunksProcessed: 0,
    totalChunks: 0
  });
  const [logs, setLogs] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    // Listen for progress updates
    const unsubscribe = window.electronAPI.onTranscriptionProgress((update: any) => {
      if (update.type === 'progress') {
        setProgress({
          percent: update.percent || 0,
          currentStep: update.status || 'Processing...',
          chunksProcessed: update.chunk || 0,
          totalChunks: update.total || 0,
          estimatedTimeRemaining: update.estimatedTime
        });

        // Add to logs
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${update.status}`]);
      } else if (update.type === 'complete') {
        setProgress({
          percent: 100,
          currentStep: 'Complete!',
          chunksProcessed: update.chunksProcessed || 0,
          totalChunks: update.chunksProcessed || 0
        });

        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Transcription completed successfully`]);

        // Wait a moment then call onComplete
        setTimeout(() => {
          onComplete();
        }, 1000);
      } else if (update.type === 'error') {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ERROR: ${update.message}`]);
        setHasError(true);
      } else if (update.type === 'log') {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${update.message}`]);
      }
    });

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [onComplete]);

  const handleCancel = async () => {
    if (confirm('Are you sure you want to cancel this transcription?')) {
      setCancelling(true);
      try {
        await window.electronAPI.cancelTranscription();
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Cancellation requested...`]);
        onCancel();
      } catch (error: any) {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Failed to cancel: ${error.message}`]);
        setCancelling(false);
      }
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <h1>Transcription in Progress</h1>
            <p className={styles.subtitle}>Job ID: {jobId}</p>
          </div>
          <button className={styles.aboutButton} onClick={() => setShowAbout(true)} title="About">
            ℹ️
          </button>
        </div>

        {/* Progress Bar */}
        <div className={styles.progressSection}>
          <div className={styles.progressInfo}>
            <span className={styles.currentStep}>{progress.currentStep}</span>
            <span className={styles.percentage}>{progress.percent}%</span>
          </div>

          <div className={styles.progressBarContainer}>
            <div
              className={styles.progressBar}
              style={{ width: `${progress.percent}%` }}
            />
          </div>

          <div className={styles.stats}>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Chunks Processed:</span>
              <span className={styles.statValue}>
                {progress.chunksProcessed} / {progress.totalChunks}
              </span>
            </div>
            {progress.estimatedTimeRemaining && (
              <div className={styles.stat}>
                <span className={styles.statLabel}>Est. Time Remaining:</span>
                <span className={styles.statValue}>{progress.estimatedTimeRemaining}</span>
              </div>
            )}
          </div>
        </div>

        {/* Logs Section */}
        <div className={styles.logsSection}>
          <button
            className={styles.logsToggle}
            onClick={() => setShowLogs(!showLogs)}
          >
            {showLogs ? '▼' : '▶'} Detailed Logs ({logs.length})
          </button>

          {showLogs && (
            <div className={styles.logsContainer}>
              {logs.map((log, index) => (
                <div key={index} className={styles.logEntry}>
                  {log}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className={styles.actions}>
          {hasError ? (
            <Button
              variant="secondary"
              onClick={onCancel}
            >
              Back to Start
            </Button>
          ) : (
            <Button
              variant="danger"
              onClick={handleCancel}
              disabled={cancelling || progress.percent === 100}
            >
              {cancelling ? 'Cancelling...' : 'Cancel Transcription'}
            </Button>
          )}
        </div>

        {/* About Modal */}
        {showAbout && <About onClose={() => setShowAbout(false)} />}
      </div>
    </div>
  );
};
