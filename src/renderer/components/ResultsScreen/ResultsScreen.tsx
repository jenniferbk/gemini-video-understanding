import React, { useState, useEffect } from 'react';
import styles from './ResultsScreen.module.css';
import { Button } from '../shared/Button';
import { About } from '../About/About';

interface ResultsScreenProps {
  jobId: number;
  onNewTranscription: () => void;
}

interface TranscriptionResult {
  outputPath: string;
  transcript: string;
  stats: {
    chunks: number;
    lines: number;
    autoAccept: number;
    reviewNeeded: number;
    processingTimeMinutes: number;
  };
}

export const ResultsScreen: React.FC<ResultsScreenProps> = ({
  jobId,
  onNewTranscription
}) => {
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => {
    async function loadResults() {
      try {
        const job = await window.electronAPI.getJob(jobId);

        if (!job || !job.output_path) {
          setError('Transcription results not found');
          setLoading(false);
          return;
        }

        // Read the transcript file
        let transcript = '';
        try {
          const transcriptData = await window.electronAPI.readTranscript(job.output_path);
          transcript = transcriptData.content;
        } catch (error) {
          console.error('Failed to read transcript:', error);
          transcript = 'Failed to load transcript. File may have been moved or deleted.';
        }

        const stats = job.stats_json ? JSON.parse(job.stats_json) : {
          chunks: 0,
          lines: 0,
          autoAccept: 0,
          reviewNeeded: 0,
          processingTimeMinutes: 0
        };

        setResult({
          outputPath: job.output_path,
          transcript,
          stats
        });
        setLoading(false);
      } catch (error: any) {
        setError(error.message || 'Failed to load results');
        setLoading(false);
      }
    }

    loadResults();
  }, [jobId]);

  const handleOpenFolder = async () => {
    if (result) {
      await window.electronAPI.openFolder(result.outputPath);
    }
  };

  const handleCopyToClipboard = () => {
    if (result) {
      navigator.clipboard.writeText(result.transcript);
      alert('Transcript copied to clipboard!');
    }
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.loading}>Loading results...</div>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.error}>{error || 'No results found'}</div>
          <Button variant="primary" onClick={onNewTranscription}>
            Start New Transcription
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div className={styles.header}>
          <div className={styles.headerContent}>
            <h1>Transcription Complete!</h1>
            <div className={styles.successIcon}>✓</div>
          </div>
          <button className={styles.aboutButton} onClick={() => setShowAbout(true)} title="About">
            ℹ️
          </button>
        </div>

        {/* Statistics */}
        <div className={styles.statsSection}>
          <div className={styles.stat}>
            <div className={styles.statValue}>{result.stats.chunks}</div>
            <div className={styles.statLabel}>Chunks Processed</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>{result.stats.lines}</div>
            <div className={styles.statLabel}>Total Lines</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>{result.stats.autoAccept}</div>
            <div className={styles.statLabel}>Auto-accepted</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>{result.stats.reviewNeeded}</div>
            <div className={styles.statLabel}>Need Review</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>{result.stats.processingTimeMinutes}</div>
            <div className={styles.statLabel}>Minutes</div>
          </div>
        </div>

        {/* Transcript Preview */}
        <div className={styles.transcriptSection}>
          <h2>Transcript Preview</h2>
          <div className={styles.transcriptPreview}>
            {result.transcript.split('\n').map((line, index) => (
              <div key={index} className={styles.transcriptLine}>
                {line}
              </div>
            ))}
          </div>
        </div>

        {/* Output Path */}
        <div className={styles.outputSection}>
          <div className={styles.outputLabel}>Saved to:</div>
          <div className={styles.outputPath}>{result.outputPath}</div>
        </div>

        {/* Actions */}
        <div className={styles.actions}>
          <Button variant="secondary" onClick={handleOpenFolder}>
            Open Folder
          </Button>
          <Button variant="secondary" onClick={handleCopyToClipboard}>
            Copy to Clipboard
          </Button>
          <Button variant="primary" onClick={onNewTranscription}>
            New Transcription
          </Button>
        </div>

        {/* About Modal */}
        {showAbout && <About onClose={() => setShowAbout(false)} />}
      </div>
    </div>
  );
};
