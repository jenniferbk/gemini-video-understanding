import React from 'react';
import styles from './About.module.css';
import { Button } from '../shared/Button';

interface AboutProps {
  onClose: () => void;
}

export const About: React.FC<AboutProps> = ({ onClose }) => {
  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2>About Gemini Video Understanding</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.content}>
          <div className={styles.logo}>
            <svg className={styles.icon} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>

          <div className={styles.version}>
            Version 1.0.1
          </div>

          <div className={styles.description}>
            AI-powered video transcription with speaker diarization for educational research
          </div>

          <div className={styles.section}>
            <h3>Developed at</h3>
            <p className={styles.institution}>University of Georgia</p>
            <p className={styles.detail}>Department of Mathematics and Science Education</p>
          </div>

          <div className={styles.section}>
            <h3>Principal Investigators</h3>
            <ul className={styles.contributors}>
              <li>AnnaMarie Conner</li>
              <li>Xiaoming Zhai</li>
            </ul>
          </div>

          <div className={styles.section}>
            <h3>Development Team</h3>
            <div className={styles.leadRole}>
              <strong>Lead Designer & Researcher</strong>
              <p className={styles.leadName}>Jennifer Kleiman</p>
              <p className={styles.leadDescription}>
                Designed and developed the sophisticated multi-stage transcription pipeline,
                integrating voice activity detection, audio denoising, adaptive chunking,
                multimodal AI analysis, and consensus-based verification for reliable
                educational video transcription.
              </p>
            </div>
            <div className={styles.teamMembers}>
              <strong>Contributing Researchers</strong>
              <ul className={styles.contributors}>
                <li>Anna Bloodworth</li>
                <li>Uyi Uyiosa</li>
              </ul>
            </div>
          </div>

          <div className={styles.section}>
            <h3>Funding</h3>
            <p className={styles.grant}>
              This work was supported by the National Science Foundation through:
            </p>
            <ul className={styles.grants}>
              <li>
                <strong>C4OMS Project</strong>
                <span className={styles.grantDetail}>
                  Collaborative Research: Characterizing Classroom Observations of Mathematics and Science
                </span>
              </li>
              <li>
                <strong>AI4STEM Center</strong>
                <span className={styles.grantDetail}>
                  Artificial Intelligence for STEM Education
                </span>
              </li>
            </ul>
            <p className={styles.disclaimer}>
              Any opinions, findings, and conclusions or recommendations expressed in this material
              are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
            </p>
          </div>

          <div className={styles.section}>
            <h3>Technology Pipeline</h3>
            <p className={styles.tech}>
              This application implements a sophisticated multi-stage processing pipeline that combines:
            </p>
            <ul className={styles.techList}>
              <li><strong>Voice Activity Detection (VAD)</strong> – Silero VAD for intelligent audio segmentation</li>
              <li><strong>Audio Preprocessing</strong> – Noise reduction and signal enhancement</li>
              <li><strong>Adaptive Chunking</strong> – Intelligent video segmentation optimized for context</li>
              <li><strong>Multimodal AI Analysis</strong> – Google Gemini 2.5 Pro for simultaneous video and audio analysis</li>
              <li><strong>Consensus Verification</strong> – Multiple-run analysis with semantic similarity scoring</li>
              <li><strong>Speaker Diarization</strong> – AI-powered speaker identification and turn-taking analysis</li>
            </ul>
            <p className={styles.techNote}>
              The integration and orchestration of these components was designed and implemented
              by Jennifer Kleiman to create a robust, research-grade transcription system.
            </p>
          </div>

          <div className={styles.footer}>
            <Button variant="primary" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
