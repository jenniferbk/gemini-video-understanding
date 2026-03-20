import React, { useState, useMemo } from 'react';
import styles from './SpeakerReview.module.css';
import { Button } from '../shared/Button';

export interface SpeakerInfo {
  label: string;
  description: string;
  type: 'teacher' | 'student' | 'researcher';
}

interface SpeakerReviewProps {
  speakers: SpeakerInfo[];
  videoFilename: string;
  onBack: () => void;
  onContinue: (speakers: SpeakerInfo[]) => void;
}

const SPEAKER_TYPES = [
  { value: 'teacher', label: 'Teacher' },
  { value: 'student', label: 'Student' },
  { value: 'researcher', label: 'Researcher' },
];

export const SpeakerReview: React.FC<SpeakerReviewProps> = ({
  speakers: initialSpeakers,
  videoFilename,
  onBack,
  onContinue
}) => {
  const [speakers, setSpeakers] = useState<SpeakerInfo[]>(initialSpeakers);

  // Check for ambiguity warnings (ported from V10 _check_ambiguity)
  const warnings = useMemo(() => {
    const result: string[] = [];
    const ignoreWords = new Set(['Boy', 'Girl', 'Teacher', 'Student', 'Shirt', 'Top',
      'Pants', 'Shorts', 'Shoes', 'Dress', 'Hair']);

    // Check for shared descriptor words
    for (let i = 0; i < speakers.length; i++) {
      for (let j = i + 1; j < speakers.length; j++) {
        const words1 = new Set((speakers[i].label.match(/[A-Z][a-z]+/g) || [])
          .filter(w => !ignoreWords.has(w)));
        const words2 = new Set((speakers[j].label.match(/[A-Z][a-z]+/g) || [])
          .filter(w => !ignoreWords.has(w)));

        const shared = [...words1].filter(w => words2.has(w));
        if (shared.length > 0) {
          result.push(
            `"${speakers[i].label}" and "${speakers[j].label}" share feature "${shared.join(', ')}" -- this may confuse the model. Consider using unique features (hair color, position, etc).`
          );
        }
      }
    }

    // Check for generic labels
    for (const s of speakers) {
      if (/^(Male|Female)?(Student|Speaker)\d*$/.test(s.label)) {
        result.push(
          `"${s.label}" is generic. Use a visual feature (e.g., Girl-BlondeHair, Boy-TallLeft) for better accuracy.`
        );
      }
    }

    return result;
  }, [speakers]);

  const updateSpeaker = (index: number, field: keyof SpeakerInfo, value: string) => {
    setSpeakers(prev => {
      const updated = [...prev];
      updated[index] = { ...updated[index], [field]: value };
      return updated;
    });
  };

  const addSpeaker = () => {
    setSpeakers(prev => [
      ...prev,
      { label: `Speaker${prev.length + 1}`, description: '', type: 'student' }
    ]);
  };

  const removeSpeaker = (index: number) => {
    if (speakers.length <= 1) return;
    setSpeakers(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>Review Speakers</h1>
        <p className={styles.subtitle}>{videoFilename}</p>
      </div>

      <div className={styles.content}>
        <p className={styles.instructions}>
          Review the auto-detected speakers below. Labels should <strong>uniquely identify</strong> each person using hair color, position, height, or accessories.
        </p>

        {/* Ambiguity Warnings */}
        {warnings.length > 0 && (
          <div className={styles.warnings}>
            {warnings.map((w, i) => (
              <div key={i} className={styles.warningItem}>
                <span className={styles.warningIcon}>⚠️</span>
                <span>{w}</span>
              </div>
            ))}
          </div>
        )}

        {/* Speaker Table */}
        <div className={styles.speakerList}>
          {speakers.map((speaker, index) => (
            <div key={index} className={styles.speakerCard}>
              <div className={styles.speakerNumber}>#{index + 1}</div>
              <div className={styles.speakerFields}>
                <div className={styles.fieldRow}>
                  <label className={styles.fieldLabel}>Label</label>
                  <input
                    type="text"
                    value={speaker.label}
                    onChange={(e) => updateSpeaker(index, 'label', e.target.value)}
                    className={styles.textInput}
                    placeholder="e.g., Girl-BlondeHair"
                  />
                </div>
                <div className={styles.fieldRow}>
                  <label className={styles.fieldLabel}>Description</label>
                  <textarea
                    value={speaker.description}
                    onChange={(e) => updateSpeaker(index, 'description', e.target.value)}
                    className={styles.textArea}
                    placeholder="Detailed physical description..."
                    rows={2}
                  />
                </div>
                <div className={styles.fieldRow}>
                  <label className={styles.fieldLabel}>Type</label>
                  <select
                    value={speaker.type}
                    onChange={(e) => updateSpeaker(index, 'type', e.target.value)}
                    className={styles.selectInput}
                  >
                    {SPEAKER_TYPES.map(t => (
                      <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                  </select>
                </div>
              </div>
              <button
                className={styles.removeButton}
                onClick={() => removeSpeaker(index)}
                disabled={speakers.length <= 1}
                title="Remove speaker"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        {/* Add Speaker */}
        <button className={styles.addButton} onClick={addSpeaker}>
          + Add Speaker
        </button>

        {/* Actions */}
        <div className={styles.actions}>
          <Button variant="secondary" onClick={onBack}>
            Back
          </Button>
          <Button
            variant="primary"
            onClick={() => onContinue(speakers)}
            disabled={speakers.some(s => !s.label.trim())}
          >
            Continue Transcription
          </Button>
        </div>
      </div>
    </div>
  );
};
