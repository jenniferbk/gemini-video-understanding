import React, { useState, useEffect, useCallback } from 'react';
import styles from './Settings.module.css';
import { Button } from '../shared/Button';
import { Input } from '../shared/Input';

interface SettingsProps {
  onClose: () => void;
}

export const Settings: React.FC<SettingsProps> = ({ onClose }) => {
  const [apiKey, setApiKey] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    async function loadSettings() {
      try {
        const { exists } = await window.electronAPI.hasApiKey();
        setHasApiKey(exists);

        const { path } = await window.electronAPI.getOutputPath();
        setOutputPath(path);
      } catch (error) {
        console.error('Failed to load settings:', error);
      }
    }
    loadSettings();
  }, []);

  const handleSaveApiKey = useCallback(async () => {
    if (!apiKey.trim()) {
      setMessage({ type: 'error', text: 'Please enter an API key' });
      return;
    }

    setSaving(true);
    setMessage(null);

    try {
      const result = await window.electronAPI.saveApiKey(apiKey);
      if (result.success) {
        setHasApiKey(true);
        setApiKey('');
        setMessage({ type: 'success', text: 'API key saved successfully' });
      }
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || 'Failed to save API key' });
    } finally {
      setSaving(false);
    }
  }, [apiKey]);

  const handleDeleteApiKey = useCallback(async () => {
    if (!confirm('Are you sure you want to delete your API key?')) {
      return;
    }

    setSaving(true);
    setMessage(null);

    try {
      const result = await window.electronAPI.deleteApiKey();
      if (result.success) {
        setHasApiKey(false);
        setMessage({ type: 'success', text: 'API key deleted' });
      }
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || 'Failed to delete API key' });
    } finally {
      setSaving(false);
    }
  }, []);

  const handleSelectOutputDirectory = useCallback(async () => {
    try {
      const result = await window.electronAPI.selectOutputDirectory();
      if (result.success && result.path) {
        setOutputPath(result.path);
        setMessage({ type: 'success', text: 'Output directory updated' });
      }
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || 'Failed to select directory' });
    }
  }, []);

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <div className={styles.header}>
          <h2>Settings</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.content}>
          {/* API Key Section */}
          <section className={styles.section}>
            <h3>Gemini API Key</h3>
            <p className={styles.description}>
              Your API key is stored securely in the system keychain.
            </p>

            {hasApiKey ? (
              <div className={styles.apiKeyStatus}>
                <div className={styles.statusBadge}>
                  <span className={styles.statusIcon}>✓</span>
                  API Key Configured
                </div>
                <Button variant="danger" size="small" onClick={handleDeleteApiKey} disabled={saving}>
                  Delete API Key
                </Button>
              </div>
            ) : (
              <div className={styles.apiKeyInput}>
                <Input
                  type="password"
                  value={apiKey}
                  onChange={(value) => setApiKey(value)}
                  placeholder="Enter your Gemini API key"
                  fullWidth
                />
                <Button variant="primary" onClick={handleSaveApiKey} disabled={saving || !apiKey.trim()}>
                  {saving ? 'Saving...' : 'Save API Key'}
                </Button>
              </div>
            )}

            <a
              href="https://aistudio.google.com/app/apikey"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.link}
            >
              Get API Key from Google AI Studio →
            </a>
          </section>

          {/* Output Directory Section */}
          <section className={styles.section}>
            <h3>Output Directory</h3>
            <p className={styles.description}>
              Where transcription files will be saved.
            </p>

            <div className={styles.outputPath}>
              <div className={styles.pathDisplay}>{outputPath}</div>
              <Button variant="secondary" size="small" onClick={handleSelectOutputDirectory}>
                Change
              </Button>
            </div>
          </section>

          {/* Message Display */}
          {message && (
            <div className={`${styles.message} ${styles[message.type]}`}>
              {message.text}
            </div>
          )}
        </div>

        <div className={styles.footer}>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
};
