import React, { useState, useEffect, useMemo, useRef } from 'react';
import { marked } from 'marked';
import styles from './PromptManager.module.css';
import { Button } from '../shared/Button';

interface Prompt {
  id: string;
  name: string;
  description: string;
  prompt_text: string;
  created_at?: string;
  updated_at?: string;
}

interface PromptManagerProps {
  onClose: () => void;
}

export const PromptManager: React.FC<PromptManagerProps> = ({ onClose }) => {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState<Prompt | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedPrompt, setEditedPrompt] = useState<Prompt | null>(null);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState<'builder' | 'raw'>('raw');
  const [viewMode, setViewMode] = useState<'formatted' | 'raw'>('formatted');
  const [builderSections, setBuilderSections] = useState<Record<string, string>>({});

  useEffect(() => {
    loadPrompts();
  }, []);

  const loadPrompts = async () => {
    try {
      const loadedPrompts = await window.electronAPI.getPrompts();
      setPrompts(loadedPrompts);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load prompts:', error);
      setLoading(false);
    }
  };

  const handleSelectPrompt = (prompt: Prompt) => {
    setSelectedPrompt(prompt);
    setIsEditing(false);
  };

  const parsePromptSections = (promptText: string) => {
    const sections: Record<string, string> = {};

    // Match **SECTION:** patterns (can be followed by more text on same line or newline)
    const sectionRegex = /\*\*([A-Z\s]+):\*\*/g;
    const matches = [...promptText.matchAll(sectionRegex)];

    if (matches.length === 0) {
      // No structured sections found, return empty to use defaults
      return sections;
    }

    for (let i = 0; i < matches.length; i++) {
      const sectionName = matches[i][1].trim();

      // Start after the **SECTION:** marker
      const startIndex = matches[i].index! + matches[i][0].length;

      // End at the next **SECTION:** or end of text
      const endIndex = i < matches.length - 1 ? matches[i + 1].index! : promptText.length;

      // Extract content and trim whitespace
      let content = promptText.substring(startIndex, endIndex).trim();

      // Remove leading newlines
      content = content.replace(/^\s*\n+/, '');

      sections[sectionName] = content;
    }

    return sections;
  };

  const buildPromptFromSections = (sections: Record<string, string>) => {
    return Object.entries(sections)
      .map(([name, content]) => `**${name}:**\n${content}`)
      .join('\n\n');
  };

  const handleEditPrompt = () => {
    if (selectedPrompt) {
      setEditedPrompt({ ...selectedPrompt });
      setBuilderSections(parsePromptSections(selectedPrompt.prompt_text));
      setIsEditing(true);
    }
  };

  const handleSavePrompt = async () => {
    if (editedPrompt) {
      try {
        // If in builder mode, build final prompt from sections
        const finalPrompt = editMode === 'builder'
          ? { ...editedPrompt, prompt_text: buildPromptFromSections(builderSections) }
          : editedPrompt;

        await window.electronAPI.savePrompt(finalPrompt);
        await loadPrompts();
        setSelectedPrompt(finalPrompt);
        setIsEditing(false);
        alert('Prompt saved successfully!');
      } catch (error: any) {
        alert('Failed to save prompt: ' + error.message);
      }
    }
  };

  const handleCancelEdit = () => {
    setEditedPrompt(null);
    setIsEditing(false);
  };

  const handleDeletePrompt = async () => {
    if (selectedPrompt && confirm(`Delete prompt "${selectedPrompt.name}"?`)) {
      try {
        await window.electronAPI.deletePrompt(selectedPrompt.id);
        await loadPrompts();
        setSelectedPrompt(null);
        alert('Prompt deleted successfully!');
      } catch (error: any) {
        alert('Failed to delete prompt: ' + error.message);
      }
    }
  };

  const handleDuplicatePrompt = async () => {
    if (selectedPrompt) {
      try {
        const duplicated = await window.electronAPI.duplicatePrompt(selectedPrompt.id);
        await loadPrompts();
        setSelectedPrompt(duplicated);
        alert('Prompt duplicated successfully!');
      } catch (error: any) {
        alert('Failed to duplicate prompt: ' + error.message);
      }
    }
  };

  const handleImportPrompt = async () => {
    try {
      const result = await window.electronAPI.importPrompt();
      if (result.success && result.prompt) {
        await loadPrompts();
        setSelectedPrompt(result.prompt);
        alert('Prompt imported successfully!');
      }
    } catch (error: any) {
      alert('Failed to import prompt: ' + error.message);
    }
  };

  const handleExportPrompt = async () => {
    if (selectedPrompt) {
      try {
        const result = await window.electronAPI.exportPrompt(selectedPrompt.id);
        if (result.success) {
          alert(`Prompt exported to: ${result.filePath}`);
        }
      } catch (error: any) {
        alert('Failed to export prompt: ' + error.message);
      }
    }
  };

  if (loading) {
    return (
      <div className={styles.overlay}>
        <div className={styles.modal}>
          <div className={styles.loading}>Loading prompts...</div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2>Prompt Manager</h2>
          <button className={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div className={styles.content}>
          {/* Left sidebar - Prompt list */}
          <div className={styles.sidebar}>
            <div className={styles.sidebarHeader}>
              <h3>Prompts ({prompts.length})</h3>
              <Button variant="primary" onClick={handleImportPrompt}>
                Import
              </Button>
            </div>
            <div className={styles.promptList}>
              {prompts.map((prompt) => (
                <div
                  key={prompt.id}
                  className={`${styles.promptItem} ${selectedPrompt?.id === prompt.id ? styles.selected : ''}`}
                  onClick={() => handleSelectPrompt(prompt)}
                >
                  <div className={styles.promptName}>{prompt.name}</div>
                  <div className={styles.promptDescription}>{prompt.description}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right panel - Prompt details/editor */}
          <div className={styles.mainPanel}>
            {selectedPrompt ? (
              isEditing && editedPrompt ? (
                // Edit mode
                <div className={styles.editor}>
                  <div className={styles.editorField}>
                    <label>Name</label>
                    <input
                      type="text"
                      value={editedPrompt.name}
                      onChange={(e) => setEditedPrompt({ ...editedPrompt, name: e.target.value })}
                      className={styles.input}
                    />
                  </div>
                  <div className={styles.editorField}>
                    <label>Description</label>
                    <input
                      type="text"
                      value={editedPrompt.description}
                      onChange={(e) => setEditedPrompt({ ...editedPrompt, description: e.target.value })}
                      className={styles.input}
                    />
                  </div>

                  {/* Mode Toggle */}
                  <div className={styles.modeToggle}>
                    <button
                      className={`${styles.modeButton} ${editMode === 'builder' ? styles.active : ''}`}
                      onClick={() => setEditMode('builder')}
                    >
                      Builder Mode
                    </button>
                    <button
                      className={`${styles.modeButton} ${editMode === 'raw' ? styles.active : ''}`}
                      onClick={() => setEditMode('raw')}
                    >
                      Raw Text
                    </button>
                  </div>

                  {editMode === 'raw' ? (
                    <div className={styles.editorField}>
                      <label>Prompt Text</label>
                      <textarea
                        value={editedPrompt.prompt_text}
                        onChange={(e) => setEditedPrompt({ ...editedPrompt, prompt_text: e.target.value })}
                        className={styles.textarea}
                        rows={20}
                      />
                    </div>
                  ) : (
                    <div className={styles.builderMode}>
                      {(() => {
                        const sectionNames = Object.keys(builderSections).length > 0
                          ? Object.keys(builderSections)
                          : ['CONTEXT', 'SPEAKER IDENTIFICATION', 'TRANSCRIPTION REQUIREMENTS', 'CRITICAL RULES'];

                        const placeholders: Record<string, string> = {
                          'CONTEXT': 'Example: This is a classroom video transcription. The video shows small group discussion in a science class. Focus on capturing student reasoning and collaborative problem-solving.',
                          'SPEAKER IDENTIFICATION': 'Example: Use speaker labels like Teacher [T], Student 1 [S1], Student 2 [S2]. Identify speakers by voice characteristics and speaking patterns.',
                          'TRANSCRIPTION REQUIREMENTS': 'Example: Include timestamps for each turn, capture overlapping speech with [overlap] markers, preserve filler words (um, uh, like), note laughter and pauses.',
                          'CRITICAL RULES': 'Example: Never merge speaker turns. Always verify speaker identity. Mark uncertain transcriptions with [verify] tags. Preserve exact wording including grammatical errors.'
                        };

                        return sectionNames.map((sectionName) => (
                          <div key={sectionName} className={styles.builderSection}>
                            <label className={styles.sectionLabel}>{sectionName}</label>
                            <textarea
                              value={builderSections[sectionName] || ''}
                              onChange={(e) => {
                                setBuilderSections({ ...builderSections, [sectionName]: e.target.value });
                              }}
                              className={styles.sectionTextarea}
                              rows={6}
                              placeholder={placeholders[sectionName] || `Enter ${sectionName.toLowerCase()} content...`}
                            />
                          </div>
                        ));
                      })()}
                    </div>
                  )}

                  <div className={styles.editorActions}>
                    <Button variant="secondary" onClick={handleCancelEdit}>
                      Cancel
                    </Button>
                    <Button variant="primary" onClick={handleSavePrompt}>
                      Save Changes
                    </Button>
                  </div>
                </div>
              ) : (
                // View mode
                <div className={styles.viewer}>
                  <div className={styles.viewerHeader}>
                    <div>
                      <h3>{selectedPrompt.name}</h3>
                      <p className={styles.description}>{selectedPrompt.description}</p>
                    </div>
                    <div className={styles.viewerActions}>
                      <Button variant="secondary" onClick={handleEditPrompt}>
                        Edit
                      </Button>
                      <Button variant="secondary" onClick={handleDuplicatePrompt}>
                        Duplicate
                      </Button>
                      <Button variant="secondary" onClick={handleExportPrompt}>
                        Export
                      </Button>
                      <Button variant="danger" onClick={handleDeletePrompt}>
                        Delete
                      </Button>
                    </div>
                  </div>
                  <div className={styles.promptTextView}>
                    <div className={styles.viewHeader}>
                      <h4>Prompt Text</h4>
                      <div className={styles.viewToggle}>
                        <button
                          className={`${styles.viewButton} ${viewMode === 'formatted' ? styles.active : ''}`}
                          onClick={() => setViewMode('formatted')}
                        >
                          Formatted
                        </button>
                        <button
                          className={`${styles.viewButton} ${viewMode === 'raw' ? styles.active : ''}`}
                          onClick={() => setViewMode('raw')}
                        >
                          Raw
                        </button>
                      </div>
                    </div>
                    {viewMode === 'formatted' ? (
                      <div
                        className={styles.markdownView}
                        dangerouslySetInnerHTML={{ __html: marked.parse(selectedPrompt.prompt_text) as string }}
                      />
                    ) : (
                      <pre className={styles.promptText}>{selectedPrompt.prompt_text}</pre>
                    )}
                  </div>
                </div>
              )
            ) : (
              <div className={styles.emptyState}>
                <p>Select a prompt to view or edit</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
