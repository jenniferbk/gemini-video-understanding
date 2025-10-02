-- SQLite Database Schema for Gemini Video Understanding
-- Version 1.0

-- Job history table
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_path TEXT NOT NULL,
  video_filename TEXT NOT NULL,
  video_duration_minutes REAL,
  prompt_name TEXT NOT NULL,
  config_json TEXT NOT NULL,  -- JSON serialized configuration
  status TEXT NOT NULL CHECK(status IN ('queued', 'processing', 'complete', 'failed', 'cancelled')),
  output_path TEXT,
  stats_json TEXT,  -- JSON serialized stats from completion
  error_message TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME,
  completed_at DATETIME
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_filename ON jobs(video_filename);

-- Settings/preferences table
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Initial default settings
INSERT OR IGNORE INTO settings (key, value) VALUES
  ('default_output_path', '~/Documents/VideoTranscripts'),
  ('auto_update_enabled', 'true'),
  ('theme', 'light'),
  ('last_prompt_used', 'smallgroup_ava');
