import { Database as SQLite3Database } from 'sqlite3';
import { app } from 'electron';
import * as path from 'path';
import * as fs from 'fs';

export interface NewJob {
  videoPath: string;
  videoFilename: string;
  videoDuration?: number;
  promptName: string;
  config: any;
}

export interface Job {
  id: number;
  video_path: string;
  video_filename: string;
  video_duration_minutes?: number;
  prompt_name: string;
  config_json: string;
  status: 'queued' | 'processing' | 'complete' | 'failed' | 'cancelled';
  output_path?: string;
  stats_json?: string;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export class Database {
  private db: SQLite3Database;
  private dbPath: string;

  constructor() {
    // Get user data directory
    const userDataPath = app.getPath('userData');
    this.dbPath = path.join(userDataPath, 'gvu.db');

    // Ensure directory exists
    if (!fs.existsSync(userDataPath)) {
      fs.mkdirSync(userDataPath, { recursive: true });
    }

    // Open database
    this.db = new SQLite3Database(this.dbPath, (err) => {
      if (err) {
        console.error('Database connection error:', err);
        throw err;
      }
      console.log(`📊 Database connected: ${this.dbPath}`);
    });

    // Initialize schema
    this.initialize();
  }

  private async initialize(): Promise<void> {
    // Inline schema (avoids file copy during build)
    const schema = `
      -- Job history table
      CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_path TEXT NOT NULL,
        video_filename TEXT NOT NULL,
        video_duration_minutes REAL,
        prompt_name TEXT NOT NULL,
        config_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('queued', 'processing', 'complete', 'failed', 'cancelled')),
        output_path TEXT,
        stats_json TEXT,
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
    `;

    // Execute schema
    return new Promise((resolve, reject) => {
      this.db.exec(schema, (err) => {
        if (err) {
          console.error('Schema initialization error:', err);
          reject(err);
        } else {
          console.log('✅ Database schema initialized');
          resolve();
        }
      });
    });
  }

  // ===== JOB OPERATIONS =====

  createJob(job: NewJob): Promise<number> {
    return new Promise((resolve, reject) => {
      const sql = `
        INSERT INTO jobs (video_path, video_filename, video_duration_minutes,
                         prompt_name, config_json, status)
        VALUES (?, ?, ?, ?, ?, 'queued')
      `;

      this.db.run(
        sql,
        [
          job.videoPath,
          job.videoFilename,
          job.videoDuration || null,
          job.promptName,
          JSON.stringify(job.config)
        ],
        function(err) {
          if (err) reject(err);
          else resolve(this.lastID);
        }
      );
    });
  }

  updateJobStatus(
    id: number,
    status: Job['status'],
    error?: string
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      let sql = 'UPDATE jobs SET status = ?';
      const params: any[] = [status];

      // Update timestamps based on status
      if (status === 'processing') {
        sql += ', started_at = CURRENT_TIMESTAMP';
      } else if (['complete', 'failed', 'cancelled'].includes(status)) {
        sql += ', completed_at = CURRENT_TIMESTAMP';
      }

      // Add error message if provided
      if (error) {
        sql += ', error_message = ?';
        params.push(error);
      }

      sql += ' WHERE id = ?';
      params.push(id);

      this.db.run(sql, params, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  updateJobOutput(id: number, outputPath: string, stats: any): Promise<void> {
    return new Promise((resolve, reject) => {
      const sql = `
        UPDATE jobs
        SET output_path = ?, stats_json = ?, status = 'complete', completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
      `;

      this.db.run(sql, [outputPath, JSON.stringify(stats), id], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  getJob(id: number): Promise<Job | null> {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT * FROM jobs WHERE id = ?', [id], (err, row) => {
        if (err) reject(err);
        else resolve((row as Job) || null);
      });
    });
  }

  getRecentJobs(limit: number = 10): Promise<Job[]> {
    return new Promise((resolve, reject) => {
      const sql = `
        SELECT * FROM jobs
        ORDER BY created_at DESC
        LIMIT ?
      `;

      this.db.all(sql, [limit], (err, rows) => {
        if (err) reject(err);
        else resolve(rows as Job[]);
      });
    });
  }

  getJobsByStatus(status: Job['status']): Promise<Job[]> {
    return new Promise((resolve, reject) => {
      this.db.all(
        'SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC',
        [status],
        (err, rows) => {
          if (err) reject(err);
          else resolve(rows as Job[]);
        }
      );
    });
  }

  // ===== SETTINGS OPERATIONS =====

  getSetting(key: string): Promise<string | null> {
    return new Promise((resolve, reject) => {
      this.db.get(
        'SELECT value FROM settings WHERE key = ?',
        [key],
        (err, row: any) => {
          if (err) reject(err);
          else resolve(row ? row.value : null);
        }
      );
    });
  }

  setSetting(key: string, value: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const sql = `
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
          value = excluded.value,
          updated_at = CURRENT_TIMESTAMP
      `;

      this.db.run(sql, [key, value], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  getAllSettings(): Promise<Record<string, string>> {
    return new Promise((resolve, reject) => {
      this.db.all('SELECT key, value FROM settings', (err, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          const settings: Record<string, string> = {};
          rows.forEach(row => {
            settings[row.key] = row.value;
          });
          resolve(settings);
        }
      });
    });
  }

  // ===== UTILITY =====

  close(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db.close((err) => {
        if (err) reject(err);
        else {
          console.log('📊 Database closed');
          resolve();
        }
      });
    });
  }
}
