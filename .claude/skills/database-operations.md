# Database Operations with SQLite

Patterns and best practices for SQLite database operations in Electron applications, including schema management, queries, transactions, and error handling.

## When to use this skill

- Working with `src/main/database/` files
- Creating or modifying database schema
- Implementing database queries
- Managing SQLite connections
- Writing database migrations
- Handling database errors

## Database Setup

### Location

```typescript
// src/main/database/paths.ts
import { app } from 'electron';
import path from 'path';

export function getDatabasePath(): string {
  if (app.isPackaged) {
    // Production: User data directory
    // macOS: ~/Library/Application Support/GeminiVideoUnderstanding/
    const userDataPath = app.getPath('userData');
    return path.join(userDataPath, 'gvu.db');
  } else {
    // Development: Project directory
    return path.join(__dirname, '..', '..', 'database', 'gvu.db');
  }
}
```

### Schema Definition

```sql
-- src/main/database/schema.sql

-- Job history table
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_path TEXT NOT NULL,
  video_filename TEXT NOT NULL,
  video_duration_minutes REAL,
  prompt_name TEXT NOT NULL,
  config_json TEXT NOT NULL,  -- JSON serialized config
  status TEXT NOT NULL CHECK(status IN ('queued', 'processing', 'complete', 'failed', 'cancelled')),
  output_path TEXT,
  stats_json TEXT,  -- JSON serialized stats
  error_message TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  started_at DATETIME,
  completed_at DATETIME
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_video_path ON jobs(video_path);

-- Settings/preferences table
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert default settings
INSERT OR IGNORE INTO settings (key, value) VALUES
  ('api_key_encrypted', ''),
  ('default_output_path', '~/Documents/VideoTranscripts'),
  ('auto_update_enabled', 'true'),
  ('theme', 'light');

-- Prompts library table (optional - could also use JSON file)
CREATE TABLE IF NOT EXISTS prompts (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  prompt_text TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);
```

## Database Class

```typescript
// src/main/database/database.ts
import sqlite3 from 'sqlite3';
import { getDatabasePath } from './paths';
import fs from 'fs/promises';
import path from 'path';

export interface Job {
  id: number;
  video_path: string;
  video_filename: string;
  video_duration_minutes: number | null;
  prompt_name: string;
  config_json: string;
  status: JobStatus;
  output_path: string | null;
  stats_json: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export type JobStatus = 'queued' | 'processing' | 'complete' | 'failed' | 'cancelled';

export interface NewJob {
  videoPath: string;
  videoFilename: string;
  videoDuration: number | null;
  promptName: string;
  config: Record<string, any>;
}

export class Database {
  private db: sqlite3.Database | null = null;
  private readonly dbPath: string;

  constructor() {
    this.dbPath = getDatabasePath();
  }

  async initialize(): Promise<void> {
    // Ensure database directory exists
    const dbDir = path.dirname(this.dbPath);
    await fs.mkdir(dbDir, { recursive: true });

    // Open database connection
    await this.open();

    // Run schema
    await this.runSchema();
  }

  private async open(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.db = new sqlite3.Database(this.dbPath, (err) => {
        if (err) {
          reject(new Error(`Failed to open database: ${err.message}`));
        } else {
          console.log('Database opened:', this.dbPath);
          resolve();
        }
      });
    });
  }

  private async runSchema(): Promise<void> {
    const schemaPath = path.join(__dirname, 'schema.sql');
    const schema = await fs.readFile(schemaPath, 'utf-8');

    return new Promise((resolve, reject) => {
      this.db!.exec(schema, (err) => {
        if (err) {
          reject(new Error(`Failed to run schema: ${err.message}`));
        } else {
          console.log('Database schema initialized');
          resolve();
        }
      });
    });
  }

  async close(): Promise<void> {
    if (!this.db) return;

    return new Promise((resolve, reject) => {
      this.db!.close((err) => {
        if (err) {
          reject(err);
        } else {
          this.db = null;
          resolve();
        }
      });
    });
  }

  // CRUD Operations

  async createJob(job: NewJob): Promise<number> {
    this.ensureConnected();

    const sql = `
      INSERT INTO jobs (
        video_path,
        video_filename,
        video_duration_minutes,
        prompt_name,
        config_json,
        status
      ) VALUES (?, ?, ?, ?, ?, 'queued')
    `;

    return new Promise((resolve, reject) => {
      this.db!.run(
        sql,
        [
          job.videoPath,
          job.videoFilename,
          job.videoDuration,
          job.promptName,
          JSON.stringify(job.config)
        ],
        function (err) {
          if (err) {
            reject(new Error(`Failed to create job: ${err.message}`));
          } else {
            resolve(this.lastID);
          }
        }
      );
    });
  }

  async getJob(id: number): Promise<Job | null> {
    this.ensureConnected();

    const sql = 'SELECT * FROM jobs WHERE id = ?';

    return new Promise((resolve, reject) => {
      this.db!.get(sql, [id], (err, row) => {
        if (err) {
          reject(new Error(`Failed to get job: ${err.message}`));
        } else {
          resolve((row as Job) || null);
        }
      });
    });
  }

  async getRecentJobs(limit: number = 10): Promise<Job[]> {
    this.ensureConnected();

    const sql = `
      SELECT * FROM jobs
      ORDER BY created_at DESC
      LIMIT ?
    `;

    return new Promise((resolve, reject) => {
      this.db!.all(sql, [limit], (err, rows) => {
        if (err) {
          reject(new Error(`Failed to get jobs: ${err.message}`));
        } else {
          resolve(rows as Job[]);
        }
      });
    });
  }

  async updateJobStatus(
    id: number,
    status: JobStatus,
    error?: string
  ): Promise<void> {
    this.ensureConnected();

    let sql = 'UPDATE jobs SET status = ?';
    const params: any[] = [status];

    // Set timestamps based on status
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

    return new Promise((resolve, reject) => {
      this.db!.run(sql, params, (err) => {
        if (err) {
          reject(new Error(`Failed to update job: ${err.message}`));
        } else {
          resolve();
        }
      });
    });
  }

  async updateJobOutput(
    id: number,
    outputPath: string,
    stats: Record<string, any>
  ): Promise<void> {
    this.ensureConnected();

    const sql = `
      UPDATE jobs
      SET output_path = ?, stats_json = ?
      WHERE id = ?
    `;

    return new Promise((resolve, reject) => {
      this.db!.run(sql, [outputPath, JSON.stringify(stats), id], (err) => {
        if (err) {
          reject(new Error(`Failed to update job output: ${err.message}`));
        } else {
          resolve();
        }
      });
    });
  }

  // Settings operations

  async getSetting(key: string): Promise<string | null> {
    this.ensureConnected();

    const sql = 'SELECT value FROM settings WHERE key = ?';

    return new Promise((resolve, reject) => {
      this.db!.get(sql, [key], (err, row: any) => {
        if (err) {
          reject(new Error(`Failed to get setting: ${err.message}`));
        } else {
          resolve(row ? row.value : null);
        }
      });
    });
  }

  async setSetting(key: string, value: string): Promise<void> {
    this.ensureConnected();

    const sql = `
      INSERT INTO settings (key, value, updated_at)
      VALUES (?, ?, CURRENT_TIMESTAMP)
      ON CONFLICT(key) DO UPDATE SET
        value = excluded.value,
        updated_at = CURRENT_TIMESTAMP
    `;

    return new Promise((resolve, reject) => {
      this.db!.run(sql, [key, value], (err) => {
        if (err) {
          reject(new Error(`Failed to set setting: ${err.message}`));
        } else {
          resolve();
        }
      });
    });
  }

  async getAllSettings(): Promise<Record<string, string>> {
    this.ensureConnected();

    const sql = 'SELECT key, value FROM settings';

    return new Promise((resolve, reject) => {
      this.db!.all(sql, [], (err, rows: any[]) => {
        if (err) {
          reject(new Error(`Failed to get settings: ${err.message}`));
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

  // Helper methods

  private ensureConnected(): void {
    if (!this.db) {
      throw new Error('Database not initialized. Call initialize() first.');
    }
  }
}

// Singleton instance
let dbInstance: Database | null = null;

export async function getDatabase(): Promise<Database> {
  if (!dbInstance) {
    dbInstance = new Database();
    await dbInstance.initialize();
  }
  return dbInstance;
}
```

## Transaction Pattern

```typescript
async function performTransaction<T>(
  db: Database,
  callback: () => Promise<T>
): Promise<T> {
  await db.run('BEGIN TRANSACTION');

  try {
    const result = await callback();
    await db.run('COMMIT');
    return result;
  } catch (error) {
    await db.run('ROLLBACK');
    throw error;
  }
}

// Usage
await performTransaction(db, async () => {
  const jobId = await db.createJob(newJob);
  await db.setSetting('last_job_id', String(jobId));
  return jobId;
});
```

## Query Builders

### Safe Parameter Binding

```typescript
// ALWAYS use parameter binding, NEVER string concatenation
// ✅ Good - prevents SQL injection
const sql = 'SELECT * FROM jobs WHERE status = ? AND prompt_name = ?';
db.all(sql, [status, promptName], callback);

// ❌ Bad - SQL injection vulnerability
const sql = `SELECT * FROM jobs WHERE status = '${status}'`;
db.all(sql, callback);
```

### Complex Queries

```typescript
async function searchJobs(filters: {
  status?: JobStatus;
  promptName?: string;
  fromDate?: Date;
  toDate?: Date;
}): Promise<Job[]> {
  const conditions: string[] = [];
  const params: any[] = [];

  if (filters.status) {
    conditions.push('status = ?');
    params.push(filters.status);
  }

  if (filters.promptName) {
    conditions.push('prompt_name = ?');
    params.push(filters.promptName);
  }

  if (filters.fromDate) {
    conditions.push('created_at >= ?');
    params.push(filters.fromDate.toISOString());
  }

  if (filters.toDate) {
    conditions.push('created_at <= ?');
    params.push(filters.toDate.toISOString());
  }

  const whereClause = conditions.length > 0
    ? `WHERE ${conditions.join(' AND ')}`
    : '';

  const sql = `
    SELECT * FROM jobs
    ${whereClause}
    ORDER BY created_at DESC
  `;

  return new Promise((resolve, reject) => {
    this.db!.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as Job[]);
    });
  });
}
```

## Migrations

```typescript
// src/main/database/migrations/001_add_tags.ts
export async function up(db: Database): Promise<void> {
  await db.run(`
    ALTER TABLE jobs
    ADD COLUMN tags TEXT
  `);

  await db.run(`
    CREATE INDEX idx_jobs_tags ON jobs(tags)
  `);
}

export async function down(db: Database): Promise<void> {
  await db.run(`DROP INDEX idx_jobs_tags`);
  await db.run(`
    ALTER TABLE jobs
    DROP COLUMN tags
  `);
}

// Migration runner
async function runMigrations(db: Database): Promise<void> {
  // Create migrations table
  await db.run(`
    CREATE TABLE IF NOT EXISTS migrations (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  const migrations = [
    { id: 1, name: '001_add_tags', up, down }
  ];

  for (const migration of migrations) {
    const applied = await db.get(
      'SELECT * FROM migrations WHERE name = ?',
      [migration.name]
    );

    if (!applied) {
      await migration.up(db);
      await db.run(
        'INSERT INTO migrations (id, name) VALUES (?, ?)',
        [migration.id, migration.name]
      );
      console.log(`Applied migration: ${migration.name}`);
    }
  }
}
```

## Error Handling

```typescript
class DatabaseError extends Error {
  constructor(
    message: string,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'DatabaseError';
  }
}

async function safeQuery<T>(
  operation: () => Promise<T>,
  errorMessage: string
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    console.error(`Database error: ${errorMessage}`, error);
    throw new DatabaseError(
      errorMessage,
      error instanceof Error ? error : undefined
    );
  }
}

// Usage
const job = await safeQuery(
  () => db.getJob(jobId),
  `Failed to retrieve job ${jobId}`
);
```

## Performance Optimization

### Prepared Statements

```typescript
// For repeated queries, prepare once and reuse
class JobRepository {
  private getJobStmt: sqlite3.Statement;

  constructor(private db: sqlite3.Database) {
    this.getJobStmt = this.db.prepare('SELECT * FROM jobs WHERE id = ?');
  }

  async getJob(id: number): Promise<Job | null> {
    return new Promise((resolve, reject) => {
      this.getJobStmt.get([id], (err, row) => {
        if (err) reject(err);
        else resolve((row as Job) || null);
      });
    });
  }

  cleanup(): void {
    this.getJobStmt.finalize();
  }
}
```

### Batch Inserts

```typescript
async function batchInsertJobs(jobs: NewJob[]): Promise<void> {
  await db.run('BEGIN TRANSACTION');

  try {
    const stmt = db.prepare(`
      INSERT INTO jobs (video_path, video_filename, prompt_name, config_json, status)
      VALUES (?, ?, ?, ?, 'queued')
    `);

    for (const job of jobs) {
      await new Promise<void>((resolve, reject) => {
        stmt.run(
          [job.videoPath, job.videoFilename, job.promptName, JSON.stringify(job.config)],
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      });
    }

    stmt.finalize();
    await db.run('COMMIT');
  } catch (error) {
    await db.run('ROLLBACK');
    throw error;
  }
}
```

## Database Utilities

### Backup

```typescript
async function backupDatabase(backupPath: string): Promise<void> {
  const dbPath = getDatabasePath();
  await fs.copyFile(dbPath, backupPath);
  console.log(`Database backed up to: ${backupPath}`);
}

// Usage in IPC handler
ipcMain.handle('database:backup', async () => {
  const timestamp = new Date().toISOString().replace(/:/g, '-');
  const backupPath = path.join(
    app.getPath('userData'),
    `backup-${timestamp}.db`
  );
  await backupDatabase(backupPath);
  return { backupPath };
});
```

### Vacuum

```typescript
async function vacuumDatabase(db: Database): Promise<void> {
  await db.run('VACUUM');
  console.log('Database vacuumed');
}
```

## Testing

```typescript
// database.test.ts
import { Database } from '../database';
import fs from 'fs/promises';

describe('Database', () => {
  let db: Database;
  const testDbPath = './test.db';

  beforeEach(async () => {
    db = new Database(testDbPath);
    await db.initialize();
  });

  afterEach(async () => {
    await db.close();
    await fs.unlink(testDbPath);
  });

  it('creates a job', async () => {
    const jobId = await db.createJob({
      videoPath: '/test/video.mp4',
      videoFilename: 'video.mp4',
      videoDuration: 45.5,
      promptName: 'test_prompt',
      config: { foo: 'bar' }
    });

    expect(jobId).toBeGreaterThan(0);

    const job = await db.getJob(jobId);
    expect(job).toBeTruthy();
    expect(job!.status).toBe('queued');
  });

  it('updates job status', async () => {
    const jobId = await db.createJob({ /* ... */ });

    await db.updateJobStatus(jobId, 'processing');

    const job = await db.getJob(jobId);
    expect(job!.status).toBe('processing');
    expect(job!.started_at).toBeTruthy();
  });
});
```

## Additional Resources

- SQLite Documentation: https://www.sqlite.org/docs.html
- node-sqlite3: https://github.com/TryGhost/node-sqlite3
- See `PROJECT_KNOWLEDGE.md` for database schema details
- See `TROUBLESHOOTING.md` for common database issues
