# GeminiVideoUnderstanding - Project Guide

## Project Overview

**Desktop application for educational researchers** to transcribe classroom videos using Google Gemini's multimodal AI with speaker diarization.

**Target Users:** 8 research colleagues at University of Georgia (non-technical, Mac users)
**Current Phase:** MVP Development - transforming working Python CLI into user-friendly Electron app
**Tech Stack:** Electron + React 18 + TypeScript, Python 3.11+ (bundled), SQLite3

## Quick Commands

```bash
# Development
npm install              # Install dependencies
npm run dev             # Start Electron with hot reload

# Building
npm run build           # Compile TypeScript
npm run package         # Create .dmg for macOS

# Python Environment (during build)
python3 -m venv src/python/venv
source src/python/venv/bin/activate
pip install -r src/python/requirements.txt
deactivate

# Backend Services (if using PM2)
pnpm pm2:start          # Start all microservices
pnpm pm2:stop           # Stop all services
pm2 logs [service-name] # View specific service logs
```

## Project Documentation

- **Architecture & Integration:** See `PROJECT_KNOWLEDGE.md`
- **Common Issues & Debugging:** See `TROUBLESHOOTING.md`
- **Coding Guidelines:** See Skills (automatically activated based on context)

## Skills Available

Use these skills for detailed implementation guidance:

- `/electron-dev-guidelines` - IPC handlers, security, preload scripts
- `/python-integration` - Bundling Python with Electron, process management
- `/react-typescript-patterns` - Component structure, hooks, error handling
- `/database-operations` - SQLite operations, schema management

## Task Management Workflow

### Starting Large Tasks

When exiting plan mode with an accepted plan:

1. **Create Task Directory:**
   ```bash
   mkdir -p ~/Documents/COMS/dev/active/[task-name]/
   ```

2. **Create Documents:**
   - `[task-name]-plan.md` - The accepted plan
   - `[task-name]-context.md` - Key files, decisions, architectural notes
   - `[task-name]-tasks.md` - Checklist of work items

3. **Update Regularly:**
   - Mark tasks complete immediately after finishing
   - Update context file with relevant decisions and file paths
   - Update "Last Updated" timestamps

### Continuing Tasks

- Check `/dev/active/` for existing tasks
- Read all three files before proceeding with work
- Run custom command `/update-dev-docs` before compacting conversation
- When resuming: Just say "continue" after compaction with dev docs updated

## Service-Specific Configuration

### Python Integration

**Script Location:**
- Development: `src/python/video_transcription_pipeline_v04.py`
- Production: App resources bundle

**JSON Progress Format:**
The Python script outputs JSON on stdout for the Electron app:
```json
{"type": "progress", "chunk": 5, "total": 16, "percent": 31, "status": "processing"}
{"type": "complete", "output_file": "/path/to/transcript.txt", "stats": {...}}
{"type": "error", "message": "Error description", "fatal": true}
```

Prefix markers: `GVU_PROGRESS:`, `GVU_COMPLETE:`, `GVU_ERROR:`

**De-identification (optional, off by default):** `--deidentify-names` runs a second Gemini pass after transcription to detect real names (students and adults) and substitute realistic pseudonyms (`Student-Hannah`, `Ms. Kelly`, etc.). Visual-description labels linked to a real name are retired in favor of the pseudonym. Writes `transcript_name_map.json` alongside the transcript as an audit trail. Module: `deidentify_names.py`; curated pool: `pseudonym_pool.json`. The name-map audit file contains the real-name↔pseudonym mapping and should be stored under separate access control from the deidentified transcript. **Caveat:** combining `--deidentify-names` with `--keep-chunks` retains PII in per-chunk files (chunks are written during transcription, before the de-identification pass); omit `--keep-chunks` when privacy matters.

### Database Location

- Development: `database/gvu.db` (created on first run)
- Production: `~/Library/Application Support/GeminiVideoUnderstanding/gvu.db`

### API Key Storage

**macOS Keychain via keytar:**
- Service: `GeminiVideoUnderstanding`
- Account: `gemini-api-key`
- See `src/main/utils/keychain.ts` for implementation

## Project-Specific Quirks

### Electron Builder Setup

Important: Python venv must be created **before** running `npm run package`. The build process:
1. Packages the entire `src/python/venv` directory into app resources
2. Includes `video_transcription_pipeline_v04.py` and related scripts
3. App uses different Python paths in dev vs production (see `src/main/python/pythonRunner.ts`)

### File Structure Note

This is a **monorepo** structure:
- `/frontend` - React Electron renderer
- `/backend` - Multiple microservices (if split into separate repos)
- Current approach: Single Electron app bundling everything

### Testing Authenticated Routes

Use the authentication testing script:
```bash
node scripts/test-auth-route.js http://localhost:3002/api/endpoint
```

This handles Keycloak token retrieval, JWT signing, and cookie headers automatically.

## Development Workflow

### Typical Feature Implementation

1. **Plan** - Use planning mode or strategic-plan-architect
2. **Create Dev Docs** - Plan, context, tasks files in `/dev/active/`
3. **Implement** - Work in chunks, update tasks as you go
4. **Review** - Run builds, check for TypeScript errors
5. **Update Dev Docs** - Before compacting, capture next steps
6. **Continue** - Resume with "continue" in new session

### Quality Checks

After making changes:
- Run `npm run build` in affected repos
- Check for TypeScript errors
- Test affected routes/features
- Update documentation if architecture changed

## Success Criteria

### MVP Complete When:

1. ✅ User can install app with zero configuration (except API key)
2. ✅ User can drag-drop video and complete transcription with 3 clicks
3. ✅ Progress bar accurately reflects processing status
4. ✅ Output transcript is RTF-compatible for Transana import
5. ✅ Prompt library supports create/edit/import/export
6. ✅ App auto-updates when new version available
7. ✅ 8 colleagues successfully use app for 2 weeks without technical support

### Quality Benchmarks:

- **Startup time:** < 3 seconds
- **Processing overhead:** < 5% slower than direct Python script
- **Memory usage:** < 500MB idle, < 2GB during processing
- **Crash rate:** < 1% of transcription jobs

## Current Status & Next Steps

**Latest Version:** v1.1.3
**Recent Changes:** Fixed prompt editing, custom prompts file path support

**Priority Tasks:**
- Complete UI for all screens (Upload → Config → Progress → Results)
- Implement prompt manager functionality
- Add comprehensive error handling throughout
- Set up electron-updater for auto-updates
- Create installer DMG with proper signing

## Contact & Support

**Project Lead:** Jennifer (UGA COMS Research)
**Repository:** [Add GitHub URL when ready]
**Issues:** GitHub Issues for bug reports and feature requests

---

**Remember:** This CLAUDE.md contains project-specific information only. For detailed coding patterns and best practices, Skills will auto-activate based on the files you're working with. For deep architectural details, refer to PROJECT_KNOWLEDGE.md.
