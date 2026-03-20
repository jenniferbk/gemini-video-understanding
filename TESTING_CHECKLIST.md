# GVU Release Testing Checklist

## Pre-Installation Testing

- [ ] **DMG Integrity**
  - [ ] DMG mounts without errors
  - [ ] Background image displays correctly
  - [ ] Drag-to-Applications works

- [ ] **File Size Verification**
  - [ ] DMG is ~450-500MB (reasonable size)
  - [ ] App bundle size is appropriate

## Fresh Installation Testing

- [ ] **First Launch**
  - [ ] App opens without Gatekeeper blocking (or shows expected warning)
  - [ ] No crash on startup
  - [ ] Welcome screen appears (if applicable)
  - [ ] API key prompt appears

- [ ] **Bundled Resources**
  - [ ] Python executable exists and is correct version
  - [ ] FFmpeg binary exists and works
  - [ ] FFprobe binary exists and works
  - [ ] Default prompts.json loads

**Verify bundled resources:**
```bash
# Check Python version
"/Applications/Gemini Video Understanding.app/Contents/Resources/python/bin/python3" --version

# Check FFmpeg
"/Applications/Gemini Video Understanding.app/Contents/Resources/bin/ffmpeg" -version

# Check FFprobe
"/Applications/Gemini Video Understanding.app/Contents/Resources/bin/ffprobe" -version

# List Python packages
"/Applications/Gemini Video Understanding.app/Contents/Resources/python/bin/python3" -m pip list
```

## Core Functionality Testing

- [ ] **API Key Management**
  - [ ] Can enter API key
  - [ ] API key persists after restart
  - [ ] Can update API key
  - [ ] Invalid key shows error

- [ ] **Video Upload**
  - [ ] Drag-and-drop works
  - [ ] File browser works
  - [ ] Validates video format (.mp4, .mov, .avi)
  - [ ] Rejects invalid files
  - [ ] Shows video metadata (duration, size)

- [ ] **Prompt Management**
  - [ ] Default prompts load correctly
  - [ ] Can create new prompt
  - [ ] Can edit existing prompt
  - [ ] Can delete prompt
  - [ ] Can export prompt to JSON
  - [ ] Can import prompt from JSON

- [ ] **Configuration Screen**
  - [ ] All presets work (Quick, Standard, High Quality)
  - [ ] Advanced settings toggle works
  - [ ] Chunk duration slider works (1-5 minutes)
  - [ ] VAD checkbox works
  - [ ] Denoising checkbox works
  - [ ] Consensus runs input works (1-10)
  - [ ] Cost estimate updates correctly

- [ ] **Transcription Processing**
  - [ ] **Short video test (2-3 minutes)**
    - [ ] Processing starts without errors
    - [ ] Progress bar updates
    - [ ] Log messages appear
    - [ ] Processing completes successfully
    - [ ] Output file created

  - [ ] **Medium video test (10-15 minutes)**
    - [ ] Processing completes
    - [ ] Multiple chunks processed
    - [ ] Consensus runs work correctly

  - [ ] **Custom prompt test**
    - [ ] Selected prompt is actually used (verify in output)
    - [ ] UUID-selected prompt works (critical bug fix v1.1.6)
    - [ ] No fallback to 'basic' when custom prompt selected

- [ ] **Progress Monitoring**
  - [ ] Progress percentage accurate
  - [ ] Chunk counter updates (X/Y chunks)
  - [ ] Status messages appear
  - [ ] Can cancel mid-process
  - [ ] Cancellation stops Python process

- [ ] **Results Screen**
  - [ ] Transcript displays correctly
  - [ ] Statistics show (chunks, lines, processing time)
  - [ ] "Open Folder" button works
  - [ ] "Copy to Clipboard" works
  - [ ] Can start new transcription

- [ ] **Error Handling**
  - [ ] Invalid API key shows clear error
  - [ ] Network error handled gracefully
  - [ ] Python crash shows error (not silent failure)
  - [ ] Out of disk space handled
  - [ ] Invalid video format rejected

## System Integration Testing

- [ ] **Database**
  - [ ] Job history saved correctly
  - [ ] Settings persist across restarts
  - [ ] Can view past jobs (if implemented)

- [ ] **File System**
  - [ ] Output files saved to correct location
  - [ ] Default output path respected
  - [ ] Can change output path in settings

- [ ] **Performance**
  - [ ] App launches in < 3 seconds
  - [ ] UI remains responsive during processing
  - [ ] Memory usage reasonable (< 2GB during processing)
  - [ ] No memory leaks (check Activity Monitor)

- [ ] **macOS Integration**
  - [ ] App icon displays correctly in Dock
  - [ ] App name correct in menu bar
  - [ ] Quit works properly (doesn't leave orphan processes)
  - [ ] Reopen works after quit

## Regression Testing (v1.1.6 Specific)

- [ ] **Critical Bug Fix: Prompt UUID Matching**
  - [ ] Create custom prompt with UUID
  - [ ] Select it from dropdown
  - [ ] Start transcription
  - [ ] **VERIFY:** Output uses selected prompt (not 'basic')
  - [ ] Check temp prompts file includes UUID fields

**Manual verification:**
```bash
# After starting a transcription with custom prompt, check temp file
ls -lt /tmp/gvu-prompts-*.json | head -1
cat /tmp/gvu-prompts-*.json
# Should see "id" and "uuid" fields for each prompt
```

- [ ] **Prompt Validation (v1.1.5 feature)**
  - [ ] Invalid prompt shows error before processing starts
  - [ ] Error message lists available prompts
  - [ ] No API money wasted on invalid prompts

## Cross-Version Testing

- [ ] **Upgrade from v1.1.5**
  - [ ] Auto-update notification appears (if v1.1.5 installed)
  - [ ] Update downloads and installs
  - [ ] Settings preserved after update
  - [ ] Prompts library preserved after update
  - [ ] API key preserved after update

## Stress Testing

- [ ] **Long Video**
  - [ ] 45+ minute video processes successfully
  - [ ] App doesn't crash during long jobs
  - [ ] Progress updates continue throughout

- [ ] **Multiple Sessions**
  - [ ] Process video → quit → relaunch → process another
  - [ ] No leftover processes after quit
  - [ ] Database doesn't corrupt after multiple uses

## Security Testing

- [ ] **API Key Security**
  - [ ] API key not visible in logs
  - [ ] API key not in plaintext files
  - [ ] API key stored in macOS Keychain (if implemented)

- [ ] **Process Isolation**
  - [ ] Python process runs with correct permissions
  - [ ] Can't access files outside expected directories

## Documentation & User Experience

- [ ] **First-Time User Experience**
  - [ ] Clear what to do on first launch
  - [ ] API key setup is intuitive
  - [ ] Error messages are helpful

- [ ] **README / Help**
  - [ ] Installation instructions accurate
  - [ ] Usage instructions clear
  - [ ] Troubleshooting section helpful

## Clean-Up Testing

- [ ] **Uninstallation**
  - [ ] Can drag app to Trash
  - [ ] Check for leftover files:
    ```bash
    # Application Support
    ls ~/Library/Application\ Support/gemini-video-understanding/

    # Logs
    ls ~/Library/Logs/gemini-video-understanding/

    # Preferences
    ls ~/Library/Preferences/edu.uga.gvu.*
    ```

- [ ] **Data Cleanup**
  - [ ] Temporary files cleaned up after transcription
  - [ ] No orphaned video chunks

---

## Test Environment Details

**Date Tested:** ___________
**Version:** ___________
**macOS Version:** ___________
**Mac Model:** ___________
**RAM:** ___________

**Test Video Used:**
- File: ___________
- Duration: ___________
- Size: ___________
- Format: ___________

**Tester:** ___________

---

## Critical Issues Found

| Issue | Severity | Description | Steps to Reproduce |
|-------|----------|-------------|-------------------|
|       |          |             |                   |

---

## Notes

