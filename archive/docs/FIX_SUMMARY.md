# Fix Summary - Missing Dependencies Issue

**Date:** November 4, 2025
**Issue:** Packaged app fails with "No such file or directory: 'ffmpeg'" and "KeyError: 'basic'"
**Status:** ✅ Solutions Created - Ready to Implement

---

## Problem Analysis

Your friend's error logs show **3 critical missing dependencies** in the packaged app:

1. **❌ ffmpeg binary** - Required for video processing
2. **❌ ffprobe binary** - Required for video metadata
3. **❌ prompts.json** - Missing or incorrectly configured

**Impact:** App completely fails - cannot process any videos

---

## Solutions Created

I've created comprehensive fixes in these files:

### 1. **BUNDLING_GUIDE.md** (~450 lines)
- Complete guide to bundling ffmpeg/ffprobe
- Step-by-step prompts.json verification
- Prevention checklist for future releases

### 2. **scripts/build-with-deps.sh** (Automated build script)
- Downloads ffmpeg/ffprobe automatically
- Verifies all dependencies
- Builds and packages correctly
- **Usage:** `./scripts/build-with-deps.sh`

### 3. **src/python/bundled_resource_paths.py** (Path resolution module)
- Auto-detects bundled binaries and prompts.json
- Works in both development and production
- Provides helpful error messages

### 4. **PYTHON_SCRIPT_FIXES.md** (Implementation guide)
- Exact code changes needed for `video_transcription_pipeline_v04.py`
- Copy-paste ready patches
- Before/after examples

### 5. **TROUBLESHOOTING.md** (Updated)
- New critical section at top
- Quick fixes for already-distributed apps
- Pre-release verification checklist

---

## Quick Fix for Your Friend (IMMEDIATE)

Your friend can fix their existing app **without rebuilding**:

```bash
# Option 1: Install ffmpeg system-wide (Easiest - 2 minutes)
brew install ffmpeg

# That's it! Relaunch the app.
```

Or if they want to patch the app bundle directly:

```bash
# Option 2: Manual patch (5 minutes)
# Download ffmpeg/ffprobe
curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ~/Downloads/ffmpeg.zip
curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ~/Downloads/ffprobe.zip
cd ~/Downloads && unzip ffmpeg.zip && unzip ffprobe.zip

# Copy to app
sudo mkdir -p "/Applications/Gemini Video Understanding.app/Contents/Resources/bin"
sudo cp ffmpeg ffprobe "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
sudo chmod +x "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"*

# Relaunch app
```

Send them either option above - they should be able to fix it immediately.

---

## Long-Term Fix (FOR YOU - Next Build)

### Step 1: Apply Python Script Fixes

**Edit:** `src/python/video_transcription_pipeline_v04.py`

**Add at top:**
```python
from bundled_resource_paths import setup_ffmpeg_environment

# Set up paths for bundled ffmpeg/ffprobe
try:
    FFMPEG_PATH, FFPROBE_PATH = setup_ffmpeg_environment()
    print(f"Using ffmpeg: {FFMPEG_PATH}")
    print(f"Using ffprobe: {FFPROBE_PATH}")
except Exception as e:
    print(f"⚠️  {e}")
    FFMPEG_PATH = 'ffmpeg'  # Fallback
    FFPROBE_PATH = 'ffprobe'
```

**Replace all calls:**
```python
# OLD
subprocess.run(['ffmpeg', ...])
subprocess.run(['ffprobe', ...])

# NEW
subprocess.run([FFMPEG_PATH, ...])
subprocess.run([FFPROBE_PATH, ...])
```

See `PYTHON_SCRIPT_FIXES.md` for complete details.

### Step 2: Use Automated Build Script

```bash
# This does everything:
./scripts/build-with-deps.sh
```

The script will:
1. Download ffmpeg/ffprobe to `resources/bin/`
2. Verify `prompts.json` exists and is valid
3. Build Python venv with all dependencies
4. Build and package the app
5. Verify the final bundle contains everything

### Step 3: Test Before Distributing

```bash
# Mount the DMG
hdiutil attach dist/GeminiVideoUnderstanding-*.dmg

# Verify contents
APP="/Volumes/Gemini Video Understanding/Gemini Video Understanding.app"
ls -lh "$APP/Contents/Resources/bin/ffmpeg"
ls -lh "$APP/Contents/Resources/bin/ffprobe"
ls -lh "$APP/Contents/Resources/python/scripts/prompts.json"

# Test with a SHORT video (30 seconds)
# - Open app
# - Transcribe test video
# - Check Console.app for "✅ Found ffmpeg at:" messages

# Unmount
hdiutil detach "/Volumes/Gemini Video Understanding"
```

---

## Files Reference

### Created/Updated Files:

| File | Purpose | Lines |
|------|---------|-------|
| **BUNDLING_GUIDE.md** | Complete bundling guide | ~450 |
| **scripts/build-with-deps.sh** | Automated build script | ~280 |
| **src/python/bundled_resource_paths.py** | Path resolution module | ~200 |
| **PYTHON_SCRIPT_FIXES.md** | Implementation guide | ~350 |
| **TROUBLESHOOTING.md** | Updated with critical section | +150 |
| **FIX_SUMMARY.md** | This file | ~150 |

### Files That Need Editing:

| File | Changes Needed |
|------|----------------|
| **src/python/video_transcription_pipeline_v04.py** | Import bundled_resource_paths, use FFMPEG_PATH/FFPROBE_PATH |
| **src/python/video_transcription_pipeline_v03.py** | Update PromptManager.get_prompt() with fallback logic |
| **electron-builder.json** | Add resources/bin to extraResources |

---

## Why This Happened

The original packaging didn't include:
1. ffmpeg/ffprobe binaries (assumed they'd be in system PATH)
2. Proper prompts.json bundling
3. Path resolution for bundled resources

Python script was looking for `'ffmpeg'` (expects it in PATH) but Electron apps run in isolated environment without access to user's PATH.

---

## Prevention for Future Releases

**Always use the build script:**
```bash
./scripts/build-with-deps.sh
```

**Pre-release checklist:** (see BUNDLING_GUIDE.md)
1. ✅ Verify ffmpeg/ffprobe in app bundle
2. ✅ Verify prompts.json in app bundle
3. ✅ Test with short video before distributing
4. ✅ Check Console.app for resource resolution messages

---

## Timeline

**Immediate (Today):**
- ✅ Send quick fix to your friend (brew install ffmpeg)
- ✅ They can use app immediately

**Next Build (When you have time):**
- [ ] Apply Python script fixes (30 minutes)
- [ ] Update electron-builder.json (5 minutes)
- [ ] Run ./scripts/build-with-deps.sh (10 minutes)
- [ ] Test packaged app (15 minutes)
- [ ] Distribute new version (5 minutes)

**Total effort:** ~1 hour to permanently fix

---

## Questions?

See these files for details:
- **BUNDLING_GUIDE.md** - Comprehensive bundling guide
- **PYTHON_SCRIPT_FIXES.md** - Code changes needed
- **TROUBLESHOOTING.md** - Diagnostic and fixes
- **scripts/build-with-deps.sh** - Automated solution

Or ask Claude Code for help with specific parts!

---

**Created:** November 4, 2025
**Status:** ✅ Ready to Implement
