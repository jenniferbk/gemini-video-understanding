# Bundling Guide - Critical Dependencies

This guide explains how to bundle ffmpeg/ffprobe and prompts.json with your Electron app.

## Problem Summary

The packaged app is missing:
1. **ffmpeg/ffprobe** binaries (video processing tools)
2. **prompts.json** file (transcription prompts)

This causes errors like:
- `[Errno 2] No such file or directory: 'ffmpeg'`
- `KeyError: 'basic'` (missing prompt)

---

## Solution 1: Bundle ffmpeg/ffprobe

### Step 1: Download Static Binaries

**macOS (recommended):**
```bash
# Create directory for binaries
mkdir -p resources/bin

# Download ffmpeg static build from official sources
# Option A: Using Homebrew
brew install ffmpeg

# Copy binaries to resources
cp $(which ffmpeg) resources/bin/ffmpeg
cp $(which ffprobe) resources/bin/ffprobe

# Option B: Download from evermeet.cx (static builds for macOS)
cd resources/bin
curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ffmpeg.zip
curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ffprobe.zip
unzip ffmpeg.zip
unzip ffprobe.zip
rm *.zip

# Make executable
chmod +x ffmpeg ffprobe
```

### Step 2: Update electron-builder.json

```json
{
  "build": {
    "appId": "edu.uga.gvu",
    "productName": "Gemini Video Understanding",
    "extraResources": [
      {
        "from": "src/python/venv",
        "to": "python",
        "filter": ["**/*", "!**/*.pyc", "!**/__pycache__"]
      },
      {
        "from": "src/python",
        "to": "python/scripts",
        "filter": ["*.py", "prompts.json", "requirements.txt"]
      },
      {
        "from": "resources/bin",
        "to": "bin",
        "filter": ["ffmpeg", "ffprobe"]
      }
    ]
  }
}
```

### Step 3: Update Python Script to Find ffmpeg/ffprobe

Add this to the top of `video_transcription_pipeline_v04.py`:

```python
import os
import sys
from pathlib import Path

def get_bundled_binary(binary_name: str) -> str:
    """
    Find ffmpeg/ffprobe binary in bundled app or system PATH.

    Search order:
    1. Bundled with app (for production)
    2. System PATH (for development)
    """
    # Check if running from bundled app
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle or Electron app
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent

        # For Electron app: resources/bin/
        possible_paths = [
            bundle_dir.parent.parent / 'bin' / binary_name,  # ../../../bin/ffmpeg
            bundle_dir / 'bin' / binary_name,
            Path('/Applications/Gemini Video Understanding.app/Contents/Resources/bin') / binary_name,
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Found {binary_name} at: {path}")
                return str(path)

    # Fall back to system PATH
    import shutil
    system_binary = shutil.which(binary_name)
    if system_binary:
        print(f"✅ Found {binary_name} in system PATH: {system_binary}")
        return system_binary

    raise FileNotFoundError(
        f"❌ {binary_name} not found. Please install ffmpeg or bundle it with the app.\n"
        f"Install: brew install ffmpeg (macOS) or https://ffmpeg.org/download.html"
    )

# Set environment variables for ffmpeg/ffprobe
try:
    FFMPEG_PATH = get_bundled_binary('ffmpeg')
    FFPROBE_PATH = get_bundled_binary('ffprobe')

    # Update PATH so subprocess calls can find them
    os.environ['PATH'] = str(Path(FFMPEG_PATH).parent) + os.pathsep + os.environ['PATH']
except FileNotFoundError as e:
    print(f"⚠️ {e}")
    FFMPEG_PATH = 'ffmpeg'  # Fallback to hoping it's in PATH
    FFPROBE_PATH = 'ffprobe'
```

Then update all calls to use the resolved paths:

```python
# OLD
subprocess.run(['ffmpeg', '-i', video_path, ...])
subprocess.run(['ffprobe', '-v', 'error', ...])

# NEW
subprocess.run([FFMPEG_PATH, '-i', video_path, ...])
subprocess.run([FFPROBE_PATH, '-v', 'error', ...])
```

---

## Solution 2: Bundle prompts.json Correctly

### Step 1: Verify prompts.json Exists and is Valid

```bash
# Check the file exists
ls -la src/python/prompts.json

# Verify JSON is valid
python3 -m json.tool src/python/prompts.json > /dev/null && echo "✅ Valid JSON"

# Check it contains the prompts being used
cat src/python/prompts.json | grep -E "basic|enhanced_vad"
```

### Step 2: Ensure It Has Required Prompts

Your `prompts.json` MUST contain at least these keys:

```json
{
  "basic": {
    "name": "Basic Transcription",
    "prompt": "Please transcribe this classroom video..."
  },
  "enhanced_vad": {
    "name": "Enhanced VAD Transcription",
    "prompt": "Please transcribe this classroom video with careful attention to speaker changes..."
  }
}
```

### Step 3: Update Python Script to Find prompts.json

Add this function to `video_transcription_pipeline_v04.py`:

```python
def get_bundled_prompts_path() -> Path:
    """
    Find prompts.json in bundled app or development location.
    """
    if getattr(sys, 'frozen', False):
        # Running in bundled app
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent

        possible_paths = [
            bundle_dir / 'prompts.json',
            bundle_dir / 'scripts' / 'prompts.json',
            Path('/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/prompts.json'),
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Found prompts.json at: {path}")
                return path

    # Development: relative to script
    script_dir = Path(__file__).parent
    prompts_path = script_dir / 'prompts.json'

    if prompts_path.exists():
        return prompts_path

    raise FileNotFoundError(
        "❌ prompts.json not found. Please ensure it's bundled with the app."
    )

# Use in PromptManager initialization
prompts_file = get_bundled_prompts_path()
prompt_manager = PromptManager(prompts_file)
```

### Step 4: Add Fallback Prompt

In `video_transcription_pipeline_v03.py` (or wherever PromptManager is), update the `get_prompt` method:

```python
def get_prompt(self, key: str) -> Dict[str, Any]:
    """
    Get a prompt by key, with fallback to basic prompt.
    """
    if key in self.prompts:
        return self.prompts[key]

    # Try to find by UUID (for saved user prompts)
    for prompt_key, prompt_data in self.prompts.items():
        if prompt_data.get('id') == key or prompt_data.get('uuid') == key:
            return prompt_data

    # Fallback to first available prompt
    if self.prompts:
        fallback_key = list(self.prompts.keys())[0]
        print(f"⚠️ Prompt '{key}' not found. Using '{fallback_key}' instead.")
        return self.prompts[fallback_key]

    # Last resort: inline basic prompt
    print(f"⚠️ No prompts available. Using inline basic prompt.")
    return {
        "name": "Basic (Inline)",
        "prompt": "Please transcribe this classroom video accurately, identifying each speaker."
    }
```

---

## Solution 3: Test Before Packaging

### Pre-Package Checklist

```bash
# 1. Verify ffmpeg/ffprobe are in resources/bin/
ls -la resources/bin/
# Should show: ffmpeg, ffprobe

# 2. Verify prompts.json exists and is valid
python3 -m json.tool src/python/prompts.json

# 3. Test Python script can find resources
cd src/python
source venv/bin/activate
python video_transcription_pipeline_v04.py --help

# 4. Check electron-builder config
cat electron-builder.json | grep -A 20 extraResources
```

### Build and Test

```bash
# Build the app
npm run build
npm run package

# Test the packaged app
# Open it, try to transcribe a short video
# Check the logs for "✅ Found ffmpeg at:" messages
```

---

## Solution 4: Quick Fix for Your Friend

If your friend already has the packaged app and it's failing, they can:

### Option A: Install ffmpeg System-Wide

```bash
# macOS
brew install ffmpeg

# This puts ffmpeg/ffprobe in PATH
# The Python script should find them
```

### Option B: Copy Binaries to App

```bash
# Download ffmpeg/ffprobe
curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ~/Downloads/ffmpeg.zip
curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ~/Downloads/ffprobe.zip

cd ~/Downloads
unzip ffmpeg.zip
unzip ffprobe.zip

# Copy to app bundle
sudo cp ffmpeg "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
sudo cp ffprobe "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"

# Make executable
sudo chmod +x "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/ffmpeg"
sudo chmod +x "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/ffprobe"
```

### Option C: Check prompts.json

```bash
# Verify prompts.json exists in app
ls -la "/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/prompts.json"

# If missing, copy from source
cp /path/to/project/src/python/prompts.json "/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/"
```

---

## Prevention: Automated Build Script

Create `scripts/build-with-deps.sh`:

```bash
#!/bin/bash
set -e

echo "🔧 Preparing dependencies for packaging..."

# 1. Download ffmpeg/ffprobe
echo "📦 Downloading ffmpeg/ffprobe..."
mkdir -p resources/bin
cd resources/bin

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ffmpeg.zip
    curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ffprobe.zip
    unzip -o ffmpeg.zip
    unzip -o ffprobe.zip
    rm *.zip
fi

chmod +x ffmpeg ffprobe
cd ../..

# 2. Verify prompts.json
echo "✅ Verifying prompts.json..."
python3 -m json.tool src/python/prompts.json > /dev/null || exit 1

# 3. Build Python venv
echo "🐍 Building Python virtual environment..."
rm -rf src/python/venv
python3 -m venv src/python/venv
source src/python/venv/bin/activate
pip install -r src/python/requirements.txt
deactivate

# 4. Build Electron app
echo "⚡ Building Electron app..."
npm run build

# 5. Package
echo "📦 Packaging..."
npm run package

echo "✅ Build complete!"
echo "Output: dist/Gemini Video Understanding-*.dmg"
```

Make it executable:
```bash
chmod +x scripts/build-with-deps.sh
```

Use it:
```bash
./scripts/build-with-deps.sh
```

---

## Verification After Packaging

After creating the `.dmg`, verify the bundle contains everything:

```bash
# Mount the DMG (if needed)
# Then check the app contents

# 1. Check ffmpeg/ffprobe
ls -la "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
# Should show: ffmpeg, ffprobe

# 2. Check prompts.json
cat "/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/prompts.json"

# 3. Check Python is bundled
ls -la "/Applications/Gemini Video Understanding.app/Contents/Resources/python/bin/python3"

# 4. Test the app
# Open it and try a short video transcription
```

---

## Summary

**Root causes:**
1. ffmpeg/ffprobe not bundled → Can't process video
2. prompts.json missing/incomplete → Can't generate transcripts
3. Python script doesn't know where to find bundled resources

**Fixes:**
1. Add ffmpeg/ffprobe to `resources/bin/` before packaging
2. Verify prompts.json has all required prompts
3. Update Python script with path resolution functions
4. Add better error messages with installation instructions

**For next build:**
Use the automated build script: `./scripts/build-with-deps.sh`

---

**Last Updated:** November 4, 2025
