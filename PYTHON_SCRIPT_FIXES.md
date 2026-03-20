# Python Script Fixes - Apply to video_transcription_pipeline_v04.py

These fixes resolve the missing ffmpeg/ffprobe and prompts.json errors.

## Quick Fix Summary

1. Import the bundled resource path module
2. Update ffmpeg/ffprobe calls to use bundled binaries
3. Fix prompts.json path resolution
4. Add better error handling for missing prompts

---

## Fix 1: Import Bundled Resource Module

**Add this at the top of `video_transcription_pipeline_v04.py`:**

```python
# Add after other imports (around line 10-20)
from bundled_resource_paths import (
    get_bundled_binary,
    get_bundled_prompts_path,
    setup_ffmpeg_environment
)
```

---

## Fix 2: Set Up ffmpeg/ffprobe Paths

**Add this after imports, before any function definitions:**

```python
# Set up paths for bundled ffmpeg/ffprobe
try:
    FFMPEG_PATH, FFPROBE_PATH = setup_ffmpeg_environment()
    print(f"Using ffmpeg: {FFMPEG_PATH}")
    print(f"Using ffprobe: {FFPROBE_PATH}")
except Exception as e:
    print(f"⚠️  Could not locate ffmpeg/ffprobe: {e}")
    print(f"⚠️  Falling back to system PATH")
    FFMPEG_PATH = 'ffmpeg'
    FFPROBE_PATH = 'ffprobe'
```

---

## Fix 3: Update All subprocess Calls

**Find all calls to ffmpeg/ffprobe and update them:**

### Before:
```python
subprocess.run(['ffmpeg', '-i', video_path, ...])
subprocess.run(['ffprobe', '-v', 'error', ...])
```

### After:
```python
subprocess.run([FFMPEG_PATH, '-i', video_path, ...])
subprocess.run([FFPROBE_PATH, '-v', 'error', ...])
```

**Common locations to update:**
- `get_video_duration()` function
- `split_video_with_vad()` function
- Any `ffmpeg`/`ffprobe` calls in chunking code

---

## Fix 4: Update PromptManager Initialization

**Find the PromptManager initialization (look for where it's imported/created):**

### Current (likely in video_transcription_pipeline_v03.py):
```python
class PromptManager:
    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = prompts_file
        # ...
```

### Update to:
```python
class PromptManager:
    def __init__(self, prompts_file: str = None):
        # Auto-detect prompts.json location if not specified
        if prompts_file is None:
            try:
                from bundled_resource_paths import get_bundled_prompts_path
                prompts_file = str(get_bundled_prompts_path())
            except Exception as e:
                print(f"⚠️  Could not auto-locate prompts.json: {e}")
                prompts_file = "prompts.json"  # Fallback

        self.prompts_file = prompts_file
        # ... rest of __init__
```

---

## Fix 5: Add Fallback Prompt Handling

**Update the `get_prompt()` method in PromptManager:**

### Current:
```python
def get_prompt(self, key: str) -> Dict[str, Any]:
    return self.prompts[key]["prompt"]
```

### Update to:
```python
def get_prompt(self, key: str) -> Dict[str, Any]:
    """
    Get a prompt by key, with intelligent fallback.
    """
    # Direct key match
    if key in self.prompts:
        return self.prompts[key]

    # Try to find by UUID (for user-created prompts)
    for prompt_key, prompt_data in self.prompts.items():
        if isinstance(prompt_data, dict):
            if prompt_data.get('id') == key or prompt_data.get('uuid') == key:
                print(f"✅ Found prompt by UUID: {key} -> {prompt_key}")
                return prompt_data

    # Fallback to first available prompt
    if self.prompts:
        fallback_key = list(self.prompts.keys())[0]
        print(f"⚠️  Prompt '{key}' not found. Using '{fallback_key}' instead.")
        return self.prompts[fallback_key]

    # Last resort: inline basic prompt
    print(f"⚠️  No prompts available in prompts.json. Using inline basic prompt.")
    return {
        "name": "Basic (Inline Fallback)",
        "prompt": (
            "Please transcribe this classroom video accurately. "
            "For each line, identify the speaker (Teacher, Student1, Student2, etc.) "
            "and provide the timestamp. Use this format:\n\n"
            "MM:SS SpeakerName: Text of what was said\n\n"
            "Example:\n"
            "00:15 Teacher: Let's begin today's lesson.\n"
            "00:23 Student1: Can I ask a question?"
        )
    }
```

---

## Fix 6: Better Error Messages

**Update error handling when video processing fails:**

### Add this helper function:
```python
def diagnose_missing_dependencies():
    """Print diagnostic information about missing dependencies."""
    print("\n" + "="*70)
    print("🔍 DEPENDENCY DIAGNOSTIC")
    print("="*70)

    # Check ffmpeg
    try:
        ffmpeg_result = subprocess.run(
            [FFMPEG_PATH, '-version'],
            capture_output=True,
            timeout=5
        )
        if ffmpeg_result.returncode == 0:
            print("✅ ffmpeg: Available")
        else:
            print("❌ ffmpeg: Found but not working")
    except FileNotFoundError:
        print("❌ ffmpeg: NOT FOUND")
        print("   Install: brew install ffmpeg (macOS)")
    except Exception as e:
        print(f"❌ ffmpeg: Error - {e}")

    # Check ffprobe
    try:
        ffprobe_result = subprocess.run(
            [FFPROBE_PATH, '-version'],
            capture_output=True,
            timeout=5
        )
        if ffprobe_result.returncode == 0:
            print("✅ ffprobe: Available")
        else:
            print("❌ ffprobe: Found but not working")
    except FileNotFoundError:
        print("❌ ffprobe: NOT FOUND")
        print("   Install: brew install ffmpeg (macOS)")
    except Exception as e:
        print(f"❌ ffprobe: Error - {e}")

    # Check prompts.json
    try:
        prompts_path = get_bundled_prompts_path()
        import json
        with open(prompts_path) as f:
            prompts = json.load(f)
        print(f"✅ prompts.json: Found with {len(prompts)} prompts")
    except FileNotFoundError:
        print("❌ prompts.json: NOT FOUND")
    except Exception as e:
        print(f"❌ prompts.json: Error - {e}")

    print("="*70 + "\n")
```

**Call this function when errors occur:**
```python
# In the main() function exception handler:
except Exception as e:
    print(f"\n❌ V04 Error: {e}")
    traceback.print_exc()

    # Add diagnostic
    diagnose_missing_dependencies()

    sys.exit(1)
```

---

## Complete Patch Template

**Here's a complete example of what to add at the top of video_transcription_pipeline_v04.py:**

```python
#!/usr/bin/env python3
"""
Video Transcription Pipeline V04
Enhanced with Hybrid VAD + Classroom AI
"""

import sys
import os
import json
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# === BUNDLED RESOURCE FIXES ===
# Import bundled resource path resolution
try:
    from bundled_resource_paths import (
        get_bundled_binary,
        get_bundled_prompts_path,
        setup_ffmpeg_environment
    )

    # Set up ffmpeg/ffprobe paths
    FFMPEG_PATH, FFPROBE_PATH = setup_ffmpeg_environment()
    print(f"✅ Using bundled ffmpeg: {FFMPEG_PATH}")
    print(f"✅ Using bundled ffprobe: {FFPROBE_PATH}")

except ImportError:
    print("⚠️  bundled_resource_paths module not found, using system PATH")
    FFMPEG_PATH = 'ffmpeg'
    FFPROBE_PATH = 'ffprobe'
except Exception as e:
    print(f"⚠️  Error setting up bundled resources: {e}")
    print(f"⚠️  Falling back to system PATH")
    FFMPEG_PATH = 'ffmpeg'
    FFPROBE_PATH = 'ffprobe'

# ... rest of imports ...
```

---

## Testing After Fixes

1. **Test in development:**
   ```bash
   cd src/python
   source venv/bin/activate
   python bundled_resource_paths.py  # Test path resolution
   python video_transcription_pipeline_v04.py --help
   ```

2. **Test subprocess calls:**
   ```python
   import subprocess
   subprocess.run([FFMPEG_PATH, '-version'])
   subprocess.run([FFPROBE_PATH, '-version'])
   ```

3. **Rebuild app:**
   ```bash
   ./scripts/build-with-deps.sh
   ```

4. **Test packaged app:**
   - Open the app
   - Try transcribing a short video
   - Check Console.app for "✅ Found ffmpeg at:" messages

---

## Alternative: Quick Manual Fix for Existing App

If you don't want to rebuild, you can manually fix an already-packaged app:

```bash
# 1. Install ffmpeg system-wide
brew install ffmpeg

# 2. Copy prompts.json to the app
# Create a basic prompts.json with at least one prompt:
cat > /tmp/prompts.json << 'EOF'
{
  "basic": {
    "name": "Basic Transcription",
    "prompt": "Please transcribe this classroom video accurately, identifying each speaker."
  }
}
EOF

# 3. Copy to app bundle
cp /tmp/prompts.json "/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/"

# 4. Relaunch app
```

---

**Last Updated:** November 4, 2025
