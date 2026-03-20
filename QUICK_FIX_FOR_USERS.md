# Quick Fix - App Won't Process Videos

**Problem:** App fails with errors like:
- "No such file or directory: 'ffmpeg'"
- "KeyError: 'basic'"
- "Could not determine video duration"

**Solution:** Install ffmpeg (2 minutes)

---

## Fix (macOS)

Open Terminal and run:

```bash
brew install ffmpeg
```

If you don't have Homebrew installed, first install it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install ffmpeg:

```bash
brew install ffmpeg
```

**That's it!** Relaunch Gemini Video Understanding and try again.

---

## Alternative: Manual Patch (if brew doesn't work)

```bash
# 1. Download ffmpeg/ffprobe
curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ~/Downloads/ffmpeg.zip
curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ~/Downloads/ffprobe.zip

# 2. Unzip
cd ~/Downloads
unzip ffmpeg.zip
unzip ffprobe.zip

# 3. Copy to app bundle
sudo mkdir -p "/Applications/Gemini Video Understanding.app/Contents/Resources/bin"
sudo cp ffmpeg "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
sudo cp ffprobe "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"
sudo chmod +x "/Applications/Gemini Video Understanding.app/Contents/Resources/bin/"*

# 4. Create prompts file
cat > /tmp/prompts.json << 'EOF'
{
  "basic": {
    "name": "Basic Transcription",
    "prompt": "Please transcribe this classroom video accurately, identifying each speaker (Teacher, Student1, Student2, etc.) with timestamps."
  }
}
EOF

sudo cp /tmp/prompts.json "/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/"

# 5. Relaunch app
```

---

## Verification

After applying the fix, you should see:
- ✅ Video processes without errors
- ✅ Progress bar updates
- ✅ Transcription completes successfully

If you check Console.app (open it → search for "Gemini"), you should see:
```
✅ Found ffmpeg at: /opt/homebrew/bin/ffmpeg
✅ Found ffprobe at: /opt/homebrew/bin/ffprobe
```

---

## Still Having Issues?

Contact the developer with:
1. Screenshot of the error
2. Output from: `which ffmpeg`
3. Output from: `ffmpeg -version`

---

**This is a temporary fix. A new version with bundled dependencies will be released soon.**
