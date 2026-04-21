#!/bin/bash

# Quick test script to verify v1.1.6 prompt UUID fix
# This verifies that the temp prompts file includes UUID fields

set -e

echo "🧪 Testing v1.1.6 Prompt UUID Fix"
echo "================================="
echo ""

# Check if app is installed
APP_PATH="/Applications/Gemini Video Understanding.app"
if [ ! -d "$APP_PATH" ]; then
    echo "❌ App not found at $APP_PATH"
    echo "Please install the DMG first"
    exit 1
fi

echo "✅ App found at $APP_PATH"
echo ""

# Check bundled Python
PYTHON_PATH="$APP_PATH/Contents/Resources/python/bin/python3"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Python not found at $PYTHON_PATH"
    exit 1
fi

echo "✅ Bundled Python found"
PYTHON_VERSION=$("$PYTHON_PATH" --version)
echo "   Version: $PYTHON_VERSION"
echo ""

# Check bundled FFmpeg
FFMPEG_PATH="$APP_PATH/Contents/Resources/bin/ffmpeg"
if [ ! -f "$FFMPEG_PATH" ]; then
    echo "❌ FFmpeg not found at $FFMPEG_PATH"
    exit 1
fi

echo "✅ Bundled FFmpeg found"
FFMPEG_VERSION=$("$FFMPEG_PATH" -version | head -1)
echo "   $FFMPEG_VERSION"
echo ""

# Check bundled FFprobe
FFPROBE_PATH="$APP_PATH/Contents/Resources/bin/ffprobe"
if [ ! -f "$FFPROBE_PATH" ]; then
    echo "❌ FFprobe not found at $FFPROBE_PATH"
    exit 1
fi

echo "✅ Bundled FFprobe found"
echo ""

# Check Python script
SCRIPT_PATH="$APP_PATH/Contents/Resources/python/scripts/video_transcription_pipeline_v04.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Python script not found at $SCRIPT_PATH"
    exit 1
fi

echo "✅ Python script found"
echo ""

# Check user prompts file
USER_PROMPTS="$HOME/Library/Application Support/gemini-video-understanding/prompts.json"
if [ -f "$USER_PROMPTS" ]; then
    echo "✅ User prompts file exists"
    echo "   Location: $USER_PROMPTS"

    # Check if prompts have UUIDs
    if grep -q '"id"' "$USER_PROMPTS"; then
        echo "   ✅ Prompts have 'id' field (UUID)"
    else
        echo "   ⚠️  Prompts missing 'id' field"
    fi
    echo ""
else
    echo "ℹ️  No user prompts file yet (normal for fresh install)"
    echo ""
fi

# Check for recent temp prompts file (if a transcription was just run)
echo "🔍 Checking for recent temp prompts files..."
TEMP_PROMPTS=$(ls -t /tmp/gvu-prompts-*.json 2>/dev/null | head -1)

if [ -n "$TEMP_PROMPTS" ]; then
    echo "✅ Found temp prompts file: $TEMP_PROMPTS"
    echo ""
    echo "📄 Contents (first 20 lines):"
    head -20 "$TEMP_PROMPTS"
    echo ""

    # Check for UUID fields
    if grep -q '"id"' "$TEMP_PROMPTS"; then
        echo "✅ PASS: Temp prompts file includes 'id' field (UUID)"
    else
        echo "❌ FAIL: Temp prompts file missing 'id' field (UUID)"
        echo "This means the v1.1.6 fix is NOT working!"
    fi

    if grep -q '"uuid"' "$TEMP_PROMPTS"; then
        echo "✅ PASS: Temp prompts file includes 'uuid' field"
    else
        echo "❌ FAIL: Temp prompts file missing 'uuid' field"
        echo "This means the v1.1.6 fix is NOT working!"
    fi
else
    echo "ℹ️  No temp prompts file found (run a transcription to generate one)"
    echo ""
    echo "To test the UUID fix:"
    echo "1. Open the app"
    echo "2. Select a custom prompt from the dropdown"
    echo "3. Start a transcription (can cancel after it starts)"
    echo "4. Run this script again"
fi

echo ""
echo "================================="
echo "✅ Basic bundle verification complete!"
echo ""
echo "For full testing, see TESTING_CHECKLIST.md"
