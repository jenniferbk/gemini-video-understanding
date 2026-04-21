#!/bin/bash

# Ultra-quick test to verify v1.1.6 UUID fix is working
# This checks if your current installation has the fix

set -e

echo "🔍 Quick UUID Fix Verification"
echo "==============================="
echo ""

# Check if app is running
APP_NAME="Gemini Video Understanding"
if pgrep -f "$APP_NAME" > /dev/null; then
    echo "⚠️  App is currently running"
    echo "   For best results, quit the app and run this again"
    echo ""
fi

# Check for recent temp prompts file
echo "Looking for temp prompts file..."
TEMP_PROMPTS=$(ls -t /tmp/gvu-prompts-*.json 2>/dev/null | head -1)

if [ -z "$TEMP_PROMPTS" ]; then
    echo "❌ No temp prompts file found"
    echo ""
    echo "This means you haven't started a transcription yet."
    echo ""
    echo "To test the UUID fix:"
    echo "  1. Open Gemini Video Understanding"
    echo "  2. Select any prompt from the dropdown"
    echo "  3. Start a transcription (you can cancel it immediately)"
    echo "  4. Run this script again"
    echo ""
    echo "The temp file is created when you start a transcription."
    exit 0
fi

echo "✅ Found: $TEMP_PROMPTS"
echo ""

# Check if it has UUID fields
HAS_ID=$(grep -c '"id"' "$TEMP_PROMPTS" || true)
HAS_UUID=$(grep -c '"uuid"' "$TEMP_PROMPTS" || true)

echo "📄 Checking temp prompts file for UUID fields..."
echo ""

if [ "$HAS_ID" -gt 0 ]; then
    echo "✅ PASS: Found 'id' field ($HAS_ID occurrences)"
else
    echo "❌ FAIL: Missing 'id' field"
fi

if [ "$HAS_UUID" -gt 0 ]; then
    echo "✅ PASS: Found 'uuid' field ($HAS_UUID occurrences)"
else
    echo "❌ FAIL: Missing 'uuid' field"
fi

echo ""

if [ "$HAS_ID" -gt 0 ] && [ "$HAS_UUID" -gt 0 ]; then
    echo "✅✅✅ SUCCESS! v1.1.6 UUID fix is working! ✅✅✅"
    echo ""
    echo "Your prompts now include UUID fields, so Python can find them correctly."
    echo "Custom prompts will no longer fall back to 'basic'."
else
    echo "❌❌❌ FAILURE! v1.1.6 UUID fix is NOT working! ❌❌❌"
    echo ""
    echo "The temp prompts file is missing UUID fields."
    echo "This means:"
    echo "  - Either you're running an old version of the app"
    echo "  - Or the fix didn't get packaged correctly"
    echo ""
    echo "Current app version should be 1.1.6"
    echo "Check: /Applications/Gemini Video Understanding.app/Contents/Info.plist"
fi

echo ""
echo "File analyzed: $TEMP_PROMPTS"
echo ""
echo "To see the full contents:"
echo "  cat '$TEMP_PROMPTS' | python3 -m json.tool"
