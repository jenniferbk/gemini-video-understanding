#!/bin/bash
# Automated build script for Gemini Video Understanding
# Ensures all dependencies (ffmpeg, prompts.json, Python venv) are bundled correctly

set -e  # Exit on any error

echo "================================================================================"
echo "🚀 Building Gemini Video Understanding with All Dependencies"
echo "================================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "📂 Project root: $PROJECT_ROOT"
echo ""

# ==============================================================================
# Step 1: Download and Bundle ffmpeg/ffprobe
# ==============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Step 1: Bundling ffmpeg/ffprobe"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p resources/bin

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS - downloading static ffmpeg/ffprobe..."

    cd resources/bin

    # Download ffmpeg
    if [ ! -f "ffmpeg" ]; then
        echo "   Downloading ffmpeg..."
        curl -L https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip -o ffmpeg.zip
        unzip -o ffmpeg.zip
        rm ffmpeg.zip
        chmod +x ffmpeg
        echo -e "   ${GREEN}✅ ffmpeg downloaded${NC}"
    else
        echo -e "   ${GREEN}✅ ffmpeg already exists${NC}"
    fi

    # Download ffprobe
    if [ ! -f "ffprobe" ]; then
        echo "   Downloading ffprobe..."
        curl -L https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip -o ffprobe.zip
        unzip -o ffprobe.zip
        rm ffprobe.zip
        chmod +x ffprobe
        echo -e "   ${GREEN}✅ ffprobe downloaded${NC}"
    else
        echo -e "   ${GREEN}✅ ffprobe already exists${NC}"
    fi

    cd "$PROJECT_ROOT"

    # Verify binaries work
    echo "   Testing ffmpeg..."
    if ./resources/bin/ffmpeg -version > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ ffmpeg is working${NC}"
    else
        echo -e "   ${RED}❌ ffmpeg test failed${NC}"
        exit 1
    fi

    echo "   Testing ffprobe..."
    if ./resources/bin/ffprobe -version > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ ffprobe is working${NC}"
    else
        echo -e "   ${RED}❌ ffprobe test failed${NC}"
        exit 1
    fi

else
    echo -e "${YELLOW}⚠️  Non-macOS system detected${NC}"
    echo "   Please manually download ffmpeg/ffprobe for your platform"
    echo "   and place in resources/bin/"
    exit 1
fi

# ==============================================================================
# Step 2: Verify prompts.json
# ==============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Step 2: Verifying prompts.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PROMPTS_FILE="src/python/prompts.json"

if [ ! -f "$PROMPTS_FILE" ]; then
    echo -e "${RED}❌ prompts.json not found at $PROMPTS_FILE${NC}"
    exit 1
fi

echo "   Validating JSON syntax..."
if python3 -m json.tool "$PROMPTS_FILE" > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ prompts.json is valid JSON${NC}"
else
    echo -e "   ${RED}❌ prompts.json has invalid JSON syntax${NC}"
    exit 1
fi

echo "   Checking for required prompts..."
PROMPT_COUNT=$(python3 -c "import json; print(len(json.load(open('$PROMPTS_FILE'))))")
echo "   Found $PROMPT_COUNT prompts"

if [ "$PROMPT_COUNT" -lt 1 ]; then
    echo -e "   ${RED}❌ prompts.json is empty${NC}"
    exit 1
fi

echo -e "   ${GREEN}✅ prompts.json verified${NC}"

# ==============================================================================
# Step 3: Build Python Virtual Environment
# ==============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐍 Step 3: Building Python Virtual Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VENV_DIR="src/python/venv"
REQUIREMENTS="src/python/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

echo "   Removing old venv (if exists)..."
rm -rf "$VENV_DIR"

echo "   Creating new virtual environment..."
python3 -m venv "$VENV_DIR"

echo "   Activating venv and installing dependencies..."
source "$VENV_DIR/bin/activate"

echo "   Upgrading pip..."
pip install --upgrade pip > /dev/null

echo "   Installing requirements..."
pip install -r "$REQUIREMENTS"

echo "   Verifying installation..."
python3 -c "import google.generativeai; import librosa; import whisper; print('Core deps OK')"

deactivate

echo -e "   ${GREEN}✅ Python venv created successfully${NC}"
echo "   Location: $VENV_DIR"

# ==============================================================================
# Step 4: Build Electron App (TypeScript)
# ==============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ Step 4: Building Electron App (TypeScript)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "   Compiling TypeScript..."
npm run build

echo -e "   ${GREEN}✅ TypeScript compilation complete${NC}"

# ==============================================================================
# Step 5: Package with electron-builder
# ==============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Step 5: Packaging with electron-builder"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

npm run package

# ==============================================================================
# Step 6: Verification
# ==============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Step 6: Verifying Package Contents"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DMG_PATH=$(find dist -name "*.dmg" -type f | head -1)

if [ -z "$DMG_PATH" ]; then
    echo -e "${RED}❌ No .dmg file found in dist/${NC}"
    exit 1
fi

echo -e "   ${GREEN}✅ Package created: $DMG_PATH${NC}"

DMG_SIZE=$(du -h "$DMG_PATH" | cut -f1)
echo "   Size: $DMG_SIZE"

# Mount and verify (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "   Mounting DMG to verify contents..."

    # Mount the DMG
    MOUNT_POINT=$(hdiutil attach "$DMG_PATH" | grep "/Volumes" | awk '{print $3}')

    if [ -z "$MOUNT_POINT" ]; then
        echo -e "   ${YELLOW}⚠️  Could not mount DMG for verification${NC}"
    else
        APP_PATH="$MOUNT_POINT/Gemini Video Understanding.app"

        echo "   Checking bundled resources..."

        # Check ffmpeg/ffprobe
        if [ -f "$APP_PATH/Contents/Resources/bin/ffmpeg" ]; then
            echo -e "   ${GREEN}✅ ffmpeg bundled${NC}"
        else
            echo -e "   ${RED}❌ ffmpeg missing${NC}"
        fi

        if [ -f "$APP_PATH/Contents/Resources/bin/ffprobe" ]; then
            echo -e "   ${GREEN}✅ ffprobe bundled${NC}"
        else
            echo -e "   ${RED}❌ ffprobe missing${NC}"
        fi

        # Check prompts.json
        if [ -f "$APP_PATH/Contents/Resources/python/scripts/prompts.json" ]; then
            echo -e "   ${GREEN}✅ prompts.json bundled${NC}"
        else
            echo -e "   ${RED}❌ prompts.json missing${NC}"
        fi

        # Check Python
        if [ -f "$APP_PATH/Contents/Resources/python/bin/python3" ]; then
            echo -e "   ${GREEN}✅ Python bundled${NC}"
        else
            echo -e "   ${RED}❌ Python missing${NC}"
        fi

        # Unmount
        hdiutil detach "$MOUNT_POINT" > /dev/null 2>&1
    fi
fi

# ==============================================================================
# Success Summary
# ==============================================================================

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ BUILD COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "📦 Package: $DMG_PATH"
echo "📏 Size: $DMG_SIZE"
echo ""
echo "Next steps:"
echo "  1. Test the app by opening it"
echo "  2. Try transcribing a short video"
echo "  3. Check Console.app for any errors"
echo "  4. If successful, distribute to users"
echo ""
echo "================================================================================"
