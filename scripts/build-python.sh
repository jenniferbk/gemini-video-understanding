#!/bin/bash
set -e

echo "🐍 Building portable Python environment..."

# Configuration
PYTHON_VERSION="3.13.8"
PYTHON_STANDALONE_VERSION="20251007"
ARCH="aarch64-apple-darwin"  # For Apple Silicon Macs
PYTHON_DIR="python-portable"

# Clean up old builds
echo "Cleaning up old Python builds..."
rm -rf $PYTHON_DIR

# Download python-build-standalone if not already downloaded
PYTHON_TARBALL="cpython-${PYTHON_VERSION}+${PYTHON_STANDALONE_VERSION}-${ARCH}-install_only.tar.gz"
DOWNLOAD_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_STANDALONE_VERSION}/${PYTHON_TARBALL}"

if [ ! -f "$PYTHON_TARBALL" ]; then
    echo "📥 Downloading portable Python from python-build-standalone..."
    echo "URL: $DOWNLOAD_URL"
    curl -L -o "$PYTHON_TARBALL" "$DOWNLOAD_URL"
else
    echo "✅ Using cached Python download"
fi

# Extract Python
echo "📦 Extracting Python..."
mkdir -p $PYTHON_DIR
tar -xzf $PYTHON_TARBALL -C $PYTHON_DIR

# The extracted structure is: python/bin/python3, python/lib/, etc.
# Move everything up one level
mv $PYTHON_DIR/python/* $PYTHON_DIR/
rmdir $PYTHON_DIR/python

# Set up pip and install dependencies
echo "📦 Installing Python dependencies..."
PYTHON_BIN="$PYTHON_DIR/bin/python3"

# Upgrade pip
$PYTHON_BIN -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    $PYTHON_BIN -m pip install -r requirements.txt
elif [ -f "src/python/requirements.txt" ]; then
    echo "Installing from src/python/requirements.txt..."
    $PYTHON_BIN -m pip install -r src/python/requirements.txt
else
    echo "⚠️  No requirements.txt found!"
fi

echo "✅ Portable Python environment built successfully!"
echo "Python location: $PYTHON_DIR/bin/python3"
echo ""
echo "Testing Python..."
$PYTHON_BIN --version
echo ""
echo "Testing imports..."
$PYTHON_BIN -c "import google.generativeai; import librosa; import torch; print('✅ All dependencies loaded successfully!')" || echo "⚠️  Some dependencies failed to load"
