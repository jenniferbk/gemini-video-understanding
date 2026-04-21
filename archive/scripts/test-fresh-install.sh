#!/bin/bash

# Simulates a fresh installation by temporarily moving user data
# Useful for testing first-launch experience without creating new user

set -e

APP_DATA="$HOME/Library/Application Support/gemini-video-understanding"
APP_PREFS="$HOME/Library/Preferences/edu.uga.gvu.plist"
BACKUP_DIR="$HOME/Desktop/GVU-Backup-$(date +%Y%m%d-%H%M%S)"

echo "🧪 Simulating Fresh Install Environment"
echo "========================================"
echo ""

# Check if app data exists
if [ ! -d "$APP_DATA" ] && [ ! -f "$APP_PREFS" ]; then
    echo "✅ Already a fresh environment (no user data found)"
    echo ""
    echo "You can now test the app as a first-time user."
    exit 0
fi

echo "Found existing user data:"
if [ -d "$APP_DATA" ]; then
    echo "  - Application Support: $APP_DATA"
fi
if [ -f "$APP_PREFS" ]; then
    echo "  - Preferences: $APP_PREFS"
fi
echo ""

read -p "Backup and move this data to simulate fresh install? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"
echo "📦 Creating backup at: $BACKUP_DIR"
echo ""

# Backup app data
if [ -d "$APP_DATA" ]; then
    echo "Backing up Application Support..."
    cp -R "$APP_DATA" "$BACKUP_DIR/"
    rm -rf "$APP_DATA"
    echo "  ✅ Moved Application Support data"
fi

# Backup preferences
if [ -f "$APP_PREFS" ]; then
    echo "Backing up Preferences..."
    cp "$APP_PREFS" "$BACKUP_DIR/"
    rm "$APP_PREFS"
    echo "  ✅ Moved Preferences"
fi

# Backup keychain entries (inform user to delete manually)
echo ""
echo "⚠️  Note: API key in macOS Keychain is NOT automatically removed"
echo "   To test API key entry, you'll need to manually delete it:"
echo "   1. Open Keychain Access app"
echo "   2. Search for 'GeminiVideoUnderstanding'"
echo "   3. Delete the entry"
echo ""

echo "✅ Fresh install environment ready!"
echo ""
echo "Your data is backed up at:"
echo "  $BACKUP_DIR"
echo ""
echo "To restore your data after testing:"
echo "  ./test-restore-data.sh \"$BACKUP_DIR\""
echo ""
echo "You can now launch the app and test the first-time experience."

# Create restore script
cat > test-restore-data.sh << 'EOF'
#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./test-restore-data.sh <backup-directory>"
    echo ""
    echo "Recent backups:"
    ls -dt ~/Desktop/GVU-Backup-* 2>/dev/null | head -5
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "Restoring data from: $BACKUP_DIR"
echo ""

# Restore app data
if [ -d "$BACKUP_DIR/gemini-video-understanding" ]; then
    echo "Restoring Application Support..."
    cp -R "$BACKUP_DIR/gemini-video-understanding" "$HOME/Library/Application Support/"
    echo "  ✅ Restored"
fi

# Restore preferences
if [ -f "$BACKUP_DIR/edu.uga.gvu.plist" ]; then
    echo "Restoring Preferences..."
    cp "$BACKUP_DIR/edu.uga.gvu.plist" "$HOME/Library/Preferences/"
    echo "  ✅ Restored"
fi

echo ""
echo "✅ Data restored successfully!"
echo ""
echo "You can delete the backup if no longer needed:"
echo "  rm -rf \"$BACKUP_DIR\""
EOF

chmod +x test-restore-data.sh
