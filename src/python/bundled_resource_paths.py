"""
Bundled Resource Path Resolution for Electron App

This module provides functions to locate ffmpeg/ffprobe binaries and prompts.json
when the Python script is bundled with an Electron application.

Usage:
    from bundled_resource_paths import get_bundled_binary, get_bundled_prompts_path

    # Get ffmpeg/ffprobe paths
    ffmpeg = get_bundled_binary('ffmpeg')
    ffprobe = get_bundled_binary('ffprobe')

    # Get prompts.json path
    prompts_file = get_bundled_prompts_path()
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional


def get_bundled_binary(binary_name: str) -> str:
    """
    Find ffmpeg/ffprobe binary in bundled app or system PATH.

    Search order:
    1. Bundled with Electron app (production)
    2. resources/bin/ (local build testing)
    3. System PATH (development/fallback)

    Args:
        binary_name: Name of binary ('ffmpeg' or 'ffprobe')

    Returns:
        Absolute path to binary

    Raises:
        FileNotFoundError: If binary not found anywhere
    """
    # Check if running from bundled app
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    else:
        # Running from script
        bundle_dir = Path(__file__).parent

    # Possible locations for bundled binaries
    possible_paths = [
        # Electron app bundle structure
        Path('/Applications/Gemini Video Understanding.app/Contents/Resources/bin') / binary_name,

        # Relative to script (when bundled)
        bundle_dir.parent.parent.parent / 'bin' / binary_name,  # ../../../bin/
        bundle_dir.parent.parent / 'bin' / binary_name,         # ../../bin/
        bundle_dir.parent / 'bin' / binary_name,                # ../bin/
        bundle_dir / 'bin' / binary_name,                       # ./bin/

        # Local development/testing
        Path(__file__).parent.parent.parent / 'resources' / 'bin' / binary_name,
    ]

    # Try each possible path
    for path in possible_paths:
        if path.exists() and path.is_file():
            # Verify it's executable
            if os.access(path, os.X_OK):
                print(f"✅ Found {binary_name} at: {path}")
                return str(path)
            else:
                print(f"⚠️  Found {binary_name} at {path} but it's not executable")

    # Fall back to system PATH
    system_binary = shutil.which(binary_name)
    if system_binary:
        print(f"✅ Found {binary_name} in system PATH: {system_binary}")
        return system_binary

    # Not found anywhere
    raise FileNotFoundError(
        f"❌ {binary_name} not found in bundle or system PATH.\n\n"
        f"Please install ffmpeg:\n"
        f"  macOS:   brew install ffmpeg\n"
        f"  Windows: choco install ffmpeg\n"
        f"  Linux:   sudo apt install ffmpeg\n\n"
        f"Or download from: https://ffmpeg.org/download.html\n\n"
        f"Searched locations:\n" +
        "\n".join(f"  - {p}" for p in possible_paths)
    )


def get_bundled_prompts_path() -> Path:
    """
    Find prompts.json in bundled app or development location.

    Search order:
    1. Bundled with Electron app
    2. Relative to script (development)

    Returns:
        Path to prompts.json

    Raises:
        FileNotFoundError: If prompts.json not found
    """
    if getattr(sys, 'frozen', False):
        # Running in bundled app
        bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
    else:
        # Running from script
        bundle_dir = Path(__file__).parent

    possible_paths = [
        # Electron app bundle structure
        Path('/Applications/Gemini Video Understanding.app/Contents/Resources/python/scripts/prompts.json'),

        # Relative to script (when bundled)
        bundle_dir / 'prompts.json',                            # ./prompts.json
        bundle_dir / 'scripts' / 'prompts.json',                # ./scripts/prompts.json
        bundle_dir.parent / 'scripts' / 'prompts.json',         # ../scripts/prompts.json

        # Development
        Path(__file__).parent / 'prompts.json',
    ]

    # Try each possible path
    for path in possible_paths:
        if path.exists() and path.is_file():
            print(f"✅ Found prompts.json at: {path}")

            # Verify it's valid JSON
            try:
                import json
                with open(path) as f:
                    data = json.load(f)

                if not data:
                    print(f"⚠️  prompts.json at {path} is empty")
                    continue

                return path
            except json.JSONDecodeError as e:
                print(f"⚠️  prompts.json at {path} has invalid JSON: {e}")
                continue

    # Not found anywhere
    raise FileNotFoundError(
        f"❌ prompts.json not found in bundle.\n\n"
        f"Please ensure prompts.json is bundled with the app.\n\n"
        f"Searched locations:\n" +
        "\n".join(f"  - {p}" for p in possible_paths)
    )


def setup_ffmpeg_environment() -> tuple[str, str]:
    """
    Set up environment to use bundled ffmpeg/ffprobe.

    Returns:
        Tuple of (ffmpeg_path, ffprobe_path)

    Raises:
        FileNotFoundError: If binaries not found
    """
    try:
        ffmpeg_path = get_bundled_binary('ffmpeg')
        ffprobe_path = get_bundled_binary('ffprobe')

        # Add directory to PATH so subprocess calls can find them
        bin_dir = str(Path(ffmpeg_path).parent)
        if bin_dir not in os.environ['PATH']:
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
            print(f"✅ Added {bin_dir} to PATH")

        return ffmpeg_path, ffprobe_path

    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print(f"⚠️  Falling back to assuming ffmpeg/ffprobe are in system PATH")
        return 'ffmpeg', 'ffprobe'


# Test/demo code
if __name__ == '__main__':
    print("🔍 Testing bundled resource path resolution...\n")

    try:
        ffmpeg, ffprobe = setup_ffmpeg_environment()
        print(f"\nFFMPEG:  {ffmpeg}")
        print(f"FFPROBE: {ffprobe}")
    except FileNotFoundError as e:
        print(f"\n{e}")

    print()

    try:
        prompts = get_bundled_prompts_path()
        print(f"PROMPTS: {prompts}")

        import json
        with open(prompts) as f:
            data = json.load(f)
        print(f"Found {len(data)} prompts")
    except FileNotFoundError as e:
        print(f"\n{e}")
