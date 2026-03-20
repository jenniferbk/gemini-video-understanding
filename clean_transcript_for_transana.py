#!/usr/bin/env python3
"""
Clean transcript file for Transana import by removing emojis and metadata.
"""

import re
import sys

def clean_transcript(input_file, output_file):
    """
    Remove emojis, chunk headers, and metadata from transcript.
    Keep only: timestamp, speaker, and dialogue text.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    in_header = True

    for line in lines:
        # Skip header section (until we hit the first chunk marker or empty line after metadata)
        if in_header:
            if line.strip().startswith('📌 CHUNK') or (line.strip() == '' and len(cleaned_lines) == 0):
                continue
            # Check if we've passed the header by looking for timestamp pattern
            if re.match(r'^\s*\d{2}:\d{2}', line):
                in_header = False
            else:
                continue

        # Skip chunk headers
        if line.strip().startswith('📌 CHUNK') or line.strip().startswith('📊 VAD:') or line.strip().startswith('----'):
            continue

        # Skip separator lines
        if line.strip() == '=' * len(line.strip()) or line.strip() == '-' * len(line.strip()):
            continue

        # Skip empty lines
        if not line.strip():
            continue

        # Process dialogue lines with timestamps
        timestamp_match = re.match(r'^(\s*)(\d{2}:\d{2})\s+([^:]+):\s+(.+)', line)
        if timestamp_match:
            timestamp = timestamp_match.group(2)
            speaker = timestamp_match.group(3).strip()
            text = timestamp_match.group(4)

            # Remove emoji and metadata at the end (everything after and including emoji)
            # Common patterns: ✅ *VAD:0.50*, 🚨 *👤28💬100📊0.50*, ⚠️ *👤68💬100📊0.50*
            text = re.sub(r'\s*[✅🚨⚠️❌📌📊🎯📅🤖🔧🧠✂️📦📈📉👤💬🚀]\s*\*[^*]+\*\s*$', '', text)
            text = re.sub(r'\s*[✅🚨⚠️❌📌📊🎯📅🤖🔧🧠✂️📦📈📉👤💬🚀]\s*$', '', text)

            # Remove any remaining standalone emojis
            text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)

            # Clean up extra whitespace
            text = text.strip()

            # Format cleaned line
            cleaned_line = f"{timestamp} {speaker}: {text}\n"
            cleaned_lines.append(cleaned_line)

    # Write cleaned transcript
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"✓ Cleaned transcript saved to: {output_file}")
    print(f"✓ Original lines: {len(lines)}")
    print(f"✓ Cleaned lines: {len(cleaned_lines)}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 clean_transcript_for_transana.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    clean_transcript(input_file, output_file)
