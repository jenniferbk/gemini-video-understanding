#!/usr/bin/env python3
"""
Reprocess specific bad chunks with a different model and stronger prompts.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Add the COMS directory to path
sys.path.insert(0, str(Path(__file__).parent))

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def extract_chunk(video_path: str, output_path: str, start_minutes: float, duration_minutes: float = 2.0):
    """Extract a video chunk using ffmpeg"""
    start_seconds = start_minutes * 60
    duration_seconds = duration_minutes * 60

    cmd = [
        "ffmpeg", "-ss", str(start_seconds), "-i", video_path,
        "-t", str(duration_seconds), "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", output_path, "-y"
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path

def get_anti_hallucination_prompt(prompt_key: str = "smallgroup_ben_day2") -> str:
    """Get prompt with strong anti-hallucination instructions"""
    import json

    prompts_file = Path(__file__).parent / "prompts.json"
    with open(prompts_file, 'r') as f:
        data = json.load(f)

    base_prompt = data.get('prompts', {}).get(prompt_key, {}).get('prompt', '')

    # Add strong anti-hallucination instructions
    anti_hallucination = """

CRITICAL QUALITY RULES - READ CAREFULLY:

1. NEVER repeat the same line more than once unless the speaker TRULY repeats themselves verbatim.
   - BAD: "I think it's 44." repeated 3 times when said once
   - GOOD: Transcribe what was actually said once

2. Do NOT fill gaps with "[inaudible]" every 2 seconds. Instead:
   - Try harder to understand unclear speech
   - If truly inaudible, use ONE "[inaudible]" to cover a gap, not multiple
   - Group unclear sections: "00:15 [inaudible conversation for ~10 seconds]"

3. Do NOT hallucinate content. If you're unsure, mark it [uncertain] rather than making up dialogue.

4. Timestamps should reflect ACTUAL speech events, not arbitrary 2-second intervals.

5. If a section has minimal audible speech, note it: "[students working quietly]" rather than fabricating dialogue.

BEGIN TRANSCRIPTION:
"""

    return base_prompt + anti_hallucination

def transcribe_chunk(video_path: str, model_name: str = "gemini-2.0-flash", max_retries: int = 3) -> str:
    """Transcribe a single chunk with the specified model, with retry logic"""

    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Upload file
    print(f"Uploading {Path(video_path).name}...", end="", flush=True)
    file = genai.upload_file(video_path)

    while file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        file = genai.get_file(file.name)

    print(f" done ({file.name})")

    if file.state.name == "FAILED":
        raise Exception(f"Upload failed: {file.state}")

    # Get prompt
    prompt = get_anti_hallucination_prompt()

    # Generate with retry logic
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Transcribing with {model_name} (attempt {attempt})...")
            response = model.generate_content(
                [file, prompt],
                safety_settings=safety_settings,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 8192,
                }
            )
            break  # Success
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait_time = 30 * attempt  # 30, 60, 90 seconds
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                if attempt == max_retries:
                    raise
            else:
                raise

    # Cleanup
    try:
        genai.delete_file(file.name)
        print(f"Cleaned up {file.name}")
    except:
        pass

    # Extract text
    if response.candidates and response.candidates[0].content:
        parts = response.candidates[0].content.parts
        text_parts = [p.text for p in parts if hasattr(p, 'text')]
        return "\n".join(text_parts)

    return "[TRANSCRIPTION_FAILED]"

def main():
    coms_dir = Path("/Users/jenniferkleiman/Documents/COMS")
    output_dir = coms_dir / "250326_3Math_Ben_Day2_SG2_v09_transcription_20251222_133821"

    # Existing chunk files (user saved them)
    bad_chunks = [13, 14, 18, 20]

    # Model to use for reprocessing - gemini-2.0-flash is more stable than 3-flash-preview
    model = "gemini-2.0-flash"

    for i, chunk_num in enumerate(bad_chunks):
        # Add delay between chunks to avoid rate limiting
        if i > 0:
            print("\nWaiting 15 seconds before next chunk...")
            time.sleep(15)
        # Use existing saved chunk file
        chunk_file = coms_dir / f"250326_3Math_Ben_Day2_SG2_chunk_{chunk_num}.mp4"

        if not chunk_file.exists():
            print(f"ERROR: Chunk file not found: {chunk_file}")
            continue

        print(f"\n{'='*60}")
        print(f"REPROCESSING CHUNK {chunk_num}")
        print(f"Using: {chunk_file}")
        print(f"{'='*60}")

        # Transcribe
        transcript = transcribe_chunk(str(chunk_file), model)

        # Save new transcript
        new_transcript_file = output_dir / f"chunk_{chunk_num:02d}_transcript_v2.txt"
        with open(new_transcript_file, 'w') as f:
            f.write(transcript)

        print(f"\nSaved new transcript: {new_transcript_file}")
        print(f"\nFirst 20 lines:")
        print("-" * 40)
        for line in transcript.split('\n')[:20]:
            print(line)
        print("-" * 40)

    print(f"\n{'='*60}")
    print("REPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"New transcripts saved as chunk_XX_transcript_v2.txt")
    print(f"Compare with original chunk_XX_transcript.txt files")

if __name__ == "__main__":
    main()
