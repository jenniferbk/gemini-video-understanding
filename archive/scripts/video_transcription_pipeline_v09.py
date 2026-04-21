#!/usr/bin/env python3
"""
Video Transcription Pipeline V09 for Educational Research
Based on V06 with key improvements for reliable classroom transcription.

V09 FEATURES (what changed from v06):
- Default model: gemini-3-flash-preview (Gemini 3 Flash)
- Default FPS: 2 (better speaker tracking)
- Traditional chunking by default (NO VAD - more reliable)
- Custom prompt support for speaker descriptions
- Simplified processing for better stability

USAGE:
  python video_transcription_pipeline_v09.py video.mp4 --prompt smallgroup_ben_day2
"""

import os
import sys
import time
import json
import argparse
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import statistics
import warnings

# Core dependencies
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    sys.exit(1)


@dataclass
class TranscriptionConfigV09:
    """Configuration for V09 pipeline - optimized for reliability"""
    # Core settings - V09.1: Shorter chunks to avoid MAX_TOKENS cutoff
    chunk_duration_minutes: float = 2.0  # 2 minutes instead of 3 to avoid token limits
    overlap_seconds: int = 10
    max_file_size_mb: int = 95
    model_name: str = "gemini-2.0-flash"  # V09: More stable than 3-flash-preview

    # V09: Traditional chunking by default (more reliable)
    use_traditional_chunking: bool = True

    # Processing settings
    fps: int = 2  # V09: FPS=2 for better speaker tracking
    prompt_key: str = "default"
    consensus_runs: int = 1
    max_retries: int = 3
    min_transcript_length: int = 50
    retry_delay: float = 5.0
    max_output_tokens: int = 16384  # V09.1: Increased from 8192

    # Output settings
    dual_output: bool = True  # Research (annotated) + Transana (clean)
    keep_chunks: bool = False  # Keep chunk files for debugging/reprocessing


class TranscriptValidator:
    """Validates transcription output quality"""

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def is_valid_transcription(self, transcript: str) -> Tuple[bool, str]:
        """Check if transcript meets quality standards"""
        if not transcript or len(transcript.strip()) < self.min_length:
            return False, "Transcript too short"

        # Check for error markers
        if transcript.strip().startswith('[') and 'ERROR' in transcript.upper():
            return False, "Contains error marker"

        # Check for excessive repetition (hallucination indicator)
        if self._detect_excessive_repetition(transcript):
            return False, "Excessive repetition detected (hallucination)"

        # Check for reasonable structure
        lines = [l for l in transcript.split('\n') if l.strip()]
        timestamp_lines = sum(1 for l in lines if re.match(r'^\d{1,2}:\d{2}', l.strip()))

        if len(lines) > 5 and timestamp_lines < len(lines) * 0.3:
            return False, "Insufficient timestamp structure"

        return True, "Valid"

    def _detect_excessive_repetition(self, transcript: str) -> bool:
        """Detect both line-level and word-level repetition hallucinations"""
        # Word-level repetition within the transcript
        words = re.findall(r'\b[a-zA-Z]{2,}\b', transcript.lower())
        if len(words) >= 20:
            word_counts = Counter(words)
            most_common_word, count = word_counts.most_common(1)[0]
            # If any single word appears more than 40% of all words, it's repetition
            if count > len(words) * 0.4:
                return True

        # Check for repeated short phrases (2-word sequences)
        if len(words) >= 30:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigram_counts:
                most_common_bigram, count = bigram_counts.most_common(1)[0]
                if count > len(bigrams) * 0.25:
                    return True

        # Line-level repetition check
        lines = [l.strip() for l in transcript.split('\n') if l.strip()]
        if len(lines) < 5:
            return False

        line_counts = Counter(lines)
        most_common_line, count = line_counts.most_common(1)[0]

        if count > len(lines) * 0.3:
            return True

        return False


class PromptManager:
    """Manages transcription prompts from prompts.json"""

    def __init__(self, prompts_file: str):
        self.prompts_file = prompts_file
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        """Load prompts from JSON file"""
        try:
            with open(self.prompts_file, 'r') as f:
                data = json.load(f)
                return data.get('prompts', data)
        except FileNotFoundError:
            print(f"Warning: Prompts file not found: {self.prompts_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in prompts file: {e}")
            return {}

    def get_prompt(self, key: str) -> str:
        """Get prompt by key"""
        if key in self.prompts:
            prompt_data = self.prompts[key]
            if isinstance(prompt_data, dict):
                return prompt_data.get('prompt', '')
            return prompt_data

        # Return default prompt if key not found
        return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Default transcription prompt"""
        return """
Please transcribe this classroom video with speaker diarization.

SPEAKERS TO IDENTIFY:
- Teacher: The main instructor (adult)
- Students: Identify by position/appearance (e.g., FemaleStudent1, MaleStudent2)

TRANSCRIPTION FORMAT:
MM:SS Speaker: Spoken content
[Action]: Non-verbal actions in brackets

REQUIREMENTS:
1. Use MM:SS timestamp format
2. Identify speakers consistently throughout
3. Capture all audible speech
4. Note significant non-verbal actions in brackets
5. Do NOT repeat words or phrases excessively

Begin transcription:
"""

    def list_prompts(self) -> List[str]:
        """List available prompt keys"""
        return list(self.prompts.keys())


class VideoCostCalculator:
    """Estimates API costs for video transcription"""

    # Gemini 3 Flash pricing (as of late 2024)
    PRICING = {
        'gemini-3-flash-preview': {'input': 0.00050, 'output': 0.00300},  # per 1K tokens
        'gemini-2.0-flash-exp': {'input': 0.00025, 'output': 0.00125},
        'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
        'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
    }

    def estimate_cost(self, duration_minutes: float, model_name: str,
                     chunk_minutes: float = 3.0, fps: int = 2) -> Dict:
        """Estimate transcription cost"""
        num_chunks = int(np.ceil(duration_minutes / chunk_minutes))

        # Estimate tokens per chunk
        frames_per_chunk = int(chunk_minutes * 60 * fps)
        tokens_per_frame = 258  # Gemini's typical frame token count
        audio_tokens_per_minute = 32

        video_tokens = frames_per_chunk * tokens_per_frame
        audio_tokens = int(chunk_minutes * 60 * audio_tokens_per_minute)
        prompt_tokens = 500

        input_tokens_per_chunk = video_tokens + audio_tokens + prompt_tokens
        output_tokens_per_chunk = 2000  # Estimated transcript length

        total_input_tokens = input_tokens_per_chunk * num_chunks
        total_output_tokens = output_tokens_per_chunk * num_chunks

        # Get pricing
        pricing = self.PRICING.get(model_name, self.PRICING['gemini-3-flash-preview'])

        input_cost = (total_input_tokens / 1000) * pricing['input']
        output_cost = (total_output_tokens / 1000) * pricing['output']

        return {
            'num_chunks': num_chunks,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens_estimated': total_input_tokens + total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }


class TraditionalChunker:
    """Simple time-based video chunking (no VAD complexity)"""

    def __init__(self, config: TranscriptionConfigV09):
        self.config = config

    def split_video(self, video_path: str, output_dir: str) -> List[Dict]:
        """Split video into fixed-duration chunks"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        duration_minutes = self._get_video_duration(str(video_path))
        if duration_minutes == 0:
            raise ValueError("Could not determine video duration")

        print(f"Processing {duration_minutes:.1f}-minute video...")

        chunk_duration_seconds = self.config.chunk_duration_minutes * 60
        chunks = []
        chunk_num = 1
        start_time = 0
        total_seconds = duration_minutes * 60

        while start_time < total_seconds:
            end_time = min(start_time + chunk_duration_seconds, total_seconds)

            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:02d}.mp4"

            success = self._extract_video_chunk(
                str(video_path), str(chunk_file), start_time, end_time - start_time
            )

            if success:
                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
                print(f"  Chunk {chunk_num}: {start_time/60:.1f}m - {end_time/60:.1f}m")

            start_time = end_time
            chunk_num += 1

        return chunks

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in minutes"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration_seconds = float(result.stdout.strip())
            return duration_seconds / 60
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            print(f"Error getting video duration: {e}")
            return 0

    def _extract_video_chunk(self, input_path: str, output_path: str,
                            start_time: float, duration: float) -> bool:
        """Extract video chunk using FFmpeg"""
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-i", input_path,
            "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", output_path, "-y"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk: {e}")
            return False


class SimpleTranscriber:
    """Simplified transcriber for V09 - focused on reliability"""

    def __init__(self, api_key: str, config: TranscriptionConfigV09):
        self.config = config
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.model_name)

        # Load prompts
        script_dir = Path(__file__).parent
        prompts_file = script_dir / "prompts.json"
        self.prompt_manager = PromptManager(str(prompts_file))

        # Initialize validation
        self.validator = TranscriptValidator(config.min_transcript_length)

        # Safety settings - permissive for educational content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def transcribe_chunk(self, chunk_info: Dict, chunk_number: int,
                        total_chunks: int, previous_transcript: str = None) -> str:
        """Transcribe a single video chunk"""

        print(f"\nProcessing chunk {chunk_number}/{total_chunks}")

        # Upload video chunk
        uploaded_file = self._upload_video_chunk(chunk_info['file_path'])

        try:
            # Build prompt with context
            prompt = self._build_prompt(chunk_number, previous_transcript)

            # Transcribe with retries
            transcript = self._transcribe_with_retry(uploaded_file, prompt)

            return transcript

        finally:
            self._cleanup_file(uploaded_file)

    def _build_prompt(self, chunk_number: int, previous_transcript: str = None) -> str:
        """Build prompt with continuity context"""
        base_prompt = self.prompt_manager.get_prompt(self.config.prompt_key)

        # Add continuity context for subsequent chunks
        if chunk_number > 1 and previous_transcript:
            # Get last few lines for context
            prev_lines = previous_transcript.strip().split('\n')
            context_lines = []

            for line in prev_lines[-8:]:
                line = line.strip()
                if line and ':' in line and any(c.isdigit() for c in line[:10]):
                    # Clean quality flags if present
                    clean_line = re.sub(r'[\u2705\u26a0\ufe0f\U0001F6A8]\s*', '', line)
                    clean_line = re.sub(r'\*[^*]+\*', '', clean_line).strip()
                    if clean_line:
                        context_lines.append(clean_line)

            if context_lines:
                context = '\n'.join(context_lines[-5:])
                continuity = f"""

CONTINUITY CONTEXT (from previous chunk):
{context}

Continue naturally from this context. Maintain speaker consistency.
Start timestamps from 00:00 for this chunk.
"""
                return base_prompt + continuity

        return base_prompt

    def _transcribe_with_retry(self, uploaded_file, prompt: str) -> str:
        """Transcribe with retry logic"""

        max_attempts = self.config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"   Retry {attempt-1}/{self.config.max_retries}")
                time.sleep(self.config.retry_delay)

            try:
                # Generate transcription
                response = self.model.generate_content(
                    [uploaded_file, prompt],
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0.1 if attempt == 1 else 0.3,
                        "max_output_tokens": self.config.max_output_tokens,
                    }
                )

                # Handle response warnings
                if hasattr(response, 'candidates') and response.candidates:
                    # Check for thinking model signature warnings
                    pass  # These are expected with gemini-3-flash-preview

                # Extract transcript
                transcript = self._extract_transcript(response)

                # Validate
                is_valid, reason = self.validator.is_valid_transcription(transcript)

                if is_valid:
                    print(f"   Valid transcript on attempt {attempt}")
                    return transcript
                else:
                    print(f"   Validation failed: {reason}")

                    # If repetition detected, retry with anti-hallucination prompt
                    if "repetition" in reason.lower() and attempt < max_attempts:
                        prompt = prompt + "\n\nCRITICAL: Do NOT repeat any word or phrase more than 3 times. Transcribe naturally."

                    if attempt == max_attempts:
                        # Return marker instead of garbage
                        return f"[CHUNK_NEEDS_REVIEW: {reason}]"

            except Exception as e:
                print(f"   Error: {e}")
                if attempt == max_attempts:
                    return f"[TRANSCRIPTION_ERROR: {str(e)}]"

        return "[TRANSCRIPTION_FAILED: Max attempts reached]"

    def _extract_transcript(self, response) -> str:
        """Extract transcript from Gemini response"""
        if not response.candidates:
            raise Exception("No response candidates")

        candidate = response.candidates[0]

        # Check finish reason
        if candidate.finish_reason != 1:
            finish_reasons = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
            reason = finish_reasons.get(candidate.finish_reason, f"UNKNOWN({candidate.finish_reason})")

            if candidate.content and candidate.content.parts:
                # Return partial content with warning
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                if text_parts:
                    return f"[PARTIAL: {reason}]\n\n" + "\n".join(text_parts)

            raise Exception(f"Generation stopped: {reason}")

        if not candidate.content or not candidate.content.parts:
            raise Exception("No content in response")

        # Extract text parts (skip thought signatures)
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, 'text'):
                text_parts.append(part.text)

        return "\n".join(text_parts)

    def _upload_video_chunk(self, chunk_path: str):
        """Upload video chunk to Gemini"""
        print(f"Uploading {Path(chunk_path).name}...", end="", flush=True)

        file = genai.upload_file(chunk_path)

        # Wait for processing
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(file.name)

        print()

        if file.state.name == "FAILED":
            raise Exception(f"File processing failed: {file.state}")

        print(f"Upload complete: {file.name}")
        return file

    def _cleanup_file(self, file):
        """Clean up uploaded file"""
        try:
            genai.delete_file(file.name)
            print(f"Cleaned up {file.name}")
        except Exception as e:
            print(f"Cleanup warning: {e}")


class VideoTranscriptionPipelineV09:
    """Main V09 pipeline - simplified and reliable"""

    def __init__(self, api_key: str, config: TranscriptionConfigV09, skip_confirmation: bool = False):
        self.config = config
        self.chunker = TraditionalChunker(config)
        self.transcriber = SimpleTranscriber(api_key, config)
        self.cost_calculator = VideoCostCalculator()
        self.skip_confirmation = skip_confirmation

    def process_video(self, video_path: str, output_dir: str = None) -> Dict:
        """Process video with V09 pipeline"""

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = video_path.parent / f"{video_path.stem}_v09_transcription_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Display processing info
        duration_minutes = self.chunker._get_video_duration(str(video_path))
        self._display_processing_info(video_path, duration_minutes)

        # Confirm processing
        if not self.skip_confirmation:
            response = input("\nProceed? (y/n): ").strip().lower()
            if response != 'y':
                print("Transcription cancelled.")
                return {}

        try:
            # Phase 1: Chunking
            print(f"\n{'='*60}")
            print("PHASE 1: CHUNKING")
            print(f"{'='*60}")

            chunks_dir = output_dir / "chunks"
            chunk_list = self.chunker.split_video(str(video_path), str(chunks_dir))

            if not chunk_list:
                raise Exception("No chunks were created")

            # Phase 2: Transcription
            print(f"\n{'='*60}")
            print("PHASE 2: TRANSCRIPTION")
            print(f"{'='*60}")

            all_transcripts = []
            previous_transcript = None

            for chunk_info in chunk_list:
                chunk_number = chunk_info['chunk_number']

                # Transcribe chunk
                transcript = self.transcriber.transcribe_chunk(
                    chunk_info, chunk_number, len(chunk_list), previous_transcript
                )

                all_transcripts.append({
                    'chunk_number': chunk_number,
                    'chunk_info': chunk_info,
                    'transcript': transcript
                })

                # Save individual chunk
                chunk_file = output_dir / f"chunk_{chunk_number:02d}_transcript.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)

                # Update context for next chunk
                if not transcript.startswith('['):
                    previous_transcript = transcript

            # Phase 3: Assembly
            print(f"\n{'='*60}")
            print("PHASE 3: ASSEMBLY")
            print(f"{'='*60}")

            # Combine transcripts with timestamp adjustment
            combined_transcript = self._combine_transcripts(all_transcripts, video_path)

            # Save outputs
            if self.config.dual_output:
                # Research version (with metadata)
                research_file = output_dir / f"{video_path.stem}_transcript.txt"
                with open(research_file, 'w', encoding='utf-8') as f:
                    f.write(combined_transcript)

                # Transana version (clean)
                clean_transcript = self._create_clean_transcript(combined_transcript)
                transana_file = output_dir / f"{video_path.stem}_transana.txt"
                with open(transana_file, 'w', encoding='utf-8') as f:
                    f.write(clean_transcript)

                print(f"\nResearch (annotated): {research_file}")
                print(f"Transana (clean):     {transana_file}")
            else:
                final_file = output_dir / f"{video_path.stem}_transcript.txt"
                with open(final_file, 'w', encoding='utf-8') as f:
                    f.write(combined_transcript)
                print(f"\nTranscript saved: {final_file}")

            # Cleanup chunks (unless keep_chunks is set)
            if self.config.keep_chunks:
                print(f"Keeping chunks: {chunks_dir}")
            else:
                self._cleanup_chunks(chunks_dir)

            # Display completion
            print(f"\n{'='*60}")
            print("V09 TRANSCRIPTION COMPLETE!")
            print(f"{'='*60}")

            return {
                'video_file': str(video_path),
                'output_dir': str(output_dir),
                'chunks_processed': len(all_transcripts),
                'successful_chunks': sum(1 for t in all_transcripts if not t['transcript'].startswith('[')),
            }

        except Exception as e:
            print(f"\nProcessing error: {e}")
            raise

    def _display_processing_info(self, video_path: Path, duration_minutes: float):
        """Display processing information"""

        print(f"\n{'='*80}")
        print("VIDEO TRANSCRIPTION PIPELINE V09")
        print(f"{'='*80}")
        print(f"Video: {video_path.name}")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Model: {self.config.model_name}")
        print(f"FPS: {self.config.fps}")

        print(f"\nV09 FEATURES:")
        print(f"   Traditional chunking: Yes (more reliable)")
        print(f"   Custom prompts: Yes")
        print(f"   Dual output: {self.config.dual_output}")

        # Cost estimate
        cost_estimate = self.cost_calculator.estimate_cost(
            duration_minutes, self.config.model_name,
            self.config.chunk_duration_minutes, self.config.fps
        )

        print(f"\nEstimated cost: ${cost_estimate['total_cost']:.3f}")

    def _combine_transcripts(self, all_transcripts: List[Dict], video_path: Path) -> str:
        """Combine chunk transcripts with timestamp adjustment"""

        combined = []

        # Header
        combined.append("=" * 80)
        combined.append("COMPLETE TRANSCRIPT - V09")
        combined.append("=" * 80)
        combined.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        combined.append(f"Model: {self.config.model_name}")
        combined.append(f"FPS: {self.config.fps}")
        combined.append("=" * 80)
        combined.append("")

        for transcript_data in all_transcripts:
            chunk_num = transcript_data['chunk_number']
            chunk_info = transcript_data['chunk_info']
            transcript = transcript_data['transcript'].strip()

            start_minutes = chunk_info['start_time'] / 60

            # Chunk header
            combined.append(f"CHUNK {chunk_num} (Starting at {start_minutes:.1f} minutes)")
            combined.append("-" * 60)

            # Check if this is a partial/failed chunk
            is_partial = transcript.startswith('[PARTIAL:')
            is_failed = transcript.startswith('[') and not is_partial

            if is_partial:
                # Extract the marker and content separately
                lines = transcript.split('\n', 2)  # Split into marker, blank, content
                marker = lines[0] if lines else ''
                content = '\n'.join(lines[2:]) if len(lines) > 2 else ''

                combined.append(f"[CHUNK {chunk_num} FAILED]")
                combined.append(marker)

                # Still adjust timestamps for the partial content
                if content.strip():
                    adjusted = self._adjust_timestamps(content, start_minutes)
                    combined.append(adjusted)
            elif is_failed or not transcript:
                combined.append(f"[CHUNK {chunk_num} FAILED]")
                combined.append(transcript)
            else:
                # Normal transcript - adjust timestamps
                adjusted = self._adjust_timestamps(transcript, start_minutes)
                combined.append(adjusted)

            combined.append("")

        return "\n".join(combined)

    def _adjust_timestamps(self, transcript: str, start_minutes: float) -> str:
        """Adjust timestamps in transcript by adding chunk start time"""

        lines = transcript.split('\n')
        adjusted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip metadata lines
            if line.startswith('===') or line.startswith('---'):
                continue
            if line.startswith('Generated:') or line.startswith('Model:'):
                continue

            # Match various timestamp formats
            # MM:SS Speaker: content
            # MM:SS [Action]: content
            # MM:SS content
            timestamp_match = re.match(r'^(\d{1,2}:\d{2}(?::\d{2})?):?\s*(.*)', line)

            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                rest_of_line = timestamp_match.group(2)

                try:
                    # Parse timestamp
                    parts = timestamp_str.split(':')
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                    elif len(parts) == 3:
                        # HH:MM:SS format
                        minutes = int(parts[0]) * 60 + int(parts[1])
                        seconds = int(parts[2])
                    else:
                        adjusted_lines.append(line)
                        continue

                    # Add chunk start time
                    total_seconds = (minutes * 60 + seconds) + (start_minutes * 60)
                    new_minutes = int(total_seconds // 60)
                    new_seconds = int(total_seconds % 60)

                    new_timestamp = f"{new_minutes:02d}:{new_seconds:02d}"
                    adjusted_lines.append(f"{new_timestamp} {rest_of_line}")

                except (ValueError, IndexError):
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)

        return '\n'.join(adjusted_lines)

    def _create_clean_transcript(self, transcript: str) -> str:
        """Create clean transcript for Transana (no metadata, no flags)"""

        lines = transcript.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()

            # Skip header/metadata
            if line.startswith('===') or line.startswith('---'):
                continue
            if line.startswith('Generated:') or line.startswith('Model:'):
                continue
            if line.startswith('FPS:') or line.startswith('COMPLETE TRANSCRIPT'):
                continue
            if line.startswith('CHUNK ') and ('Starting at' in line or 'FAILED' in line):
                continue
            if line.startswith('[PARTIAL:') or line.startswith('[CHUNK'):
                continue
            if not line:
                continue

            # Remove quality flags
            clean_line = re.sub(r'[\u2705\u26a0\ufe0f\U0001F6A8]\s*', '', line)
            clean_line = re.sub(r'\*[^*]+\*', '', clean_line).strip()

            if clean_line:
                clean_lines.append(clean_line)

        return '\n'.join(clean_lines)

    def _cleanup_chunks(self, chunks_dir: Path):
        """Clean up temporary chunk files"""
        try:
            import shutil
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
                print(f"Cleaned up: {chunks_dir}")
        except Exception as e:
            print(f"Cleanup warning: {e}")


def main():
    """Main entry point for V09 pipeline"""

    parser = argparse.ArgumentParser(
        description="V09 Video Transcription Pipeline - Reliable Classroom Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V09 FEATURES:
  - Default model: gemini-3-flash-preview (Gemini 3 Flash)
  - Default FPS: 2 (better speaker tracking)
  - Traditional chunking (more reliable than VAD)
  - Custom prompt support for speaker descriptions
  - Dual output: Research (annotated) + Transana (clean)

EXAMPLES:
  Basic transcription:
    python video_transcription_pipeline_v09.py video.mp4

  With custom prompt for speaker descriptions:
    python video_transcription_pipeline_v09.py video.mp4 --prompt smallgroup_ben_day2

  Skip confirmation:
    python video_transcription_pipeline_v09.py video.mp4 --no-confirm
        """
    )

    # Core arguments
    parser.add_argument("video_path", nargs='?', help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output directory")

    # Model and prompt
    parser.add_argument("-m", "--model", default="gemini-2.0-flash",
                       help="Gemini model (default: gemini-2.0-flash)")
    parser.add_argument("-p", "--prompt", default="default",
                       help="Prompt key from prompts.json")

    # Processing settings
    parser.add_argument("--fps", type=int, default=2,
                       help="Video analysis FPS (default: 2)")
    parser.add_argument("-c", "--chunk-minutes", type=float, default=2.0,
                       help="Chunk duration in minutes (default: 2.0)")
    parser.add_argument("--consensus-runs", type=int, default=1,
                       help="Number of consensus runs (default: 1)")

    # Other options
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Max retry attempts (default: 3)")
    parser.add_argument("--api-key", help="Gemini API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--no-confirm", action="store_true",
                       help="Skip confirmation prompt")
    parser.add_argument("--estimate-only", action="store_true",
                       help="Show cost estimate only")
    parser.add_argument("--single-output", action="store_true",
                       help="Single output file instead of dual (research + transana)")
    parser.add_argument("--keep-chunks", action="store_true",
                       help="Keep chunk files after processing (for debugging/reprocessing)")

    args = parser.parse_args()

    # Require video path
    if not args.video_path:
        parser.error("video_path is required")

    # Get API key
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please provide API key via --api-key or GOOGLE_API_KEY environment variable")
        sys.exit(1)

    # Create configuration
    config = TranscriptionConfigV09(
        chunk_duration_minutes=args.chunk_minutes,
        model_name=args.model,
        fps=args.fps,
        prompt_key=args.prompt,
        consensus_runs=args.consensus_runs,
        max_retries=args.max_retries,
        dual_output=not args.single_output,
        keep_chunks=args.keep_chunks
    )

    try:
        if args.estimate_only:
            # Cost estimation only
            chunker = TraditionalChunker(config)
            duration = chunker._get_video_duration(args.video_path)

            if duration == 0:
                print("Could not determine video duration")
                sys.exit(1)

            cost_calc = VideoCostCalculator()
            estimate = cost_calc.estimate_cost(duration, args.model, args.chunk_minutes, args.fps)

            print(f"\n{'='*60}")
            print("V09 COST ESTIMATE")
            print(f"{'='*60}")
            print(f"Video: {args.video_path}")
            print(f"Duration: {duration:.1f} minutes")
            print(f"Model: {args.model}")
            print(f"FPS: {args.fps}")
            print(f"Chunks: {estimate['num_chunks']}")
            print(f"Estimated cost: ${estimate['total_cost']:.3f}")
        else:
            # Full processing
            processor = VideoTranscriptionPipelineV09(
                api_key, config, skip_confirmation=args.no_confirm
            )
            result = processor.process_video(args.video_path, args.output)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nV09 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
