#!/usr/bin/env python3
"""
Video Transcription Pipeline V10 for Educational Research
CLI-first tool for batch transcription of classroom videos with speaker diarization.

V10 KEY IMPROVEMENTS:
- Model: gemini-3-flash-preview (Gemini 3 Flash, 86.9% Video-MMMU)
- SDK: google-genai (unlocks caching, media_resolution, thinking)
- 15-second video overlap between chunks for speaker continuity
- Two-pass speaker auto-detection with descriptive pseudonyms + human-in-the-loop
- Speaker registry passed as structured context to every chunk
- Stream copy FFmpeg chunking (fast) with re-encode fallback
- HIGH media resolution (280 tok/frame on Gemini 3) for rich visual detail
- 2 FPS video sampling for gesture/action capture in small group work
- 60s chunks for focused per-chunk attention
- Educational research prompt framing (activity understanding, not exhaustive description)
- 5 parallel video workers for batch processing
- max_output_tokens=16384

SUBCOMMANDS:
  identify /path/to/videos/    Interactive speaker ID, saves manifests
  process video.mp4             Single video (interactive speaker ID + transcription)
  batch /path/to/videos/        Unattended batch using saved manifests
  estimate video.mp4            Cost estimate only

USAGE:
  python video_transcription_pipeline_v10.py process video.mp4
  python video_transcription_pipeline_v10.py identify ./videos/ --prompt smallgroup_ben
  python video_transcription_pipeline_v10.py batch ./videos/ --workers 5
  python video_transcription_pipeline_v10.py estimate video.mp4
"""

import os
import sys
import time
import json
import math
import argparse
import re
import shutil
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
import statistics
import warnings

# Core dependency
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai>=1.0.0")
    sys.exit(1)

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SpeakerInfo:
    """Speaker information for registry"""
    label: str
    description: str
    speaker_type: str = "student"  # teacher, student, researcher

    def to_dict(self) -> Dict:
        return {"label": self.label, "description": self.description, "type": self.speaker_type}

    @classmethod
    def from_dict(cls, data: Dict) -> 'SpeakerInfo':
        return cls(
            label=data["label"],
            description=data["description"],
            speaker_type=data.get("type", "student")
        )


@dataclass
class TranscriptionConfigV10:
    """Configuration for V10 pipeline"""
    # Chunking
    chunk_duration_seconds: int = 60  # 1 minute (better per-chunk focus for detail)
    overlap_seconds: int = 15

    # Model
    model_name: str = "gemini-3-flash-preview"
    media_resolution: str = "HIGH"  # LOW, MEDIUM, HIGH
    thinking_budget: int = 4096  # Medium thinking
    max_output_tokens: int = 16384
    temperature: float = 0.2
    video_fps: int = 2  # Frames per second for video sampling (2 recommended for small groups)

    # Processing
    parallel_videos: int = 10
    parallel_uploads: int = 3
    max_retries: int = 3
    retry_delay: float = 5.0
    min_transcript_length: int = 50

    # Speaker ID
    speaker_id_chunks: int = 2  # Chunks to use for speaker identification
    continuity_context_lines: int = 8

    # Output
    dual_output: bool = True  # Research (annotated) + Transana (clean)
    keep_chunks: bool = False

    # Prompt
    prompt_key: str = "default"

    # Progress reporting (for future Electron integration)
    json_progress: bool = False

    # Context caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 7200  # 2 hours


# =============================================================================
# UTILITY CLASSES (ported from v08/v09)
# =============================================================================

class TranscriptValidator:
    """Validates transcription output quality (ported from v09)"""

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def is_valid(self, transcript: str) -> Tuple[bool, str]:
        """Check if transcript meets quality standards"""
        if not transcript or len(transcript.strip()) < self.min_length:
            return False, "Transcript too short"

        if transcript.strip().startswith('[') and 'ERROR' in transcript.upper():
            return False, "Contains error marker"

        if self._detect_excessive_repetition(transcript):
            return False, "Excessive repetition detected (hallucination)"

        lines = [l for l in transcript.split('\n') if l.strip()]
        timestamp_lines = sum(1 for l in lines if re.match(r'^\d{1,2}:\d{2}', l.strip()))
        if len(lines) > 5 and timestamp_lines < len(lines) * 0.3:
            return False, "Insufficient timestamp structure"

        return True, "Valid"

    def _detect_excessive_repetition(self, transcript: str) -> bool:
        """Detect both line-level and word-level repetition hallucinations"""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', transcript.lower())
        if len(words) >= 20:
            word_counts = Counter(words)
            most_common_word, count = word_counts.most_common(1)[0]
            if count > len(words) * 0.4:
                return True

        if len(words) >= 30:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigram_counts:
                most_common_bigram, count = bigram_counts.most_common(1)[0]
                if count > len(bigrams) * 0.25:
                    return True

        lines = [l.strip() for l in transcript.split('\n') if l.strip()]
        if len(lines) >= 5:
            line_counts = Counter(lines)
            most_common_line, count = line_counts.most_common(1)[0]
            if count > len(lines) * 0.3:
                return True

        return False


class TimestampNormalizer:
    """Normalize and adjust timestamps in transcripts (ported from v08)"""

    @staticmethod
    def normalize(timestamp: str) -> str:
        """Convert various timestamp formats to MM:SS"""
        match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})$', timestamp)
        if match:
            hours, mins, secs = int(match.group(1)), int(match.group(2)), int(match.group(3))
            total_mins = hours * 60 + mins
            return f"{total_mins:02d}:{secs:02d}"

        match = re.match(r'^(\d{1,2}):(\d{2})$', timestamp)
        if match:
            mins, secs = int(match.group(1)), int(match.group(2))
            return f"{mins:02d}:{secs:02d}"

        return timestamp

    @staticmethod
    def to_seconds(timestamp: str) -> float:
        """Convert MM:SS or HH:MM:SS to seconds"""
        normalized = TimestampNormalizer.normalize(timestamp)
        parts = normalized.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return 0.0

    @staticmethod
    def from_seconds(seconds: float) -> str:
        """Convert seconds to MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def adjust_transcript(transcript: str, offset_seconds: float) -> str:
        """Adjust all timestamps in transcript by offset"""
        lines = transcript.split('\n')
        adjusted = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            timestamp_match = re.match(r'^(\d{1,2}:\d{2}(?::\d{2})?):?\s*(.*)$', line)
            if timestamp_match:
                ts_str = timestamp_match.group(1)
                rest = timestamp_match.group(2)

                normalized = TimestampNormalizer.normalize(ts_str)
                original_secs = TimestampNormalizer.to_seconds(normalized)
                new_secs = original_secs + offset_seconds
                new_ts = TimestampNormalizer.from_seconds(max(0, new_secs))

                if rest:
                    adjusted.append(f"{new_ts} {rest}")
                else:
                    adjusted.append(new_ts)
            else:
                adjusted.append(line)

        return '\n'.join(adjusted)


class SubtitleExporter:
    """Export transcripts to SRT format (ported from v08)"""

    @staticmethod
    def to_srt(transcript: str, output_path: str):
        """Export transcript to SRT format"""
        lines = transcript.split('\n')
        srt_entries = []
        entry_num = 1

        for i, line in enumerate(lines):
            line = line.strip()
            match = re.match(r'^(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)$', line)
            if not match:
                continue

            timestamp, speaker, content = match.groups()
            start_seconds = TimestampNormalizer.to_seconds(timestamp)

            end_seconds = start_seconds + 3
            for next_line in lines[i+1:]:
                next_match = re.match(r'^(\d{1,2}:\d{2})', next_line.strip())
                if next_match:
                    end_seconds = TimestampNormalizer.to_seconds(next_match.group(1))
                    break

            start_srt = SubtitleExporter._seconds_to_srt(start_seconds)
            end_srt = SubtitleExporter._seconds_to_srt(end_seconds)

            srt_entries.append(f"{entry_num}\n{start_srt} --> {end_srt}\n{speaker}: {content}\n")
            entry_num += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_entries))

        print(f"SRT exported: {output_path}")

    @staticmethod
    def _seconds_to_srt(seconds: float) -> str:
        """Convert seconds to SRT time format HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


class PromptManager:
    """Manages transcription prompts from prompts.json (ported from v09)"""

    def __init__(self, prompts_file: str):
        self.prompts_file = prompts_file
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
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
        """Get prompt by key, falls back to default"""
        if key in self.prompts:
            prompt_data = self.prompts[key]
            if isinstance(prompt_data, dict):
                return prompt_data.get('prompt', '')
            return prompt_data
        return self._default_prompt()

    def _default_prompt(self) -> str:
        return """Please transcribe this classroom video with speaker diarization.

SPEAKERS TO IDENTIFY:
- Teacher: The main instructor (adult)
- Students: Identify by position/appearance (e.g., Boy-RedShirt, Girl-Glasses)

TRANSCRIPTION FORMAT:
MM:SS Speaker: Spoken content
[Action]: Non-verbal actions in brackets

REQUIREMENTS:
1. Use MM:SS timestamp format
2. Identify speakers consistently throughout
3. Capture all audible speech
4. Note significant non-verbal actions in brackets
5. Do NOT repeat words or phrases excessively
6. Use [inaudible] for unclear speech

Begin transcription:"""

    def list_prompts(self) -> List[str]:
        return list(self.prompts.keys())


class VideoCostCalculator:
    """Estimates API costs for video transcription (updated for v10)"""

    # Gemini pricing per 1K tokens
    PRICING = {
        'gemini-3-flash-preview': {'input': 0.00015, 'output': 0.0006, 'cached_input': 0.0000375},
        'gemini-2.5-flash': {'input': 0.00015, 'output': 0.0006, 'cached_input': 0.0000375},
        'gemini-2.0-flash': {'input': 0.00010, 'output': 0.00040, 'cached_input': 0.000025},
    }

    # Tokens per frame by media resolution
    TOKENS_PER_FRAME = {
        'HIGH': 258,
        'MEDIUM': 70,
        'LOW': 35,
    }

    AUDIO_TOKENS_PER_SECOND = 32

    def estimate(self, duration_minutes: float, config: 'TranscriptionConfigV10') -> Dict:
        """Estimate transcription cost for a single video"""
        total_seconds = duration_minutes * 60
        chunk_duration = config.chunk_duration_seconds
        overlap = config.overlap_seconds
        stride = chunk_duration - overlap

        # First chunk has no overlap
        num_chunks = 1 + max(0, math.ceil((total_seconds - chunk_duration) / stride))
        # Speaker ID uses first N chunks
        speaker_id_calls = min(config.speaker_id_chunks, num_chunks)

        tokens_per_frame = self.TOKENS_PER_FRAME.get(config.media_resolution, 70)

        # Per-chunk token estimate (FPS controls frame sampling rate)
        fps = config.video_fps
        frames_per_chunk = chunk_duration * fps
        video_tokens = frames_per_chunk * tokens_per_frame
        audio_tokens = chunk_duration * self.AUDIO_TOKENS_PER_SECOND
        prompt_tokens = 800  # Base prompt + speaker registry + context
        input_per_chunk = video_tokens + audio_tokens + prompt_tokens
        output_per_chunk = 3000  # Estimated transcript output

        # Speaker ID: extra calls with shorter output
        speaker_id_input = input_per_chunk * speaker_id_calls
        speaker_id_output = 500 * speaker_id_calls

        # Transcription calls
        transcription_input = input_per_chunk * num_chunks
        transcription_output = output_per_chunk * num_chunks

        total_input = speaker_id_input + transcription_input
        total_output = speaker_id_output + transcription_output

        pricing = self.PRICING.get(config.model_name, self.PRICING['gemini-3-flash-preview'])
        input_cost = (total_input / 1000) * pricing['input']
        output_cost = (total_output / 1000) * pricing['output']

        return {
            'num_chunks': num_chunks,
            'speaker_id_calls': speaker_id_calls,
            'tokens_per_frame': tokens_per_frame,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'duration_minutes': duration_minutes,
        }

    def estimate_batch(self, video_paths: List[Path], config: 'TranscriptionConfigV10') -> Dict:
        """Estimate cost for a batch of videos"""
        total_cost = 0
        total_chunks = 0
        video_estimates = []

        for path in video_paths:
            duration = get_video_duration(str(path))
            if duration > 0:
                est = self.estimate(duration, config)
                total_cost += est['total_cost']
                total_chunks += est['num_chunks']
                video_estimates.append({'video': path.name, **est})

        return {
            'num_videos': len(video_estimates),
            'total_chunks': total_chunks,
            'total_cost': total_cost,
            'videos': video_estimates,
        }


# =============================================================================
# HELPERS
# =============================================================================

def get_video_duration(video_path: str) -> float:
    """Get video duration in minutes using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip()) / 60
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        print(f"Error getting video duration: {e}")
        return 0


def find_videos(directory: Path) -> List[Path]:
    """Find all video files in a directory"""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(directory.glob(f"*{ext}"))
    return sorted(videos)


def report_progress(config: 'TranscriptionConfigV10', msg_type: str, **kwargs):
    """Report progress in JSON format for Electron integration"""
    if not config.json_progress:
        return
    msg = json.dumps({"type": msg_type, **kwargs})
    prefix = {"progress": "GVU_PROGRESS", "complete": "GVU_COMPLETE", "error": "GVU_ERROR"}
    print(f"{prefix.get(msg_type, 'GVU_INFO')}:{msg}", flush=True)


# =============================================================================
# CORE CLASSES
# =============================================================================

class GeminiClient:
    """Wrapper for google-genai SDK with media_resolution, thinking, and caching"""

    def __init__(self, api_key: str, config: TranscriptionConfigV10):
        self.config = config
        self.client = genai.Client(api_key=api_key)
        self.validator = TranscriptValidator(config.min_transcript_length)

    def upload_file(self, file_path: str) -> Any:
        """Upload file to Gemini and wait for processing"""
        name = Path(file_path).name
        print(f"  Uploading {name}...", end="", flush=True)
        uploaded = self.client.files.upload(file=file_path)

        while uploaded.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)

        print(" done")

        if uploaded.state.name == "FAILED":
            raise Exception(f"File processing failed for {name}")

        return uploaded

    def upload_files_parallel(self, file_paths: List[str]) -> Dict[str, Any]:
        """Upload multiple files in parallel"""
        uploaded = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_uploads) as executor:
            future_to_path = {
                executor.submit(self.upload_file, p): p for p in file_paths
            }
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    uploaded[path] = future.result()
                except Exception as e:
                    print(f"  Upload failed for {Path(path).name}: {e}")
                    uploaded[path] = None
        return uploaded

    def delete_file(self, uploaded_file: Any):
        """Delete uploaded file from Gemini"""
        try:
            self.client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Best-effort cleanup

    def _apply_video_fps(self, contents: List) -> List:
        """Wrap uploaded video files with videoMetadata(fps=N) when fps != 1."""
        if self.config.video_fps == 1:
            return contents

        wrapped = []
        video_count = 0
        for item in contents:
            # Check if this is an uploaded file reference with a video mime type
            if hasattr(item, 'uri') and hasattr(item, 'mime_type') and \
               item.mime_type and item.mime_type.startswith('video/'):
                try:
                    part = types.Part(
                        file_data=types.FileData(file_uri=item.uri, mime_type=item.mime_type),
                        video_metadata=types.VideoMetadata(fps=self.config.video_fps)
                    )
                    wrapped.append(part)
                    video_count += 1
                except Exception:
                    wrapped.append(item)  # Fallback to raw file
            else:
                wrapped.append(item)
        if video_count > 0 and os.environ.get('V10_DEBUG'):
            print(f"    [FPS={self.config.video_fps} applied to {video_count} video part(s)]")
        return wrapped

    def generate(self, contents: List, temperature: float = None,
                 cached_content: str = None) -> str:
        """Generate content with v10 settings (media_resolution, thinking, safety)"""
        if temperature is None:
            temperature = self.config.temperature

        # Apply custom FPS to video parts
        contents = self._apply_video_fps(contents)

        gen_config = self._build_gen_config(temperature, cached_content)

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=gen_config,
            )
            return self._extract_text(response)
        except Exception as e:
            error_msg = str(e)
            # If media_resolution or thinking not supported, retry without them
            if "media_resolution" in error_msg.lower() or "thinking" in error_msg.lower():
                print(f"  Note: Falling back to basic config ({e})")
                return self._generate_fallback(contents, temperature, cached_content)
            raise

    def _build_gen_config(self, temperature: float, cached_content: str = None) -> types.GenerateContentConfig:
        """Build generation config with all v10 features"""
        safety = [
            types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
        ]

        # Map config string to SDK enum
        resolution_map = {
            'LOW': 'MEDIA_RESOLUTION_LOW',
            'MEDIUM': 'MEDIA_RESOLUTION_MEDIUM',
            'HIGH': 'MEDIA_RESOLUTION_HIGH',
            'ULTRA_HIGH': 'MEDIA_RESOLUTION_HIGH',  # ULTRA_HIGH not yet supported by API, use HIGH
        }

        kwargs = {
            'temperature': temperature,
            'max_output_tokens': self.config.max_output_tokens,
            'safety_settings': safety,
        }

        # Add media_resolution if available in SDK
        try:
            res_value = resolution_map.get(self.config.media_resolution, 'MEDIA_RESOLUTION_MEDIUM')
            media_res = getattr(types.MediaResolution, res_value, None)
            if media_res is not None:
                kwargs['media_resolution'] = media_res
        except (AttributeError, TypeError):
            pass  # SDK version doesn't support media_resolution

        # Add thinking config if available
        try:
            kwargs['thinking_config'] = types.ThinkingConfig(
                thinking_budget=self.config.thinking_budget
            )
        except (AttributeError, TypeError):
            pass  # SDK version doesn't support thinking_config

        if cached_content:
            kwargs['cached_content'] = cached_content

        return types.GenerateContentConfig(**kwargs)

    def _generate_fallback(self, contents: List, temperature: float,
                           cached_content: str = None) -> str:
        """Fallback generation without media_resolution/thinking"""
        safety = [
            types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
        ]

        kwargs = {
            'temperature': temperature,
            'max_output_tokens': self.config.max_output_tokens,
            'safety_settings': safety,
        }
        if cached_content:
            kwargs['cached_content'] = cached_content

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**kwargs),
        )
        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """Extract text from response, handling thinking models"""
        try:
            # response.text should work for most cases
            return response.text
        except Exception:
            pass

        # Manual extraction from candidates
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts or []
                text_parts = []
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        # Skip thinking parts if they have a thought flag
                        if hasattr(part, 'thought') and part.thought:
                            continue
                        text_parts.append(part.text)
                if text_parts:
                    return "\n".join(text_parts)

        raise Exception("No text content in response")

    def create_cache(self, system_instruction: str, contents: List = None) -> Optional[str]:
        """Create a context cache for repeated use. Returns cache name or None."""
        if not self.config.enable_caching:
            return None

        try:
            cache_config = types.CreateCachedContentConfig(
                display_name=f'v10-speaker-context-{int(time.time())}',
                system_instruction=system_instruction,
                ttl=f"{self.config.cache_ttl_seconds}s",
            )
            if contents:
                cache_config.contents = contents

            cache = self.client.caches.create(
                model=self.config.model_name,
                config=cache_config,
            )
            print(f"  Context cache created: {cache.name}")
            return cache.name
        except Exception as e:
            print(f"  Note: Context caching not available ({e}), proceeding without cache")
            return None

    def delete_cache(self, cache_name: str):
        """Delete a context cache"""
        try:
            self.client.caches.delete(name=cache_name)
        except Exception:
            pass


class OverlapChunker:
    """FFmpeg video chunking with 15s overlap and stream copy optimization"""

    def __init__(self, config: TranscriptionConfigV10):
        self.config = config

    def split_video(self, video_path: str, output_dir: str) -> List[Dict]:
        """Split video into overlapping chunks"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        duration_minutes = get_video_duration(str(video_path))
        if duration_minutes <= 0:
            raise ValueError(f"Could not determine duration for {video_path.name}")

        total_seconds = duration_minutes * 60
        chunk_duration = self.config.chunk_duration_seconds
        overlap = self.config.overlap_seconds
        stride = chunk_duration - overlap

        print(f"  Video: {duration_minutes:.1f} min, chunk: {chunk_duration}s, overlap: {overlap}s")

        chunks = []
        chunk_num = 1
        start_time = 0.0

        while start_time < total_seconds:
            end_time = min(start_time + chunk_duration, total_seconds)
            actual_duration = end_time - start_time

            # Skip very short trailing chunks
            if actual_duration < 10 and chunk_num > 1:
                break

            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:03d}.mp4"

            success = self._extract_chunk(
                str(video_path), str(chunk_file), start_time, actual_duration
            )

            if success:
                # transcript_start_time: where actual new content begins
                # For chunk 1: 0. For subsequent chunks: overlap seconds into the chunk.
                transcript_start_offset = overlap if chunk_num > 1 else 0

                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': actual_duration,
                    'transcript_start_offset': transcript_start_offset,
                    'transcript_start_time': start_time + transcript_start_offset,
                    'has_overlap': chunk_num > 1,
                })
                print(f"  Chunk {chunk_num}: {start_time/60:.1f}m - {end_time/60:.1f}m"
                      f"{' (15s overlap)' if chunk_num > 1 else ''}")
            else:
                print(f"  WARNING: Failed to create chunk {chunk_num}")

            start_time += stride
            chunk_num += 1

        return chunks

    def extract_speaker_id_chunks(self, video_path: str, output_dir: str,
                                   num_chunks: int = 2) -> List[Dict]:
        """Extract just the first N chunks for speaker identification"""
        all_chunks = self.split_video(video_path, output_dir)
        return all_chunks[:num_chunks]

    def _extract_chunk(self, input_path: str, output_path: str,
                       start_time: float, duration: float) -> bool:
        """Extract chunk using stream copy first, re-encode fallback"""
        # Try stream copy (fast, no quality loss)
        if self._try_stream_copy(input_path, output_path, start_time, duration):
            return True

        # Fallback: re-encode
        print(f"    Stream copy failed, re-encoding...", end="", flush=True)
        success = self._try_reencode(input_path, output_path, start_time, duration)
        if success:
            print(" done")
        else:
            print(" FAILED")
        return success

    def _try_stream_copy(self, input_path: str, output_path: str,
                         start_time: float, duration: float) -> bool:
        """Try extracting chunk with stream copy (fast)"""
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-i", input_path,
            "-t", str(duration), "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path, "-y"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                return False
            # Verify output file exists and has reasonable size
            output = Path(output_path)
            if output.exists() and output.stat().st_size > 1000:
                return True
            return False
        except (subprocess.TimeoutExpired, Exception):
            return False

    def _try_reencode(self, input_path: str, output_path: str,
                      start_time: float, duration: float) -> bool:
        """Re-encode chunk (slower but more compatible)"""
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-i", input_path,
            "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", output_path, "-y"
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            return Path(output_path).exists()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False


class SpeakerRegistry:
    """Two-pass speaker auto-detection with interactive terminal editor"""

    SPEAKER_ID_PROMPT = """Analyze this classroom video carefully and identify ALL distinct speakers you can see and hear.

For each speaker, provide:
1. A descriptive label that UNIQUELY identifies this person and NO ONE ELSE in the video.
   The label must use a feature that distinguishes them from every other person visible.
   CRITICAL: Before finalizing labels, check that no two people share the same distinguishing
   feature in their label. If two girls both wear grey shirts, do NOT use "GreyShirt" for either.
   Instead use their MOST UNIQUE feature: hair color, hair length, height, specific accessories,
   seating position, etc.

   Good labels (unique identifiers):
   - "Girl-BlondeHair" (only one blonde girl)
   - "Boy-RedHoodie" (only one red hoodie)
   - "Girl-Ponytail" (only one girl with ponytail)
   - "Boy-TallLeft" (the tall boy on the left)

   Bad labels (shared features, will cause confusion):
   - "Girl-GreyShirt" (when multiple girls wear grey)
   - "Boy-Sitting" (when multiple boys are sitting)
   - "Student1", "Student2" (not visually descriptive at all)

2. A detailed physical description covering: hair color/style, clothing specifics,
   position in frame, height relative to others, and any distinguishing accessories.
   This description will be used throughout a long video to consistently identify this person,
   so include as many distinguishing details as possible.

3. Their role: "teacher", "student", or "researcher".

Return ONLY a JSON array with no other text:
[
  {"label": "Teacher", "description": "Adult woman with brown hair in bun, wearing blue striped dress, standing at front of room", "type": "teacher"},
  {"label": "Girl-BlondeHair", "description": "Girl with shoulder-length blonde hair, sitting at center desk, wearing grey t-shirt with blue text", "type": "student"}
]"""

    def __init__(self, gemini_client: GeminiClient, config: TranscriptionConfigV10):
        self.client = gemini_client
        self.config = config

    def identify_speakers(self, uploaded_files: List[Any]) -> List[SpeakerInfo]:
        """Auto-detect speakers from uploaded video chunks"""
        print("\n  Auto-detecting speakers...")

        contents = []
        for f in uploaded_files:
            if f is not None:
                contents.append(f)
        contents.append(self.SPEAKER_ID_PROMPT)

        try:
            response_text = self.client.generate(contents, temperature=0.1)
            speakers = self._parse_speakers_json(response_text)
            if speakers:
                print(f"  Detected {len(speakers)} speakers")
                return speakers
            else:
                print("  Warning: Could not parse speaker detection, using defaults")
                return self._default_speakers()
        except Exception as e:
            print(f"  Speaker detection error: {e}")
            return self._default_speakers()

    def _parse_speakers_json(self, response_text: str) -> List[SpeakerInfo]:
        """Parse JSON speaker list from Gemini response"""
        text = response_text.strip()
        # Strip markdown code blocks
        if '```' in text:
            text = re.sub(r'```(?:json)?\s*\n?', '', text)
            text = text.strip()

        # Find JSON array in response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group(0))
            speakers = []
            for item in data:
                if isinstance(item, dict) and 'label' in item:
                    speakers.append(SpeakerInfo(
                        label=item['label'],
                        description=item.get('description', ''),
                        speaker_type=item.get('type', 'student'),
                    ))
            return speakers
        except json.JSONDecodeError:
            return []

    def _default_speakers(self) -> List[SpeakerInfo]:
        return [
            SpeakerInfo("Teacher", "Main instructor", "teacher"),
            SpeakerInfo("Student1", "Student", "student"),
        ]

    def interactive_edit(self, speakers: List[SpeakerInfo], video_name: str) -> List[SpeakerInfo]:
        """Interactive terminal editor for speaker list"""
        print(f"\n{'='*60}")
        print(f"  SPEAKER IDENTIFICATION: {video_name}")
        print(f"{'='*60}")
        print(f"\n  Review the auto-detected speakers below.")
        print(f"  TIP: Labels should UNIQUELY identify each person.")
        print(f"  Use hair color, position, height, accessories - whatever")
        print(f"  makes them different from everyone else in the video.")

        while True:
            self._display_speakers(speakers)
            self._check_ambiguity(speakers)
            print("\nCommands:")
            print("  <#> rename <NewLabel>     - Rename speaker")
            print("  <#> desc <Description>    - Edit description")
            print("  <#> type <teacher|student> - Change type")
            print("  add <Label> <Description> - Add speaker")
            print("  remove <#>                - Remove speaker")
            print("  done                      - Save and continue")

            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Using current speaker list.")
                break

            if not cmd:
                continue

            if cmd.lower() == 'done':
                break

            speakers = self._process_command(cmd, speakers)

        return speakers

    def _display_speakers(self, speakers: List[SpeakerInfo]):
        """Display formatted speaker table"""
        print(f"\n  {'#':<4} {'Label':<20} {'Type':<10} Description")
        print(f"  {'-'*4} {'-'*20} {'-'*10} {'-'*30}")
        for i, s in enumerate(speakers, 1):
            print(f"  {i:<4} {s.label:<20} {s.speaker_type:<10} {s.description}")

    def _check_ambiguity(self, speakers: List[SpeakerInfo]):
        """Check for potentially ambiguous labels and warn the user"""
        # Extract the descriptor part of each label (after the dash)
        warnings = []
        labels = [s.label for s in speakers]

        # Check for shared descriptor words across labels of the same gender prefix
        for i, s1 in enumerate(speakers):
            for j, s2 in enumerate(speakers):
                if i >= j:
                    continue
                # Extract words from labels
                words1 = set(re.findall(r'[A-Z][a-z]+', s1.label))
                words2 = set(re.findall(r'[A-Z][a-z]+', s2.label))
                # Ignore generic clothing words that are too common to be meaningful
                ignore = {'Boy', 'Girl', 'Teacher', 'Student', 'Shirt', 'Top',
                          'Pants', 'Shorts', 'Shoes', 'Dress', 'Hair'}
                shared = words1 & words2 - ignore
                if shared:
                    warnings.append(
                        f"  WARNING: '{s1.label}' and '{s2.label}' share feature "
                        f"'{', '.join(shared)}' - this may confuse the model. "
                        f"Consider renaming to use unique features (hair, position, etc)."
                    )

        # Check for generic/non-descriptive labels
        for s in speakers:
            if re.match(r'^(Male|Female)?(Student|Speaker)\d*$', s.label):
                warnings.append(
                    f"  WARNING: '{s.label}' is generic. Use a visual feature "
                    f"(e.g., Girl-BlondeHair, Boy-TallLeft) for better accuracy."
                )

        if warnings:
            print()
            for w in warnings:
                print(w)

    def _process_command(self, cmd: str, speakers: List[SpeakerInfo]) -> List[SpeakerInfo]:
        """Process an interactive command"""
        parts = cmd.split(None, 2)

        if parts[0].lower() == 'add' and len(parts) >= 2:
            label = parts[1]
            desc = parts[2] if len(parts) > 2 else ""
            speakers.append(SpeakerInfo(label, desc, "student"))
            print(f"  Added: {label}")
            return speakers

        if parts[0].lower() == 'remove' and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
                if 0 <= idx < len(speakers):
                    removed = speakers.pop(idx)
                    print(f"  Removed: {removed.label}")
                else:
                    print(f"  Invalid index: {parts[1]}")
            except ValueError:
                print(f"  Invalid number: {parts[1]}")
            return speakers

        # Numbered commands: <#> rename/desc/type <value>
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                idx = int(parts[0]) - 1
                action = parts[1].lower()
                value = parts[2]

                if 0 <= idx < len(speakers):
                    if action == 'rename':
                        old = speakers[idx].label
                        speakers[idx].label = value
                        print(f"  Renamed: {old} -> {value}")
                    elif action == 'desc':
                        speakers[idx].description = value
                        print(f"  Updated description for {speakers[idx].label}")
                    elif action == 'type':
                        if value in ('teacher', 'student', 'researcher'):
                            speakers[idx].speaker_type = value
                            print(f"  Updated type for {speakers[idx].label} -> {value}")
                        else:
                            print(f"  Invalid type: {value} (use teacher/student/researcher)")
                else:
                    print(f"  Invalid index: {parts[0]}")
            except (ValueError, IndexError):
                print(f"  Invalid command: {cmd}")
            return speakers

        print(f"  Unknown command: {cmd}")
        return speakers

    @staticmethod
    def save_manifest(speakers: List[SpeakerInfo], output_path: str):
        """Save speaker manifest to JSON"""
        data = [s.to_dict() for s in speakers]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"  Speaker manifest saved: {output_path}")

    @staticmethod
    def load_manifest(manifest_path: str) -> List[SpeakerInfo]:
        """Load speaker manifest from JSON"""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return [SpeakerInfo.from_dict(item) for item in data]


class PromptBuilder:
    """Assembles chunk prompts from components"""

    def __init__(self, prompt_manager: PromptManager, config: TranscriptionConfigV10):
        self.prompt_manager = prompt_manager
        self.config = config

    def build_system_prompt(self, speakers: List[SpeakerInfo]) -> str:
        """Build system instruction with speaker registry (for caching)"""
        base = self.prompt_manager.get_prompt(self.config.prompt_key)

        speaker_section = self._build_speaker_section(speakers)

        anti_hallucination = """
FORMAT RULES:
1. Use MM:SS timestamp format for each speaker turn and visual description.
2. Use the exact speaker labels from the SPEAKER REGISTRY above.
3. Do NOT repeat any word, phrase, or line more than 3 times.

PURPOSE - EDUCATIONAL RESEARCH OBSERVATION:
You are a research assistant helping education researchers understand what is happening
in a classroom. These videos may show small groups at a table OR whole-class instruction.
Your job is to produce a transcript that lets a researcher who wasn't in the room understand:
  - What activity or lesson is happening (what's the task? what's being taught?)
  - What students are physically doing (measuring, counting, drawing, pointing, raising hands)
  - What they're saying - both to each other and to the teacher
  - What the teacher is saying, demonstrating, or presenting
  - What's on the board, projector screen, or shared materials

SPEECH TRANSCRIPTION:
- Write what you actually HEAR, not what "makes sense" in context.
- Messy, partial, incomplete speech is expected and valuable. Kids talk over each other.
- If audio is unclear: use [inaudible]. Use [word?] for uncertain words.
- Do NOT rephrase or clean up what someone said. "We gotta do the thing" stays as-is.
- Teachers typically project more clearly than students - listen carefully for their words.
- For stretches with no audible speech: ONE "[inaudible conversation, ~Ns]" line.

VISUAL DESCRIPTIONS - WHAT MATTERS FOR UNDERSTANDING THE ACTIVITY:
Interleave [bracketed descriptions] that help a researcher understand the learning activity.
Focus on actions that reveal what students are thinking and doing:
  - GESTURES: pointing at specific parts of a shape, tracing along edges, counting with fingers, raising hands
  - MATERIALS: what's written on whiteboards, projector slides, worksheets, or papers (read actual content)
  - MANIPULATION: measuring with fingers, arranging materials, drawing diagrams, writing numbers
  - COLLABORATION: who's showing work to whom, who's leading the discussion
  - WHOLE CLASS: what the teacher writes/draws on the board, what's on the projected slide, how students respond

SPEAKER IDENTIFICATION:
When many students are present, identify speakers by visible features (hair, clothing, position).
If you can't identify a specific student, use "Student" rather than guessing wrong.
Track recurring speakers consistently - if Girl-PinkShirt speaks in one segment, use the same label later.

Do NOT waste description on irrelevant actions (adjusting hair, fidgeting, generic "looking at desk").
DO describe where exactly someone points, what part of a shape they trace, what they write.

Example - GOOD (helps understand the activity):
  00:08 [Boy-RedShirt points at the bottom-right corner of a triangle made of paper strips on the desk]
  00:09 Girl-Glasses: One, two, three, four, five, six, seven, eight.
  00:09 [Girl-Glasses traces her finger along the right side of the triangle, counting marks between the bottom-right and top corners]
  00:16 Girl-Glasses: That's only eight steps.
  00:18 [Boy-RedShirt picks up a marker and starts counting marks along the hypotenuse of the triangle]
  00:22 [Girl-Glasses points to the whiteboard where "10 steps = 30 inches" is written]
  00:25 [inaudible conversation, ~5s]
  00:25 [Both students lean over the triangle, comparing the side lengths]

Example - BAD (generic, doesn't help understand the activity):
  00:08 [The students look at items on the desk]
  00:09 Girl-Glasses: One, two, three, four, five, six, seven, eight.
  00:16 Girl-Glasses: That's only eight steps.
  00:18 [Boy-RedShirt picks up a marker]
  00:22 [Girl-Glasses points to the whiteboard]

HALLUCINATION CHECK - watch for these patterns in your own output:
- Rapid back-and-forth where each person says one clean sentence (real kids interrupt and overlap)
- Synonymous rephrasing in consecutive lines ("rotate/turn/pivot/spin" - pick what you hear)
- Dialogue that reads like a textbook example of classroom interaction
- Visual actions you're inferring rather than seeing (e.g., "cuts with scissors" when they're tracing)"""

        return f"{base}\n\n{speaker_section}\n{anti_hallucination}"

    def build_chunk_prompt(self, chunk_info: Dict, chunk_number: int,
                           total_chunks: int, speakers: List[SpeakerInfo],
                           previous_context: str = None,
                           include_base: bool = False) -> str:
        """Build prompt for a specific chunk"""
        parts = []

        # Include base prompt + speakers if not using cache
        if include_base:
            parts.append(self.build_system_prompt(speakers))

        # Chunk info with duration
        chunk_duration_secs = int(chunk_info['end_time'] - chunk_info['start_time'])
        chunk_duration_mm = chunk_duration_secs // 60
        chunk_duration_ss = chunk_duration_secs % 60
        parts.append(f"\n--- CHUNK {chunk_number}/{total_chunks} ---")
        parts.append(f"This video clip is {chunk_duration_mm}:{chunk_duration_ss:02d} long ({chunk_duration_secs} seconds).")
        parts.append(f"Transcribe the ENTIRE clip from start to finish. Your last timestamp should be near {chunk_duration_mm}:{chunk_duration_ss:02d}.")
        parts.append(f"Do NOT stop early and do NOT generate timestamps beyond {chunk_duration_mm}:{chunk_duration_ss:02d}.")

        # Overlap instruction for chunks 2+
        if chunk_info.get('has_overlap', False):
            overlap_secs = self.config.overlap_seconds
            parts.append(f"""
IMPORTANT - VIDEO OVERLAP HANDLING:
The first {overlap_secs} seconds of this video (00:00 to 00:{overlap_secs:02d}) are REPEATED
from the previous chunk. You have already transcribed this content. Do NOT transcribe it again.

Use those first {overlap_secs} seconds ONLY to visually identify which speakers are which,
so you can maintain consistent speaker labels. Then begin your actual transcript at 00:{overlap_secs:02d}.

Your FIRST transcript line must have a timestamp of 00:{overlap_secs:02d} or later.
Do NOT output any lines with timestamps before 00:{overlap_secs:02d}.
Do NOT re-transcribe dialogue from the CONTINUITY CONTEXT below - that is already captured.""")

        # Continuity context
        if previous_context and chunk_number > 1:
            context_lines = self._extract_context_lines(previous_context)
            if context_lines:
                parts.append(f"""
CONTINUITY CONTEXT (last lines from previous chunk):
{context_lines}

Continue naturally from this context. Maintain speaker label consistency.
Start timestamps from 00:{self.config.overlap_seconds:02d} for this chunk.""")

        parts.append("\nBegin transcription:")
        return "\n".join(parts)

    def _build_speaker_section(self, speakers: List[SpeakerInfo]) -> str:
        """Build speaker registry section"""
        if not speakers:
            return ""

        lines = ["SPEAKER REGISTRY (use these exact labels):"]
        for s in speakers:
            lines.append(f"- {s.label}: {s.description} [{s.speaker_type}]")
        lines.append("")
        lines.append("Use these exact speaker labels consistently throughout the transcript.")
        return "\n".join(lines)

    def _extract_context_lines(self, previous_transcript: str) -> str:
        """Extract last N clean transcript lines for context.
        Skips [inaudible]-only lines to avoid encouraging that pattern."""
        lines = previous_transcript.strip().split('\n')
        context = []

        for line in reversed(lines):
            line = line.strip()
            if line and re.match(r'^\d{1,2}:\d{2}', line):
                # Skip lines that are just [inaudible]
                if re.match(r'^\d{1,2}:\d{2}\s+\S+:\s*\[inaudible\]\s*$', line):
                    continue
                # Clean quality flags
                clean = re.sub(r'[\u2705\u26a0\ufe0f\U0001F6A8]\s*', '', line)
                clean = re.sub(r'\*[^*]+\*', '', clean).strip()
                if clean:
                    context.insert(0, clean)
            if len(context) >= self.config.continuity_context_lines:
                break

        return "\n".join(context)


# =============================================================================
# ORCHESTRATION
# =============================================================================

class VideoTranscriptionPipelineV10:
    """Single-video transcription pipeline"""

    def __init__(self, api_key: str, config: TranscriptionConfigV10):
        self.config = config
        self.client = GeminiClient(api_key, config)
        self.chunker = OverlapChunker(config)
        self.speaker_registry = SpeakerRegistry(self.client, config)
        self.cost_calculator = VideoCostCalculator()

        # Load prompts
        script_dir = Path(__file__).parent
        prompts_file = script_dir / "prompts.json"
        self.prompt_manager = PromptManager(str(prompts_file))
        self.prompt_builder = PromptBuilder(self.prompt_manager, config)
        self.validator = TranscriptValidator(config.min_transcript_length)

    def process(self, video_path: str, output_dir: str = None,
                speakers: List[SpeakerInfo] = None,
                skip_confirmation: bool = False) -> Dict:
        """Process a single video end-to-end"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = video_path.parent / f"{video_path.stem}_v10_{timestamp}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        duration_minutes = get_video_duration(str(video_path))
        if duration_minutes <= 0:
            raise ValueError(f"Could not determine duration for {video_path.name}")

        # Display info and cost estimate
        self._display_info(video_path, duration_minutes)
        if not skip_confirmation:
            response = input("\nProceed? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return {}

        start_time = time.time()
        chunks_dir = output_dir / "chunks"
        uploaded_files = {}
        cache_name = None

        try:
            # Phase 1: Chunking
            print(f"\n{'='*60}")
            print("PHASE 1: CHUNKING")
            print(f"{'='*60}")
            report_progress(self.config, "progress", chunk=0, total=100, percent=0, status="chunking")

            chunk_list = self.chunker.split_video(str(video_path), str(chunks_dir))
            if not chunk_list:
                raise Exception("No chunks created")
            print(f"  Created {len(chunk_list)} chunks")

            # Phase 2: Upload first chunks for speaker ID (if needed)
            print(f"\n{'='*60}")
            print("PHASE 2: SPEAKER IDENTIFICATION")
            print(f"{'='*60}")
            report_progress(self.config, "progress", chunk=0, total=100, percent=10, status="identifying speakers")

            if speakers is None:
                # Upload first N chunks for speaker detection
                id_chunks = chunk_list[:self.config.speaker_id_chunks]
                id_paths = [c['file_path'] for c in id_chunks]
                id_uploaded = self.client.upload_files_parallel(id_paths)

                # Auto-detect speakers
                id_files = [id_uploaded[p] for p in id_paths if id_uploaded.get(p)]
                speakers = self.speaker_registry.identify_speakers(id_files)

                # Interactive editing
                speakers = self.speaker_registry.interactive_edit(speakers, video_path.name)

                # Save manifest
                manifest_path = output_dir / f"{video_path.stem}_speakers.json"
                SpeakerRegistry.save_manifest(speakers, str(manifest_path))

                # Track uploaded files for cleanup later
                for path, f in id_uploaded.items():
                    uploaded_files[path] = f
            else:
                print(f"  Using provided speaker manifest ({len(speakers)} speakers)")

            # Phase 3: Upload remaining chunks
            print(f"\n{'='*60}")
            print("PHASE 3: UPLOADING CHUNKS")
            print(f"{'='*60}")
            report_progress(self.config, "progress", chunk=0, total=100, percent=20, status="uploading")

            remaining_paths = [c['file_path'] for c in chunk_list if c['file_path'] not in uploaded_files]
            if remaining_paths:
                new_uploaded = self.client.upload_files_parallel(remaining_paths)
                uploaded_files.update(new_uploaded)

            # Phase 4: Context cache (optional)
            print(f"\n{'='*60}")
            print("PHASE 4: CONTEXT SETUP")
            print(f"{'='*60}")

            system_prompt = self.prompt_builder.build_system_prompt(speakers)
            cache_name = self.client.create_cache(system_prompt)
            use_cache = cache_name is not None
            if use_cache:
                print(f"  Using context cache for {len(chunk_list)} chunks")
            else:
                print(f"  Inline prompt mode (no cache)")

            # Phase 5: Sequential transcription
            print(f"\n{'='*60}")
            print("PHASE 5: TRANSCRIPTION")
            print(f"{'='*60}")

            all_transcripts = []
            previous_transcript = None

            for chunk_info in chunk_list:
                chunk_num = chunk_info['chunk_number']
                total = len(chunk_list)
                percent = 30 + int((chunk_num / total) * 60)

                print(f"\n  Transcribing chunk {chunk_num}/{total}...")
                report_progress(self.config, "progress", chunk=chunk_num, total=total,
                              percent=percent, status="transcribing")

                uploaded = uploaded_files.get(chunk_info['file_path'])
                if uploaded is None:
                    print(f"  WARNING: No upload for chunk {chunk_num}, skipping")
                    all_transcripts.append({
                        'chunk_number': chunk_num,
                        'chunk_info': chunk_info,
                        'transcript': f"[CHUNK_{chunk_num}_UPLOAD_FAILED]",
                    })
                    continue

                # Build prompt
                chunk_prompt = self.prompt_builder.build_chunk_prompt(
                    chunk_info, chunk_num, total, speakers,
                    previous_context=previous_transcript,
                    include_base=not use_cache,
                )

                # Transcribe with retries
                transcript = self._transcribe_chunk(
                    uploaded, chunk_prompt, cache_name, chunk_num
                )

                all_transcripts.append({
                    'chunk_number': chunk_num,
                    'chunk_info': chunk_info,
                    'transcript': transcript,
                })

                # Save individual chunk transcript
                chunk_file = output_dir / f"chunk_{chunk_num:03d}_transcript.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)

                # Update context for next chunk
                if not transcript.startswith('['):
                    previous_transcript = transcript

            # Phase 6: Assembly
            print(f"\n{'='*60}")
            print("PHASE 6: ASSEMBLY")
            print(f"{'='*60}")
            report_progress(self.config, "progress", chunk=0, total=100, percent=95, status="assembling")

            combined = self._assemble_transcript(all_transcripts, video_path, speakers)

            # Save outputs
            if self.config.dual_output:
                research_file = output_dir / f"{video_path.stem}_transcript.txt"
                with open(research_file, 'w', encoding='utf-8') as f:
                    f.write(combined)

                clean = self._create_clean_transcript(combined)
                transana_file = output_dir / f"{video_path.stem}_transana.txt"
                with open(transana_file, 'w', encoding='utf-8') as f:
                    f.write(clean)

                # SRT export
                srt_file = output_dir / f"{video_path.stem}.srt"
                SubtitleExporter.to_srt(clean, str(srt_file))

                print(f"\n  Research (annotated): {research_file}")
                print(f"  Transana (clean):     {transana_file}")
                print(f"  Subtitles (SRT):      {srt_file}")
            else:
                final_file = output_dir / f"{video_path.stem}_transcript.txt"
                with open(final_file, 'w', encoding='utf-8') as f:
                    f.write(combined)
                print(f"\n  Transcript: {final_file}")

            # Cleanup
            if not self.config.keep_chunks:
                self._cleanup_chunks(chunks_dir)

            elapsed = time.time() - start_time
            successful = sum(1 for t in all_transcripts if not t['transcript'].startswith('['))

            print(f"\n{'='*60}")
            print(f"V10 TRANSCRIPTION COMPLETE")
            print(f"{'='*60}")
            print(f"  Chunks: {successful}/{len(all_transcripts)} successful")
            print(f"  Time: {elapsed/60:.1f} minutes")
            print(f"  Output: {output_dir}")

            result = {
                'video_file': str(video_path),
                'output_dir': str(output_dir),
                'chunks_processed': len(all_transcripts),
                'successful_chunks': successful,
                'elapsed_seconds': elapsed,
                'speakers': [s.to_dict() for s in speakers],
            }

            report_progress(self.config, "complete",
                          output_file=str(output_dir), stats=result)
            return result

        except Exception as e:
            report_progress(self.config, "error", message=str(e), fatal=True)
            raise

        finally:
            # Cleanup uploaded files
            for f in uploaded_files.values():
                if f is not None:
                    self.client.delete_file(f)
            if cache_name:
                self.client.delete_cache(cache_name)

    def _transcribe_chunk(self, uploaded_file: Any, prompt: str,
                          cache_name: Optional[str], chunk_num: int) -> str:
        """Transcribe a single chunk with retry logic"""
        max_attempts = self.config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"    Retry {attempt-1}/{self.config.max_retries}...")
                time.sleep(self.config.retry_delay)

            try:
                temp = self.config.temperature if attempt == 1 else min(0.3, self.config.temperature + 0.1)
                contents = [uploaded_file, prompt]

                transcript = self.client.generate(
                    contents, temperature=temp, cached_content=cache_name
                )

                is_valid, reason = self.validator.is_valid(transcript)
                if is_valid:
                    print(f"    Valid transcript (attempt {attempt})")
                    return transcript
                else:
                    print(f"    Validation failed: {reason}")
                    if "repetition" in reason.lower() and attempt < max_attempts:
                        prompt = prompt + "\n\nCRITICAL: Do NOT repeat any word or phrase. Transcribe naturally."
                    if attempt == max_attempts:
                        return f"[CHUNK_{chunk_num}_NEEDS_REVIEW: {reason}]"

            except Exception as e:
                print(f"    Error: {e}")
                if attempt == max_attempts:
                    return f"[CHUNK_{chunk_num}_ERROR: {str(e)}]"

        return f"[CHUNK_{chunk_num}_FAILED]"

    def _assemble_transcript(self, all_transcripts: List[Dict],
                             video_path: Path, speakers: List[SpeakerInfo]) -> str:
        """Combine chunk transcripts with timestamp adjustment"""
        combined = []

        # Header
        combined.append("=" * 80)
        combined.append("COMPLETE TRANSCRIPT - V10")
        combined.append("=" * 80)
        combined.append(f"Video: {video_path.name}")
        combined.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        combined.append(f"Model: {self.config.model_name}")
        combined.append(f"Media Resolution: {self.config.media_resolution}")
        combined.append(f"Chunk Duration: {self.config.chunk_duration_seconds}s, Overlap: {self.config.overlap_seconds}s")
        combined.append(f"Speakers: {', '.join(s.label for s in speakers)}")
        combined.append("=" * 80)
        combined.append("")

        for td in all_transcripts:
            chunk_num = td['chunk_number']
            chunk_info = td['chunk_info']
            transcript = td['transcript'].strip()

            # The model outputs timestamps relative to the chunk video file.
            # For chunk 2+, the model starts at 00:15 (skipping overlap).
            # We add start_time (when the chunk begins in absolute video time)
            # to convert to absolute timestamps. NOT transcript_start_time,
            # which would double-count the overlap offset.
            offset_seconds = chunk_info['start_time']
            start_label = TimestampNormalizer.from_seconds(chunk_info['start_time'])
            end_label = TimestampNormalizer.from_seconds(chunk_info['end_time'])

            combined.append(f"--- CHUNK {chunk_num} ({start_label} - {end_label}) ---")

            is_failed = transcript.startswith('[')
            if is_failed or not transcript:
                combined.append(f"[CHUNK {chunk_num} FAILED]")
                if transcript:
                    combined.append(transcript)
            else:
                # Strip any echoed speaker registry or preamble from model output
                cleaned_transcript = self._strip_preamble(transcript)
                adjusted = TimestampNormalizer.adjust_transcript(cleaned_transcript, offset_seconds)
                # Clip lines with timestamps past the chunk's end time
                clipped = self._clip_past_end(adjusted, chunk_info['end_time'])
                combined.append(clipped)

            combined.append("")

        return "\n".join(combined)

    def _clip_past_end(self, transcript: str, end_time_seconds: float) -> str:
        """Remove lines with timestamps that exceed the chunk's end time.
        Prevents hallucinated content past the actual video end."""
        lines = transcript.split('\n')
        kept = []
        for line in lines:
            # Check if line has a timestamp
            match = re.match(r'^(\d{1,2}):(\d{2})', line.strip())
            if match:
                mins, secs = int(match.group(1)), int(match.group(2))
                line_seconds = mins * 60 + secs
                if line_seconds > end_time_seconds + 2:  # 2s grace for rounding
                    continue  # Skip lines past video end
            kept.append(line)
        return '\n'.join(kept)

    def _strip_preamble(self, transcript: str) -> str:
        """Remove echoed speaker registry or preamble before first timestamp line"""
        lines = transcript.split('\n')
        # Find first line that starts with a timestamp (MM:SS)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^\d{1,2}:\d{2}\s', stripped) or re.match(r'^\[\d{1,2}:\d{2}', stripped):
                return '\n'.join(lines[i:])
            # Also keep lines that start with [ (visual descriptions with timestamps)
            if stripped.startswith('[') and re.search(r'\d{1,2}:\d{2}', stripped):
                return '\n'.join(lines[i:])
        # No timestamp found, return as-is
        return transcript

    def _create_clean_transcript(self, transcript: str) -> str:
        """Create clean Transana-compatible transcript"""
        lines = transcript.split('\n')
        clean = []

        for line in lines:
            line = line.strip()
            # Skip metadata/headers
            if line.startswith('===') or line.startswith('---'):
                continue
            if any(line.startswith(p) for p in [
                'Generated:', 'Model:', 'Media Resolution:', 'Chunk Duration:',
                'Speakers:', 'COMPLETE TRANSCRIPT', 'Video:'
            ]):
                continue
            if line.startswith('[CHUNK') or line.startswith('[PARTIAL'):
                continue
            if not line:
                continue

            # Remove quality flags
            clean_line = re.sub(r'[\u2705\u26a0\ufe0f\U0001F6A8]\s*', '', line)
            clean_line = re.sub(r'\*[^*]+\*', '', clean_line).strip()

            if clean_line:
                clean.append(clean_line)

        return '\n'.join(clean)

    def _display_info(self, video_path: Path, duration_minutes: float):
        """Display processing info and cost estimate"""
        print(f"\n{'='*60}")
        print("VIDEO TRANSCRIPTION PIPELINE V10")
        print(f"{'='*60}")
        print(f"  Video:       {video_path.name}")
        print(f"  Duration:    {duration_minutes:.1f} minutes")
        print(f"  Model:       {self.config.model_name}")
        print(f"  Resolution:  {self.config.media_resolution}")
        print(f"  FPS:         {self.config.video_fps}")
        print(f"  Chunk:       {self.config.chunk_duration_seconds}s + {self.config.overlap_seconds}s overlap")
        print(f"  Thinking:    budget={self.config.thinking_budget}")

        est = self.cost_calculator.estimate(duration_minutes, self.config)
        print(f"\n  Estimated chunks: {est['num_chunks']}")
        print(f"  Estimated cost:   ${est['total_cost']:.3f}")
        print(f"  Tokens/frame:     {est['tokens_per_frame']}")

    def _cleanup_chunks(self, chunks_dir: Path):
        """Remove temporary chunk files"""
        try:
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
                print(f"  Cleaned up: {chunks_dir}")
        except Exception as e:
            print(f"  Cleanup warning: {e}")


class BatchProcessor:
    """Multi-video parallel batch processing"""

    MANIFEST_FILENAME = "batch_manifest.json"

    def __init__(self, api_key: str, config: TranscriptionConfigV10):
        self.api_key = api_key
        self.config = config
        self.client = GeminiClient(api_key, config)
        self.chunker = OverlapChunker(config)
        self.speaker_registry = SpeakerRegistry(self.client, config)

    def identify_all(self, video_dir: Path, prompt_key: str = "default") -> Dict:
        """Interactive speaker identification for all videos in directory"""
        videos = find_videos(video_dir)
        if not videos:
            print(f"No video files found in {video_dir}")
            return {}

        print(f"\n{'='*60}")
        print(f"BATCH SPEAKER IDENTIFICATION")
        print(f"{'='*60}")
        print(f"  Directory: {video_dir}")
        print(f"  Videos found: {len(videos)}")
        print(f"  Prompt: {prompt_key}")

        # Load or create batch manifest
        manifest_path = video_dir / self.MANIFEST_FILENAME
        manifest = self._load_manifest(manifest_path)

        for i, video_path in enumerate(videos, 1):
            video_name = video_path.name
            print(f"\n{'='*60}")
            print(f"  VIDEO {i}/{len(videos)}: {video_name}")
            print(f"{'='*60}")

            # Skip if already identified
            if video_name in manifest.get('videos', {}) and \
               manifest['videos'][video_name].get('status') == 'identified':
                print(f"  Already identified. Skipping. (delete manifest entry to redo)")
                continue

            # Extract first chunks for speaker ID
            with tempfile.TemporaryDirectory() as tmp_dir:
                id_chunks = self.chunker.extract_speaker_id_chunks(
                    str(video_path), tmp_dir, self.config.speaker_id_chunks
                )

                if not id_chunks:
                    print(f"  WARNING: Could not extract chunks, skipping")
                    continue

                # Upload chunks
                id_paths = [c['file_path'] for c in id_chunks]
                id_uploaded = self.client.upload_files_parallel(id_paths)
                id_files = [id_uploaded[p] for p in id_paths if id_uploaded.get(p)]

                try:
                    # Auto-detect speakers
                    speakers = self.speaker_registry.identify_speakers(id_files)

                    # Interactive editing
                    speakers = self.speaker_registry.interactive_edit(speakers, video_name)

                    # Save speaker manifest
                    speaker_manifest_name = f"{video_path.stem}_speakers.json"
                    speaker_manifest_path = video_dir / speaker_manifest_name
                    SpeakerRegistry.save_manifest(speakers, str(speaker_manifest_path))

                    # Update batch manifest
                    if 'videos' not in manifest:
                        manifest['videos'] = {}
                    manifest['videos'][video_name] = {
                        'speaker_manifest': speaker_manifest_name,
                        'prompt_key': prompt_key,
                        'status': 'identified',
                    }
                    self._save_manifest(manifest, manifest_path)

                finally:
                    # Cleanup uploaded files
                    for f in id_uploaded.values():
                        if f is not None:
                            self.client.delete_file(f)

        print(f"\n{'='*60}")
        print(f"IDENTIFICATION COMPLETE")
        print(f"{'='*60}")
        identified = sum(1 for v in manifest.get('videos', {}).values()
                        if v.get('status') == 'identified')
        print(f"  Identified: {identified}/{len(videos)} videos")
        print(f"  Manifest: {manifest_path}")

        return manifest

    def process_batch(self, video_dir: Path, output_base: str = None) -> List[Dict]:
        """Process all identified videos in parallel"""
        manifest_path = video_dir / self.MANIFEST_FILENAME
        manifest = self._load_manifest(manifest_path)

        if not manifest.get('videos'):
            print("No videos in manifest. Run 'identify' first.")
            return []

        # Filter to identified videos
        to_process = {
            name: info for name, info in manifest['videos'].items()
            if info.get('status') == 'identified'
        }

        if not to_process:
            print("No identified videos to process. Run 'identify' first.")
            return []

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING")
        print(f"{'='*60}")
        print(f"  Videos: {len(to_process)}")
        print(f"  Workers: {self.config.parallel_videos}")

        # Cost estimate
        video_paths = [video_dir / name for name in to_process.keys()]
        existing_paths = [p for p in video_paths if p.exists()]
        cost_calc = VideoCostCalculator()
        batch_est = cost_calc.estimate_batch(existing_paths, self.config)
        print(f"  Estimated total cost: ${batch_est['total_cost']:.2f}")

        confirm = input("\nProceed with batch processing? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return []

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_videos) as executor:
            futures = {}
            for video_name, video_info in to_process.items():
                video_path = video_dir / video_name
                if not video_path.exists():
                    print(f"  WARNING: {video_name} not found, skipping")
                    continue

                # Load speaker manifest
                speaker_path = video_dir / video_info['speaker_manifest']
                if not speaker_path.exists():
                    print(f"  WARNING: Speaker manifest not found for {video_name}, skipping")
                    continue

                speakers = SpeakerRegistry.load_manifest(str(speaker_path))

                # Output directory
                if output_base:
                    out_dir = Path(output_base) / video_path.stem
                else:
                    out_dir = video_dir / f"{video_path.stem}_v10_transcript"

                # Create config with video-specific prompt
                video_config = TranscriptionConfigV10(
                    chunk_duration_seconds=self.config.chunk_duration_seconds,
                    overlap_seconds=self.config.overlap_seconds,
                    model_name=self.config.model_name,
                    media_resolution=self.config.media_resolution,
                    thinking_budget=self.config.thinking_budget,
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    video_fps=self.config.video_fps,
                    parallel_videos=1,  # Each worker handles one video
                    parallel_uploads=self.config.parallel_uploads,
                    max_retries=self.config.max_retries,
                    retry_delay=self.config.retry_delay,
                    min_transcript_length=self.config.min_transcript_length,
                    continuity_context_lines=self.config.continuity_context_lines,
                    dual_output=self.config.dual_output,
                    keep_chunks=self.config.keep_chunks,
                    prompt_key=video_info.get('prompt_key', self.config.prompt_key),
                    json_progress=False,  # No JSON in batch mode
                    enable_caching=self.config.enable_caching,
                    cache_ttl_seconds=self.config.cache_ttl_seconds,
                )

                future = executor.submit(
                    self._process_single_video,
                    str(video_path), str(out_dir), speakers, video_config
                )
                futures[future] = video_name

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                video_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    manifest['videos'][video_name]['status'] = 'completed'
                    print(f"\n  COMPLETED: {video_name} "
                          f"({result.get('successful_chunks', 0)}/{result.get('chunks_processed', 0)} chunks)")
                except Exception as e:
                    print(f"\n  FAILED: {video_name}: {e}")
                    manifest['videos'][video_name]['status'] = 'failed'
                    manifest['videos'][video_name]['error'] = str(e)

                # Save manifest after each video
                self._save_manifest(manifest, manifest_path)

        elapsed = time.time() - start_time
        completed = sum(1 for r in results if r)

        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Completed: {completed}/{len(to_process)} videos")
        print(f"  Time: {elapsed/60:.1f} minutes")

        return results

    def _process_single_video(self, video_path: str, output_dir: str,
                               speakers: List[SpeakerInfo],
                               config: TranscriptionConfigV10) -> Dict:
        """Process a single video (called from batch worker)"""
        pipeline = VideoTranscriptionPipelineV10(self.api_key, config)
        return pipeline.process(
            video_path, output_dir, speakers=speakers, skip_confirmation=True
        )

    def _load_manifest(self, path: Path) -> Dict:
        """Load or create batch manifest"""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {"version": "1.0", "videos": {}}

    def _save_manifest(self, manifest: Dict, path: Path):
        """Save batch manifest"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def cmd_identify(args):
    """Handle 'identify' subcommand"""
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: Provide API key via --api-key or GOOGLE_API_KEY env var")
        sys.exit(1)

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    config = TranscriptionConfigV10(
        model_name=args.model,
        prompt_key=args.prompt,
        media_resolution=args.resolution,
        video_fps=args.fps,
    )

    if path.is_file():
        # Single video: identify speakers and save manifest
        video_dir = path.parent
        batch = BatchProcessor(api_key, config)

        # Temporarily modify manifest to only process this video
        manifest_path = video_dir / BatchProcessor.MANIFEST_FILENAME
        manifest = batch._load_manifest(manifest_path)

        # Process just this video through identify_all logic
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = GeminiClient(api_key, config)
            chunker = OverlapChunker(config)
            registry = SpeakerRegistry(client, config)

            id_chunks = chunker.extract_speaker_id_chunks(str(path), tmp_dir, config.speaker_id_chunks)
            if not id_chunks:
                print("ERROR: Could not extract chunks")
                sys.exit(1)

            id_paths = [c['file_path'] for c in id_chunks]
            id_uploaded = client.upload_files_parallel(id_paths)
            id_files = [id_uploaded[p] for p in id_paths if id_uploaded.get(p)]

            try:
                speakers = registry.identify_speakers(id_files)
                speakers = registry.interactive_edit(speakers, path.name)
                manifest_out = video_dir / f"{path.stem}_speakers.json"
                SpeakerRegistry.save_manifest(speakers, str(manifest_out))
            finally:
                for f in id_uploaded.values():
                    if f is not None:
                        client.delete_file(f)
    else:
        # Directory: identify all videos
        batch = BatchProcessor(api_key, config)
        batch.identify_all(path, prompt_key=args.prompt)


def cmd_process(args):
    """Handle 'process' subcommand"""
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: Provide API key via --api-key or GOOGLE_API_KEY env var")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    config = TranscriptionConfigV10(
        model_name=args.model,
        prompt_key=args.prompt,
        media_resolution=args.resolution,
        chunk_duration_seconds=int(args.chunk_minutes * 60),
        overlap_seconds=args.overlap,
        max_retries=args.max_retries,
        dual_output=not args.single_output,
        keep_chunks=args.keep_chunks,
        json_progress=args.json_progress,
        thinking_budget=args.thinking_budget,
        video_fps=args.fps,
    )

    # Load speakers from manifest if provided
    speakers = None
    if args.speakers:
        speaker_path = Path(args.speakers)
        if speaker_path.exists():
            speakers = SpeakerRegistry.load_manifest(str(speaker_path))
            print(f"Loaded {len(speakers)} speakers from {speaker_path.name}")

    pipeline = VideoTranscriptionPipelineV10(api_key, config)
    pipeline.process(
        str(video_path), args.output,
        speakers=speakers,
        skip_confirmation=args.no_confirm,
    )


def cmd_batch(args):
    """Handle 'batch' subcommand"""
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: Provide API key via --api-key or GOOGLE_API_KEY env var")
        sys.exit(1)

    video_dir = Path(args.path)
    if not video_dir.is_dir():
        print(f"ERROR: Directory not found: {video_dir}")
        sys.exit(1)

    config = TranscriptionConfigV10(
        model_name=args.model,
        media_resolution=args.resolution,
        parallel_videos=args.workers,
        chunk_duration_seconds=int(args.chunk_minutes * 60),
        overlap_seconds=args.overlap,
        dual_output=not args.single_output,
        keep_chunks=args.keep_chunks,
        thinking_budget=args.thinking_budget,
        video_fps=args.fps,
    )

    batch = BatchProcessor(api_key, config)
    batch.process_batch(video_dir, args.output)


def cmd_estimate(args):
    """Handle 'estimate' subcommand"""
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    config = TranscriptionConfigV10(
        model_name=args.model,
        media_resolution=args.resolution,
        chunk_duration_seconds=int(args.chunk_minutes * 60),
        overlap_seconds=args.overlap,
        video_fps=args.fps,
    )

    cost_calc = VideoCostCalculator()

    if path.is_file():
        duration = get_video_duration(str(path))
        if duration <= 0:
            print("ERROR: Could not determine video duration")
            sys.exit(1)

        est = cost_calc.estimate(duration, config)

        print(f"\n{'='*60}")
        print("V10 COST ESTIMATE")
        print(f"{'='*60}")
        print(f"  Video:          {path.name}")
        print(f"  Duration:       {duration:.1f} minutes")
        print(f"  Model:          {config.model_name}")
        print(f"  Resolution:     {config.media_resolution}")
        print(f"  Tokens/frame:   {est['tokens_per_frame']}")
        print(f"  Chunks:         {est['num_chunks']}")
        print(f"  Speaker ID:     {est['speaker_id_calls']} API calls")
        print(f"  Input tokens:   {est['total_input_tokens']:,}")
        print(f"  Output tokens:  {est['total_output_tokens']:,}")
        print(f"  Input cost:     ${est['input_cost']:.3f}")
        print(f"  Output cost:    ${est['output_cost']:.3f}")
        print(f"  TOTAL COST:     ${est['total_cost']:.3f}")
    else:
        videos = find_videos(path)
        if not videos:
            print(f"No video files found in {path}")
            sys.exit(1)

        batch_est = cost_calc.estimate_batch(videos, config)

        print(f"\n{'='*60}")
        print("V10 BATCH COST ESTIMATE")
        print(f"{'='*60}")
        print(f"  Directory:      {path}")
        print(f"  Videos:         {batch_est['num_videos']}")
        print(f"  Total chunks:   {batch_est['total_chunks']}")
        print(f"  Model:          {config.model_name}")
        print(f"  Resolution:     {config.media_resolution}")
        print(f"  TOTAL COST:     ${batch_est['total_cost']:.2f}")

        print(f"\n  Per-video breakdown:")
        for v in batch_est['videos']:
            print(f"    {v['video']:<40} {v['duration_minutes']:.0f}min  "
                  f"{v['num_chunks']} chunks  ${v['total_cost']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        prog="v10",
        description="V10 Video Transcription Pipeline - Batch Classroom Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Estimate cost for a single video
  python video_transcription_pipeline_v10.py estimate video.mp4

  # Process a single video (interactive speaker ID + transcription)
  python video_transcription_pipeline_v10.py process video.mp4

  # Identify speakers for all videos in a folder
  python video_transcription_pipeline_v10.py identify ./videos/ --prompt smallgroup_ben

  # Batch process (unattended, uses saved manifests)
  python video_transcription_pipeline_v10.py batch ./videos/ --workers 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommand')

    # Common arguments
    def add_common_args(p):
        p.add_argument("--api-key", help="Gemini API key (or set GOOGLE_API_KEY)")
        p.add_argument("-m", "--model", default="gemini-3-flash-preview",
                       help="Gemini model (default: gemini-3-flash-preview)")
        p.add_argument("--resolution", default="HIGH", choices=["LOW", "MEDIUM", "HIGH"],
                       help="Media resolution (default: HIGH)")
        p.add_argument("--fps", type=int, default=2,
                       help="Video frame sampling rate in FPS (default: 2). Higher = more visual detail but more tokens.")

    def add_chunk_args(p):
        p.add_argument("-c", "--chunk-minutes", type=float, default=1.0,
                       help="Chunk duration in minutes (default: 1.0)")
        p.add_argument("--overlap", type=int, default=15,
                       help="Overlap seconds between chunks (default: 15)")

    # identify
    p_identify = subparsers.add_parser('identify',
        help='Interactive speaker identification for videos')
    p_identify.add_argument("path", help="Video file or directory")
    p_identify.add_argument("-p", "--prompt", default="default",
                           help="Prompt key from prompts.json")
    add_common_args(p_identify)
    p_identify.set_defaults(func=cmd_identify)

    # process
    p_process = subparsers.add_parser('process',
        help='Process a single video (interactive speaker ID + transcription)')
    p_process.add_argument("video", help="Path to video file")
    p_process.add_argument("-o", "--output", help="Output directory")
    p_process.add_argument("-p", "--prompt", default="default",
                          help="Prompt key from prompts.json")
    p_process.add_argument("--speakers", help="Path to speaker manifest JSON")
    p_process.add_argument("--no-confirm", action="store_true",
                          help="Skip confirmation prompt")
    p_process.add_argument("--single-output", action="store_true",
                          help="Single output instead of dual (research + transana)")
    p_process.add_argument("--keep-chunks", action="store_true",
                          help="Keep chunk files after processing")
    p_process.add_argument("--max-retries", type=int, default=3,
                          help="Max retry attempts per chunk (default: 3)")
    p_process.add_argument("--json-progress", action="store_true",
                          help="Output JSON progress for Electron integration")
    p_process.add_argument("--thinking-budget", type=int, default=4096,
                          help="Thinking token budget (default: 4096)")
    add_common_args(p_process)
    add_chunk_args(p_process)
    p_process.set_defaults(func=cmd_process)

    # batch
    p_batch = subparsers.add_parser('batch',
        help='Batch process videos using saved speaker manifests')
    p_batch.add_argument("path", help="Directory containing videos and manifests")
    p_batch.add_argument("-o", "--output", help="Base output directory")
    p_batch.add_argument("-w", "--workers", type=int, default=10,
                        help="Parallel video workers (default: 10)")
    p_batch.add_argument("--single-output", action="store_true",
                        help="Single output instead of dual")
    p_batch.add_argument("--keep-chunks", action="store_true",
                        help="Keep chunk files after processing")
    p_batch.add_argument("--thinking-budget", type=int, default=1024,
                        help="Thinking token budget (default: 1024)")
    add_common_args(p_batch)
    add_chunk_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    # estimate
    p_estimate = subparsers.add_parser('estimate',
        help='Estimate transcription cost')
    p_estimate.add_argument("path", help="Video file or directory")
    add_common_args(p_estimate)
    add_chunk_args(p_estimate)
    p_estimate.set_defaults(func=cmd_estimate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nV10 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
