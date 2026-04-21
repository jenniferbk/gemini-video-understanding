#!/usr/bin/env python3
"""
Video Transcription Pipeline V07 for Educational Research
Clean, self-contained pipeline with Gemini 3 Flash support.

NEW IN V07:
- Updated to Gemini 3 Flash (gemini-3-flash-preview)
- Self-contained: no external pipeline dependencies
- Default FPS increased to 2 for better speaker tracking
- Automatic temp file cleanup (--keep-chunks to preserve)
- Updated cost calculator with Gemini 3 pricing
- Code cleanup: fixed exception handling, removed duplicates

V04/V06 FEATURES RETAINED:
- Hybrid VAD preprocessing (Frame-level VAD + Whisper ASR timestamps)
- Classroom-optimized denoising with student voice preservation
- VAD-informed intelligent chunking at speech boundaries
- VAD confidence integration into consensus analysis
- Enhanced short segment detection for student voices
"""

import os
import sys
import time
import json
import argparse
import re
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
import warnings

# Core dependencies
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    sys.exit(1)

# VAD and audio processing dependencies
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available - audio preprocessing will be limited")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not available - denoising will be disabled")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not available - ASR-based VAD will be disabled")

try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not available - advanced VAD will be disabled")

# BERT dependencies for consensus
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: BERT libraries not available - falling back to basic similarity")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TranscriptionConfigV07:
    """Configuration for V07 transcription pipeline"""
    # Core settings
    chunk_duration_minutes: float = 3.0
    overlap_seconds: int = 10
    max_file_size_mb: int = 95
    model_name: str = "gemini-3-flash-preview"  # V07: Updated to Gemini 3

    # FPS setting - V07: default increased to 2
    fps: int = 2

    # VAD settings
    enable_vad_preprocessing: bool = True
    vad_chunk_overlap: float = 0.5
    vad_confidence_threshold: float = 0.6
    whisper_model: str = "large-v3"
    frame_vad_model: str = "wav2vec2-large-robust"

    # Denoising settings
    enable_denoising: bool = True
    denoising_strength: float = 0.6
    denoise_as_augmentation: bool = True

    # Chunking settings
    vad_informed_chunking: bool = True
    min_speech_gap: float = 2.0
    preserve_speech_boundaries: bool = True

    # Consensus settings
    vad_weight_in_consensus: float = 0.3
    consensus_runs: int = 1
    consensus_threshold: float = 0.7

    # Other settings
    thinking: bool = True
    prompt_key: str = "enhanced_vad"
    precise_chunking: bool = True
    enable_repetition_filter: bool = True
    max_retries: int = 3
    min_transcript_length: int = 50
    retry_delay: float = 5.0

    # V07: Temp file management
    keep_chunks: bool = False


# =============================================================================
# CONSTANTS (extracted magic numbers)
# =============================================================================

# VAD constants
VAD_FRAME_DURATION_SEC = 0.02  # 20ms frames
VAD_MIN_SEGMENT_DURATION_SEC = 0.1  # 100ms minimum
VAD_SMOOTHING_WINDOW = 5
VAD_ALPHA_WEIGHT = 0.7  # Weight for frame-level VAD vs Whisper

# Temporal matching
TEMPORAL_MATCH_WINDOW_SEC = 5  # Seconds tolerance for matching
CHUNK_BOUNDARY_SEARCH_SEC = 30  # Search range for optimal boundaries

# Token calculation
FRAME_TOKENS_PER_SECOND_BASE = 258  # At 1 FPS
AUDIO_TOKENS_PER_SECOND = 32
METADATA_TOKENS_PER_SECOND = 10
OUTPUT_TOKEN_RATIO = 0.15


# =============================================================================
# INLINED CLASSES FROM V03 (self-contained)
# =============================================================================

class TranscriptValidator:
    """Validate transcription results and detect failures"""

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def is_valid_transcription(self, transcript: str, file_name: str = "") -> Tuple[bool, str]:
        """
        Validate if transcription is successful or failed
        Returns (is_valid, failure_reason)
        """
        if not transcript or not isinstance(transcript, str):
            return False, "Empty or invalid transcript"

        transcript = transcript.strip()

        # Check for explicit error markers
        error_patterns = [
            r'\[ERROR:',
            r'\[PARTIAL:.*Generation stopped',
            r'Transcription failed',
            r'No response candidates',
            r'Invalid response',
            r'No content parts'
        ]

        for pattern in error_patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                return False, f"Error marker detected: {pattern}"

        # Check minimum length
        if len(transcript) < self.min_length:
            return False, f"Transcript too short: {len(transcript)} chars (min: {self.min_length})"

        # Check for reasonable content structure
        lines = transcript.split('\n')
        valid_lines = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for timestamp patterns
            timestamp_patterns = [
                r'^\d{1,2}:\d{2}',      # MM:SS
                r'^\(\d{1,2}:\d{2}',    # (MM:SS
                r'^\d{1,2}:\d{2}:\d{2}', # HH:MM:SS
                r'^\[\d{1,2}:\d{2}'     # [MM:SS
            ]

            has_timestamp = any(re.search(pattern, line) for pattern in timestamp_patterns)
            has_speaker = ':' in line and len(line.split(':', 1)) == 2

            if has_timestamp or has_speaker:
                valid_lines += 1

        if valid_lines == 0:
            return False, "No valid transcript lines with timestamps/speakers found"

        # Check for excessive repetition
        if self._detect_excessive_repetition(transcript):
            return False, "Excessive repetition detected (likely AI generation loop)"

        return True, "Valid transcript"

    def _detect_excessive_repetition(self, transcript: str) -> bool:
        """Detect if transcript has excessive repetitive content"""
        lines = transcript.split('\n')
        if len(lines) < 10:
            return False

        content_lines = []
        for line in lines:
            if ':' in line:
                try:
                    content = line.split(':', 1)[1].strip()
                    content_lines.append(content)
                except Exception:  # V07: Fixed bare except
                    pass

        if len(content_lines) < 5:
            return False

        content_counts = Counter(content_lines)
        most_common = content_counts.most_common(1)[0]

        if most_common[1] > len(content_lines) * 0.3:
            return True

        return False


class PromptManager:
    """Manage transcription prompts from external files"""

    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = Path(prompts_file)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        """Load prompts from JSON file"""
        if not self.prompts_file.exists():
            default_prompts = {
                "basic": {
                    "name": "Basic Transcription",
                    "description": "Simple speaker identification and timestamps",
                    "prompt": "Transcribe this classroom video with speaker identification and timestamps.\n\nFormat: MM:SS Speaker: content\n\nInclude brief [visual actions] when relevant.\n\nProvide an accurate, concise transcript."
                },
                "enhanced_vad": {
                    "name": "VAD-Enhanced Classroom Transcription",
                    "description": "VAD-guided transcription with hybrid speech detection",
                    "prompt": """Please transcribe this classroom video with enhanced speaker diarization.

CONTEXT: Classroom video.

SPEAKERS TO IDENTIFY:
- Teacher_1: The main teacher
- Teacher_2: If a second adult is present
- Student(brief description): Identify student speakers by position and description

TRANSCRIPTION REQUIREMENTS:
1. FORMAT: MM:SS SPEAKER: content [visual actions]
2. Use [uncertain] for ambiguous speaker identification

QUALITY PRIORITIES:
- Accurate speaker identification using multi-modal cues
- Precise timestamp alignment
- Capture all speech including quiet student voices
- Minimize repetition and false transcriptions

Begin transcription:"""
                }
            }
            with open(self.prompts_file, 'w') as f:
                json.dump(default_prompts, f, indent=2)
            return default_prompts

        try:
            with open(self.prompts_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading prompts from {self.prompts_file}: {e}")
            return {"basic": {"name": "Basic", "description": "Default prompt", "prompt": "Transcribe this video."}}

    def get_prompt(self, key: str) -> str:
        """Get prompt text by key"""
        if key not in self.prompts:
            print(f"Warning: Prompt '{key}' not found. Using 'basic' instead.")
            key = "basic"
        return self.prompts[key]["prompt"]

    def list_prompts(self) -> None:
        """Display available prompts"""
        print("\nAvailable prompts:")
        print("=" * 50)
        for key, data in self.prompts.items():
            print(f"  {key:20} - {data['name']}")
            print(f"  {' ' * 20}   {data['description']}")
            print()

    def get_prompt_keys(self) -> List[str]:
        """Get list of available prompt keys"""
        return list(self.prompts.keys())


class VideoCostCalculator:
    """Calculate processing costs for video transcription - V07 Updated"""

    # V07: Updated token costs for Gemini 3 (December 2025)
    TOKEN_COSTS = {
        "gemini-3-flash-preview": {
            "input": 0.0005,   # $0.50 per 1M tokens = $0.0005 per 1K
            "output": 0.003    # $3.00 per 1M tokens = $0.003 per 1K
        },
        "gemini-2.5-pro-preview-05-06": {
            "input_low": 0.00125,
            "input_high": 0.0025,
            "output_low": 0.010,
            "output_high": 0.015,
            "threshold": 200000
        },
        "gemini-2.0-flash-exp": {
            "input": 0.000075,
            "output": 0.0003
        },
        "gemini-2.5-flash-preview-05-20": {
            "input": 0.000075,
            "output": 0.0003
        },
    }

    @classmethod
    def estimate_cost(cls, duration_minutes: float, model: str, chunk_minutes: float, fps: int = 2) -> Dict:
        """Estimate processing cost for video"""
        total_seconds = duration_minutes * 60

        # Token calculation scaled by FPS
        frame_tokens_per_second = FRAME_TOKENS_PER_SECOND_BASE * fps
        tokens_per_second = frame_tokens_per_second + AUDIO_TOKENS_PER_SECOND + METADATA_TOKENS_PER_SECOND
        total_input_tokens = total_seconds * tokens_per_second

        # Output tokens estimate
        estimated_output_tokens = total_input_tokens * OUTPUT_TOKEN_RATIO

        num_chunks = max(1, int(duration_minutes / chunk_minutes))

        if model in cls.TOKEN_COSTS:
            rates = cls.TOKEN_COSTS[model]

            if "threshold" in rates:  # Tiered pricing
                threshold = rates["threshold"]

                if total_input_tokens <= threshold:
                    input_cost = (total_input_tokens / 1000) * rates["input_low"]
                else:
                    low_tier_cost = (threshold / 1000) * rates["input_low"]
                    high_tier_cost = ((total_input_tokens - threshold) / 1000) * rates["input_high"]
                    input_cost = low_tier_cost + high_tier_cost

                if estimated_output_tokens <= threshold:
                    output_cost = (estimated_output_tokens / 1000) * rates["output_low"]
                else:
                    low_tier_cost = (threshold / 1000) * rates["output_low"]
                    high_tier_cost = ((estimated_output_tokens - threshold) / 1000) * rates["output_high"]
                    output_cost = low_tier_cost + high_tier_cost

            else:  # Flat pricing
                input_cost = (total_input_tokens / 1000) * rates["input"]
                output_cost = (estimated_output_tokens / 1000) * rates["output"]

            total_cost = input_cost + output_cost
        else:
            input_cost = output_cost = total_cost = 0

        return {
            "duration_minutes": duration_minutes,
            "fps": fps,
            "tokens_per_second": int(tokens_per_second),
            "total_tokens_estimated": int(total_input_tokens + estimated_output_tokens),
            "num_chunks": num_chunks,
            "input_cost": round(input_cost, 3),
            "output_cost": round(output_cost, 3),
            "total_cost": round(total_cost, 3),
            "model": model
        }


# =============================================================================
# AUDIO PROCESSING CLASSES
# =============================================================================

class ClassroomDenoiser:
    """Implements classroom-optimized denoising that preserves student voices"""

    def __init__(self, strength: float = 0.6):
        self.strength = strength
        self.available = NOISEREDUCE_AVAILABLE and LIBROSA_AVAILABLE

        if not self.available:
            print("Warning: Denoising disabled - missing dependencies (noisereduce, librosa)")

    def process_audio_file(self, audio_path: str) -> Tuple[Optional[str], str]:
        """
        Process audio file and return (clean_path, original_path)
        Returns (None, original_path) if denoising unavailable
        """
        if not self.available:
            return None, audio_path

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            denoised_audio = self._classroom_specific_denoise(audio, sr)

            audio_path_obj = Path(audio_path)
            clean_path = audio_path_obj.parent / f"{audio_path_obj.stem}_denoised{audio_path_obj.suffix}"

            sf.write(str(clean_path), denoised_audio, sr)

            return str(clean_path), audio_path

        except Exception as e:
            print(f"Warning: Denoising failed for {audio_path}: {e}")
            return None, audio_path

    def _classroom_specific_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Optimized denoising that preserves student voices"""
        try:
            denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,
                prop_decrease=self.strength,
                freq_mask_smooth_hz=100,
                time_mask_smooth_ms=50
            )
            return denoised

        except Exception as e:
            print(f"Denoising error: {e}, returning original audio")
            return audio


class HybridVADPreprocessor:
    """
    Hybrid VAD combining frame-level VAD and Whisper ASR timestamps
    """

    def __init__(self, config: TranscriptionConfigV07):
        self.config = config
        self.whisper_model = None
        self.frame_vad_available = False
        self._initialize_models()

    def _initialize_models(self):
        """Initialize VAD models with fallback handling"""
        if WHISPER_AVAILABLE and self.config.enable_vad_preprocessing:
            try:
                print(f"Loading Whisper {self.config.whisper_model} for hybrid VAD...")
                self.whisper_model = whisper.load_model(self.config.whisper_model)
                print("Whisper loaded successfully")
            except Exception as e:
                print(f"Failed to load Whisper: {e}")
                self.whisper_model = None

        if TRANSFORMERS_AVAILABLE and self.config.enable_vad_preprocessing:
            try:
                print("Loading frame-level VAD model...")
                print("Frame-level VAD using simplified implementation")
                self.frame_vad_available = True
            except Exception as e:
                print(f"Failed to load frame VAD: {e}")
                self.frame_vad_available = False

    def process_audio_chunk(self, audio_path: str) -> Dict:
        """Process audio chunk with hybrid VAD approach"""
        if not self.config.enable_vad_preprocessing:
            return self._create_fallback_result(audio_path)

        results = {
            'audio_path': audio_path,
            'speech_segments': [],
            'vad_confidence': [],
            'whisper_result': None,
            'hybrid_vad_available': False,
            'processing_stats': {}
        }

        try:
            if not LIBROSA_AVAILABLE:
                return self._create_fallback_result(audio_path)

            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            whisper_vad = self._extract_whisper_vad(audio_path)
            if whisper_vad:
                results['whisper_result'] = whisper_vad
                results['speech_segments'].extend(whisper_vad.get('segments', []))

            frame_vad = self._extract_frame_vad(audio, sr)

            if whisper_vad and frame_vad is not None:
                hybrid_confidence = self._combine_vad_outputs(frame_vad, whisper_vad, duration)
                results['vad_confidence'] = hybrid_confidence
                results['hybrid_vad_available'] = True
                results['speech_segments'] = self._extract_enhanced_segments(hybrid_confidence, sr)

            results['processing_stats'] = {
                'duration_seconds': duration,
                'num_segments': len(results['speech_segments']),
                'speech_ratio': self._calculate_speech_ratio(results['speech_segments'], duration),
                'avg_segment_duration': self._calculate_avg_segment_duration(results['speech_segments']),
                'whisper_available': whisper_vad is not None,
                'frame_vad_available': frame_vad is not None
            }

        except Exception as e:
            print(f"Warning: VAD processing error for {audio_path}: {e}")
            return self._create_fallback_result(audio_path)

        return results

    def _extract_whisper_vad(self, audio_path: str) -> Optional[Dict]:
        """Extract word-level timestamps using Whisper"""
        if not self.whisper_model:
            return None

        try:
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0
            )

            return {
                'text': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'en')
            }

        except Exception as e:
            print(f"Whisper VAD error: {e}")
            return None

    def _extract_frame_vad(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Simplified frame-level VAD implementation"""
        if not self.frame_vad_available:
            return None

        try:
            frame_length = int(VAD_FRAME_DURATION_SEC * sr)
            hop_length = frame_length // 2

            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)

            threshold = np.percentile(energy, 30)
            vad_predictions = (energy > threshold).astype(float)

            vad_smoothed = self._smooth_vad_predictions(vad_predictions)

            return vad_smoothed

        except Exception as e:
            print(f"Frame VAD error: {e}")
            return None

    def _combine_vad_outputs(self, frame_vad: np.ndarray, whisper_result: Dict, duration: float) -> np.ndarray:
        """Combine frame-level VAD with Whisper timestamps"""
        frame_rate = len(frame_vad) / duration
        whisper_frames = np.zeros_like(frame_vad)

        for segment in whisper_result.get('segments', []):
            start_frame = int(segment['start'] * frame_rate)
            end_frame = int(segment['end'] * frame_rate)
            end_frame = min(end_frame, len(whisper_frames))
            whisper_frames[start_frame:end_frame] = 1.0

        combined = VAD_ALPHA_WEIGHT * frame_vad + (1 - VAD_ALPHA_WEIGHT) * whisper_frames

        return combined

    def _extract_enhanced_segments(self, vad_confidence: np.ndarray, sr: int) -> List[Dict]:
        """Extract speech segments from hybrid VAD confidence"""
        segments = []

        threshold = self.config.vad_confidence_threshold
        binary_vad = vad_confidence > threshold

        in_speech = False
        start_time = 0

        for i, is_speech in enumerate(binary_vad):
            current_time = i * VAD_FRAME_DURATION_SEC

            if is_speech and not in_speech:
                start_time = current_time
                in_speech = True
            elif not is_speech and in_speech:
                if current_time - start_time > VAD_MIN_SEGMENT_DURATION_SEC:
                    start_idx = int(start_time / VAD_FRAME_DURATION_SEC)
                    segments.append({
                        'start': start_time,
                        'end': current_time,
                        'duration': current_time - start_time,
                        'confidence': float(np.mean(vad_confidence[start_idx:i]))
                    })
                in_speech = False

        if in_speech:
            end_time = len(binary_vad) * VAD_FRAME_DURATION_SEC
            if end_time - start_time > VAD_MIN_SEGMENT_DURATION_SEC:
                start_idx = int(start_time / VAD_FRAME_DURATION_SEC)
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'confidence': float(np.mean(vad_confidence[start_idx:]))
                })

        return segments

    def _smooth_vad_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Apply smoothing to VAD predictions"""
        kernel = np.ones(VAD_SMOOTHING_WINDOW) / VAD_SMOOTHING_WINDOW
        smoothed = np.convolve(predictions, kernel, mode='same')
        return smoothed

    def _calculate_speech_ratio(self, segments: List[Dict], total_duration: float) -> float:
        """Calculate ratio of speech to total duration"""
        if not segments or total_duration == 0:
            return 0.0
        total_speech = sum(seg['duration'] for seg in segments)
        return total_speech / total_duration

    def _calculate_avg_segment_duration(self, segments: List[Dict]) -> float:
        """Calculate average segment duration"""
        if not segments:
            return 0.0
        return sum(seg['duration'] for seg in segments) / len(segments)

    def _create_fallback_result(self, audio_path: str) -> Dict:
        """Create fallback result when VAD is unavailable"""
        return {
            'audio_path': audio_path,
            'speech_segments': [],
            'vad_confidence': [],
            'whisper_result': None,
            'hybrid_vad_available': False,
            'processing_stats': {
                'fallback_mode': True,
                'reason': 'VAD preprocessing disabled or dependencies unavailable'
            }
        }


# =============================================================================
# CHUNKING AND PROCESSING
# =============================================================================

class VADInformedChunker:
    """Create video chunks that respect speech boundaries using VAD analysis"""

    def __init__(self, config: TranscriptionConfigV07):
        self.config = config
        self.vad_processor = HybridVADPreprocessor(config)
        self.denoiser = ClassroomDenoiser(config.denoising_strength) if config.enable_denoising else None
        self._temp_files: List[str] = []  # V07: Track temp files for cleanup

    def split_video_with_vad(self, video_path: str, output_dir: str) -> List[Dict]:
        """Split video using VAD analysis to preserve speech boundaries"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        duration_minutes = self._get_video_duration(str(video_path))
        if duration_minutes == 0:
            raise ValueError("Could not determine video duration")

        print(f"VAD-informed chunking for {duration_minutes:.1f}-minute video...")

        audio_path = self._extract_audio(video_path, output_dir)
        self._temp_files.append(audio_path)  # Track for cleanup

        vad_results = self.vad_processor.process_audio_chunk(audio_path)

        if not vad_results['hybrid_vad_available'] or not self.config.vad_informed_chunking:
            print("Falling back to traditional chunking")
            return self._traditional_chunking(video_path, output_dir, vad_results)

        chunk_boundaries = self._find_optimal_boundaries(vad_results, duration_minutes * 60)

        chunks = self._create_chunks_at_boundaries(video_path, output_dir, chunk_boundaries, vad_results)

        if self.denoiser and self.config.denoise_as_augmentation:
            chunks = self._add_denoised_versions(chunks)

        print(f"Created {len(chunks)} VAD-informed chunks")
        return chunks

    def cleanup_temp_files(self):
        """V07: Clean up temporary files"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file}: {e}")
        self._temp_files.clear()

    def _find_optimal_boundaries(self, vad_results: Dict, total_duration: float) -> List[float]:
        """Find optimal chunk boundaries using VAD analysis"""
        target_duration = self.config.chunk_duration_minutes * 60
        speech_segments = vad_results['speech_segments']

        if not speech_segments:
            return list(range(int(target_duration), int(total_duration), int(target_duration)))

        boundaries = []
        current_time = 0

        while current_time + target_duration < total_duration:
            ideal_boundary = current_time + target_duration

            optimal_boundary = self._find_nearest_speech_gap(
                speech_segments, ideal_boundary, self.config.min_speech_gap
            )

            if optimal_boundary and optimal_boundary > current_time + target_duration * 0.7:
                boundaries.append(optimal_boundary)
                current_time = optimal_boundary
            else:
                boundaries.append(ideal_boundary)
                current_time = ideal_boundary

        return boundaries

    def _find_nearest_speech_gap(self, segments: List[Dict], target_time: float, min_gap: float) -> Optional[float]:
        """Find speech gap nearest to target time"""
        best_gap_time = None
        best_distance = float('inf')

        for i in range(len(segments) - 1):
            gap_start = segments[i]['end']
            gap_end = segments[i + 1]['start']
            gap_duration = gap_end - gap_start

            if gap_duration >= min_gap:
                gap_center = (gap_start + gap_end) / 2
                distance = abs(gap_center - target_time)

                if distance < best_distance and distance < CHUNK_BOUNDARY_SEARCH_SEC:
                    best_distance = distance
                    best_gap_time = gap_center

        return best_gap_time

    def _create_chunks_at_boundaries(self, video_path: Path, output_dir: Path,
                                   boundaries: List[float], vad_results: Dict) -> List[Dict]:
        """Create video chunks at specified boundaries"""
        chunks = []
        start_time = 0

        for i, end_time in enumerate(boundaries + [None]):
            chunk_num = i + 1

            if end_time is None:
                duration_minutes = self._get_video_duration(str(video_path))
                end_time = duration_minutes * 60

            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:02d}.mp4"

            success = self._extract_video_chunk(str(video_path), str(chunk_file), start_time, end_time - start_time)

            if success:
                chunk_vad_info = self._extract_chunk_vad_info(vad_results, start_time, end_time)

                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'vad_info': chunk_vad_info
                })

                print(f"  Chunk {chunk_num}: {start_time//60:02.0f}:{start_time%60:05.2f} - {end_time//60:02.0f}:{end_time%60:05.2f}")

            start_time = end_time

        return chunks

    def _extract_chunk_vad_info(self, vad_results: Dict, start_time: float, end_time: float) -> Dict:
        """Extract VAD information for specific time range"""
        chunk_segments = []

        for segment in vad_results.get('speech_segments', []):
            if segment['end'] > start_time and segment['start'] < end_time:
                adjusted_segment = {
                    'start': max(0, segment['start'] - start_time),
                    'end': min(end_time - start_time, segment['end'] - start_time),
                    'duration': 0,
                    'confidence': segment.get('confidence', 1.0)
                }
                adjusted_segment['duration'] = adjusted_segment['end'] - adjusted_segment['start']

                if adjusted_segment['duration'] > VAD_MIN_SEGMENT_DURATION_SEC:
                    chunk_segments.append(adjusted_segment)

        chunk_duration = end_time - start_time
        return {
            'speech_segments': chunk_segments,
            'speech_ratio': sum(s['duration'] for s in chunk_segments) / chunk_duration if chunk_duration > 0 else 0,
            'num_segments': len(chunk_segments),
            'avg_confidence': np.mean([s['confidence'] for s in chunk_segments]) if chunk_segments else 0.0
        }

    def _add_denoised_versions(self, chunks: List[Dict]) -> List[Dict]:
        """Add denoised versions of chunks for data augmentation"""
        if not self.denoiser:
            return chunks

        for chunk in chunks:
            audio_path = self._extract_audio_from_chunk(chunk['file_path'])
            self._temp_files.append(audio_path)  # Track for cleanup

            clean_audio_path, noisy_audio_path = self.denoiser.process_audio_file(audio_path)
            if clean_audio_path:
                self._temp_files.append(clean_audio_path)  # Track for cleanup

            chunk['audio_files'] = {
                'original': noisy_audio_path,
                'denoised': clean_audio_path
            }

        return chunks

    def _traditional_chunking(self, video_path: Path, output_dir: Path, vad_results: Dict) -> List[Dict]:
        """Fallback to traditional time-based chunking"""
        duration_minutes = self._get_video_duration(str(video_path))
        chunk_duration_seconds = self.config.chunk_duration_minutes * 60

        chunks = []
        chunk_num = 1
        start_time = 0

        while start_time < duration_minutes * 60:
            end_time = min(start_time + chunk_duration_seconds, duration_minutes * 60)

            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:02d}.mp4"

            success = self._extract_video_chunk(str(video_path), str(chunk_file), start_time, end_time - start_time)

            if success:
                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'vad_info': {'fallback_mode': True}
                })
                print(f"  Chunk {chunk_num}: {start_time:.0f}s-{end_time:.0f}s ({end_time-start_time:.0f}s duration)")

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

    def _extract_audio(self, video_path: Path, output_dir: Path) -> str:
        """Extract audio track from video for VAD analysis"""
        audio_path = output_dir / f"{video_path.stem}_audio.wav"

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            raise

    def _extract_video_chunk(self, input_path: str, output_path: str, start_time: float, duration: float) -> bool:
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

    def _extract_audio_from_chunk(self, chunk_path: str) -> str:
        """Extract audio from video chunk"""
        chunk_path_obj = Path(chunk_path)
        audio_path = chunk_path_obj.parent / f"{chunk_path_obj.stem}_audio.wav"

        cmd = [
            "ffmpeg", "-i", chunk_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(audio_path)
        except subprocess.CalledProcessError:
            return chunk_path


# =============================================================================
# CONSENSUS ANALYSIS
# =============================================================================

class VADEnhancedConsensusAnalyzer:
    """Enhanced consensus analyzer that incorporates VAD confidence"""

    def __init__(self, consensus_threshold: float = 0.7, vad_weight: float = 0.3):
        self.consensus_threshold = consensus_threshold
        self.vad_weight = vad_weight

        self.bert_model = None
        self.bert_available = BERT_AVAILABLE
        if self.bert_available:
            try:
                print("Loading BERT model for hybrid semantic similarity...")
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("BERT model loaded successfully")
            except Exception as e:
                print(f"Failed to load BERT model: {e}")
                self.bert_available = False

    def analyze_with_vad_confidence(self, transcript_runs: List[str], vad_info_list: List[Dict]) -> Dict:
        """Enhanced consensus analysis using VAD confidence"""

        if len(transcript_runs) == 1:
            return {
                'flagged_transcript': transcript_runs[0],
                'analysis_summary': {
                    'total_runs': 1,
                    'vad_enhanced': False,
                    'quality_level': 'SINGLE_RUN'
                }
            }

        print(f"\n=== VAD-ENHANCED CONSENSUS ANALYSIS ===")
        algorithm_type = "Hybrid BERT + VAD" if self.bert_available else "Basic + VAD"
        print(f"Using {algorithm_type} algorithm")

        baseline_transcript, baseline_index, baseline_info = self._select_baseline_with_vad(
            transcript_runs, vad_info_list
        )

        aligned_data = self._align_transcripts_with_vad(
            baseline_transcript, transcript_runs, vad_info_list[baseline_index]
        )

        flagged_lines = self._flag_with_vad_confidence(aligned_data)

        return self._generate_vad_enhanced_output(flagged_lines, transcript_runs, vad_info_list)

    def _select_baseline_with_vad(self, transcript_runs: List[str],
                                 vad_info_list: List[Dict]) -> Tuple[str, int, Dict]:
        """Select baseline transcript considering VAD quality"""

        scores = []
        for i, (transcript, vad_info) in enumerate(zip(transcript_runs, vad_info_list)):
            base_score = self._score_transcript_quality(transcript, transcript_runs)
            vad_bonus = self._calculate_vad_quality_bonus(vad_info)
            total_score = base_score + vad_bonus

            scores.append({
                'index': i,
                'score': total_score,
                'base_score': base_score,
                'vad_bonus': vad_bonus,
                'transcript': transcript
            })

            print(f"    Run {i+1}: {total_score:.1f}/110 points (base: {base_score:.1f}, VAD: {vad_bonus:.1f})")

        best_run = max(scores, key=lambda x: x['score'])
        print(f"  BASELINE SELECTED: Run {best_run['index']+1} with {best_run['score']:.1f}/110 points")

        return best_run['transcript'], best_run['index'], {
            'score': best_run['score'],
            'vad_enhanced': True
        }

    def _calculate_vad_quality_bonus(self, vad_info: Dict) -> float:
        """Calculate quality bonus based on VAD analysis (0-10 points)"""
        if vad_info.get('fallback_mode', False):
            return 0.0

        bonus = 0.0

        speech_ratio = vad_info.get('speech_ratio', 0.0)
        if 0.3 <= speech_ratio <= 0.8:
            bonus += 3.0
        elif speech_ratio > 0.1:
            bonus += 1.0

        avg_confidence = vad_info.get('avg_confidence', 0.0)
        bonus += avg_confidence * 4.0

        num_segments = vad_info.get('num_segments', 0)
        if num_segments > 5:
            bonus += 2.0
        elif num_segments > 2:
            bonus += 1.0

        if vad_info.get('hybrid_vad_available', False):
            bonus += 1.0

        return min(bonus, 10.0)

    def _align_transcripts_with_vad(self, baseline: str, all_transcripts: List[str],
                                   vad_info: Dict) -> List[Dict]:
        """Align transcripts with VAD confidence integration"""

        baseline_lines = self._parse_transcript(baseline)
        other_transcript_lines = [self._parse_transcript(t) for t in all_transcripts]
        vad_confidence_map = self._create_vad_confidence_map(vad_info)

        aligned_data = []

        for baseline_line in baseline_lines:
            baseline_time = self._timestamp_to_seconds(baseline_line['timestamp'])
            vad_confidence = vad_confidence_map.get(int(baseline_time), 0.5)

            alignment = {
                'baseline': baseline_line,
                'matches': [],
                'speaker_agreements': [],
                'content_agreements': [],
                'vad_confidence': vad_confidence
            }

            for other_lines in other_transcript_lines:
                best_match = self._find_temporal_match(baseline_line, other_lines)

                if best_match:
                    alignment['matches'].append(best_match)

                    speaker_match = baseline_line['speaker'].lower() == best_match['speaker'].lower()
                    content_sim = self._enhanced_content_similarity(
                        baseline_line['content'], best_match['content']
                    )

                    alignment['speaker_agreements'].append(speaker_match)
                    alignment['content_agreements'].append(content_sim)

            aligned_data.append(alignment)

        return aligned_data

    def _create_vad_confidence_map(self, vad_info: Dict) -> Dict[int, float]:
        """Create timestamp -> confidence mapping from VAD info"""
        confidence_map = {}

        for segment in vad_info.get('speech_segments', []):
            start_sec = int(segment['start'])
            end_sec = int(segment['end'])
            confidence = segment.get('confidence', 0.5)

            for sec in range(start_sec, end_sec + 1):
                confidence_map[sec] = max(confidence_map.get(sec, 0), confidence)

        return confidence_map

    def _flag_with_vad_confidence(self, aligned_data: List[Dict]) -> List[Dict]:
        """Generate quality flags incorporating VAD confidence"""

        flagged_lines = []

        for alignment in aligned_data:
            baseline_line = alignment['baseline']
            speaker_agreements = alignment['speaker_agreements']
            content_agreements = alignment['content_agreements']
            vad_confidence = alignment['vad_confidence']

            speaker_confidence = sum(speaker_agreements) / max(len(speaker_agreements), 1) if speaker_agreements else 1.0
            content_confidence = sum(content_agreements) / max(len(content_agreements), 1) if content_agreements else 1.0

            vad_boost = vad_confidence * self.vad_weight
            enhanced_speaker_conf = min(1.0, speaker_confidence + vad_boost * 0.5)
            enhanced_content_conf = min(1.0, content_confidence + vad_boost)

            if enhanced_speaker_conf >= 0.7 and enhanced_content_conf >= 0.75:
                flag = "OK"
                review_needed = False
                flag_reason = f"VAD:{vad_confidence:.2f}"
            elif enhanced_speaker_conf >= 0.5 and enhanced_content_conf >= 0.6:
                flag = "REVIEW"
                review_needed = True
                flag_reason = f"SPK{enhanced_speaker_conf*100:.0f}CNT{enhanced_content_conf*100:.0f}VAD{vad_confidence:.2f}"
            else:
                flag = "CHECK"
                review_needed = True
                flag_reason = f"SPK{enhanced_speaker_conf*100:.0f}CNT{enhanced_content_conf*100:.0f}VAD{vad_confidence:.2f}"

            flagged_lines.append({
                'timestamp': baseline_line['timestamp'],
                'speaker': baseline_line['speaker'],
                'content': baseline_line['content'],
                'visual': baseline_line.get('visual', ''),
                'flag': flag,
                'review_needed': review_needed,
                'flag_reason': flag_reason,
                'confidence': {
                    'speaker': enhanced_speaker_conf,
                    'content': enhanced_content_conf,
                    'vad': vad_confidence
                },
                'original_confidence': {
                    'speaker': speaker_confidence,
                    'content': content_confidence
                }
            })

        return flagged_lines

    def _generate_vad_enhanced_output(self, flagged_lines: List[Dict],
                                     transcript_runs: List[str], vad_info_list: List[Dict]) -> Dict:
        """Generate final output with VAD enhancement statistics"""

        formatted_lines = []
        quality_stats = {'auto_accept': 0, 'manual_review': 0, 'critical_review': 0}

        for line_data in flagged_lines:
            line = f"{line_data['timestamp']} {line_data['speaker']}: {line_data['content']}"
            if line_data['visual']:
                line += f" [{line_data['visual']}]"

            line += f" [{line_data['flag']}]"
            if line_data['flag_reason']:
                line += f" *{line_data['flag_reason']}*"

            formatted_lines.append(line)

            if line_data['flag'] == "OK":
                quality_stats['auto_accept'] += 1
            elif line_data['flag'] == "REVIEW":
                quality_stats['manual_review'] += 1
            else:
                quality_stats['critical_review'] += 1

        flagged_transcript = '\n'.join(formatted_lines)

        vad_stats = self._calculate_vad_enhancement_stats(flagged_lines, vad_info_list)

        total_lines = len(flagged_lines)
        analysis_summary = {
            'total_runs': len(transcript_runs),
            'vad_enhanced': True,
            'algorithm_used': 'hybrid_bert_vad' if self.bert_available else 'basic_vad',
            'total_lines': total_lines,
            'quality_distribution': quality_stats,
            'vad_enhancement_stats': vad_stats,
            'auto_accept_rate': quality_stats['auto_accept'] / max(total_lines, 1),
            'overall_confidence': {
                'speaker': statistics.mean([float(line['confidence']['speaker']) for line in flagged_lines]) if flagged_lines else 0,
                'content': statistics.mean([float(line['confidence']['content']) for line in flagged_lines]) if flagged_lines else 0,
                'vad': statistics.mean([float(line['confidence']['vad']) for line in flagged_lines]) if flagged_lines else 0
            }
        }

        print(f"  VAD ENHANCEMENT: {vad_stats['vad_enhanced_lines']} lines enhanced by VAD")
        print(f"  RESULTS: {quality_stats['auto_accept']} auto-accept, {quality_stats['manual_review']} review, {quality_stats['critical_review']} critical")

        return {
            'flagged_transcript': flagged_transcript,
            'analysis_summary': analysis_summary,
            'detailed_analysis': flagged_lines
        }

    def _calculate_vad_enhancement_stats(self, flagged_lines: List[Dict], vad_info_list: List[Dict]) -> Dict:
        """Calculate statistics about VAD enhancement impact"""

        vad_enhanced_count = 0
        high_vad_confidence_count = 0
        vad_boost_impact = 0

        for line in flagged_lines:
            vad_conf = line['confidence']['vad']
            original_speaker = line['original_confidence']['speaker']
            enhanced_speaker = line['confidence']['speaker']

            if vad_conf > 0.6:
                high_vad_confidence_count += 1

            if enhanced_speaker > original_speaker:
                vad_enhanced_count += 1
                vad_boost_impact += enhanced_speaker - original_speaker

        avg_vad_confidence = statistics.mean([float(line['confidence']['vad']) for line in flagged_lines]) if flagged_lines else 0

        return {
            'vad_enhanced_lines': vad_enhanced_count,
            'high_confidence_vad_lines': high_vad_confidence_count,
            'avg_vad_confidence': avg_vad_confidence,
            'total_vad_boost_impact': vad_boost_impact,
            'vad_availability_rate': sum(1 for info in vad_info_list if not info.get('fallback_mode', True)) / len(vad_info_list) if vad_info_list else 0
        }

    def _parse_transcript(self, transcript: str) -> List[Dict]:
        """Parse transcript into structured format"""
        lines = []
        for line in transcript.strip().split('\n'):
            match = re.match(r'^(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)', line.strip())
            if match:
                timestamp, speaker, content = match.groups()
                lines.append({
                    'timestamp': timestamp,
                    'speaker': speaker.strip(),
                    'content': content.strip()
                })
        return lines

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert MM:SS to seconds"""
        parts = timestamp.split(':')
        return float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else 0.0

    def _find_temporal_match(self, baseline_line: Dict, other_lines: List[Dict]) -> Optional[Dict]:
        """Find temporally closest match in other transcript"""
        baseline_time = self._timestamp_to_seconds(baseline_line['timestamp'])

        best_match = None
        min_time_diff = float('inf')

        for other_line in other_lines:
            other_time = self._timestamp_to_seconds(other_line['timestamp'])
            time_diff = abs(baseline_time - other_time)

            if time_diff < min_time_diff and time_diff <= TEMPORAL_MATCH_WINDOW_SEC:
                min_time_diff = time_diff
                best_match = other_line

        return best_match

    def _enhanced_content_similarity(self, content1: str, content2: str) -> float:
        """Enhanced content similarity using BERT if available"""
        if self.bert_available and self.bert_model:
            try:
                embeddings = self.bert_model.encode([content1, content2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return (similarity + 1) / 2
            except Exception:  # V07: Fixed bare except
                pass

        return self._basic_content_similarity(content1, content2)

    def _basic_content_similarity(self, content1: str, content2: str) -> float:
        """Basic content similarity as fallback"""
        if not content1 and not content2:
            return 1.0
        if not content1 or not content2:
            return 0.0

        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _score_transcript_quality(self, transcript: str, all_transcripts: List[str]) -> float:
        """Basic quality scoring"""
        lines = transcript.strip().split('\n')
        parseable_lines = sum(1 for line in lines if re.match(r'^\d{1,2}:\d{2}\s+\w+:', line.strip()))

        if not lines:
            return 0.0

        parse_rate = parseable_lines / len(lines)
        return 100 * parse_rate


# =============================================================================
# TRANSCRIBER
# =============================================================================

class VADEnhancedTranscriber:
    """Enhanced transcriber using VAD preprocessing and context"""

    def __init__(self, api_key: str, config: TranscriptionConfigV07):
        self.config = config
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.model_name)

        script_dir = Path(__file__).parent
        prompts_file = script_dir / "prompts.json"
        self.prompt_manager = PromptManager(str(prompts_file))
        self._ensure_vad_prompts()

        self.validator = TranscriptValidator(config.min_transcript_length)

        if config.consensus_runs > 1:
            self.consensus_analyzer = VADEnhancedConsensusAnalyzer(
                config.consensus_threshold, config.vad_weight_in_consensus
            )

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _ensure_vad_prompts(self):
        """Ensure VAD-enhanced prompts exist"""
        if self.config.prompt_key == "enhanced_vad":
            try:
                self.prompt_manager.get_prompt("enhanced_vad")
            except Exception:  # V07: Fixed bare except
                enhanced_vad_prompt = """Please transcribe this classroom video with enhanced speaker diarization.

CONTEXT: Classroom video.

SPEAKERS TO IDENTIFY:
- Teacher_1: The main teacher
- Teacher_2: If a second adult is present
- Student(brief description): Identify student speakers by position and description

TRANSCRIPTION REQUIREMENTS:
1. FORMAT: MM:SS SPEAKER: content [visual actions]
2. Use [uncertain] for ambiguous speaker identification

Begin transcription:"""

                if hasattr(self.prompt_manager, 'prompts'):
                    self.prompt_manager.prompts["enhanced_vad"] = {
                        "name": "VAD-Enhanced Classroom Transcription",
                        "description": "VAD-guided transcription with hybrid speech detection",
                        "prompt": enhanced_vad_prompt
                    }

    def transcribe_chunk_with_vad_enhancement(self, chunk_info: Dict, chunk_number: int,
                                            previous_chunk_transcript: str = None, output_dir: Path = None) -> str:
        """Transcribe chunk using VAD enhancement and context"""

        uploaded_file = self._upload_video_chunk(chunk_info['file_path'])

        try:
            if self.config.consensus_runs > 1:
                return self._transcribe_with_vad_consensus(
                    uploaded_file, chunk_info, chunk_number, previous_chunk_transcript, output_dir
                )
            else:
                return self._transcribe_single_with_vad(
                    uploaded_file, chunk_info, chunk_number, previous_chunk_transcript
                )
        finally:
            self._cleanup_file(uploaded_file)

    def _transcribe_with_vad_consensus(self, uploaded_file, chunk_info: Dict,
                                     chunk_number: int, previous_chunk_transcript: str, output_dir: Path = None) -> str:
        """Multi-run transcription with VAD-enhanced consensus"""

        print(f"VAD-Enhanced Consensus: {self.config.consensus_runs} runs for chunk {chunk_number}")

        transcript_runs = []
        vad_info_list = []

        for run_num in range(1, self.config.consensus_runs + 1):
            print(f"  Run {run_num}/{self.config.consensus_runs}")

            transcript = self._transcribe_single_with_vad(
                uploaded_file, chunk_info, chunk_number, previous_chunk_transcript, run_num
            )

            is_valid, failure_reason = self.validator.is_valid_transcription(transcript)

            if is_valid:
                transcript_runs.append(transcript)
                vad_info_list.append(chunk_info.get('vad_info', {}))
                print(f"    Valid run added to consensus")

                if output_dir:
                    individual_run_file = output_dir / f"chunk_{chunk_number:02d}_run_{len(transcript_runs):02d}_transcript.txt"
                    with open(individual_run_file, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    print(f"    Saved individual run to: {individual_run_file.name}")

                print(f"\n  TRANSCRIPT RUN {len(transcript_runs)}:")
                print(f"  {'-' * 60}")
                lines = [line.strip() for line in transcript.split('\n') if line.strip()][:15]
                for line in lines:
                    print(f"  {line}")
                if len(transcript.split('\n')) > 15:
                    remaining = len([line for line in transcript.split('\n') if line.strip()]) - 15
                    print(f"  ... ({remaining} more lines)")
                print(f"  {'-' * 60}\n")
            else:
                print(f"    Invalid run: {failure_reason}")

            if run_num < self.config.consensus_runs:
                time.sleep(2)

        if not transcript_runs:
            return f"[VAD_CONSENSUS_FAILED: No valid transcripts for chunk {chunk_number}]"
        elif len(transcript_runs) == 1:
            return transcript_runs[0]
        else:
            consensus_result = self.consensus_analyzer.analyze_with_vad_confidence(
                transcript_runs, vad_info_list
            )
            return consensus_result['flagged_transcript']

    def _transcribe_single_with_vad(self, uploaded_file, chunk_info: Dict, chunk_number: int,
                                   previous_chunk_transcript: str = None, run_num: int = 1) -> str:
        """Single transcription with VAD context"""

        base_prompt = self.prompt_manager.get_prompt(self.config.prompt_key)
        vad_context = self._create_vad_context(chunk_info)
        continuity_context = self._create_continuity_context(chunk_number, previous_chunk_transcript)

        enhanced_prompt = f"{base_prompt}\n\n{vad_context}\n\n{continuity_context}"

        max_attempts = self.config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"    Retry {attempt-1}/{self.config.max_retries}")
                time.sleep(self.config.retry_delay)

            try:
                response = self.model.generate_content(
                    [uploaded_file, enhanced_prompt],
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0.1 if run_num == 1 else 0.3,
                        "max_output_tokens": 8192,
                    }
                )

                transcript = self._extract_transcript_from_response(response)

                is_valid, failure_reason = self._validate_with_vad_context(transcript, chunk_info)

                if is_valid:
                    print(f"    Valid VAD-enhanced transcription on attempt {attempt}")
                    return transcript
                else:
                    print(f"    Validation failed: {failure_reason}")
                    if attempt == max_attempts:
                        return f"[VAD_VALIDATION_FAILED: {failure_reason}]\n\n{transcript}"

            except Exception as e:
                print(f"    Transcription error: {e}")
                if attempt == max_attempts:
                    return f"[VAD_TRANSCRIPTION_ERROR: {str(e)}]"

        return f"[VAD_TRANSCRIPTION_FAILED: Max attempts reached]"

    def _create_vad_context(self, chunk_info: Dict) -> str:
        """Create VAD context for enhanced prompting"""
        vad_info = chunk_info.get('vad_info', {})

        if not self.config.enable_vad_preprocessing or vad_info.get('fallback_mode', False):
            if self.config.enable_denoising:
                return "AUDIO PREPROCESSING: Denoising applied for enhanced clarity."
            else:
                return "AUDIO PREPROCESSING: Standard processing mode."

        context_parts = ["VAD PREPROCESSING RESULTS:"]

        speech_ratio = vad_info.get('speech_ratio', 0.0)
        num_segments = vad_info.get('num_segments', 0)
        avg_confidence = vad_info.get('avg_confidence', 0.0)

        context_parts.append(f"- Speech coverage: {speech_ratio:.1%} of total duration")
        context_parts.append(f"- Detected {num_segments} speech segments")
        context_parts.append(f"- Average VAD confidence: {avg_confidence:.2f}")

        if num_segments > 10:
            context_parts.append("- High speech activity detected - expect multiple speaker turns")
        elif num_segments < 3:
            context_parts.append("- Low speech activity - focus on clear, distinct utterances")

        if avg_confidence > 0.8:
            context_parts.append("- High confidence speech detection - expect clear audio")
        elif avg_confidence < 0.5:
            context_parts.append("- Lower confidence detection - some segments may be unclear")

        context_parts.append("\nFocus transcription on VAD-identified speech regions for optimal accuracy.")

        return "\n".join(context_parts)

    def _create_continuity_context(self, chunk_number: int, previous_transcript: str) -> str:
        """Create context for chunk continuity"""
        if chunk_number == 1:
            return "SEQUENCE CONTEXT: This is the start of the video - begin transcription from the beginning."

        if not previous_transcript or previous_transcript.startswith('['):
            return f"SEQUENCE CONTEXT: Continuing from chunk {chunk_number-1} (previous context unavailable)."

        prev_lines = previous_transcript.strip().split('\n')
        context_lines = []

        for line in prev_lines[-10:]:
            line = line.strip()
            if line and ':' in line and any(c.isdigit() for c in line[:10]):
                clean_line = re.sub(r'\[OK\]|\[REVIEW\]|\[CHECK\]', '', line)
                clean_line = re.sub(r'\*[^*]+\*', '', clean_line).strip()
                context_lines.append(clean_line)

        if context_lines:
            context = '\n'.join(context_lines[-5:])
            return f"""SEQUENCE CONTEXT: Continuing from chunk {chunk_number-1}.
Recent conversation:

{context}

Continue naturally from this context, maintaining speaker consistency."""

        return f"SEQUENCE CONTEXT: Continuing from chunk {chunk_number-1} - maintain speaker consistency."

    def _validate_with_vad_context(self, transcript: str, chunk_info: Dict) -> Tuple[bool, str]:
        """Enhanced validation using VAD context"""

        is_valid, failure_reason = self.validator.is_valid_transcription(transcript)
        if not is_valid:
            return False, failure_reason

        vad_info = chunk_info.get('vad_info', {})

        if not vad_info.get('fallback_mode', True):
            transcript_lines = [line for line in transcript.split('\n') if ':' in line and line.strip()]
            expected_segments = vad_info.get('num_segments', 0)

            if expected_segments > 5 and len(transcript_lines) < 2:
                return False, f"VAD detected {expected_segments} segments but transcript has {len(transcript_lines)} lines"

            if expected_segments == 0 and len(transcript_lines) > 10:
                return False, f"VAD detected no clear speech but transcript has {len(transcript_lines)} lines"

        return True, "Valid with VAD context"

    def _extract_transcript_from_response(self, response) -> str:
        """Extract transcript from Gemini response"""
        if not response.candidates:
            raise Exception("No response candidates")

        candidate = response.candidates[0]

        if candidate.finish_reason != 1:
            finish_reasons = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
            reason = finish_reasons.get(candidate.finish_reason, f"UNKNOWN({candidate.finish_reason})")

            if candidate.content and candidate.content.parts:
                partial_text = candidate.content.parts[0].text
                return f"[PARTIAL_VAD: Generation stopped due to {reason}]\n\n{partial_text}"
            else:
                raise Exception(f"Generation stopped due to {reason}")

        if not candidate.content or not candidate.content.parts:
            raise Exception("No content parts in response")

        return candidate.content.parts[0].text

    def _upload_video_chunk(self, chunk_path: str):
        """Upload video chunk to Gemini"""
        print(f"Uploading {Path(chunk_path).name}...")

        file = genai.upload_file(chunk_path)

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
            print(f"Warning: Cleanup failed: {e}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class VideoTranscriptionPipelineV07:
    """Main V07 pipeline with all enhancements"""

    def __init__(self, api_key: str, config: TranscriptionConfigV07, skip_confirmation: bool = False):
        self.config = config
        self.chunker = VADInformedChunker(config)
        self.transcriber = VADEnhancedTranscriber(api_key, config)
        self.cost_calculator = VideoCostCalculator()
        self.skip_confirmation = skip_confirmation
        self._chunks_dir: Optional[Path] = None  # V07: Track for cleanup

    def process_video(self, video_path: str, output_dir: str = None) -> Dict:
        """Process video with V07 enhancements"""

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = video_path.parent / f"{video_path.stem}_v07_transcription_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Display processing info
        self._display_processing_info(video_path)

        # Confirm processing
        if not self.skip_confirmation:
            response = input("\nProceed with V07 transcription? (y/n): ").strip().lower()
            if response != 'y':
                print("Transcription cancelled.")
                return {}
        else:
            print("\nStarting V07 transcription (confirmation skipped)...")

        try:
            # Phase 1: VAD-informed chunking
            print(f"\n{'='*60}")
            print("PHASE 1: VAD-INFORMED CHUNKING")
            print(f"{'='*60}")

            self._chunks_dir = output_dir / "chunks"
            chunk_info_list = self.chunker.split_video_with_vad(str(video_path), str(self._chunks_dir))

            if not chunk_info_list:
                raise Exception("No chunks were created")

            # Phase 2: Enhanced transcription
            print(f"\n{'='*60}")
            print("PHASE 2: VAD-ENHANCED TRANSCRIPTION")
            print(f"{'='*60}")

            all_transcripts = []
            previous_chunk_transcript = None

            for chunk_info in chunk_info_list:
                chunk_number = chunk_info['chunk_number']
                print(f"\nProcessing chunk {chunk_number}/{len(chunk_info_list)}")
                print(f"   VAD Stats: {chunk_info['vad_info'].get('num_segments', 'N/A')} segments, "
                      f"{chunk_info['vad_info'].get('speech_ratio', 0):.1%} speech")

                transcript = self.transcriber.transcribe_chunk_with_vad_enhancement(
                    chunk_info, chunk_number, previous_chunk_transcript, output_dir
                )

                all_transcripts.append({
                    'chunk_number': chunk_number,
                    'chunk_info': chunk_info,
                    'transcript': transcript
                })

                # V07: Single file save (removed duplicate)
                chunk_file = output_dir / f"chunk_{chunk_number:02d}_transcript.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)

                if not transcript.startswith('['):
                    previous_chunk_transcript = transcript
                    print(f"   Context saved for chunk {chunk_number + 1}")
                else:
                    print(f"   Chunk {chunk_number} failed - no context for next chunk")

            # Phase 3: Combine transcripts
            print(f"\n{'='*60}")
            print("PHASE 3: TRANSCRIPT ASSEMBLY")
            print(f"{'='*60}")

            combined_transcript = self._combine_transcripts(all_transcripts)

            final_file = output_dir / f"{video_path.stem}_v07_complete_transcript.txt"
            with open(final_file, 'w', encoding='utf-8') as f:
                f.write(combined_transcript)

            # Generate summary
            summary = self._generate_summary(video_path, all_transcripts, output_dir)

            summary_file = output_dir / "v07_processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            self._display_completion_info(final_file, summary)

            return summary

        finally:
            # V07: Cleanup temp files
            if not self.config.keep_chunks and self._chunks_dir:
                self._cleanup_chunks()
            self.chunker.cleanup_temp_files()

    def _cleanup_chunks(self):
        """V07: Clean up chunk files"""
        if self._chunks_dir and self._chunks_dir.exists():
            try:
                shutil.rmtree(self._chunks_dir)
                print(f"Cleaned up chunks directory: {self._chunks_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up chunks: {e}")

    def _display_processing_info(self, video_path: Path):
        """Display comprehensive processing information"""

        duration_minutes = self.chunker._get_video_duration(str(video_path))

        print(f"\n{'='*80}")
        print("VIDEO TRANSCRIPTION PIPELINE V07")
        print("   Gemini 3 Flash + Hybrid VAD + Classroom AI")
        print(f"{'='*80}")

        print(f"Video: {video_path.name}")
        print(f"Duration: {duration_minutes:.1f} minutes")
        print(f"Model: {self.config.model_name}")
        print(f"FPS: {self.config.fps}")
        print(f"Prompt: {self.config.prompt_key}")

        print(f"\nV07 CONFIGURATION:")

        vad_status = "ENABLED" if self.config.enable_vad_preprocessing else "DISABLED"
        print(f"   Hybrid VAD Preprocessing: {vad_status}")
        if self.config.enable_vad_preprocessing:
            print(f"      Frame-level VAD: {'Available' if TRANSFORMERS_AVAILABLE else 'Unavailable'}")
            print(f"      Whisper ASR VAD: {'Available' if WHISPER_AVAILABLE else 'Unavailable'}")

        denoise_status = "ENABLED" if self.config.enable_denoising else "DISABLED"
        print(f"   Classroom Denoising: {denoise_status}")

        chunking_mode = "VAD-Informed" if self.config.vad_informed_chunking else "Traditional"
        print(f"   Chunking Strategy: {chunking_mode}")
        print(f"      Target Duration: {self.config.chunk_duration_minutes} minutes")

        if self.config.consensus_runs > 1:
            consensus_type = "VAD-Enhanced BERT" if BERT_AVAILABLE else "VAD-Enhanced Basic"
            print(f"   Consensus Analysis: {consensus_type}")
            print(f"      Runs per chunk: {self.config.consensus_runs}")
        else:
            print(f"   Consensus Analysis: Single Run Mode")

        cleanup_mode = "Keep chunks (--keep-chunks)" if self.config.keep_chunks else "Auto-delete chunks"
        print(f"   Temp File Handling: {cleanup_mode}")

        # Cost estimate
        cost_estimate = self.cost_calculator.estimate_cost(
            duration_minutes, self.config.model_name,
            self.config.chunk_duration_minutes, self.config.fps
        )

        if self.config.consensus_runs > 1:
            cost_estimate['total_cost'] *= self.config.consensus_runs

        print(f"\nESTIMATED COST: ${cost_estimate['total_cost']:.3f}")
        print(f"   Estimated tokens: {cost_estimate['total_tokens_estimated']:,}")
        print(f"   Chunks: {cost_estimate['num_chunks']}")

    def _combine_transcripts(self, all_transcripts: List[Dict]) -> str:
        """Combine transcripts with metadata"""

        combined = []

        combined.append("=" * 80)
        combined.append("COMPLETE VIDEO TRANSCRIPT - V07")
        combined.append("   Gemini 3 Flash + Hybrid VAD + Classroom AI")
        combined.append("=" * 80)
        combined.append("")

        combined.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        combined.append(f"Model: {self.config.model_name}")
        combined.append(f"FPS: {self.config.fps}")
        combined.append(f"Version: V07")
        combined.append(f"VAD Preprocessing: {'Enabled' if self.config.enable_vad_preprocessing else 'Disabled'}")
        combined.append(f"Denoising: {'Enabled' if self.config.enable_denoising else 'Disabled'}")
        combined.append(f"Consensus: {'Multi-run' if self.config.consensus_runs > 1 else 'Single-run'}")
        combined.append(f"Chunking: {'VAD-Informed' if self.config.vad_informed_chunking else 'Traditional'}")

        total_chunks = len(all_transcripts)
        vad_enhanced_chunks = sum(1 for t in all_transcripts
                                 if not t['chunk_info']['vad_info'].get('fallback_mode', True))

        combined.append(f"Total chunks: {total_chunks}")
        combined.append(f"VAD-enhanced chunks: {vad_enhanced_chunks}")
        combined.append(f"VAD enhancement rate: {vad_enhanced_chunks/total_chunks:.1%}")
        combined.append("")
        combined.append("=" * 80)
        combined.append("")

        for transcript_data in all_transcripts:
            chunk_num = transcript_data['chunk_number']
            chunk_info = transcript_data['chunk_info']
            transcript = transcript_data['transcript'].strip()

            start_minutes = chunk_info['start_time'] / 60

            combined.append(f"CHUNK {chunk_num} (Starting at {start_minutes:.1f} minutes)")

            vad_info = chunk_info['vad_info']
            if not vad_info.get('fallback_mode', True):
                combined.append(f"   VAD: {vad_info.get('num_segments', 0)} segments, "
                               f"{vad_info.get('speech_ratio', 0):.1%} speech, "
                               f"confidence {vad_info.get('avg_confidence', 0):.2f}")
            else:
                combined.append(f"   VAD: Traditional chunking mode")

            combined.append("-" * 60)

            if transcript and not transcript.startswith('['):
                adjusted_transcript = self._adjust_chunk_timestamps(transcript, start_minutes, chunk_info)
                combined.append(adjusted_transcript)
            else:
                combined.append(f"CHUNK {chunk_num} FAILED:")
                combined.append(transcript)

            combined.append("")

        return "\n".join(combined)

    def _adjust_chunk_timestamps(self, transcript: str, start_minutes: float, chunk_info: Dict) -> str:
        """Adjust timestamps for combined transcript"""

        lines = transcript.split('\n')
        adjusted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if (line.startswith('===') or line.startswith('---') or
                line.startswith('Generated:') or line.startswith('Model:')):
                continue

            # Match timestamp + speaker (including parenthetical descriptions) + content
            match = re.match(r'^(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)$', line)
            if match:
                timestamp, speaker, content = match.groups()
                speaker = speaker.strip()

                try:
                    time_parts = timestamp.split(':')
                    minutes = int(time_parts[0])
                    seconds = int(time_parts[1])

                    total_seconds = (minutes * 60 + seconds) + (start_minutes * 60)
                    new_minutes = int(total_seconds // 60)
                    new_seconds = int(total_seconds % 60)

                    new_timestamp = f"{new_minutes:02d}:{new_seconds:02d}"
                    adjusted_line = f"{new_timestamp} {speaker}: {content}"
                    adjusted_lines.append(adjusted_line)

                except (ValueError, IndexError):
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)

        return '\n'.join(adjusted_lines)

    def _generate_summary(self, video_path: Path, all_transcripts: List[Dict], output_dir: Path) -> Dict:
        """Generate processing summary"""

        vad_stats = {
            'total_chunks': len(all_transcripts),
            'vad_enhanced_chunks': 0,
            'traditional_chunks': 0,
            'total_speech_segments': 0,
            'avg_speech_ratio': 0.0,
            'avg_vad_confidence': 0.0
        }

        speech_ratios = []
        vad_confidences = []

        for transcript_data in all_transcripts:
            vad_info = transcript_data['chunk_info']['vad_info']

            if vad_info.get('fallback_mode', True):
                vad_stats['traditional_chunks'] += 1
            else:
                vad_stats['vad_enhanced_chunks'] += 1
                vad_stats['total_speech_segments'] += vad_info.get('num_segments', 0)

                speech_ratio = vad_info.get('speech_ratio', 0.0)
                confidence = vad_info.get('avg_confidence', 0.0)

                if speech_ratio > 0:
                    speech_ratios.append(speech_ratio)
                if confidence > 0:
                    vad_confidences.append(confidence)

        if speech_ratios:
            vad_stats['avg_speech_ratio'] = statistics.mean(speech_ratios)
        if vad_confidences:
            vad_stats['avg_vad_confidence'] = statistics.mean(vad_confidences)

        return {
            'video_file': str(video_path),
            'processing_date': datetime.now().isoformat(),
            'version': 'V07',
            'config': {
                'model_name': self.config.model_name,
                'fps': self.config.fps,
                'vad_preprocessing_enabled': self.config.enable_vad_preprocessing,
                'denoising_enabled': self.config.enable_denoising,
                'vad_informed_chunking': self.config.vad_informed_chunking,
                'consensus_runs': self.config.consensus_runs,
                'vad_weight_in_consensus': self.config.vad_weight_in_consensus,
                'chunk_duration_minutes': self.config.chunk_duration_minutes,
                'keep_chunks': self.config.keep_chunks
            },
            'dependency_status': {
                'librosa_available': LIBROSA_AVAILABLE,
                'noisereduce_available': NOISEREDUCE_AVAILABLE,
                'whisper_available': WHISPER_AVAILABLE,
                'transformers_available': TRANSFORMERS_AVAILABLE,
                'bert_available': BERT_AVAILABLE
            },
            'vad_statistics': vad_stats,
            'processing_results': {
                'chunks_processed': len(all_transcripts),
                'successful_chunks': sum(1 for t in all_transcripts if not t['transcript'].startswith('[')),
                'failed_chunks': sum(1 for t in all_transcripts if t['transcript'].startswith('[')),
                'vad_enhancement_rate': vad_stats['vad_enhanced_chunks'] / vad_stats['total_chunks'] if vad_stats['total_chunks'] > 0 else 0
            },
            'output_files': {
                'complete_transcript': str(output_dir / f"{video_path.stem}_v07_complete_transcript.txt"),
                'chunks_directory': str(output_dir / "chunks") if self.config.keep_chunks else "cleaned up",
                'processing_summary': str(output_dir / "v07_processing_summary.json")
            }
        }

    def _display_completion_info(self, final_file: Path, summary: Dict):
        """Display completion information"""

        print(f"\n{'='*80}")
        print("V07 TRANSCRIPTION COMPLETE!")
        print(f"{'='*80}")

        print(f"Final transcript: {final_file}")

        processing_results = summary['processing_results']
        vad_stats = summary['vad_statistics']

        print(f"\nPROCESSING STATISTICS:")
        print(f"   Total chunks: {processing_results['chunks_processed']}")
        print(f"   Successful: {processing_results['successful_chunks']}")
        print(f"   Failed: {processing_results['failed_chunks']}")
        print(f"   Success rate: {processing_results['successful_chunks']/processing_results['chunks_processed']:.1%}")

        print(f"\nVAD ENHANCEMENT RESULTS:")
        print(f"   VAD-enhanced chunks: {vad_stats['vad_enhanced_chunks']}")
        print(f"   Enhancement rate: {processing_results['vad_enhancement_rate']:.1%}")

        if vad_stats['avg_speech_ratio'] > 0:
            print(f"   Average speech ratio: {vad_stats['avg_speech_ratio']:.1%}")
            print(f"   Average VAD confidence: {vad_stats['avg_vad_confidence']:.2f}")
            print(f"   Total speech segments: {vad_stats['total_speech_segments']}")

        if not self.config.keep_chunks:
            print(f"\nChunk files cleaned up (use --keep-chunks to preserve)")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point for V07 pipeline"""

    parser = argparse.ArgumentParser(
        description="V07 Video Transcription Pipeline - Gemini 3 Flash + Hybrid VAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V07 FEATURES:
  - Gemini 3 Flash (gemini-3-flash-preview) - fast and affordable
  - Default 2 FPS for better speaker tracking
  - Hybrid VAD preprocessing (Frame-level + Whisper ASR)
  - Classroom-optimized denoising with student voice preservation
  - VAD-informed intelligent chunking at speech boundaries
  - Automatic temp file cleanup (--keep-chunks to preserve)
  - Self-contained: no external pipeline dependencies

EXAMPLES:
  Basic processing:
    python video_transcription_pipeline_v07.py video.mp4

  Higher FPS for fast discussions:
    python video_transcription_pipeline_v07.py video.mp4 --fps 5

  Keep chunks for debugging:
    python video_transcription_pipeline_v07.py video.mp4 --keep-chunks

  Multiple consensus runs:
    python video_transcription_pipeline_v07.py video.mp4 --consensus-runs 3

  Skip confirmation for automation:
    python video_transcription_pipeline_v07.py video.mp4 --no-confirm

  Cost estimate only:
    python video_transcription_pipeline_v07.py video.mp4 --estimate-only
        """
    )

    # Core arguments
    parser.add_argument("video_path", nargs='?', help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output directory")

    # V07: FPS argument with new default
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for analysis (default: 2)")

    # V07: Keep chunks flag
    parser.add_argument("--keep-chunks", action="store_true", help="Keep chunk files after processing (default: delete)")

    # VAD arguments
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD preprocessing")
    parser.add_argument("--vad-confidence", type=float, default=0.6, help="VAD confidence threshold (default: 0.6)")
    parser.add_argument("--vad-weight", type=float, default=0.3, help="VAD weight in consensus (default: 0.3)")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model for VAD (default: large-v3)")

    # Denoising arguments
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    parser.add_argument("--denoise-strength", type=float, default=0.6, help="Denoising strength (default: 0.6)")

    # Chunking arguments
    parser.add_argument("-c", "--chunk-minutes", type=float, default=3.0, help="Target chunk duration (default: 3.0)")
    parser.add_argument("--traditional-chunking", action="store_true", help="Use traditional time-based chunking")
    parser.add_argument("--min-speech-gap", type=float, default=2.0, help="Minimum speech gap for chunking (default: 2.0)")

    # Model and consensus arguments
    parser.add_argument("-m", "--model", default="gemini-3-flash-preview", help="Gemini model (default: gemini-3-flash-preview)")
    parser.add_argument("-p", "--prompt", default="enhanced_vad", help="Prompt to use")
    parser.add_argument("--consensus-runs", type=int, default=1, help="Consensus runs per chunk (default: 1)")
    parser.add_argument("--consensus-threshold", type=float, default=0.7, help="Consensus threshold (default: 0.7)")

    # Other arguments
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts (default: 3)")
    parser.add_argument("--api-key", help="Gemini API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--estimate-only", action="store_true", help="Show cost estimate only")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    # Require video path unless estimating
    if not args.video_path and not args.estimate_only:
        parser.error("video_path is required")

    # Get API key
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: Please provide API key via --api-key or GOOGLE_API_KEY environment variable")
        sys.exit(1)

    # Create V07 configuration
    config = TranscriptionConfigV07(
        chunk_duration_minutes=args.chunk_minutes,
        model_name=args.model,
        fps=args.fps,
        prompt_key=args.prompt,
        max_retries=args.max_retries,
        enable_vad_preprocessing=not args.no_vad,
        vad_confidence_threshold=args.vad_confidence,
        whisper_model=args.whisper_model,
        vad_weight_in_consensus=args.vad_weight,
        enable_denoising=not args.no_denoise,
        denoising_strength=args.denoise_strength,
        denoise_as_augmentation=True,
        vad_informed_chunking=not args.traditional_chunking,
        min_speech_gap=args.min_speech_gap,
        preserve_speech_boundaries=True,
        consensus_runs=args.consensus_runs,
        consensus_threshold=args.consensus_threshold,
        keep_chunks=args.keep_chunks
    )

    try:
        if args.estimate_only:
            if not args.video_path:
                parser.error("video_path required for cost estimation")

            print("Calculating V07 cost estimate...")

            chunker = VADInformedChunker(config)
            duration = chunker._get_video_duration(args.video_path)

            if duration == 0:
                print("Error: Could not determine video duration")
                sys.exit(1)

            estimate = VideoCostCalculator.estimate_cost(duration, args.model, args.chunk_minutes, args.fps)

            if args.consensus_runs > 1:
                estimate['total_cost'] *= args.consensus_runs

            print(f"\n{'='*60}")
            print("V07 COST ESTIMATE")
            print(f"{'='*60}")
            print(f"Video: {args.video_path}")
            print(f"Duration: {duration:.1f} minutes")
            print(f"Model: {args.model}")
            print(f"FPS: {args.fps}")
            print(f"VAD Preprocessing: {'Enabled' if not args.no_vad else 'Disabled'}")
            print(f"Denoising: {'Enabled' if not args.no_denoise else 'Disabled'}")
            print(f"Consensus runs: {args.consensus_runs}")
            print(f"Estimated chunks: {estimate['num_chunks']}")
            print(f"Estimated cost: ${estimate['total_cost']:.3f}")

        else:
            processor = VideoTranscriptionPipelineV07(api_key, config, skip_confirmation=args.no_confirm)
            result = processor.process_video(args.video_path, args.output)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nV07 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
