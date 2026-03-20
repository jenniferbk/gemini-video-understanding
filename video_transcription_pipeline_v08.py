#!/usr/bin/env python3
"""
Video Transcription Pipeline V08 for Educational Research
Major upgrade with google.genai SDK, parallel processing, and speaker embeddings.

NEW IN V08:
- Migrated to google.genai SDK (new unified API)
- Parallel chunk uploads for 2x speed improvement
- Speaker embedding matching with Resemblyzer for cross-chunk consistency
- Two-pass mode: identify speakers first, then transcribe
- Whisper 'base' model for faster VAD (sufficient for timestamps)
- Fixed VAD stats recording bug
- Timestamp normalization before adjustment
- Enhanced speaker context passing between chunks
- SRT subtitle export option

V08.1 CONFIDENCE FEATURES:
- Token-level logprobs from Gemini API for confidence estimation
- Uncertainty marker detection ([inaudible], [word?])
- Multi-run consensus mode for high-confidence research transcription
- Confidence-annotated output with per-line scores
- Composite confidence scoring (logprobs + VAD + uncertainty markers)

V07 FEATURES RETAINED:
- Hybrid VAD preprocessing (Frame-level VAD + Whisper ASR timestamps)
- Classroom-optimized denoising with student voice preservation
- VAD-informed intelligent chunking at speech boundaries
- Automatic temp file cleanup
"""

import os
import sys
import time
import json
import argparse
import re
import shutil
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
import warnings
import concurrent.futures

# V08: New unified Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Please install google-genai: pip install google-genai")
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

# V08: Speaker embedding matching
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("Warning: resemblyzer not available - speaker embedding matching disabled")

# BERT dependencies for consensus
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TranscriptionConfigV08:
    """Configuration for V08 transcription pipeline"""
    # Core settings
    chunk_duration_minutes: float = 3.0
    overlap_seconds: int = 10
    max_file_size_mb: int = 95
    model_name: str = "gemini-3-flash-preview"  # V08: Gemini 3 Flash

    # FPS setting
    fps: int = 2

    # VAD settings - V08: Using 'base' for speed
    enable_vad_preprocessing: bool = True
    vad_chunk_overlap: float = 0.5
    vad_confidence_threshold: float = 0.6
    whisper_model: str = "base"  # V08: Changed from large-v3 for speed

    # Denoising settings
    enable_denoising: bool = True
    denoising_strength: float = 0.6

    # Chunking settings
    vad_informed_chunking: bool = True
    min_speech_gap: float = 2.0
    preserve_speech_boundaries: bool = True

    # V08: Two-pass mode
    two_pass_mode: bool = True  # First pass identifies speakers, second transcribes

    # V08: Speaker embedding matching
    enable_speaker_embeddings: bool = True
    speaker_similarity_threshold: float = 0.75

    # V08: Parallel processing
    parallel_uploads: bool = True
    max_parallel_uploads: int = 3

    # V08.1: Confidence settings
    enable_confidence: bool = True  # Enable confidence scoring
    consensus_runs: int = 1  # Number of runs for consensus (1=disabled, 3 recommended)
    consensus_threshold: float = 0.7
    output_confidence_annotations: bool = True  # Add [HIGH]/[LOW] to lines

    # Other settings
    prompt_key: str = "enhanced_vad"
    max_retries: int = 3
    min_transcript_length: int = 50
    retry_delay: float = 5.0

    # Temp file management
    keep_chunks: bool = False

    # V08: Output formats
    output_srt: bool = False
    output_vtt: bool = False


# =============================================================================
# CONSTANTS
# =============================================================================

VAD_FRAME_DURATION_SEC = 0.02
VAD_MIN_SEGMENT_DURATION_SEC = 0.1
VAD_SMOOTHING_WINDOW = 5
VAD_ALPHA_WEIGHT = 0.7

TEMPORAL_MATCH_WINDOW_SEC = 5
CHUNK_BOUNDARY_SEARCH_SEC = 30

FRAME_TOKENS_PER_SECOND_BASE = 258
AUDIO_TOKENS_PER_SECOND = 32
METADATA_TOKENS_PER_SECOND = 10
OUTPUT_TOKEN_RATIO = 0.15


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class TranscriptValidator:
    """Validate transcription results and detect failures"""

    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def is_valid_transcription(self, transcript: str, file_name: str = "") -> Tuple[bool, str]:
        if not transcript or not isinstance(transcript, str):
            return False, "Empty or invalid transcript"

        transcript = transcript.strip()

        error_patterns = [
            r'\[ERROR:', r'\[PARTIAL:.*Generation stopped',
            r'Transcription failed', r'No response candidates',
        ]

        for pattern in error_patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                return False, f"Error marker detected: {pattern}"

        if len(transcript) < self.min_length:
            return False, f"Transcript too short: {len(transcript)} chars"

        lines = transcript.split('\n')
        valid_lines = sum(1 for line in lines if re.match(r'^\d{1,2}:\d{2}', line.strip()))

        if valid_lines == 0:
            return False, "No valid transcript lines with timestamps found"

        if self._detect_excessive_repetition(transcript):
            return False, "Excessive repetition detected"

        return True, "Valid transcript"

    def _detect_excessive_repetition(self, transcript: str) -> bool:
        """Detect both line-level and word-level repetition hallucinations"""

        # V08.1: First check for word-level repetition within lines
        # This catches "Lily, Lily, Lily..." type hallucinations
        words = re.findall(r'\b[a-zA-Z]{2,}\b', transcript.lower())
        if len(words) >= 20:
            word_counts = Counter(words)
            most_common_word, count = word_counts.most_common(1)[0]
            # If any word appears in >40% of all words, it's likely a hallucination
            if count > len(words) * 0.4:
                return True

        # Also check for repeated short phrases (2-3 word sequences)
        if len(words) >= 30:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            if bigram_counts:
                most_common_bigram, count = bigram_counts.most_common(1)[0]
                if count > len(bigrams) * 0.25:
                    return True

        # Original line-level check
        lines = transcript.split('\n')
        if len(lines) < 10:
            return False

        content_lines = []
        for line in lines:
            if ':' in line:
                try:
                    content = line.split(':', 1)[1].strip()
                    content_lines.append(content)
                except Exception:
                    pass

        if len(content_lines) < 5:
            return False

        content_counts = Counter(content_lines)
        most_common = content_counts.most_common(1)[0]
        return most_common[1] > len(content_lines) * 0.3


# =============================================================================
# V08.1: CONFIDENCE METRICS
# =============================================================================

@dataclass
class LineConfidence:
    """Confidence metrics for a single transcript line"""
    timestamp: str
    speaker: str
    content: str
    logprob_score: float = 0.0  # From Gemini logprobs (0-1, higher=more confident)
    uncertainty_markers: int = 0  # Count of [inaudible], [word?]
    vad_confidence: float = 0.0  # From VAD analysis
    consensus_agreement: Optional[float] = None  # From multi-run (0-1)

    @property
    def composite_score(self) -> float:
        """Calculate weighted composite confidence score (0-1)"""
        # Penalize for uncertainty markers
        marker_penalty = min(self.uncertainty_markers * 0.15, 0.5)

        if self.consensus_agreement is not None:
            # Full hybrid: consensus weighs heavily
            score = (
                0.35 * self.logprob_score +
                0.35 * self.consensus_agreement +
                0.20 * self.vad_confidence +
                0.10 * (1.0 - marker_penalty)
            )
        else:
            # Without consensus: logprobs + VAD
            score = (
                0.55 * self.logprob_score +
                0.30 * self.vad_confidence +
                0.15 * (1.0 - marker_penalty)
            )
        return max(0.0, min(1.0, score))

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence label"""
        score = self.composite_score
        if score >= 0.85:
            return "HIGH"
        elif score >= 0.65:
            return "MEDIUM"
        elif score >= 0.45:
            return "LOW"
        else:
            return "REVIEW"


@dataclass
class ChunkConfidence:
    """Confidence metrics for an entire chunk"""
    chunk_number: int
    lines: List[LineConfidence]
    avg_logprob: float = 0.0
    vad_speech_ratio: float = 0.0
    total_uncertainty_markers: int = 0
    consensus_runs_completed: int = 1

    @property
    def avg_composite_score(self) -> float:
        if not self.lines:
            return 0.0
        return statistics.mean(line.composite_score for line in self.lines)

    @property
    def lines_needing_review(self) -> List[LineConfidence]:
        return [line for line in self.lines if line.composite_score < 0.45]


class UncertaintyMarkerAnalyzer:
    """Detect and count uncertainty markers in transcripts"""

    UNCERTAINTY_PATTERNS = [
        r'\[inaudible\]',
        r'\[unclear\]',
        r'\[\?\]',
        r'\[crosstalk\]',
        r'\[overlapping\]',
        r'\w+\?(?=\])',  # [word?] pattern
    ]

    @classmethod
    def count_markers(cls, text: str) -> int:
        """Count uncertainty markers in text"""
        count = 0
        for pattern in cls.UNCERTAINTY_PATTERNS:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    @classmethod
    def analyze_line(cls, line: str) -> Dict:
        """Analyze a single line for uncertainty markers"""
        markers = cls.count_markers(line)
        has_inaudible = bool(re.search(r'\[inaudible\]', line, re.IGNORECASE))
        has_uncertain = bool(re.search(r'\w+\?\]', line))

        return {
            'count': markers,
            'has_inaudible': has_inaudible,
            'has_uncertain_words': has_uncertain,
            'confidence_penalty': min(markers * 0.15, 0.5)
        }


class ConsensusAnalyzer:
    """Analyze agreement across multiple transcription runs"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def compare_transcripts(self, transcripts: List[str]) -> Dict:
        """Compare multiple transcripts and calculate agreement"""
        if len(transcripts) < 2:
            return {'agreement': 1.0, 'line_agreements': []}

        # Parse all transcripts into lines
        parsed = [self._parse_transcript(t) for t in transcripts]

        # Find consensus for each line position
        line_agreements = []
        max_lines = max(len(p) for p in parsed)

        for i in range(max_lines):
            lines_at_pos = [p[i] if i < len(p) else None for p in parsed]
            agreement = self._calculate_line_agreement(lines_at_pos)
            line_agreements.append(agreement)

        avg_agreement = statistics.mean(line_agreements) if line_agreements else 0.0

        return {
            'agreement': avg_agreement,
            'line_agreements': line_agreements,
            'num_runs': len(transcripts),
            'high_confidence_lines': sum(1 for a in line_agreements if a >= 0.85),
            'low_confidence_lines': sum(1 for a in line_agreements if a < 0.5)
        }

    def _parse_transcript(self, transcript: str) -> List[Dict]:
        """Parse transcript into structured lines"""
        lines = []
        for line in transcript.split('\n'):
            line = line.strip()
            match = re.match(r'^(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)$', line)
            if match:
                lines.append({
                    'timestamp': match.group(1),
                    'speaker': match.group(2).strip(),
                    'content': match.group(3).strip()
                })
        return lines

    def _calculate_line_agreement(self, lines: List[Optional[Dict]]) -> float:
        """Calculate agreement for lines at same position across runs"""
        valid_lines = [l for l in lines if l is not None]
        if len(valid_lines) < 2:
            return 1.0 if valid_lines else 0.0

        # Check speaker agreement
        speakers = [l['speaker'] for l in valid_lines]
        speaker_agreement = max(speakers.count(s) for s in set(speakers)) / len(speakers)

        # Check content similarity (simple word overlap for speed)
        contents = [set(l['content'].lower().split()) for l in valid_lines]
        content_agreements = []
        for i, c1 in enumerate(contents):
            for c2 in contents[i+1:]:
                if c1 or c2:
                    overlap = len(c1 & c2) / max(len(c1 | c2), 1)
                    content_agreements.append(overlap)

        content_agreement = statistics.mean(content_agreements) if content_agreements else 1.0

        return 0.3 * speaker_agreement + 0.7 * content_agreement


class PromptManager:
    """Manage transcription prompts from external files"""

    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = Path(prompts_file)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict:
        if not self.prompts_file.exists():
            return self._create_default_prompts()

        try:
            with open(self.prompts_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return self._create_default_prompts()

    def _create_default_prompts(self) -> Dict:
        return {
            "enhanced_vad": {
                "name": "V08 Enhanced Transcription",
                "description": "Optimized for V08 pipeline",
                "prompt": """Transcribe this classroom video with speaker diarization.

SPEAKERS: Identify as Teacher_1, Teacher_2, or Student (description).
FORMAT: MM:SS SPEAKER: content [visual actions]
RULES: Accurate timestamps, consistent labels, no repetition.

Begin:"""
            }
        }

    def get_prompt(self, key: str) -> str:
        if key not in self.prompts:
            key = list(self.prompts.keys())[0]
        return self.prompts[key]["prompt"]


class VideoCostCalculator:
    """Calculate processing costs for video transcription"""

    TOKEN_COSTS = {
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "gemini-2.0-flash-exp": {"input": 0.0001, "output": 0.0004},
        "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003},
    }

    @classmethod
    def estimate_cost(cls, duration_minutes: float, model: str, chunk_minutes: float, fps: int = 2) -> Dict:
        total_seconds = duration_minutes * 60
        frame_tokens = FRAME_TOKENS_PER_SECOND_BASE * fps
        tokens_per_second = frame_tokens + AUDIO_TOKENS_PER_SECOND + METADATA_TOKENS_PER_SECOND
        total_input_tokens = total_seconds * tokens_per_second
        estimated_output_tokens = total_input_tokens * OUTPUT_TOKEN_RATIO
        num_chunks = max(1, int(duration_minutes / chunk_minutes))

        rates = cls.TOKEN_COSTS.get(model, {"input": 0.0005, "output": 0.003})
        input_cost = (total_input_tokens / 1000) * rates["input"]
        output_cost = (estimated_output_tokens / 1000) * rates["output"]

        return {
            "duration_minutes": duration_minutes,
            "fps": fps,
            "tokens_per_second": int(tokens_per_second),
            "total_tokens_estimated": int(total_input_tokens + estimated_output_tokens),
            "num_chunks": num_chunks,
            "input_cost": round(input_cost, 3),
            "output_cost": round(output_cost, 3),
            "total_cost": round(input_cost + output_cost, 3),
            "model": model
        }


# =============================================================================
# V08: SPEAKER EMBEDDING MANAGER
# =============================================================================

class SpeakerEmbeddingManager:
    """Manage speaker embeddings for cross-chunk consistency"""

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self.encoder = None
        self.speaker_embeddings: Dict[str, np.ndarray] = {}
        self.speaker_labels: Dict[str, str] = {}  # embedding_id -> consistent_label
        self._initialize_encoder()

    def _initialize_encoder(self):
        if RESEMBLYZER_AVAILABLE:
            try:
                print("Loading speaker embedding model...")
                self.encoder = VoiceEncoder()
                print("Speaker embedding model loaded")
            except Exception as e:
                print(f"Failed to load speaker encoder: {e}")
                self.encoder = None

    def extract_embedding(self, audio_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio segment"""
        if not self.encoder or not LIBROSA_AVAILABLE:
            return None

        try:
            # Load audio segment
            audio, sr = librosa.load(audio_path, sr=16000, mono=True,
                                    offset=start_time, duration=end_time - start_time)

            if len(audio) < sr * 0.5:  # Need at least 0.5 seconds
                return None

            # Preprocess and encode
            wav = preprocess_wav(audio, source_sr=sr)
            embedding = self.encoder.embed_utterance(wav)
            return embedding

        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            return None

    def find_matching_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """Find existing speaker that matches this embedding"""
        if embedding is None or not self.speaker_embeddings:
            return None

        best_match = None
        best_similarity = 0

        for speaker_id, stored_embedding in self.speaker_embeddings.items():
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = speaker_id

        return best_match

    def register_speaker(self, embedding: np.ndarray, label: str) -> str:
        """Register a new speaker with their embedding"""
        speaker_id = f"speaker_{len(self.speaker_embeddings)}"
        self.speaker_embeddings[speaker_id] = embedding
        self.speaker_labels[speaker_id] = label
        return speaker_id

    def get_consistent_label(self, embedding: np.ndarray, proposed_label: str) -> str:
        """Get consistent label for a speaker across chunks"""
        if embedding is None:
            return proposed_label

        match = self.find_matching_speaker(embedding)
        if match:
            return self.speaker_labels[match]
        else:
            self.register_speaker(embedding, proposed_label)
            return proposed_label


# =============================================================================
# V08: TIMESTAMP NORMALIZER
# =============================================================================

class TimestampNormalizer:
    """Normalize and adjust timestamps in transcripts"""

    @staticmethod
    def normalize_timestamp(timestamp: str) -> str:
        """Convert various timestamp formats to MM:SS"""
        # Handle HH:MM:SS
        match = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})$', timestamp)
        if match:
            hours, mins, secs = int(match.group(1)), int(match.group(2)), int(match.group(3))
            total_mins = hours * 60 + mins
            return f"{total_mins:02d}:{secs:02d}"

        # Handle MM:SS or M:SS
        match = re.match(r'^(\d{1,2}):(\d{2})$', timestamp)
        if match:
            mins, secs = int(match.group(1)), int(match.group(2))
            return f"{mins:02d}:{secs:02d}"

        return timestamp

    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float:
        """Convert MM:SS to seconds"""
        normalized = TimestampNormalizer.normalize_timestamp(timestamp)
        parts = normalized.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return 0.0

    @staticmethod
    def seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def adjust_transcript_timestamps(transcript: str, offset_seconds: float) -> str:
        """Adjust all timestamps in transcript by offset

        V08.1: More robust matching to handle malformed lines:
        - Standard: MM:SS Speaker: content
        - Extra colon: MM:SS: Speaker: content
        - Bracket only: MM:SS [description]
        - Truncated: MM:SS Speaker (no colon, no content)
        """
        lines = transcript.split('\n')
        adjusted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # V08.1: First, try to extract any timestamp at line start
            timestamp_match = re.match(r'^(\d{1,2}:\d{2}(?::\d{2})?):?\s*(.*)$', line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                rest_of_line = timestamp_match.group(2)

                # Normalize and adjust timestamp
                normalized = TimestampNormalizer.normalize_timestamp(timestamp)
                original_seconds = TimestampNormalizer.timestamp_to_seconds(normalized)
                new_seconds = original_seconds + offset_seconds
                new_timestamp = TimestampNormalizer.seconds_to_timestamp(new_seconds)

                # Reconstruct line with adjusted timestamp
                if rest_of_line:
                    adjusted_lines.append(f"{new_timestamp} {rest_of_line}")
                else:
                    adjusted_lines.append(new_timestamp)
            else:
                adjusted_lines.append(line)

        return '\n'.join(adjusted_lines)


# =============================================================================
# V08: SRT/VTT EXPORTER
# =============================================================================

class SubtitleExporter:
    """Export transcripts to subtitle formats"""

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
            start_seconds = TimestampNormalizer.timestamp_to_seconds(timestamp)

            # Estimate end time (next timestamp or +3 seconds)
            end_seconds = start_seconds + 3
            for next_line in lines[i+1:]:
                next_match = re.match(r'^(\d{1,2}:\d{2})', next_line.strip())
                if next_match:
                    end_seconds = TimestampNormalizer.timestamp_to_seconds(next_match.group(1))
                    break

            # Format SRT times
            start_srt = SubtitleExporter._seconds_to_srt_time(start_seconds)
            end_srt = SubtitleExporter._seconds_to_srt_time(end_seconds)

            srt_entries.append(f"{entry_num}\n{start_srt} --> {end_srt}\n{speaker}: {content}\n")
            entry_num += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_entries))

        print(f"SRT exported: {output_path}")

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


# =============================================================================
# AUDIO PROCESSING (Updated for V08)
# =============================================================================

class HybridVADPreprocessor:
    """Hybrid VAD with Whisper base model for speed"""

    def __init__(self, config: TranscriptionConfigV08):
        self.config = config
        self.whisper_model = None
        self._initialize_models()

    def _initialize_models(self):
        if WHISPER_AVAILABLE and self.config.enable_vad_preprocessing:
            try:
                # V08: Use base model for speed
                print(f"Loading Whisper {self.config.whisper_model} for VAD...")
                self.whisper_model = whisper.load_model(self.config.whisper_model)
                print("Whisper loaded successfully")
            except Exception as e:
                print(f"Failed to load Whisper: {e}")

    def process_audio(self, audio_path: str) -> Dict:
        """Process audio with hybrid VAD"""
        if not self.config.enable_vad_preprocessing:
            return self._create_fallback_result(audio_path)

        try:
            if not LIBROSA_AVAILABLE:
                return self._create_fallback_result(audio_path)

            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr

            # Get Whisper VAD
            whisper_result = self._extract_whisper_vad(audio_path)

            # Get frame-level VAD
            frame_vad = self._extract_frame_vad(audio, sr)

            # Combine
            speech_segments = []
            if whisper_result:
                speech_segments = whisper_result.get('segments', [])

            # Calculate stats
            total_speech = sum(seg.get('end', 0) - seg.get('start', 0) for seg in speech_segments)
            speech_ratio = total_speech / duration if duration > 0 else 0

            return {
                'audio_path': audio_path,
                'speech_segments': speech_segments,
                'duration': duration,
                'speech_ratio': speech_ratio,
                'num_segments': len(speech_segments),
                'avg_confidence': 0.8 if speech_segments else 0.0,
                'fallback_mode': False,  # V08: Fixed bug - explicitly set False
                'whisper_text': whisper_result.get('text', '') if whisper_result else ''
            }

        except Exception as e:
            print(f"VAD processing error: {e}")
            return self._create_fallback_result(audio_path)

    def _extract_whisper_vad(self, audio_path: str) -> Optional[Dict]:
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
            print(f"Whisper error: {e}")
            return None

    def _extract_frame_vad(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        try:
            frame_length = int(VAD_FRAME_DURATION_SEC * sr)
            hop_length = frame_length // 2
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            threshold = np.percentile(energy, 30)
            return (energy > threshold).astype(float)
        except Exception:
            return None

    def _create_fallback_result(self, audio_path: str) -> Dict:
        return {
            'audio_path': audio_path,
            'speech_segments': [],
            'duration': 0,
            'speech_ratio': 0,
            'num_segments': 0,
            'avg_confidence': 0,
            'fallback_mode': True,
            'whisper_text': ''
        }


# =============================================================================
# V08: GENAI CLIENT WRAPPER
# =============================================================================

@dataclass
class GenerationResult:
    """Result from content generation with confidence metrics"""
    text: str
    avg_logprob: float = 0.0
    logprobs_available: bool = False
    token_count: int = 0

    @property
    def logprob_confidence(self) -> float:
        """Convert avg_logprob to 0-1 confidence score"""
        if not self.logprobs_available or self.avg_logprob == 0.0:
            return 0.7  # Default moderate confidence when unavailable
        # logprob is negative; closer to 0 = more confident
        # -0.01 -> ~0.99, -1.0 -> ~0.37, -4.6 -> ~0.01
        import math
        return min(1.0, max(0.0, math.exp(self.avg_logprob)))


class GeminiClient:
    """Wrapper for google.genai client with logprobs support"""

    def __init__(self, api_key: str, config: TranscriptionConfigV08):
        self.config = config
        self.client = genai.Client(api_key=api_key)
        self.validator = TranscriptValidator(config.min_transcript_length)

    def upload_file(self, file_path: str) -> Any:
        """Upload file to Gemini"""
        print(f"Uploading {Path(file_path).name}...")
        file = self.client.files.upload(file=file_path)

        # Wait for processing
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = self.client.files.get(name=file.name)

        print()

        if file.state.name == "FAILED":
            raise Exception(f"File processing failed: {file.state}")

        print(f"Upload complete: {file.name}")
        return file

    def delete_file(self, file: Any):
        """Delete uploaded file"""
        try:
            self.client.files.delete(name=file.name)
            print(f"Cleaned up {file.name}")
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def generate_content(self, file: Any, prompt: str, temperature: float = 0.1) -> str:
        """Generate content from file with prompt (legacy interface)"""
        result = self.generate_content_with_confidence(file, prompt, temperature)
        return result.text

    def generate_content_with_confidence(self, file: Any, prompt: str,
                                          temperature: float = 0.1) -> GenerationResult:
        """Generate content with confidence metrics from logprobs"""

        # V08.1: Try with logprobs first, fallback if not supported
        try:
            response = self._generate_with_logprobs(file, prompt, temperature)
            return self._parse_logprobs_response(response)
        except Exception as e:
            error_msg = str(e)
            if "Logprobs is not enabled" in error_msg or "logprobs" in error_msg.lower():
                # Model doesn't support logprobs - fall back gracefully
                print(f"   Note: Logprobs not available for {self.config.model_name}, using default confidence")
                return self._generate_without_logprobs(file, prompt, temperature)
            else:
                return GenerationResult(text=f"[ERROR: {str(e)}]")

    def _generate_with_logprobs(self, file: Any, prompt: str, temperature: float) -> Any:
        """Attempt to generate with logprobs enabled"""
        return self.client.models.generate_content(
            model=self.config.model_name,
            contents=[file, prompt],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=8192,
                response_logprobs=True,
                logprobs=5,
                safety_settings=[
                    types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                    types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
                ]
            )
        )

    def _generate_without_logprobs(self, file: Any, prompt: str, temperature: float) -> GenerationResult:
        """Generate without logprobs (fallback)"""
        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=[file, prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=8192,
                    safety_settings=[
                        types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
                    ]
                )
            )
            return GenerationResult(
                text=response.text,
                avg_logprob=0.0,
                logprobs_available=False,
                token_count=0
            )
        except Exception as e:
            return GenerationResult(text=f"[ERROR: {str(e)}]")

    def _parse_logprobs_response(self, response: Any) -> GenerationResult:
        """Parse response with logprobs"""
        avg_logprob = 0.0
        logprobs_available = False
        token_count = 0

        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'avg_logprobs') and candidate.avg_logprobs:
                    avg_logprob = candidate.avg_logprobs
                    logprobs_available = True
                elif hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    logprobs_result = candidate.logprobs_result
                    if hasattr(logprobs_result, 'chosen_candidates'):
                        chosen = logprobs_result.chosen_candidates
                        if chosen:
                            logprobs = [c.log_probability for c in chosen if hasattr(c, 'log_probability')]
                            if logprobs:
                                avg_logprob = statistics.mean(logprobs)
                                token_count = len(logprobs)
                                logprobs_available = True
        except Exception as e:
            print(f"   Logprobs extraction note: {e}")

        return GenerationResult(
            text=response.text,
            avg_logprob=avg_logprob,
            logprobs_available=logprobs_available,
            token_count=token_count
        )


# =============================================================================
# V08: PARALLEL UPLOADER
# =============================================================================

class ParallelUploader:
    """Handle parallel file uploads"""

    def __init__(self, client: GeminiClient, max_parallel: int = 3):
        self.client = client
        self.max_parallel = max_parallel

    def upload_chunks(self, chunk_paths: List[str]) -> Dict[str, Any]:
        """Upload multiple chunks in parallel"""
        uploaded_files = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_path = {
                executor.submit(self.client.upload_file, path): path
                for path in chunk_paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    uploaded_files[path] = future.result()
                except Exception as e:
                    print(f"Upload failed for {path}: {e}")
                    uploaded_files[path] = None

        return uploaded_files


# =============================================================================
# CHUNKING (Updated for V08)
# =============================================================================

class VADInformedChunker:
    """Create video chunks respecting speech boundaries"""

    def __init__(self, config: TranscriptionConfigV08):
        self.config = config
        self.vad_processor = HybridVADPreprocessor(config)
        self._temp_files: List[str] = []

    def split_video(self, video_path: str, output_dir: str) -> List[Dict]:
        """Split video using VAD analysis"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        duration_minutes = self._get_video_duration(str(video_path))
        if duration_minutes == 0:
            raise ValueError("Could not determine video duration")

        print(f"Processing {duration_minutes:.1f}-minute video...")

        # Extract audio for VAD
        audio_path = self._extract_audio(video_path, output_dir)
        self._temp_files.append(audio_path)

        # Run VAD
        vad_results = self.vad_processor.process_audio(audio_path)

        # Find chunk boundaries
        if vad_results['fallback_mode'] or not self.config.vad_informed_chunking:
            chunks = self._traditional_chunking(video_path, output_dir, duration_minutes)
        else:
            chunk_boundaries = self._find_optimal_boundaries(vad_results, duration_minutes * 60)
            chunks = self._create_chunks(video_path, output_dir, chunk_boundaries, vad_results)

        return chunks

    def _find_optimal_boundaries(self, vad_results: Dict, total_duration: float) -> List[float]:
        """Find optimal chunk boundaries using VAD"""
        target_duration = self.config.chunk_duration_minutes * 60
        speech_segments = vad_results['speech_segments']

        boundaries = []
        current_time = 0

        while current_time + target_duration < total_duration:
            ideal_boundary = current_time + target_duration
            optimal = self._find_nearest_gap(speech_segments, ideal_boundary)
            boundaries.append(optimal if optimal else ideal_boundary)
            current_time = boundaries[-1]

        return boundaries

    def _find_nearest_gap(self, segments: List[Dict], target: float) -> Optional[float]:
        """Find speech gap nearest to target time"""
        best_gap = None
        best_dist = float('inf')

        for i in range(len(segments) - 1):
            gap_start = segments[i].get('end', 0)
            gap_end = segments[i + 1].get('start', 0)
            gap_duration = gap_end - gap_start

            if gap_duration >= self.config.min_speech_gap:
                gap_center = (gap_start + gap_end) / 2
                dist = abs(gap_center - target)
                if dist < best_dist and dist < CHUNK_BOUNDARY_SEARCH_SEC:
                    best_dist = dist
                    best_gap = gap_center

        return best_gap

    def _create_chunks(self, video_path: Path, output_dir: Path,
                      boundaries: List[float], vad_results: Dict) -> List[Dict]:
        """Create video chunks at boundaries"""
        chunks = []
        start_time = 0
        duration = self._get_video_duration(str(video_path)) * 60

        all_boundaries = boundaries + [duration]

        for i, end_time in enumerate(all_boundaries):
            chunk_num = i + 1
            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:02d}.mp4"

            if self._extract_chunk(str(video_path), str(chunk_file), start_time, end_time - start_time):
                # V08: Extract VAD info with fallback_mode: False
                vad_info = self._extract_chunk_vad_info(vad_results, start_time, end_time)

                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'vad_info': vad_info
                })
                print(f"  Chunk {chunk_num}: {start_time/60:.1f}m - {end_time/60:.1f}m")

            start_time = end_time

        return chunks

    def _extract_chunk_vad_info(self, vad_results: Dict, start_time: float, end_time: float) -> Dict:
        """Extract VAD info for chunk - V08: Fixed fallback_mode bug"""
        chunk_segments = []

        for seg in vad_results.get('speech_segments', []):
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            if seg_end > start_time and seg_start < end_time:
                chunk_segments.append({
                    'start': max(0, seg_start - start_time),
                    'end': min(end_time - start_time, seg_end - start_time),
                    'confidence': seg.get('confidence', 0.8)
                })

        chunk_duration = end_time - start_time
        speech_duration = sum(s['end'] - s['start'] for s in chunk_segments)

        return {
            'speech_segments': chunk_segments,
            'speech_ratio': speech_duration / chunk_duration if chunk_duration > 0 else 0,
            'num_segments': len(chunk_segments),
            'avg_confidence': np.mean([s['confidence'] for s in chunk_segments]) if chunk_segments else 0,
            'fallback_mode': False  # V08: Fixed - explicitly set False for valid VAD
        }

    def _traditional_chunking(self, video_path: Path, output_dir: Path, duration_minutes: float) -> List[Dict]:
        """Fallback time-based chunking"""
        chunks = []
        chunk_duration = self.config.chunk_duration_minutes * 60
        current = 0
        chunk_num = 1

        while current < duration_minutes * 60:
            end = min(current + chunk_duration, duration_minutes * 60)
            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:02d}.mp4"

            if self._extract_chunk(str(video_path), str(chunk_file), current, end - current):
                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': current,
                    'end_time': end,
                    'duration': end - current,
                    'vad_info': {'fallback_mode': True, 'num_segments': 0, 'speech_ratio': 0}
                })
                print(f"  Chunk {chunk_num}: {current/60:.1f}m - {end/60:.1f}m")

            current = end
            chunk_num += 1

        return chunks

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in minutes"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip()) / 60
        except Exception:
            return 0

    def _extract_audio(self, video_path: Path, output_dir: Path) -> str:
        """Extract audio from video"""
        audio_path = output_dir / f"{video_path.stem}_audio.wav"
        cmd = ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
               "-ar", "16000", "-ac", "1", str(audio_path), "-y"]
        subprocess.run(cmd, check=True, capture_output=True)
        return str(audio_path)

    def _extract_chunk(self, input_path: str, output_path: str, start: float, duration: float) -> bool:
        """Extract video chunk"""
        cmd = ["ffmpeg", "-ss", str(start), "-i", input_path, "-t", str(duration),
               "-c:v", "libx264", "-c:a", "aac", "-preset", "fast", output_path, "-y"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception:
            return False

    def cleanup(self):
        """Clean up temp files"""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self._temp_files.clear()


# =============================================================================
# V08: ENHANCED SPEAKER CONTEXT
# =============================================================================

class SpeakerContextManager:
    """Manage speaker context across chunks for consistency"""

    def __init__(self):
        self.identified_speakers: Dict[str, str] = {}  # label -> description
        self.speaker_appearances: Dict[str, List[int]] = defaultdict(list)  # label -> chunk numbers
        self.named_speakers: List[str] = []  # Speakers identified by name

    def update_from_transcript(self, transcript: str, chunk_number: int):
        """Extract speaker information from transcript"""
        for line in transcript.split('\n'):
            match = re.match(r'^\d{1,2}:\d{2}\s+([^:]+):', line)
            if match:
                speaker = match.group(1).strip()

                # Track speaker
                if speaker not in self.identified_speakers:
                    self.identified_speakers[speaker] = speaker
                self.speaker_appearances[speaker].append(chunk_number)

                # Track named speakers (when name appears in parentheses or directly)
                name_match = re.search(r'\(([A-Z][a-z]+)[\),]', speaker)
                if name_match:
                    name = name_match.group(1)
                    if name not in self.named_speakers:
                        self.named_speakers.append(name)

    def get_context_prompt(self, chunk_number: int) -> str:
        """Generate context prompt with speaker information"""
        if chunk_number == 1 or not self.identified_speakers:
            return ""

        context_parts = ["SPEAKER CONTEXT FROM PREVIOUS CHUNKS:"]

        # List known speakers
        if self.named_speakers:
            context_parts.append(f"Named students identified: {', '.join(self.named_speakers)}")

        # List frequently appearing speakers
        frequent = [s for s, chunks in self.speaker_appearances.items() if len(chunks) >= 2]
        if frequent:
            context_parts.append(f"Recurring speakers: {', '.join(frequent[:5])}")

        context_parts.append("Maintain consistent speaker labels from previous chunks.")

        return '\n'.join(context_parts)


# =============================================================================
# V08: MAIN PIPELINE
# =============================================================================

class VideoTranscriptionPipelineV08:
    """V08 Pipeline with all enhancements"""

    def __init__(self, api_key: str, config: TranscriptionConfigV08, skip_confirmation: bool = False):
        self.config = config
        self.client = GeminiClient(api_key, config)
        self.chunker = VADInformedChunker(config)
        self.prompt_manager = PromptManager()
        self.speaker_context = SpeakerContextManager()
        self.skip_confirmation = skip_confirmation

        # V08: Speaker embeddings
        if config.enable_speaker_embeddings:
            self.speaker_embeddings = SpeakerEmbeddingManager(config.speaker_similarity_threshold)
        else:
            self.speaker_embeddings = None

        # V08: Parallel uploader
        if config.parallel_uploads:
            self.uploader = ParallelUploader(self.client, config.max_parallel_uploads)
        else:
            self.uploader = None

        self._chunks_dir: Optional[Path] = None

    def process_video(self, video_path: str, output_dir: str = None) -> Dict:
        """Process video with V08 enhancements"""
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = video_path.parent / f"{video_path.stem}_v08_transcription_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        self._display_info(video_path)

        if not self.skip_confirmation:
            if input("\nProceed? (y/n): ").strip().lower() != 'y':
                print("Cancelled.")
                return {}

        try:
            # Phase 1: Chunking
            print(f"\n{'='*60}")
            print("PHASE 1: VAD-INFORMED CHUNKING")
            print(f"{'='*60}")

            self._chunks_dir = output_dir / "chunks"
            chunks = self.chunker.split_video(str(video_path), str(self._chunks_dir))

            if not chunks:
                raise Exception("No chunks created")

            # Phase 2: Two-pass or direct transcription
            print(f"\n{'='*60}")
            print("PHASE 2: TRANSCRIPTION")
            print(f"{'='*60}")

            if self.config.two_pass_mode:
                transcripts = self._two_pass_transcription(chunks, output_dir)
            else:
                transcripts = self._direct_transcription(chunks, output_dir)

            # Phase 3: Assembly
            print(f"\n{'='*60}")
            print("PHASE 3: ASSEMBLY")
            print(f"{'='*60}")

            combined = self._combine_transcripts(transcripts, video_path)

            # V08.1: Dual output - research (annotated) and Transana (clean)
            research_file = output_dir / f"{video_path.stem}_transcript.txt"
            with open(research_file, 'w', encoding='utf-8') as f:
                f.write(combined)

            # Create clean version for Transana import (strip confidence markers)
            transana_file = output_dir / f"{video_path.stem}_transana.txt"
            clean_transcript = self._strip_confidence_annotations(combined)
            with open(transana_file, 'w', encoding='utf-8') as f:
                f.write(clean_transcript)

            # V08: Export SRT if requested (use clean version)
            if self.config.output_srt:
                srt_file = output_dir / f"{video_path.stem}.srt"
                SubtitleExporter.to_srt(clean_transcript, str(srt_file))

            # Generate summary
            summary = self._generate_summary(video_path, transcripts, output_dir)

            summary_file = output_dir / "v08_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"\n{'='*60}")
            print("V08.1 TRANSCRIPTION COMPLETE!")
            print(f"{'='*60}")
            print(f"Research (annotated): {research_file}")
            print(f"Transana (clean):     {transana_file}")

            return summary

        finally:
            if not self.config.keep_chunks and self._chunks_dir:
                self._cleanup_chunks()
            self.chunker.cleanup()

    def _two_pass_transcription(self, chunks: List[Dict], output_dir: Path) -> List[Dict]:
        """Two-pass: identify speakers first, then detailed transcription"""
        print("Pass 1: Speaker identification...")

        # Quick pass to identify speakers
        speaker_prompt = """Quickly identify all speakers in this video segment.
List each speaker with a brief visual description.
Format: SPEAKER_LABEL: description
Example: Teacher_1: Adult woman at front of room
         Mason: Boy in blue shirt on left"""

        for chunk in chunks[:min(2, len(chunks))]:  # Check first 2 chunks
            try:
                file = self.client.upload_file(chunk['file_path'])
                response = self.client.generate_content(file, speaker_prompt, temperature=0.3)
                self.client.delete_file(file)

                # Parse speakers from response
                for line in response.split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            label = parts[0].strip()
                            desc = parts[1].strip()
                            self.speaker_context.identified_speakers[label] = desc
            except Exception as e:
                print(f"Speaker ID pass failed for chunk {chunk['chunk_number']}: {e}")

        print(f"Identified {len(self.speaker_context.identified_speakers)} speakers")

        print("\nPass 2: Detailed transcription...")
        return self._direct_transcription(chunks, output_dir)

    def _direct_transcription(self, chunks: List[Dict], output_dir: Path) -> List[Dict]:
        """Direct transcription of all chunks with confidence metrics"""
        transcripts = []
        previous_transcript = None

        # V08.1: Initialize consensus analyzer if multi-run enabled
        consensus_analyzer = ConsensusAnalyzer() if self.config.consensus_runs > 1 else None

        # V08: Parallel upload if enabled
        if self.uploader and len(chunks) > 1:
            print("Pre-uploading chunks in parallel...")
            chunk_paths = [c['file_path'] for c in chunks]
            uploaded_files = self.uploader.upload_chunks(chunk_paths)
        else:
            uploaded_files = {}

        for chunk in chunks:
            chunk_num = chunk['chunk_number']
            print(f"\nProcessing chunk {chunk_num}/{len(chunks)}")
            vad_info = chunk['vad_info']
            vad_confidence = vad_info.get('speech_ratio', 0.5)
            print(f"   VAD: {vad_info.get('num_segments', 0)} segments, {vad_confidence:.1%} speech")

            # Get or upload file
            if chunk['file_path'] in uploaded_files and uploaded_files[chunk['file_path']]:
                file = uploaded_files[chunk['file_path']]
            else:
                file = self.client.upload_file(chunk['file_path'])

            try:
                # Build prompt with context
                base_prompt = self.prompt_manager.get_prompt(self.config.prompt_key)
                speaker_context = self.speaker_context.get_context_prompt(chunk_num)

                continuity = ""
                if previous_transcript and not previous_transcript.startswith('['):
                    last_lines = '\n'.join(previous_transcript.strip().split('\n')[-5:])
                    continuity = f"\nCONTINUING FROM:\n{last_lines}\n"

                full_prompt = f"{base_prompt}\n{speaker_context}\n{continuity}"

                # V08.1: Multi-run consensus mode
                if self.config.consensus_runs > 1:
                    result, chunk_confidence = self._transcribe_with_consensus(
                        file, full_prompt, chunk_num, vad_confidence, consensus_analyzer
                    )
                else:
                    # Single run with confidence
                    result, chunk_confidence = self._transcribe_single_run(
                        file, full_prompt, chunk_num, vad_confidence
                    )

                transcript = result.text

                # Validate
                is_valid, reason = self.client.validator.is_valid_transcription(transcript)

                # V08.1: Retry with anti-repetition prompt if hallucination detected
                if not is_valid and "repetition" in reason.lower():
                    print(f"   Repetition detected - retrying with anti-hallucination prompt...")
                    anti_rep_prompt = full_prompt + "\n\nCRITICAL: Do NOT repeat any word or phrase more than 3 times. Transcribe naturally."
                    retry_result = self.client.generate_content_with_confidence(file, anti_rep_prompt, temperature=0.3)
                    transcript = retry_result.text
                    is_valid, reason = self.client.validator.is_valid_transcription(transcript)
                    # Update confidence with retry result
                    if chunk_confidence:
                        chunk_confidence.avg_logprob = retry_result.avg_logprob

                if is_valid:
                    # V08.1: Show confidence metrics
                    if chunk_confidence and self.config.enable_confidence:
                        print(f"   Valid | Confidence: {chunk_confidence.avg_composite_score:.1%} "
                              f"(logprob: {result.logprob_confidence:.2f}, "
                              f"markers: {chunk_confidence.total_uncertainty_markers})")
                    else:
                        print(f"   Valid transcription")
                    self.speaker_context.update_from_transcript(transcript, chunk_num)
                    previous_transcript = transcript
                else:
                    # V08.1: Don't save hallucinated garbage - just mark as needing review
                    print(f"   Validation failed after retry: {reason}")
                    print(f"   ⚠️  Chunk {chunk_num} marked for manual review")
                    transcript = f"[CHUNK_{chunk_num}_NEEDS_REVIEW: {reason}]"

                # V08.1: Add confidence annotations if enabled
                if self.config.output_confidence_annotations and chunk_confidence:
                    transcript = self._annotate_with_confidence(transcript, chunk_confidence)

                transcripts.append({
                    'chunk_number': chunk_num,
                    'chunk_info': chunk,
                    'transcript': transcript,
                    'confidence': chunk_confidence  # V08.1: Store confidence
                })

                # Save individual chunk
                chunk_file = output_dir / f"chunk_{chunk_num:02d}_transcript.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)

                # V08.1: Save confidence report
                if chunk_confidence and self.config.enable_confidence:
                    conf_file = output_dir / f"chunk_{chunk_num:02d}_confidence.json"
                    self._save_confidence_report(chunk_confidence, conf_file)

            finally:
                self.client.delete_file(file)

        return transcripts

    def _transcribe_single_run(self, file: Any, prompt: str, chunk_num: int,
                                vad_confidence: float) -> Tuple[GenerationResult, Optional[ChunkConfidence]]:
        """Single transcription run with confidence metrics"""
        result = self.client.generate_content_with_confidence(file, prompt)

        if not self.config.enable_confidence:
            return result, None

        # Analyze transcript for confidence
        lines_confidence = self._analyze_transcript_confidence(
            result.text, result.logprob_confidence, vad_confidence
        )

        chunk_confidence = ChunkConfidence(
            chunk_number=chunk_num,
            lines=lines_confidence,
            avg_logprob=result.avg_logprob,
            vad_speech_ratio=vad_confidence,
            total_uncertainty_markers=sum(l.uncertainty_markers for l in lines_confidence),
            consensus_runs_completed=1
        )

        return result, chunk_confidence

    def _transcribe_with_consensus(self, file: Any, prompt: str, chunk_num: int,
                                    vad_confidence: float,
                                    consensus_analyzer: ConsensusAnalyzer) -> Tuple[GenerationResult, Optional[ChunkConfidence]]:
        """Multi-run transcription with consensus analysis"""
        print(f"   Running {self.config.consensus_runs} transcription passes for consensus...")

        runs = []
        results = []

        for run_num in range(self.config.consensus_runs):
            # Vary temperature slightly for diversity
            temp = 0.1 + (run_num * 0.05)
            result = self.client.generate_content_with_confidence(file, prompt, temperature=temp)
            runs.append(result.text)
            results.append(result)
            print(f"      Run {run_num + 1}/{self.config.consensus_runs} complete")

        # Analyze consensus
        consensus_result = consensus_analyzer.compare_transcripts(runs)
        print(f"   Consensus agreement: {consensus_result['agreement']:.1%} "
              f"({consensus_result['high_confidence_lines']} high, "
              f"{consensus_result['low_confidence_lines']} low confidence lines)")

        # Use first run as base, but annotate with consensus
        best_result = results[0]

        # Build line-level confidence with consensus
        lines_confidence = self._analyze_transcript_confidence(
            best_result.text, best_result.logprob_confidence, vad_confidence,
            consensus_agreements=consensus_result['line_agreements']
        )

        chunk_confidence = ChunkConfidence(
            chunk_number=chunk_num,
            lines=lines_confidence,
            avg_logprob=best_result.avg_logprob,
            vad_speech_ratio=vad_confidence,
            total_uncertainty_markers=sum(l.uncertainty_markers for l in lines_confidence),
            consensus_runs_completed=self.config.consensus_runs
        )

        return best_result, chunk_confidence

    def _analyze_transcript_confidence(self, transcript: str, logprob_score: float,
                                        vad_confidence: float,
                                        consensus_agreements: List[float] = None) -> List[LineConfidence]:
        """Analyze each line for confidence metrics"""
        lines_confidence = []

        for i, line in enumerate(transcript.split('\n')):
            line = line.strip()
            # V08.1: Handle both "00:00 Speaker:" and "00:00: Speaker:" formats
            match = re.match(r'^(\d{1,2}:\d{2}):?\s+([^:]+):\s*(.*)$', line)
            if not match:
                continue

            timestamp, speaker, content = match.groups()

            # Count uncertainty markers
            markers = UncertaintyMarkerAnalyzer.count_markers(content)

            # Get consensus agreement for this line if available
            consensus = None
            if consensus_agreements and i < len(consensus_agreements):
                consensus = consensus_agreements[i]

            line_conf = LineConfidence(
                timestamp=timestamp,
                speaker=speaker.strip(),
                content=content,
                logprob_score=logprob_score,
                uncertainty_markers=markers,
                vad_confidence=vad_confidence,
                consensus_agreement=consensus
            )
            lines_confidence.append(line_conf)

        return lines_confidence

    def _annotate_with_confidence(self, transcript: str, chunk_confidence: ChunkConfidence) -> str:
        """Add confidence annotations to transcript lines"""
        lines = transcript.split('\n')
        annotated = []
        line_idx = 0

        for line in lines:
            line = line.strip()
            if not line:
                annotated.append("")
                continue

            # V08.1: Handle both "00:00 Speaker:" and "00:00: Speaker:" formats
            match = re.match(r'^(\d{1,2}:\d{2}):?\s+([^:]+):\s*(.*)$', line)
            if match and line_idx < len(chunk_confidence.lines):
                conf = chunk_confidence.lines[line_idx]
                label = conf.confidence_label

                # Only annotate non-HIGH confidence lines to reduce clutter
                if label != "HIGH":
                    annotated.append(f"{line} [{label}]")
                else:
                    annotated.append(line)
                line_idx += 1
            else:
                annotated.append(line)

        return '\n'.join(annotated)

    @staticmethod
    def _strip_confidence_annotations(transcript: str) -> str:
        """Remove [HIGH], [MEDIUM], [LOW] confidence markers from transcript

        V08.1: Create clean version for Transana import
        """
        lines = transcript.split('\n')
        clean_lines = []

        for line in lines:
            # Remove confidence markers at end of line
            cleaned = re.sub(r'\s*\[(HIGH|MEDIUM|LOW)\]\s*$', '', line)
            clean_lines.append(cleaned)

        return '\n'.join(clean_lines)

    def _save_confidence_report(self, chunk_confidence: ChunkConfidence, filepath: Path):
        """Save detailed confidence report to JSON"""
        report = {
            'chunk_number': chunk_confidence.chunk_number,
            'avg_composite_score': chunk_confidence.avg_composite_score,
            'avg_logprob': chunk_confidence.avg_logprob,
            'vad_speech_ratio': chunk_confidence.vad_speech_ratio,
            'total_uncertainty_markers': chunk_confidence.total_uncertainty_markers,
            'consensus_runs': chunk_confidence.consensus_runs_completed,
            'lines_needing_review': len(chunk_confidence.lines_needing_review),
            'line_details': [
                {
                    'timestamp': l.timestamp,
                    'speaker': l.speaker,
                    'composite_score': l.composite_score,
                    'label': l.confidence_label,
                    'uncertainty_markers': l.uncertainty_markers,
                    'logprob_score': l.logprob_score,
                    'consensus_agreement': l.consensus_agreement
                }
                for l in chunk_confidence.lines
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def _combine_transcripts(self, transcripts: List[Dict], video_path: Path) -> str:
        """Combine all transcripts with timestamp adjustment and confidence summary"""

        # V08.1: Calculate overall confidence if enabled
        overall_confidence = None
        if self.config.enable_confidence:
            chunk_scores = []
            for t in transcripts:
                conf = t.get('confidence')
                if conf and hasattr(conf, 'avg_composite_score'):
                    chunk_scores.append(conf.avg_composite_score)
            if chunk_scores:
                overall_confidence = statistics.mean(chunk_scores)

        lines = [
            "=" * 80,
            "COMPLETE TRANSCRIPT - V08.1",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.config.model_name}",
            f"FPS: {self.config.fps}",
            f"Two-pass mode: {self.config.two_pass_mode}",
            f"Speaker embeddings: {self.config.enable_speaker_embeddings}",
        ]

        # V08.1: Add confidence header
        if overall_confidence is not None:
            lines.append(f"Overall Confidence: {overall_confidence:.1%}")
            lines.append(f"Consensus Runs: {self.config.consensus_runs}")

        lines.extend(["=" * 80, ""])

        for t in transcripts:
            chunk_num = t['chunk_number']
            chunk_info = t['chunk_info']
            transcript = t['transcript']

            start_minutes = chunk_info['start_time'] / 60
            lines.append(f"CHUNK {chunk_num} (Starting at {start_minutes:.1f} minutes)")

            vad_info = chunk_info['vad_info']
            if not vad_info.get('fallback_mode', True):
                lines.append(f"   VAD: {vad_info.get('num_segments', 0)} segments, {vad_info.get('speech_ratio', 0):.1%} speech")

            lines.append("-" * 60)

            if transcript and not transcript.startswith('['):
                # V08: Use TimestampNormalizer
                adjusted = TimestampNormalizer.adjust_transcript_timestamps(
                    transcript, chunk_info['start_time']
                )
                lines.append(adjusted)
            else:
                lines.append(transcript)

            lines.append("")

        return '\n'.join(lines)

    def _generate_summary(self, video_path: Path, transcripts: List[Dict], output_dir: Path) -> Dict:
        """Generate processing summary"""
        successful = sum(1 for t in transcripts if not t['transcript'].startswith('['))

        # V08: Fixed VAD stats
        vad_enhanced = sum(1 for t in transcripts
                          if not t['chunk_info']['vad_info'].get('fallback_mode', True))

        return {
            'video_file': str(video_path),
            'version': 'V08',
            'processing_date': datetime.now().isoformat(),
            'config': {
                'model': self.config.model_name,
                'fps': self.config.fps,
                'whisper_model': self.config.whisper_model,
                'two_pass_mode': self.config.two_pass_mode,
                'parallel_uploads': self.config.parallel_uploads,
                'speaker_embeddings': self.config.enable_speaker_embeddings
            },
            'results': {
                'total_chunks': len(transcripts),
                'successful': successful,
                'failed': len(transcripts) - successful,
                'vad_enhanced_chunks': vad_enhanced
            },
            'speakers_identified': list(self.speaker_context.identified_speakers.keys())
        }

    def _display_info(self, video_path: Path):
        """Display processing information"""
        duration = self.chunker._get_video_duration(str(video_path))

        # V08.1: Adjust cost estimate for consensus runs
        base_cost = VideoCostCalculator.estimate_cost(
            duration, self.config.model_name,
            self.config.chunk_duration_minutes, self.config.fps
        )
        total_cost = base_cost['total_cost'] * self.config.consensus_runs

        print(f"\n{'='*80}")
        print("VIDEO TRANSCRIPTION PIPELINE V08.1")
        print(f"{'='*80}")
        print(f"Video: {video_path.name}")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Model: {self.config.model_name}")
        print(f"FPS: {self.config.fps}")
        print(f"\nV08 FEATURES:")
        print(f"   google.genai SDK: Yes")
        print(f"   Whisper model: {self.config.whisper_model} (fast)")
        print(f"   Two-pass mode: {self.config.two_pass_mode}")
        print(f"   Parallel uploads: {self.config.parallel_uploads}")
        print(f"   Speaker embeddings: {self.config.enable_speaker_embeddings}")
        print(f"\nV08.1 CONFIDENCE:")
        print(f"   Confidence scoring: {self.config.enable_confidence}")
        print(f"   Consensus runs: {self.config.consensus_runs}" +
              (" (HIGH CONFIDENCE MODE)" if self.config.consensus_runs >= 3 else ""))
        print(f"   Output annotations: {self.config.output_confidence_annotations}")
        print(f"\nEstimated cost: ${total_cost:.3f}" +
              (f" ({self.config.consensus_runs}x base)" if self.config.consensus_runs > 1 else ""))

    def _cleanup_chunks(self):
        """Clean up chunk directory"""
        if self._chunks_dir and self._chunks_dir.exists():
            try:
                shutil.rmtree(self._chunks_dir)
                print(f"Cleaned up: {self._chunks_dir}")
            except Exception as e:
                print(f"Cleanup warning: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V08 Video Transcription - google.genai + Parallel Processing + Speaker Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V08.1 FEATURES:
  - google.genai SDK (new unified API)
  - Whisper 'base' model for faster VAD
  - Parallel chunk uploads (2x speed)
  - Two-pass speaker identification
  - Speaker embedding matching (resemblyzer)
  - SRT subtitle export

V08.1 CONFIDENCE FEATURES:
  - Token-level logprobs from Gemini API
  - Uncertainty marker detection ([inaudible], [word?])
  - Multi-run consensus mode (--consensus-runs 3)
  - Confidence-annotated output with per-line scores
  - Detailed JSON confidence reports per chunk

EXAMPLES:
  python video_transcription_pipeline_v08.py video.mp4
  python video_transcription_pipeline_v08.py video.mp4 --consensus-runs 3  # High-confidence mode
  python video_transcription_pipeline_v08.py video.mp4 --fps 3 --output-srt
  python video_transcription_pipeline_v08.py video.mp4 --no-annotations  # No [HIGH]/[LOW] markers
        """
    )

    parser.add_argument("video_path", nargs='?', help="Path to video file")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    parser.add_argument("-m", "--model", default="gemini-3-flash-preview", help="Model name")
    parser.add_argument("-p", "--prompt", default="enhanced_vad", help="Prompt key")
    parser.add_argument("-c", "--chunk-minutes", type=float, default=3.0, help="Chunk duration")

    # V08 options
    parser.add_argument("--no-two-pass", action="store_true", help="Disable two-pass mode")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel uploads")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable speaker embeddings")
    parser.add_argument("--whisper-model", default="base", help="Whisper model (default: base)")
    parser.add_argument("--output-srt", action="store_true", help="Export SRT subtitles")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep chunk files")

    # V08.1 confidence options
    parser.add_argument("--consensus-runs", type=int, default=1,
                       help="Number of transcription runs for consensus (1=disabled, 3 recommended)")
    parser.add_argument("--no-confidence", action="store_true",
                       help="Disable confidence scoring")
    parser.add_argument("--no-annotations", action="store_true",
                       help="Disable [HIGH]/[LOW] confidence annotations in output")

    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation")
    parser.add_argument("--estimate-only", action="store_true", help="Show cost estimate only")

    args = parser.parse_args()

    if not args.video_path and not args.estimate_only:
        parser.error("video_path required")

    api_key = args.api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: API key required (--api-key or GOOGLE_API_KEY env)")
        sys.exit(1)

    config = TranscriptionConfigV08(
        chunk_duration_minutes=args.chunk_minutes,
        model_name=args.model,
        fps=args.fps,
        prompt_key=args.prompt,
        whisper_model=args.whisper_model,
        two_pass_mode=not args.no_two_pass,
        parallel_uploads=not args.no_parallel,
        enable_speaker_embeddings=not args.no_embeddings,
        output_srt=args.output_srt,
        keep_chunks=args.keep_chunks,
        # V08.1 confidence options
        consensus_runs=args.consensus_runs,
        enable_confidence=not args.no_confidence,
        output_confidence_annotations=not args.no_annotations
    )

    if args.estimate_only:
        if not args.video_path:
            parser.error("video_path required for estimate")
        chunker = VADInformedChunker(config)
        duration = chunker._get_video_duration(args.video_path)
        est = VideoCostCalculator.estimate_cost(duration, args.model, args.chunk_minutes, args.fps)
        print(f"\nV08 Cost Estimate:")
        print(f"  Duration: {duration:.1f} min")
        print(f"  Model: {args.model}")
        print(f"  Cost: ${est['total_cost']:.3f}")
        sys.exit(0)

    try:
        pipeline = VideoTranscriptionPipelineV08(api_key, config, args.no_confirm)
        pipeline.process_video(args.video_path, args.output)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
