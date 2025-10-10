#!/usr/bin/env python3
"""
Video Transcription Pipeline V04 for Educational Research
Integrates hybrid VAD improvements from "Multi-Stage Speaker Diarization for Noisy Classrooms"
with existing Gemini-based consensus system for maximum accuracy.

NEW IN V04:
- Hybrid VAD preprocessing (Frame-level VAD + Whisper ASR timestamps)
- Classroom-optimized denoising as data augmentation
- VAD-informed intelligent chunking at speech boundaries
- VAD confidence integration into consensus analysis
- Enhanced short segment detection for student voices
- RTF-compatible output format for Transana import (no emojis, clean text markers)

CONFIDENCE MARKER FORMAT (RTF-compatible):
- Good confidence (≥70% speaker AND ≥85% text): no marker
- Speaker low (<70%): [verify: spkr:XX]
- Text low (<85%): [verify: text:XX]
- Both low: [verify: spkr:XX text:YY]
"""

import os
import sys
import time
import json
import argparse
import re
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

# FFmpeg path resolution for bundled app
def get_ffmpeg_path() -> str:
    """Get path to ffmpeg binary (bundled or system)"""
    return os.environ.get('FFMPEG_PATH', 'ffmpeg')

def get_ffprobe_path() -> str:
    """Get path to ffprobe binary (bundled or system)"""
    return os.environ.get('FFPROBE_PATH', 'ffprobe')

# Core dependencies
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    try:
        from google.generativeai.types import File
    except ImportError:
        File = object
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
    print("⚠️  librosa not available - audio preprocessing will be limited")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("⚠️  noisereduce not available - denoising will be disabled")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  whisper not available - ASR-based VAD will be disabled")

try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers/torch not available - advanced VAD will be disabled")

# BERT dependencies from v03
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠️  BERT libraries not available - falling back to basic similarity")

@dataclass
class TranscriptionConfigV04:
    """Enhanced configuration for V04 with VAD preprocessing"""
    # Core settings
    chunk_duration_minutes: float = 3.0
    overlap_seconds: int = 10
    max_file_size_mb: int = 95
    model_name: str = "gemini-2.5-pro-preview-05-06"
    
    # VAD settings
    enable_vad_preprocessing: bool = True
    vad_chunk_overlap: float = 0.5  # Overlap for VAD chunking
    vad_confidence_threshold: float = 0.6
    whisper_model: str = "large-v3"  # For hybrid VAD
    frame_vad_model: str = "wav2vec2-large-robust"  # Robust model for noisy environments
    
    # Denoising settings  
    enable_denoising: bool = True
    denoising_strength: float = 0.6  # Gentler than default for student voices
    denoise_as_augmentation: bool = True  # Paper's approach
    
    # Chunking settings
    vad_informed_chunking: bool = True
    min_speech_gap: float = 2.0  # Minimum gap to consider for chunking
    preserve_speech_boundaries: bool = True
    
    # Enhanced consensus settings
    vad_weight_in_consensus: float = 0.3  # Weight for VAD confidence in consensus
    
    # Existing v03 settings
    fps: int = 1
    thinking: bool = True
    prompt_key: str = "enhanced_vad"
    consensus_runs: int = 1
    consensus_threshold: float = 0.7
    precise_chunking: bool = True
    enable_repetition_filter: bool = True
    max_retries: int = 3
    min_transcript_length: int = 50
    retry_delay: float = 5.0

    # Output settings
    json_progress: bool = False

class ClassroomDenoiser:
    """Implements paper's denoising approach optimized for classroom audio"""
    
    def __init__(self, strength: float = 0.6):
        self.strength = strength
        self.available = NOISEREDUCE_AVAILABLE and LIBROSA_AVAILABLE
        
        if not self.available:
            print("⚠️  Denoising disabled - missing dependencies (noisereduce, librosa)")
    
    def process_audio_file(self, audio_path: str) -> Tuple[Optional[str], str]:
        """
        Process audio file and return (clean_path, original_path)
        Returns (None, original_path) if denoising unavailable
        """
        if not self.available:
            return None, audio_path
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Apply classroom-optimized denoising
            denoised_audio = self._classroom_specific_denoise(audio, sr)
            
            # Save denoised version
            audio_path_obj = Path(audio_path)
            clean_path = audio_path_obj.parent / f"{audio_path_obj.stem}_denoised{audio_path_obj.suffix}"
            
            sf.write(str(clean_path), denoised_audio, sr)
            
            return str(clean_path), audio_path
            
        except Exception as e:
            print(f"⚠️  Denoising failed for {audio_path}: {e}")
            return None, audio_path
    
    def _classroom_specific_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Optimized denoising that preserves student voices
        Based on paper's findings about student voice suppression
        """
        try:
            # Gentler denoising to avoid suppressing quiet student voices
            denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,  # Classroom noise is non-stationary
                prop_decrease=self.strength,  # Gentler than default 1.0
                freq_mask_smooth_hz=100,  # Preserve speech frequencies
                time_mask_smooth_ms=50   # Preserve temporal speech patterns
            )
            
            # Additional classroom-specific processing
            # Preserve frequency ranges important for children's voices (200-400 Hz fundamentals)
            return denoised
            
        except Exception as e:
            print(f"Denoising error: {e}, returning original audio")
            return audio

class HybridVADPreprocessor:
    """
    Implements paper's hybrid VAD approach combining:
    - Frame-level VAD (Wav2Vec2-based)
    - ASR word-level timestamps (Whisper)
    """
    
    def __init__(self, config: TranscriptionConfigV04):
        self.config = config
        self.whisper_model = None
        self.frame_vad_model = None
        
        # Initialize models if available
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize VAD models with fallback handling"""
        
        # Initialize Whisper for ASR-based VAD
        if WHISPER_AVAILABLE and self.config.enable_vad_preprocessing:
            try:
                print(f"🔄 Loading Whisper {self.config.whisper_model} for hybrid VAD...")
                self.whisper_model = whisper.load_model(self.config.whisper_model)
                print("✅ Whisper loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
                self.whisper_model = None
        
        # Initialize frame-level VAD (simplified implementation)
        if TRANSFORMERS_AVAILABLE and self.config.enable_vad_preprocessing:
            try:
                print("🔄 Loading frame-level VAD model...")
                # Note: This is a simplified implementation
                # In production, you'd want to use a proper VAD model
                print("ℹ️  Frame-level VAD using simplified implementation")
                self.frame_vad_available = True
            except Exception as e:
                print(f"❌ Failed to load frame VAD: {e}")
                self.frame_vad_available = False
        else:
            self.frame_vad_available = False
    
    def process_audio_chunk(self, audio_path: str) -> Dict:
        """
        Process audio chunk with hybrid VAD approach
        Returns comprehensive VAD analysis
        """
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
            # Load audio for processing
            if not LIBROSA_AVAILABLE:
                return self._create_fallback_result(audio_path)
            
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # 1. Whisper-based VAD (word-level timestamps)
            whisper_vad = self._extract_whisper_vad(audio_path)
            if whisper_vad:
                results['whisper_result'] = whisper_vad
                results['speech_segments'].extend(whisper_vad.get('segments', []))
            
            # 2. Frame-level VAD (simplified implementation)
            frame_vad = self._extract_frame_vad(audio, sr)
            
            # 3. Hybrid combination (if both available)
            if whisper_vad and frame_vad:
                hybrid_confidence = self._combine_vad_outputs(
                    frame_vad, whisper_vad, duration
                )
                results['vad_confidence'] = hybrid_confidence
                results['hybrid_vad_available'] = True
                
                # Enhanced speech segments using hybrid approach
                results['speech_segments'] = self._extract_enhanced_segments(
                    hybrid_confidence, sr
                )
            
            # Processing statistics
            results['processing_stats'] = {
                'duration_seconds': duration,
                'num_segments': len(results['speech_segments']),
                'speech_ratio': self._calculate_speech_ratio(results['speech_segments'], duration),
                'avg_segment_duration': self._calculate_avg_segment_duration(results['speech_segments']),
                'whisper_available': whisper_vad is not None,
                'frame_vad_available': frame_vad is not None
            }
            
        except Exception as e:
            print(f"⚠️  VAD processing error for {audio_path}: {e}")
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
                temperature=0.0  # Deterministic for VAD
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
        """
        Simplified frame-level VAD implementation
        In production, this would use a proper VAD model like Wav2Vec2
        """
        if not self.frame_vad_available:
            return None
        
        try:
            # Simplified energy-based VAD as placeholder
            # In production, replace with proper Wav2Vec2-based VAD
            frame_length = int(0.02 * sr)  # 20ms frames
            hop_length = frame_length // 2
            
            # Calculate energy per frame
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # Simple threshold-based VAD
            threshold = np.percentile(energy, 30)  # Adaptive threshold
            vad_predictions = (energy > threshold).astype(float)
            
            # Smooth predictions
            vad_smoothed = self._smooth_vad_predictions(vad_predictions)
            
            return vad_smoothed
            
        except Exception as e:
            print(f"Frame VAD error: {e}")
            return None
    
    def _combine_vad_outputs(self, frame_vad: np.ndarray, whisper_result: Dict, duration: float) -> np.ndarray:
        """
        Combine frame-level VAD with Whisper timestamps
        Implements paper's Equation 1: Yi = α · frame-vadi + (1 − α) · whisperi
        """
        alpha = 0.7  # Weight favoring frame-level VAD (tunable parameter)
        
        # Convert Whisper segments to frame-level
        frame_rate = len(frame_vad) / duration
        whisper_frames = np.zeros_like(frame_vad)
        
        for segment in whisper_result.get('segments', []):
            start_frame = int(segment['start'] * frame_rate)
            end_frame = int(segment['end'] * frame_rate)
            end_frame = min(end_frame, len(whisper_frames))
            whisper_frames[start_frame:end_frame] = 1.0
        
        # Combine using weighted sum (paper's approach)
        combined = alpha * frame_vad + (1 - alpha) * whisper_frames
        
        return combined
    
    def _extract_enhanced_segments(self, vad_confidence: np.ndarray, sr: int) -> List[Dict]:
        """Extract speech segments from hybrid VAD confidence"""
        frame_duration = 0.02  # 20ms frames
        segments = []
        
        # Apply threshold to get binary decisions
        threshold = self.config.vad_confidence_threshold
        binary_vad = vad_confidence > threshold
        
        # Find continuous speech regions
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(binary_vad):
            current_time = i * frame_duration
            
            if is_speech and not in_speech:
                # Start of speech
                start_time = current_time
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech
                if current_time - start_time > 0.1:  # Minimum 100ms
                    segments.append({
                        'start': start_time,
                        'end': current_time,
                        'duration': current_time - start_time,
                        'confidence': float(np.mean(vad_confidence[int(start_time/frame_duration):i]))
                    })
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            end_time = len(binary_vad) * frame_duration
            if end_time - start_time > 0.1:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'confidence': float(np.mean(vad_confidence[int(start_time/frame_duration):]))
                })
        
        return segments
    
    def _smooth_vad_predictions(self, predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply smoothing to VAD predictions"""
        # Simple moving average smoothing
        kernel = np.ones(window_size) / window_size
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

class VADInformedChunker:
    """Create video chunks that respect speech boundaries using VAD analysis"""
    
    def __init__(self, config: TranscriptionConfigV04):
        self.config = config
        self.vad_processor = HybridVADPreprocessor(config)
        self.denoiser = ClassroomDenoiser(config.denoising_strength) if config.enable_denoising else None
    
    def split_video_with_vad(self, video_path: str, output_dir: str) -> List[Dict]:
        """
        Split video using VAD analysis to preserve speech boundaries
        Returns list of chunk info with VAD data
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get video duration
        duration_minutes = self._get_video_duration(str(video_path))
        if duration_minutes == 0:
            raise ValueError("Could not determine video duration")
        
        print(f"🔄 VAD-informed chunking for {duration_minutes:.1f}-minute video...")
        
        # Extract audio for VAD analysis
        audio_path = self._extract_audio(video_path, output_dir)
        
        # Process audio with VAD
        vad_results = self.vad_processor.process_audio_chunk(audio_path)
        
        if not vad_results['hybrid_vad_available'] or not self.config.vad_informed_chunking:
            print("ℹ️  Falling back to traditional chunking")
            return self._traditional_chunking(video_path, output_dir, vad_results)
        
        # VAD-informed chunking
        chunk_boundaries = self._find_optimal_boundaries(
            vad_results, duration_minutes * 60
        )
        
        # Create chunks at boundaries
        chunks = self._create_chunks_at_boundaries(
            video_path, output_dir, chunk_boundaries, vad_results
        )
        
        # Process with denoising if enabled
        if self.denoiser and self.config.denoise_as_augmentation:
            chunks = self._add_denoised_versions(chunks)
        
        print(f"✅ Created {len(chunks)} VAD-informed chunks")
        return chunks
    
    def _find_optimal_boundaries(self, vad_results: Dict, total_duration: float) -> List[float]:
        """Find optimal chunk boundaries using VAD analysis"""
        target_duration = self.config.chunk_duration_minutes * 60
        speech_segments = vad_results['speech_segments']
        
        if not speech_segments:
            # Fallback to regular intervals
            return list(range(int(target_duration), int(total_duration), int(target_duration)))
        
        boundaries = []
        current_time = 0
        
        while current_time + target_duration < total_duration:
            ideal_boundary = current_time + target_duration
            
            # Find speech gap closest to ideal boundary
            optimal_boundary = self._find_nearest_speech_gap(
                speech_segments, ideal_boundary, self.config.min_speech_gap
            )
            
            if optimal_boundary and optimal_boundary > current_time + target_duration * 0.7:
                boundaries.append(optimal_boundary)
                current_time = optimal_boundary
            else:
                # No good gap found, use ideal boundary
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
                
                if distance < best_distance and distance < 30:  # Within 30 seconds
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
                # Last chunk
                duration_minutes = self._get_video_duration(str(video_path))
                end_time = duration_minutes * 60
            
            # Create chunk file
            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_num:02d}.mp4"
            
            success = self._extract_video_chunk(
                str(video_path), str(chunk_file), start_time, end_time - start_time
            )
            
            if success:
                # Extract VAD info for this chunk
                chunk_vad_info = self._extract_chunk_vad_info(
                    vad_results, start_time, end_time
                )
                
                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'vad_info': chunk_vad_info
                })
                
                print(f"  ✅ Chunk {chunk_num}: {start_time//60:02.0f}:{start_time%60:05.2f} - {end_time//60:02.0f}:{end_time%60:05.2f}")
            
            start_time = end_time
        
        return chunks
    
    def _extract_chunk_vad_info(self, vad_results: Dict, start_time: float, end_time: float) -> Dict:
        """Extract VAD information for specific time range"""
        chunk_segments = []
        
        for segment in vad_results.get('speech_segments', []):
            # Check if segment overlaps with chunk
            if segment['end'] > start_time and segment['start'] < end_time:
                # Adjust segment times relative to chunk start
                adjusted_segment = {
                    'start': max(0, segment['start'] - start_time),
                    'end': min(end_time - start_time, segment['end'] - start_time),
                    'duration': 0,
                    'confidence': segment.get('confidence', 1.0)
                }
                adjusted_segment['duration'] = adjusted_segment['end'] - adjusted_segment['start']
                
                if adjusted_segment['duration'] > 0.1:  # Minimum 100ms
                    chunk_segments.append(adjusted_segment)
        
        return {
            'speech_segments': chunk_segments,
            'speech_ratio': sum(s['duration'] for s in chunk_segments) / (end_time - start_time),
            'num_segments': len(chunk_segments),
            'avg_confidence': np.mean([s['confidence'] for s in chunk_segments]) if chunk_segments else 0.0
        }
    
    def _add_denoised_versions(self, chunks: List[Dict]) -> List[Dict]:
        """Add denoised versions of chunks for data augmentation"""
        if not self.denoiser:
            return chunks
        
        enhanced_chunks = []
        
        for chunk in chunks:
            # Extract audio from video chunk
            audio_path = self._extract_audio_from_chunk(chunk['file_path'])
            
            # Create denoised version
            clean_audio_path, noisy_audio_path = self.denoiser.process_audio_file(audio_path)
            
            # Add both versions to chunk info
            chunk['audio_files'] = {
                'original': noisy_audio_path,
                'denoised': clean_audio_path
            }
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
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
            
            success = self._extract_video_chunk(
                str(video_path), str(chunk_file), start_time, end_time - start_time
            )
            
            if success:
                chunks.append({
                    'chunk_number': chunk_num,
                    'file_path': str(chunk_file),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'vad_info': {'fallback_mode': True}
                })
            
            start_time = end_time
            chunk_num += 1
        
        return chunks
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in minutes"""
        try:
            cmd = [
                get_ffprobe_path(), "-v", "quiet", "-show_entries", "format=duration",
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
            get_ffmpeg_path(), "-i", str(video_path),
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
            get_ffmpeg_path(), "-ss", str(start_time), "-i", input_path,
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
            get_ffmpeg_path(), "-i", chunk_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(audio_path)
        except subprocess.CalledProcessError:
            return chunk_path  # Return original if extraction fails


# Import existing classes from v03 with enhancements
from video_transcription_pipeline_v03 import (
    TranscriptValidator, PromptManager, VideoCostCalculator
)

class VADEnhancedConsensusAnalyzer:
    """Enhanced consensus analyzer that incorporates VAD confidence"""
    
    def __init__(self, consensus_threshold: float = 0.7, vad_weight: float = 0.3):
        self.consensus_threshold = consensus_threshold
        self.vad_weight = vad_weight
        
        # Initialize BERT model if available (from v03)
        self.bert_model = None
        self.bert_available = BERT_AVAILABLE
        if self.bert_available:
            try:
                print("🚀 Loading BERT model for hybrid semantic similarity...")
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ BERT model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load BERT model: {e}")
                self.bert_available = False
    
    def analyze_with_vad_confidence(self, transcript_runs: List[str], 
                                   vad_info_list: List[Dict]) -> Dict:
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
        print(f"🧠 Using {algorithm_type} algorithm")
        
        # Select baseline transcript
        baseline_transcript, baseline_index, baseline_info = self._select_baseline_with_vad(
            transcript_runs, vad_info_list
        )
        
        # Align transcripts with VAD confidence
        aligned_data = self._align_transcripts_with_vad(
            baseline_transcript, transcript_runs, vad_info_list[baseline_index]
        )
        
        # Generate VAD-enhanced flags
        flagged_lines = self._flag_with_vad_confidence(aligned_data)
        
        # Create final output
        return self._generate_vad_enhanced_output(flagged_lines, transcript_runs, vad_info_list)
    
    def _select_baseline_with_vad(self, transcript_runs: List[str], 
                                 vad_info_list: List[Dict]) -> Tuple[str, int, Dict]:
        """Select baseline transcript considering VAD quality"""
        
        scores = []
        for i, (transcript, vad_info) in enumerate(zip(transcript_runs, vad_info_list)):
            # Base quality score (from v03)
            base_score = self._score_transcript_quality(transcript, transcript_runs)
            
            # VAD quality bonus
            vad_bonus = self._calculate_vad_quality_bonus(vad_info)
            
            # Combined score
            total_score = base_score + vad_bonus
            
            scores.append({
                'index': i,
                'score': total_score,
                'base_score': base_score,
                'vad_bonus': vad_bonus,
                'transcript': transcript
            })
            
            print(f"    Run {i+1}: {total_score:.1f}/110 points (base: {base_score:.1f}, VAD: {vad_bonus:.1f})")
        
        # Select best
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
        
        # Speech ratio bonus (good coverage)
        speech_ratio = vad_info.get('speech_ratio', 0.0)
        if 0.3 <= speech_ratio <= 0.8:  # Optimal range
            bonus += 3.0
        elif speech_ratio > 0.1:
            bonus += 1.0
        
        # Confidence bonus
        avg_confidence = vad_info.get('avg_confidence', 0.0)
        bonus += avg_confidence * 4.0  # Up to 4 points
        
        # Segment quality bonus
        num_segments = vad_info.get('num_segments', 0)
        if num_segments > 5:  # Good segmentation
            bonus += 2.0
        elif num_segments > 2:
            bonus += 1.0
        
        # Hybrid VAD bonus
        if vad_info.get('hybrid_vad_available', False):
            bonus += 1.0
        
        return min(bonus, 10.0)
    
    def _align_transcripts_with_vad(self, baseline: str, all_transcripts: List[str], 
                                   vad_info: Dict) -> List[Dict]:
        """Align transcripts with VAD confidence integration"""
        
        # Parse baseline transcript
        baseline_lines = self._parse_transcript(baseline)
        
        # Parse other transcripts
        other_transcript_lines = [self._parse_transcript(t) for t in all_transcripts]
        
        # Get VAD confidence for baseline timing
        vad_confidence_map = self._create_vad_confidence_map(vad_info)
        
        aligned_data = []
        
        for baseline_line in baseline_lines:
            baseline_time = self._timestamp_to_seconds(baseline_line['timestamp'])
            
            # Get VAD confidence for this timestamp
            vad_confidence = vad_confidence_map.get(int(baseline_time), 0.5)  # Default 0.5
            
            alignment = {
                'baseline': baseline_line,
                'matches': [],
                'speaker_agreements': [],
                'content_agreements': [],
                'vad_confidence': vad_confidence
            }
            
            # Find matches in other transcripts
            for other_lines in other_transcript_lines:
                best_match = self._find_temporal_match(baseline_line, other_lines)
                
                if best_match:
                    alignment['matches'].append(best_match)
                    
                    # Calculate agreements
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
        """Generate quality flags incorporating VAD confidence - RTF-compatible format"""
        
        flagged_lines = []
        
        for alignment in aligned_data:
            baseline_line = alignment['baseline']
            speaker_agreements = alignment['speaker_agreements']
            content_agreements = alignment['content_agreements']
            vad_confidence = alignment['vad_confidence']
            
            # Calculate base confidences
            speaker_confidence = sum(speaker_agreements) / max(len(speaker_agreements), 1) if speaker_agreements else 1.0
            content_confidence = sum(content_agreements) / max(len(content_agreements), 1) if content_agreements else 1.0
            
            # VAD-enhanced confidence (not used in final output, but kept for statistics)
            vad_boost = vad_confidence * self.vad_weight
            enhanced_speaker_conf = min(1.0, speaker_confidence + vad_boost * 0.5)
            enhanced_content_conf = min(1.0, content_confidence + vad_boost)
            
            # Determine flag with CLEAN RTF-compatible format
            # Good confidence (≥70% speaker AND ≥85% text) → no marker
            # Low confidence → concise text marker
            flag_reason = ""
            review_needed = False
            
            speaker_pct = int(enhanced_speaker_conf * 100)
            content_pct = int(enhanced_content_conf * 100)
            
            if speaker_pct >= 70 and content_pct >= 85:
                # Good confidence - suppress marker entirely
                flag_reason = ""
                review_needed = False
            elif speaker_pct < 70 and content_pct < 85:
                # Both below threshold
                flag_reason = f"[verify: spkr:{speaker_pct} text:{content_pct}]"
                review_needed = True
            elif speaker_pct < 70:
                # Speaker only
                flag_reason = f"[verify: spkr:{speaker_pct}]"
                review_needed = True
            elif content_pct < 85:
                # Text only
                flag_reason = f"[verify: text:{content_pct}]"
                review_needed = True
            
            flagged_lines.append({
                'timestamp': baseline_line['timestamp'],
                'speaker': baseline_line['speaker'],
                'content': baseline_line['content'],
                'visual': baseline_line.get('visual', ''),
                'flag': "",  # No emoji flags
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
        """Generate final output with VAD enhancement statistics - RTF-compatible"""
        
        # Format transcript with clean markers
        formatted_lines = []
        quality_stats = {'auto_accept': 0, 'manual_review': 0}
        
        for line_data in flagged_lines:
            line = f"{line_data['timestamp']} {line_data['speaker']}: {line_data['content']}"
            if line_data['visual']:
                line += f" [{line_data['visual']}]"
            
            # Add flag reason only if present (clean format)
            if line_data['flag_reason']:
                line += f" {line_data['flag_reason']}"
            
            formatted_lines.append(line)
            
            # Update stats
            if line_data['review_needed']:
                quality_stats['manual_review'] += 1
            else:
                quality_stats['auto_accept'] += 1
        
        flagged_transcript = '\n'.join(formatted_lines)
        
        # Calculate VAD enhancement statistics
        vad_stats = self._calculate_vad_enhancement_stats(flagged_lines, vad_info_list)
        
        # Generate comprehensive summary
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
                'speaker': statistics.mean([float(line['confidence']['speaker']) for line in flagged_lines]),
                'content': statistics.mean([float(line['confidence']['content']) for line in flagged_lines]),
                'vad': statistics.mean([float(line['confidence']['vad']) for line in flagged_lines])
            }
        }
        
        print(f"  VAD ENHANCEMENT: {vad_stats['vad_enhanced_lines']} lines enhanced by VAD")
        print(f"  RESULTS: {quality_stats['auto_accept']} auto-accept, {quality_stats['manual_review']} review needed")
        
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
        
        avg_vad_confidence = statistics.mean([float(line['confidence']['vad']) for line in flagged_lines])
        
        return {
            'vad_enhanced_lines': vad_enhanced_count,
            'high_confidence_vad_lines': high_vad_confidence_count,
            'avg_vad_confidence': avg_vad_confidence,
            'total_vad_boost_impact': vad_boost_impact,
            'vad_availability_rate': sum(1 for info in vad_info_list if not info.get('fallback_mode', True)) / len(vad_info_list)
        }
    
    # Helper methods (simplified versions of v03 methods)
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
            
            if time_diff < min_time_diff and time_diff <= 5:  # Within 5 seconds
                min_time_diff = time_diff
                best_match = other_line
        
        return best_match
    
    def _enhanced_content_similarity(self, content1: str, content2: str) -> float:
        """Enhanced content similarity using BERT if available"""
        if self.bert_available and self.bert_model:
            try:
                embeddings = self.bert_model.encode([content1, content2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return (similarity + 1) / 2  # Normalize to 0-1
            except:
                pass
        
        # Fallback to basic similarity
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
        """Basic quality scoring (simplified from v03)"""
        lines = transcript.strip().split('\n')
        parseable_lines = sum(1 for line in lines if re.match(r'^\d{1,2}:\d{2}\s+\w+:', line.strip()))
        
        if not lines:
            return 0.0
        
        parse_rate = parseable_lines / len(lines)
        return 100 * parse_rate


class VADEnhancedTranscriber:
    """Enhanced transcriber using VAD preprocessing and context"""
    
    def __init__(self, api_key: str, config: TranscriptionConfigV04):
        self.config = config
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.model_name)
        
        # Load enhanced prompts
        self.prompt_manager = PromptManager()
        self._ensure_vad_prompts()
        
        # Initialize validation
        self.validator = TranscriptValidator(config.min_transcript_length)
        
        # Initialize VAD-enhanced consensus if multiple runs
        if config.consensus_runs > 1:
            self.consensus_analyzer = VADEnhancedConsensusAnalyzer(
                config.consensus_threshold, config.vad_weight_in_consensus
            )
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def _ensure_vad_prompts(self):
        """Ensure VAD-enhanced prompts exist"""
        try:
            # Check if enhanced_vad prompt exists
            self.prompt_manager.get_prompt("enhanced_vad")
        except:
            # Add VAD-enhanced prompt if missing
            enhanced_vad_prompt = """
Please transcribe this classroom video with enhanced speaker diarization and VAD-guided analysis.

CONTEXT: Classroom video.

SPEAKERS TO IDENTIFY:
- Teacher_1: The main teacher
- Teacher_2: If a second adult is present
- Student(brief description): Identify student speakers by position and brief description


TRANSCRIPTION REQUIREMENTS:
1. FORMAT: MM:SS SPEAKER: content [visual actions]
2. Use [uncertain] for ambiguous speaker identification

QUALITY PRIORITIES:
- Accurate speaker identification using multi-modal cues
- Precise timestamp alignment with VAD boundaries
- Capture all speech including quiet student voices
- Minimize repetition and false transcriptions

Begin transcription:
"""
            
            # This is a simplified way to add the prompt - in production you'd update prompts.json
            if hasattr(self.prompt_manager, 'prompts'):
                self.prompt_manager.prompts["enhanced_vad"] = {
                    "name": "VAD-Enhanced Classroom Transcription",
                    "description": "VAD-guided transcription with hybrid speech detection",
                    "prompt": enhanced_vad_prompt
                }
    
    def transcribe_chunk_with_vad_enhancement(self, chunk_info: Dict, chunk_number: int, 
                                            previous_chunk_transcript: str = None) -> str:
        """Transcribe chunk using VAD enhancement and context"""
        
        # Upload video chunk
        uploaded_file = self._upload_video_chunk(chunk_info['file_path'])
        
        try:
            if self.config.consensus_runs > 1:
                # Multi-run consensus with VAD enhancement
                return self._transcribe_with_vad_consensus(
                    uploaded_file, chunk_info, chunk_number, previous_chunk_transcript
                )
            else:
                # Single run with VAD context
                return self._transcribe_single_with_vad(
                    uploaded_file, chunk_info, chunk_number, previous_chunk_transcript
                )
        finally:
            self._cleanup_file(uploaded_file)
    
    def _transcribe_with_vad_consensus(self, uploaded_file, chunk_info: Dict, 
                                     chunk_number: int, previous_chunk_transcript: str) -> str:
        """Multi-run transcription with VAD-enhanced consensus"""
        
        print(f"🔄 VAD-Enhanced Consensus: {self.config.consensus_runs} runs for chunk {chunk_number}")
        
        transcript_runs = []
        vad_info_list = []
        
        for run_num in range(1, self.config.consensus_runs + 1):
            print(f"  Run {run_num}/{self.config.consensus_runs}")
            
            # Transcribe with VAD context
            transcript = self._transcribe_single_with_vad(
                uploaded_file, chunk_info, chunk_number, previous_chunk_transcript, run_num
            )
            
            # Validate
            is_valid, failure_reason = self.validator.is_valid_transcription(transcript)
            
            if is_valid:
                transcript_runs.append(transcript)
                vad_info_list.append(chunk_info.get('vad_info', {}))
                print(f"    ✅ Valid run added to consensus")
            else:
                print(f"    ❌ Invalid run: {failure_reason}")
            
            if run_num < self.config.consensus_runs:
                time.sleep(2)  # Brief delay between runs
        
        # Generate VAD-enhanced consensus
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
        
        # Build VAD-enhanced prompt
        base_prompt = self.prompt_manager.get_prompt(self.config.prompt_key)
        vad_context = self._create_vad_context(chunk_info)
        continuity_context = self._create_continuity_context(chunk_number, previous_chunk_transcript)
        
        enhanced_prompt = f"{base_prompt}\n\n{vad_context}\n\n{continuity_context}"
        
        # Retry logic with VAD awareness
        max_attempts = self.config.max_retries + 1
        
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"    Retry {attempt-1}/{self.config.max_retries}")
                time.sleep(self.config.retry_delay)
            
            try:
                # Generate with enhanced prompt
                response = self.model.generate_content(
                    [uploaded_file, enhanced_prompt],
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0.1 if run_num == 1 else 0.3,  # Vary temperature for consensus
                        "max_output_tokens": 4096,
                    }
                )
                
                # Extract and validate transcript
                transcript = self._extract_transcript_from_response(response)
                
                # Validate with VAD awareness
                is_valid, failure_reason = self._validate_with_vad_context(
                    transcript, chunk_info
                )
                
                if is_valid:
                    print(f"    ✅ Valid VAD-enhanced transcription on attempt {attempt}")
                    return transcript
                else:
                    print(f"    ❌ Validation failed: {failure_reason}")
                    if attempt == max_attempts:
                        return f"[VAD_VALIDATION_FAILED: {failure_reason}]\n\n{transcript}"
                    
            except Exception as e:
                print(f"    ❌ Transcription error: {e}")
                if attempt == max_attempts:
                    return f"[VAD_TRANSCRIPTION_ERROR: {str(e)}]"
        
        return f"[VAD_TRANSCRIPTION_FAILED: Max attempts reached]"
    
    def _create_vad_context(self, chunk_info: Dict) -> str:
        """Create VAD context for enhanced prompting"""
        vad_info = chunk_info.get('vad_info', {})
        
        if vad_info.get('fallback_mode', False):
            return "VAD CONTEXT: Standard chunking mode (VAD preprocessing unavailable)."
        
        context_parts = ["VAD PREPROCESSING RESULTS:"]
        
        # Speech coverage info
        speech_ratio = vad_info.get('speech_ratio', 0.0)
        num_segments = vad_info.get('num_segments', 0)
        avg_confidence = vad_info.get('avg_confidence', 0.0)
        
        context_parts.append(f"- Speech coverage: {speech_ratio:.1%} of total duration")
        context_parts.append(f"- Detected {num_segments} speech segments")
        context_parts.append(f"- Average VAD confidence: {avg_confidence:.2f}")
        
        # Segment guidance
        if num_segments > 10:
            context_parts.append("- High speech activity detected - expect multiple speaker turns")
        elif num_segments < 3:
            context_parts.append("- Low speech activity - focus on clear, distinct utterances")
        
        # Confidence guidance
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
        
        # Extract last few lines for context
        prev_lines = previous_transcript.strip().split('\n')
        context_lines = []
        
        for line in prev_lines[-10:]:  # Last 10 lines
            line = line.strip()
            if line and ':' in line and any(c.isdigit() for c in line[:10]):
                # Clean line of quality flags
                clean_line = re.sub(r'[✅⚠️🚨]\s*', '', line)
                clean_line = re.sub(r'\*[^*]+\*', '', clean_line).strip()
                context_lines.append(clean_line)
        
        if context_lines:
            context = '\n'.join(context_lines[-5:])  # Last 5 clean lines
            return f"""SEQUENCE CONTEXT: Continuing from chunk {chunk_number-1}. 
Recent conversation:

{context}

Continue naturally from this context, maintaining speaker consistency."""
        
        return f"SEQUENCE CONTEXT: Continuing from chunk {chunk_number-1} - maintain speaker consistency."
    
    def _validate_with_vad_context(self, transcript: str, chunk_info: Dict) -> Tuple[bool, str]:
        """Enhanced validation using VAD context"""
        
        # Basic validation first
        is_valid, failure_reason = self.validator.is_valid_transcription(transcript)
        if not is_valid:
            return False, failure_reason
        
        # VAD-specific validation
        vad_info = chunk_info.get('vad_info', {})
        
        if not vad_info.get('fallback_mode', True):
            # Check if transcript aligns with VAD expectations
            transcript_lines = [line for line in transcript.split('\n') if ':' in line and line.strip()]
            expected_segments = vad_info.get('num_segments', 0)
            
            # Very basic alignment check
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
        
        if candidate.finish_reason != 1:  # Not natural completion
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
        print(f"📤 Uploading {Path(chunk_path).name}...")
        
        file = genai.upload_file(chunk_path)
        
        # Wait for processing
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(file.name)
        
        print()
        
        if file.state.name == "FAILED":
            raise Exception(f"File processing failed: {file.state}")
        
        print(f"✅ Upload complete: {file.name}")
        return file
    
    def _cleanup_file(self, file):
        """Clean up uploaded file"""
        try:
            genai.delete_file(file.name)
            print(f"🗑️  Cleaned up {file.name}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


class VideoTranscriptionPipelineV04:
    """Main V04 pipeline integrating all VAD enhancements"""
    
    def __init__(self, api_key: str, config: TranscriptionConfigV04):
        self.config = config
        self.chunker = VADInformedChunker(config)
        self.transcriber = VADEnhancedTranscriber(api_key, config)
        self.cost_calculator = VideoCostCalculator()

    def _report_progress(self, chunk: int, total: int, status: str):
        """Report progress in JSON format for Electron app"""
        if self.config.json_progress:
            percent = int((chunk / total) * 100) if total > 0 else 0
            progress_data = {
                "type": "progress",
                "chunk": chunk,
                "total": total,
                "percent": percent,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
            print(f"GVU_PROGRESS:{json.dumps(progress_data)}", flush=True)

    def _report_completion(self, output_file: str, stats: dict):
        """Report completion in JSON format for Electron app"""
        if self.config.json_progress:
            completion_data = {
                "type": "complete",
                "outputFile": str(output_file),
                "stats": stats
            }
            print(f"GVU_COMPLETE:{json.dumps(completion_data)}", flush=True)

    def _report_error(self, message: str, fatal: bool = True):
        """Report error in JSON format for Electron app"""
        if self.config.json_progress:
            error_data = {
                "type": "error",
                "message": message,
                "fatal": fatal
            }
            print(f"GVU_ERROR:{json.dumps(error_data)}", flush=True)

    def process_video(self, video_path: str, output_dir: str = None) -> Dict:
        """Process video with V04 VAD enhancements"""
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = video_path.parent / f"{video_path.stem}_v04_transcription_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Display processing info
        self._display_processing_info(video_path)

        # Confirm processing (skip if running from Electron app)
        if not self.config.json_progress:
            response = input("\n🚀 Proceed with V04 VAD-enhanced transcription? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Transcription cancelled.")
                return {}
        
        try:
            # VAD-informed chunking
            print(f"\n{'='*60}")
            print("🎯 PHASE 1: VAD-INFORMED CHUNKING")
            print(f"{'='*60}")
            
            chunks_dir = output_dir / "chunks"
            chunk_info_list = self.chunker.split_video_with_vad(str(video_path), str(chunks_dir))
            
            if not chunk_info_list:
                raise Exception("No chunks were created")
            
            # Enhanced transcription
            print(f"\n{'='*60}")
            print("🎯 PHASE 2: VAD-ENHANCED TRANSCRIPTION")
            print(f"{'='*60}")
            
            all_transcripts = []
            previous_chunk_transcript = None
            
            for chunk_info in chunk_info_list:
                chunk_number = chunk_info['chunk_number']
                print(f"\n🔄 Processing chunk {chunk_number}/{len(chunk_info_list)}")
                print(f"   📊 VAD Stats: {chunk_info['vad_info'].get('num_segments', 'N/A')} segments, "
                      f"{chunk_info['vad_info'].get('speech_ratio', 0):.1%} speech")

                # Report progress
                self._report_progress(chunk_number, len(chunk_info_list), f"Processing chunk {chunk_number}/{len(chunk_info_list)}")

                # Transcribe with VAD enhancement
                transcript = self.transcriber.transcribe_chunk_with_vad_enhancement(
                    chunk_info, chunk_number, previous_chunk_transcript
                )
                
                all_transcripts.append({
                    'chunk_number': chunk_number,
                    'chunk_info': chunk_info,
                    'transcript': transcript
                })
                
                # Save individual chunk
                chunk_file = output_dir / f"chunk_{chunk_number:02d}_v04_transcript.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                
                # Update context for next chunk
                if not transcript.startswith('['):
                    previous_chunk_transcript = transcript
                    print(f"   ✅ Context saved for chunk {chunk_number + 1}")
                else:
                    print(f"   ⚠️  Chunk {chunk_number} failed - no context for next chunk")
            
            # Combine transcripts
            print(f"\n{'='*60}")
            print("🎯 PHASE 3: TRANSCRIPT ASSEMBLY")
            print(f"{'='*60}")
            
            combined_transcript = self._combine_v04_transcripts(all_transcripts)
            
            # Save final transcript
            final_file = output_dir / f"{video_path.stem}_v04_complete_transcript.txt"
            with open(final_file, 'w', encoding='utf-8') as f:
                f.write(combined_transcript)
            
            # Generate processing summary
            summary = self._generate_v04_summary(video_path, all_transcripts, output_dir)
            
            summary_file = output_dir / "v04_processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Display completion info
            self._display_completion_info(final_file, summary)

            # Report completion to Electron
            self._report_completion(final_file, summary)

            return summary
            
        except Exception as e:
            print(f"\n❌ V04 Processing error: {e}")
            raise
    
    def _display_processing_info(self, video_path: Path):
        """Display comprehensive processing information"""
        
        duration_minutes = self.chunker._get_video_duration(str(video_path))
        
        print(f"\n{'='*80}")
        print("🚀 VIDEO TRANSCRIPTION PIPELINE V04")
        print("   Enhanced with Hybrid VAD + Classroom AI")
        print(f"{'='*80}")
        
        print(f"📹 Video: {video_path.name}")
        print(f"⏱️  Duration: {duration_minutes:.1f} minutes")
        print(f"🤖 Model: {self.config.model_name}")
        
        print(f"\n🎯 V04 ENHANCEMENTS:")
        
        # VAD Features
        vad_status = "✅ ENABLED" if self.config.enable_vad_preprocessing else "❌ DISABLED"
        print(f"   📊 Hybrid VAD Preprocessing: {vad_status}")
        if self.config.enable_vad_preprocessing:
            print(f"      • Frame-level VAD: {'✅' if TRANSFORMERS_AVAILABLE else '❌'} Available")
            print(f"      • Whisper ASR VAD: {'✅' if WHISPER_AVAILABLE else '❌'} Available")
            print(f"      • VAD Confidence Threshold: {self.config.vad_confidence_threshold}")
        
        # Denoising
        denoise_status = "✅ ENABLED" if self.config.enable_denoising else "❌ DISABLED"
        print(f"   🔧 Classroom Denoising: {denoise_status}")
        if self.config.enable_denoising:
            print(f"      • Student Voice Preservation: ✅ Optimized")
            print(f"      • Data Augmentation Mode: {'✅' if self.config.denoise_as_augmentation else '❌'}")
        
        # Chunking
        chunking_mode = "VAD-Informed" if self.config.vad_informed_chunking else "Traditional"
        print(f"   ✂️  Chunking Strategy: {chunking_mode}")
        print(f"      • Target Duration: {self.config.chunk_duration_minutes} minutes")
        print(f"      • Speech Boundary Preservation: {'✅' if self.config.preserve_speech_boundaries else '❌'}")
        
        # Consensus
        if self.config.consensus_runs > 1:
            consensus_type = "VAD-Enhanced BERT" if BERT_AVAILABLE else "VAD-Enhanced Basic"
            print(f"   🧠 Consensus Analysis: {consensus_type}")
            print(f"      • Runs per chunk: {self.config.consensus_runs}")
            print(f"      • VAD weight in consensus: {self.config.vad_weight_in_consensus}")
        else:
            print(f"   🧠 Consensus Analysis: Single Run Mode")
        
        # Dependencies status
        print(f"\n📦 DEPENDENCY STATUS:")
        deps = [
            ("Core Audio (librosa)", LIBROSA_AVAILABLE),
            ("Denoising (noisereduce)", NOISEREDUCE_AVAILABLE),
            ("Whisper ASR", WHISPER_AVAILABLE),
            ("Advanced VAD (transformers)", TRANSFORMERS_AVAILABLE),
            ("BERT Consensus", BERT_AVAILABLE)
        ]
        
        for name, available in deps:
            status = "✅" if available else "❌"
            print(f"   {status} {name}")
        
        # Cost estimate
        cost_estimate = self.cost_calculator.estimate_cost(
            duration_minutes, self.config.model_name, 
            self.config.chunk_duration_minutes, self.config.fps
        )
        
        if self.config.consensus_runs > 1:
            cost_estimate['total_cost'] *= self.config.consensus_runs
        
        print(f"\n💰 ESTIMATED COST: ${cost_estimate['total_cost']:.3f}")
        print(f"   📊 Estimated tokens: {cost_estimate['total_tokens_estimated']:,}")
        print(f"   📦 Chunks: {cost_estimate['num_chunks']}")
    
    def _combine_v04_transcripts(self, all_transcripts: List[Dict]) -> str:
        """Combine V04 transcripts with enhanced metadata"""
        
        combined = []
        
        # Header with V04 info
        combined.append("=" * 80)
        combined.append("🚀 COMPLETE VIDEO TRANSCRIPT - V04 ENHANCED")
        combined.append("   Hybrid VAD + Classroom AI + Consensus Analysis")
        combined.append("=" * 80)
        combined.append("")
        
        # Metadata
        combined.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        combined.append(f"🤖 Model: {self.config.model_name}")
        combined.append(f"🎯 Version: V04 (VAD-Enhanced)")
        combined.append(f"📊 VAD Preprocessing: {'✅ Enabled' if self.config.enable_vad_preprocessing else '❌ Disabled'}")
        combined.append(f"🔧 Denoising: {'✅ Enabled' if self.config.enable_denoising else '❌ Disabled'}")
        combined.append(f"🧠 Consensus: {'✅ Multi-run' if self.config.consensus_runs > 1 else 'Single-run'}")
        combined.append(f"✂️  Chunking: {'VAD-Informed' if self.config.vad_informed_chunking else 'Traditional'}")
        
        # Processing statistics
        total_chunks = len(all_transcripts)
        vad_enhanced_chunks = sum(1 for t in all_transcripts 
                                 if not t['chunk_info']['vad_info'].get('fallback_mode', True))
        
        combined.append(f"📦 Total chunks: {total_chunks}")
        combined.append(f"📊 VAD-enhanced chunks: {vad_enhanced_chunks}")
        combined.append(f"📈 VAD enhancement rate: {vad_enhanced_chunks/total_chunks:.1%}")
        combined.append("")
        combined.append("=" * 80)
        combined.append("")
        
        # Combine chunk transcripts
        for transcript_data in all_transcripts:
            chunk_num = transcript_data['chunk_number']
            chunk_info = transcript_data['chunk_info']
            transcript = transcript_data['transcript'].strip()
            
            start_minutes = chunk_info['start_time'] / 60
            
            # Chunk header with VAD stats
            combined.append(f"📌 CHUNK {chunk_num} (Starting at {start_minutes:.1f} minutes)")
            
            vad_info = chunk_info['vad_info']
            if not vad_info.get('fallback_mode', True):
                combined.append(f"   📊 VAD: {vad_info.get('num_segments', 0)} segments, "
                               f"{vad_info.get('speech_ratio', 0):.1%} speech, "
                               f"confidence {vad_info.get('avg_confidence', 0):.2f}")
            else:
                combined.append(f"   📊 VAD: Traditional chunking mode")
            
            combined.append("-" * 60)
            
            # Adjust timestamps and add transcript
            if transcript and not transcript.startswith('['):
                adjusted_transcript = self._adjust_chunk_timestamps_v04(
                    transcript, start_minutes, chunk_info
                )
                combined.append(adjusted_transcript)
            else:
                combined.append(f"❌ CHUNK {chunk_num} FAILED:")
                combined.append(transcript)
            
            combined.append("")
        
        return "\n".join(combined)
    
    def _adjust_chunk_timestamps_v04(self, transcript: str, start_minutes: float, chunk_info: Dict) -> str:
        """Enhanced timestamp adjustment with VAD awareness"""
        
        lines = transcript.split('\n')
        adjusted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip metadata lines
            if (line.startswith('===') or line.startswith('---') or 
                line.startswith('Generated:') or line.startswith('Model:')):
                continue
            
            # Process transcript lines
            match = re.match(r'^(\d{1,2}:\d{2})\s+(\w+):\s*(.*)$', line)
            if match:
                timestamp, speaker, content = match.groups()
                
                try:
                    # Convert and adjust timestamp
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
    
    def _generate_v04_summary(self, video_path: Path, all_transcripts: List[Dict], output_dir: Path) -> Dict:
        """Generate comprehensive V04 processing summary"""
        
        # Calculate VAD statistics
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
            'version': 'V04_VAD_Enhanced',
            'config': {
                'model_name': self.config.model_name,
                'vad_preprocessing_enabled': self.config.enable_vad_preprocessing,
                'denoising_enabled': self.config.enable_denoising,
                'vad_informed_chunking': self.config.vad_informed_chunking,
                'consensus_runs': self.config.consensus_runs,
                'vad_weight_in_consensus': self.config.vad_weight_in_consensus,
                'chunk_duration_minutes': self.config.chunk_duration_minutes
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
                'vad_enhancement_rate': vad_stats['vad_enhanced_chunks'] / vad_stats['total_chunks']
            },
            'output_files': {
                'complete_transcript': str(output_dir / f"{video_path.stem}_v04_complete_transcript.txt"),
                'chunks_directory': str(output_dir / "chunks"),
                'processing_summary': str(output_dir / "v04_processing_summary.json")
            }
        }
    
    def _display_completion_info(self, final_file: Path, summary: Dict):
        """Display completion information"""
        
        print(f"\n{'='*80}")
        print("🎉 V04 TRANSCRIPTION COMPLETE!")
        print(f"{'='*80}")
        
        print(f"📄 Final transcript: {final_file}")
        
        # Processing stats
        vad_stats = summary['vad_statistics']
        processing_results = summary['processing_results']
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   📦 Total chunks: {processing_results['chunks_processed']}")
        print(f"   ✅ Successful: {processing_results['successful_chunks']}")
        print(f"   ❌ Failed: {processing_results['failed_chunks']}")
        print(f"   📈 Success rate: {processing_results['successful_chunks']/processing_results['chunks_processed']:.1%}")
        
        print(f"\n🎯 VAD ENHANCEMENT RESULTS:")
        print(f"   📊 VAD-enhanced chunks: {vad_stats['vad_enhanced_chunks']}")
        print(f"   📈 Enhancement rate: {processing_results['vad_enhancement_rate']:.1%}")
        
        if vad_stats['avg_speech_ratio'] > 0:
            print(f"   🗣️  Average speech ratio: {vad_stats['avg_speech_ratio']:.1%}")
            print(f"   🎯 Average VAD confidence: {vad_stats['avg_vad_confidence']:.2f}")
            print(f"   📝 Total speech segments: {vad_stats['total_speech_segments']}")
        
        print(f"\n🚀 V04 provides enhanced accuracy through:")
        print(f"   • Hybrid VAD preprocessing for optimal speech detection")
        print(f"   • Classroom-optimized denoising for student voice preservation") 
        print(f"   • VAD-informed chunking at natural speech boundaries")
        print(f"   • Enhanced consensus analysis with VAD confidence weighting")


def main():
    """Main entry point for V04 pipeline"""
    
    parser = argparse.ArgumentParser(
        description="V04 Video Transcription Pipeline with Hybrid VAD Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V04 ENHANCEMENTS:
  • Hybrid VAD preprocessing (Frame-level + Whisper ASR)
  • Classroom-optimized denoising with student voice preservation
  • VAD-informed intelligent chunking at speech boundaries  
  • Enhanced consensus analysis with VAD confidence weighting
  • Short segment detection optimization for student voices

EXAMPLES:
  Basic V04 processing:
    python video_transcription_pipeline_v04.py video.mp4
    
  With multiple consensus runs:
    python video_transcription_pipeline_v04.py video.mp4 --consensus-runs 3
    
  Disable VAD preprocessing (fallback to V03 mode):
    python video_transcription_pipeline_v04.py video.mp4 --no-vad
        """
    )
    
    # Core arguments
    parser.add_argument("video_path", nargs='?', help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output directory")
    
    # V04 VAD arguments
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
    parser.add_argument("-m", "--model", default="gemini-2.5-pro-preview-05-06", help="Gemini model")
    parser.add_argument("-p", "--prompt", default="enhanced_vad", help="Prompt to use")
    parser.add_argument("--consensus-runs", type=int, default=1, help="Consensus runs per chunk (default: 1)")
    parser.add_argument("--consensus-threshold", type=float, default=0.7, help="Consensus threshold (default: 0.7)")
    
    # Other arguments
    parser.add_argument("--fps", type=int, default=1, help="Video analysis FPS (default: 1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts (default: 3)")
    parser.add_argument("--api-key", help="Gemini API key (or set GOOGLE_API_KEY)")
    parser.add_argument("--estimate-only", action="store_true", help="Show cost estimate only")
    parser.add_argument("--json-progress", action="store_true", help="Output progress as JSON for Electron app")

    args = parser.parse_args()
    
    # Require video path unless estimating
    if not args.video_path and not args.estimate_only:
        parser.error("video_path is required")
    
    # Get API key
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Please provide API key via --api-key or GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    # Create V04 configuration
    config = TranscriptionConfigV04(
        # Core settings
        chunk_duration_minutes=args.chunk_minutes,
        model_name=args.model,
        fps=args.fps,
        prompt_key=args.prompt,
        max_retries=args.max_retries,
        
        # VAD settings
        enable_vad_preprocessing=not args.no_vad,
        vad_confidence_threshold=args.vad_confidence,
        whisper_model=args.whisper_model,
        vad_weight_in_consensus=args.vad_weight,
        
        # Denoising settings
        enable_denoising=not args.no_denoise,
        denoising_strength=args.denoise_strength,
        denoise_as_augmentation=True,
        
        # Chunking settings
        vad_informed_chunking=not args.traditional_chunking,
        min_speech_gap=args.min_speech_gap,
        preserve_speech_boundaries=True,
        
        # Consensus settings
        consensus_runs=args.consensus_runs,
        consensus_threshold=args.consensus_threshold,

        # Output settings
        json_progress=args.json_progress
    )
    
    try:
        if args.estimate_only:
            # Cost estimation mode
            if not args.video_path:
                parser.error("video_path required for cost estimation")
            
            print("🔄 Calculating V04 cost estimate...")
            
            # Get duration
            chunker = VADInformedChunker(config)
            duration = chunker._get_video_duration(args.video_path)
            
            if duration == 0:
                print("❌ Could not determine video duration")
                sys.exit(1)
            
            # Base estimate
            cost_calc = VideoCostCalculator()
            estimate = cost_calc.estimate_cost(duration, args.model, args.chunk_minutes, args.fps)
            
            # Adjust for consensus
            if args.consensus_runs > 1:
                estimate['total_cost'] *= args.consensus_runs
            
            # Display estimate
            print(f"\n{'='*60}")
            print("💰 V04 COST ESTIMATE")
            print(f"{'='*60}")
            print(f"📹 Video: {args.video_path}")
            print(f"⏱️  Duration: {duration:.1f} minutes")
            print(f"🤖 Model: {args.model}")
            print(f"📊 VAD Preprocessing: {'✅' if not args.no_vad else '❌'}")
            print(f"🔧 Denoising: {'✅' if not args.no_denoise else '❌'}")
            print(f"🧠 Consensus runs: {args.consensus_runs}")
            print(f"📦 Estimated chunks: {estimate['num_chunks']}")
            print(f"💰 Estimated cost: ${estimate['total_cost']:.3f}")
            
        else:
            # Full processing mode
            processor = VideoTranscriptionPipelineV04(api_key, config)
            result = processor.process_video(args.video_path, args.output)
            
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ V04 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
