"""Test that _cleanup_chunks removes both chunks/ dir AND per-chunk transcript .txt files."""
import shutil
import tempfile
from pathlib import Path

import pytest

from video_transcription_pipeline_v10 import (
    TranscriptionConfigV10,
    VideoTranscriptionPipelineV10,
)


@pytest.fixture
def fake_run_dir():
    d = Path(tempfile.mkdtemp(prefix="cleanup_test_"))
    (d / "chunks").mkdir()
    (d / "chunks" / "chunk_001.mp4").write_bytes(b"fake video")
    (d / "chunks" / "chunk_002.mp4").write_bytes(b"fake video")
    # Per-chunk transcript files (PII-bearing in real runs)
    (d / "chunk_001_transcript.txt").write_text("chunk 1 text")
    (d / "chunk_002_transcript.txt").write_text("chunk 2 text")
    (d / "chunk_003_transcript.txt").write_text("chunk 3 text")
    # Keep files (must NOT be deleted)
    (d / "foo_transcript.txt").write_text("final transcript")
    (d / "foo_speakers.json").write_text("{}")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_engine():
    cfg = TranscriptionConfigV10()
    # Bypass __init__ network/client setup by directly instantiating with object.__new__
    eng = object.__new__(VideoTranscriptionPipelineV10)
    eng.config = cfg
    return eng


def test_cleanup_removes_chunks_dir_and_per_chunk_txt(fake_run_dir):
    eng = _make_engine()
    eng._cleanup_chunks(fake_run_dir / "chunks", fake_run_dir)

    assert not (fake_run_dir / "chunks").exists(), "chunks/ dir should be removed"
    assert not (fake_run_dir / "chunk_001_transcript.txt").exists()
    assert not (fake_run_dir / "chunk_002_transcript.txt").exists()
    assert not (fake_run_dir / "chunk_003_transcript.txt").exists()
    assert (fake_run_dir / "foo_transcript.txt").exists(), "final transcript must survive"
    assert (fake_run_dir / "foo_speakers.json").exists(), "speaker manifest must survive"


def test_cleanup_tolerates_missing_chunks_dir(fake_run_dir):
    shutil.rmtree(fake_run_dir / "chunks")
    eng = _make_engine()
    eng._cleanup_chunks(fake_run_dir / "chunks", fake_run_dir)
    assert not (fake_run_dir / "chunk_001_transcript.txt").exists()
