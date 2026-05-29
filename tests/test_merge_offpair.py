# tests/test_merge_offpair.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from merge_offpair_transcript import Entry, parse_transcript_text


def test_parse_speech_visual_and_skips_headers():
    text = (
        "================================================================\n"
        "COMPLETE TRANSCRIPT - V10\n"
        "Speakers: Student-Maya, Teacher-Lee\n"
        "================================================================\n"
        "\n"
        "--- CHUNK 54 (39:45 - 40:45) ---\n"
        "40:00 [Student-Omar types \"45\" into the turn block]\n"
        "40:03 Student-Maya: Spin it again.\n"
        "75:10 Teacher-Lee: Eyes up here, please.\n"
    )
    entries = parse_transcript_text(text, source="video")
    assert len(entries) == 3
    visual, maya, teacher = entries
    assert visual.kind == "visual" and visual.speaker is None and visual.time_s == 40 * 60
    assert maya.kind == "speech" and maya.speaker == "Student-Maya"
    assert maya.text == "Spin it again." and maya.time_s == 40 * 60 + 3
    assert teacher.time_s == 75 * 60 + 10  # minutes > 59 supported
    assert all(e.source == "video" for e in entries)


def test_text_similarity():
    from merge_offpair_transcript import text_similarity
    assert text_similarity("Spin it again!", "spin it again") == 1.0
    assert text_similarity("rotate it sideways", "no way") == 0.0
    mid = text_similarity("we counted to forty", "we counted to fifty")
    assert 0.4 < mid < 0.9
    assert text_similarity("", "anything") == 0.0


def test_parse_skips_malformed_and_empty():
    text = (
        "40:60 Student-Maya: hi\n"   # malformed seconds (>= 60), skipped
        "41:00 \n"                   # timestamp-only empty body, skipped
        "41:05 Student-Maya: ok\n"   # normal line, parsed
    )
    entries = parse_transcript_text(text, source="video")
    assert len(entries) == 1
    only = entries[0]
    assert only.kind == "speech"
    assert only.speaker == "Student-Maya"
    assert only.text == "ok"
    assert only.time_s == 41 * 60 + 5


def test_cross_correlate_offset_finds_excerpt():
    import numpy as np
    from merge_offpair_transcript import cross_correlate_offset
    rng = np.random.default_rng(0)
    sr = 16000
    ref = rng.standard_normal(sr * 2).astype(np.float32)  # 2 s reference
    start = int(0.4 * sr)
    sig = ref[start:start + sr // 2].copy()                # 0.5 s excerpt at t=0.4s
    offset, strength = cross_correlate_offset(ref, sig, sr)
    assert abs(offset - 0.4) < 0.01
    assert strength > 0.9


def test_fit_time_map_recovers_offset_and_drift():
    from merge_offpair_transcript import fit_time_map
    # true map: video_t = 1.002*mp3_t + 480   (drift + 8 min offset)
    pairs = [(t, 1.002 * t + 480.0) for t in (60.0, 1200.0, 2400.0, 3600.0)]
    tm = fit_time_map(pairs)
    assert abs(tm.a - 1.002) < 1e-4
    assert abs(tm.b - 480.0) < 0.5
    assert abs(tm.map(1800.0) - (1.002 * 1800.0 + 480.0)) < 0.5
    assert tm.residual < 1e-3
