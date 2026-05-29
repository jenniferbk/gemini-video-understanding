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


def test_energy_envelope_and_gating():
    import numpy as np
    from merge_offpair_transcript import rms_envelope, choose_threshold, is_close
    sr = 16000
    quiet = 0.02 * np.random.default_rng(1).standard_normal(sr * 4).astype(np.float32)
    loud = quiet.copy()
    loud[sr * 1: sr * 2] += 0.8 * np.random.default_rng(2).standard_normal(sr).astype(np.float32)
    env = rms_envelope(loud, sr, hop_s=0.5)
    assert len(env) == 8  # 4 s / 0.5 s
    thr = choose_threshold(env, k=1.0)
    assert is_close(env, 0.5, 1.4, thr) is True     # inside the loud second
    assert is_close(env, 0.5, 3.2, thr) is False    # quiet region


def test_detect_pair2_maps_speakers_to_best_match():
    from merge_offpair_transcript import Entry, TimeMap, detect_pair2
    tm = TimeMap(a=1.0, b=0.0)
    video = [
        Entry(100.0, "Student-Maya", "spin it again you got this", "speech", "video"),
        Entry(130.0, "Student-Omar", "try the other arrow key", "speech", "video"),
        Entry(160.0, "Teacher-Lee", "eyes up here everyone", "speech", "video"),
    ]
    offpair_close_overlap = [
        Entry(101.0, "Speaker-A", "spin it again you got this", "speech", "offpair"),
        Entry(131.0, "Speaker-B", "try the other arrow key", "speech", "offpair"),
    ]
    pm = detect_pair2(video, offpair_close_overlap, tm, window=8.0)
    assert pm.mapping["Speaker-A"] == "Student-Maya"
    assert pm.mapping["Speaker-B"] == "Student-Omar"
    assert pm.label_for("Speaker-A") == "Student-Maya"
    assert pm.label_for("Speaker-Z") == "Speaker-Z"  # unknown passes through
    assert pm.confidence > 0.0
