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


def test_merge_gated_gapfill():
    import numpy as np
    from merge_offpair_transcript import Entry, TimeMap, PairMap, merge
    tm = TimeMap(a=1.0, b=0.0)
    video = [
        Entry(100.0, "Student-Maya", "spin it again", "speech", "video"),      # reliable
        Entry(200.0, "Student-Maya", "[inaudible]", "speech", "video"),        # gap (inaudible)
        Entry(300.0, "Teacher-Lee", "eyes up here", "speech", "video"),            # teacher only -> gap for students
    ]
    offpair = [
        Entry(100.0, "Speaker-B", "spin it again", "speech", "offpair"),        # redundant -> drop
        Entry(200.0, "Speaker-B", "maybe its a hexagon", "speech", "offpair"),  # fills inaudible -> insert
        Entry(300.0, "Speaker-B", "try the green block", "speech", "offpair"),  # student gap under teacher -> insert
        Entry(400.0, "Speaker-B", "faint bleed words", "speech", "offpair"),   # faint -> drop
    ]
    # energy: close at 100/200/300, faint at 400 (hop_s=0.5 -> index = t/0.5)
    env = np.zeros(900)
    for t in (100.0, 200.0, 300.0):
        env[int(t / 0.5)] = 1.0
    threshold = 0.5
    pm = PairMap(mapping={"Speaker-B": "Student-Omar"}, confidence=0.9)
    merged = merge(video, offpair, tm, env, 0.5, threshold, pm, window=8.0)

    texts = [(e.time_s, e.speaker, e.text, e.source) for e in merged]
    # all video lines preserved
    assert (100.0, "Student-Maya", "spin it again", "video") in texts
    assert (300.0, "Teacher-Lee", "eyes up here", "video") in texts
    # inaudible-gap filled, relabeled, sourced offpair
    assert (200.0, "Student-Omar", "maybe its a hexagon", "offpair") in texts
    # student gap under teacher filled
    assert (300.0, "Student-Omar", "try the green block", "offpair") in texts
    # redundant + faint NOT inserted
    assert all(e.text != "spin it again" or e.source == "video" for e in merged)
    assert all(e.text != "faint bleed words" for e in merged)
    # sorted by time
    assert [e.time_s for e in merged] == sorted(e.time_s for e in merged)


def test_format_transcript_and_audit():
    from merge_offpair_transcript import Entry, TimeMap, PairMap, format_transcript, build_audit
    entries = [
        Entry(63.0, "Student-Maya", "spin it again", "speech", "video"),
        Entry(75.0, None, "[Student-Omar points at screen]", "visual", "video"),
        Entry(200.0, "Student-Omar", "maybe its a hexagon", "speech", "offpair"),
    ]
    out = format_transcript(entries, header_lines=["Unified transcript", "Source: SG2"])
    assert "Unified transcript" in out
    assert "01:03 Student-Maya: spin it again" in out
    assert "01:15 [Student-Omar points at screen]" in out
    assert "03:20 Student-Omar: maybe its a hexagon" in out

    tm = TimeMap(a=1.0, b=480.0, residual=0.2, confidence=0.8)
    pm = PairMap(mapping={"Speaker-B": "Student-Omar"}, confidence=0.9)
    audit = build_audit(tm, threshold=0.5, close_count=120, faint_count=300,
                        pair_map=pm, inserted=2, discarded=5, warnings=["low overlap"])
    assert audit["time_map"]["b"] == 480.0
    assert audit["energy"]["close_count"] == 120
    assert audit["pair2"]["mapping"]["Speaker-B"] == "Student-Omar"
    assert audit["counts"] == {"inserted": 2, "discarded": 5}
    assert audit["warnings"] == ["low overlap"]
