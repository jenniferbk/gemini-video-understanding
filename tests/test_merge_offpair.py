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
