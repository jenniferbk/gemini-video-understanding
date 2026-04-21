"""Tests for name de-identification module."""
import json
from pathlib import Path
import pytest

from deidentify_names import (
    NameEntry, AdultEntry, NameMap, load_pseudonym_pool,
)

REPO_ROOT = Path(__file__).parent
POOL_PATH = REPO_ROOT / "pseudonym_pool.json"


def test_name_entry_defaults():
    e = NameEntry(real_name="Melanie", gender="F", visual_label=None, pseudonym="Student-Hannah")
    assert e.nicknames == []


def test_name_map_serializes_roundtrip():
    m = NameMap(
        students=[NameEntry(real_name="Melanie", gender="F",
                            visual_label="Girl-PinkShirtBlackPants",
                            pseudonym="Student-Hannah", nicknames=["Mel"])],
        adults=[AdultEntry(real_name="Sheridan", honorific="Ms.",
                          pseudonym="Ms. Kelly")],
    )
    d = m.to_dict()
    m2 = NameMap.from_dict(d)
    assert m2 == m


def test_load_pool_returns_four_buckets():
    pool = load_pseudonym_pool(str(POOL_PATH))
    assert set(pool.keys()) == {
        "student_female", "student_male", "student_neutral", "adult_last",
    }
    assert len(pool["student_female"]) >= 20
    assert len(pool["student_male"]) >= 20
    assert len(pool["adult_last"]) >= 20


def test_load_pool_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pseudonym_pool("/nonexistent/pool.json")


from deidentify_names import assign_pseudonym, assign_adult_pseudonym


def test_assign_pseudonym_avoids_real_names():
    # If "Hannah" is in the transcript as a real name, don't pick Student-Hannah
    pool = {"student_female": ["Hannah", "Ava", "Sophia"]}
    result = assign_pseudonym(
        gender="F", pool=pool, avoid_real_names={"Hannah", "Melanie"},
        already_assigned=set(),
    )
    assert result == "Student-Ava"


def test_assign_pseudonym_avoids_already_assigned():
    pool = {"student_female": ["Hannah", "Ava", "Sophia"]}
    result = assign_pseudonym(
        gender="F", pool=pool, avoid_real_names=set(),
        already_assigned={"Student-Hannah", "Student-Ava"},
    )
    assert result == "Student-Sophia"


def test_assign_pseudonym_neutral_when_unknown_gender():
    pool = {
        "student_female": ["Hannah"],
        "student_male": ["Michael"],
        "student_neutral": ["Alex", "Jordan"],
    }
    result = assign_pseudonym(
        gender="N", pool=pool, avoid_real_names=set(),
        already_assigned=set(),
    )
    assert result == "Student-Alex"


def test_assign_pseudonym_exhausted_pool_raises():
    pool = {"student_female": ["Hannah"]}
    with pytest.raises(ValueError, match="pool exhausted"):
        assign_pseudonym(
            gender="F", pool=pool, avoid_real_names={"Hannah"},
            already_assigned=set(),
        )


def test_assign_adult_pseudonym():
    pool = {"adult_last": ["Kelly", "Walker"]}
    result = assign_adult_pseudonym(
        honorific="Ms.", pool=pool, avoid_real_names=set(),
        already_assigned=set(),
    )
    assert result == "Ms. Kelly"


def test_assign_adult_pseudonym_avoids_real_last_name():
    pool = {"adult_last": ["Kelly", "Walker"]}
    result = assign_adult_pseudonym(
        honorific="Ms.", pool=pool, avoid_real_names={"Kelly"},
        already_assigned=set(),
    )
    assert result == "Ms. Walker"


from deidentify_names import build_name_extraction_prompt


def test_prompt_includes_transcript_and_schema():
    transcript = "39:43 Teacher-PinkPants: Melanie, come on up."
    prompt = build_name_extraction_prompt(transcript)
    # Prompt must embed the transcript verbatim
    assert transcript in prompt
    # Prompt must describe the output JSON schema
    assert '"students"' in prompt
    assert '"adults"' in prompt
    assert "real_name" in prompt
    assert "visual_label" in prompt
    assert "honorific" in prompt
    # Prompt must tell the model to output JSON only
    assert "JSON" in prompt


def test_prompt_handles_empty_transcript():
    prompt = build_name_extraction_prompt("")
    # An empty transcript should produce an empty TRANSCRIPT: section,
    # not raise and not drop the section header.
    assert prompt.rstrip("\n").endswith("TRANSCRIPT:")


from deidentify_names import parse_name_extraction_response


def test_parse_happy_path():
    raw = '''{"students": [{"real_name": "Melanie", "gender": "F", "visual_label": "Girl-PinkShirtBlackPants", "nicknames": ["Mel"]}], "adults": [{"real_name": "Sheridan", "honorific": "Ms.", "visual_label": null}]}'''
    detected = parse_name_extraction_response(raw)
    assert len(detected["students"]) == 1
    assert detected["students"][0]["real_name"] == "Melanie"
    assert detected["students"][0]["nicknames"] == ["Mel"]
    assert detected["adults"][0]["honorific"] == "Ms."


def test_parse_strips_code_fences():
    raw = '```json\n{"students": [], "adults": []}\n```'
    detected = parse_name_extraction_response(raw)
    assert detected == {"students": [], "adults": []}


def test_parse_empty():
    raw = '{"students": [], "adults": []}'
    detected = parse_name_extraction_response(raw)
    assert detected == {"students": [], "adults": []}


def test_parse_malformed_raises():
    with pytest.raises(ValueError, match="could not parse"):
        parse_name_extraction_response("not json at all")


def test_parse_fills_missing_nicknames():
    raw = '{"students": [{"real_name": "Piper", "gender": "F", "visual_label": null}], "adults": []}'
    detected = parse_name_extraction_response(raw)
    assert detected["students"][0]["nicknames"] == []


def test_parse_rejects_bad_gender():
    raw = '{"students": [{"real_name": "Piper", "gender": "Q", "visual_label": null, "nicknames": []}], "adults": []}'
    detected = parse_name_extraction_response(raw)
    assert detected["students"][0]["gender"] == "N"  # coerced to neutral


def test_parse_rejects_non_dict_top_level():
    with pytest.raises(ValueError, match="expected JSON object"):
        parse_name_extraction_response("[]")
    with pytest.raises(ValueError, match="expected JSON object"):
        parse_name_extraction_response('"not a dict"')


def test_parse_rejects_adult_missing_honorific():
    raw = '{"students": [], "adults": [{"real_name": "Sheridan", "visual_label": null}]}'
    with pytest.raises(ValueError, match="missing 'honorific'"):
        parse_name_extraction_response(raw)


def test_parse_rejects_student_missing_real_name():
    raw = '{"students": [{"gender": "F"}], "adults": []}'
    with pytest.raises(ValueError, match="missing 'real_name'"):
        parse_name_extraction_response(raw)


from deidentify_names import build_name_map


def test_build_name_map_assigns_distinct_pseudonyms():
    detected = {
        "students": [
            {"real_name": "Melanie", "gender": "F", "visual_label": "Girl-PinkShirtBlackPants", "nicknames": ["Mel"]},
            {"real_name": "Piper", "gender": "F", "visual_label": None, "nicknames": []},
            {"real_name": "James", "gender": "M", "visual_label": None, "nicknames": []},
        ],
        "adults": [
            {"real_name": "Sheridan", "honorific": "Ms.", "visual_label": None},
        ],
    }
    pool = {
        "student_female": ["Hannah", "Ava", "Sophia"],
        "student_male": ["Michael", "Ethan"],
        "student_neutral": ["Alex"],
        "adult_last": ["Kelly", "Walker"],
    }
    name_map = build_name_map(detected, pool)
    assert len(name_map.students) == 3
    pseudonyms = [s.pseudonym for s in name_map.students]
    assert len(set(pseudonyms)) == 3  # all distinct
    # Melanie is F → female bucket
    melanie = next(s for s in name_map.students if s.real_name == "Melanie")
    assert melanie.pseudonym.startswith("Student-")
    assert melanie.pseudonym in {"Student-Hannah", "Student-Ava", "Student-Sophia"}
    # James is M → male bucket
    james = next(s for s in name_map.students if s.real_name == "James")
    assert james.pseudonym in {"Student-Michael", "Student-Ethan"}
    # Adult
    assert name_map.adults[0].pseudonym == "Ms. Kelly"


def test_build_name_map_avoids_real_name_collision():
    # If "Hannah" appears as a real student name, don't pick Student-Hannah for anyone
    detected = {
        "students": [
            {"real_name": "Hannah", "gender": "F", "visual_label": None, "nicknames": []},
            {"real_name": "Melanie", "gender": "F", "visual_label": None, "nicknames": []},
        ],
        "adults": [],
    }
    pool = {
        "student_female": ["Hannah", "Ava", "Sophia"],
        "student_male": [],
        "student_neutral": [],
        "adult_last": [],
    }
    name_map = build_name_map(detected, pool)
    pseudonyms = {s.pseudonym for s in name_map.students}
    assert "Student-Hannah" not in pseudonyms


def test_build_name_map_preserves_visual_labels_and_nicknames():
    detected = {
        "students": [
            {"real_name": "Melanie", "gender": "F",
             "visual_label": "Girl-PinkShirtBlackPants", "nicknames": ["Mel"]},
        ],
        "adults": [],
    }
    pool = {"student_female": ["Hannah"], "student_male": [],
            "student_neutral": [], "adult_last": []}
    name_map = build_name_map(detected, pool)
    entry = name_map.students[0]
    assert entry.visual_label == "Girl-PinkShirtBlackPants"
    assert entry.nicknames == ["Mel"]
