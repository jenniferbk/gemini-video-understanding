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


from deidentify_names import apply_name_map


def _name_map_melanie():
    return NameMap(
        students=[NameEntry(
            real_name="Melanie", gender="F",
            visual_label="Girl-PinkShirtBlackPants",
            pseudonym="Student-Hannah", nicknames=["Mel"],
        )],
        adults=[AdultEntry(
            real_name="Sheridan", honorific="Ms.",
            pseudonym="Ms. Kelly",
        )],
    )


def test_apply_replaces_real_name_label():
    src = "43:02 Melanie: Two sides are the same."
    out = apply_name_map(src, _name_map_melanie())
    assert out == "43:02 Student-Hannah: Two sides are the same."


def test_apply_replaces_visual_label_to_retire_it():
    # Policy A: the visual label is also rewritten to the pseudonym
    src = "39:58 Girl-PinkShirtBlackPants: I have a question."
    out = apply_name_map(src, _name_map_melanie())
    assert out == "39:58 Student-Hannah: I have a question."


def test_apply_replaces_real_name_in_dialogue():
    src = "39:43 Teacher-PinkPants: Melanie, come on up."
    out = apply_name_map(src, _name_map_melanie())
    assert out == "39:43 Teacher-PinkPants: Student-Hannah, come on up."


def test_apply_replaces_possessive():
    src = "Teacher-PinkPants: That's Melanie's answer."
    out = apply_name_map(src, _name_map_melanie())
    assert out == "Teacher-PinkPants: That's Student-Hannah's answer."


def test_apply_replaces_nickname():
    src = "Teacher-PinkPants: Good job, Mel."
    out = apply_name_map(src, _name_map_melanie())
    assert out == "Teacher-PinkPants: Good job, Student-Hannah."


def test_apply_replaces_adult_honorific_name():
    src = "Teacher-PinkPants: You've done it with Ms. Sheridan."
    out = apply_name_map(src, _name_map_melanie())
    assert out == "Teacher-PinkPants: You've done it with Ms. Kelly."


def test_apply_does_not_touch_unrelated_words():
    # "Graham" is a real name in some classrooms but also a cracker;
    # if not in the map, must not be replaced.
    src = "Teacher-PinkPants: Let's eat graham crackers."
    out = apply_name_map(src, _name_map_melanie())
    assert out == src


def test_apply_is_case_sensitive_word_boundary():
    # Lowercase "melanie" at a proper word boundary -- should NOT match
    # because classroom transcripts preserve name casing and lowercase
    # occurrences are almost always common-word collisions.
    src = "Teacher-PinkPants: Let's go see melanie at lunch."
    out = apply_name_map(src, _name_map_melanie())
    assert out == src


def test_apply_multiple_names_across_transcript():
    name_map = NameMap(
        students=[
            NameEntry(real_name="Melanie", gender="F", visual_label=None,
                      pseudonym="Student-Hannah", nicknames=[]),
            NameEntry(real_name="James", gender="M", visual_label=None,
                      pseudonym="Student-Michael", nicknames=[]),
        ],
        adults=[],
    )
    src = (
        "Teacher-PinkPants: Melanie, walk forward.\n"
        "Teacher-PinkPants: James, what's next?\n"
        "James: Turn left."
    )
    expected = (
        "Teacher-PinkPants: Student-Hannah, walk forward.\n"
        "Teacher-PinkPants: Student-Michael, what's next?\n"
        "Student-Michael: Turn left."
    )
    assert apply_name_map(src, name_map) == expected


def test_apply_no_pseudonym_chain_corruption():
    # Regression test for the chain-corruption bug. If Melanie's pseudonym
    # contains "Hannah" as a substring and "Hannah" is also a real student,
    # the naive sequential-replace approach would corrupt the output.
    nm = NameMap(
        students=[
            NameEntry(real_name="Melanie", gender="F", visual_label=None,
                      pseudonym="Student-Hannah", nicknames=[]),
            NameEntry(real_name="Hannah", gender="F", visual_label=None,
                      pseudonym="Student-Ava", nicknames=[]),
        ],
        adults=[],
    )
    src = "Teacher-PinkPants: Melanie and Hannah are here."
    out = apply_name_map(src, nm)
    assert out == "Teacher-PinkPants: Student-Hannah and Student-Ava are here."


def test_apply_handles_trailing_punctuation():
    # Names followed by various punctuation should still match.
    src = "Teacher: Is James? (James, not Jim.) James's book."
    nm = NameMap(
        students=[NameEntry(real_name="James", gender="M", visual_label=None,
                            pseudonym="Student-Michael", nicknames=[])],
        adults=[],
    )
    out = apply_name_map(src, nm)
    assert out == "Teacher: Is Student-Michael? (Student-Michael, not Jim.) Student-Michael's book."


def test_apply_visual_label_possessive():
    # Visual labels can have possessive too (though rare in practice).
    src = "That is Girl-PinkShirtBlackPants's book."
    nm = NameMap(
        students=[NameEntry(real_name="Melanie", gender="F",
                            visual_label="Girl-PinkShirtBlackPants",
                            pseudonym="Student-Hannah", nicknames=[])],
        adults=[],
    )
    out = apply_name_map(src, nm)
    assert out == "That is Student-Hannah's book."


from unittest.mock import MagicMock
from deidentify_names import deidentify_transcript


def test_deidentify_transcript_end_to_end_with_mock():
    transcript = (
        "39:43 Teacher-PinkPants: Melanie, come on up.\n"
        "40:45 Teacher-PinkPants: Piper, what's pseudo-code?\n"
        "40:47 Piper: Fake code.\n"
        "43:02 Girl-PinkShirtBlackPants: Two sides are the same."
    )
    # Mock Gemini: GeminiClient.generate() returns a str (the response text).
    mock_client = MagicMock()
    canned = '''{"students": [
        {"real_name": "Melanie", "gender": "F", "visual_label": "Girl-PinkShirtBlackPants", "nicknames": []},
        {"real_name": "Piper", "gender": "F", "visual_label": null, "nicknames": []}
    ], "adults": []}'''
    mock_client.generate.return_value = canned

    result_text, name_map = deidentify_transcript(
        transcript, mock_client, str(POOL_PATH),
    )

    # Melanie -> Student-<female>; Piper -> Student-<different female>
    melanie_pseudo = next(s.pseudonym for s in name_map.students if s.real_name == "Melanie")
    piper_pseudo = next(s.pseudonym for s in name_map.students if s.real_name == "Piper")
    assert melanie_pseudo != piper_pseudo
    # No real names leak
    assert "Melanie" not in result_text
    assert "Piper" not in result_text
    # Visual label retired
    assert "Girl-PinkShirtBlackPants" not in result_text
    # Pseudonyms appear in place of both labels and in-text mentions
    assert melanie_pseudo in result_text
    assert piper_pseudo in result_text
    # Gemini was called exactly once with a prompt containing the transcript
    assert mock_client.generate.call_count == 1


FIXTURE_DIR = REPO_ROOT / "test_fixtures" / "deidentify"


def test_melanie_excerpt_no_real_names_leak():
    """Integration check: run the pipeline with a realistic canned Gemini
    response and verify no real names survive in the output."""
    excerpt = (FIXTURE_DIR / "melanie_excerpt.txt").read_text()

    # Real names observed in the excerpt by manual review:
    real_names = {"Melanie", "Piper", "Vera", "Chevy", "River", "Graham",
                  "James", "Shavy", "Aubrey"}
    # Plus the adult reference "Ms. Sheridan"

    canned = json.dumps({
        "students": [
            {"real_name": "Melanie", "gender": "F",
             "visual_label": "Girl-PinkShirtBlackPants", "nicknames": []},
            {"real_name": "Piper", "gender": "F", "visual_label": None, "nicknames": []},
            {"real_name": "Vera", "gender": "F", "visual_label": None, "nicknames": []},
            {"real_name": "Chevy", "gender": "N", "visual_label": None, "nicknames": []},
            {"real_name": "River", "gender": "N", "visual_label": None, "nicknames": []},
            {"real_name": "Graham", "gender": "M", "visual_label": None, "nicknames": []},
            {"real_name": "James", "gender": "M", "visual_label": None, "nicknames": []},
            {"real_name": "Shavy", "gender": "N", "visual_label": None, "nicknames": []},
            {"real_name": "Aubrey", "gender": "F", "visual_label": None, "nicknames": []},
        ],
        "adults": [
            {"real_name": "Sheridan", "honorific": "Ms.", "visual_label": None},
        ],
    })
    mock_client = MagicMock()
    mock_client.generate.return_value = canned

    result_text, name_map = deidentify_transcript(
        excerpt, mock_client, str(POOL_PATH),
    )

    # No real name should appear anywhere in the output
    for name in real_names:
        assert name not in result_text, (
            f"Real name {name!r} leaked into deidentified output"
        )
    # "Ms. Sheridan" should be gone
    assert "Ms. Sheridan" not in result_text
    # The visual label linked to Melanie should be retired
    assert "Girl-PinkShirtBlackPants" not in result_text
    # All students got distinct pseudonyms
    pseudonyms = [s.pseudonym for s in name_map.students]
    assert len(set(pseudonyms)) == len(pseudonyms)
    # All pseudonyms use the Student- prefix
    assert all(p.startswith("Student-") for p in pseudonyms)
    # Adult pseudonym uses Ms. prefix
    assert name_map.adults[0].pseudonym.startswith("Ms. ")
