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
