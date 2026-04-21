"""Post-transcription name de-identification for classroom videos.

Runs a second Gemini pass over the stitched transcript, detects real first
names used in the classroom, and substitutes realistic pseudonyms for both
speaker labels and in-text mentions. Off by default; opt in with the
`deidentify_names` config flag.

Policy (A, 2026-04-20): when a real name is linked to an existing
visual-description label (e.g., Girl-PinkShirtBlackPants), that visual label
is RETIRED for that student — the pseudonym replaces both the real-name
mention and the visual-description label everywhere in the transcript.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class NameEntry:
    real_name: str
    gender: str  # "F", "M", or "N"
    visual_label: Optional[str]
    pseudonym: str
    nicknames: List[str] = field(default_factory=list)


@dataclass
class AdultEntry:
    real_name: str
    honorific: str  # "Ms.", "Mr.", "Mx.", "Mrs.", "Dr."
    pseudonym: str  # e.g., "Ms. Kelly"
    visual_label: Optional[str] = None


@dataclass
class NameMap:
    students: List[NameEntry] = field(default_factory=list)
    adults: List[AdultEntry] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "NameMap":
        return cls(
            students=[NameEntry(**s) for s in d.get("students", [])],
            adults=[AdultEntry(**a) for a in d.get("adults", [])],
        )


def load_pseudonym_pool(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pseudonym pool not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
