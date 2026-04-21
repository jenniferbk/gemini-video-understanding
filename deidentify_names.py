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
import re
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


_GENDER_BUCKET = {
    "F": "student_female",
    "M": "student_male",
    "N": "student_neutral",
}


def assign_pseudonym(
    gender: str,
    pool: Dict[str, List[str]],
    avoid_real_names: set,
    already_assigned: set,
) -> str:
    """Pick the first name from the gender-appropriate pool bucket that does
    not collide with any real name or previously-assigned pseudonym.

    Returns: "Student-<FirstName>"
    Raises: ValueError if no eligible name remains in the bucket.
    """
    bucket_key = _GENDER_BUCKET.get(gender, "student_neutral")
    bucket = pool.get(bucket_key, [])
    lowered_avoid = {n.lower() for n in avoid_real_names}
    for candidate in bucket:
        if candidate.lower() in lowered_avoid:
            continue
        proposed = f"Student-{candidate}"
        if proposed in already_assigned:
            continue
        return proposed
    raise ValueError(f"pool exhausted for bucket {bucket_key!r}")


def assign_adult_pseudonym(
    honorific: str,
    pool: Dict[str, List[str]],
    avoid_real_names: set,
    already_assigned: set,
) -> str:
    """Pick the first last name from the adult pool that does not collide.

    Returns: "<Honorific> <LastName>" (e.g., "Ms. Kelly").
    Raises: ValueError if no eligible name remains.
    """
    bucket = pool.get("adult_last", [])
    lowered_avoid = {n.lower() for n in avoid_real_names}
    for candidate in bucket:
        if candidate.lower() in lowered_avoid:
            continue
        proposed = f"{honorific} {candidate}"
        if proposed in already_assigned:
            continue
        return proposed
    raise ValueError("pool exhausted for bucket 'adult_last'")


_NAME_EXTRACTION_PROMPT_TEMPLATE = '''You are helping de-identify a classroom transcript for educational research.

The transcript uses VISUAL-DESCRIPTION labels for speakers (e.g., "Teacher-PinkPants", "Boy-BlackTShirtGlasses", "Girl-PinkShirtBlackPants"). But real first names may still appear in TWO places:

1. As speaker labels — when the transcription model switched from a visual description to a real name it heard (e.g., `Melanie: Two sides are the same.`).
2. Inside dialogue — when someone addresses another person by name (e.g., `Teacher-PinkPants: Melanie, come on up.`, `Piper, what's pseudo-code?`).

YOUR TASK: Identify every real first name and adult honorific-name that refers to an actual person present in this classroom, and for each one return structured JSON.

For each student real name:
- `real_name`: the first name as spelled in the transcript
- `gender`: "F", "M", or "N" (use "N" only when you genuinely cannot tell from context/pronouns)
- `visual_label`: if you can link this name to an existing visual-description speaker label in the transcript, give the EXACT label string. Otherwise null.
- `nicknames`: list of nicknames that clearly refer to this same person (e.g., "Mel" for "Melanie"). ONLY include when linkage is unambiguous. Prefer [] when in doubt.

For each adult real name (teacher, aide, principal mentioned by name):
- `real_name`: last name only (e.g., "Sheridan")
- `honorific`: "Ms.", "Mr.", "Mrs.", "Mx.", or "Dr."
- `visual_label`: usually null unless the named adult is also a visible speaker
- Do NOT include the primary classroom teacher if they are already labeled `Teacher-*` with no name spoken.

EXCLUDE:
- Names from curriculum content (historical figures, book characters, math-problem characters)
- Pronoun antecedents that are not actual names
- Last names used alone without honorific (too risky; skip them)

OUTPUT JSON ONLY, matching this schema exactly. No prose, no code fences.

{{"students": [{{"real_name": "Melanie", "gender": "F", "visual_label": "Girl-PinkShirtBlackPants", "nicknames": ["Mel"]}}], "adults": [{{"real_name": "Sheridan", "honorific": "Ms.", "visual_label": null}}]}}

If no names are found, return {{"students": [], "adults": []}}.

TRANSCRIPT:
{transcript}
'''


def build_name_extraction_prompt(transcript_text: str) -> str:
    return _NAME_EXTRACTION_PROMPT_TEMPLATE.format(transcript=transcript_text)


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_VALID_GENDERS = {"F", "M", "N"}


def parse_name_extraction_response(raw: str) -> Dict:
    """Parse Gemini's JSON response into a normalized dict.

    Strips code fences if present, coerces missing `nicknames` to `[]`, and
    coerces out-of-vocabulary genders to `N`.

    Returns dict with keys "students", "adults" — each a list of plain dicts.
    Raises ValueError on unparseable input.
    """
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse name extraction response: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")

    students = []
    for s in data.get("students", []):
        if "real_name" not in s:
            raise ValueError(f"student entry missing 'real_name': {s!r}")
        gender = s.get("gender", "N")
        if gender not in _VALID_GENDERS:
            gender = "N"
        students.append({
            "real_name": s["real_name"],
            "gender": gender,
            "visual_label": s.get("visual_label"),
            "nicknames": s.get("nicknames") or [],
        })

    adults = []
    for a in data.get("adults", []):
        if "real_name" not in a:
            raise ValueError(f"adult entry missing 'real_name': {a!r}")
        if "honorific" not in a:
            raise ValueError(f"adult entry missing 'honorific': {a!r}")
        adults.append({
            "real_name": a["real_name"],
            "honorific": a["honorific"],
            "visual_label": a.get("visual_label"),
        })

    return {"students": students, "adults": adults}


def build_name_map(detected: Dict, pool: Dict[str, List[str]]) -> NameMap:
    """Assign non-colliding pseudonyms to detected names and return a NameMap.

    `detected` is the dict returned by `parse_name_extraction_response`.
    """
    # All real first and last names detected — pseudonyms must avoid any of these
    avoid_real = set()
    for s in detected.get("students", []):
        avoid_real.add(s["real_name"])
        for nick in s.get("nicknames", []):
            avoid_real.add(nick)
    for a in detected.get("adults", []):
        avoid_real.add(a["real_name"])

    assigned: set = set()
    students: List[NameEntry] = []
    for s in detected.get("students", []):
        pseudonym = assign_pseudonym(
            gender=s["gender"],
            pool=pool,
            avoid_real_names=avoid_real,
            already_assigned=assigned,
        )
        assigned.add(pseudonym)
        students.append(NameEntry(
            real_name=s["real_name"],
            gender=s["gender"],
            visual_label=s.get("visual_label"),
            pseudonym=pseudonym,
            nicknames=list(s.get("nicknames", [])),
        ))

    adults: List[AdultEntry] = []
    for a in detected.get("adults", []):
        pseudonym = assign_adult_pseudonym(
            honorific=a["honorific"],
            pool=pool,
            avoid_real_names=avoid_real,
            already_assigned=assigned,
        )
        assigned.add(pseudonym)
        adults.append(AdultEntry(
            real_name=a["real_name"],
            honorific=a["honorific"],
            pseudonym=pseudonym,
            visual_label=a.get("visual_label"),
        ))

    return NameMap(students=students, adults=adults)


def apply_name_map(text: str, name_map: NameMap) -> str:
    """Substitute real names and visual labels with pseudonyms throughout.

    Single-pass substitution: one regex matches any token (real name, visual
    label, nickname, or honorific-last-name pair) and the callback looks up
    the pseudonym. Because each input position is consumed at most once,
    pseudonyms written into the output are never re-scanned -- this prevents
    chain-corruption when a real name happens to be a substring of another
    student's pseudonym.

    Longer tokens are tried first so "Mel" doesn't eat part of "Melanie" and
    "Ms. Sheridan" wins over a lone "Sheridan" (which wouldn't match anyway
    because bare last names are not registered).
    """
    # Build (token, pseudonym) list. Later tokens with the same text overwrite
    # earlier ones in the lookup; if a nickname collides with another student's
    # real name the LLM prompt should have prevented it, but last-write-wins
    # gives deterministic behavior.
    mappings: List[tuple] = []
    for entry in name_map.students:
        if entry.visual_label:
            mappings.append((entry.visual_label, entry.pseudonym))
        mappings.append((entry.real_name, entry.pseudonym))
        for nick in entry.nicknames:
            mappings.append((nick, entry.pseudonym))
    for adult in name_map.adults:
        mappings.append((f"{adult.honorific} {adult.real_name}", adult.pseudonym))
        if adult.visual_label:
            mappings.append((adult.visual_label, adult.pseudonym))

    if not mappings:
        return text

    # Sort by token length descending so the alternation prefers the longest
    # match at any given position (regex alternation is left-to-right / first-
    # match, not longest-match, so ordering matters).
    mappings.sort(key=lambda m: len(m[0]), reverse=True)

    lookup = {tok: ps for tok, ps in mappings}
    alternation = "|".join(re.escape(tok) for tok, _ in mappings)
    pattern = re.compile(rf"\b(?P<tok>{alternation})(?P<poss>'s)?\b")

    def _sub(m: "re.Match") -> str:
        return lookup[m.group("tok")] + (m.group("poss") or "")

    return pattern.sub(_sub, text)


def deidentify_transcript(
    transcript_text: str,
    gemini_client,
    pool_path: str,
) -> "tuple[str, NameMap]":
    """Run the full de-identification pipeline.

    Args:
        transcript_text: the stitched v10 transcript.
        gemini_client: a `GeminiClient` (from video_transcription_pipeline_v10)
            or any object with a `.generate(contents, temperature=...)` method
            that returns the model's response text as a string.
        pool_path: path to pseudonym_pool.json.

    Returns:
        (deidentified_text, name_map) -- writing the name_map alongside the
        transcript gives researchers an audit trail.
    """
    pool = load_pseudonym_pool(pool_path)
    prompt = build_name_extraction_prompt(transcript_text)
    raw = gemini_client.generate([prompt], temperature=0.0)
    detected = parse_name_extraction_response(raw)
    name_map = build_name_map(detected, pool)
    return apply_name_map(transcript_text, name_map), name_map
