#!/usr/bin/env python3
"""Merge an off-pair audio transcript into a video transcript on the video timeline.

The off-pair recorder is a stereo desk mic on two focal students ("Pair 2"); it hears
them close/loud but also picks up the rest of the room as faint bleed, which Gemini
tends to hallucinate. So the off-pair is used conservatively: energy-gate to keep only
close speech, then fill ONLY the gaps the video missed. The video stays authoritative.

See docs/superpowers/specs/2026-05-29-offpair-merge-design.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field  # field used by later tasks  # noqa: F401
from typing import List, Literal, Optional, Tuple  # Tuple used by later tasks  # noqa: F401

_TS_RE = re.compile(r"^(\d{1,3}):(\d{2})\s+(.*)$")


@dataclass
class Entry:
    time_s: float
    speaker: Optional[str]
    text: str
    kind: Literal["speech", "visual"]
    source: Literal["video", "offpair"]


def parse_transcript_text(text: str, source: str) -> List[Entry]:
    """Parse a v10 / off-pair transcript into timestamped entries.

    Lines look like 'MM:SS Speaker: words' (speech) or 'MM:SS [action]' (visual).
    Chunk headers ('--- CHUNK ... ---'), file headers, and blank lines are skipped.
    Minutes may exceed 59 (e.g. '75:10').
    """
    entries: List[Entry] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TS_RE.match(line)
        if not m:
            continue  # header / banner / non-timestamped line
        mm, ss, rest = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        if ss >= 60:
            continue  # malformed seconds field (e.g. '40:60')
        if not rest:
            continue  # timestamp with no body (e.g. '41:00 ')
        t = mm * 60 + ss
        if rest.startswith("[") and rest.endswith("]"):
            entries.append(Entry(t, None, rest, "visual", source))
            continue
        # speech: 'Speaker: text' — speaker labels never contain a colon
        sm = re.match(r"^([^:\[]+):\s*(.*)$", rest)
        if sm:
            entries.append(Entry(t, sm.group(1).strip(), sm.group(2).strip(), "speech", source))
        else:
            entries.append(Entry(t, None, rest, "visual", source))
    return entries
