#!/usr/bin/env python3
"""
TIMSS benchmark runner for the v10 multimodal transcription pipeline.

Pipeline:
  1. yt-dlp the TIMSS YouTube video to mp4
  2. Run video_transcription_pipeline_v10.py on it
  3. Project v10 output -> TIMSS-comparable form using projection_config.yaml
  4. Score three layers:
        (a) Speech WER vs the TIMSS gold transcript
        (b) Speaker-label agreement on T/S/SN/SS axis
        (c) Visual yield: bracketed visual events per minute (descriptive only)
  5. Write a JSON report next to the v10 output dir.

Usage:
    python benchmark_timss.py \
        --lesson US1 \
        --youtube https://www.youtube.com/watch?v=5Eg1fJ-ZpQs \
        --gold "/Users/jenniferkleiman/Documents/COMS/Math+transcripts+as+txt+files+(1)/Math US1 transcript.txt" \
        --workdir ./benchmark_runs \
        [--skip-pipeline]   # reuse an existing v10 output dir
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: pip install pyyaml jiwer")

try:
    from jiwer import wer
except ImportError:
    sys.exit("Missing dependency: pip install jiwer")

# Shared content-equivalence rules (same logic as the HTML review tool).
from content_equivalence import (
    score_content_equivalence,
    Turn as EqTurn,
    EquivalenceScore,
)


# ---------------------------------------------------------------------------
# Data classes

@dataclass
class Turn:
    """One speech turn after projection — directly comparable to TIMSS rows."""
    t_seconds: float | None
    speaker: str        # projected label: T/S/SN/SS/BS/UNK
    text: str           # cleaned text, no visual annotations, no badges


@dataclass
class VisualEvent:
    """A bracketed visual annotation extracted from v10 output."""
    t_seconds: float
    speaker_raw: str
    description: str    # contents of the [ ... ]


@dataclass
class DroppedTurn:
    """A gold turn with no matching pred turn nearby — a candidate omission."""
    t_seconds: float
    speaker: str
    text: str
    best_pred_overlap: float    # jaccard of word sets, 0..1
    best_pred_text: str | None  # nearest pred turn text within window, if any


@dataclass
class BenchmarkReport:
    lesson: str
    youtube_url: str
    gold_path: str
    v10_output_path: str
    n_gold_turns: int
    n_pred_turns: int
    n_gold_words: int
    n_pred_words: int
    n_visual_events: int
    visual_events_per_minute: float
    wer: float                   # aggressive normalization (primary)
    wer_strict: float            # legacy normalization
    wer_per_role: dict           # per-speaker-role WER (T / student / weighted)
    content_equivalence: dict    # primary metric: mirrors review-tool rules
    speaker_role_content_aware: dict  # content-aligned speaker accuracy (fair)
    wer_n_words_ref: int
    audit: dict | None            # audit-based stats if --audit supplied
    speaker_label_accuracy: float
    speaker_confusion: dict
    n_dropped_turns: int
    dropped_turns: list[dict] = field(default_factory=list)
    projection_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsers

GOLD_LINE_RE = re.compile(r'^(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)$')
V10_LINE_RE  = re.compile(r'^(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)$')
BRACKET_RE   = re.compile(r'\[([^\]]+)\]')


def hms_to_seconds(s: str) -> float:
    parts = [int(p) for p in s.split(':')]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    raise ValueError(f"bad timestamp: {s}")


def parse_gold(path: Path) -> list[Turn]:
    """TIMSS format: 'HH:MM:SS\\tSPEAKER\\ttext'."""
    turns: list[Turn] = []
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        m = GOLD_LINE_RE.match(line)
        if not m:
            # Some TIMSS lines are tab-separated rather than space-separated.
            parts = line.split('\t', 2)
            if len(parts) == 3 and re.match(r'^\d{2}:\d{2}:\d{2}$', parts[0]):
                ts, sp, txt = parts
                turns.append(Turn(hms_to_seconds(ts), sp.strip(), txt.strip()))
            continue
        ts, sp, txt = m.groups()
        turns.append(Turn(hms_to_seconds(ts), sp.strip(), txt.strip()))
    return turns


def parse_v10(path: Path, cfg: dict) -> tuple[list[Turn], list[VisualEvent], list[str]]:
    """
    Parse the v10 complete-transcript file. Returns (projected_turns, visual_events, warnings).
    Lines look like:
        00:09 A06: What? 🚨 *LOW CONFIDENCE - Speaker: 40%, Content: 60%*
        01:09 Ava: Let's see. ⚠️ *Speaker: 60%, Content: 53%*
        00:16 A06: I did that... [points to screen] ✅
    """
    strip_res = [re.compile(p) for p in cfg.get('strip_patterns', [])]
    normalize = cfg.get('normalize', {}) or {}
    speaker_rules = [(re.compile(r['pattern']), r['label'])
                     for r in cfg.get('speaker_map', [])]

    turns: list[Turn] = []
    visuals: list[VisualEvent] = []
    warnings: list[str] = []

    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line or line.startswith('---') or line.startswith('=='):
            continue
        m = V10_LINE_RE.match(line)
        if not m:
            continue
        ts_str, speaker_raw, body = m.groups()
        try:
            t_sec = hms_to_seconds(ts_str)
        except ValueError:
            continue

        # Standalone visual-only lines look like:
        #   00:32 [S-BlackShirt walks around handing out paper.]
        #   00:40 [Close-up of worksheet... it says: "Using a pencil..."]
        # The second form fools V10_LINE_RE because the bracketed content
        # contains a colon. Detect by leading '[' and treat as a pure visual
        # event with no speech contribution.
        if speaker_raw.strip().startswith('['):
            # Reconstruct the full bracketed string from the original line.
            inner_full = line[len(ts_str):].strip()
            inner_full = inner_full.lstrip('[').rstrip(']').strip()
            visuals.append(VisualEvent(t_sec, '<visual>', inner_full))
            continue

        # Pull visual events out BEFORE stripping, so we can record them.
        for bm in BRACKET_RE.finditer(body):
            inner = bm.group(1).strip()
            # 'inaudible' is speech metadata, not a visual event.
            if 'inaudible' in inner.lower() or 'unclear' in inner.lower():
                continue
            visuals.append(VisualEvent(t_sec, speaker_raw.strip(), inner))

        # Apply strip patterns to get clean text.
        cleaned = body
        for r in strip_res:
            cleaned = r.sub('', cleaned)
        for k, v in normalize.items():
            cleaned = cleaned.replace(k, v)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned:
            continue

        # Map speaker.
        sp_token = speaker_raw.strip()
        label = None
        for rx, lab in speaker_rules:
            if rx.match(sp_token):
                label = lab
                break
        if label is None:
            label = 'UNK'
            warnings.append(f"unmapped speaker token: {sp_token!r}")

        turns.append(Turn(t_sec, label, cleaned))

    return turns, visuals, warnings


# ---------------------------------------------------------------------------
# Scoring

# Contraction / spelling normalizations applied to BOTH gold and pred before WER.
# Goal: collapse differences that are scoring artifacts, not real errors.
WER_SUBSTITUTIONS = [
    (r'\(inaudible\)', '[inaudible]'),
    (r'\(unclear\)',   '[inaudible]'),
    (r'\balright\b',   'all right'),
    (r'\bgonna\b',     'going to'),
    (r'\bwanna\b',     'want to'),
    (r'\bgotta\b',     'got to'),
    (r'\bkinda\b',     'kind of'),
    (r'\bsorta\b',     'sort of'),
    (r"\b'cause\b",    'because'),
    (r'\bcause\b',     'because'),
    (r'\bokay\b',      'ok'),
    (r"\bain't\b",     'aint'),
    (r"\bmm-?hmm\b",   'mhm'),
    (r"\buh-huh\b",    'mhm'),
    (r"y'all",         'you all'),
]
_WER_SUB_RES = [(re.compile(p, re.IGNORECASE), r) for p, r in WER_SUBSTITUTIONS]


_NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12',
}

# Aggressive normalizations applied *in addition* to WER_SUBSTITUTIONS.
# Goal: collapse differences a human reviewer has already said "don't matter":
# punctuation, hyphens, titles, and segmentation-sensitive compound words.
_AGGRESSIVE_SUBS = [
    (r'\bmr\b\.?', 'mr'),
    (r'\bmrs\b\.?', 'mrs'),
    (r'\bms\b\.?', 'ms'),
    (r'\bdr\b\.?', 'dr'),
    # Hyphens and slashes to spaces so "two-thirds" == "two thirds" and
    # "2/3" splits predictably.
    (r'[-/]', ' '),
]
_AGGRESSIVE_SUB_RES = [(re.compile(p, re.IGNORECASE), r) for p, r in _AGGRESSIVE_SUBS]


def normalize_for_wer(text: str, aggressive: bool = True) -> str:
    text = text.lower()
    for rx, repl in _WER_SUB_RES:
        text = rx.sub(repl, text)
    if aggressive:
        for rx, repl in _AGGRESSIVE_SUB_RES:
            text = rx.sub(repl, text)
        # Strip ALL punctuation including apostrophes so "zero's" == "zeros"
        # and "don't" == "dont".
        text = re.sub(r"[^\w\s\[\]]", ' ', text)
        # Word->digit mapping so "eight" == "8" and "two thirds" has
        # predictable numeric form.
        toks = text.split()
        toks = [_NUMBER_WORDS.get(t, t) for t in toks]
        text = ' '.join(toks)
    else:
        # Legacy: keep apostrophes, strip other punctuation.
        text = re.sub(r"[^\w\s'\[\]]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def project_gold_speaker(sp: str) -> str:
    """TIMSS already uses T/S/SN/SS/BS, but normalize edge cases."""
    sp = sp.strip()
    if sp in {'T', 'S', 'SN', 'SS', 'BS'}:
        return sp
    if sp.upper().startswith('T'):
        return 'T'
    return 'S'


def score_wer(pred: list[Turn], gold: list[Turn],
              aggressive: bool = True) -> tuple[float, int]:
    ref = ' '.join(normalize_for_wer(t.text, aggressive=aggressive) for t in gold)
    hyp = ' '.join(normalize_for_wer(t.text, aggressive=aggressive) for t in pred)
    return wer(ref, hyp), len(ref.split())


def _speaker_role(label: str) -> str:
    """Project any speaker label down to the TIMSS role axis: T/S/SN/SS/BS.
    We use this for merging same-speaker runs; named-student variants all
    collapse to 'S' so consecutive student turns merge together."""
    lab = label.strip()
    if lab.upper() == 'T' or lab.lower().startswith('t'):
        return 'T'
    if lab in {'SN', 'S', 'SS', 'BS'}:
        return lab
    # Anything matching an identified student pattern → S.
    if re.match(r'^(S[-_(\s]|A\d|Boy|Girl)', lab):
        return 'S'
    return 'S'  # default: treat as identified student


def _merge_same_speaker_blocks(turns: list[Turn]) -> list[Turn]:
    """Collapse consecutive turns with the same speaker role into one block.
    Timestamps are taken from the first turn in the block; texts are joined
    with a space."""
    if not turns:
        return []
    blocks: list[Turn] = []
    cur_role = _speaker_role(turns[0].speaker)
    cur_texts = [turns[0].text]
    cur_t = turns[0].t_seconds
    for t in turns[1:]:
        role = _speaker_role(t.speaker)
        if role == cur_role:
            cur_texts.append(t.text)
        else:
            blocks.append(Turn(t_seconds=cur_t, speaker=cur_role,
                               text=' '.join(cur_texts)))
            cur_role = role
            cur_texts = [t.text]
            cur_t = t.t_seconds
    blocks.append(Turn(t_seconds=cur_t, speaker=cur_role,
                       text=' '.join(cur_texts)))
    return blocks


def score_wer_per_role(pred: list[Turn], gold: list[Turn]) -> dict:
    """Segmentation-insensitive, timestamp-free, speaker-aware WER.

    Algorithm:
      1. Project every speaker label to a role (T / student / other).
      2. Concatenate all gold content per role into one stream; same for pred.
      3. Compute word-level WER separately for teacher content and student
         content, then report a weighted average by gold word count.

    This is the metric that answers Jennifer's question:
      "If we remove timestamps and merge utterances by the same speaker, does
       our sequenced diarized transcript match theirs?"

    Properties:
      - Insensitive to turn division WITHIN a speaker role (all T words are
        pooled regardless of how many turns gold or pred split them into).
      - Insensitive to timestamps (never consulted).
      - Sensitive to speaker-role attribution: if gold says T and pred says S
        for the same word, that word counts as both a T-deletion and an
        S-insertion (appropriately — it measures diarization accuracy).
      - Normalization is the aggressive variant (punctuation, contractions,
        compound-word collapse).
    """
    def role_of(speaker: str) -> str:
        r = _speaker_role(speaker)
        if r == 'T':
            return 'T'
        if r in {'S', 'SN', 'SS'}:
            return 'student'
        return 'other'

    def pool(turns: list[Turn], target_role: str) -> str:
        parts = [normalize_for_wer(t.text, aggressive=True)
                 for t in turns if role_of(t.speaker) == target_role]
        return ' '.join(p for p in parts if p)

    gold_t = pool(gold, 'T')
    pred_t = pool(pred, 'T')
    gold_s = pool(gold, 'student')
    pred_s = pool(pred, 'student')

    wer_t = wer(gold_t, pred_t) if gold_t else 0.0
    wer_s = wer(gold_s, pred_s) if gold_s else 0.0
    n_t = len(gold_t.split())
    n_s = len(gold_s.split())
    total = n_t + n_s
    weighted = ((wer_t * n_t + wer_s * n_s) / total) if total else 0.0

    return {
        'wer_weighted': weighted,
        'wer_teacher': wer_t,
        'wer_student': wer_s,
        'gold_teacher_words': n_t,
        'gold_student_words': n_s,
        'pred_teacher_words': len(pred_t.split()),
        'pred_student_words': len(pred_s.split()),
    }


def score_wer_audit_adjusted(
    audit_path: Path,
    pred: list[Turn],
    gold: list[Turn],
) -> dict:
    """Compute WER using human audit decisions as the source of truth.

    Only gold turns marked 'gold' (pred was wrong) contribute errors.
    Turns marked 'pred' (equivalent) or 'inaud' (over-transcribed) are
    excluded from both numerator and denominator.
    Unreviewed turns are excluded entirely (marked as 'unscored').

    Returns a dict with headline numbers and per-verdict counts.
    """
    audit = json.loads(audit_path.read_text())
    decs = audit.get('decisions', {})

    # Index audit by (rounded seconds, gold text)
    by_key: dict[tuple[int, str], dict] = {}
    for k, v in decs.items():
        if not k.startswith('g'):
            continue
        gt = v.get('gold_t')
        gtext = v.get('gold_text')
        if gt is None or gtext is None:
            continue
        by_key[(int(gt), gtext)] = v

    n_gold_verdict = 0
    n_pred_verdict = 0
    n_inaud_verdict = 0
    n_unreviewed = 0
    gold_verdict_error_words = 0.0
    gold_verdict_total_words = 0

    for g in gold:
        key = (int(g.t_seconds), g.text)
        dec = by_key.get(key)
        verdict = (dec or {}).get('verdict')
        if verdict == 'gold':
            n_gold_verdict += 1
            # Compute edit distance for this turn against its paired pred
            pred_texts = [p['text'] for p in dec.get('pred', [])]
            ref = normalize_for_wer(g.text, aggressive=True)
            hyp = normalize_for_wer(' '.join(pred_texts), aggressive=True)
            n_words = len(ref.split())
            gold_verdict_total_words += n_words
            if ref and hyp:
                gold_verdict_error_words += wer(ref, hyp) * n_words
            else:
                gold_verdict_error_words += n_words
        elif verdict == 'pred':
            n_pred_verdict += 1
        elif verdict == 'inaud':
            n_inaud_verdict += 1
        else:
            n_unreviewed += 1

    n_reviewed = n_gold_verdict + n_pred_verdict + n_inaud_verdict
    scored_turns = n_gold_verdict + n_pred_verdict  # inaud is excluded
    return {
        'n_gold_verdict': n_gold_verdict,
        'n_pred_verdict': n_pred_verdict,
        'n_inaud_verdict': n_inaud_verdict,
        'n_unreviewed': n_unreviewed,
        'n_reviewed': n_reviewed,
        'audit_wer_on_reviewed': (
            gold_verdict_error_words / gold_verdict_total_words
            if gold_verdict_total_words else 0.0
        ),
        'audit_row_error_rate': (
            n_gold_verdict / scored_turns if scored_turns else 0.0
        ),
        'audit_error_words': gold_verdict_error_words,
        'audit_scored_words': gold_verdict_total_words,
    }


def score_speaker_labels(pred: list[Turn], gold: list[Turn]) -> tuple[float, dict]:
    """
    Naive nearest-timestamp alignment (kept for backwards compatibility).
    Prefer score_speaker_labels_content_aware for paper numbers — that one
    aligns gold and pred turns by text similarity instead of raw time
    proximity, which is much more robust when the teacher dominates the
    transcript (a brief student utterance would otherwise be nearest-aligned
    to an adjacent teacher turn, producing a spurious student→teacher error).
    """
    if not pred or not gold:
        return 0.0, {}
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    pred_by_t = sorted(pred, key=lambda t: t.t_seconds or 0)
    pred_times = [p.t_seconds or 0 for p in pred_by_t]

    import bisect
    for g in gold:
        if g.t_seconds is None:
            continue
        i = bisect.bisect_left(pred_times, g.t_seconds)
        candidates = []
        if i < len(pred_by_t): candidates.append(pred_by_t[i])
        if i > 0:              candidates.append(pred_by_t[i - 1])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda p: abs((p.t_seconds or 0) - g.t_seconds))
        gold_lab = project_gold_speaker(g.speaker)
        confusion.setdefault(gold_lab, {}).setdefault(nearest.speaker, 0)
        confusion[gold_lab][nearest.speaker] += 1
        if gold_lab == nearest.speaker:
            correct += 1

    total = sum(sum(row.values()) for row in confusion.values())
    return (correct / total) if total else 0.0, confusion


# ---------------------------------------------------------------------------
# Content-aware speaker accuracy (the fair metric)

from content_equivalence import expand as _expand_word

def _turn_content_set(text: str) -> set[str]:
    """Word-set for a turn, expanded through the equivalence table."""
    out: set[str] = set()
    for w in re.split(r'\s+', text):
        if not w:
            continue
        out |= _expand_word(w)
    return out


def _to_role(label: str) -> str:
    """Collapse any speaker label to one of: T, student, other.

    We treat TIMSS's SN/S/SS AND pred's per-student visual labels
    (S-Jenna, S-Boy-Afro, etc.) as the SAME role class 'student'. This is
    what matters for paper-grade scoring: did the system correctly identify
    speech as teacher vs student? Per-student identity is a separate
    qualitative contribution measured elsewhere.
    """
    lab = (label or '').strip()
    u = lab.upper()
    if u == 'T' or u.startswith('T'):
        return 'T'
    if u in {'S', 'SN', 'SS'}:
        return 'student'
    # v10 per-student patterns
    if re.match(r'^S[-_(\s]', lab) or re.match(r'^S$', lab):
        return 'student'
    if re.match(r'^(Boy|Girl|Student)', lab, re.IGNORECASE):
        return 'student'
    # Named teacher-style (e.g. "Ava", "Mr. X")
    if re.match(r'^(Mr|Mrs|Ms|Dr)\.?', lab, re.IGNORECASE):
        return 'T'
    return 'other'


def score_speaker_labels_content_aware(
    pred: list[Turn],
    gold: list[Turn],
    window_seconds: float = 12.0,
    min_overlap: float = 0.5,
) -> dict:
    """
    Content-aware speaker-role accuracy.

    For each gold turn with meaningful content:
      1. Find candidate pred turns within ±window_seconds.
      2. Score each candidate by F1 of word-set intersection (harmonic mean
         of gold-side recall and pred-side precision). F1 is crucial here:
         a 1-word gold "Zero." trivially hits 100% gold-recall against a
         long teacher monologue that also contains "zero", but its F1 is
         low because pred-side precision is tiny. Using F1 correctly picks
         the matching pred turn "Zero." over the monologue.
      3. Pick the candidate with the highest F1.
      4. If F1 is >= min_overlap, this gold turn is considered "matched
         to a specific pred turn" and we compare their collapsed role
         labels (T / student / other).
      5. If no candidate meets the threshold, the gold turn is counted as
         "unmatched" — excluded from speaker accuracy (we can't judge a
         speaker label if we can't even locate the corresponding pred turn).

    Returns a dict with:
      - role_accuracy: matched rows correct / matched rows total
      - confusion: role confusion matrix
      - n_matched: rows that reached min_overlap
      - n_unmatched: rows where no pred turn had enough content overlap
      - n_total_gold: total gold turns with content

    This metric is MUCH more informative than nearest-timestamp alignment
    because it only scores speaker labels on rows where we're confident we
    found the right pred turn. The unmatched count tells you how much of
    the transcript is genuinely missed content (which is a content WER issue,
    not a speaker issue).
    """
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    matched = 0
    total = 0

    # Pre-compute pred content sets for speed
    pred_with_sets = [
        (p, _turn_content_set(p.text)) for p in pred
    ]

    for g in gold:
        if g.t_seconds is None:
            continue
        gold_set = _turn_content_set(g.text)
        if not gold_set:
            continue  # skip turns with no content (pure meta)
        total += 1

        # Find candidates in time window. Score by F1 so the tightest
        # match wins over a long turn that happens to contain one of
        # gold's words.
        candidates: list[tuple[float, float, Turn]] = []  # (f1, time_delta, turn)
        for p, p_set in pred_with_sets:
            if p.t_seconds is None:
                continue
            time_delta = abs(p.t_seconds - g.t_seconds)
            if time_delta > window_seconds:
                continue
            if not p_set:
                continue
            inter = len(gold_set & p_set)
            if not inter:
                continue
            recall = inter / len(gold_set)
            precision = inter / len(p_set)
            f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
            if f1 >= min_overlap:
                candidates.append((f1, time_delta, p))

        if not candidates:
            continue  # unmatched — don't count

        # Classroom call-and-response is pervasive: teacher and student often
        # say the same short phrase ("Okay", "Six", "Yeah", etc.). When two
        # candidates have near-tied F1 and one matches gold's role while the
        # other doesn't, the role-matching one is the correct attribution —
        # v10 really did produce a separate student turn AND a separate
        # teacher turn with similar content, and we should credit it with
        # the correct speaker ID.
        candidates.sort(key=lambda c: (-c[0], c[1]))  # best F1 first
        f1_max = candidates[0][0]
        gold_role = _to_role(g.speaker)
        near_tie = [c for c in candidates if c[0] >= f1_max - 0.05]
        role_match = next((c for c in near_tie if _to_role(c[2].speaker) == gold_role), None)
        best_pred = (role_match or candidates[0])[2]

        matched += 1
        gold_role = _to_role(g.speaker)
        pred_role = _to_role(best_pred.speaker)
        confusion.setdefault(gold_role, {}).setdefault(pred_role, 0)
        confusion[gold_role][pred_role] += 1
        if gold_role == pred_role:
            correct += 1

    return {
        'role_accuracy': (correct / matched) if matched else 0.0,
        'confusion': confusion,
        'n_matched': matched,
        'n_unmatched': total - matched,
        'n_total_gold': total,
        'window_seconds': window_seconds,
        'min_overlap': min_overlap,
    }


_STOP_WORDS_FOR_DROP = {
    'the', 'a', 'an', 'is', 'it', 'of', 'to', 'and', 'or', 'in', 'on', 'at',
    'i', 'you', 'we', 'they', 'he', 'she', 'me', 'my', 'your', 'that', 'this',
    'so', 'but', 'for', 'as', 'with', 'be', 'do', 'have', 'has', 'had', 'was',
    'were', 'are', 'am', "'s", "'re", "'m", "'ll", "'ve", "'d", 'mhm', 'uh',
}


def find_dropped_turns(
    gold: list[Turn],
    pred: list[Turn],
    window_seconds: float = 12.0,
    min_recall: float = 0.5,
) -> list[DroppedTurn]:
    """
    For each gold turn, find the pred turn within +/- window_seconds whose
    word set best COVERS the gold turn's words (gold-side recall). A gold turn
    is flagged as dropped if no pred turn in the window covers >= min_recall
    of its content words.

    We use recall (not Jaccard) so that pred turns which MERGE multiple gold
    turns into one longer line still get credit for capturing each gold turn.
    Stop-word-only gold turns ("yeah", "okay") are scored on the raw word set
    since recall on a 1-word turn is binary.
    """
    dropped: list[DroppedTurn] = []
    for g in gold:
        if g.t_seconds is None:
            continue
        gold_words_all = normalize_for_wer(g.text).split()
        if not gold_words_all:
            continue
        gold_set = set(gold_words_all)
        # For longer turns, score on content words; for very short turns
        # ("yeah", "okay") keep all words so backchannels still get scored.
        gold_content = {w for w in gold_set if w not in _STOP_WORDS_FOR_DROP}
        scoring_set = gold_content if len(gold_content) >= 2 else gold_set
        if not scoring_set:
            continue

        candidates = [p for p in pred
                      if p.t_seconds is not None
                      and abs((p.t_seconds or 0) - g.t_seconds) <= window_seconds]
        best_recall = 0.0
        best_text: str | None = None
        for p in candidates:
            pred_words = set(normalize_for_wer(p.text).split())
            if not pred_words:
                continue
            inter = len(scoring_set & pred_words)
            recall = inter / len(scoring_set)
            if recall > best_recall:
                best_recall = recall
                best_text = p.text
        if best_recall < min_recall:
            dropped.append(DroppedTurn(
                t_seconds=g.t_seconds,
                speaker=g.speaker,
                text=g.text,
                best_pred_overlap=round(best_recall, 3),
                best_pred_text=best_text,
            ))
    return dropped


# ---------------------------------------------------------------------------
# Pipeline orchestration

def run_yt_dlp(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    cmd = ['yt-dlp', '-f', 'b[ext=mp4]', '-o', str(out_path), url]
    print(f"[yt-dlp] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return out_path


def run_v10_pipeline(video: Path, workdir: Path) -> Path:
    """
    Returns the path to the *_complete_transcript.txt file produced by v10.
    NOTE: the v10 script's CLI flags should be confirmed before first run;
    we shell out and let it create its own dated output dir under workdir.
    """
    pipeline = Path('/Users/jenniferkleiman/Documents/COMS/video_transcription_pipeline_v10.py')
    # v10 uses subcommands. `process` is single-video; common args go BEFORE
    # the subcommand in argparse subparser style — but here add_common_args is
    # attached to each subparser, so they go AFTER 'process'.
    speakers_stub = Path(__file__).parent / 'timss_speakers_stub.json'
    cmd = [
        sys.executable, str(pipeline), 'process', str(video),
        '-o', str(workdir),
        '--speakers', str(speakers_stub),
        '--no-confirm',
        '--single-output',
        '--thinking-budget', '4096',
        # Common args (from add_common_args), validated config in MEMORY.md:
        '--model', 'gemini-3-flash-preview',
        '--resolution', 'HIGH',
        '--fps', '2',
        '--chunk-minutes', '1.0',
        '--overlap', '15',
    ]
    # NOTE: v10 has no --temperature flag exposed; the in-code default is used.
    # Speaker manifest is the generic TIMSS stub (T/S/SN/SS/BS) — same coding
    # scheme as TIMSS gold transcripts, so projection is a near-identity map.
    print(f"[v10] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # v10 in --single-output mode writes <video_stem>_transcript.txt directly
    # under -o. Older multi-output mode wrote *_complete_transcript.txt under
    # a dated subdir. Try both.
    candidates = sorted(workdir.glob(f'{video.stem}_transcript.txt'))
    if not candidates:
        candidates = sorted(workdir.glob('**/*_complete_transcript.txt'))
    if not candidates:
        candidates = sorted(workdir.glob('**/*_transcript.txt'))
    if not candidates:
        raise FileNotFoundError(f"no v10 transcript file under {workdir}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lesson', required=True, help='e.g. US1')
    ap.add_argument('--youtube', required=True)
    ap.add_argument('--gold', required=True, type=Path)
    ap.add_argument('--workdir', default='./benchmark_runs', type=Path)
    ap.add_argument('--config', default=Path(__file__).parent / 'projection_config.yaml', type=Path)
    ap.add_argument('--clip-seconds', type=int, default=None,
                    help='If set, ffmpeg-trim the downloaded video to the first N seconds before running v10. Useful for cheap smoke tests.')
    ap.add_argument('--audit', type=Path, default=None,
                    help='Optional review-tool audit JSON for audit-adjusted WER.')
    ap.add_argument('--skip-pipeline', action='store_true',
                    help='reuse v10 output already in workdir')
    ap.add_argument('--v10-transcript', type=Path,
                    help='explicit path to a v10 *_complete_transcript.txt (skips download+pipeline)')
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.workdir.mkdir(parents=True, exist_ok=True)

    if args.v10_transcript:
        v10_txt = args.v10_transcript
    elif args.skip_pipeline:
        cands = sorted(args.workdir.glob('**/*_complete_transcript.txt'))
        if not cands:
            sys.exit("--skip-pipeline set but no v10 transcript found in workdir")
        v10_txt = cands[-1]
    else:
        video = run_yt_dlp(args.youtube, args.workdir / f'{args.lesson}.mp4')
        if args.clip_seconds:
            clipped = args.workdir / f'{args.lesson}_first{args.clip_seconds}s.mp4'
            if not clipped.exists():
                print(f"[ffmpeg] trimming to first {args.clip_seconds}s -> {clipped}", flush=True)
                subprocess.run(
                    ['ffmpeg', '-y', '-loglevel', 'error',
                     '-i', str(video), '-t', str(args.clip_seconds),
                     '-c', 'copy', str(clipped)],
                    check=True,
                )
            video = clipped
        v10_txt = run_v10_pipeline(video, args.workdir)

    gold_turns = parse_gold(args.gold)
    pred_turns, visuals, warnings = parse_v10(v10_txt, cfg)

    if args.clip_seconds:
        gold_turns = [t for t in gold_turns
                      if t.t_seconds is not None and t.t_seconds < args.clip_seconds]
        pred_turns = [t for t in pred_turns
                      if t.t_seconds is not None and t.t_seconds < args.clip_seconds]
        visuals = [v for v in visuals if v.t_seconds < args.clip_seconds]

    wer_score, n_ref_words = score_wer(pred_turns, gold_turns, aggressive=True)
    wer_strict_score, _ = score_wer(pred_turns, gold_turns, aggressive=False)
    wer_role_stats = score_wer_per_role(pred_turns, gold_turns)
    content_spk_stats = score_speaker_labels_content_aware(pred_turns, gold_turns)

    # Content-equivalence score (the primary number for the paper).
    # Mirrors the HTML review tool's matching rules: time-windowed word-set
    # membership with variant/contraction/number/compound expansion and
    # inaudible-marker ignore.
    eq_gold = [EqTurn(t_seconds=t.t_seconds, speaker=t.speaker, text=t.text)
               for t in gold_turns]
    eq_pred = [EqTurn(t_seconds=t.t_seconds, speaker=t.speaker, text=t.text)
               for t in pred_turns]
    eq = score_content_equivalence(eq_gold, eq_pred)
    content_eq_dict = {
        'window_seconds': eq.window_seconds,
        'n_gold_words': eq.n_gold_words,
        'n_pred_words': eq.n_pred_words,
        'n_gold_matched': eq.n_gold_matched,
        'n_pred_matched': eq.n_pred_matched,
        'recall': round(eq.recall, 4),
        'precision': round(eq.precision, 4),
        'f1': round(eq.f1, 4),
        'wer_equivalent': round(eq.wer_equivalent, 4),
        'gold_unmatched_count': len(eq.gold_unmatched_words),
        'pred_unmatched_count': len(eq.pred_unmatched_words),
        # Sample of unmatched words for qualitative inspection (cap at 40).
        'gold_unmatched_sample': [
            {'t': round(t, 1), 'w': w}
            for t, w in eq.gold_unmatched_words[:40]
        ],
        'pred_unmatched_sample': [
            {'t': round(t, 1), 'w': w}
            for t, w in eq.pred_unmatched_words[:40]
        ],
    }
    spk_acc, confusion = score_speaker_labels(pred_turns, gold_turns)
    dropped = find_dropped_turns(gold_turns, pred_turns)
    n_gold_words = sum(len(normalize_for_wer(t.text).split()) for t in gold_turns)
    n_pred_words = sum(len(normalize_for_wer(t.text).split()) for t in pred_turns)
    audit_stats = None
    if args.audit:
        audit_stats = score_wer_audit_adjusted(args.audit, pred_turns, gold_turns)

    duration_min = max(
        (gold_turns[-1].t_seconds or 0) / 60.0 if gold_turns else 0,
        (pred_turns[-1].t_seconds or 0) / 60.0 if pred_turns else 0,
        1.0,
    )
    report = BenchmarkReport(
        lesson=args.lesson,
        youtube_url=args.youtube,
        gold_path=str(args.gold),
        v10_output_path=str(v10_txt),
        n_gold_turns=len(gold_turns),
        n_pred_turns=len(pred_turns),
        n_gold_words=n_gold_words,
        n_pred_words=n_pred_words,
        n_visual_events=len(visuals),
        visual_events_per_minute=round(len(visuals) / duration_min, 2),
        wer=round(wer_score, 4),
        wer_strict=round(wer_strict_score, 4),
        wer_per_role={k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in wer_role_stats.items()},
        content_equivalence=content_eq_dict,
        speaker_role_content_aware={
            'role_accuracy': round(content_spk_stats['role_accuracy'], 4),
            'n_matched': content_spk_stats['n_matched'],
            'n_unmatched': content_spk_stats['n_unmatched'],
            'n_total_gold': content_spk_stats['n_total_gold'],
            'confusion': content_spk_stats['confusion'],
            'window_seconds': content_spk_stats['window_seconds'],
            'min_overlap': content_spk_stats['min_overlap'],
        },
        wer_n_words_ref=n_ref_words,
        audit=audit_stats,
        speaker_label_accuracy=round(spk_acc, 4),
        speaker_confusion=confusion,
        n_dropped_turns=len(dropped),
        dropped_turns=[asdict(d) for d in dropped],
        projection_warnings=sorted(set(warnings)),
    )

    out_json = v10_txt.parent / f'benchmark_{args.lesson}.json'
    out_json.write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2))
    print(f"\nReport written to {out_json}")


if __name__ == '__main__':
    main()
