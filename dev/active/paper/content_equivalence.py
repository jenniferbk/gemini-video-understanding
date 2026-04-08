"""
Content-equivalence matching rules shared by the benchmark scorer and the
HTML review tool. This is the single source of truth for "are these two
transcripts saying the same thing, modulo segmentation, timing jitter, and
normalization quirks that don't matter for educational research use".

Design principles:
  - Time-windowed, not turn-aligned. Arbitrary turn division must not affect
    matching; a word counts as matched if it appears anywhere on the other
    side within ±WINDOW seconds.
  - Equivalence classes, not exact strings. "alright" ↔ "all right",
    "you're" ↔ "you are", "8" ↔ "eight", "Mr." ↔ "Mister", "can" ↔ "could",
    "XY" ↔ "X Y", "Y-intercept" ↔ "Y intercept", "this" ↔ "these", etc.
  - Inaudible/meta markers are ignored on both sides. "(inaudible)" and
    silence are treated as equivalent.
  - All matching is done on lowercased, punctuation-stripped words.

The VARIANTS table and expand() function are ported from the review tool so
the review and the scorer always agree on what counts as an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Default time window for "nearby on the other side" matching. Generous
# enough to tolerate pipeline timestamp jitter while tight enough that a
# cross-minute repetition doesn't produce spurious matches.
DEFAULT_WINDOW_SECONDS = 12.0


# Meta / transcription-marker words that represent silence or non-transcribed
# audio. Treated as invisible on both sides: never contribute to the word
# set, never highlighted, never counted as errors.
IGNORE_WORDS: frozenset[str] = frozenset({
    'inaudible', 'unclear', 'unintelligible', 'crosstalk',
    'overlap', 'pause', 'silence', 'laughter', 'noise',
})


# Bidirectional equivalence classes. Every form in the tuple is considered
# equivalent to every other form. Expand() unions these on token lookup so a
# match in either direction succeeds.
_EQUIVALENCE_CLASSES: tuple[tuple[str, ...], ...] = (
    # Compounds
    ("alright", "all", "right"),
    ("anyway", "any", "way"),
    ("twothirds", "two", "thirds", "two-thirds"),
    ("onehalf", "one", "half"),
    ("threequarters", "three", "quarters"),
    # Colloquial
    ("gonna", "going", "to"),
    ("wanna", "want", "to"),
    ("gotta", "got", "to"),
    ("kinda", "kind", "of"),
    ("sorta", "sort", "of"),
    ("dunno", "don't", "know", "dont"),
    ("cause", "because", "'cause"),
    ("y'all", "you", "all", "yall"),
    # Contractions
    ("youre", "you", "are", "you're"),
    ("theyre", "they", "are", "they're"),
    ("were", "we", "are", "we're"),
    ("dont", "do", "not", "don't"),
    ("didnt", "did", "not", "didn't"),
    ("cant", "can", "not", "can't", "cannot"),
    ("wont", "will", "not", "won't"),
    ("isnt", "is", "not", "isn't"),
    ("arent", "are", "not", "aren't"),
    ("wasnt", "was", "not", "wasn't"),
    ("wouldnt", "would", "not", "wouldn't"),
    ("couldnt", "could", "not", "couldn't"),
    ("shouldnt", "should", "not", "shouldn't"),
    ("lets", "let", "us", "let's"),
    ("its", "it", "is", "it's"),
    ("thats", "that", "is", "that's"),
    ("whats", "what", "is", "what's"),
    ("wheres", "where", "is", "where's"),
    ("hows", "how", "is", "how's"),
    ("heres", "here", "is", "here's"),
    ("theres", "there", "is", "there's"),
    ("im", "i", "am", "i'm"),
    ("ive", "i", "have", "i've"),
    ("ill", "i", "will", "i'll"),
    ("id", "i", "would", "i'd"),
    ("hes", "he", "is", "he's"),
    ("shes", "she", "is", "she's"),
    ("youve", "you", "have", "you've"),
    ("youll", "you", "will", "you'll"),
    ("youd", "you", "would", "you'd"),
    ("weve", "we", "have", "we've"),
    ("theyve", "they", "have", "they've"),
    ("theyll", "they", "will", "they'll"),
    ("thatll", "that", "will", "that'll"),
    ("thisll", "this", "will", "this'll"),
    ("itll", "it", "will", "it'll"),
    ("therell", "there", "will", "there'll"),
    ("wholl", "who", "will", "who'll"),
    ("whatll", "what", "will", "what'll"),
    ("whod", "who", "would", "who'd"),
    ("whos", "who", "is", "who's"),
    # Affirmatives / backchannels
    ("okay", "ok", "k"),
    ("mhm", "mm-hmm", "mmhmm", "uh-huh", "uhhuh", "mm",
     "yeah", "yep", "yes"),
    # Titles
    ("mr", "mister"),
    ("mrs", "missus"),
    ("ms", "miss"),
    # Numbers
    ("zero", "0"),
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
    ("ten", "10"),
    # Modal verb slack
    ("can", "could"),
    ("would", "could", "will"),
    # Demonstrative slack
    ("this", "these", "that", "those"),
)


def _build_variants() -> dict[str, frozenset[str]]:
    v: dict[str, set[str]] = {}
    for cls in _EQUIVALENCE_CLASSES:
        s = set(cls)
        for f in cls:
            v.setdefault(f, set()).update(s)
    return {k: frozenset(val) for k, val in v.items()}


VARIANTS: dict[str, frozenset[str]] = _build_variants()


def norm_word(w: str) -> str:
    """Lowercase and strip everything except word chars and apostrophes."""
    return re.sub(r"[^\w']", '', w).lower()


def expand(w: str) -> frozenset[str]:
    """Return the set of equivalent forms for a raw word token.

    Applies: lowercasing, punctuation stripping, hyphen splitting (so
    "Y-intercept" becomes {y, intercept, yintercept}), short all-alpha char
    splitting (so "XY" becomes {x, y, xy}), meta/ignore filtering, and
    equivalence-class expansion.

    Returns empty set for tokens that carry no content (empty, meta marker).
    """
    raw = w.lower()
    results: set[str] = set()

    # Hyphenated compounds → both the joined form and each piece.
    if '-' in raw:
        for piece in raw.split('-'):
            p = re.sub(r"[^\w']", '', piece)
            if p:
                results.add(p)

    nw = norm_word(w)
    if nw:
        results.add(nw)

    # Short all-alpha words → also yield individual characters (handles
    # "XY" ↔ "X Y").
    if nw and len(nw) <= 3 and nw.isalpha():
        for ch in nw:
            results.add(ch)

    # Drop meta/inaudible markers entirely.
    results -= set(IGNORE_WORDS)

    # Expand each surviving form through the equivalence table.
    for form in list(results):
        key = form.lstrip("'")
        forms = VARIANTS.get(key)
        if forms:
            results |= forms

    return frozenset(results)


# ---------------------------------------------------------------------------
# Scoring

@dataclass
class Turn:
    t_seconds: float
    speaker: str
    text: str


@dataclass
class EquivalenceScore:
    """Headline content-equivalence stats.

    A gold word is MATCHED if any of its expansion forms appears in the pred
    word set within ±window seconds. Similarly for pred→gold.

    The intuition: "would a human reviewer who allows segmentation, timing,
    and normalization slack say this word is present on both sides?"
    """
    window_seconds: float
    n_gold_words: int           # words that survive normalization/ignore
    n_pred_words: int
    n_gold_matched: int         # gold words found in nearby pred
    n_pred_matched: int         # pred words found in nearby gold
    gold_unmatched_words: list[tuple[float, str]]   # (t_seconds, word) — true deletions
    pred_unmatched_words: list[tuple[float, str]]   # (t_seconds, word) — true insertions

    @property
    def recall(self) -> float:
        """Fraction of gold content captured by pred."""
        return (self.n_gold_matched / self.n_gold_words) if self.n_gold_words else 1.0

    @property
    def precision(self) -> float:
        """Fraction of pred content that was actually in gold."""
        return (self.n_pred_matched / self.n_pred_words) if self.n_pred_words else 1.0

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        return (2 * r * p / (r + p)) if (r + p) else 0.0

    @property
    def wer_equivalent(self) -> float:
        """Content-equivalence WER: (unmatched_gold + unmatched_pred) / gold words.

        Gold unmatched → deletions; pred unmatched → insertions. No separate
        substitution count because our matching is word-set, not alignment.
        """
        if not self.n_gold_words:
            return 0.0
        dels = self.n_gold_words - self.n_gold_matched
        ins = self.n_pred_words - self.n_pred_matched
        return (dels + ins) / self.n_gold_words


def _windowed_forms(turns: list[Turn], t_center: float,
                    window: float) -> frozenset[str]:
    """Union of expand(w) for every word in turns within ±window of t_center."""
    out: set[str] = set()
    for t in turns:
        if abs(t.t_seconds - t_center) > window:
            continue
        for w in re.findall(r"\S+", t.text):
            out |= expand(w)
    return frozenset(out)


def score_content_equivalence(
    gold: list[Turn],
    pred: list[Turn],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> EquivalenceScore:
    """Compute content-equivalence recall/precision/WER between two transcripts.

    For each gold turn, each of its content-bearing words is checked for
    presence in the pred word set within ±window seconds of the gold turn's
    timestamp. Symmetric check for pred→gold.

    This is the primary metric that mirrors the visual review tool's
    highlighting rules. The `wer_equivalent` field is the headline number.
    """
    n_gold_words = 0
    n_gold_matched = 0
    gold_unmatched: list[tuple[float, str]] = []

    for t in gold:
        pred_set = _windowed_forms(pred, t.t_seconds, window_seconds)
        for w in re.findall(r"\S+", t.text):
            forms = expand(w)
            if not forms:
                continue  # meta / ignore / empty
            n_gold_words += 1
            if forms & pred_set:
                n_gold_matched += 1
            else:
                gold_unmatched.append((t.t_seconds, w))

    n_pred_words = 0
    n_pred_matched = 0
    pred_unmatched: list[tuple[float, str]] = []

    for t in pred:
        gold_set = _windowed_forms(gold, t.t_seconds, window_seconds)
        for w in re.findall(r"\S+", t.text):
            forms = expand(w)
            if not forms:
                continue
            n_pred_words += 1
            if forms & gold_set:
                n_pred_matched += 1
            else:
                pred_unmatched.append((t.t_seconds, w))

    return EquivalenceScore(
        window_seconds=window_seconds,
        n_gold_words=n_gold_words,
        n_pred_words=n_pred_words,
        n_gold_matched=n_gold_matched,
        n_pred_matched=n_pred_matched,
        gold_unmatched_words=gold_unmatched,
        pred_unmatched_words=pred_unmatched,
    )
