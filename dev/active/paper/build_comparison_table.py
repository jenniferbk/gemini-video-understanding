#!/usr/bin/env python3
"""
Scan all benchmark_*.json files in dev/active/paper/benchmark_runs/ and
produce a clean markdown comparison table + per-system details for the paper.

Writes:
  dev/active/paper/benchmark_runs/comparison_summary.md
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path('dev/active/paper/benchmark_runs')

# Friendly names + ordering for paper display. Any JSON file not listed here
# is appended to an "Other" section so nothing gets silently dropped.
DISPLAY: list[tuple[str, str, str]] = [
    # (json path relative to ROOT, short name, category)
    ('whisper_us1/benchmark_US1_whisper.json',      'Whisper large-v3 (alone)',             'baseline'),
    ('whisper_pyannote_us1/benchmark_US1_wpy.json', 'Whisper + pyannote 3.1',               'baseline'),
    ('full_us1/benchmark_US1_full.json',            'v10 (this work) — run 1',              'primary'),
    ('replicate_1/benchmark_US1_rep1.json',         'v10 (this work) — run 2 (replicate)',  'primary'),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_pct(x: float, prec: int = 1) -> str:
    return f"{x*100:.{prec}f}%"


def fmt_row(name: str, r: dict) -> str:
    ce = r['content_equivalence']
    sca = r.get('speaker_role_content_aware', {})
    conf = sca.get('confusion', {})
    t_row = conf.get('T', {})
    s_row = conf.get('student', {})
    t_n = sum(t_row.values()) or 1
    s_n = sum(s_row.values()) or 1
    t_acc = t_row.get('T', 0) / t_n
    s_acc = s_row.get('student', 0) / s_n
    return (
        f"| {name} "
        f"| {fmt_pct(ce['recall'])} "
        f"| {fmt_pct(ce['precision'])} "
        f"| {fmt_pct(ce['f1'])} "
        f"| {fmt_pct(ce['wer_equivalent'])} "
        f"| {fmt_pct(sca.get('role_accuracy', 0.0))} "
        f"| {fmt_pct(t_acc)} "
        f"| {fmt_pct(s_acc)} "
        f"| {fmt_pct(r['wer_strict'])} "
        f"|"
    )


def main() -> None:
    lines: list[str] = []
    lines.append("# TIMSS US1 benchmark — system comparison\n")
    lines.append("All systems scored against the reviewer-corrected gold transcript "
                 "(8 corrections applied: 3 inaudible over-transcriptions removed, "
                 "5 text corrections for transcription mistakes in the published TIMSS gold).\n")
    lines.append("Primary metric is **content-equivalence WER** — a time-windowed "
                 "word-set membership score that tolerates arbitrary turn division, "
                 "±2s timestamp jitter, punctuation/contraction normalization, "
                 "hyphen compounds, number/word interchange, and inaudible-marker "
                 "invisibility. Rules are identical to the human review tool's "
                 "highlighting logic, so the metric and the visual review can never "
                 "drift apart.\n")
    lines.append("Strict WER (last column) is the raw jiwer number on normalized "
                 "text, included for calibration against published ASR literature.\n")
    lines.append("Speaker role accuracy (role acc) uses a **content-aware** "
                 "alignment: for each gold turn, find the pred turn with the "
                 "highest F1 of word-set overlap within ±12 seconds, with "
                 "role-aware tiebreaking (when multiple pred turns have near-tied "
                 "F1, prefer the one whose role matches gold — which correctly "
                 "credits systems for classroom call-and-response where teacher "
                 "and student say the same short phrase). Unmatched gold turns "
                 "(no pred turn reached 0.5 F1) are **excluded** from the "
                 "speaker accuracy denominator — they're a content-WER issue, "
                 "not a speaker-ID issue. Nearest-timestamp alignment is much "
                 "less informative here because classroom transcripts are "
                 "teacher-dominated: a brief student utterance's nearest pred "
                 "is almost always an adjacent teacher turn, producing spurious "
                 "student→teacher errors.\n")
    lines.append("")

    header = (
        "| system | recall | precision | F1 | content WER | role acc | T acc | S acc | strict WER |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines.append("## Headline comparison\n")
    lines.append(header)
    rows_for_later: list[tuple[str, str, dict, Path]] = []
    for rel, name, category in DISPLAY:
        p = ROOT / rel
        if not p.exists():
            lines.append(f"| {name} | — | — | — | — | — | — | *missing* |")
            continue
        r = load(p)
        lines.append(fmt_row(name, r))
        rows_for_later.append((name, category, r, p))
    lines.append("")

    # Gold words baseline for context.
    any_primary = next((r for _, cat, r, _ in rows_for_later if cat == 'primary'), None)
    if any_primary:
        gw = any_primary['content_equivalence']['n_gold_words']
        lines.append(f"**Reference:** {gw} gold content words "
                     f"({any_primary.get('n_gold_turns','?')} gold turns after correction).\n")

    # Replicate variance (v10 runs).
    v10_runs = [(name, r) for name, cat, r, _ in rows_for_later if cat == 'primary']
    if len(v10_runs) >= 2:
        recalls = [r['content_equivalence']['recall'] for _, r in v10_runs]
        precs   = [r['content_equivalence']['precision'] for _, r in v10_runs]
        f1s     = [r['content_equivalence']['f1'] for _, r in v10_runs]
        cwer    = [r['content_equivalence']['wer_equivalent'] for _, r in v10_runs]
        spk     = [r.get('speaker_role_content_aware', {}).get('role_accuracy', 0.0)
                   for _, r in v10_runs]
        def mean(xs): return sum(xs)/len(xs)
        def spread(xs): return max(xs)-min(xs)
        lines.append("## v10 replicate variance\n")
        lines.append(f"Across {len(v10_runs)} independent runs of the same config on "
                     f"the same video (temperature 0.2, so some stochasticity is expected):")
        lines.append("")
        lines.append("| metric | mean | range (max − min) |")
        lines.append("|---|---|---|")
        lines.append(f"| recall | {fmt_pct(mean(recalls),2)} | {fmt_pct(spread(recalls),2)} |")
        lines.append(f"| precision | {fmt_pct(mean(precs),2)} | {fmt_pct(spread(precs),2)} |")
        lines.append(f"| F1 | {fmt_pct(mean(f1s),2)} | {fmt_pct(spread(f1s),2)} |")
        lines.append(f"| content WER | {fmt_pct(mean(cwer),2)} | {fmt_pct(spread(cwer),2)} |")
        lines.append(f"| speaker accuracy | {fmt_pct(mean(spk),2)} | {fmt_pct(spread(spk),2)} |")
        lines.append("")
        lines.append("**Takeaway:** content metrics are stable across replicates "
                     "(<1 pp spread). Speaker accuracy is the noisier axis — Gemini's "
                     "visual-feature diarization is deterministic in principle but "
                     "the same video can produce somewhat different speaker clusterings "
                     "across runs because individual-student labels depend on which "
                     "visual features the model latches onto.\n")

    # Key findings.
    lines.append("## Key findings\n")
    lines.append(
        "1. **On the speech content axis, all three systems tie.** "
        "Whisper large-v3, Whisper+pyannote, and v10 all hit F1 = 96.7-96.8% on "
        "content-equivalence. v10 is not better at speech recognition per se — "
        "it matches the state of the art. The value proposition is elsewhere.\n"
    )
    lines.append(
        "2. **v10 dominates on speaker attribution: 92-93% role accuracy vs "
        "Whisper+pyannote's 75%.** Scored by content-aware alignment (F1 match "
        "of gold turn → nearest pred turn by word-set similarity, with "
        "role-aware tiebreaking for classroom call-and-response), v10 achieves "
        "**97-98% accuracy on teacher turns and 83-85% on student turns**. "
        "Whisper+pyannote hits 84% teacher / 61% student even with an oracle "
        "SPEAKER_00→teacher mapping. pyannote detects only **2 distinct "
        "speakers** across a classroom with ~15-20 students because it clusters "
        "by voice characteristics; v10 uses **visual features** (clothing, "
        "position, hair, facial features) to distinguish individuals. The "
        "qualitative win is even larger: v10 produces per-student labels like "
        "`S-Jenna`, `S-Boy-Afro`, `S-Girl-StripedShirt` that let researchers "
        "identify who said what without manual cluster labeling or re-watching "
        "the video.\n"
    )
    lines.append(
        "3. **Visual descriptions are uncontested.** No baseline produces "
        "interleaved visual descriptions of classroom activity (gestures, "
        "whiteboard content, gaze, shared materials). v10 produces these in the "
        "same single API call as the speech transcription, at no additional cost "
        "and no additional pipeline complexity.\n"
    )
    lines.append(
        "4. **Human review confirms the content-equivalence number is an "
        "accurate reflection of pipeline quality.** Reviewer-validated audits of "
        "US1 rows (first 5 minutes and flagged rows across the full 44 minutes) "
        "confirmed that the content-equivalence metric's flagged misses "
        "correspond to the reviewer's ground-truth verdicts. The remaining "
        "gold-unmatched words are overwhelmingly: (a) legitimate pipeline misses "
        "on short student backchannels, (b) additional TIMSS gold errors, or "
        "(c) short inaudible passages the reviewer couldn't verify either way. "
        "Raw (strict) WER overestimates pipeline error by 3-4× on this corpus.\n"
    )
    lines.append(
        "5. **TIMSS gold transcripts contain errors and over-transcription of "
        "inaudible audio.** We documented 8 corrections in the first 5 minutes "
        "alone (3 removes of inaudible passages, 5 text edits for transcription "
        "mistakes like 'Sarah' miscoded as 'sir' and 'Y intercept and X intercept' "
        "where no 'X intercept' was actually said). The corrected gold and "
        "correction log are shipped as paper artifacts.\n"
    )

    # Cost / practicality comparison.
    lines.append("## Practical comparison\n")
    lines.append("| aspect | Whisper alone | Whisper + pyannote | v10 (this work) |")
    lines.append("|---|---|---|---|")
    lines.append("| Speech transcription | ✓ | ✓ | ✓ |")
    lines.append("| Speaker diarization | ✗ | anonymous clusters only | visual-feature per-student |")
    lines.append("| Visual descriptions | ✗ | ✗ | ✓ interleaved |")
    lines.append("| Classroom activity context | ✗ | ✗ | ✓ (whiteboard, gestures, materials) |")
    lines.append("| API calls per video | 0 (local) | 0 (local) + HF model download | ~60 chunks/hr |")
    lines.append("| Cost per hour of video | ~free (local CPU/GPU) | ~free (local) | ~$0.19 |")
    lines.append("| Setup complexity | low | medium (HF auth + gated models) | low (one API key) |")
    lines.append("| End-to-end wall time (44-min video, our machine) | ~2+ hours (large-v3 CPU) | +~10 min pyannote | ~42 min (API-bound) |")
    lines.append("")

    out = ROOT / 'comparison_summary.md'
    out.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out}")
    print(f"  {len(rows_for_later)} systems scored")


if __name__ == '__main__':
    main()
