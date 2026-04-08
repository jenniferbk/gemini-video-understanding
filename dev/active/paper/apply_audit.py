#!/usr/bin/env python3
"""
Apply a review-tool audit JSON to an original TIMSS gold transcript and
produce a corrected version + a human-readable change log.

Audit verdicts:
  gold   → keep gold turn unchanged (pred was wrong)
  pred   → keep gold turn unchanged (pred was equivalent; metric artifact only)
  inaud  → REMOVE gold turn (gold over-transcribed inaudible audio)
  both   → flag for manual fix (kept in corrected gold with a warning comment)
  (none) → undecided; keep gold turn unchanged

Usage:
    python apply_audit.py \
        --gold "Math+transcripts+as+txt+files+(1)/Math US1 transcript.txt" \
        --audit ~/Downloads/review_US1_*.json \
        --out-gold dev/active/paper/benchmark_runs/US1_gold_corrected.txt \
        --out-log dev/active/paper/benchmark_runs/US1_audit_log.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


GOLD_LINE_RE = re.compile(r'^(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)$')


@dataclass
class GoldTurn:
    raw_line: str               # original file line, preserved for output
    timestamp: str              # "HH:MM:SS"
    t_seconds: float
    speaker: str
    text: str


def parse_gold(path: Path) -> list[GoldTurn]:
    turns: list[GoldTurn] = []
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.rstrip('\n')
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split('\t', 2)
        if len(parts) == 3 and re.match(r'^\d{2}:\d{2}:\d{2}$', parts[0]):
            ts, sp, txt = parts
        else:
            m = GOLD_LINE_RE.match(stripped)
            if not m:
                continue
            ts, sp, txt = m.groups()
        h, m_, s = [int(x) for x in ts.split(':')]
        turns.append(GoldTurn(
            raw_line=line,
            timestamp=ts,
            t_seconds=h*3600 + m_*60 + s,
            speaker=sp.strip(),
            text=txt.strip(),
        ))
    return turns


def _load_corrections(path: Path) -> dict[tuple[str, str], dict]:
    """Load gold corrections YAML into a lookup keyed by (timestamp, original)."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml")
    data = yaml.safe_load(path.read_text())
    out: dict[tuple[str, str], dict] = {}
    for entry in data.get('corrections', []):
        key = (entry['timestamp'], entry['original'])
        out[key] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True, type=Path)
    ap.add_argument('--audit', required=False, type=Path)
    ap.add_argument('--corrections', type=Path, default=None,
                    help='YAML file of explicit gold corrections (edit/remove).')
    ap.add_argument('--out-gold', required=True, type=Path)
    ap.add_argument('--out-log', required=True, type=Path)
    args = ap.parse_args()

    gold = parse_gold(args.gold)
    decs = {}
    if args.audit:
        audit = json.loads(args.audit.read_text())
        decs = audit.get('decisions', {})

    corrections = _load_corrections(args.corrections) if args.corrections else {}

    # Build a lookup by gold text + timestamp so we can match audit entries
    # to gold turns without relying on the "g<idx>" key (which depends on the
    # review tool's ordering that includes clip_seconds filtering).
    decision_by_text: dict[tuple[int, str], dict] = {}
    for k, v in decs.items():
        if not k.startswith('g'):
            continue
        gt = v.get('gold_t')
        gtext = v.get('gold_text')
        if gt is None or gtext is None:
            continue
        decision_by_text[(int(gt), gtext)] = v

    corrected: list[GoldTurn] = []
    log_entries: list[str] = []
    n_corrections_applied = 0
    n_corrections_unmatched = []

    for t in gold:
        key = (int(t.t_seconds), t.text)
        dec = decision_by_text.get(key)
        verdict = (dec or {}).get('verdict')
        note = (dec or {}).get('note', '')

        # Explicit corrections (from the YAML config) take precedence over
        # audit verdicts. Match by exact (timestamp, original text).
        corr_key = (t.timestamp, t.text)
        corr = corrections.get(corr_key)
        if corr:
            n_corrections_applied += 1
            action = corr['action']
            reason = corr.get('reason', '')
            if action == 'remove':
                log_entries.append(
                    f"- **REMOVED (correction)** `{t.timestamp} {t.speaker}` — {t.text!r}  \n"
                    f"  *reason: {reason}*"
                )
                continue
            elif action == 'edit':
                new_text = corr['corrected']
                log_entries.append(
                    f"- **EDITED (correction)** `{t.timestamp} {t.speaker}`  \n"
                    f"  from: {t.text!r}  \n"
                    f"  to:   {new_text!r}  \n"
                    f"  *reason: {reason}*"
                )
                # Rebuild raw_line with corrected text (preserve tab format).
                new_raw = f"{t.timestamp}\t{t.speaker}\t{new_text}"
                t = GoldTurn(
                    raw_line=new_raw,
                    timestamp=t.timestamp, t_seconds=t.t_seconds,
                    speaker=t.speaker, text=new_text,
                )
                corrected.append(t)
                continue

        if verdict == 'inaud':
            log_entries.append(
                f"- **REMOVED** `{t.timestamp} {t.speaker}` — {t.text!r}  \n"
                f"  *verdict: inaudible; gold over-transcribed*"
                + (f"  \n  *note: {note}*" if note else '')
            )
            continue  # drop from corrected gold

        if verdict == 'both':
            log_entries.append(
                f"- **FLAGGED** `{t.timestamp} {t.speaker}` — {t.text!r}  \n"
                f"  *verdict: both wrong; needs manual fix*"
                + (f"  \n  *note: {note}*" if note else '')
            )
            # Keep in corrected gold but mark the line with a trailing comment.
            t = GoldTurn(
                raw_line=t.raw_line + '    # AUDIT_FLAG: both wrong',
                timestamp=t.timestamp, t_seconds=t.t_seconds,
                speaker=t.speaker, text=t.text,
            )

        if verdict == 'gold':
            log_entries.append(
                f"- **KEPT (confirmed correct)** `{t.timestamp} {t.speaker}` — {t.text!r}  \n"
                f"  *verdict: gold-correct; pred had a genuine error*"
                + (f"  \n  *note: {note}*" if note else '')
            )
        elif verdict == 'pred':
            log_entries.append(
                f"- **KEPT (metric artifact)** `{t.timestamp} {t.speaker}` — {t.text!r}  \n"
                f"  *verdict: pred-equivalent; difference was segmentation/normalization/timing, not content*"
                + (f"  \n  *note: {note}*" if note else '')
            )

        corrected.append(t)

    # Detect any YAML corrections that never matched a gold turn (= typo).
    applied_keys = set()
    for t in gold:
        k = (t.timestamp, t.text)
        if k in corrections:
            applied_keys.add(k)
    for k in corrections.keys():
        if k not in applied_keys:
            n_corrections_unmatched.append(k)
    if n_corrections_unmatched:
        print("WARNING: corrections that didn't match any gold turn:")
        for k in n_corrections_unmatched:
            print(f"  {k}")

    # Write corrected gold preserving original format
    with args.out_gold.open('w', encoding='utf-8') as f:
        for t in corrected:
            f.write(t.raw_line + '\n')

    # Write change log
    n_total = len(gold)
    n_kept = len(corrected)
    n_removed = n_total - n_kept
    n_reviewed = sum(1 for v in decs.values() if v.get('verdict'))
    verdict_counts: dict[str, int] = {}
    for v in decs.values():
        verd = v.get('verdict')
        if verd:
            verdict_counts[verd] = verdict_counts.get(verd, 0) + 1

    log_header = (
        f"# TIMSS gold correction log\n\n"
        f"- Source gold: `{args.gold.name}`\n"
        f"- Audit: `{args.audit.name if args.audit else '(none)'}`\n"
        f"- Corrections config: `{args.corrections.name if args.corrections else '(none)'}`\n"
        f"- Original gold turns: **{n_total}**\n"
        f"- Reviewed turns: **{n_reviewed}** ({n_reviewed/n_total:.0%})\n"
        f"- Explicit corrections applied: **{n_corrections_applied}**\n"
        f"- Corrected gold turns: **{n_kept}** (removed {n_removed})\n\n"
        f"## Verdict counts\n\n"
    )
    for verd, count in sorted(verdict_counts.items()):
        log_header += f"- **{verd}**: {count}\n"
    log_header += "\n## Changes applied\n\n"

    args.out_log.write_text(log_header + '\n'.join(log_entries) + '\n')

    print(f"Wrote corrected gold: {args.out_gold} ({n_kept}/{n_total} turns kept)")
    print(f"Wrote audit log:      {args.out_log}")


if __name__ == '__main__':
    main()
