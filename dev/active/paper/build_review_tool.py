#!/usr/bin/env python3
"""
Build a self-contained HTML side-by-side review tool for comparing a gold
TIMSS transcript against a v10 pipeline transcript.

The HTML page:
  - Embeds an aligned list of (gold_turn, nearest_pred_turn(s)) pairs
  - Shows word-level diffs (LCS) using safe DOM construction (no innerHTML)
  - Has an inline <video> player that jumps to each turn's timestamp
  - Offers four classification buttons per row:
        gold ✓  | pred ✓  | both wrong  | inaudible
  - Persists decisions to localStorage, with an Export button that downloads
    a JSON audit log

Usage:
    python build_review_tool.py \
        --gold "Math+transcripts+as+txt+files+(1)/Math US1 transcript.txt" \
        --pred dev/active/paper/benchmark_runs/US1_first300s_transcript.txt \
        --video dev/active/paper/benchmark_runs/US1_first300s.mp4 \
        --clip-seconds 300 \
        --out dev/active/paper/benchmark_runs/review_US1.html
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


# ----------------------------------------------------------------------
# Parsing

GOLD_LINE_RE = re.compile(r'^(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)$')
V10_LINE_RE  = re.compile(r'^(\d{1,2}:\d{2})\s+([^:]+):\s*(.*)$')
STRIP_RES = [
    re.compile(r'\*LOW CONFIDENCE[^*]*\*'),
    re.compile(r'\*Speaker:[^*]*\*'),
    re.compile(r'🚨|⚠️|✅'),
]


@dataclass
class Turn:
    t_seconds: float
    speaker: str
    text: str


def hms_to_seconds(s: str) -> float:
    parts = [int(p) for p in s.split(':')]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(s)


def parse_gold(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split('\t', 2)
        if len(parts) == 3 and re.match(r'^\d{2}:\d{2}:\d{2}$', parts[0]):
            ts, sp, txt = parts
            turns.append(Turn(hms_to_seconds(ts), sp.strip(), txt.strip()))
            continue
        m = GOLD_LINE_RE.match(line)
        if m:
            ts, sp, txt = m.groups()
            turns.append(Turn(hms_to_seconds(ts), sp.strip(), txt.strip()))
    return turns


def parse_pred(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw.strip()
        if not line or line.startswith('---') or line.startswith('=='):
            continue
        m = V10_LINE_RE.match(line)
        if not m:
            continue
        ts_str, speaker_raw, body = m.groups()
        if speaker_raw.strip().startswith('['):
            continue  # standalone visual line
        try:
            t_sec = hms_to_seconds(ts_str)
        except ValueError:
            continue
        cleaned = body
        for r in STRIP_RES:
            cleaned = r.sub('', cleaned)
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned:
            continue
        turns.append(Turn(t_sec, speaker_raw.strip(), cleaned))
    return turns


# ----------------------------------------------------------------------
# Alignment

def align_turns(gold: list[Turn], pred: list[Turn],
                window_seconds: float = 10.0) -> list[dict]:
    pairs: list[dict] = []
    used_pred_ids: set[int] = set()

    for g_idx, g in enumerate(gold):
        cands = [(p_idx, p, abs(p.t_seconds - g.t_seconds))
                 for p_idx, p in enumerate(pred)
                 if abs(p.t_seconds - g.t_seconds) <= window_seconds]
        cands.sort(key=lambda c: c[2])
        chosen = cands[:3]
        for p_idx, _, _ in chosen:
            used_pred_ids.add(p_idx)
        pairs.append({
            'kind': 'gold',
            'gold_idx': g_idx,
            'gold_t': g.t_seconds,
            'gold_speaker': g.speaker,
            'gold_text': g.text,
            'pred_matches': [
                {'idx': p_idx, 't': p.t_seconds, 'speaker': p.speaker, 'text': p.text}
                for p_idx, p, _ in chosen
            ],
        })

    for p_idx, p in enumerate(pred):
        if p_idx not in used_pred_ids:
            pairs.append({
                'kind': 'pred_orphan',
                'gold_idx': None,
                'gold_t': None,
                'gold_speaker': None,
                'gold_text': None,
                'pred_matches': [
                    {'idx': p_idx, 't': p.t_seconds, 'speaker': p.speaker, 'text': p.text}
                ],
            })

    pairs.sort(key=lambda r: (r['gold_t'] if r['gold_t'] is not None
                              else r['pred_matches'][0]['t']))
    return pairs


# ----------------------------------------------------------------------
# Word-diff: emit STRUCTURED segments rather than HTML strings.
# Each segment is {"t": text, "d": bool} — d=true means "differs from other side".

def word_diff_segments(a: str, b: str) -> tuple[list[dict], list[dict]]:
    def tokenize(s: str) -> list[str]:
        return re.findall(r"\S+|\s+", s)

    def norm(t: str) -> str:
        return re.sub(r"[^\w']", '', t).lower()

    a_toks = tokenize(a)
    b_toks = tokenize(b)
    a_norm = [norm(t) for t in a_toks]
    b_norm = [norm(t) for t in b_toks]

    sm = SequenceMatcher(a=a_norm, b=b_norm, autojunk=False)
    out_a: list[dict] = []
    out_b: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_span = ''.join(a_toks[i1:i2])
        b_span = ''.join(b_toks[j1:j2])
        if tag == 'equal':
            if a_span:
                out_a.append({'t': a_span, 'd': False})
            if b_span:
                out_b.append({'t': b_span, 'd': False})
        else:
            if a_span:
                out_a.append({'t': a_span, 'd': True})
            if b_span:
                out_b.append({'t': b_span, 'd': True})
    return out_a, out_b


def plain_segments(s: str) -> list[dict]:
    return [{'t': s, 'd': False}]


# ----------------------------------------------------------------------
# HTML — uses textContent + createElement everywhere.

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>TIMSS __LESSON__ review</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 0;
         background: #fafafa; color: #222; }
  header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
           padding: 12px 20px; display: flex; gap: 20px; align-items: center;
           z-index: 10; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 18px; }
  #player { width: 480px; height: 270px; background: #000; }
  #stats { font-size: 13px; color: #555; }
  #stats b { color: #222; }
  button.export { background: #0a66c2; color: white; border: 0; padding: 8px 14px;
                  border-radius: 4px; cursor: pointer; font-size: 13px; }
  button.export:hover { background: #084d93; }
  main { padding: 20px; max-width: 1500px; margin: 0 auto; }
  table { border-collapse: collapse; width: 100%; background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #eee;
           vertical-align: top; font-size: 14px; }
  th { background: #f0f0f0; font-weight: 600; font-size: 12px;
       text-transform: uppercase; color: #555; }
  tr.orphan { background: #fff8e1; }
  tr.decided-gold { background: #e8f5e9; }
  tr.decided-pred { background: #e3f2fd; }
  tr.decided-both { background: #fce4ec; }
  tr.decided-inaud { background: #eeeeee; color: #888; }
  tr.decided-golderr { background: #fff3e0; }
  .t { color: #0a66c2; font-family: ui-monospace, monospace; cursor: pointer;
       white-space: nowrap; user-select: none; }
  .t:hover { text-decoration: underline; }
  .sp { font-weight: 600; color: #555; font-size: 12px; }
  .d { background: #ffe082; padding: 0 2px; border-radius: 2px; }
  .pm { display: block; padding: 2px 0; }
  .pm + .pm { border-top: 1px dashed #ddd; margin-top: 4px; padding-top: 4px; }
  .btns { display: flex; flex-direction: column; gap: 4px; width: 110px; }
  .btns button { border: 1px solid #ccc; background: white; padding: 4px 8px;
                 font-size: 11px; border-radius: 3px; cursor: pointer;
                 text-align: left; }
  .btns button:hover { border-color: #888; }
  .btns button.active { background: #0a66c2; color: white; border-color: #0a66c2; }
  .notes { width: 140px; font-size: 12px; font-family: inherit;
           border: 1px solid #ddd; padding: 4px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>TIMSS __LESSON__ · gold vs v10 review</h1>
  <video id="player" src="__VIDEO__" controls preload="metadata"></video>
  <div id="stats">
    Total: <b id="n-total">0</b> &nbsp;|&nbsp;
    Decided: <b id="n-decided">0</b> &nbsp;|&nbsp;
    gold✓ <b id="n-gold">0</b> &nbsp;
    pred✓ <b id="n-pred">0</b> &nbsp;
    both✗ <b id="n-both">0</b> &nbsp;
    inaud <b id="n-inaud">0</b> &nbsp;
    golderr <b id="n-golderr">0</b>
  </div>
  <button class="export" id="export-btn">Export audit JSON</button>
</header>
<main>
<table>
<thead>
<tr>
  <th>time</th><th>spk</th><th>gold</th>
  <th>pred t</th><th>spk</th><th>pred</th>
  <th>classify</th><th>note</th>
</tr>
</thead>
<tbody id="rows"></tbody>
</table>
</main>

<script>
const ROWS = __ROWS_JSON__;
const LESSON = "__LESSON__";
const STORAGE_KEY = "timss_review_" + LESSON;

function loadDecisions() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (e) { return {}; }
}
function saveDecisions(d) { localStorage.setItem(STORAGE_KEY, JSON.stringify(d)); }
const decisions = loadDecisions();
const player = document.getElementById("player");

function seekTo(seconds) {
  player.currentTime = Math.max(0, seconds - 1);
  player.play();
}

function fmtTime(sec) {
  if (sec == null) return "—";
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return m + ":" + String(s).padStart(2, "0");
}

// Stable row key keyed by content (timestamp + text prefix), not index.
// Lets us regenerate the review sheet against a corrected gold without losing
// existing decisions — as long as a turn's text is unchanged, its key is the
// same.
function stableHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h).toString(36);
}
function rowKey(row) {
  if (row.kind === "gold") {
    const sig = Math.round(row.gold_t) + "|" + (row.gold_text || "").slice(0, 40);
    return "g_" + stableHash(sig);
  }
  const p = row.pred_matches[0];
  const sig = Math.round(p.t) + "|" + (p.text || "").slice(0, 40);
  return "p_" + stableHash(sig);
}

// Build a span containing text-only (or diff-highlighted) segments. SAFE: uses
// createElement + textContent, never innerHTML.
function renderSegments(parent, segs) {
  for (const seg of segs) {
    if (seg.d) {
      const s = document.createElement("span");
      s.className = "d";
      s.textContent = seg.t;
      parent.appendChild(s);
    } else {
      parent.appendChild(document.createTextNode(seg.t));
    }
  }
}

function makeTimeSpan(seconds) {
  const span = document.createElement("span");
  span.className = "t";
  span.textContent = fmtTime(seconds);
  if (seconds != null) {
    span.addEventListener("click", () => seekTo(seconds));
  } else {
    span.style.color = "#bbb";
  }
  return span;
}

function makeSpeakerSpan(label) {
  const span = document.createElement("span");
  span.className = "sp";
  span.textContent = label || "";
  return span;
}

function updateStats() {
  const vals = Object.values(decisions);
  document.getElementById("n-total").textContent = ROWS.length;
  document.getElementById("n-decided").textContent = vals.filter(v => v && v.verdict).length;
  document.getElementById("n-gold").textContent = vals.filter(v => v && v.verdict === "gold").length;
  document.getElementById("n-pred").textContent = vals.filter(v => v && v.verdict === "pred").length;
  document.getElementById("n-both").textContent = vals.filter(v => v && v.verdict === "both").length;
  document.getElementById("n-inaud").textContent = vals.filter(v => v && v.verdict === "inaud").length;
  document.getElementById("n-golderr").textContent = vals.filter(v => v && v.verdict === "golderr").length;
}

function setVerdict(rowIdx, verdict) {
  const row = ROWS[rowIdx];
  const k = rowKey(row);
  if (!decisions[k]) decisions[k] = {};
  decisions[k].verdict = verdict;
  decisions[k].gold_text = row.gold_text;
  decisions[k].gold_t = row.gold_t;
  decisions[k].pred = row.pred_matches.map(p => ({ t: p.t, speaker: p.speaker, text: p.text }));
  saveDecisions(decisions);
  const tr = document.getElementById("row-" + rowIdx);
  tr.classList.remove("decided-gold","decided-pred","decided-both","decided-inaud","decided-golderr");
  tr.classList.add("decided-" + verdict);
  tr.querySelectorAll(".btns button").forEach(b => b.classList.remove("active"));
  const btn = tr.querySelector('.btns button[data-v="' + verdict + '"]');
  if (btn) btn.classList.add("active");
  updateStats();
}

function setNote(rowIdx, text) {
  const row = ROWS[rowIdx];
  const k = rowKey(row);
  if (!decisions[k]) decisions[k] = {};
  decisions[k].note = text;
  saveDecisions(decisions);
}

function render() {
  const tbody = document.getElementById("rows");
  tbody.textContent = "";
  ROWS.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.id = "row-" + i;
    if (row.kind === "pred_orphan") tr.classList.add("orphan");
    const k = rowKey(row);
    const saved = decisions[k];
    if (saved && saved.verdict) tr.classList.add("decided-" + saved.verdict);

    // gold time
    const td1 = document.createElement("td");
    td1.appendChild(makeTimeSpan(row.gold_t));
    tr.appendChild(td1);

    // gold speaker
    const td2 = document.createElement("td");
    td2.appendChild(makeSpeakerSpan(row.gold_speaker));
    tr.appendChild(td2);

    // gold text (with diff highlighting)
    const td3 = document.createElement("td");
    if (row.gold_segments && row.gold_segments.length) {
      renderSegments(td3, row.gold_segments);
    } else if (row.gold_text) {
      td3.textContent = row.gold_text;
    } else {
      const i_ = document.createElement("i");
      i_.style.color = "#999";
      i_.textContent = "(no gold)";
      td3.appendChild(i_);
    }
    tr.appendChild(td3);

    // pred first time
    const td4 = document.createElement("td");
    if (row.pred_matches.length > 0) {
      td4.appendChild(makeTimeSpan(row.pred_matches[0].t));
    } else {
      td4.textContent = "—";
    }
    tr.appendChild(td4);

    // pred first speaker
    const td5 = document.createElement("td");
    if (row.pred_matches.length > 0) {
      td5.appendChild(makeSpeakerSpan(row.pred_matches[0].speaker));
    }
    tr.appendChild(td5);

    // pred text (multi-row, with diff highlighting on the primary)
    const td6 = document.createElement("td");
    if (row.pred_matches.length === 0) {
      const i_ = document.createElement("i");
      i_.style.color = "#c00";
      i_.textContent = "(missing from pred)";
      td6.appendChild(i_);
    } else {
      row.pred_matches.forEach((p, j) => {
        const pm = document.createElement("span");
        pm.className = "pm";
        pm.appendChild(makeTimeSpan(p.t));
        pm.appendChild(document.createTextNode(" "));
        pm.appendChild(makeSpeakerSpan(p.speaker));
        pm.appendChild(document.createTextNode(" "));
        if (j === 0 && row.pred_segments && row.pred_segments.length) {
          renderSegments(pm, row.pred_segments);
        } else {
          pm.appendChild(document.createTextNode(p.text));
        }
        td6.appendChild(pm);
      });
    }
    tr.appendChild(td6);

    // classify buttons
    const td7 = document.createElement("td");
    td7.className = "btns";
    [["gold","gold ✓"],["pred","pred ✓"],["both","both ✗"],["inaud","inaudible"],["golderr","gold error"]].forEach(([v, label]) => {
      const b = document.createElement("button");
      b.dataset.v = v;
      b.textContent = label;
      b.addEventListener("click", () => setVerdict(i, v));
      if (saved && saved.verdict === v) b.classList.add("active");
      td7.appendChild(b);
    });
    tr.appendChild(td7);

    // note
    const td8 = document.createElement("td");
    const inp = document.createElement("input");
    inp.className = "notes";
    inp.placeholder = "note…";
    inp.value = (saved && saved.note) || "";
    inp.addEventListener("input", () => setNote(i, inp.value));
    td8.appendChild(inp);
    tr.appendChild(td8);

    tbody.appendChild(tr);
  });
  updateStats();
}

document.getElementById("export-btn").addEventListener("click", () => {
  const out = {
    lesson: LESSON,
    exported_at: new Date().toISOString(),
    decisions: decisions,
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "review_" + LESSON + "_" + Date.now() + ".json";
  a.click();
});

render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True, type=Path)
    ap.add_argument('--pred', required=True, type=Path)
    ap.add_argument('--video', required=True, type=Path)
    ap.add_argument('--lesson', default='US1')
    ap.add_argument('--clip-seconds', type=int, default=None)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    gold = parse_gold(args.gold)
    pred = parse_pred(args.pred)
    if args.clip_seconds is not None:
        gold = [t for t in gold if t.t_seconds < args.clip_seconds]
        # Give pred a 15s grace margin beyond the clip so end-of-clip gold
        # rows can still match against pred content whose timestamps drift
        # slightly past the boundary. Without this, gold turns near the clip
        # edge wrongly appear as "missing from pred" when the content is just
        # filtered out.
        pred = [t for t in pred if t.t_seconds < args.clip_seconds + 15]

    rows = align_turns(gold, pred)

    # Diff logic (time-windowed, both directions).
    #
    # The reviewer's criterion is: "is this content anywhere nearby on the
    # other side?" — cross-row segmentation should not be highlighted.
    #
    # For each row:
    #   gold highlighting: a gold word is marked "diff" only if the normalized
    #     word is NOT present anywhere in the nearby pred window
    #     ([row.t - W, row.t + W]).
    #   pred highlighting: same in reverse — pred word is marked "diff" only
    #     if not present anywhere in the nearby gold window.
    # This is membership-based, not sequence-based, so it tolerates different
    # word orders and turn divisions.
    # Equivalence rules (VARIANTS table, IGNORE_WORDS, hyphen-split, char-split)
    # are imported from the shared content_equivalence module. Both the
    # benchmark scorer and this HTML review tool now use the same expand()
    # function so the visual highlighting and the scoring can never drift.
    from content_equivalence import (
        expand, DEFAULT_WINDOW_SECONDS as WINDOW,
    )

    def windowed_word_set(turns: list[Turn], t_center: float) -> set[str]:
        words: set[str] = set()
        for tt in turns:
            if abs(tt.t_seconds - t_center) > WINDOW:
                continue
            for w in re.findall(r"\S+", tt.text):
                words |= expand(w)
        return words

    def tokenize_keep_ws(s: str) -> list[str]:
        return re.findall(r"\S+|\s+", s)

    def segments_against(text: str, other_words: set[str]) -> list[dict]:
        segs: list[dict] = []
        for tok in tokenize_keep_ws(text):
            if tok.isspace():
                segs.append({'t': tok, 'd': False})
                continue
            # Expand the token into its equivalence set; if ANY form appears
            # in the other side's word set, it's a match.
            forms = expand(tok)
            is_diff = bool(forms) and not (forms & other_words)
            segs.append({'t': tok, 'd': is_diff})
        return segs

    for row in rows:
        if row['gold_text'] is None and not row['pred_matches']:
            row['gold_segments'] = []
            row['pred_segments'] = []
            continue

        # Time center: prefer gold timestamp; fall back to first pred.
        t_center = (row['gold_t'] if row['gold_t'] is not None
                    else row['pred_matches'][0]['t'])

        nearby_pred_words = windowed_word_set(pred, t_center)
        nearby_gold_words = windowed_word_set(gold, t_center)

        if row['gold_text']:
            row['gold_segments'] = segments_against(row['gold_text'], nearby_pred_words)
        else:
            row['gold_segments'] = []

        if row['pred_matches']:
            primary = row['pred_matches'][0]['text']
            row['pred_segments'] = segments_against(primary, nearby_gold_words)
        else:
            row['pred_segments'] = []

    try:
        video_rel = args.video.resolve().relative_to(args.out.parent.resolve())
        video_path_str = str(video_rel)
    except ValueError:
        video_path_str = str(args.video.resolve())

    rows_json = json.dumps(rows, ensure_ascii=False)
    html_doc = (HTML_TEMPLATE
                .replace('__LESSON__', args.lesson)
                .replace('__VIDEO__', video_path_str)
                .replace('__ROWS_JSON__', rows_json))
    args.out.write_text(html_doc, encoding='utf-8')

    print(f"Wrote {args.out}")
    print(f"  Gold turns: {len(gold)}")
    print(f"  Pred turns: {len(pred)}")
    print(f"  Rows (gold + pred-orphans): {len(rows)}")


if __name__ == '__main__':
    main()
