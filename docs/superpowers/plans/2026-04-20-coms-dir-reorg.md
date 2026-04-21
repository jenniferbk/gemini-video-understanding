# COMS Directory Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `/Users/jenniferkleiman/Documents/COMS/` from 317 cluttered top-level entries (~34 GB) into a focused layout: keep the Electron app and active paper work in place; consolidate 91 transcription runs into `transcription_runs/` (keepers only); archive pre-v10 docs/scripts/logs; sequester IRB + name-maps in `sensitive/`; delete redundant media; fix a real PII-leak gap in the v10 pipeline's chunk-cleanup.

**Architecture:** Six sequential phases with approval gates on destructive actions: (1) read-only inventory → (2) create empty target dirs + gitignore → (3) file moves (mostly gitignored, `git mv` for tracked files) → (4) four approval-gated deletion batches → (5) pipeline code change with TDD → (6) commits.

**Tech Stack:** Bash file operations, Python 3.11+ (for pipeline change), pytest (for test), git (for `git mv` and commit hygiene).

**Design spec:** `docs/superpowers/specs/2026-04-20-coms-dir-reorg-design.md`

**Key constraints:**
- `dev/active/paper/*` modifications and all currently-modified git-tracked files MUST NOT be touched
- `src/`, `scripts/`, `resources/`, `database/`, `test_fixtures/` stay put (Electron build paths depend on them)
- TIMSS videos preserved (identified via `dev/active/paper/benchmark_runs/*_gold.txt` references)
- No transcript deleted unless a newer complete run for the same source video exists
- Every deletion batch requires explicit user approval with file list shown

---

## Task 1: Generate Inventory (read-only)

Produce `REORG_INVENTORY.md` at repo root describing what will move, what will be deleted, and what needs human triage. Zero destructive actions.

**Files:**
- Create: `/Users/jenniferkleiman/Documents/COMS/REORG_INVENTORY.md`

- [ ] **Step 1: Enumerate all transcription run folders**

Run from `/Users/jenniferkleiman/Documents/COMS/`:
```bash
ls -d *_transcription_* 2>/dev/null | sort > /tmp/all_runs.txt
wc -l /tmp/all_runs.txt
```
Expected: ~91 folders.

- [ ] **Step 2: Group runs by base video name**

Base name = everything before `_v04_transcription_`, `_v08_transcription_`, `_v09_transcription_`, `_v10_`, or `_transcription_` (earliest pattern). Implement as a python one-liner or small script:

```bash
python3 <<'EOF'
import re, os
from pathlib import Path
from collections import defaultdict

root = Path("/Users/jenniferkleiman/Documents/COMS")
runs = [d.name for d in root.iterdir() if d.is_dir() and re.search(r"_transcription_\d{8}_\d{6}$|_v10_\d{8}_\d{6}$", d.name)]
groups = defaultdict(list)
pat = re.compile(r"^(.*?)(?:_v04_transcription|_v08_transcription|_v09_transcription|_v10|_transcription)_\d{8}_\d{6}$")
for r in runs:
    m = pat.match(r)
    key = m.group(1) if m else r
    groups[key].append(r)
for k in sorted(groups):
    print(f"=== {k}")
    for r in sorted(groups[k]):
        print(f"  {r}")
EOF
```
Save output to `/tmp/grouped_runs.txt` for reference.

- [ ] **Step 3: Pick keeper per group**

Rule: newest `v10` run wins; fallback = newest overall. Extend the script:

```bash
python3 <<'EOF' > /tmp/keepers.txt
import re
from pathlib import Path
from collections import defaultdict

root = Path("/Users/jenniferkleiman/Documents/COMS")
runs = [d.name for d in root.iterdir() if d.is_dir() and re.search(r"_transcription_\d{8}_\d{6}$|_v10_\d{8}_\d{6}$", d.name)]
groups = defaultdict(list)
pat = re.compile(r"^(.*?)(?:_v04_transcription|_v08_transcription|_v09_transcription|_v10|_transcription)_(\d{8}_\d{6})$")
for r in runs:
    m = pat.match(r)
    if m:
        key = m.group(1)
        ts = m.group(2)
        is_v10 = "_v10_" in r
        groups[key].append((ts, is_v10, r))

print("KEEPER\tNON-KEEPERS")
for k in sorted(groups):
    entries = groups[k]
    v10s = sorted([e for e in entries if e[1]], key=lambda x: x[0], reverse=True)
    if v10s:
        keeper = v10s[0][2]
    else:
        keeper = sorted(entries, key=lambda x: x[0], reverse=True)[0][2]
    non = [e[2] for e in entries if e[2] != keeper]
    print(f"{keeper}\t{','.join(non)}")
EOF
cat /tmp/keepers.txt
```

- [ ] **Step 4: Identify TIMSS videos**

TIMSS videos are referenced by gold files in `dev/active/paper/benchmark_runs/`. Extract base names:

```bash
ls /Users/jenniferkleiman/Documents/COMS/dev/active/paper/benchmark_runs/*_gold.txt 2>/dev/null
ls /Users/jenniferkleiman/Documents/COMS/dev/active/paper/benchmark_runs/ | grep -iE "^(us|au|cz|jp)[0-9]" | sort -u
```

Then in the top level, find any `.mp4`/`.mov` whose name contains `US1`, `US2`, `US3`, `US4`, `AU1`, `AU2`, `AU3`, `AU4`, `CZ1`, `JP1` (exact codes depend on what gold files exist):
```bash
find /Users/jenniferkleiman/Documents/COMS -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" \) | grep -iE "US[1-4]|AU[1-4]|CZ1|JP1"
```

Note: TIMSS videos may be inside subfolders too — check the paper's benchmark scripts to find their actual paths. If none at top level, no moves needed for `timss_videos/` (folder stays empty or is skipped).

- [ ] **Step 5: Find name-maps in keeper runs**

```bash
find /Users/jenniferkleiman/Documents/COMS -maxdepth 3 -name "transcript_name_map.json" 2>/dev/null
```

- [ ] **Step 6: Identify loose top-level items to categorize**

Run the three categorization lists from the spec against reality:
```bash
cd /Users/jenniferkleiman/Documents/COMS
ls -l PHASE*_COMPLETE.md V04_*.md FIX_SUMMARY.md PROMPT_VALIDATION_SUMMARY.md PYTHON_SCRIPT_FIXES.md \
   BETA_TESTING_GUIDE.md BUNDLING_GUIDE.md TESTING_CHECKLIST.md TODO_MVP.md QUICK_FIX_FOR_USERS.md \
   README_V04.md V03_to_V04_Migration_Guide.md 2>/dev/null
ls -l batch_process_v04.py compare_v03_v04.py consensus_*.py fix_*.py patch_v04.py reprocess_chunks.py \
   v04_*.py test_imports.py test_rtf_output.py test_v04_installation.py setup_v04.sh run_ava_batch.sh \
   test-fresh-install.sh quick-test-uuid-fix.sh test-prompt-uuid-fix.sh 2>/dev/null
ls -l build-log*.txt ben_day*_log.txt *_test.log debug_response_object.log transcription_debug.log \
   mystery_solved.log 2>/dev/null
ls -l transcripts_annaB.xlsx transcripts_combined.xlsx 3Math_Ben_Day4_SG2_Arg2.drawing \
   Y2_whole_class_episodes.html whole_class_episodes.html v04_processing_summary.json 2>/dev/null
```

Any file in the spec lists not found on disk is simply skipped in later move tasks (no error).

- [ ] **Step 7: Write REORG_INVENTORY.md**

Consolidate findings into a single markdown file with sections:
```
# COMS Reorganization Inventory (2026-04-20)

## Transcription runs
### Keepers (move to transcription_runs/) — N folders
| Base video | Keeper folder | Size |
| ...

### Non-keepers (DELETE in Batch A) — N folders
| Base video | Folder | Why superseded |
| ...

### Singletons (only one run, keep it) — N folders

## TIMSS videos (move to timss_videos/)
- path1
- path2

## Sensitive content
- IRB/ → sensitive/IRB/
- name-maps found: path1, path2, ...

## Clarke/GENIUS content (move to _to_relocate/)
- file1
- file2

## Archive candidates
### archive/docs/: files found on disk (list)
### archive/scripts/: files found on disk (list)
### archive/logs/: files found on disk (list)

## misc/: files found on disk (list)

## Loose complete-transcript files (triage manually)
- 240920_4Science_Ava_Post_0509_v04_complete_transcript.txt — duplicate of run X? standalone?
- 250124_3Science_Faith_Day4_0408_v04_complete_transcript.txt — duplicate of run X? standalone?

## Deletion summary (projected)
- Batch A: N run folders, total ~X GB
- Batch B: chunks/ subdirs (N) + video_chunks/ (287 MB)
- Batch C: raw COMS videos — list
- Batch D: loose chunk transcripts — N files

Projected reclaim: ~X GB.
```

- [ ] **Step 8: Show inventory to user, wait for approval**

```bash
cat /Users/jenniferkleiman/Documents/COMS/REORG_INVENTORY.md
```
Then stop and prompt: "Inventory ready — please review. OK to proceed with structure creation?" **No further tasks run without explicit yes.**

---

## Task 2: Create Empty Directory Structure and Update Gitignore

Non-destructive scaffolding. Directories are empty until Task 3+.

**Files:**
- Create: `/Users/jenniferkleiman/Documents/COMS/transcription_runs/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/archive/docs/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/archive/scripts/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/archive/logs/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/sensitive/IRB/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/sensitive/name_maps/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/timss_videos/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/_to_relocate/` (dir)
- Create: `/Users/jenniferkleiman/Documents/COMS/misc/` (dir)
- Modify: `/Users/jenniferkleiman/Documents/COMS/.gitignore`

- [ ] **Step 1: Create directories**

```bash
cd /Users/jenniferkleiman/Documents/COMS
mkdir -p transcription_runs archive/{docs,scripts,logs} sensitive/{IRB,name_maps} \
  timss_videos _to_relocate misc
```

- [ ] **Step 2: Restrict perms on sensitive and _to_relocate**

```bash
chmod 700 sensitive _to_relocate
ls -ld sensitive _to_relocate
```
Expected: `drwx------`.

- [ ] **Step 3: Add gitignore entries**

Append to `/Users/jenniferkleiman/Documents/COMS/.gitignore` (via Edit tool, adding at end):
```
# Reorganization (2026-04-20)
sensitive/
_to_relocate/
transcription_runs/
timss_videos/
misc/
archive/logs/
REORG_INVENTORY.md
```
Note: `archive/docs/` and `archive/scripts/` are NOT gitignored — these preserve project history and should be tracked.

- [ ] **Step 4: Verify gitignore works**

```bash
cd /Users/jenniferkleiman/Documents/COMS
git status --short sensitive/ _to_relocate/ transcription_runs/ timss_videos/ misc/ archive/logs/ 2>&1 | head
```
Expected: no output (all ignored).

- [ ] **Step 5: Commit structure**

```bash
cd /Users/jenniferkleiman/Documents/COMS
git add .gitignore
git commit -m "$(cat <<'EOF'
chore(reorg): add target directory structure to gitignore

Scaffolds transcription_runs/, archive/, sensitive/, timss_videos/,
_to_relocate/, misc/ — preparation for directory reorganization.
archive/docs and archive/scripts will be tracked; runtime/sensitive
dirs stay out of git.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extract Name-Maps to sensitive/name_maps/

Before moving runs, copy out `transcript_name_map.json` files to the secured location. The files are audit trails and per project memory should have tighter access control than the transcripts themselves.

- [ ] **Step 1: List all name-maps**

```bash
find /Users/jenniferkleiman/Documents/COMS -name "transcript_name_map.json" 2>/dev/null -not -path "*/sensitive/*"
```

- [ ] **Step 2: Copy (not move) each into sensitive/name_maps/ with a unique name**

For each found path, copy preserving provenance:
```bash
for f in $(find /Users/jenniferkleiman/Documents/COMS -name "transcript_name_map.json" 2>/dev/null -not -path "*/sensitive/*"); do
    parent=$(basename $(dirname "$f"))
    cp -v "$f" "/Users/jenniferkleiman/Documents/COMS/sensitive/name_maps/${parent}_transcript_name_map.json"
done
chmod 600 /Users/jenniferkleiman/Documents/COMS/sensitive/name_maps/*.json
```
We **copy** not move — the originals stay in their runs so the run folders remain self-contained. The sensitive/ copy is the audit-trail backup. Redundant is fine.

- [ ] **Step 3: Verify**

```bash
ls -l /Users/jenniferkleiman/Documents/COMS/sensitive/name_maps/
```

---

## Task 4: Move IRB to sensitive/IRB/

- [ ] **Step 1: Show what's in IRB/ first**

```bash
ls /Users/jenniferkleiman/Documents/COMS/IRB/ | head
du -sh /Users/jenniferkleiman/Documents/COMS/IRB/
```

- [ ] **Step 2: Move contents (not the folder itself — we already created sensitive/IRB/)**

```bash
cd /Users/jenniferkleiman/Documents/COMS
mv IRB/* IRB/.* sensitive/IRB/ 2>/dev/null || true
# Move dotfiles safely — the '|| true' swallows the "." and ".." entries
rmdir IRB
```

- [ ] **Step 3: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/sensitive/IRB/ | head
test ! -e /Users/jenniferkleiman/Documents/COMS/IRB && echo "original IRB/ removed"
```

---

## Task 5: Move Clarke/GENIUS Content to _to_relocate/

- [ ] **Step 1: List Clarke items**

```bash
cd /Users/jenniferkleiman/Documents/COMS
ls -d Clarke* 2>/dev/null
```

- [ ] **Step 2: Move**

```bash
cd /Users/jenniferkleiman/Documents/COMS
# Use a loop to handle spaces in filenames reliably
for item in Clarke*; do
    [ -e "$item" ] && mv -v "$item" _to_relocate/
done
```

- [ ] **Step 3: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/_to_relocate/
ls /Users/jenniferkleiman/Documents/COMS/Clarke* 2>/dev/null && echo "UNEXPECTED: leftovers" || echo "clean"
```

---

## Task 6: Move TIMSS Videos to timss_videos/

Uses the list from Task 1 Step 4. If no TIMSS videos at top level, skip this task and note in REORG_INVENTORY.md that `timss_videos/` is empty or TIMSS is elsewhere (e.g., in `dev/` structure).

- [ ] **Step 1: Move each identified TIMSS video**

For each video path identified in inventory:
```bash
mv -v "<path>" /Users/jenniferkleiman/Documents/COMS/timss_videos/
```

- [ ] **Step 2: Verify**

```bash
ls -lh /Users/jenniferkleiman/Documents/COMS/timss_videos/
```

---

## Task 7: Move Keeper Transcription Runs to transcription_runs/

Uses the keeper list from Task 1 Step 3.

- [ ] **Step 1: Move each keeper**

```bash
cd /Users/jenniferkleiman/Documents/COMS
# For each keeper from inventory:
for keeper in $(awk -F'\t' 'NR>1 {print $1}' /tmp/keepers.txt); do
    [ -d "$keeper" ] && mv -v "$keeper" transcription_runs/
done
```

- [ ] **Step 2: Verify counts**

```bash
ls /Users/jenniferkleiman/Documents/COMS/transcription_runs/ | wc -l
du -sh /Users/jenniferkleiman/Documents/COMS/transcription_runs/
```

Expected: number matches keeper list from inventory.

---

## Task 8: Resolve Loose Top-Level Complete-Transcript Files

Per spec: `240920_4Science_Ava_Post_0509_v04_complete_transcript.txt` and `250124_3Science_Faith_Day4_0408_v04_complete_transcript.txt` need triage.

- [ ] **Step 1: Check if each has a keeper run for the same base video**

```bash
cd /Users/jenniferkleiman/Documents/COMS
for f in 240920_4Science_Ava_Post_0509_v04_complete_transcript.txt \
         250124_3Science_Faith_Day4_0408_v04_complete_transcript.txt; do
    [ -e "$f" ] || continue
    base=$(echo "$f" | sed 's/_v04_complete_transcript\.txt$//')
    echo "=== $f"
    echo "  base: $base"
    ls -d "transcription_runs/${base}"* 2>/dev/null | head
    echo "---"
done
```

- [ ] **Step 2: For each, move to keeper's folder if a keeper exists; else to transcription_runs/ as a standalone**

```bash
cd /Users/jenniferkleiman/Documents/COMS
for f in 240920_4Science_Ava_Post_0509_v04_complete_transcript.txt \
         250124_3Science_Faith_Day4_0408_v04_complete_transcript.txt; do
    [ -e "$f" ] || continue
    base=$(echo "$f" | sed 's/_v04_complete_transcript\.txt$//')
    keeper=$(ls -d "transcription_runs/${base}"* 2>/dev/null | head -1)
    if [ -n "$keeper" ]; then
        mv -v "$f" "$keeper/"
    else
        # No keeper for this video — preserve as standalone
        mv -v "$f" transcription_runs/
    fi
done
```

- [ ] **Step 3: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/*_complete_transcript.txt 2>/dev/null && echo "UNEXPECTED leftovers" || echo "clean"
```

---

## Task 9: Move Docs to archive/docs/

Per spec categorization list. Use `git mv` for tracked files to preserve history.

- [ ] **Step 1: Check which candidate files are git-tracked**

```bash
cd /Users/jenniferkleiman/Documents/COMS
for f in PHASE1_COMPLETE.md PHASE2_COMPLETE.md PHASE3_PROGRESS.md \
         V04_IMPLEMENTATION_SUMMARY.md V04_QUICK_FIX_INSTRUCTIONS.md V03_to_V04_Migration_Guide.md \
         FIX_SUMMARY.md PROMPT_VALIDATION_SUMMARY.md PYTHON_SCRIPT_FIXES.md \
         BETA_TESTING_GUIDE.md BUNDLING_GUIDE.md TESTING_CHECKLIST.md TODO_MVP.md \
         QUICK_FIX_FOR_USERS.md README_V04.md; do
    [ -e "$f" ] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        echo "TRACKED: $f"
    else
        echo "UNTRACKED: $f"
    fi
done
```

- [ ] **Step 2: Move tracked files with `git mv`**

```bash
cd /Users/jenniferkleiman/Documents/COMS
for f in PHASE1_COMPLETE.md PHASE2_COMPLETE.md PHASE3_PROGRESS.md \
         V04_IMPLEMENTATION_SUMMARY.md V04_QUICK_FIX_INSTRUCTIONS.md V03_to_V04_Migration_Guide.md \
         FIX_SUMMARY.md PROMPT_VALIDATION_SUMMARY.md PYTHON_SCRIPT_FIXES.md \
         BETA_TESTING_GUIDE.md BUNDLING_GUIDE.md TESTING_CHECKLIST.md TODO_MVP.md \
         QUICK_FIX_FOR_USERS.md README_V04.md; do
    [ -e "$f" ] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        git mv "$f" "archive/docs/$f"
    else
        mv -v "$f" "archive/docs/$f"
    fi
done
```

- [ ] **Step 3: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/archive/docs/
```

---

## Task 10: Move Old Scripts to archive/scripts/

Per spec categorization list. Same tracked-vs-untracked logic as Task 9.

- [ ] **Step 1: Move each script**

```bash
cd /Users/jenniferkleiman/Documents/COMS
SCRIPTS="batch_process_v04.py compare_v03_v04.py consensus_analysis_script.js consensus_tester.py
fix_float_issue.py fix_thinking_parameter.py fix_v04_setup.sh patch_v04.py
reprocess_chunks.py v04_config_generator.py v04_migration_tool.py
test_imports.py test_rtf_output.py test_v04_installation.py
setup_v04.sh run_ava_batch.sh test-fresh-install.sh quick-test-uuid-fix.sh test-prompt-uuid-fix.sh
requirements_v04.txt requirements_v04_fixed.txt requirements_v04_simple.txt"
for f in $SCRIPTS; do
    [ -e "$f" ] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        git mv "$f" "archive/scripts/$f"
    else
        mv -v "$f" "archive/scripts/$f"
    fi
done
```

- [ ] **Step 2: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/archive/scripts/
```

---

## Task 11: Move Logs to archive/logs/

`archive/logs/` is gitignored (from Task 2) so plain `mv` only — no tracked-file handling needed. But still use the tracked-check in case a log got committed somehow.

- [ ] **Step 1: Move**

```bash
cd /Users/jenniferkleiman/Documents/COMS
LOGS="build-log.txt build-log-v1.1.4.txt build-log-v1.1.5.txt build-log-v1.1.6.txt
ben_day2_log.txt ben_day3_log.txt debug_response_object.log transcription_debug.log
fixed_chunking_test.log fixed_test.log ultra_fixed_test.log mystery_solved.log"
for f in $LOGS; do
    [ -e "$f" ] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        git mv "$f" "archive/logs/$f"
    else
        mv -v "$f" "archive/logs/$f"
    fi
done
```

- [ ] **Step 2: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/archive/logs/
```

---

## Task 12: Move Misc Items

- [ ] **Step 1: Move**

```bash
cd /Users/jenniferkleiman/Documents/COMS
MISC="transcripts_annaB.xlsx transcripts_combined.xlsx
3Math_Ben_Day4_SG2_Arg2.drawing
Y2_whole_class_episodes.html whole_class_episodes.html
v04_processing_summary.json"
for f in $MISC; do
    [ -e "$f" ] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        git mv "$f" "misc/$f"
    else
        mv -v "$f" "misc/$f"
    fi
done
```

- [ ] **Step 2: Verify**

```bash
ls /Users/jenniferkleiman/Documents/COMS/misc/
```

---

## Task 13: Checkpoint — Review and Commit Moves

Before any deletion. Last chance to back out — everything so far is reversible.

- [ ] **Step 1: Show current state**

```bash
cd /Users/jenniferkleiman/Documents/COMS
echo "=== Top level ==="
ls -la | head -60
echo "=== git status ==="
git status --short | head -60
echo "=== Dir sizes ==="
du -sh transcription_runs archive misc sensitive _to_relocate timss_videos 2>/dev/null
```

- [ ] **Step 2: User review**

Prompt: "All moves done, no deletions yet. `git status` shows renames for tracked files; everything else is gitignored. OK to commit moves and proceed to deletions?" **Wait for yes.**

- [ ] **Step 3: Commit moves**

```bash
cd /Users/jenniferkleiman/Documents/COMS
git add -A archive/docs archive/scripts misc
git commit -m "$(cat <<'EOF'
chore(reorg): archive pre-v10 docs and scripts, consolidate misc

Tracked moves only (gitignored moves are invisible to git):
- pre-v10 status/fix docs → archive/docs/
- old pipeline versions, fix/patch scripts → archive/scripts/
- low-priority keep-for-now items → misc/

IRB/, keeper transcription runs, Clarke/GENIUS content, TIMSS videos,
and logs were also moved but are gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Deletion Batch A — Non-Keeper Transcription Runs

Per inventory: all run folders that have a newer complete v10 run for the same source video.

- [ ] **Step 1: Show file list and size**

```bash
cd /Users/jenniferkleiman/Documents/COMS
# Non-keepers = any *_transcription_* still at top level after Task 7
ls -d *_transcription_* *_v10_* 2>/dev/null | sort > /tmp/batch_a.txt
wc -l /tmp/batch_a.txt
du -sh $(cat /tmp/batch_a.txt) 2>/dev/null | tail -5
echo "=== Total Batch A size: ==="
du -sh $(cat /tmp/batch_a.txt) 2>/dev/null | awk '{print $1}' | head -1  # rough
```

- [ ] **Step 2: Sanity check against inventory**

Compare the list in `/tmp/batch_a.txt` to the non-keeper list in `REORG_INVENTORY.md`. Any surprise inclusions or omissions → stop and revisit inventory.

- [ ] **Step 3: User approval gate**

Prompt: "Batch A: delete N non-keeper run folders totaling X GB. Show the full list if needed. Proceed?" **Wait for yes.**

- [ ] **Step 4: Delete**

```bash
cd /Users/jenniferkleiman/Documents/COMS
while IFS= read -r f; do
    [ -d "$f" ] && rm -rf "$f" && echo "deleted: $f"
done < /tmp/batch_a.txt
```

- [ ] **Step 5: Verify top level is clean**

```bash
ls -d /Users/jenniferkleiman/Documents/COMS/*_transcription_* /Users/jenniferkleiman/Documents/COMS/*_v10_* 2>/dev/null && echo "UNEXPECTED leftovers" || echo "clean"
```

---

## Task 15: Deletion Batch B — chunks/ Subdirectories and video_chunks/

- [ ] **Step 1: Find all chunks/ subdirs inside keeper runs**

```bash
find /Users/jenniferkleiman/Documents/COMS/transcription_runs -type d -name "chunks" -maxdepth 2
du -sh /Users/jenniferkleiman/Documents/COMS/video_chunks 2>/dev/null
```

- [ ] **Step 2: Show counts and sizes**

```bash
find /Users/jenniferkleiman/Documents/COMS/transcription_runs -type d -name "chunks" -maxdepth 2 -exec du -sh {} \;
```

- [ ] **Step 3: User approval gate**

Prompt: "Batch B: delete N chunks/ subdirs inside kept runs + top-level video_chunks/ (287 MB). Proceed?" **Wait for yes.**

- [ ] **Step 4: Delete**

```bash
find /Users/jenniferkleiman/Documents/COMS/transcription_runs -type d -name "chunks" -maxdepth 2 -exec rm -rf {} +
rm -rf /Users/jenniferkleiman/Documents/COMS/video_chunks
```

- [ ] **Step 5: Verify**

```bash
find /Users/jenniferkleiman/Documents/COMS/transcription_runs -type d -name "chunks" 2>/dev/null && echo "UNEXPECTED leftovers" || echo "clean"
test ! -e /Users/jenniferkleiman/Documents/COMS/video_chunks && echo "video_chunks removed"
```

---

## Task 16: Deletion Batch C — Raw COMS Videos at Top Level

- [ ] **Step 1: List top-level videos that are NOT TIMSS and NOT Clarke**

```bash
cd /Users/jenniferkleiman/Documents/COMS
ls -lh Y2_4S_Ava_Day*.mp4 250415_4Math_Ava_Day1.mp4 250416_4Math_Ava_Day2.mp4 2>/dev/null
```

- [ ] **Step 2: Also scan for any other top-level mp4/mov/m4a/MP3 NOT in timss_videos/ or _to_relocate/**

```bash
cd /Users/jenniferkleiman/Documents/COMS
find . -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4a" -o -iname "*.MP3" \)
```
Any surprise files → ask user rather than deleting.

- [ ] **Step 3: User approval gate**

Prompt: "Batch C: delete top-level raw COMS videos — shown list. (Everything in Teams; TIMSS already in timss_videos/; Clarke in _to_relocate/.) Proceed?" **Wait for yes.**

- [ ] **Step 4: Delete**

```bash
cd /Users/jenniferkleiman/Documents/COMS
for f in Y2_4S_Ava_Day*.mp4 250415_4Math_Ava_Day1.mp4 250416_4Math_Ava_Day2.mp4; do
    [ -e "$f" ] && rm -v "$f"
done
```

---

## Task 17: Deletion Batch D — Loose Chunk Transcripts

Top-level `chunk_01_v04_transcript.txt` through `chunk_XX_v04_transcript.txt` (per spec).

- [ ] **Step 1: List**

```bash
cd /Users/jenniferkleiman/Documents/COMS
ls chunk_[0-9]*_v04_transcript.txt 2>/dev/null
```

- [ ] **Step 2: User approval gate**

Prompt: "Batch D: delete loose top-level chunk transcripts (chunk_NN_v04_transcript.txt). These are stray fragments from old runs; full transcripts live in kept run folders. Proceed?" **Wait for yes.**

- [ ] **Step 3: Delete**

```bash
cd /Users/jenniferkleiman/Documents/COMS
rm -v chunk_[0-9]*_v04_transcript.txt
```

---

## Task 18: Pipeline Code Change — Extend `_cleanup_chunks` to also remove per-chunk transcript files

**Context:** `video_transcription_pipeline_v10.py` already deletes the `chunks/` directory on successful runs when `keep_chunks=False` (default). However, the per-chunk transcript files at `{output_dir}/chunk_NNN_transcript.txt` are NOT cleaned up. The comment at lines 1553-1557 claims they are, but the code doesn't match. Those files contain original PII-bearing transcripts (pre-deidentification). This is a real gap.

**Files:**
- Modify: `/Users/jenniferkleiman/Documents/COMS/video_transcription_pipeline_v10.py`
  - `_cleanup_chunks` method at line 1819 → accept `output_dir` param, also delete per-chunk transcript files
  - Call site at line 1618 → pass `output_dir`
- Create: `/Users/jenniferkleiman/Documents/COMS/test_cleanup_chunks.py` (new TDD test)

- [ ] **Step 1: Write failing test**

Create `/Users/jenniferkleiman/Documents/COMS/test_cleanup_chunks.py`:
```python
"""Test that _cleanup_chunks removes both chunks/ dir AND per-chunk transcript .txt files."""
import shutil
import tempfile
from pathlib import Path

import pytest

from video_transcription_pipeline_v10 import (
    TranscriptionConfigV10,
    TranscriptionEngineV10,
)


@pytest.fixture
def fake_run_dir():
    d = Path(tempfile.mkdtemp(prefix="cleanup_test_"))
    (d / "chunks").mkdir()
    (d / "chunks" / "chunk_001.mp4").write_bytes(b"fake video")
    (d / "chunks" / "chunk_002.mp4").write_bytes(b"fake video")
    # Per-chunk transcript files (PII-bearing in real runs)
    (d / "chunk_001_transcript.txt").write_text("chunk 1 text")
    (d / "chunk_002_transcript.txt").write_text("chunk 2 text")
    (d / "chunk_003_transcript.txt").write_text("chunk 3 text")
    # Keep files (must NOT be deleted)
    (d / "foo_transcript.txt").write_text("final transcript")
    (d / "foo_speakers.json").write_text("{}")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_engine():
    cfg = TranscriptionConfigV10(api_key="fake")
    # Bypass __init__ network/client setup by directly instantiating with object.__new__
    eng = object.__new__(TranscriptionEngineV10)
    eng.config = cfg
    return eng


def test_cleanup_removes_chunks_dir_and_per_chunk_txt(fake_run_dir):
    eng = _make_engine()
    eng._cleanup_chunks(fake_run_dir / "chunks", fake_run_dir)

    assert not (fake_run_dir / "chunks").exists(), "chunks/ dir should be removed"
    assert not (fake_run_dir / "chunk_001_transcript.txt").exists()
    assert not (fake_run_dir / "chunk_002_transcript.txt").exists()
    assert not (fake_run_dir / "chunk_003_transcript.txt").exists()
    # Keepers
    assert (fake_run_dir / "foo_transcript.txt").exists(), "final transcript must survive"
    assert (fake_run_dir / "foo_speakers.json").exists(), "speaker manifest must survive"


def test_cleanup_tolerates_missing_chunks_dir(fake_run_dir):
    shutil.rmtree(fake_run_dir / "chunks")
    eng = _make_engine()
    # Should not raise even though chunks/ is gone
    eng._cleanup_chunks(fake_run_dir / "chunks", fake_run_dir)
    # Per-chunk txt files still removed
    assert not (fake_run_dir / "chunk_001_transcript.txt").exists()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
cd /Users/jenniferkleiman/Documents/COMS
source venv/bin/activate 2>/dev/null || true
python -m pytest test_cleanup_chunks.py -v
```
Expected: FAIL — current `_cleanup_chunks` takes only `chunks_dir`, not `output_dir`; per-chunk .txt files remain.

- [ ] **Step 3: Implement change in pipeline**

Edit `/Users/jenniferkleiman/Documents/COMS/video_transcription_pipeline_v10.py`:

**Change 1 — modify `_cleanup_chunks` method (currently at line 1819):**

```python
    def _cleanup_chunks(self, chunks_dir: Path, output_dir: Path = None):
        """Remove temporary chunk files and per-chunk transcript .txt files.

        The per-chunk transcript files (chunk_NNN_transcript.txt) hold the
        original PII-bearing text before any de-identification pass, so they
        must be removed on successful completion unless --keep-chunks is set.
        """
        try:
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
                print(f"  Cleaned up: {chunks_dir}")
        except Exception as e:
            print(f"  Cleanup warning (chunks dir): {e}")

        if output_dir is not None:
            try:
                for p in output_dir.glob("chunk_*_transcript.txt"):
                    p.unlink()
                    print(f"  Cleaned up: {p.name}")
            except Exception as e:
                print(f"  Cleanup warning (per-chunk txt): {e}")
```

**Change 2 — update the call site (currently at line 1618):**

Old:
```python
            # Cleanup
            if not self.config.keep_chunks:
                self._cleanup_chunks(chunks_dir)
```

New:
```python
            # Cleanup
            if not self.config.keep_chunks:
                self._cleanup_chunks(chunks_dir, output_dir)
```

- [ ] **Step 4: Run test, verify it passes**

```bash
cd /Users/jenniferkleiman/Documents/COMS
python -m pytest test_cleanup_chunks.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Run existing test suite to check for regression**

```bash
cd /Users/jenniferkleiman/Documents/COMS
python -m pytest test_deidentify_names.py -v 2>&1 | tail -20
```
Expected: same pass/fail count as before the change (no new failures).

- [ ] **Step 6: Commit**

```bash
cd /Users/jenniferkleiman/Documents/COMS
git add video_transcription_pipeline_v10.py test_cleanup_chunks.py
git commit -m "$(cat <<'EOF'
fix(pipeline): also remove per-chunk transcript .txt files in cleanup

_cleanup_chunks previously only removed the chunks/ video directory, but
the per-chunk transcript .txt files (chunk_NNN_transcript.txt) contain
original PII-bearing transcripts pre-deidentification. The docstring at
the phase-6.5 call site already claimed they were cleaned up — this
makes the code match the contract. --keep-chunks continues to preserve
both for debugging/resume.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Final Review and Summary

- [ ] **Step 1: Show final state**

```bash
cd /Users/jenniferkleiman/Documents/COMS
echo "=== Top level count ==="
ls | wc -l
echo "=== Top level entries ==="
ls
echo "=== Total size ==="
du -sh .
echo "=== Key dir sizes ==="
du -sh transcription_runs archive sensitive misc timss_videos _to_relocate 2>/dev/null
echo "=== git status ==="
git status --short
```

- [ ] **Step 2: Update REORG_INVENTORY.md with final tallies**

Append a "Done" section with actual numbers: final top-level count, total GB reclaimed, what's where now.

- [ ] **Step 3: Show user**

Present the diff between "before" numbers (from Task 1 inventory) and "after" (final state).

---

## Self-Review

**Spec coverage check:**
- Target directory structure (all dirs) → Tasks 2, 7 (transcription_runs populated), 3-4 (sensitive), 5 (_to_relocate), 6 (timss_videos), 9-11 (archive/*), 12 (misc). ✓
- Categorization rules for archive/docs, archive/scripts, archive/logs → Tasks 9, 10, 11. ✓
- Sensitive content handling → Tasks 3 (name-maps), 4 (IRB). ✓
- Keeper-selection rule → Task 1 Step 3. ✓
- TIMSS identification → Task 1 Step 4, Task 6. ✓
- Four deletion batches A/B/C/D → Tasks 14, 15, 16, 17. ✓
- Pipeline chunk-cleanup change → Task 18. ✓
- Approval-gated deletions → User-approval steps in each deletion task. ✓
- Git history preserved via `git mv` → Tasks 9-12 Step 2. ✓
- Three commits (structure+gitignore / moves / pipeline) → Tasks 2 Step 5, 13 Step 3, 18 Step 6. ✓
- Inventory-triage for loose complete-transcript files → Task 8. ✓
- `dev/active/paper/*` untouched → not referenced anywhere in plan. ✓

**Placeholder scan:** no TBD/TODO; all code steps contain actual code; all command steps contain actual commands. ✓

**Type consistency:** `_cleanup_chunks(chunks_dir, output_dir=None)` signature matches both the implementation step and the test's call. ✓
