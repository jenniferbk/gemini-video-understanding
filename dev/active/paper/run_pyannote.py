#!/usr/bin/env python3
"""
Run pyannote.audio speaker diarization on a WAV file and align with a
Whisper transcript to produce a Whisper+pyannote combined transcript in the
same MM:SS format as our v10 outputs (so the benchmark scorer can read it).

Usage:
    python run_pyannote.py \
        --audio dev/active/paper/benchmark_runs/US1.wav \
        --whisper dev/active/paper/benchmark_runs/whisper_us1/result.json \
        --token hf_xxxxxxxx \
        --out dev/active/paper/benchmark_runs/whisper_pyannote_us1/US1_wpy_transcript.txt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True, type=Path)
    ap.add_argument('--whisper', required=True, type=Path)
    ap.add_argument('--token', required=True)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    from pyannote.audio import Pipeline
    print("Loading pyannote/speaker-diarization-3.1 ...", flush=True)
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=args.token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=args.token,
        )

    print(f"Diarizing {args.audio} ...", flush=True)
    # torchaudio + pyannote's native file loader both route through torchcodec
    # which on this machine can't find an older libavutil. Load the WAV with
    # soundfile directly (pure libsndfile, no ffmpeg) and convert to a torch
    # tensor matching pyannote's expected (channels, samples) shape.
    import soundfile as sf
    import torch
    audio_np, sample_rate = sf.read(str(args.audio), dtype='float32', always_2d=True)
    # soundfile gives (samples, channels); pyannote wants (channels, samples).
    waveform = torch.from_numpy(audio_np.T.copy())
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Collect (start, end, speaker_label) segments.
    # pyannote 4.x returns a DiarizeOutput with .speaker_diarization; older
    # versions return an Annotation directly. Handle both.
    ann = getattr(diarization, 'speaker_diarization', diarization)
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    segments.sort(key=lambda x: x[0])
    print(f"  {len(segments)} diarization segments", flush=True)
    speakers_seen = sorted({s for _,_,s in segments})
    print(f"  distinct speakers: {speakers_seen}", flush=True)

    # Load whisper word/segment output.
    whisper = json.loads(args.whisper.read_text())
    whisper_segs = whisper.get('segments', [])

    # For each whisper segment, assign the speaker whose diarization
    # segment has the most temporal overlap with it.
    def assign_speaker(ws_start: float, ws_end: float) -> str:
        best_speaker = None
        best_overlap = 0.0
        for s, e, sp in segments:
            overlap = max(0.0, min(e, ws_end) - max(s, ws_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp
        return best_speaker or 'UNK'

    out_lines = ['=== WHISPER large-v3 + pyannote 3.1 ===', '']
    for ws in whisper_segs:
        start = ws['start']
        text = ws['text'].strip()
        if not text:
            continue
        sp = assign_speaker(start, ws['end'])
        mm = int(start // 60); ss = int(start % 60)
        out_lines.append(f"{mm:02d}:{ss:02d} {sp}: {text}")

    args.out.write_text('\n'.join(out_lines) + '\n')
    print(f"Wrote {args.out}  ({len(out_lines)-2} segments)")


if __name__ == '__main__':
    main()
