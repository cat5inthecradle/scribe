# scribe

Local audio transcription with speaker separation. Drop a file in, get an
accurate speaker-attributed transcript out. No cloud, no per-hour cost, no audio
leaving the machine.

## Setup

```bash
brew install ffmpeg          # if you don't have it
uv sync --extra mlx --extra diarize
```

The diarization model is **gated**, so this is a required one-time step:

1. Accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>
2. Create a read token at <https://huggingface.co/settings/tokens>
3. `export HF_TOKEN=hf_...`

Then check everything is in place:

```bash
uv run scribe doctor
```

## Use

```bash
uv run scribe run recording.m4a
```

Outputs land in `data/out/<name>-<date>-<hash>/`:

| file | what it's for |
|---|---|
| `transcript.json` | canonical, word-level timings — everything else derives from it |
| `transcript.md` | the readable one |
| `transcript.txt` | flat `[time] Speaker: text`, grep-friendly |
| `transcript.srt` / `.vtt` | subtitles, re-chunked to readable cue lengths |
| `speakers.yaml` | map speakers to real names |

### Naming speakers

Diarization can tell voices apart but not who they belong to. Edit the `name:`
fields in `speakers.yaml`, then:

```bash
uv run scribe rerender data/out/<dir>
```

That re-runs **no inference** — word-level timings live in `transcript.json`, so
renaming is a pure re-render.

### Useful flags

```bash
uv run scribe run call.mp4 --speakers 3      # pin the count when you know it
uv run scribe run call.mp4 --max-speakers 5  # or just bound it
uv run scribe run call.mp4 --device cpu      # MPS is not always faster
```

Telling it the speaker count materially improves the split. Video files work —
ffmpeg drops the video stream.

## How it works

```
intake  →  normalize   →  ASR         ─┐
           (ffmpeg,       (words +     ├→  merge  →  render
            16k mono       timestamps) │   (word↔     (json, md,
            wav)                       │    speaker)   srt, vtt, txt)
                          diarize     ─┘
                          (exclusive timeline)
```

**ASR** is NVIDIA Parakeet TDT. It is a transducer, so word-level timestamps come
out natively — no forced-alignment pass, which is why WhisperX isn't used here.
English-focused and faster and more accurate than Whisper for it.

**Diarization** is pyannote `community-1`. Beyond being clearly better than 3.1
on noisy real-world audio, it exposes `exclusive_speaker_diarization` — a
one-speaker-at-a-time timeline built for aligning against ASR word timestamps.
That removes the overlap-conflict guesswork from attribution entirely.

**Merge** ([`src/scribe/merge.py`](src/scribe/merge.py)) is where quality is won
or lost: it assigns each word a speaker by timeline overlap, smooths out
single-word speaker flapping (the most visible diarization artifact), and groups
words into readable turns. It has no model dependencies, so it is tested
exhaustively with plain fixtures.

### Two runtimes, one model

MLX needs Metal, which containers on macOS cannot reach — so the fast path is a
native host process and the portable path is CPU-only.

| | runtime | speed |
|---|---|---|
| host (macOS) | `parakeet-mlx` → Metal | ~70x realtime on an M5 |
| container (k8s) | Parakeet INT8 ONNX → `sherpa-onnx` | ~1–3x realtime, CPU |

Both are the same Parakeet model, so transcripts stay comparable. Select with
`--backend mlx|onnx`. Kubernetes buys always-on availability here, not speed.

## Configuration

Environment (`SCRIBE_*`) or `scribe.toml`. Every merge threshold is tunable
because the right values depend on the audio:

```toml
[merge]
nearest_window_s = 0.25       # how far a word reaches for a speaker
smooth_max_words = 2          # runs this short can be absorbed...
smooth_max_duration_s = 0.4   # ...if they're also this brief
turn_gap_s = 1.5              # silence that splits a turn
```

## Development

```bash
uv run pytest          # 68 tests, no models or network needed
uv run ruff check src tests
```

The test suite deliberately stubs the models out, so it stays fast and runs
without a Hugging Face token.
