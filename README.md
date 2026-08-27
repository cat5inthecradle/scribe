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

### One file, right now

```bash
uv run scribe run recording.m4a
```

No database needed. Best for trying things out and for judging output quality.

### The folder-drop service

```bash
docker compose up -d          # Postgres
uv run scribe db upgrade      # apply migrations
uv run scribe dev             # watch intake + run one worker
```

Drop files into `data/intake/` and they get transcribed. `scribe dev` is the
convenience form; the pieces run separately in production:

```bash
uv run scribe watch     # queue files as they settle (run exactly one)
uv run scribe worker    # process the queue (run as many as you like)
```

Inspect and control the queue:

```bash
uv run scribe queue --watch    # live status
uv run scribe logs 77748a40    # one job's event log (id prefix is enough)
uv run scribe submit a.m4a     # enqueue explicitly
uv run scribe retry 77748a40   # revive a dead job
uv run scribe scan             # one-shot intake scan
```

### Where files go

| directory | holds | safe to delete? |
|---|---|---|
| `data/intake/` | files you drop in | yes |
| `data/archive/` | **your original recordings**, moved out of intake | **no** |
| `data/work/` | scratch (normalized WAV) | yes, any time |
| `data/out/` | transcripts | no |

Intake **moves** files rather than copying, so `archive/` briefly holds the only
copy of a recording. That is why originals never go in `work/`, which is scratch
and gets emptied.

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

## Queue design

`jobs` is both the record and the queue. Claiming a job is
`SELECT ... FOR UPDATE SKIP LOCKED`, so many workers can race for the same row
and exactly one wins without any of them blocking. That single primitive is why
there is no Redis, Celery, or RabbitMQ here.

- **Dedupe by content hash** — re-dropping a recording under a new name is the
  same job. `--force` re-runs it.
- **Crash recovery** — a worker holds its job with a heartbeat. If the process
  dies, the heartbeat lapses and another worker reclaims the job. Nothing needs
  unwinding on an abrupt exit, so SIGTERM just stops claiming new work.
- **Bounded retries with backoff** — failures wait 15s, then 60s, then 300s
  before the next attempt. Without the delay a job that fails in 50ms would burn
  every attempt in a tenth of a second, and retrying would be pointless.

Run workers on the **host**, not in a container, to get Metal acceleration.

## A note on credentials

The Postgres username and password default to `scribe`/`scribe`, and the
container publishes only to `127.0.0.1:5433`. That is deliberate convenience for
a laptop, **not** a safe default for anything a network can reach. Before
deploying, set `POSTGRES_PASSWORD` and `SCRIBE_DATABASE_URL` to real values from
a secret store.

Nothing in this repository should ever hold a Hugging Face token: `hf auth login`
stores one outside the project, and containers should receive it as an injected
env var. `data/` — your recordings, archived originals, and transcripts — is
gitignored, along with media and transcript filenames anywhere in the tree.

## Configuration

Environment (`SCRIBE_*`) or `scribe.toml`. Every merge threshold is tunable
because the right values depend on the audio:

```toml
[merge]
nearest_window_s = 0.25       # how far a word reaches for a speaker
min_segment_s = 0.12          # ignore segments too short to own a word
smooth_max_words = 5          # runs this short can be absorbed...
smooth_max_duration_s = 0.4   # ...if they're also this brief
turn_gap_s = 1.5              # silence that splits a turn
```

## Development

```bash
uv run pytest          # 155 tests; models are stubbed
uv run ruff check src tests
```

The test suite stubs the models out, so it stays fast and needs no Hugging Face
token. Queue and intake tests need Postgres (`docker compose up -d`) and skip
cleanly without it — they use a real database because the correctness argument
rests on `SKIP LOCKED`, which no fake reproduces.
