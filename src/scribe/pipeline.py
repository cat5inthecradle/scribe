"""Stage orchestration.

ASR and diarization are independent — both only need the normalized WAV — but
they run sequentially because on a single machine they contend for the same
cores, and interleaving them would slow both down. Per-stage wall times are
recorded so that assumption stays checkable with data rather than belief.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from scribe import media
from scribe import speakers as sidecar
from scribe.config import Settings
from scribe.merge import merge
from scribe.render import render_all
from scribe.types import (
    BackendInfo,
    PipelineInfo,
    SourceInfo,
    Transcript,
)

STAGES = ("inspect", "normalize", "transcribe", "diarize", "merge", "render")

ProgressFn = Callable[[str, float], None]
"""Called as ``(stage, fraction)``. `fraction` is 0..1 within that stage."""


@dataclass
class PipelineResult:
    transcript: Transcript
    out_dir: Path
    written: list[Path] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)


def slugify(name: str) -> str:
    """Filesystem-safe stem for an output directory."""
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return (slug or "audio")[:60]


def output_dir_for(
    settings: Settings, info: media.MediaInfo, display_name: str | None = None
) -> Path:
    """``out/<slug>-<date>-<hash8>``.

    The hash suffix keeps two same-named recordings from colliding, and makes the
    directory name a stable function of content — re-running a file lands in the
    same place and overwrites cleanly.

    `display_name` matters when the file on disk is not what the user called it:
    the intake watcher parks sources at ``work/<job id>/source.ext``, so slugging
    the actual path would name every queued output "source".
    """
    date = datetime.now(UTC).strftime("%Y%m%d")
    stem = slugify(display_name or info.path.name)
    return settings.out / f"{stem}-{date}-{info.sha256[:8]}"


class _Timer:
    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    def __call__(self, stage: str):
        return _Span(self, stage)


class _Span:
    def __init__(self, timer: _Timer, stage: str) -> None:
        self._timer, self._stage = timer, stage

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._timer.durations[self._stage] = round(time.perf_counter() - self._t0, 3)


def run(
    source: Path,
    settings: Settings,
    *,
    out_dir: Path | None = None,
    on_progress: ProgressFn | None = None,
    keep_wav: bool = False,
    display_name: str | None = None,
) -> PipelineResult:
    """Transcribe and diarize `source`, writing every output format.

    `display_name` overrides the name recorded in the transcript and used for
    the output directory. The queue path needs it because the file it hands over
    has been renamed to a neutral ``source.ext`` under the work directory.
    """
    report: ProgressFn = on_progress or (lambda _s, _f: None)
    timer = _Timer()

    with timer("inspect"):
        report("inspect", 0.0)
        info = media.inspect(source)
        report("inspect", 1.0)

    name = display_name or info.path.name
    out_dir = out_dir or output_dir_for(settings, info, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = settings.work / info.sha256[:16]
    wav = work / "audio.wav"

    with timer("normalize"):
        report("normalize", 0.0)
        media.normalize(source, wav)
        report("normalize", 1.0)

    try:
        with timer("transcribe"):
            # The pipeline owns stage boundaries so the label advances even for a
            # backend that reports no intra-stage progress; backends only refine.
            report("transcribe", 0.0)
            from scribe.asr.base import load_backend as load_asr

            asr = load_asr(
                settings.asr.backend,
                model=(
                    settings.asr.mlx_model
                    if settings.asr.backend == "mlx"
                    else settings.asr.onnx_model_dir
                ),
                on_progress=lambda f: report("transcribe", f),
            )
            words = asr.transcribe(wav)
            report("transcribe", 1.0)

        with timer("diarize"):
            report("diarize", 0.0)
            from scribe.diarize.base import load_backend as load_diar

            diarizer = load_diar(
                "pyannote",
                model=settings.diarize.model,
                token=settings.resolved_hf_token(),
                device=settings.diarize.device,
                num_speakers=settings.diarize.num_speakers,
                min_speakers=settings.diarize.min_speakers,
                max_speakers=settings.diarize.max_speakers,
                clustering_threshold=settings.diarize.clustering_threshold,
                min_duration_off=settings.diarize.min_duration_off,
                on_progress=lambda f: report("diarize", f),
            )
            diarization = diarizer.diarize(wav)
            report("diarize", 1.0)

        with timer("merge"):
            report("merge", 0.0)
            turns, speaker_list = merge(words, diarization, settings.merge)
            report("merge", 1.0)

        transcript = Transcript(
            source=SourceInfo(
                filename=name,
                sha256=info.sha256,
                duration_s=round(info.duration_s, 3),
                bytes=info.bytes,
            ),
            pipeline=PipelineInfo(
                asr=BackendInfo(backend=asr.name, model=str(asr.model_id)),
                diarization=BackendInfo(
                    backend=diarizer.name, model=diarizer.model_id
                ),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                durations_s=timer.durations,
            ),
            speakers=speaker_list,
            turns=turns,
        )

        # Voice embeddings are written beside the outputs so speakers can be
        # identified later without re-running any inference.
        candidates: dict[str, list[dict[str, object]]] = {}
        if diarization.embeddings:
            write_embeddings(out_dir, diarization.embeddings, transcript)
            auto, candidates = identify_speakers(
                settings, diarization.embeddings
            )
            for speaker in transcript.speakers:
                if name := auto.get(speaker.id):
                    speaker.name = name

        # Names set by hand always win over an automatic match, and survive
        # a re-run of the same audio.
        if names := sidecar.read(out_dir):
            transcript = sidecar.apply(transcript, names)

        with timer("render"):
            report("render", 0.0)
            transcript.pipeline.durations_s = dict(timer.durations)
            written = render_all(transcript, out_dir)
            written.append(sidecar.write(transcript, out_dir, candidates))
            report("render", 1.0)

    finally:
        # Only the derived WAV is removed, and only by name. `work` holds
        # nothing irreplaceable by design (see Settings.archive), but deleting
        # narrowly rather than recursively keeps that true even if that changes.
        if not keep_wav:
            wav.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                work.rmdir()  # succeeds only if empty

    return PipelineResult(
        transcript=transcript,
        out_dir=out_dir,
        written=written,
        durations=timer.durations,
    )


EMBEDDINGS_FILE = "embeddings.json"


def write_embeddings(
    out_dir: Path, embeddings: dict[str, list[float]], transcript: Transcript
) -> Path:
    """Save per-speaker voice embeddings alongside the transcript.

    Kept with the outputs rather than in the database so an output directory
    stays self-describing, and so `scribe identify` can name speakers later
    without re-running diarization on the audio — which may since have been
    archived or deleted.
    """
    import json

    speech = {s.id: s.speech_s for s in transcript.speakers}
    payload = {
        "speakers": [
            {
                "id": label,
                "label": next(
                    (s.label for s in transcript.speakers if s.id == label), label
                ),
                "speech_s": speech.get(label, 0.0),
                "embedding": vector,
            }
            for label, vector in embeddings.items()
        ]
    }
    path = out_dir / EMBEDDINGS_FILE
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def read_embeddings(out_dir: Path) -> dict[str, list[float]]:
    """Load saved embeddings, or an empty dict if there are none."""
    import json

    path = out_dir / EMBEDDINGS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        entry["id"]: entry["embedding"]
        for entry in data.get("speakers", [])
        if entry.get("id") and entry.get("embedding")
    }


def identify_speakers(
    settings: Settings, embeddings: dict[str, list[float]]
) -> tuple[dict[str, str], dict[str, list[dict[str, object]]]]:
    """Match diarized speakers against the enrolled-voice library.

    Returns confident names and, separately, the closest candidates for every
    speaker so a human can confirm the near misses.

    Deliberately best-effort: `scribe run` is meant to work with no database at
    all, so an unreachable Postgres degrades to anonymous speakers rather than
    failing a transcription that is otherwise complete.
    """
    from scribe import voices
    from scribe.db import session_scope

    try:
        with session_scope(settings) as session:
            if not voices.enrolled(session):
                return {}, {}

            matched = voices.identify_all(
                session, embeddings, settings.diarize.embedding_match_threshold
            )
            names = {label: m.name for label, m in matched.items()}
            hints = {
                label: [
                    {"name": m.name, "similarity": round(m.similarity, 3)}
                    for m in voices.rank(session, vector, limit=3)
                ]
                for label, vector in embeddings.items()
            }
            return names, hints
    except Exception:  # noqa: BLE001 - enrollment is an enhancement, not a requirement
        return {}, {}


def rerender(out_dir: Path) -> PipelineResult:
    """Re-render an existing output directory after a speaker rename.

    Reads `transcript.json`, applies `speakers.yaml`, and rewrites every derived
    format. Runs no inference — which is the whole point of keeping word-level
    detail in the canonical JSON.
    """
    path = out_dir / "transcript.json"
    if not path.is_file():
        raise FileNotFoundError(f"no transcript.json in {out_dir}")

    transcript = Transcript.from_json(path.read_text(encoding="utf-8"))
    transcript = sidecar.apply(transcript, sidecar.read(out_dir))
    written = render_all(transcript, out_dir)
    written.append(sidecar.write(transcript, out_dir))
    return PipelineResult(
        transcript=transcript, out_dir=out_dir, written=written, durations={}
    )
