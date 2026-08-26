"""Diarization via pyannote.audio 4.x + community-1.

community-1 is chosen over speaker-diarization-3.1 for two reasons: markedly
lower speaker confusion on real-world noisy audio, and native
`exclusive_speaker_diarization` — a one-speaker-at-a-time timeline built for
alignment against ASR word timestamps. Deriving that ourselves from an
overlapping timeline is guesswork; getting it from the model is not.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from scribe.types import Diarization, Segment

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

_GATED_HELP = """\
Could not load {model!r}.

This model is gated. One-time setup:
  1. Accept the terms at https://huggingface.co/{model}
  2. Create a read token at https://huggingface.co/settings/tokens
  3. Export it:  export HF_TOKEN=hf_...

Original error: {err}"""


class DiarizationError(RuntimeError):
    pass


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device.

    'auto' prefers CUDA, then MPS, then CPU. MPS is *not* a guaranteed win here —
    several pyannote ops fall back to CPU and the copies can cost more than the
    acceleration saves — so the choice stays overridable and worth benchmarking
    on real audio rather than assuming.
    """
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_waveform(wav: Path) -> dict:
    """Read a WAV into the dict form pyannote accepts.

    Deliberately not passing a file path. pyannote 4.x decodes paths through
    torchcodec, which links against a specific FFmpeg major version and breaks
    against whatever the host actually has (Homebrew ships 8.x; torchcodec
    probes for 4.x first and raises). We have already normalized to 16 kHz mono
    upstream, so handing over the samples directly removes that coupling
    entirely, skips a redundant decode, and behaves identically in a container
    where the FFmpeg version will differ again.
    """
    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(str(wav), dtype="float32", always_2d=True)
    # soundfile gives (frames, channels); pyannote wants (channels, frames).
    waveform = torch.from_numpy(samples.T).contiguous()
    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
        "uri": wav.stem,
    }


class PyannoteDiarizer:
    name = "pyannote"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        token: str | None = None,
        device: str = "auto",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        self.model_id = model
        self._token = token
        self._device = device
        self._num_speakers = num_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._on_progress = on_progress
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from pyannote.audio import Pipeline

        device = resolve_device(self._device)
        if device == "mps":
            # Some ops have no MPS kernel; without this the load hard-fails
            # instead of quietly running those ops on CPU.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        try:
            pipeline = Pipeline.from_pretrained(self.model_id, token=self._token)
        except Exception as err:  # noqa: BLE001 - surfaced with setup guidance
            raise DiarizationError(
                _GATED_HELP.format(model=self.model_id, err=err)
            ) from err

        if pipeline is None:
            raise DiarizationError(
                _GATED_HELP.format(
                    model=self.model_id,
                    err="from_pretrained returned None (usually means the token "
                    "is missing or the licence has not been accepted)",
                )
            )

        self._pipeline = pipeline.to(torch.device(device))
        self.device = device
        return self._pipeline

    def _constraints(self) -> dict[str, int]:
        # Only forward what was set: passing num_speakers=None alongside
        # min/max is an error in pyannote.
        out: dict[str, int] = {}
        if self._num_speakers is not None:
            out["num_speakers"] = self._num_speakers
        else:
            if self._min_speakers is not None:
                out["min_speakers"] = self._min_speakers
            if self._max_speakers is not None:
                out["max_speakers"] = self._max_speakers
        return out

    def diarize(self, wav: Path) -> Diarization:
        pipeline = self._load()
        hook = _ProgressHook(self._on_progress) if self._on_progress else None

        output = pipeline(
            _load_waveform(Path(wav)), hook=hook, **self._constraints()
        )

        exclusive = _to_segments(
            getattr(output, "exclusive_speaker_diarization", None)
        )
        overlapped = _to_segments(getattr(output, "speaker_diarization", output))

        if not exclusive:
            # Older pipelines don't expose the exclusive timeline; fall back to
            # deriving one so the merge stage's contract still holds.
            from scribe.diarize.base import derive_exclusive

            exclusive = derive_exclusive(overlapped)

        if self._on_progress:
            self._on_progress(1.0)
        return Diarization(exclusive=exclusive, overlapped=overlapped)


def _to_segments(annotation) -> list[Segment]:
    """Flatten a pyannote `Annotation` into our own segment list."""
    if annotation is None:
        return []
    out = [
        Segment(speaker=str(label), start=float(seg.start), end=float(seg.end))
        for seg, _track, label in annotation.itertracks(yield_label=True)
        if seg.end > seg.start
    ]
    out.sort(key=lambda s: (s.start, s.end))
    return out


class _ProgressHook:
    """Adapts pyannote's staged hook to a single 0..1 fraction.

    pyannote reports progress per internal step (segmentation, embedding,
    clustering) with no global total, so weight the steps by their rough share of
    wall time to get a monotonic overall number.
    """

    _WEIGHTS = {"segmentation": 0.45, "embeddings": 0.45}

    def __init__(self, report: Callable[[float], None]) -> None:
        self._report = report
        self._done = 0.0

    def __call__(
        self,
        step_name: str,
        step_artifact=None,
        file=None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        weight = self._WEIGHTS.get(step_name, 0.05)
        if total and completed is not None:
            fraction = self._done + weight * (completed / total)
            self._report(min(0.99, fraction))
            if completed >= total:
                self._done = min(0.99, self._done + weight)
        elif completed is None and total is None:
            self._report(min(0.99, self._done))

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None
