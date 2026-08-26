"""Core data types.

These are the contract between pipeline stages. ASR backends produce `Word`s,
diarization backends produce `Segment`s, `merge` combines them into `Turn`s, and
renderers consume a `Transcript`.

The pydantic models are the on-disk schema for `transcript.json` and round-trip
exactly, so a transcript can be re-rendered (e.g. after renaming speakers)
without re-running any inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

# pyannote's own label for "no speaker could be attributed".
UNKNOWN_SPEAKER = "UNKNOWN"


class Word(BaseModel):
    """A single token with timing, as emitted by an ASR backend."""

    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(alias="t")
    start: float
    end: float
    conf: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous stretch attributed to one speaker by diarization.

    Internal only — never serialized. `speaker` is the backend's raw label
    (e.g. ``SPEAKER_00``), not a display label.
    """

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, start: float, end: float) -> float:
        """Seconds of overlap with the interval ``[start, end]``. Never negative."""
        return max(0.0, min(self.end, end) - max(self.start, start))


@dataclass(frozen=True, slots=True)
class Diarization:
    """Output of a diarization backend.

    `exclusive` is the one-speaker-at-a-time timeline: overlapping speech has
    already been resolved to a single speaker per instant. This is what `merge`
    aligns ASR words against. pyannote community-1 produces it directly; a
    backend without native support must derive one.

    `overlapped` is the raw diarization, retained for reporting (it is the only
    place crosstalk is visible) but not used for word attribution.
    """

    exclusive: list[Segment]
    overlapped: list[Segment]

    @property
    def speakers(self) -> list[str]:
        """Raw speaker labels, ordered by first appearance in the timeline."""
        seen: dict[str, None] = {}
        for seg in sorted(self.exclusive, key=lambda s: s.start):
            seen.setdefault(seg.speaker, None)
        return list(seen)


class Speaker(BaseModel):
    """A speaker in the finished transcript."""

    id: str
    """Backend's raw label, e.g. ``SPEAKER_00``. Stable within one transcript only."""

    label: str
    """Display label assigned by first appearance, e.g. ``Speaker 1``."""

    name: str | None = None
    """Human name, from ``speakers.yaml``. None until someone fills it in."""

    speech_s: float = 0.0
    """Total attributed speaking time, for a quick sanity check on the split."""

    @property
    def display(self) -> str:
        return self.name or self.label


class Turn(BaseModel):
    """Consecutive words from one speaker, the unit of a readable transcript."""

    speaker: str
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class SourceInfo(BaseModel):
    filename: str
    sha256: str
    duration_s: float
    bytes: int | None = None


class BackendInfo(BaseModel):
    backend: str
    model: str


class PipelineInfo(BaseModel):
    asr: BackendInfo
    diarization: BackendInfo
    created_at: str
    durations_s: dict[str, float] = Field(default_factory=dict)
    """Per-stage wall time. Makes it obvious whether a re-render skipped inference."""


class Transcript(BaseModel):
    """The canonical artifact. Everything else is rendered from this."""

    schema_version: int = SCHEMA_VERSION
    source: SourceInfo
    pipeline: PipelineInfo
    speakers: list[Speaker]
    turns: list[Turn]

    def speaker_map(self) -> dict[str, Speaker]:
        return {s.id: s for s in self.speakers}

    def display_for(self, speaker_id: str) -> str:
        """Display name for a raw speaker id, tolerant of ids not in the map."""
        s = self.speaker_map().get(speaker_id)
        return s.display if s else speaker_id

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, by_alias=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> Transcript:
        return cls.model_validate_json(raw)
