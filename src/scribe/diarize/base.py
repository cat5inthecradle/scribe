"""Diarization backend interface.

A backend answers "who spoke when" and must supply an *exclusive* timeline — one
speaker per instant. `merge` relies on that: attributing a word to a speaker is
ambiguous if two speakers hold the same moment, and resolving overlap is the
diarizer's job, not the merger's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from scribe.types import Diarization, Segment


@runtime_checkable
class DiarizeBackend(Protocol):
    name: str
    model_id: str

    def diarize(self, wav: Path) -> Diarization:
        ...


def derive_exclusive(overlapped: list[Segment]) -> list[Segment]:
    """Collapse an overlapping timeline to one speaker per instant.

    Only needed for backends that don't produce an exclusive timeline natively
    (pyannote community-1 does). Splits at every boundary and keeps, for each
    resulting slice, the speaker whose segment is shortest — the shortest
    segment is the most confident/specific claim on that moment, so this favours
    brief interjections over a long segment that merely spans them.
    """
    if not overlapped:
        return []

    bounds = sorted({b for s in overlapped for b in (s.start, s.end)})
    out: list[Segment] = []
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2
        covering = [s for s in overlapped if s.start <= mid < s.end]
        if not covering:
            continue
        winner = min(covering, key=lambda s: s.duration)
        # Extend the previous slice instead of emitting an adjacent duplicate.
        if out and out[-1].speaker == winner.speaker and abs(out[-1].end - lo) < 1e-6:
            out[-1] = Segment(winner.speaker, out[-1].start, hi)
        else:
            out.append(Segment(winner.speaker, lo, hi))
    return out


def load_backend(name: str = "pyannote", **kwargs: object) -> DiarizeBackend:
    if name == "pyannote":
        from scribe.diarize.pyannote_ import PyannoteDiarizer

        return PyannoteDiarizer(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown diarization backend: {name!r}")
