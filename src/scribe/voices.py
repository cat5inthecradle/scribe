"""Speaker enrollment: recognising the same person across recordings.

Diarization can tell voices apart within one recording but numbers them
arbitrarily, so the same colleague is "Speaker 1" in one meeting and
"Speaker 3" in the next. An embedding centroid is a stable voice fingerprint,
so storing one and matching future recordings against it turns anonymous
labels into names automatically.

Measured on synthetic voices: the same speaker across different recordings
scored 0.92-0.96 cosine similarity while different speakers stayed at or below
0.25. Real voices vary more with microphone, room, and health, so the default
threshold is deliberately well below the observed same-speaker floor, and a
near miss is left unnamed rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from scribe.models import Voice


@dataclass(frozen=True, slots=True)
class Match:
    name: str
    similarity: float
    voice_id: object
    sample_count: int

    def __str__(self) -> str:
        return f"{self.name} ({self.similarity:.2f})"


@dataclass(frozen=True, slots=True)
class EnrolledSpeaker:
    """A person on file, summarised for listing."""

    name: str
    samples: int
    total_speech_s: float


def _unit(vector) -> np.ndarray:
    """Normalize to unit length so a dot product is cosine similarity.

    pyannote's centroids are not unit norm, and comparing raw dot products
    would let a loud or long-winded speaker score higher purely from magnitude.
    """
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def similarity(a, b) -> float:
    """Cosine similarity, clamped to [-1, 1] against float drift."""
    return float(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))


def enroll(
    session: Session,
    name: str,
    embedding: list[float],
    *,
    source_name: str | None = None,
    speech_s: float | None = None,
) -> Voice:
    """Add a voice sample for `name`."""
    name = name.strip()
    if not name:
        raise ValueError("speaker name cannot be blank")
    if not embedding:
        raise ValueError("embedding cannot be empty")

    voice = Voice(
        name=name,
        embedding=[float(x) for x in embedding],
        dim=len(embedding),
        source_name=source_name,
        speech_s=speech_s,
    )
    session.add(voice)
    session.flush()
    return voice


def rank(session: Session, embedding: list[float], *, limit: int = 3) -> list[Match]:
    """Enrolled speakers most similar to `embedding`, best first.

    Compares against every stored sample and keeps each person's best score.
    The library is small enough (tens of people, a few samples each) that
    loading it is cheaper than any indexing scheme would be.
    """
    if not embedding:
        return []

    probe = _unit(embedding)
    rows = session.scalars(select(Voice).where(Voice.dim == len(embedding))).all()

    best: dict[str, tuple[float, object]] = {}
    counts: dict[str, int] = {}
    for voice in rows:
        counts[voice.name] = counts.get(voice.name, 0) + 1
        score = float(np.clip(probe @ _unit(voice.embedding), -1.0, 1.0))
        if voice.name not in best or score > best[voice.name][0]:
            best[voice.name] = (score, voice.id)

    matches = [
        Match(
            name=name,
            similarity=score,
            voice_id=voice_id,
            sample_count=counts[name],
        )
        for name, (score, voice_id) in best.items()
    ]
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:limit]


def best_match(
    session: Session, embedding: list[float], threshold: float
) -> Match | None:
    """The single confident match, or None.

    Returning None rather than the closest candidate is deliberate: an unnamed
    "Speaker 2" is honest, whereas a wrong name is silently misleading in a
    transcript someone will trust months later.
    """
    ranked = rank(session, embedding, limit=1)
    if not ranked or ranked[0].similarity < threshold:
        return None
    return ranked[0]


def identify_all(
    session: Session, embeddings: dict[str, list[float]], threshold: float
) -> dict[str, Match]:
    """Match every diarized speaker in one recording against the library.

    A person is assigned at most once per recording: if two diarized speakers
    both match the same enrolled voice, only the stronger one gets the name.
    The same person genuinely appearing as two speakers means diarization split
    them, and naming both would hide that rather than surface it.
    """
    scored: list[tuple[float, str, Match]] = []
    for label, vector in embeddings.items():
        if match := best_match(session, vector, threshold):
            scored.append((match.similarity, label, match))

    scored.sort(reverse=True, key=lambda item: item[0])
    assigned: dict[str, Match] = {}
    taken: set[str] = set()
    for _, label, match in scored:
        if match.name in taken:
            continue
        assigned[label] = match
        taken.add(match.name)
    return assigned


def enrolled(session: Session) -> list[EnrolledSpeaker]:
    """Everyone on file, with sample counts."""
    rows = session.execute(
        select(
            Voice.name,
            func.count(Voice.id),
            func.coalesce(func.sum(Voice.speech_s), 0.0),
        )
        .group_by(Voice.name)
        .order_by(Voice.name)
    ).all()
    return [
        EnrolledSpeaker(name=name, samples=int(n), total_speech_s=float(total))
        for name, n, total in rows
    ]


def forget(session: Session, name: str) -> int:
    """Remove every sample for a name. Returns how many were deleted."""
    result = session.execute(delete(Voice).where(Voice.name == name.strip()))
    return int(result.rowcount or 0)
