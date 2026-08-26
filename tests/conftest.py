from __future__ import annotations

import pytest

from scribe.types import BackendInfo, PipelineInfo, SourceInfo, Transcript, Word


def w(text: str, start: float, end: float, conf: float | None = None) -> Word:
    return Word(t=text, start=start, end=end, conf=conf)


def sequence(*texts: str, start: float = 0.0, dur: float = 0.30, gap: float = 0.05):
    """Evenly spaced words, so tests can talk about attribution not arithmetic."""
    out, t = [], start
    for text in texts:
        out.append(w(text, round(t, 3), round(t + dur, 3)))
        t += dur + gap
    return out


@pytest.fixture
def make_transcript():
    def _make(turns, speakers, *, filename="meeting.m4a", duration_s=60.0):
        return Transcript(
            source=SourceInfo(filename=filename, sha256="deadbeef", duration_s=duration_s),
            pipeline=PipelineInfo(
                asr=BackendInfo(backend="fake", model="fake-asr"),
                diarization=BackendInfo(backend="fake", model="fake-diar"),
                created_at="2026-08-26T12:00:00Z",
            ),
            speakers=speakers,
            turns=turns,
        )

    return _make
