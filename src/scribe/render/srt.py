"""Subtitle output (SRT and WebVTT).

A turn can run a minute or more, which is unusable as a subtitle cue, so turns
are re-chunked into short cues. Word-level timing makes this exact rather than
interpolated — each cue's in/out points are real word boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from scribe.merge import join_words
from scribe.render.timecode import subtitle_time
from scribe.types import Transcript, Word

MAX_CUE_CHARS = 84  # two 42-char lines, the usual readability ceiling
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.7


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    speaker: str
    text: str


def _chunk(words: list[Word]) -> list[list[Word]]:
    """Split a turn's words into cue-sized groups."""
    chunks: list[list[Word]] = []
    buf: list[Word] = []
    chars = 0

    for word in words:
        token = word.text.strip()
        if not token:
            continue
        would_be = chars + len(token) + (1 if buf else 0)
        too_wide = buf and would_be > MAX_CUE_CHARS
        too_long = buf and (word.end - buf[0].start) > MAX_CUE_SECONDS
        if too_wide or too_long:
            chunks.append(buf)
            buf, chars = [], 0
            would_be = len(token)
        buf.append(word)
        chars = would_be

    if buf:
        chunks.append(buf)
    return chunks


def build_cues(t: Transcript) -> list[Cue]:
    cues: list[Cue] = []
    for turn in t.turns:
        speaker = t.display_for(turn.speaker)
        # A turn with no word detail (e.g. a hand-edited transcript) still yields
        # one usable cue rather than disappearing.
        groups = _chunk(turn.words) if turn.words else []
        if not groups:
            cues.append(Cue(turn.start, turn.end, speaker, turn.text))
            continue
        for group in groups:
            start, end = group[0].start, group[-1].end
            cues.append(
                Cue(start, max(end, start + MIN_CUE_SECONDS), speaker, join_words(group))
            )
    return cues


def render_srt(t: Transcript) -> str:
    blocks = []
    for i, cue in enumerate(build_cues(t), start=1):
        blocks.append(
            f"{i}\n"
            f"{subtitle_time(cue.start)} --> {subtitle_time(cue.end)}\n"
            f"{cue.speaker}: {cue.text}\n"
        )
    return "\n".join(blocks)


def render_vtt(t: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for cue in build_cues(t):
        start = subtitle_time(cue.start, sep=".")
        end = subtitle_time(cue.end, sep=".")
        lines += [f"{start} --> {end}", f"<v {cue.speaker}>{cue.text}", ""]
    return "\n".join(lines)
