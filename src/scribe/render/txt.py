"""Flat timestamped text — the grep-friendly format."""

from __future__ import annotations

from scribe.render.timecode import hhmmss
from scribe.types import Transcript


def render_txt(t: Transcript) -> str:
    lines = [
        f"[{hhmmss(turn.start)}] {t.display_for(turn.speaker)}: {turn.text}"
        for turn in t.turns
    ]
    return "\n".join(lines) + ("\n" if lines else "")
