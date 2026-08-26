"""Output renderers.

`transcript.json` is canonical and retains word-level timing; every other format
is derived from it. That is what makes speaker renaming a pure re-render with no
inference.
"""

from __future__ import annotations

from pathlib import Path

from scribe.render.md import render_md
from scribe.render.srt import render_srt, render_vtt
from scribe.render.txt import render_txt
from scribe.types import Transcript

__all__ = ["render_all", "render_md", "render_srt", "render_txt", "render_vtt"]

RENDERERS = {
    "transcript.md": render_md,
    "transcript.txt": render_txt,
    "transcript.srt": render_srt,
    "transcript.vtt": render_vtt,
}


def render_all(transcript: Transcript, out_dir: Path) -> list[Path]:
    """Write every format into `out_dir`. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "transcript.json"]
    written[0].write_text(transcript.to_json(), encoding="utf-8")

    for filename, fn in RENDERERS.items():
        path = out_dir / filename
        path.write_text(fn(transcript), encoding="utf-8")
        written.append(path)
    return written
