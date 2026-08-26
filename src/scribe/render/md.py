"""Markdown transcript — the format meant to actually be read."""

from __future__ import annotations

from scribe.render.timecode import hhmmss
from scribe.types import Transcript


def render_md(t: Transcript) -> str:
    lines = [f"# {t.source.filename}", ""]

    speaker_list = ", ".join(s.display for s in t.speakers) or "none detected"
    lines += [
        f"- **Duration:** {hhmmss(t.source.duration_s)}",
        f"- **Speakers:** {len(t.speakers)} — {speaker_list}",
        f"- **Transcribed:** {t.pipeline.created_at}",
        f"- **Models:** {t.pipeline.asr.model} / {t.pipeline.diarization.model}",
        "",
        "---",
        "",
    ]

    # Only repeat the speaker heading when the speaker actually changes, so a
    # back-and-forth exchange doesn't drown the text in headings.
    previous: str | None = None
    for turn in t.turns:
        display = t.display_for(turn.speaker)
        if display != previous:
            lines += [f"**{display}** · `{hhmmss(turn.start)}`", ""]
            previous = display
        lines += [turn.text, ""]

    return "\n".join(lines).rstrip() + "\n"
