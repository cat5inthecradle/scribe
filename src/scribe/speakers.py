"""The `speakers.yaml` sidecar.

Names live beside the outputs rather than in the database so an output directory
stays self-describing and portable. Editing the file and re-rendering is the
whole rename workflow — no inference re-runs.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scribe.types import Transcript

FILENAME = "speakers.yaml"

_HEADER = """\
# Map diarized speakers to real names, then re-render:
#
#     scribe rerender {out_dir}
#
# Only the `name:` values are read back; everything else is regenerated.
"""


def write(transcript: Transcript, out_dir: Path) -> Path:
    """Write (or refresh) the sidecar, preserving any names already set."""
    path = out_dir / FILENAME
    existing = read(out_dir)

    payload = {
        "speakers": [
            {
                "id": s.id,
                "label": s.label,
                "name": existing.get(s.id) or s.name,
                "speech_s": s.speech_s,
            }
            for s in transcript.speakers
        ]
    }
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(_HEADER.format(out_dir=out_dir) + body, encoding="utf-8")
    return path


def read(out_dir: Path) -> dict[str, str]:
    """Speaker id -> name, for whichever speakers have a name set."""
    path = out_dir / FILENAME
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}

    names: dict[str, str] = {}
    for entry in data.get("speakers") or []:
        if not isinstance(entry, dict):
            continue
        sid, name = entry.get("id"), entry.get("name")
        if sid and isinstance(name, str) and name.strip():
            names[str(sid)] = name.strip()
    return names


def apply(transcript: Transcript, names: dict[str, str]) -> Transcript:
    """Return a copy of `transcript` with names applied."""
    updated = transcript.model_copy(deep=True)
    for speaker in updated.speakers:
        if name := names.get(speaker.id):
            speaker.name = name
    return updated
