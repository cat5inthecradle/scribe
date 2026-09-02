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
# `matched` and `candidates` are hints from the enrolled-voice library --
# confirm one interactively with `scribe identify {out_dir}`, which also
# enrolls the confirmation as a new sample.
"""


def write(
    transcript: Transcript,
    out_dir: Path,
    candidates: dict[str, list[dict[str, object]]] | None = None,
) -> Path:
    """Write (or refresh) the sidecar, preserving any names already set.

    `candidates` carries the closest enrolled voices per speaker. They are
    informational only — written so someone editing this file by hand can see
    who the library thought it might be, and regenerated on every write.
    """
    path = out_dir / FILENAME
    existing = read(out_dir)
    candidates = candidates or {}

    entries: list[dict[str, object]] = []
    for s in transcript.speakers:
        entry: dict[str, object] = {
            "id": s.id,
            "label": s.label,
            "name": existing.get(s.id) or s.name,
            "speech_s": s.speech_s,
        }
        if hints := candidates.get(s.id):
            entry["candidates"] = hints
        entries.append(entry)

    payload = {"speakers": entries}
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
