"""Media inspection and normalization.

Both Parakeet and pyannote want 16 kHz mono PCM. Normalizing once up front means
neither backend re-decodes the source, and video containers (Zoom/Meet exports)
work with no special handling — ffmpeg just drops the video stream.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_RATE = 16_000
TARGET_CHANNELS = 1

# Containers ffmpeg handles that we accept from the intake folder. Deliberately
# a allowlist: the intake dir will accumulate .DS_Store, partial downloads, etc.
MEDIA_SUFFIXES = frozenset(
    {
        ".mp3", ".m4a", ".mp4", ".wav", ".flac", ".ogg", ".opus", ".aac",
        ".webm", ".mkv", ".mov", ".avi", ".wma", ".aiff", ".aif", ".m4v", ".3gp",
    }
)


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    duration_s: float
    bytes: int
    sha256: str


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(
                f"{tool} not found on PATH. Install it with: brew install ffmpeg"
            )


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_SUFFIXES


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Content hash, used as the dedupe key so re-dropping a file is a no-op."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def probe_duration(path: Path) -> float:
    """Duration in seconds via ffprobe.

    Falls back to the first audio stream's duration when the container has no
    format-level duration (common for streamed .webm captures).
    """
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=duration,codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed on {path.name}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    if raw := data.get("format", {}).get("duration"):
        try:
            return float(raw)
        except ValueError:
            pass
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio" and stream.get("duration"):
            try:
                return float(stream["duration"])
            except ValueError:
                continue
    raise MediaError(f"could not determine duration of {path.name}")


def inspect(path: Path) -> MediaInfo:
    require_ffmpeg()
    if not path.is_file():
        raise MediaError(f"not a file: {path}")
    return MediaInfo(
        path=path,
        duration_s=probe_duration(path),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def normalize(src: Path, dest: Path) -> Path:
    """Decode `src` to 16 kHz mono 16-bit PCM WAV at `dest`.

    No loudness normalization or denoising: both models were trained on ordinary
    audio, and filtering ahead of them tends to cost accuracy rather than buy it.
    """
    require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first so a crash can't leave a truncated WAV that
    # looks complete to the next stage.
    tmp = dest.with_suffix(dest.suffix + ".partial")
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(src),
            "-vn",
            "-map", "0:a:0",
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_RATE),
            "-c:a", "pcm_s16le",
            # State the container explicitly: the temp file ends in .partial, so
            # ffmpeg cannot infer the format from the extension.
            "-f", "wav",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise MediaError(f"ffmpeg failed to normalize {src.name}:\n{tail}")
    tmp.replace(dest)
    return dest
