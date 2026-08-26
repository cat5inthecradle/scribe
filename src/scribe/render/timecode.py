"""Timestamp formatting shared by the renderers."""

from __future__ import annotations


def hhmmss(seconds: float) -> str:
    """``H:MM:SS``, dropping the hour when zero. For human-readable output."""
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def subtitle_time(seconds: float, *, sep: str = ",") -> str:
    """``HH:MM:SS,mmm`` for SRT, or with ``sep="."`` for WebVTT."""
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding can carry
        ms, s = 0, s + 1
        if s == 60:
            s, m = 0, m + 1
            if m == 60:
                m, h = 0, h + 1
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
