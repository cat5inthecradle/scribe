"""ASR backend interface.

A backend turns a normalized 16 kHz mono WAV into words with timestamps. That is
the whole contract — no speaker awareness, no segmentation. Keeping it this
narrow is what lets the Metal and CPU backends be swapped freely, and what would
let a Whisper backend drop in later for non-English audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from scribe.types import Word

ProgressFn = "None | object"


@runtime_checkable
class AsrBackend(Protocol):
    name: str
    model_id: str

    def transcribe(self, wav: Path) -> list[Word]:
        """Words in chronological order, with absolute timestamps in seconds.

        `text` on each word must be display-ready and stripped of surrounding
        whitespace; `merge` handles the joining.
        """
        ...


def load_backend(name: str, **kwargs: object) -> AsrBackend:
    """Import a backend lazily.

    The imports are deliberately inside the branches: `mlx` cannot even be
    imported on Linux, and `onnx` pulls a large runtime we don't want loaded on
    the host fast path.
    """
    if name == "mlx":
        from scribe.asr.mlx_parakeet import MlxParakeet

        return MlxParakeet(**kwargs)  # type: ignore[arg-type]
    if name == "onnx":
        from scribe.asr.onnx_parakeet import OnnxParakeet

        return OnnxParakeet(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown ASR backend: {name!r} (expected 'mlx' or 'onnx')")
