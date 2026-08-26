"""Parakeet TDT on Metal via MLX. The host fast path.

macOS + Apple Silicon only — MLX needs Metal, which containers on this platform
cannot reach. The portable equivalent is `scribe.asr.onnx_parakeet`, which runs
the same model family so transcripts stay comparable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from scribe.types import Word

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# Long audio is processed in overlapping chunks to bound peak memory. 120s with
# 15s of overlap is parakeet-mlx's own recommendation; the overlap is what stops
# words being clipped at chunk seams.
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 15.0

# Below this length chunking only adds overhead, so transcribe in one pass.
CHUNK_THRESHOLD_SECONDS = 150.0


def _geometric_mean(values: list[float]) -> float:
    """Match parakeet's own confidence aggregation.

    Geometric rather than arithmetic so one very low-confidence subword drags the
    word's score down instead of being averaged away.
    """
    if not values:
        return 1.0
    return math.exp(sum(math.log(max(v, 1e-10)) for v in values) / len(values))


def tokens_to_words(tokens: list) -> list[Word]:
    """Merge Parakeet's subword tokens into whole words.

    The tokenizer is SentencePiece, so a word boundary is signalled by a leading
    space on the token that *starts* the next word. Emitting subwords directly
    would break attribution — merge would assign speakers to fragments like
    "ing" and the transcript text would be unreadable.
    """
    words: list[Word] = []
    text = ""
    start = 0.0
    end = 0.0
    confs: list[float] = []

    def flush() -> None:
        nonlocal text, confs
        if stripped := text.strip():
            words.append(
                Word(
                    t=stripped,
                    start=round(start, 3),
                    end=round(end, 3),
                    conf=round(_geometric_mean(confs), 4),
                )
            )
        text = ""
        confs = []

    for token in tokens:
        raw = token.text
        # A leading space opens a new word; the very first token does too.
        if raw.startswith(" ") and text.strip():
            flush()
        if not text:
            start = token.start
        text += raw
        end = token.end
        confs.append(float(getattr(token, "confidence", 1.0)))

    flush()
    return words


class MlxParakeet:
    name = "mlx"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        self.model_id = model
        self._on_progress = on_progress
        self._model = None  # loaded lazily; ~600MB of weights

    def _load(self):
        if self._model is None:
            from parakeet_mlx import from_pretrained

            self._model = from_pretrained(self.model_id)
        return self._model

    def transcribe(self, wav: Path) -> list[Word]:
        from parakeet_mlx.audio import load_audio

        model = self._load()

        # Decide on chunking from the real sample count rather than trusting the
        # container metadata we probed earlier.
        rate = model.preprocessor_config.sample_rate
        duration = len(load_audio(Path(wav), rate)) / rate

        callback = None
        if self._on_progress is not None:
            report = self._on_progress

            def callback(done: int, total: int) -> None:  # noqa: F811
                report(min(1.0, done / total) if total else 0.0)

        result = model.transcribe(
            Path(wav),
            chunk_duration=(
                CHUNK_SECONDS if duration > CHUNK_THRESHOLD_SECONDS else None
            ),
            overlap_duration=OVERLAP_SECONDS,
            chunk_callback=callback,
        )
        if self._on_progress is not None:
            self._on_progress(1.0)
        return tokens_to_words(result.tokens)
