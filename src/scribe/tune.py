"""Hyperparameter sweeps for diarization.

Tuning `clustering.threshold` by hand means running the same recording several
times and comparing, which is slow and easy to get wrong. This does it in one
pass and reports the differences.

The key saving: ASR output does not depend on any diarization setting, so the
audio is transcribed exactly once and only diarization is repeated. A five-value
sweep therefore costs five diarizations rather than five full pipelines — and
since diarization is already ~75% of runtime, that is most of the work anyway.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scribe import media
from scribe.config import Settings
from scribe.merge import merge
from scribe.types import Turn, Word

#: Bracketing the 0.6 default in both directions. Lower splits more eagerly.
DEFAULT_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)

#: A turn this short is usually an attribution artifact rather than speech.
FRAGMENT_WORDS = 4


@dataclass(frozen=True, slots=True)
class SweepRow:
    threshold: float
    speakers: int
    turns: list[Turn]
    seconds: float

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def fragments(self) -> int:
        """Turns too short to be real speech.

        The most useful single number in the sweep: speaker flapping shows up
        as a burst of one- and two-word turns, so this counts the artifact
        directly rather than requiring someone to spot it by reading.
        """
        return sum(1 for t in self.turns if len(t.words) < FRAGMENT_WORDS)

    @property
    def median_turn_words(self) -> int:
        if not self.turns:
            return 0
        counts = sorted(len(t.words) for t in self.turns)
        return counts[len(counts) // 2]

    def shape(self, limit: int = 28) -> str:
        """Compact turn structure, e.g. ``S1·12 S2·3 S1·45``.

        Reads at a glance: long alternating runs are healthy, a string of
        single-digit turns is flapping.
        """
        order: dict[str, int] = {}
        parts = []
        for turn in self.turns[:limit]:
            n = order.setdefault(turn.speaker, len(order) + 1)
            parts.append(f"S{n}·{len(turn.words)}")
        if len(self.turns) > limit:
            parts.append("…")
        return " ".join(parts)


def sweep(
    source: Path,
    settings: Settings,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    min_duration_off: float | None = None,
    on_step: object | None = None,
) -> tuple[list[Word], list[SweepRow]]:
    """Transcribe once, then diarize at each threshold.

    Returns the shared word list and one row per setting.
    """
    from scribe.asr.base import load_backend as load_asr
    from scribe.diarize.base import load_backend as load_diar

    settings.ensure_dirs()
    info = media.inspect(source)
    wav = settings.work / f"tune-{info.sha256[:12]}.wav"
    media.normalize(source, wav)

    try:
        asr = load_asr(
            settings.asr.backend,
            model=(
                settings.asr.mlx_model
                if settings.asr.backend == "mlx"
                else settings.asr.onnx_model_dir
            ),
        )
        words = asr.transcribe(wav)

        rows: list[SweepRow] = []
        for threshold in thresholds:
            if callable(on_step):
                on_step(threshold)
            started = time.perf_counter()
            diarizer = load_diar(
                "pyannote",
                model=settings.diarize.model,
                token=settings.resolved_hf_token(),
                device=settings.diarize.device,
                num_speakers=settings.diarize.num_speakers,
                min_speakers=settings.diarize.min_speakers,
                max_speakers=settings.diarize.max_speakers,
                clustering_threshold=threshold,
                min_duration_off=(
                    min_duration_off
                    if min_duration_off is not None
                    else settings.diarize.min_duration_off
                ),
            )
            diarization = diarizer.diarize(wav)
            turns, speakers = merge(words, diarization, settings.merge)
            rows.append(
                SweepRow(
                    threshold=threshold,
                    speakers=len(speakers),
                    turns=turns,
                    seconds=time.perf_counter() - started,
                )
            )
        return words, rows
    finally:
        wav.unlink(missing_ok=True)


def disagreements(rows: list[SweepRow], words: list[Word]) -> list[int]:
    """Indices of words the settings do not agree about.

    Speaker labels are arbitrary per run, so raw labels cannot be compared.
    What *is* comparable is the boundary structure: for each word, whether the
    speaker changed at that point. Settings agree where they place changes in
    the same places, regardless of what they call the speakers.
    """
    if len(rows) < 2:
        return []

    def change_mask(row: SweepRow) -> list[bool]:
        mask: list[bool] = []
        previous: str | None = None
        for turn in row.turns:
            for i, _ in enumerate(turn.words):
                mask.append(i == 0 and previous is not None)
            previous = turn.speaker
        # Pad or trim: a run may drop words that fell outside every segment.
        mask += [False] * (len(words) - len(mask))
        return mask[: len(words)]

    masks = [change_mask(r) for r in rows]
    return [i for i in range(len(words)) if len({m[i] for m in masks}) > 1]
