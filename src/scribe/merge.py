"""Word-to-speaker attribution and turn grouping.

This is the quality core of the whole pipeline: ASR decides *what* was said and
diarization decides *who* was talking, but this module decides who said which
word. Its failure mode is the most visible artifact in a transcript — a single
word flipping to the wrong speaker mid-sentence.

Deliberately free of model dependencies: everything here is pure data
manipulation over `Word` and `Segment`, so it is exhaustively testable with
JSON fixtures and no inference.
"""

from __future__ import annotations

import re
from bisect import bisect_right

from scribe.config import MergeSettings
from scribe.types import UNKNOWN_SPEAKER, Diarization, Segment, Speaker, Turn, Word

_PUNCT_TIGHTEN = (
    (re.compile(r"\s+([,.!?;:])"), r"\1"),          # "word ," -> "word,"
    (re.compile(r"([(\[“])\s+"), r"\1"),        # "( word" -> "(word"
    (re.compile(r"\s+([)\]”])"), r"\1"),        # "word )" -> "word)"
)

_SENTENCE_END = re.compile(r"[.!?][\"'”\)\]]*$")


def join_words(words: list[Word]) -> str:
    """Render words as text.

    Backends emit whole words, so a plain space join is correct; the regexes only
    repair spacing where punctuation arrived as its own token.
    """
    text = " ".join(t for w in words if (t := w.text.strip()))
    for pattern, repl in _PUNCT_TIGHTEN:
        text = pattern.sub(repl, text)
    return text.strip()


class _Timeline:
    """Index over a non-overlapping, sorted segment list.

    Words number in the tens of thousands on a long recording and segments in
    the thousands, so the naive scan is ~10^8 comparisons. Because an exclusive
    timeline has non-overlapping segments, both starts *and* ends are
    monotonically increasing, which makes a bisect plus a short backward walk
    exact rather than approximate.
    """

    def __init__(self, segments: list[Segment]) -> None:
        self.segs = sorted(segments, key=lambda s: (s.start, s.end))
        self.starts = [s.start for s in self.segs]

    def overlapping(self, start: float, end: float) -> list[Segment]:
        """Segments intersecting ``[start, end]``."""
        out: list[Segment] = []
        i = bisect_right(self.starts, end) - 1
        while i >= 0 and self.segs[i].end > start:
            out.append(self.segs[i])
            i -= 1
        return out

    def best_speaker(self, word: Word, window_s: float) -> str | None:
        """Speaker with the most overlap, else the nearest within `window_s`."""
        if not self.segs:
            return None

        candidates = self.overlapping(word.start, word.end)
        if candidates:
            totals: dict[str, float] = {}
            for seg in candidates:
                totals[seg.speaker] = totals.get(seg.speaker, 0.0) + seg.overlap(
                    word.start, word.end
                )
            # Ties broken by earlier onset for determinism, not by dict order.
            best = max(totals.items(), key=lambda kv: (kv[1], -self._onset(kv[0])))
            if best[1] > 0:
                return best[0]

        # Zero-overlap: a word landing in a diarization gap. Reach out a little.
        near = self.overlapping(word.start - window_s, word.end + window_s)
        if not near:
            return None
        return min(near, key=lambda s: self._distance(s, word)).speaker

    def _onset(self, speaker: str) -> float:
        return next((s.start for s in self.segs if s.speaker == speaker), 0.0)

    @staticmethod
    def _distance(seg: Segment, word: Word) -> float:
        if seg.end <= word.start:
            return word.start - seg.end
        if seg.start >= word.end:
            return seg.start - word.end
        return 0.0


def attribute(
    words: list[Word], exclusive: list[Segment], cfg: MergeSettings
) -> list[str]:
    """Assign a speaker to every word. Returns one label per word, positionally."""
    timeline = _Timeline(exclusive)
    raw: list[str | None] = [
        timeline.best_speaker(w, cfg.nearest_window_s) for w in words
    ]

    # Fill gaps by continuity: carry the previous speaker forward, and for a
    # leading gap borrow from the first word we did attribute. Speech doesn't
    # change speaker in silence, so continuity beats guessing.
    filled: list[str] = []
    last: str | None = None
    for spk in raw:
        if spk is not None:
            last = spk
        filled.append(spk or last or "")
    if not any(filled):
        return [UNKNOWN_SPEAKER] * len(words)

    first_known = next(s for s in filled if s)
    return [s or first_known for s in filled]


def _runs(speakers: list[str]) -> list[tuple[int, int, str]]:
    """Consecutive same-speaker spans as ``(start_idx, end_idx_exclusive, speaker)``."""
    if not speakers:
        return []
    out: list[tuple[int, int, str]] = []
    lo = 0
    for i in range(1, len(speakers) + 1):
        if i == len(speakers) or speakers[i] != speakers[lo]:
            out.append((lo, i, speakers[lo]))
            lo = i
    return out


def smooth(words: list[Word], speakers: list[str], cfg: MergeSettings) -> list[str]:
    """Absorb implausibly short speaker runs into their neighbours.

    Diarization boundaries are accurate to a few hundred milliseconds, so a word
    near a real speaker change can be pulled across it. The result reads as one
    speaker's sentence with a stray word attributed to someone else. A run is
    only absorbed when it is *both* short in words and short in time, and when
    doing so is unambiguous: either it sits between two runs of the same speaker,
    or it is at an edge with only one neighbour.

    Runs between two *different* speakers are left alone — that is a genuine
    speaker change with a brief utterance inside it, not an artifact.
    """
    result = list(speakers)
    if not result:
        return result

    # Absorbing merges runs, which can expose a newly-absorbable neighbour, so
    # iterate to a fixed point. Bounded because every pass strictly reduces the
    # run count when it changes anything.
    for _ in range(4):
        runs = _runs(result)
        if len(runs) < 2:
            break

        changed = False
        for idx, (lo, hi, speaker) in enumerate(runs):
            if hi - lo > cfg.smooth_max_words:
                continue
            span = words[hi - 1].end - words[lo].start
            if span >= cfg.smooth_max_duration_s:
                continue

            prev_spk = runs[idx - 1][2] if idx > 0 else None
            next_spk = runs[idx + 1][2] if idx + 1 < len(runs) else None

            if prev_spk is not None and prev_spk == next_spk:
                target = prev_spk
            elif prev_spk is None and next_spk is not None:
                target = next_spk
            elif next_spk is None and prev_spk is not None:
                target = prev_spk
            else:
                continue  # flanked by two different speakers: a real change.

            if target != speaker:
                result[lo:hi] = [target] * (hi - lo)
                changed = True

        if not changed:
            break
    return result


def group(words: list[Word], speakers: list[str], cfg: MergeSettings) -> list[Turn]:
    """Collect attributed words into readable turns."""
    turns: list[Turn] = []
    buf: list[Word] = []

    def flush(speaker: str) -> None:
        if buf:
            turns.append(
                Turn(
                    speaker=speaker,
                    start=buf[0].start,
                    end=buf[-1].end,
                    text=join_words(buf),
                    words=list(buf),
                )
            )
            buf.clear()

    for i, (word, speaker) in enumerate(zip(words, speakers, strict=True)):
        if buf:
            prev_speaker = speakers[i - 1]
            gap = word.start - buf[-1].end
            too_long = (buf[-1].end - buf[0].start) >= cfg.turn_max_duration_s
            # A long monologue is split only at a sentence boundary, so turns
            # never break mid-clause.
            if (
                speaker != prev_speaker
                or gap > cfg.turn_gap_s
                or (too_long and _SENTENCE_END.search(buf[-1].text.strip()))
            ):
                flush(prev_speaker)
        buf.append(word)

    if buf:
        flush(speakers[-1])
    return turns


def build_speakers(turns: list[Turn]) -> list[Speaker]:
    """Assign display labels by order of first appearance.

    Diarization backends number speakers arbitrarily, so ``SPEAKER_02`` may well
    talk first. Relabelling by onset means "Speaker 1" is whoever opened the
    recording, which is what a reader expects.
    """
    order: list[str] = []
    speech: dict[str, float] = {}
    for turn in turns:
        if turn.speaker not in speech:
            order.append(turn.speaker)
            speech[turn.speaker] = 0.0
        speech[turn.speaker] += turn.duration

    return [
        Speaker(id=sid, label=f"Speaker {n}", name=None, speech_s=round(speech[sid], 2))
        for n, sid in enumerate(order, start=1)
    ]


def merge(
    words: list[Word], diarization: Diarization, cfg: MergeSettings | None = None
) -> tuple[list[Turn], list[Speaker]]:
    """Full attribution pipeline: attribute -> smooth -> group -> label."""
    cfg = cfg or MergeSettings()
    if not words:
        return [], []

    words = sorted(words, key=lambda w: (w.start, w.end))
    speakers = attribute(words, diarization.exclusive, cfg)
    speakers = smooth(words, speakers, cfg)
    turns = group(words, speakers, cfg)
    return turns, build_speakers(turns)
