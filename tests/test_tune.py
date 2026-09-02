"""Sweep reporting logic.

Pure data: the metrics that let someone read a sweep at a glance, and the
comparison that says whether two settings actually disagreed.
"""

from __future__ import annotations

from conftest import sequence

from scribe.tune import FRAGMENT_WORDS, SweepRow, disagreements
from scribe.types import Turn, Word

A, B = "SPEAKER_00", "SPEAKER_01"


def turn(speaker: str, n_words: int, start: float = 0.0) -> Turn:
    words = [
        Word(t=f"w{i}", start=start + i * 0.3, end=start + i * 0.3 + 0.25)
        for i in range(n_words)
    ]
    return Turn(
        speaker=speaker,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        words=words,
    )


def row(threshold: float, spec: list[tuple[str, int]]) -> SweepRow:
    turns, t = [], 0.0
    for speaker, n in spec:
        turns.append(turn(speaker, n, start=t))
        t += n * 0.3 + 0.5
    speakers = len({s for s, _ in spec})
    return SweepRow(threshold=threshold, speakers=speakers, turns=turns, seconds=1.0)


class TestMetrics:
    def test_fragments_count_turns_too_short_to_be_speech(self):
        r = row(0.6, [(A, 20), (B, 1), (A, 15), (B, 2), (A, 10)])
        assert r.fragments == 2

    def test_a_clean_conversation_has_no_fragments(self):
        assert row(0.6, [(A, 20), (B, 18), (A, 15)]).fragments == 0

    def test_fragment_boundary_is_exclusive(self):
        # Exactly FRAGMENT_WORDS is not a fragment; one fewer is.
        assert row(0.6, [(A, FRAGMENT_WORDS)]).fragments == 0
        assert row(0.6, [(A, FRAGMENT_WORDS - 1)]).fragments == 1

    def test_median_turn_length(self):
        assert row(0.6, [(A, 5), (B, 10), (A, 30)]).median_turn_words == 10

    def test_empty_sweep_metrics_do_not_raise(self):
        empty = SweepRow(threshold=0.6, speakers=0, turns=[], seconds=0.0)
        assert empty.fragments == 0
        assert empty.median_turn_words == 0
        assert empty.shape() == ""

    def test_shape_numbers_speakers_by_first_appearance(self):
        # Diarizers label arbitrarily; the shape must read consistently.
        r = row(0.6, [(B, 5), (A, 3), (B, 4)])
        assert r.shape() == "S1·5 S2·3 S1·4"

    def test_shape_truncates_long_conversations(self):
        r = row(0.6, [(A, 3)] * 40)
        assert r.shape(limit=5).endswith("…")
        assert r.shape(limit=5).count("·") == 5


class TestDisagreements:
    def test_identical_settings_disagree_nowhere(self):
        spec = [(A, 5), (B, 5)]
        rows = [row(0.4, spec), row(0.6, spec), row(0.8, spec)]
        assert disagreements(rows, sequence(*[f"w{i}" for i in range(10)])) == []

    def test_a_single_setting_has_nothing_to_compare(self):
        assert disagreements([row(0.6, [(A, 5)])], sequence("a", "b")) == []

    def test_different_boundaries_are_reported(self):
        words = sequence(*[f"w{i}" for i in range(10)])
        merged = row(0.8, [(A, 10)])          # one speaker throughout
        split = row(0.4, [(A, 5), (B, 5)])    # boundary at word 5
        assert disagreements([merged, split], words) == [5]

    def test_speaker_relabelling_alone_is_not_a_disagreement(self):
        """Labels are arbitrary per run; only boundary placement is comparable.

        A run that calls the same split SPEAKER_01/00 instead of 00/01 has
        found the identical structure and must not be reported as differing.
        """
        words = sequence(*[f"w{i}" for i in range(10)])
        one = row(0.6, [(A, 5), (B, 5)])
        flipped = row(0.5, [(B, 5), (A, 5)])
        assert disagreements([one, flipped], words) == []
