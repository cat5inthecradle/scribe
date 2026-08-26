"""Attribution tests.

Everything here is synthetic: the point is to pin down the decisions `merge`
makes at boundaries, which is exactly where transcripts visibly go wrong.
"""

from __future__ import annotations

from conftest import sequence, w

from scribe.config import MergeSettings
from scribe.merge import (
    attribute,
    build_speakers,
    group,
    join_words,
    merge,
    smooth,
    usable_segments,
)
from scribe.types import UNKNOWN_SPEAKER, Diarization, Segment

CFG = MergeSettings()

A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


def diar(*segments: Segment) -> Diarization:
    return Diarization(exclusive=list(segments), overlapped=list(segments))


def spans_of(segs: list[Segment]) -> list[tuple[str, float, float]]:
    return [(s.speaker, round(s.start, 3), round(s.end, 3)) for s in segs]


class TestAttribute:
    def test_back_to_back_turns_split_at_the_boundary(self):
        words = sequence("one", "two", "three", "four")  # ends 0.30/0.65/1.00/1.35
        segs = [Segment(A, 0.0, 0.70), Segment(B, 0.70, 2.0)]
        assert attribute(words, segs, CFG) == [A, A, B, B]

    def test_word_straddling_a_boundary_goes_to_the_larger_overlap(self):
        # 0.9-1.2 overlaps A by 0.1s and B by 0.2s.
        words = [w("borderline", 0.9, 1.2)]
        segs = [Segment(A, 0.0, 1.0), Segment(B, 1.0, 2.0)]
        assert attribute(words, segs, CFG) == [B]

    def test_word_in_a_diarization_gap_inherits_the_previous_speaker(self):
        words = [w("early", 0.1, 0.4), w("orphan", 1.8, 2.0)]
        segs = [Segment(A, 0.0, 1.0), Segment(B, 3.0, 4.0)]
        # Nearest segment is 0.8s away, well past nearest_window_s.
        assert attribute(words, segs, CFG) == [A, A]

    def test_word_just_outside_a_segment_snaps_to_it(self):
        words = [w("close", 1.05, 1.30)]
        segs = [Segment(A, 0.0, 1.0)]
        assert attribute(words, segs, CFG) == [A]

    def test_leading_orphan_borrows_from_the_first_attributed_word(self):
        words = [w("stray", 0.0, 0.2), w("real", 5.1, 5.4)]
        segs = [Segment(B, 5.0, 6.0)]
        assert attribute(words, segs, CFG) == [B, B]

    def test_no_diarization_at_all_yields_unknown(self):
        words = sequence("a", "b")
        assert attribute(words, [], CFG) == [UNKNOWN_SPEAKER] * 2

    def test_attribution_is_positional_and_total(self):
        words = sequence(*[f"w{i}" for i in range(25)])
        segs = [Segment(A, 0.0, 4.0), Segment(B, 4.0, 20.0)]
        assert len(attribute(words, segs, CFG)) == len(words)


class TestUsableSegments:
    """Micro-segment filtering. Regression: real crosstalk audio."""

    def test_sub_word_segments_are_discarded(self):
        segs = [
            Segment(A, 0.0, 5.0),
            Segment(B, 5.00, 5.02),   # 20ms — pyannote flicker, not speech
            Segment(A, 5.02, 5.04),
            Segment(B, 6.0, 7.5),
        ]
        kept = usable_segments(segs, CFG)
        assert spans_of(kept) == [(A, 0.0, 5.0), (B, 6.0, 7.5)]

    def test_filtering_everything_falls_back_to_the_original(self):
        # A noisy timeline still beats no timeline: without this, every word
        # would go UNKNOWN.
        segs = [Segment(A, 0.0, 0.02), Segment(B, 0.02, 0.04)]
        assert usable_segments(segs, CFG) == segs

    def test_flicker_no_longer_steals_a_word(self):
        # Observed on real audio: "But wait a minute, do you do your own
        # stunts?" was split across three turns because a 0.02s segment won
        # the overlap vote for a word in the middle of it.
        words = sequence("But", "wait", "a", "minute", "do", "you", "do", "your", "own")
        segs = [
            Segment(A, 0.0, 1.5),
            Segment(B, 1.50, 1.52),   # flicker inside one continuous question
            Segment(A, 1.52, 5.0),
        ]
        turns, _ = merge(words, diar(*segs), CFG)
        assert len(turns) == 1, f"question split into {len(turns)} turns"
        assert turns[0].speaker == A


class TestSmooth:
    def test_single_word_flap_between_one_speaker_is_absorbed(self):
        words = sequence("I", "think", "yes", "we", "should")
        assert smooth(words, [A, A, B, A, A], CFG) == [A] * 5

    def test_flap_flanked_by_two_different_speakers_is_kept(self):
        # A genuine three-way exchange, not an artifact — nothing to absorb into.
        words = sequence("I", "think", "yes", "we", "should")
        assert smooth(words, [A, A, B, C, C], CFG) == [A, A, B, C, C]

    def test_rapid_multiword_fragment_is_absorbed(self):
        # Regression: 3 words in 0.32s escaped smoothing under the old
        # 2-word cap, splitting one spoken question across three turns.
        # Duration is the meaningful test; the word cap is only a guard.
        words = [
            w("But", 0.00, 0.20), w("wait", 0.22, 0.45),
            w("do", 0.50, 0.60), w("you", 0.62, 0.72), w("do", 0.74, 0.82),
            w("your", 0.90, 1.20), w("own", 1.22, 1.50),
        ]
        speakers = [A, A, B, B, B, A, A]
        assert smooth(words, speakers, CFG) == [A] * 7

    def test_long_run_is_never_absorbed(self):
        words = sequence(*[f"w{i}" for i in range(8)])
        speakers = [A, A, B, B, B, B, A, A]
        assert smooth(words, speakers, CFG) == speakers

    def test_short_word_count_but_long_duration_is_kept(self):
        # Two words spanning 2s is real speech, not a boundary artifact.
        words = [w("well", 0.0, 0.9), w("actually", 1.0, 2.0), w("no", 2.1, 2.4)]
        assert smooth(words, [B, B, A], CFG)[:2] == [B, B]

    def test_leading_flap_absorbs_into_its_only_neighbour(self):
        words = sequence("uh", "so", "anyway", "right")
        assert smooth(words, [B, A, A, A], CFG) == [A] * 4

    def test_trailing_flap_absorbs_into_its_only_neighbour(self):
        words = sequence("so", "anyway", "right", "uh")
        assert smooth(words, [A, A, A, B], CFG) == [A] * 4

    def test_rapid_two_speaker_exchange_survives(self):
        # Real back-and-forth: each run is short, but they alternate between two
        # speakers, so no run is flanked by a matching pair.
        words = sequence(*[f"w{i}" for i in range(6)])
        speakers = [A, A, B, B, A, A]
        assert smooth(words, speakers, CFG) == speakers


class TestGroup:
    def test_speaker_change_starts_a_new_turn(self):
        words = sequence("hello", "there", "hi", "back")
        turns = group(words, [A, A, B, B], CFG)
        assert [t.speaker for t in turns] == [A, B]
        assert turns[0].text == "hello there"
        assert turns[1].text == "hi back"

    def test_long_silence_splits_one_speaker(self):
        words = [w("before", 0.0, 0.5), w("after", 9.0, 9.5)]
        turns = group(words, [A, A], CFG)
        assert len(turns) == 2
        assert turns[0].end == 0.5 and turns[1].start == 9.0

    def test_turn_bounds_and_words_are_preserved(self):
        words = sequence("a", "b", "c")
        turn = group(words, [A, A, A], CFG)[0]
        assert turn.start == words[0].start
        assert turn.end == words[-1].end
        assert len(turn.words) == 3

    def test_long_turn_splits_at_the_first_sentence_end_past_the_limit(self):
        cfg = MergeSettings(turn_max_duration_s=1.0, turn_gap_s=99.0)
        words = [
            w("one", 0.0, 0.4), w("two", 0.5, 0.9),
            w("three.", 1.0, 1.4), w("four", 1.5, 1.9),
        ]
        turns = group(words, [A] * 4, cfg)
        assert [t.text for t in turns] == ["one two three.", "four"]

    def test_sentence_end_before_the_limit_does_not_split(self):
        # The limit is a floor, not a target: short sentences stay in one turn.
        cfg = MergeSettings(turn_max_duration_s=5.0, turn_gap_s=99.0)
        words = [
            w("one.", 0.0, 0.4), w("two.", 0.5, 0.9), w("three.", 1.0, 1.4),
        ]
        assert len(group(words, [A] * 3, cfg)) == 1

    def test_long_turn_with_no_sentence_end_is_never_split_mid_clause(self):
        # Rambling with no punctuation stays whole rather than breaking randomly.
        cfg = MergeSettings(turn_max_duration_s=0.5, turn_gap_s=99.0)
        words = sequence(*[f"w{i}" for i in range(12)])
        assert len(group(words, [A] * 12, cfg)) == 1


class TestSpeakerLabels:
    def test_labels_follow_first_appearance_not_backend_numbering(self):
        words = sequence("second", "speaker", "first", "one")
        turns = group(words, [C, C, A, A], CFG)
        speakers = build_speakers(turns)
        assert [(s.id, s.label) for s in speakers] == [
            (C, "Speaker 1"),
            (A, "Speaker 2"),
        ]

    def test_speech_time_accumulates_across_turns(self):
        words = sequence(*[f"w{i}" for i in range(4)])
        speakers = build_speakers(group(words, [A, B, A, B], CFG))
        by_id = {s.id: s for s in speakers}
        assert by_id[A].speech_s > 0
        assert by_id[B].speech_s > 0


class TestJoinWords:
    def test_punctuation_tokens_are_tightened(self):
        assert join_words(sequence("Hello", ",", "world", ".")) == "Hello, world."

    def test_blank_tokens_are_dropped(self):
        assert join_words(sequence("a", " ", "b")) == "a b"

    def test_empty_input(self):
        assert join_words([]) == ""


class TestMergeEndToEnd:
    def test_full_pipeline_produces_clean_two_speaker_dialogue(self):
        words = sequence(*"q1 q2 q3 a1 a2 a3".split())
        # Boundary intentionally set mid-word to exercise smoothing + grouping.
        d = diar(Segment(A, 0.0, 1.10), Segment(B, 1.10, 3.0))
        turns, speakers = merge(words, d, CFG)
        assert [t.speaker for t in turns] == [A, B]
        assert [s.label for s in speakers] == ["Speaker 1", "Speaker 2"]

    def test_empty_words_is_not_an_error(self):
        assert merge([], diar(Segment(A, 0.0, 1.0)), CFG) == ([], [])

    def test_unsorted_input_is_ordered_first(self):
        words = [w("second", 1.0, 1.4), w("first", 0.0, 0.4)]
        turns, _ = merge(words, diar(Segment(A, 0.0, 2.0)), CFG)
        assert turns[0].text == "first second"
