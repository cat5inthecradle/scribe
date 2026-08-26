"""Tests for the overlap-collapsing fallback.

community-1 supplies an exclusive timeline directly, so `derive_exclusive` only
runs for backends that don't. It still has to be correct, because everything
downstream assumes one speaker per instant.
"""

from __future__ import annotations

from scribe.diarize.base import derive_exclusive
from scribe.types import Segment

A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


def spans(segs: list[Segment]) -> list[tuple[str, float, float]]:
    return [(s.speaker, round(s.start, 3), round(s.end, 3)) for s in segs]


def test_empty_input():
    assert derive_exclusive([]) == []


def test_non_overlapping_input_is_preserved():
    segs = [Segment(A, 0.0, 1.0), Segment(B, 2.0, 3.0)]
    assert spans(derive_exclusive(segs)) == [(A, 0.0, 1.0), (B, 2.0, 3.0)]


def test_adjacent_same_speaker_segments_are_coalesced():
    segs = [Segment(A, 0.0, 1.0), Segment(A, 1.0, 2.0)]
    assert spans(derive_exclusive(segs)) == [(A, 0.0, 2.0)]


def test_overlap_resolves_to_the_shorter_claim():
    # A brief interjection inside a long turn wins its own slice, rather than
    # being swallowed by the segment that merely spans it.
    segs = [Segment(A, 0.0, 10.0), Segment(B, 4.0, 5.0)]
    assert spans(derive_exclusive(segs)) == [
        (A, 0.0, 4.0),
        (B, 4.0, 5.0),
        (A, 5.0, 10.0),
    ]


def test_result_is_strictly_non_overlapping_and_ordered():
    segs = [
        Segment(A, 0.0, 5.0),
        Segment(B, 2.0, 7.0),
        Segment(C, 3.0, 4.0),
    ]
    out = derive_exclusive(segs)
    assert out == sorted(out, key=lambda s: s.start)
    for prev, nxt in zip(out, out[1:], strict=False):
        assert prev.end <= nxt.start, f"{prev} overlaps {nxt}"


def test_gaps_between_segments_stay_empty():
    segs = [Segment(A, 0.0, 1.0), Segment(B, 5.0, 6.0)]
    out = derive_exclusive(segs)
    assert not any(s.start >= 1.0 and s.end <= 5.0 for s in out)
