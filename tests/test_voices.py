"""Speaker enrollment.

Measured behaviour on real embeddings: the same voice across different
recordings scores 0.92-0.96 cosine similarity while different voices stay at or
below 0.34. These tests pin the logic that turns those numbers into names — and
particularly the refusals, since a wrong name is worse than no name.
"""

from __future__ import annotations

import math

import pytest

from scribe import voices
from scribe.models import Voice


def vec(*values: float, dim: int = 8) -> list[float]:
    """A padded vector, so tests can express direction compactly."""
    out = list(values) + [0.0] * (dim - len(values))
    return out[:dim]


ALICE = vec(1.0, 0.0, 0.0)
ALICE_VARIANT = vec(0.96, 0.28, 0.0)   # same person, different day (~0.96)
BOB = vec(0.0, 1.0, 0.0)
CAROL = vec(0.0, 0.0, 1.0)


class TestSimilarity:
    def test_identical_vectors_score_one(self):
        assert voices.similarity(ALICE, ALICE) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert voices.similarity(ALICE, BOB) == pytest.approx(0.0)

    def test_magnitude_is_ignored(self):
        # pyannote centroids are not unit norm; a loud speaker must not win on
        # magnitude alone.
        loud = [x * 17.0 for x in ALICE]
        assert voices.similarity(ALICE, loud) == pytest.approx(1.0)

    def test_symmetric(self):
        assert voices.similarity(ALICE, BOB) == pytest.approx(
            voices.similarity(BOB, ALICE)
        )

    def test_zero_vector_does_not_divide_by_zero(self):
        assert math.isfinite(voices.similarity(ALICE, vec(0.0)))


class TestEnroll:
    def test_creates_a_sample(self, session):
        voice = voices.enroll(session, "Alice", ALICE, source_name="a.wav")
        session.commit()
        assert voice.name == "Alice"
        assert voice.dim == len(ALICE)
        assert voice.source_name == "a.wav"

    def test_name_is_stripped(self, session):
        voice = voices.enroll(session, "  Alice  ", ALICE)
        assert voice.name == "Alice"

    def test_blank_name_is_rejected(self, session):
        with pytest.raises(ValueError, match="blank"):
            voices.enroll(session, "   ", ALICE)

    def test_empty_embedding_is_rejected(self, session):
        with pytest.raises(ValueError, match="empty"):
            voices.enroll(session, "Alice", [])

    def test_a_person_can_have_several_samples(self, session):
        voices.enroll(session, "Alice", ALICE)
        voices.enroll(session, "Alice", ALICE_VARIANT)
        session.commit()
        assert len(session.query(Voice).filter_by(name="Alice").all()) == 2


class TestRank:
    def test_orders_by_similarity(self, session):
        for name, v in (("Alice", ALICE), ("Bob", BOB), ("Carol", CAROL)):
            voices.enroll(session, name, v)
        session.commit()

        ranked = voices.rank(session, ALICE_VARIANT, limit=3)
        assert ranked[0].name == "Alice"
        assert ranked[0].similarity > ranked[1].similarity

    def test_uses_a_persons_best_sample_not_their_average(self, session):
        """Averaging would blur a voice across recording conditions.

        Alice has one sample that matches the probe well and one that does not.
        Her score must come from the good one, so adding samples can only help.
        """
        voices.enroll(session, "Alice", ALICE_VARIANT)
        voices.enroll(session, "Alice", BOB)  # a poor-quality enrollment
        session.commit()

        ranked = voices.rank(session, ALICE, limit=1)
        assert ranked[0].similarity == pytest.approx(
            voices.similarity(ALICE, ALICE_VARIANT), abs=1e-6
        )

    def test_reports_sample_count(self, session):
        voices.enroll(session, "Alice", ALICE)
        voices.enroll(session, "Alice", ALICE_VARIANT)
        session.commit()
        assert voices.rank(session, ALICE)[0].sample_count == 2

    def test_incompatible_dimensions_are_skipped(self, session):
        # A model change producing different-sized vectors must not be compared
        # against old samples as if they were meaningful.
        voices.enroll(session, "Alice", vec(1.0, dim=8))
        session.commit()
        assert voices.rank(session, vec(1.0, dim=192)) == []

    def test_empty_library_and_empty_probe(self, session):
        assert voices.rank(session, ALICE) == []
        voices.enroll(session, "Alice", ALICE)
        session.commit()
        assert voices.rank(session, []) == []


class TestBestMatch:
    def test_returns_a_confident_match(self, session):
        voices.enroll(session, "Alice", ALICE)
        session.commit()
        match = voices.best_match(session, ALICE_VARIANT, threshold=0.5)
        assert match is not None and match.name == "Alice"

    def test_refuses_a_weak_match(self, session):
        """The important refusal.

        An unnamed "Speaker 2" is honest; a wrong name is silently misleading
        in a transcript someone will trust months later.
        """
        voices.enroll(session, "Alice", ALICE)
        session.commit()
        assert voices.best_match(session, BOB, threshold=0.5) is None

    def test_threshold_boundary(self, session):
        voices.enroll(session, "Alice", ALICE)
        session.commit()
        score = voices.similarity(ALICE, ALICE_VARIANT)
        assert voices.best_match(session, ALICE_VARIANT, threshold=score - 0.01)
        assert voices.best_match(session, ALICE_VARIANT, threshold=score + 0.01) is None


class TestIdentifyAll:
    def test_names_each_diarized_speaker(self, session):
        voices.enroll(session, "Alice", ALICE)
        voices.enroll(session, "Bob", BOB)
        session.commit()

        result = voices.identify_all(
            session, {"SPEAKER_00": BOB, "SPEAKER_01": ALICE}, threshold=0.5
        )
        assert result["SPEAKER_00"].name == "Bob"
        assert result["SPEAKER_01"].name == "Alice"

    def test_unknown_speakers_are_left_out(self, session):
        voices.enroll(session, "Alice", ALICE)
        session.commit()
        result = voices.identify_all(
            session, {"SPEAKER_00": ALICE, "SPEAKER_01": CAROL}, threshold=0.5
        )
        assert "SPEAKER_00" in result
        assert "SPEAKER_01" not in result

    def test_a_person_is_assigned_at_most_once(self, session):
        """Two diarized speakers matching one voice means diarization split
        them. Naming both would hide that; the weaker one stays anonymous."""
        voices.enroll(session, "Alice", ALICE)
        session.commit()

        result = voices.identify_all(
            session,
            {"SPEAKER_00": ALICE_VARIANT, "SPEAKER_01": ALICE},
            threshold=0.5,
        )
        assert len(result) == 1
        # The stronger match wins: SPEAKER_01 is an exact match.
        assert result["SPEAKER_01"].name == "Alice"

    def test_no_embeddings_is_not_an_error(self, session):
        assert voices.identify_all(session, {}, threshold=0.5) == {}


class TestEnrolledAndForget:
    def test_lists_people_with_counts(self, session):
        voices.enroll(session, "Alice", ALICE, speech_s=10.0)
        voices.enroll(session, "Alice", ALICE_VARIANT, speech_s=5.0)
        voices.enroll(session, "Bob", BOB, speech_s=7.0)
        session.commit()

        people = {p.name: p for p in voices.enrolled(session)}
        assert people["Alice"].samples == 2
        assert people["Alice"].total_speech_s == pytest.approx(15.0)
        assert people["Bob"].samples == 1

    def test_empty_library(self, session):
        assert voices.enrolled(session) == []

    def test_forget_removes_every_sample(self, session):
        voices.enroll(session, "Alice", ALICE)
        voices.enroll(session, "Alice", ALICE_VARIANT)
        session.commit()

        assert voices.forget(session, "Alice") == 2
        session.commit()
        assert voices.enrolled(session) == []

    def test_forget_an_unknown_name(self, session):
        assert voices.forget(session, "Nobody") == 0
