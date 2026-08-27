"""Queue semantics against a real Postgres.

The correctness argument for this queue is entirely about concurrency: exactly
one worker wins a contested row, a crashed worker's job returns to the queue,
and a poison job eventually stops retrying. None of that can be tested against
a fake, so these run against Postgres or skip.
"""

from __future__ import annotations

from sqlalchemy import select, text

from scribe import queue
from scribe.db import session_factory
from scribe.models import Job, JobEvent, JobState


def add(session, name="rec.m4a", sha=None, **kw) -> Job:
    job, _ = queue.submit(
        session,
        sha256=sha or name.ljust(64, "0")[:64],
        name=name,
        path=f"/tmp/{name}",
        size_bytes=1234,
        duration_s=60.0,
        **kw,
    )
    session.commit()
    return job


class TestSubmit:
    def test_creates_a_pending_job(self, session):
        job = add(session)
        assert job.state == JobState.PENDING
        assert job.attempts == 0
        assert job.progress == 0.0

    def test_identical_content_is_deduplicated(self, session):
        first = add(session, "a.m4a", sha="x" * 64)
        second, created = queue.submit(
            session, sha256="x" * 64, name="renamed.m4a", path="/tmp/renamed.m4a"
        )
        assert not created
        assert second.id == first.id

    def test_force_resets_an_existing_job(self, session):
        job = add(session, "a.m4a", sha="y" * 64)
        job.state = JobState.DONE
        job.attempts = 3
        job.last_error = "boom"
        session.commit()

        again, created = queue.submit(
            session, sha256="y" * 64, name="a.m4a", path="/new/path", force=True
        )
        session.commit()
        assert not created and again.id == job.id
        assert again.state == JobState.PENDING
        assert again.attempts == 0
        assert again.last_error is None
        assert again.source_path == "/new/path"

    def test_submission_is_logged(self, session):
        job = add(session)
        events = session.scalars(
            select(JobEvent).where(JobEvent.job_id == job.id)
        ).all()
        assert any("queued" in e.message for e in events)


class TestClaim:
    def test_claims_the_oldest_job_first(self, session):
        first = add(session, "first.m4a")
        second = add(session, "second.m4a")
        # Make ordering unambiguous regardless of clock resolution.
        session.execute(
            text("UPDATE jobs SET created_at = now() - interval '1 hour' WHERE id=:i"),
            {"i": first.id},
        )
        session.commit()

        claimed = queue.claim(session, "worker-1")
        assert claimed is not None and claimed.id == first.id
        assert second.state == JobState.PENDING

    def test_claim_records_worker_and_increments_attempts(self, session):
        add(session)
        job = queue.claim(session, "worker-7")
        assert job is not None
        assert job.state == JobState.CLAIMED
        assert job.claimed_by == "worker-7"
        assert job.attempts == 1
        assert job.heartbeat_at is not None
        assert job.started_at is not None

    def test_empty_queue_returns_none(self, session):
        assert queue.claim(session, "worker-1") is None

    def test_only_one_of_two_workers_wins_the_same_job(self, db):
        """The core guarantee. Two live sessions, one row, no double-processing."""
        factory = session_factory(db)
        setup = factory()
        add(setup, "contested.m4a")
        setup.close()

        a, b = factory(), factory()
        try:
            first = queue.claim(a, "worker-a")
            # SKIP LOCKED: b must step over the locked row, not block on it.
            second = queue.claim(b, "worker-b")
            assert first is not None
            assert second is None, "two workers claimed the same job"
            a.commit()
            b.commit()
        finally:
            a.close()
            b.close()

    def test_two_workers_take_two_different_jobs(self, db):
        factory = session_factory(db)
        setup = factory()
        add(setup, "one.m4a")
        add(setup, "two.m4a")
        setup.close()

        a, b = factory(), factory()
        try:
            first = queue.claim(a, "worker-a")
            second = queue.claim(b, "worker-b")
            assert first is not None and second is not None
            assert first.id != second.id
            a.commit()
            b.commit()
        finally:
            a.close()
            b.close()


class TestHeartbeatAndReclaim:
    def test_fresh_claim_is_not_reclaimed(self, session):
        add(session)
        queue.claim(session, "worker-1")
        session.commit()
        assert queue.reclaim_stale(session, timeout_s=60).total == 0

    def test_stale_claim_is_requeued(self, session):
        add(session)
        job = queue.claim(session, "worker-1")
        assert job is not None
        session.execute(
            text("UPDATE jobs SET heartbeat_at = now() - interval '10 minutes'")
        )
        session.commit()

        result = queue.reclaim_stale(session, timeout_s=60)
        assert (result.requeued, result.died) == (1, 0)
        session.refresh(job)
        assert job.state == JobState.PENDING
        assert job.claimed_by is None

    def test_stale_claim_out_of_attempts_dies(self, session):
        add(session, max_attempts=1)
        queue.claim(session, "worker-1")
        session.execute(
            text("UPDATE jobs SET heartbeat_at = now() - interval '10 minutes'")
        )
        session.commit()

        result = queue.reclaim_stale(session, timeout_s=60)
        assert (result.requeued, result.died) == (0, 1)

    def test_reclaimed_job_can_be_claimed_again(self, session):
        add(session)
        queue.claim(session, "dead-worker")
        session.execute(
            text("UPDATE jobs SET heartbeat_at = now() - interval '10 minutes'")
        )
        session.commit()
        queue.reclaim_stale(session, timeout_s=60)
        session.commit()

        revived = queue.claim(session, "live-worker")
        assert revived is not None
        assert revived.claimed_by == "live-worker"
        assert revived.attempts == 2

    def test_heartbeat_marks_running_and_records_progress(self, session):
        add(session)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.heartbeat(session, job, stage="diarize", progress=0.5)
        session.commit()
        session.refresh(job)
        assert job.state == JobState.RUNNING
        assert job.stage == "diarize"
        assert job.progress == 0.5

    def test_progress_is_clamped(self, session):
        add(session)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.heartbeat(session, job, progress=1.7)
        session.commit()
        assert job.progress == 1.0


class TestCompleteAndFail:
    def test_complete_records_outputs_and_models(self, session):
        add(session)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.complete(
            session,
            job,
            output_dir="/data/out/rec",
            asr_backend="mlx",
            asr_model="parakeet",
            diar_model="community-1",
        )
        session.commit()
        session.refresh(job)
        assert job.state == JobState.DONE
        assert job.progress == 1.0
        assert job.output_dir == "/data/out/rec"
        assert job.asr_model == "parakeet"
        assert job.finished_at is not None

    def test_failure_requeues_while_attempts_remain(self, session):
        add(session, max_attempts=3)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.fail(session, job, "ffmpeg exploded")
        session.commit()
        session.refresh(job)
        assert job.state == JobState.PENDING
        assert job.last_error == "ffmpeg exploded"
        assert job.finished_at is None

    def test_failure_dies_once_attempts_are_exhausted(self, session):
        add(session, max_attempts=2)
        for _ in range(2):
            job = queue.claim(session, "worker-1")
            assert job is not None
            queue.fail(session, job, "still broken")
            session.commit()
            # Skip past the retry backoff so the next attempt can be claimed.
            session.execute(text("UPDATE jobs SET available_at = NULL"))
            session.commit()
        session.refresh(job)
        assert job.state == JobState.DEAD
        assert job.finished_at is not None
        assert queue.claim(session, "worker-1") is None, "dead job was re-claimed"

    def test_retry_revives_a_dead_job(self, session):
        add(session, max_attempts=1)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.fail(session, job, "nope")
        session.commit()
        assert job.state == JobState.DEAD

        queue.retry(session, job)
        session.commit()
        assert job.state == JobState.PENDING
        assert job.attempts == 0
        assert queue.claim(session, "worker-1") is not None

    def test_failures_are_logged_with_attempt_numbers(self, session):
        add(session, max_attempts=2)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.fail(session, job, "disk full")
        session.commit()
        messages = [
            e.message
            for e in session.scalars(
                select(JobEvent).where(JobEvent.job_id == job.id)
            )
        ]
        assert any("attempt 1/2" in m and "disk full" in m for m in messages)


class TestBackoff:
    """Retries are spaced so a transient fault has time to clear.

    Without this, a job failing in 50ms exhausts every attempt within a tenth
    of a second, and retrying is pure theatre.
    """

    def test_backoff_grows_then_plateaus(self):
        delays = [queue.backoff_for(n) for n in range(1, 6)]
        assert delays == sorted(delays), "backoff must not decrease"
        assert delays[0] > 0
        assert delays[-1] == delays[-2], "final delay should repeat"

    def test_out_of_range_attempts_are_clamped(self):
        assert queue.backoff_for(0) == queue.backoff_for(1)
        assert queue.backoff_for(99) == queue.RETRY_BACKOFF_S[-1]

    def test_failed_job_is_not_immediately_claimable(self, session):
        add(session, max_attempts=3)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.fail(session, job, "transient")
        session.commit()
        assert queue.claim(session, "worker-1") is None

    def test_claimable_again_once_the_window_passes(self, session):
        add(session, max_attempts=3)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.fail(session, job, "transient")
        session.commit()

        session.execute(
            text("UPDATE jobs SET available_at = now() - interval '1 second'")
        )
        session.commit()
        assert queue.claim(session, "worker-1") is not None

    def test_reclaim_clears_backoff(self, session):
        """A crashed worker is not a failed attempt; it should not be delayed."""
        add(session)
        queue.claim(session, "worker-1")
        session.execute(
            text(
                "UPDATE jobs SET heartbeat_at = now() - interval '10 minutes', "
                "available_at = now() + interval '10 minutes'"
            )
        )
        session.commit()

        queue.reclaim_stale(session, timeout_s=60)
        session.commit()
        assert queue.claim(session, "worker-1") is not None

    def test_force_resubmit_clears_backoff(self, session):
        add(session, "a.m4a", sha="z" * 64, max_attempts=3)
        job = queue.claim(session, "worker-1")
        assert job is not None
        queue.fail(session, job, "transient")
        session.commit()

        queue.submit(
            session, sha256="z" * 64, name="a.m4a", path="/tmp/a.m4a", force=True
        )
        session.commit()
        assert queue.claim(session, "worker-1") is not None


class TestCounts:
    def test_groups_by_state(self, session):
        add(session, "a.m4a")
        add(session, "b.m4a")
        queue.claim(session, "worker-1")
        session.commit()
        assert queue.counts(session) == {"pending": 1, "claimed": 1}
