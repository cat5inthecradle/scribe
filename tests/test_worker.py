"""Worker behaviour, with the pipeline stubbed.

Covers the parts that are about the queue rather than about audio: that success
and failure land in the right state, that retries are bounded, and that a
worker's own progress reporting cannot break the job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select, text

from scribe import queue
from scribe.models import Job, JobEvent, JobState
from scribe.pipeline import PipelineResult
from scribe.types import (
    BackendInfo,
    PipelineInfo,
    SourceInfo,
    Transcript,
    Turn,
    Word,
)
from scribe.worker import Worker


def fake_result(out_dir: Path) -> PipelineResult:
    transcript = Transcript(
        source=SourceInfo(filename="rec.wav", sha256="a" * 64, duration_s=42.0),
        pipeline=PipelineInfo(
            asr=BackendInfo(backend="mlx", model="parakeet-test"),
            diarization=BackendInfo(backend="pyannote", model="community-1"),
            created_at="2026-08-26T12:00:00Z",
        ),
        speakers=[],
        turns=[
            Turn(
                speaker="SPEAKER_00",
                start=0.0,
                end=1.0,
                text="hello",
                words=[Word(t="hello", start=0.0, end=1.0)],
            )
        ],
    )
    return PipelineResult(
        transcript=transcript,
        out_dir=out_dir,
        written=[out_dir / "transcript.json"],
        durations={"transcribe": 2.0, "diarize": 8.0},
    )


@pytest.fixture
def enqueued(db, session):
    source = db.work / "rec.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"\0" * 64)
    job, _ = queue.submit(
        session,
        sha256="a" * 64,
        name="rec.wav",
        path=str(source),
        duration_s=42.0,
    )
    session.commit()
    return job.id


def events_for(session, job_id) -> list[str]:
    return [
        e.message
        for e in session.scalars(select(JobEvent).where(JobEvent.job_id == job_id))
    ]


class TestSuccess:
    def test_job_reaches_done_with_metadata(self, db, session, enqueued, monkeypatch):
        out = db.out / "rec-out"
        monkeypatch.setattr(
            "scribe.pipeline.run", lambda *a, **k: fake_result(out)
        )
        worker = Worker(db, once=True)
        assert worker.run() == 1

        job = session.get(Job, enqueued)
        session.refresh(job)
        assert job.state == JobState.DONE
        assert job.progress == 1.0
        assert job.output_dir == str(out)
        assert job.asr_backend == "mlx"
        assert job.asr_model == "parakeet-test"
        assert job.diar_model == "community-1"
        assert job.duration_s == 42.0
        assert job.finished_at is not None

    def test_stage_timings_are_logged(self, db, session, enqueued, monkeypatch):
        monkeypatch.setattr(
            "scribe.pipeline.run", lambda *a, **k: fake_result(db.out / "o")
        )
        Worker(db, once=True).run()
        messages = events_for(session, enqueued)
        assert any("diarize took 8.0s" in m for m in messages)

    def test_once_exits_on_an_empty_queue(self, db, monkeypatch):
        monkeypatch.setattr(
            "scribe.pipeline.run", lambda *a, **k: fake_result(db.out / "o")
        )
        assert Worker(db, once=True).run() == 0

    def test_once_drains_multiple_jobs(self, db, session, monkeypatch):
        for i in range(3):
            source = db.work / f"r{i}.wav"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"\0" * 8)
            queue.submit(
                session, sha256=str(i) * 64, name=f"r{i}.wav", path=str(source)
            )
        session.commit()
        monkeypatch.setattr(
            "scribe.pipeline.run", lambda *a, **k: fake_result(db.out / "o")
        )
        assert Worker(db, once=True).run() == 3


class TestFailure:
    def test_failure_requeues_with_backoff_and_records_the_error(
        self, db, session, enqueued, monkeypatch
    ):
        def explode(*a, **k):
            raise RuntimeError("ffmpeg died")

        monkeypatch.setattr("scribe.pipeline.run", explode)
        worker = Worker(db, once=True)
        worker.run()

        job = session.get(Job, enqueued)
        session.refresh(job)
        assert worker.failed == 1, "backoff should stop it retrying immediately"
        assert job.state == JobState.PENDING
        assert "ffmpeg died" in (job.last_error or "")
        assert job.attempts == 1
        assert job.available_at is not None, "no backoff was applied"

    def test_a_backed_off_job_is_not_claimable_yet(
        self, db, session, enqueued, monkeypatch
    ):
        monkeypatch.setattr(
            "scribe.pipeline.run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        Worker(db, once=True).run()
        assert queue.claim(session, "another-worker") is None

    def test_a_transient_failure_succeeds_on_the_next_attempt(
        self, db, session, enqueued, monkeypatch
    ):
        """The point of backoff: attempt 2 happens after the fault clears."""
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("model download hiccup")
            return fake_result(db.out / "o")

        monkeypatch.setattr("scribe.pipeline.run", flaky)
        Worker(db, once=True).run()

        # Simulate the backoff window elapsing.
        session.execute(
            text("UPDATE jobs SET available_at = now() - interval '1 second'")
        )
        session.commit()
        assert Worker(db, once=True).run() == 1

        job = session.get(Job, enqueued)
        session.refresh(job)
        assert job.state == JobState.DONE
        assert job.attempts == 2

    def test_retries_are_bounded_and_the_job_dies(
        self, db, session, enqueued, monkeypatch
    ):
        monkeypatch.setattr(
            "scribe.pipeline.run",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("always broken")),
        )
        # Backoff now separates attempts, so step past each window in turn.
        for _ in range(3):
            Worker(db, once=True).run()
            session.execute(text("UPDATE jobs SET available_at = NULL"))
            session.commit()

        job = session.get(Job, enqueued)
        session.refresh(job)
        assert job.state == JobState.DEAD
        assert job.attempts == 3
        assert job.finished_at is not None

    def test_a_poison_job_does_not_block_the_queue(
        self, db, session, enqueued, monkeypatch
    ):
        good = db.work / "good.wav"
        good.write_bytes(b"\0" * 8)
        queue.submit(session, sha256="b" * 64, name="good.wav", path=str(good))
        session.commit()

        def selective(source, *a, **k):
            if Path(source).name == "rec.wav":
                raise RuntimeError("poison")
            return fake_result(db.out / "good-out")

        monkeypatch.setattr("scribe.pipeline.run", selective)
        Worker(db, once=True).run()

        jobs = {j.source_name: j for j in session.scalars(select(Job)).all()}
        # The healthy job completes in the same pass...
        assert jobs["good.wav"].state == JobState.DONE
        # ...and the failing one steps aside into its backoff window rather
        # than being retried ahead of it.
        assert jobs["rec.wav"].state == JobState.PENDING
        assert jobs["rec.wav"].available_at is not None
        assert queue.claim(session, "next-worker") is None


class TestProvenance:
    """Regression: the queue renames sources, and the name must survive.

    The intake watcher parks files at ``work/<job id>/source.ext``, so a
    pipeline that slugs its actual path names every queued output "source" and
    records ``source.ext`` as the filename — losing all provenance. Invisible
    via `scribe run`; only the queue path exposes it.
    """

    def test_original_filename_reaches_the_pipeline(
        self, db, session, enqueued, monkeypatch
    ):
        seen: dict[str, object] = {}

        def capture(source, settings, *, display_name=None, **k):
            seen["display_name"] = display_name
            seen["source"] = Path(source).name
            return fake_result(db.out / "o")

        monkeypatch.setattr("scribe.pipeline.run", capture)
        Worker(db, once=True).run()
        assert seen["display_name"] == "rec.wav"


class TestProgress:
    def test_progress_and_stage_are_recorded(
        self, db, session, enqueued, monkeypatch
    ):
        def run_with_progress(source, settings, *, on_progress=None, **k):
            if on_progress:
                on_progress("transcribe", 0.5)
                on_progress("diarize", 0.5)
            return fake_result(db.out / "o")

        monkeypatch.setattr("scribe.pipeline.run", run_with_progress)
        Worker(db, once=True).run()
        assert any("stage: diarize" in m for m in events_for(session, enqueued))

    def test_a_broken_progress_write_does_not_fail_the_job(
        self, db, session, enqueued, monkeypatch
    ):
        def run_with_progress(source, settings, *, on_progress=None, **k):
            if on_progress:
                on_progress("nonexistent-stage", 0.5)
            return fake_result(db.out / "o")

        monkeypatch.setattr("scribe.pipeline.run", run_with_progress)
        assert Worker(db, once=True).run() == 1

        job = session.get(Job, enqueued)
        session.refresh(job)
        assert job.state == JobState.DONE


class TestStopping:
    def test_request_stop_prevents_new_claims(self, db, enqueued, monkeypatch):
        monkeypatch.setattr(
            "scribe.pipeline.run", lambda *a, **k: fake_result(db.out / "o")
        )
        worker = Worker(db)
        worker.request_stop()
        assert worker.run() == 0, "claimed work after being told to stop"
