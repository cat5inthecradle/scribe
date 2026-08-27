"""The job queue, implemented in Postgres.

`SELECT ... FOR UPDATE SKIP LOCKED` is what makes a broker unnecessary: many
workers can race for the same row and exactly one wins, without any of them
blocking. That single primitive replaces Redis, Celery, and RabbitMQ here.

Every timestamp comparison uses the server's `now()`. Workers may sit on
different machines with skewed clocks, and heartbeat staleness judged against a
worker's own clock would let a lagging worker have live work stolen from it.
"""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from scribe.models import ACTIVE_STATES, Job, JobEvent, JobState

#: Retry backoff, in seconds, indexed by attempt number. The last value
#: repeats. Short enough that a homelab queue still drains promptly, long
#: enough that a transient fault has a chance to clear.
RETRY_BACKOFF_S = (15.0, 60.0, 300.0)


def backoff_for(attempt: int) -> float:
    """Delay before a failed job's next attempt."""
    index = min(max(attempt, 1), len(RETRY_BACKOFF_S)) - 1
    return RETRY_BACKOFF_S[index]


def default_worker_id() -> str:
    """Host and pid: enough to tell two workers apart in the log."""
    return f"{socket.gethostname()}:{os.getpid()}"


# Claim exactly one pending job. The inner SELECT takes the row lock; SKIP
# LOCKED makes a competing worker step over it rather than wait.
_CLAIM = text(
    """
    UPDATE jobs SET
        state        = 'claimed',
        claimed_by   = :worker,
        claimed_at   = now(),
        heartbeat_at = now(),
        attempts     = attempts + 1,
        started_at   = COALESCE(started_at, now()),
        updated_at   = now()
    WHERE id = (
        SELECT id FROM jobs
        WHERE state = 'pending'
          AND (available_at IS NULL OR available_at <= now())
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    RETURNING id
    """
)

# A worker that crashed leaves its job claimed forever. Anything past the
# heartbeat window goes back in the queue, or dies if it is out of attempts.
_RECLAIM = text(
    """
    UPDATE jobs SET
        state        = CASE WHEN attempts >= max_attempts THEN 'dead'
                            ELSE 'pending' END,
        claimed_by   = NULL,
        claimed_at   = NULL,
        heartbeat_at = NULL,
        stage        = NULL,
        progress     = 0,
        available_at = NULL,
        last_error   = COALESCE(last_error, 'worker heartbeat expired'),
        finished_at  = CASE WHEN attempts >= max_attempts THEN now() END,
        updated_at   = now()
    WHERE state = ANY(:active)
      AND heartbeat_at < now() - make_interval(secs => :timeout)
    RETURNING id, state
    """
)


@dataclass(frozen=True, slots=True)
class Reclaimed:
    requeued: int
    died: int

    @property
    def total(self) -> int:
        return self.requeued + self.died


def log(
    session: Session,
    job_id: uuid.UUID,
    message: str,
    *,
    level: str = "info",
    stage: str | None = None,
) -> None:
    """Append to a job's event log. This is what the UI streams."""
    session.add(
        JobEvent(job_id=job_id, message=message, level=level, stage=stage)
    )


def submit(
    session: Session,
    *,
    sha256: str,
    name: str,
    path: str,
    size_bytes: int | None = None,
    duration_s: float | None = None,
    force: bool = False,
    max_attempts: int = 3,
) -> tuple[Job, bool]:
    """Enqueue a file, deduplicating by content hash.

    Returns ``(job, created)``. An identical file already known is a no-op
    unless `force`, which resets the existing row rather than inserting a
    duplicate — so history stays one row per distinct recording.
    """
    existing = session.scalar(select(Job).where(Job.source_sha256 == sha256))
    if existing is not None:
        if force:
            _reset(existing, path)
            log(session, existing.id, f"resubmitted (force): {name}")
        return existing, False

    job = Job(
        source_sha256=sha256,
        source_name=name,
        source_path=path,
        source_bytes=size_bytes,
        duration_s=duration_s,
        state=JobState.PENDING,
        max_attempts=max_attempts,
    )
    try:
        # Savepoint: two workers can submit the same file simultaneously, and
        # the loser must recover rather than poison the whole transaction.
        with session.begin_nested():
            session.add(job)
            session.flush()
    except IntegrityError:
        existing = session.scalar(select(Job).where(Job.source_sha256 == sha256))
        if existing is None:
            raise
        return existing, False

    log(session, job.id, f"queued: {name}")
    return job, True


def _reset(job: Job, path: str) -> None:
    job.state = JobState.PENDING
    job.stage = None
    job.progress = 0.0
    job.attempts = 0
    job.last_error = None
    job.claimed_by = None
    job.claimed_at = None
    job.heartbeat_at = None
    job.started_at = None
    job.finished_at = None
    job.available_at = None
    job.source_path = path


def claim(session: Session, worker_id: str) -> Job | None:
    """Take the oldest pending job, or None if the queue is empty."""
    row = session.execute(_CLAIM, {"worker": worker_id}).first()
    if row is None:
        return None
    job = session.get(Job, row[0])
    if job is not None:
        session.refresh(job)
        log(session, job.id, f"claimed by {worker_id}")
    return job


def reclaim_stale(session: Session, timeout_s: float) -> Reclaimed:
    """Requeue jobs whose worker stopped heartbeating."""
    rows = session.execute(
        _RECLAIM,
        {"active": [s.value for s in ACTIVE_STATES], "timeout": timeout_s},
    ).all()
    requeued = sum(1 for _, state in rows if state == JobState.PENDING)
    died = sum(1 for _, state in rows if state == JobState.DEAD)
    for job_id, state in rows:
        log(
            session,
            job_id,
            "worker heartbeat expired; "
            + ("requeued" if state == JobState.PENDING else "gave up"),
            level="warning",
        )
    return Reclaimed(requeued=requeued, died=died)


def heartbeat(
    session: Session,
    job: Job,
    *,
    stage: str | None = None,
    progress: float | None = None,
) -> None:
    """Refresh the liveness timestamp, optionally recording progress."""
    job.state = JobState.RUNNING
    if stage is not None:
        job.stage = stage
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    # Assigned as a SQL expression, not a Python datetime: the server's clock is
    # the only one all workers agree on. Going through the ORM (rather than a
    # raw UPDATE) keeps the in-memory instance and the row from diverging.
    job.heartbeat_at = func.now()


def complete(
    session: Session,
    job: Job,
    *,
    output_dir: str,
    duration_s: float | None = None,
    asr_backend: str | None = None,
    asr_model: str | None = None,
    diar_model: str | None = None,
) -> None:
    job.state = JobState.DONE
    job.stage = None
    job.progress = 1.0
    job.output_dir = output_dir
    job.last_error = None
    job.claimed_by = None
    if duration_s is not None:
        job.duration_s = duration_s
    if asr_backend:
        job.asr_backend = asr_backend
    if asr_model:
        job.asr_model = asr_model
    if diar_model:
        job.diar_model = diar_model
    job.finished_at = func.now()
    job.available_at = None
    log(session, job.id, f"done -> {output_dir}")


def fail(session: Session, job: Job, error: str) -> None:
    """Record a failure, requeueing if attempts remain."""
    job.last_error = error[:4000]
    job.stage = None
    job.progress = 0.0
    job.claimed_by = None
    job.claimed_at = None
    job.heartbeat_at = None

    exhausted = job.attempts >= job.max_attempts
    job.state = JobState.DEAD if exhausted else JobState.PENDING

    if exhausted:
        job.finished_at = func.now()
        job.available_at = None
        delay = 0.0
    else:
        delay = backoff_for(job.attempts)
        job.available_at = func.now() + timedelta(seconds=delay)

    log(
        session,
        job.id,
        f"attempt {job.attempts}/{job.max_attempts} failed: {error[:500]}"
        + ("" if exhausted else f" (retrying in {delay:.0f}s)"),
        level="error",
    )


def retry(session: Session, job: Job) -> None:
    """Manually revive a dead job."""
    _reset(job, job.source_path)
    log(session, job.id, "manually retried")


def counts(session: Session) -> dict[str, int]:
    """Job totals per state, for the CLI and UI summaries."""
    rows = session.execute(
        text("SELECT state, count(*) FROM jobs GROUP BY state")
    ).all()
    return {state: n for state, n in rows}
