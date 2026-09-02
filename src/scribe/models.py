"""Database schema.

Two tables. `jobs` is both the record and the queue — claiming is a row-level
lock, which is why no broker (Redis, Celery, RabbitMQ) appears anywhere in this
project. `job_events` is the append-only log the UI streams.

Every timestamp is set server-side with `now()`. Workers may run on different
machines with skewed clocks, and heartbeat staleness has to be judged against
one clock or a lagging worker gets its work stolen mid-job.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobState(enum.StrEnum):
    """Job lifecycle.

    There is deliberately no separate "failed but retrying" state: a job that
    errored with attempts left goes straight back to PENDING, and
    ``attempts > 0 AND last_error IS NOT NULL`` already distinguishes it from
    one that has never run. `max_attempts` caps the retry loop, so no backoff
    timestamp is needed either.
    """

    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    DONE = "done"
    DEAD = "dead"


#: States a worker currently holds. Used for staleness reclaim.
ACTIVE_STATES = (JobState.CLAIMED, JobState.RUNNING)

#: States that mean "no worker will touch this again without being asked".
TERMINAL_STATES = (JobState.DONE, JobState.DEAD)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Content hash, not path: re-dropping the same recording under a new name is
    # the same job. Unique, so two concurrent submits cannot both win.
    source_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_name: Mapped[str] = mapped_column(String(512))
    source_path: Mapped[str] = mapped_column(Text)
    source_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    state: Mapped[str] = mapped_column(
        String(16), default=JobState.PENDING, index=True
    )
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    available_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """Earliest time this job may be claimed. Set on failure, for backoff.

    Without it, retries fire instantly: a job that fails in 50ms burns all
    three attempts inside a tenth of a second, so a transient fault (a model
    download hiccup, a brief database blip) is guaranteed to consume every
    attempt before it could possibly have cleared. NULL means "claimable now".
    """

    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    output_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    asr_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    asr_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    diar_model: Mapped[str | None] = mapped_column(String(256), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[list[JobEvent]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("progress >= 0 AND progress <= 1", name="ck_jobs_progress"),
        CheckConstraint("attempts >= 0", name="ck_jobs_attempts"),
        # Partial index: the claim query only ever scans pending rows, and this
        # keeps that lookup cheap once thousands of finished jobs accumulate.
        Index(
            "ix_jobs_pending",
            "available_at",
            "created_at",
            postgresql_where=(state == JobState.PENDING),
        ),
    )

    def __repr__(self) -> str:
        return f"<Job {str(self.id)[:8]} {self.state} {self.source_name!r}>"


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    level: Mapped[str] = mapped_column(String(16), default="info")
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="events")

    __table_args__ = (
        # The UI reads one job's events newest-last; this serves that directly.
        Index("ix_job_events_job_ts", "job_id", "ts"),
    )


class Voice(Base):
    """One enrolled voice sample.

    A person gets a *row per sample*, not one averaged vector. Averaging blurs
    a voice across recording conditions, so a hoarse day or a different mic
    would drag the reference toward the middle and weaken every future match.
    Matching takes the maximum similarity over a person's samples instead, so
    each new sample can only help.
    """

    __tablename__ = "voices"

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), index=True)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(Float))
    dim: Mapped[int] = mapped_column(Integer)
    """Stored explicitly so a model change producing different-sized vectors is
    detected rather than silently compared against incompatible samples."""

    source_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    """Recording this sample came from, so a bad enrollment can be traced."""

    speech_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    """How much speech backed this sample. Short samples are less reliable."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("dim > 0", name="ck_voices_dim"),
        Index("ix_voices_name_created", "name", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Voice {self.name!r} dim={self.dim}>"
