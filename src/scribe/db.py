"""Engine and session management.

Synchronous on purpose. The work here is CPU-bound inference in a worker
process, not IO concurrency, so async would add colour to every function
signature and buy nothing. FastAPI runs sync handlers in a threadpool, which is
the right trade for a single-user service.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from scribe.config import Settings

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def engine_for(settings: Settings) -> Engine:
    """Process-wide engine, created once."""
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            # A worker holds one connection for a long transcription; recycling
            # avoids a silently dropped connection surfacing as a mid-job error.
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=5,
            future=True,
        )
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def session_factory(settings: Settings) -> sessionmaker[Session]:
    engine_for(settings)
    assert _Session is not None
    return _Session


@contextmanager
def session_scope(settings: Settings) -> Iterator[Session]:
    """Transaction boundary: commit on success, roll back on any exception."""
    factory = session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine. Tests use this to switch databases."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None


def check_connection(settings: Settings) -> tuple[bool, str]:
    """Whether the database is reachable, and its version or the error."""
    try:
        with engine_for(settings).connect() as conn:
            version = conn.execute(text("select version()")).scalar_one()
        return True, str(version).split(" on ")[0]
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return False, str(exc).strip().splitlines()[0]
