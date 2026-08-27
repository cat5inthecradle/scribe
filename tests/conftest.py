from __future__ import annotations

import pytest

from scribe.types import BackendInfo, PipelineInfo, SourceInfo, Transcript, Word


def w(text: str, start: float, end: float, conf: float | None = None) -> Word:
    return Word(t=text, start=start, end=end, conf=conf)


def sequence(*texts: str, start: float = 0.0, dur: float = 0.30, gap: float = 0.05):
    """Evenly spaced words, so tests can talk about attribution not arithmetic."""
    out, t = [], start
    for text in texts:
        out.append(w(text, round(t, 3), round(t + dur, 3)))
        t += dur + gap
    return out


@pytest.fixture
def make_transcript():
    def _make(turns, speakers, *, filename="meeting.m4a", duration_s=60.0):
        return Transcript(
            source=SourceInfo(filename=filename, sha256="deadbeef", duration_s=duration_s),
            pipeline=PipelineInfo(
                asr=BackendInfo(backend="fake", model="fake-asr"),
                diarization=BackendInfo(backend="fake", model="fake-diar"),
                created_at="2026-08-26T12:00:00Z",
            ),
            speakers=speakers,
            turns=turns,
        )

    return _make


# --- Postgres-backed fixtures -------------------------------------------------
#
# These use a real Postgres because the queue's whole correctness argument rests
# on `FOR UPDATE SKIP LOCKED`, which no SQLite or in-memory fake reproduces.
# Testing it against a substitute would test the substitute.

import os  # noqa: E402
from urllib.parse import urlsplit, urlunsplit  # noqa: E402

DEFAULT_DEV_URL = "postgresql+psycopg://scribe:scribe@localhost:5433/scribe"
TEST_DB_NAME = "scribe_test"


def _test_database_url() -> str:
    """Point at a dedicated database so a test run cannot touch dev data."""
    base = os.environ.get("SCRIBE_TEST_DATABASE_URL")
    if base:
        return base
    parts = urlsplit(os.environ.get("SCRIBE_DATABASE_URL", DEFAULT_DEV_URL))
    return urlunsplit(parts._replace(path=f"/{TEST_DB_NAME}"))


def _create_test_database(url: str) -> None:
    import psycopg

    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    admin = urlunsplit(parts._replace(scheme="postgresql", path="/postgres"))
    with psycopg.connect(admin, autocommit=True, connect_timeout=5) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session")
def db_settings(tmp_path_factory):
    """Session-wide settings pointed at a freshly created test database."""
    from scribe.config import Settings
    from scribe.db import engine_for, reset_engine
    from scribe.models import Base

    url = _test_database_url()
    try:
        _create_test_database(url)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable ({exc}); run: docker compose up -d")

    settings = Settings(
        database_url=url, data_dir=tmp_path_factory.mktemp("scribe-data")
    )
    settings.ensure_dirs()

    reset_engine()
    Base.metadata.drop_all(engine_for(settings))
    Base.metadata.create_all(engine_for(settings))
    yield settings
    reset_engine()


@pytest.fixture
def db(db_settings):
    """Reset both the queue and the data directories between tests.

    Truncating the tables alone is not enough: intake/work/out are
    session-scoped, so files dropped by one test would show up in the next
    test's scan.
    """
    import shutil

    from sqlalchemy import text

    from scribe.db import engine_for

    with engine_for(db_settings).begin() as conn:
        conn.execute(text("TRUNCATE jobs, job_events RESTART IDENTITY CASCADE"))

    for directory in (db_settings.intake, db_settings.work, db_settings.out):
        shutil.rmtree(directory, ignore_errors=True)
    db_settings.ensure_dirs()
    return db_settings


@pytest.fixture
def session(db):
    """An open session. Commit explicitly when a test needs durability."""
    from scribe.db import session_factory

    s = session_factory(db)()
    try:
        yield s
    finally:
        s.rollback()
        s.close()
