"""Intake discovery.

The hard requirement is not noticing files — it is *not* grabbing them too
early. Copying a large recording yields a readable, growing file long before the
copy finishes, and transcribing a half-written file produces a plausible-looking
truncated transcript, which is worse than an obvious failure.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
from sqlalchemy import select

from scribe import intake
from scribe.models import Job, JobState


def write_audio(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", str(seconds),
         "-i", "anullsrc=r=16000:cl=mono", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def settle(path: Path, age_s: float = 30.0) -> Path:
    """Backdate mtime so the quiescence check passes without a real sleep."""
    when = time.time() - age_s
    os.utime(path, (when, when))
    return path


@pytest.fixture
def scanner(db):
    return intake.IntakeScanner(db)


class TestStabilityGate:
    def test_a_just_written_file_is_withheld(self, db, scanner):
        # The core guard: a file touched moments ago may still be copying.
        write_audio(db.intake / "new.wav")
        assert scanner.stable_files() == []
        assert scanner.stable_files() == [], "still too fresh on a second look"

    def test_a_settled_file_is_offered_on_the_first_scan(self, db, scanner):
        # Stateless acceptance is what lets `scribe scan` and a freshly
        # restarted watcher pick up files at all.
        path = settle(write_audio(db.intake / "old.wav"))
        assert scanner.stable_files() == [path]

    def test_a_growing_file_is_withheld(self, db, scanner):
        path = db.intake / "copying.wav"
        for size in (1024, 4096, 8192):
            path.write_bytes(b"\0" * size)
            assert scanner.stable_files() == [], f"claimed at {size} bytes"

    def test_a_chunked_writer_pausing_is_withheld(self, db, scanner):
        """A writer appending in bursts can look quiescent between bursts.

        Backdated mtime makes the age check pass, so only agreement across
        scans can catch this — which is why both checks exist.
        """
        path = settle(write_audio(db.intake / "bursty.wav"))
        assert scanner.stable_files() == [path]      # looks settled

        path.write_bytes(path.read_bytes() + b"\0" * 4096)
        settle(path)                                 # backdated again
        assert scanner.stable_files() == [], "size changed since last scan"

    def test_empty_file_is_withheld(self, db, scanner):
        empty = db.intake / "touched.wav"
        empty.touch()
        settle(empty)
        assert scanner.stable_files() == []

    def test_junk_is_ignored(self, db, scanner):
        for name in (".DS_Store", "notes.txt"):
            (db.intake / name).write_text("junk")
            settle(db.intake / name)
        settle(write_audio(db.intake / "real.wav"))
        assert [p.name for p in scanner.stable_files()] == ["real.wav"]

    def test_subdirectories_are_not_scanned(self, db, scanner):
        settle(write_audio(db.intake / intake.DUPLICATES_DIRNAME / "old.wav"))
        assert scanner.stable_files() == []

    def test_vanishing_file_does_not_raise(self, db, scanner):
        path = settle(write_audio(db.intake / "fleeting.wav"))
        path.unlink()
        assert scanner.stable_files() == []


class TestSubmitStable:
    def test_queues_and_moves_the_file_out_of_intake(self, db, session, scanner):
        path = settle(write_audio(db.intake / "meeting.wav"))

        report = intake.submit_stable(session, db, scanner)
        session.commit()

        assert report.queued == ["meeting.wav"]
        assert not path.exists(), "source left in intake; it would be re-queued"

        job = session.scalars(select(Job)).one()
        assert job.state == JobState.PENDING
        assert job.source_name == "meeting.wav"
        assert Path(job.source_path).is_file()

    def test_originals_are_archived_never_left_in_scratch(self, db, session, scanner):
        """Originals must not live under `work`.

        The move out of intake means the destination briefly holds the only
        copy of the recording, while `work` is scratch that gets emptied —
        this combination once destroyed a real file.
        """
        settle(write_audio(db.intake / "irreplaceable.wav"))
        intake.submit_stable(session, db, scanner)
        session.commit()

        job = session.scalars(select(Job)).one()
        source = Path(job.source_path)
        assert source.is_file()
        assert db.archive in source.parents, f"{source} is not in the archive"
        assert db.work not in source.parents, "original placed in scratch"
        assert source.name == "irreplaceable.wav", "archive should stay browsable"

    def test_archive_collision_keeps_both_files(self, db, session, scanner):
        # Same name, different content: neither may be silently overwritten.
        (db.archive).mkdir(parents=True, exist_ok=True)
        existing = write_audio(db.archive / "dup.wav", seconds=1.0)
        original_bytes = existing.read_bytes()

        settle(write_audio(db.intake / "dup.wav", seconds=2.0))
        intake.submit_stable(session, db, scanner)
        session.commit()

        assert existing.read_bytes() == original_bytes, "existing file clobbered"
        assert len(list(db.archive.glob("dup*.wav"))) == 2

    def test_emptying_work_does_not_lose_the_source(self, db, session, scanner):
        """`work` must be safe to delete at any time."""
        import shutil

        settle(write_audio(db.intake / "safe.wav"))
        intake.submit_stable(session, db, scanner)
        session.commit()

        shutil.rmtree(db.work, ignore_errors=True)
        job = session.scalars(select(Job)).one()
        assert Path(job.source_path).is_file(), "source lost when work was cleared"

    def test_duration_and_size_are_recorded_at_submit(self, db, session, scanner):
        settle(write_audio(db.intake / "clip.wav", seconds=3.0))
        intake.submit_stable(session, db, scanner)
        session.commit()

        job = session.scalars(select(Job)).one()
        assert job.duration_s == pytest.approx(3.0, abs=0.2)
        assert job.source_bytes and job.source_bytes > 0

    def test_duplicate_content_is_set_aside_not_requeued(self, db, session, scanner):
        original = settle(write_audio(db.intake / "first.wav"))
        payload = original.read_bytes()
        intake.submit_stable(session, db, scanner)
        session.commit()

        # Same bytes, different name — the classic re-drop.
        again = db.intake / "second.wav"
        again.write_bytes(payload)
        settle(again)
        report = intake.submit_stable(session, db, scanner)
        session.commit()

        assert report.duplicates == ["second.wav"]
        assert len(session.scalars(select(Job)).all()) == 1
        assert (db.intake / intake.DUPLICATES_DIRNAME / "second.wav").is_file()
        assert not again.exists()

    def test_unreadable_media_is_reported_not_queued(self, db, session, scanner):
        # A .wav extension is trusted without probing, so this reaches inspect()
        # and must fail there rather than becoming a doomed job.
        corrupt = db.intake / "corrupt.wav"
        corrupt.write_text("this is not audio")
        settle(corrupt)
        report = intake.submit_stable(session, db, scanner)
        session.commit()

        assert [n for n, _ in report.errors] == ["corrupt.wav"]
        assert report.queued == []
        assert session.scalars(select(Job)).all() == []

    def test_scan_once_is_idempotent(self, db, session, scanner):
        settle(write_audio(db.intake / "a.wav"))
        intake.submit_stable(session, db, scanner)
        session.commit()

        second = intake.submit_stable(session, db, scanner)
        session.commit()
        assert second.total == 0


class TestWatchOnce:
    def test_single_pass_withholds_a_just_written_file(self, db):
        write_audio(db.intake / "brand-new.wav")
        assert intake.watch(db, once=True).total == 0

    def test_single_pass_queues_a_settled_file(self, db):
        # Regression: a fresh scanner has no cross-scan history, so requiring it
        # meant `scribe scan` could never queue anything at all.
        settle(write_audio(db.intake / "settled.wav"))
        assert intake.watch(db, once=True).queued == ["settled.wav"]
