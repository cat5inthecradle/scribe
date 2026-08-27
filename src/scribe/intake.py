"""Watched-folder discovery.

A periodic reconcile scan is the *primary* mechanism, not a fallback. inotify is
unreliable over NFS, which the eventual homelab will use, and a watcher that was
down during a copy misses the file permanently. Polling a directory is cheap and
cannot miss anything. `watchfiles` is layered on only to cut latency when it
does work.

The subtle part is not noticing files — it is not grabbing them too early.
Copying a 2 GB recording fires filesystem events, and yields a readable file,
long before the copy finishes.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from scribe import media, queue
from scribe.config import Settings
from scribe.db import session_scope

log = logging.getLogger(__name__)

#: Files skipped as duplicates land here so the scan does not re-hash them
#: forever. Prefixed so it is visibly not user content.
DUPLICATES_DIRNAME = "_duplicates"


@dataclass(frozen=True, slots=True)
class _Stat:
    size: int
    mtime: float


@dataclass(frozen=True, slots=True)
class SubmitReport:
    queued: list[str]
    duplicates: list[str]
    errors: list[tuple[str, str]]

    @property
    def total(self) -> int:
        return len(self.queued) + len(self.duplicates) + len(self.errors)


class IntakeScanner:
    """Finds media files in the intake directory that have stopped changing.

    Two independent checks, both of which must pass:

    * **Quiescence** — the file's mtime is at least `min_stable_age_s` old. This
      is stateless, so it works on the very first scan, which is what makes a
      one-shot `scribe scan` and a restarted watcher able to pick anything up.
    * **Agreement across scans** — if we observed the file previously, its size
      and mtime must be unchanged. This catches a slow writer that appends in
      bursts, where any single quiescence check could land in a lull.

    Neither alone is sufficient: quiescence misses a chunked writer mid-pause,
    and cross-scan agreement can never fire in a process that only scans once.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._previous: dict[Path, _Stat] = {}

    def _candidates(self) -> list[Path]:
        directory = self.settings.intake
        if not directory.is_dir():
            return []
        # Non-recursive by design: subdirectories are for our own bookkeeping
        # (see DUPLICATES_DIRNAME), and recursing would reclaim them.
        return [p for p in sorted(directory.iterdir()) if p.is_file()]

    def stable_files(self) -> list[Path]:
        """Media files unchanged since the previous scan."""
        stable: list[Path] = []
        current: dict[Path, _Stat] = {}

        for path in self._candidates():
            try:
                stat = path.stat()
            except OSError:
                continue  # vanished mid-scan; it will reappear or it won't

            now = _Stat(size=stat.st_size, mtime=stat.st_mtime)
            current[path] = now

            if stat.st_size == 0:
                continue  # a freshly created, not-yet-written file

            age = time.time() - stat.st_mtime
            if age < self.settings.min_stable_age_s:
                log.debug("intake: %s touched %.1fs ago; waiting", path.name, age)
                continue

            before = self._previous.get(path)
            if before is not None and before != now:
                log.debug("intake: still growing %s", path.name)
                continue

            # Stable for a full interval. Probing is the expensive check, so it
            # happens last and only once per file.
            if media.is_media_file(path):
                stable.append(path)
            else:
                log.debug("intake: ignoring non-media %s", path.name)

        self._previous = current
        return stable

    def forget(self, path: Path) -> None:
        """Stop tracking a file we have taken ownership of."""
        self._previous.pop(path, None)


def _move_into(path: Path, destination: Path) -> Path | None:
    """Move a file, tolerating it having already been taken.

    Two scanners can legitimately run at once (a worker and the API, say).
    `queue.submit` already dedupes by content hash, so the loser of that race
    just finds the file gone — which is success, not an error.
    """
    if not path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(path), str(destination))
    except (FileNotFoundError, shutil.Error) as exc:
        log.debug("intake: could not move %s: %s", path.name, exc)
        return None
    return destination


def submit_stable(
    session: Session,
    settings: Settings,
    scanner: IntakeScanner,
    *,
    max_attempts: int = 3,
) -> SubmitReport:
    """Queue every stable intake file, moving it out of the watched directory.

    Files are moved rather than left in place so that a single scan can never
    enqueue the same recording twice, and so the intake folder reflects only
    what is still waiting.
    """
    queued: list[str] = []
    duplicates: list[str] = []
    errors: list[tuple[str, str]] = []

    for path in scanner.stable_files():
        name = path.name
        try:
            info = media.inspect(path)
        except media.MediaError as exc:
            errors.append((name, str(exc)))
            scanner.forget(path)
            continue

        job, created = queue.submit(
            session,
            sha256=info.sha256,
            name=name,
            path=str(path),
            size_bytes=info.bytes,
            duration_s=info.duration_s,
            max_attempts=max_attempts,
        )

        if created:
            # Move the original into the archive, never into `work`. Because
            # this is a move, the archive briefly holds the only copy of the
            # recording, so it must not live anywhere that gets cleaned out.
            # `work` is scratch for derived files (the normalized WAV) and is
            # emptied freely; putting sources there once cost a real recording.
            #
            # The name is preserved, with the content hash appended only on
            # collision, so the archive stays browsable by a human.
            final = settings.archive / name
            if final.exists():
                final = final.with_name(
                    f"{path.stem}-{info.sha256[:8]}{path.suffix}"
                )
            if moved := _move_into(path, final):
                job.source_path = str(moved)
            queued.append(name)
        else:
            target = settings.intake / DUPLICATES_DIRNAME / name
            if target.exists():
                target = target.with_name(f"{path.stem}-{info.sha256[:8]}{path.suffix}")
            _move_into(path, target)
            queue.log(
                session,
                job.id,
                f"duplicate of an existing job; moved {name} to "
                f"{DUPLICATES_DIRNAME}/",
                level="warning",
            )
            duplicates.append(name)

        scanner.forget(path)

    return SubmitReport(queued=queued, duplicates=duplicates, errors=errors)


def watch(
    settings: Settings,
    *,
    stop: object | None = None,
    once: bool = False,
) -> SubmitReport:
    """Scan the intake directory on an interval, queueing what settles.

    `watchfiles` is used only to shorten the wait between a file appearing and
    the next scan; the scan itself remains the source of truth, so a missed or
    unsupported filesystem event costs latency and never correctness.

    Pass `stop` as a `threading.Event` to shut down cleanly.
    """
    scanner = IntakeScanner(settings)
    settings.ensure_dirs()
    totals = SubmitReport(queued=[], duplicates=[], errors=[])

    while True:
        with session_scope(settings) as session:
            report = submit_stable(session, settings, scanner)
        if report.total:
            log.info(
                "intake: %d queued, %d duplicate, %d error",
                len(report.queued),
                len(report.duplicates),
                len(report.errors),
            )
        totals = SubmitReport(
            queued=totals.queued + report.queued,
            duplicates=totals.duplicates + report.duplicates,
            errors=totals.errors + report.errors,
        )
        if once:
            return totals
        if stop is not None and stop.wait(settings.scan_interval_s):
            return totals
        if stop is None:
            time.sleep(settings.scan_interval_s)
