"""The worker loop.

Claims one job at a time and runs the Phase 1 pipeline against it. Concurrency
within a worker would be counterproductive: ASR and diarization are already
CPU/GPU-bound, so two jobs on one machine finish no sooner and each reports
progress more slowly. Scale by running more worker processes instead.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from sqlalchemy import text

from scribe import pipeline, queue
from scribe.config import Settings
from scribe.db import session_scope
from scribe.models import Job

log = logging.getLogger(__name__)

#: Liveness pings per staleness window. Several, so one slow tick is harmless.
HEARTBEAT_DIVISOR = 6

#: Don't write a progress update for changes smaller than this.
PROGRESS_EPSILON = 0.05


class _Heartbeat:
    """Background liveness pings for the job currently being processed.

    Deliberately independent of pipeline progress: a single-pass ASR run can go
    minutes without emitting a callback, and a worker that is alive but quiet
    must not have its job reclaimed. Uses its own session per tick because
    SQLAlchemy sessions are not thread-safe.
    """

    def __init__(self, settings: Settings, job_id, interval_s: float) -> None:
        self._settings = settings
        self._job_id = job_id
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Heartbeat:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                with session_scope(self._settings) as session:
                    session.execute(
                        text(
                            "UPDATE jobs SET heartbeat_at = now(), "
                            "updated_at = now() WHERE id = :id"
                        ),
                        {"id": self._job_id},
                    )
            except Exception:  # noqa: BLE001 - a missed beat must not kill the job
                log.warning("heartbeat failed", exc_info=True)


class Worker:
    def __init__(
        self,
        settings: Settings,
        *,
        worker_id: str | None = None,
        once: bool = False,
    ) -> None:
        self.settings = settings
        self.worker_id = worker_id or settings.worker_id or queue.default_worker_id()
        self.once = once
        self._stop = threading.Event()
        self.processed = 0
        self.failed = 0

    def request_stop(self) -> None:
        """Stop claiming new work.

        The job in flight is left to finish. If the process is killed before it
        does, the heartbeat lapses and another worker reclaims it — which is
        exactly what the reclaim path exists for, so there is nothing to unwind.
        """
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run(self) -> int:
        """Process jobs until stopped (or until the queue empties, if `once`)."""
        log.info("worker %s starting", self.worker_id)
        while not self.stopping:
            job_id = self._claim_one()
            if job_id is None:
                if self.once:
                    break
                self._stop.wait(self.settings.poll_interval_s)
                continue
            self._process(job_id)
        log.info(
            "worker %s stopped (%d done, %d failed)",
            self.worker_id,
            self.processed,
            self.failed,
        )
        return self.processed

    def _claim_one(self):
        """Reclaim abandoned work, then take a job. Returns a job id or None."""
        with session_scope(self.settings) as session:
            reclaimed = queue.reclaim_stale(session, self.settings.stale_after_s)
            if reclaimed.total:
                log.warning(
                    "reclaimed %d stale job(s): %d requeued, %d dead",
                    reclaimed.total,
                    reclaimed.requeued,
                    reclaimed.died,
                )
            job = queue.claim(session, self.worker_id)
            # Return the id, not the ORM object: the session closes here and the
            # pipeline runs long enough that a detached instance would be a trap.
            return job.id if job else None

    def _process(self, job_id) -> None:
        with session_scope(self.settings) as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            source = Path(job.source_path)
            name = job.source_name
            attempt = f"{job.attempts}/{job.max_attempts}"

        log.info("job %s attempt %s: %s", str(job_id)[:8], attempt, name)
        interval = max(1.0, self.settings.stale_after_s / HEARTBEAT_DIVISOR)

        try:
            with _Heartbeat(self.settings, job_id, interval):
                result = pipeline.run(
                    source,
                    self.settings,
                    on_progress=self._reporter(job_id),
                    # The source was parked at work/<job id>/source.ext by the
                    # intake watcher; carry the original name for the output.
                    display_name=name,
                )
        except Exception as exc:  # noqa: BLE001 - recorded on the job
            self.failed += 1
            detail = f"{type(exc).__name__}: {exc}"
            log.error("job %s failed: %s", str(job_id)[:8], detail)
            log.debug("%s", traceback.format_exc())
            with session_scope(self.settings) as session:
                job = session.get(Job, job_id)
                if job is not None:
                    queue.fail(session, job, detail)
            return

        self.processed += 1
        transcript = result.transcript
        with session_scope(self.settings) as session:
            job = session.get(Job, job_id)
            if job is not None:
                queue.complete(
                    session,
                    job,
                    output_dir=str(result.out_dir),
                    duration_s=transcript.source.duration_s,
                    asr_backend=transcript.pipeline.asr.backend,
                    asr_model=transcript.pipeline.asr.model,
                    diar_model=transcript.pipeline.diarization.model,
                )
                for stage, seconds in result.durations.items():
                    queue.log(
                        session, job_id, f"{stage} took {seconds:.1f}s", stage=stage
                    )
        log.info(
            "job %s done: %d speakers, %d turns -> %s",
            str(job_id)[:8],
            len(transcript.speakers),
            len(transcript.turns),
            result.out_dir,
        )

    def _reporter(self, job_id):
        """Throttled progress writer.

        Every ASR chunk would otherwise be an UPDATE. Writing only on a stage
        change or a meaningful progress jump keeps the log readable and the
        write volume trivial.
        """
        state = {"stage": None, "progress": -1.0}

        def report(stage: str, fraction: float) -> None:
            changed_stage = stage != state["stage"]
            moved = fraction - float(state["progress"]) >= PROGRESS_EPSILON
            if not (changed_stage or moved or fraction >= 1.0):
                return
            state["stage"] = stage
            state["progress"] = fraction

            overall = _overall_progress(stage, fraction)
            try:
                with session_scope(self.settings) as session:
                    job = session.get(Job, job_id)
                    if job is None:
                        return
                    queue.heartbeat(session, job, stage=stage, progress=overall)
                    if changed_stage:
                        queue.log(session, job_id, f"stage: {stage}", stage=stage)
            except Exception:  # noqa: BLE001 - progress is not worth failing over
                log.debug("progress update failed", exc_info=True)

        return report


def _overall_progress(stage: str, fraction: float) -> float:
    """Map (stage, fraction) onto a single 0..1 for the whole pipeline.

    Stages are weighted by observed share of wall time rather than counted
    evenly: diarization dominates (roughly 75% on real recordings), so an even
    split would park the bar at 50% for most of the run.
    """
    weights = {
        "inspect": 0.02,
        "normalize": 0.03,
        "transcribe": 0.20,
        "diarize": 0.70,
        "merge": 0.02,
        "render": 0.03,
    }
    order = list(weights)
    if stage not in weights:
        return 0.0
    before = sum(weights[s] for s in order[: order.index(stage)])
    return min(1.0, before + weights[stage] * max(0.0, min(1.0, fraction)))
