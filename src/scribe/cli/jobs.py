"""Queue inspection and manual control."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table
from sqlalchemy import desc, select

from scribe import intake, media, queue
from scribe.cli._common import console, err, resolve_job, short, state_badge
from scribe.config import load_settings
from scribe.db import session_scope
from scribe.models import Job, JobEvent, JobState
from scribe.render.timecode import hhmmss


def submit_cmd(
    sources: Annotated[list[Path], typer.Argument(help="Files to enqueue.")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-run even if already transcribed.")
    ] = False,
) -> None:
    """Add files to the queue. A worker picks them up."""
    settings = load_settings()
    settings.ensure_dirs()

    with session_scope(settings) as session:
        for source in sources:
            if not source.is_file():
                err.print(f"[red]skipped[/] {source} (not a file)")
                continue
            try:
                info = media.inspect(source)
            except media.MediaError as exc:
                err.print(f"[red]skipped[/] {source.name}: {exc}")
                continue

            job, created = queue.submit(
                session,
                sha256=info.sha256,
                name=source.name,
                path=str(source.resolve()),
                size_bytes=info.bytes,
                duration_s=info.duration_s,
                force=force,
            )
            if created:
                console.print(
                    f"[green]queued[/]  {short(job.id)}  {source.name} "
                    f"[dim]({hhmmss(info.duration_s)})[/]"
                )
            elif force:
                console.print(f"[yellow]requeued[/] {short(job.id)}  {source.name}")
            else:
                console.print(
                    f"[dim]already known[/] {short(job.id)}  {source.name} "
                    f"[dim]({job.state}; use --force to re-run)[/]"
                )


def queue_cmd(
    state: Annotated[
        str | None, typer.Option("--state", "-s", help="Filter by state.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    watch_mode: Annotated[
        bool, typer.Option("--watch", "-w", help="Refresh until interrupted.")
    ] = False,
) -> None:
    """List jobs, newest first."""
    settings = load_settings()

    def render() -> Table:
        table = Table(box=None, header_style="dim")
        for column in ("id", "state", "stage", "prog", "duration", "source", "note"):
            table.add_column(column)
        with session_scope(settings) as session:
            stmt = select(Job).order_by(desc(Job.created_at)).limit(limit)
            if state:
                stmt = stmt.where(Job.state == state)
            for job in session.scalars(stmt):
                note = ""
                if job.state == JobState.DEAD and job.last_error:
                    note = f"[red]{job.last_error[:40]}[/]"
                elif job.attempts > 1:
                    note = f"[yellow]attempt {job.attempts}/{job.max_attempts}[/]"
                elif job.output_dir:
                    note = f"[dim]{Path(job.output_dir).name}[/]"
                table.add_row(
                    short(job.id),
                    state_badge(job.state),
                    job.stage or "",
                    f"{job.progress:.0%}" if 0 < job.progress < 1 else "",
                    hhmmss(job.duration_s) if job.duration_s else "",
                    job.source_name[:36],
                    note,
                )
            summary = queue.counts(session)
        table.caption = (
            "  ".join(f"{k}: {v}" for k, v in sorted(summary.items())) or "no jobs"
        )
        table.caption_justify = "left"
        return table

    if not watch_mode:
        console.print(render())
        return

    from rich.live import Live

    try:
        with Live(render(), console=console, refresh_per_second=1) as live:
            while True:
                live.update(render())
                time.sleep(2)
    except KeyboardInterrupt:
        pass


def logs_cmd(
    reference: Annotated[str, typer.Argument(help="Job id, or an id prefix.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 60,
) -> None:
    """Show a job's event log."""
    settings = load_settings()
    with session_scope(settings) as session:
        job = resolve_job(session, reference)
        console.print(
            f"[bold]{short(job.id)}[/] {job.source_name}  {state_badge(job.state)}"
        )
        if job.output_dir:
            console.print(f"[dim]output:[/] {job.output_dir}")
        if job.last_error:
            console.print(f"[red]last error:[/] {job.last_error}")
        console.print()

        events = session.scalars(
            select(JobEvent)
            .where(JobEvent.job_id == job.id)
            .order_by(desc(JobEvent.ts))
            .limit(limit)
        ).all()
        style = {"error": "red", "warning": "yellow"}
        for event in reversed(events):
            colour = style.get(event.level, "dim")
            stamp = event.ts.strftime("%H:%M:%S")
            console.print(f"[dim]{stamp}[/] [{colour}]{event.message}[/]")


def retry_cmd(
    reference: Annotated[str, typer.Argument(help="Job id, or an id prefix.")],
) -> None:
    """Requeue a dead job."""
    settings = load_settings()
    with session_scope(settings) as session:
        job = resolve_job(session, reference)
        if job.state not in (JobState.DEAD, JobState.DONE):
            err.print(
                f"[yellow]{short(job.id)} is {job.state}[/]; nothing to retry."
            )
            raise typer.Exit(1)
        queue.retry(session, job)
        console.print(f"[green]requeued[/] {short(job.id)}  {job.source_name}")


def scan_cmd() -> None:
    """Scan the intake folder once and queue whatever has settled."""
    settings = load_settings()
    settings.ensure_dirs()
    report = intake.watch(settings, once=True)

    for name in report.queued:
        console.print(f"[green]queued[/]     {name}")
    for name in report.duplicates:
        console.print(f"[dim]duplicate  {name}[/]")
    for name, reason in report.errors:
        err.print(f"[red]error[/]      {name}: {reason}")
    if not report.total:
        console.print(
            f"[dim]Nothing new in {settings.intake}.\n"
            "A file must be unchanged across two scans before it is picked up.[/]"
        )
