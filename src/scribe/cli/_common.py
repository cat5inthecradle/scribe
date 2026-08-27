"""Shared CLI helpers."""

from __future__ import annotations

import uuid

import typer
from rich.console import Console
from sqlalchemy import String, cast, select
from sqlalchemy.orm import Session

from scribe.models import Job

console = Console()
err = Console(stderr=True)

STATE_STYLE = {
    "pending": "yellow",
    "claimed": "cyan",
    "running": "bold cyan",
    "done": "green",
    "dead": "red",
}


def state_badge(state: str) -> str:
    return f"[{STATE_STYLE.get(state, 'white')}]{state}[/]"


def short(value: uuid.UUID | str) -> str:
    return str(value)[:8]


def resolve_job(session: Session, reference: str) -> Job:
    """Look up a job by full uuid or by any unambiguous id prefix.

    Typing eight characters is far friendlier than a full uuid, and the CLI
    already prints ids truncated to eight — so accepting that form back is the
    least surprising behaviour.
    """
    try:
        parsed = uuid.UUID(reference)
    except ValueError:
        pass
    else:
        job = session.get(Job, parsed)
        if job is None:
            err.print(f"[red]No job with id[/] {reference}")
            raise typer.Exit(2)
        return job

    matches = list(
        session.scalars(
            select(Job).where(cast(Job.id, String).like(f"{reference}%")).limit(6)
        )
    )
    if not matches:
        err.print(f"[red]No job matching[/] {reference!r}")
        raise typer.Exit(2)
    if len(matches) > 1:
        err.print(
            f"[red]Ambiguous id[/] {reference!r} matches "
            + ", ".join(short(j.id) for j in matches[:5])
        )
        raise typer.Exit(2)
    return matches[0]
