"""Environment checks and database administration."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from scribe import media
from scribe.cli._common import console
from scribe.config import load_settings
from scribe.db import check_connection

db_app = typer.Typer(no_args_is_help=True, help="Database administration.")


def doctor_cmd() -> None:
    """Check the prerequisites. Run this first."""
    settings = load_settings()
    ok = True

    table = Table(show_header=True, box=None)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", style="dim")

    def row(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        table.add_row(name, "[green]ok[/]" if good else "[red]missing[/]", detail)

    try:
        media.require_ffmpeg()
        row("ffmpeg", True)
    except media.MediaError as exc:
        row("ffmpeg", False, str(exc))

    token = settings.resolved_hf_token()
    row(
        "HF token",
        bool(token),
        "found" if token else "run: hf auth login   (needed for pyannote weights)",
    )

    try:
        from scribe.diarize.pyannote_ import resolve_device

        row("torch", True, f"device: {resolve_device(settings.diarize.device)}")
    except Exception as exc:  # noqa: BLE001
        row("torch", False, str(exc))

    if settings.asr.backend == "mlx":
        try:
            import mlx.core as mx  # noqa: F401

            row("mlx", True, "Metal available")
        except Exception as exc:  # noqa: BLE001
            row("mlx", False, str(exc))

    reachable, detail = check_connection(settings)
    row(
        "database",
        reachable,
        detail if reachable else f"{detail}  (try: docker compose up -d)",
    )

    console.print(table)
    console.print(f"\n[dim]intake[/] {settings.intake}\n[dim]out   [/] {settings.out}")
    if not ok:
        console.print(
            "\n[yellow]If the pyannote licence is the problem, accept it at[/] "
            "https://huggingface.co/pyannote/speaker-diarization-community-1"
        )
        raise typer.Exit(1)


@db_app.command("upgrade")
def db_upgrade(
    revision: str = typer.Argument("head", help="Target revision."),
) -> None:
    """Apply database migrations."""
    from alembic.config import Config

    from alembic import command

    root = Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    command.upgrade(config, revision)
    console.print(f"[green]Database at revision[/] {revision}")


@db_app.command("current")
def db_current() -> None:
    """Show the applied migration revision."""
    from alembic.config import Config

    from alembic import command

    root = Path(__file__).resolve().parents[3]
    command.current(Config(str(root / "alembic.ini")), verbose=True)
