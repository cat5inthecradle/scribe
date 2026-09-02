"""The `scribe tune` command: sweep diarization settings on one recording."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from scribe.cli._common import console, err
from scribe.config import load_settings
from scribe.tune import DEFAULT_THRESHOLDS, FRAGMENT_WORDS, disagreements, sweep


def tune_cmd(
    source: Annotated[Path, typer.Argument(help="A recording to tune against.")],
    thresholds: Annotated[
        str | None,
        typer.Option(
            "--thresholds",
            "-t",
            help="Comma-separated clustering thresholds. Lower splits more eagerly.",
        ),
    ] = None,
    min_duration_off: Annotated[
        float | None,
        typer.Option("--min-duration-off", help="Bridge silences shorter than this."),
    ] = None,
    speakers: Annotated[
        int | None, typer.Option("--speakers", "-s", help="Pin the speaker count.")
    ] = None,
    show: Annotated[
        float | None,
        typer.Option("--show", help="Print the full transcript for one threshold."),
    ] = None,
) -> None:
    """Compare diarization settings on one file.

    Transcribes once and re-runs only diarization, so this costs about one
    extra diarization per setting rather than a full pipeline each.
    """
    settings = load_settings()
    if speakers is not None:
        settings.diarize.num_speakers = speakers

    if not source.is_file():
        err.print(f"[red]No such file:[/] {source}")
        raise typer.Exit(2)

    try:
        values = (
            tuple(float(v) for v in thresholds.split(","))
            if thresholds
            else DEFAULT_THRESHOLDS
        )
    except ValueError as exc:
        err.print(f"[red]Could not parse --thresholds:[/] {thresholds}")
        raise typer.Exit(2) from exc

    console.print(
        f"[dim]Transcribing once, then diarizing at {len(values)} settings…[/]"
    )
    try:
        words, rows = sweep(
            source,
            settings,
            thresholds=values,
            min_duration_off=min_duration_off,
            on_step=lambda t: console.print(f"[dim]  threshold {t}[/]"),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        err.print(f"[red]Failed:[/] {exc}")
        raise typer.Exit(1) from exc

    if not rows:
        err.print("[red]Nothing to compare.[/]")
        raise typer.Exit(1)

    default_row = min(rows, key=lambda r: abs(r.threshold - 0.6))
    fewest_fragments = min(r.fragments for r in rows)

    # Metrics and turn shapes are printed separately: a shape string is far
    # wider than any numeric column, and Rich drops columns rather than
    # wrapping when a table cannot fit the terminal.
    table = Table(box=None, header_style="dim", title_justify="left")
    table.title = f"{source.name} — {len(words)} words"
    for column in (
        "threshold",
        "speakers",
        "turns",
        f"<{FRAGMENT_WORDS}w",
        "median",
        "time",
    ):
        table.add_column(column, justify="right")

    for row in rows:
        marker = " (default)" if row is default_row else ""
        best = row.fragments == fewest_fragments
        table.add_row(
            f"{row.threshold:.2f}{marker}",
            str(row.speakers),
            str(row.turn_count),
            f"[green]{row.fragments}[/]" if best else str(row.fragments),
            str(row.median_turn_words),
            f"{row.seconds:.0f}s",
        )
    console.print(table)

    console.print("\n[dim]Turn shape (speaker·words, in order):[/]")
    for row in rows:
        console.print(f"  [dim]{row.threshold:.2f}[/]  {row.shape(limit=18)}")

    differing = disagreements(rows, words)
    console.print(
        f"\n[dim]Settings disagree about the speaker boundary at "
        f"{len(differing)} of {len(words)} words "
        f"({100 * len(differing) / max(len(words), 1):.1f}%).[/]"
    )
    console.print(
        f"[dim]`<{FRAGMENT_WORDS}w` counts turns too short to be real speech — "
        "the flapping artifact. Lower is usually better.[/]"
    )

    if show is not None:
        chosen = min(rows, key=lambda r: abs(r.threshold - show))
        console.print(f"\n[bold]Transcript at threshold {chosen.threshold:.2f}[/]\n")
        order: dict[str, int] = {}
        for turn in chosen.turns:
            n = order.setdefault(turn.speaker, len(order) + 1)
            console.print(f"[cyan]Speaker {n}[/] [dim]{turn.start:7.2f}s[/]  {turn.text}")

    if len(differing) == 0 and len(rows) > 1:
        console.print(
            "[yellow]Every setting agreed.[/] This recording is unambiguous, so "
            "the threshold makes no difference here — tune against audio that "
            "actually gives the diarizer trouble (overlap, similar voices)."
        )

    console.print("\n[dim]To keep a setting, add to scribe.toml:[/]")
    console.print("  [diarize]", markup=False, style="dim")
    console.print("  clustering_threshold = <value>", markup=False, style="dim")
