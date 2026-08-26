"""Command line interface.

`run` is deliberately database-free: it is the shortest path from a file to a
transcript, which makes it the right tool for judging output quality and for
debugging a backend without any service infrastructure in the way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from scribe import media
from scribe.config import load_settings
from scribe.pipeline import STAGES, rerender, run
from scribe.render.timecode import hhmmss

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local audio transcription with speaker separation.",
)
console = Console()
err = Console(stderr=True)


@app.command("run")
def run_cmd(
    source: Annotated[Path, typer.Argument(help="Audio or video file.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory.")] = None,
    backend: Annotated[str | None, typer.Option("--backend", "-b", help="mlx | onnx")] = None,
    speakers: Annotated[
        int | None,
        typer.Option("--speakers", "-s", help="Exact speaker count, when known."),
    ] = None,
    min_speakers: Annotated[int | None, typer.Option("--min-speakers")] = None,
    max_speakers: Annotated[int | None, typer.Option("--max-speakers")] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="auto | cpu | mps | cuda")
    ] = None,
    keep_wav: Annotated[
        bool, typer.Option("--keep-wav", help="Retain the normalized WAV.")
    ] = False,
) -> None:
    """Transcribe one file end to end. No database involved."""
    settings = load_settings()
    if backend:
        settings.asr.backend = backend  # type: ignore[assignment]
    if device:
        settings.diarize.device = device  # type: ignore[assignment]
    for name, value in (
        ("num_speakers", speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None:
            setattr(settings.diarize, name, value)
    settings.ensure_dirs()

    if not source.is_file():
        err.print(f"[red]No such file:[/] {source}")
        raise typer.Exit(2)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[stage]:<11}"),
        BarColumn(bar_width=32),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("work", total=len(STAGES) * 100, stage="starting")

        def on_progress(stage: str, fraction: float) -> None:
            index = STAGES.index(stage) if stage in STAGES else 0
            progress.update(
                task, completed=index * 100 + fraction * 100, stage=stage
            )

        try:
            result = run(
                source, settings, out_dir=out, on_progress=on_progress, keep_wav=keep_wav
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            progress.stop()
            err.print(f"\n[red]Failed:[/] {exc}")
            raise typer.Exit(1) from exc

        progress.update(task, completed=len(STAGES) * 100, stage="done")

    _report(result)


def _report(result) -> None:
    t = result.transcript
    console.print()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("source", t.source.filename)
    table.add_row("duration", hhmmss(t.source.duration_s))
    table.add_row("speakers", str(len(t.speakers)))
    table.add_row("turns", str(len(t.turns)))
    table.add_row("words", str(sum(len(x.words) for x in t.turns)))

    total = sum(result.durations.values())
    if total and t.source.duration_s:
        speed = t.source.duration_s / total
        table.add_row("time", f"{total:.1f}s  ({speed:.1f}x realtime)")
    stages = "  ".join(f"{k} {v:.1f}s" for k, v in result.durations.items())
    if stages:
        table.add_row("stages", stages)
    table.add_row("output", str(result.out_dir))
    console.print(table)

    if t.speakers:
        console.print()
        breakdown = Table(title="Speaking time", title_justify="left", box=None)
        breakdown.add_column("speaker")
        breakdown.add_column("time", justify="right")
        breakdown.add_column("share", justify="right")
        total_speech = sum(s.speech_s for s in t.speakers) or 1.0
        for s in t.speakers:
            breakdown.add_row(
                s.display, hhmmss(s.speech_s), f"{100 * s.speech_s / total_speech:.0f}%"
            )
        console.print(breakdown)

    console.print(
        f"\n[dim]Rename speakers in[/] {result.out_dir / 'speakers.yaml'}"
        f"[dim], then:[/] scribe rerender {result.out_dir}"
    )


@app.command("rerender")
def rerender_cmd(
    out_dir: Annotated[Path, typer.Argument(help="An existing output directory.")],
) -> None:
    """Rewrite outputs from transcript.json, applying speakers.yaml. No inference."""
    try:
        result = rerender(out_dir)
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc

    names = [s.display for s in result.transcript.speakers]
    console.print(f"Re-rendered {len(result.written)} files in {out_dir}")
    console.print(f"Speakers: {', '.join(names)}")


@app.command("doctor")
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

    console.print(table)
    console.print(
        f"\n[dim]intake[/] {settings.intake}\n[dim]out   [/] {settings.out}"
    )
    if not ok:
        console.print(
            "\n[yellow]Accept the model licence at[/] "
            "https://huggingface.co/pyannote/speaker-diarization-community-1"
        )
        raise typer.Exit(1)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
