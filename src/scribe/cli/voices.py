"""Speaker enrollment commands.

`identify` is the primary workflow: after a recording is transcribed, confirm
who each speaker was. It reads the embeddings saved beside the transcript, so
naming people costs no inference and works even if the audio has since been
archived away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from scribe import speakers as sidecar
from scribe import voices
from scribe.cli._common import console, err
from scribe.config import load_settings
from scribe.db import session_scope
from scribe.pipeline import read_embeddings, rerender
from scribe.render.timecode import hhmmss
from scribe.types import Transcript

SKIP = "-"


def voices_cmd(
    forget: Annotated[
        str | None, typer.Option("--forget", help="Remove every sample for a name.")
    ] = None,
) -> None:
    """List enrolled speakers."""
    settings = load_settings()
    with session_scope(settings) as session:
        if forget:
            removed = voices.forget(session, forget)
            if removed:
                console.print(f"[yellow]Removed[/] {removed} sample(s) for {forget!r}")
            else:
                err.print(f"[red]No enrolled speaker named[/] {forget!r}")
                raise typer.Exit(1)
            return

        people = voices.enrolled(session)

    if not people:
        console.print(
            "[dim]No speakers enrolled yet.\n"
            "Transcribe a recording, then: scribe identify <output dir>[/]"
        )
        return

    table = Table(box=None, header_style="dim")
    table.add_column("speaker")
    table.add_column("samples", justify="right")
    table.add_column("speech", justify="right")
    for person in people:
        table.add_row(person.name, str(person.samples), hhmmss(person.total_speech_s))
    console.print(table)
    console.print(
        f"\n[dim]match threshold: "
        f"{load_settings().diarize.embedding_match_threshold}[/]"
    )


def identify_cmd(
    out_dir: Annotated[Path, typer.Argument(help="A transcript output directory.")],
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Override the auto-match similarity."),
    ] = None,
    reidentify: Annotated[
        bool,
        typer.Option("--all", help="Ask about speakers that already have names too."),
    ] = False,
) -> None:
    """Name the speakers in a transcript, enrolling each confirmation.

    Runs no inference: the voice embeddings were saved when the recording was
    transcribed.
    """
    settings = load_settings()
    if threshold is not None:
        settings.diarize.embedding_match_threshold = threshold

    transcript_path = out_dir / "transcript.json"
    if not transcript_path.is_file():
        err.print(f"[red]No transcript.json in[/] {out_dir}")
        raise typer.Exit(2)

    transcript = Transcript.from_json(transcript_path.read_text(encoding="utf-8"))
    embeddings = read_embeddings(out_dir)
    if not embeddings:
        err.print(
            "[red]No embeddings.json in this output directory.[/]\n"
            "[dim]It predates speaker enrollment — re-run the recording to "
            "generate one.[/]"
        )
        raise typer.Exit(2)

    existing = sidecar.read(out_dir)
    console.print(f"[bold]{transcript.source.filename}[/]  {out_dir}\n")

    resolved: dict[str, str] = {}
    with session_scope(settings) as session:
        for speaker in transcript.speakers:
            current = existing.get(speaker.id) or speaker.name
            if current and not reidentify:
                console.print(f"[green]{speaker.label}[/] already named {current!r}")
                continue

            vector = embeddings.get(speaker.id)
            _show_speaker(transcript, speaker, current)
            if vector:
                for match in voices.rank(session, vector, limit=3):
                    console.print(
                        f"    [dim]closest:[/] {match.name} "
                        f"[dim]({match.similarity:.2f}, "
                        f"{match.sample_count} sample(s))[/]"
                    )

            answer = typer.prompt(
                f"    Who is {speaker.label}? "
                f"(name, {SKIP!r} to leave anonymous)",
                default=current or SKIP,
                show_default=bool(current),
            ).strip()
            console.print()

            if not answer or answer == SKIP:
                continue

            resolved[speaker.id] = answer
            if vector:
                # Every confirmation becomes a new sample, so recognition
                # improves with use rather than depending on one enrollment.
                voices.enroll(
                    session,
                    answer,
                    vector,
                    source_name=transcript.source.filename,
                    speech_s=speaker.speech_s,
                )

    if not resolved:
        console.print("[dim]Nothing changed.[/]")
        return

    # Persist names to the sidecar, then re-render from the canonical JSON.
    merged = {**existing, **resolved}
    sidecar.write(sidecar.apply(transcript, merged), out_dir)
    result = rerender(out_dir)
    console.print(
        f"[green]Named {len(resolved)} speaker(s)[/] and enrolled their voices."
    )
    console.print(
        "Speakers: " + ", ".join(s.display for s in result.transcript.speakers)
    )


def _show_speaker(transcript: Transcript, speaker, current: str | None) -> None:
    """Print a speaker's share and a sample line, to jog recognition."""
    share = ""
    total = sum(s.speech_s for s in transcript.speakers)
    if total:
        share = f", {100 * speaker.speech_s / total:.0f}% of speech"
    console.print(
        f"[cyan]{speaker.label}[/] [dim]({hhmmss(speaker.speech_s)}{share})[/]"
    )
    # The longest turn is the most recognisable thing they said.
    turns = [t for t in transcript.turns if t.speaker == speaker.id]
    if turns:
        longest = max(turns, key=lambda t: len(t.words))
        snippet = longest.text[:150] + ("…" if len(longest.text) > 150 else "")
        console.print(f'    [dim]"{snippet}"[/]')


def enroll_cmd(
    name: Annotated[str, typer.Argument(help="The person's name.")],
    source: Annotated[
        Path, typer.Argument(help="A recording of them, ideally speaking alone.")
    ],
    speaker: Annotated[
        str | None,
        typer.Option("--speaker", help="Which diarized label to enroll, if several."),
    ] = None,
) -> None:
    """Enroll a voice directly from an audio file.

    Prefer `scribe identify` on an already-transcribed recording — it needs no
    inference. This is for enrolling from a clean solo sample.
    """
    from scribe import media
    from scribe.diarize.base import load_backend

    settings = load_settings()
    settings.ensure_dirs()
    if not source.is_file():
        err.print(f"[red]No such file:[/] {source}")
        raise typer.Exit(2)

    console.print(f"[dim]Diarizing {source.name}…[/]")
    info = media.inspect(source)
    wav = settings.work / f"enroll-{info.sha256[:12]}.wav"
    media.normalize(source, wav)
    try:
        diarizer = load_backend(
            "pyannote",
            model=settings.diarize.model,
            token=settings.resolved_hf_token(),
            device=settings.diarize.device,
            clustering_threshold=settings.diarize.clustering_threshold,
            min_duration_off=settings.diarize.min_duration_off,
        )
        diarization = diarizer.diarize(wav)
    finally:
        wav.unlink(missing_ok=True)

    if not diarization.embeddings:
        err.print("[red]No voice embeddings were produced for this file.[/]")
        raise typer.Exit(1)

    talk = {
        label: sum(s.duration for s in diarization.exclusive if s.speaker == label)
        for label in diarization.embeddings
    }

    if speaker is None:
        if len(diarization.embeddings) == 1:
            speaker = next(iter(diarization.embeddings))
        else:
            console.print(
                f"[yellow]{len(diarization.embeddings)} speakers detected.[/] "
                "Pick the one to enroll:"
            )
            for label in sorted(talk, key=lambda k: -talk[k]):
                console.print(f"  {label}  [dim]{hhmmss(talk[label])} of speech[/]")
            speaker = typer.prompt("  Label").strip()

    if speaker not in diarization.embeddings:
        err.print(f"[red]No such speaker in this file:[/] {speaker}")
        raise typer.Exit(2)

    with session_scope(settings) as session:
        voices.enroll(
            session,
            name,
            diarization.embeddings[speaker],
            source_name=source.name,
            speech_s=talk.get(speaker),
        )
    console.print(
        f"[green]Enrolled[/] {name} [dim]from {speaker} "
        f"({hhmmss(talk.get(speaker, 0))} of speech)[/]"
    )
