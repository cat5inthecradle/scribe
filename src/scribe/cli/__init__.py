"""Command line interface.

Commands live in focused modules and are registered here, so the full command
surface is visible in one place and stays flat (`scribe run`, `scribe submit`)
rather than nesting behind group names.
"""

from __future__ import annotations

import sys

import typer

from scribe.cli import admin, jobs, service, transcribe, tune

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local audio transcription with speaker separation.",
)

# One-shot transcription, no database.
app.command("run")(transcribe.run_cmd)
app.command("rerender")(transcribe.rerender_cmd)
app.command("tune")(tune.tune_cmd)

# Queue control.
app.command("submit")(jobs.submit_cmd)
app.command("queue")(jobs.queue_cmd)
app.command("logs")(jobs.logs_cmd)
app.command("retry")(jobs.retry_cmd)
app.command("scan")(jobs.scan_cmd)

# Long-running processes.
app.command("worker")(service.worker_cmd)
app.command("watch")(service.watch_cmd)
app.command("dev")(service.dev_cmd)

# Diagnostics and migrations.
app.command("doctor")(admin.doctor_cmd)
app.add_typer(admin.db_app, name="db")

__all__ = ["app", "main"]


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
