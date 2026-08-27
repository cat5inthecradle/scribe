"""Long-running processes: the worker, the intake watcher, and both together."""

from __future__ import annotations

import logging
import signal
import threading
from typing import Annotated

import typer

from scribe import intake
from scribe.cli._common import console
from scribe.config import load_settings
from scribe.worker import Worker


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at DEBUG and drown out anything useful.
    for noisy in ("urllib3", "httpx", "filelock", "speechbrain", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def worker_cmd(
    once: Annotated[
        bool, typer.Option("--once", help="Drain the queue, then exit.")
    ] = False,
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="mlx | onnx")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Process queued jobs.

    Run this on the host (not in a container) to get Metal acceleration — a
    container on macOS cannot reach it.
    """
    _setup_logging(verbose)
    settings = load_settings()
    if backend:
        settings.asr.backend = backend  # type: ignore[assignment]
    settings.ensure_dirs()

    worker = Worker(settings, once=once)
    _on_signal(worker.request_stop, "worker")
    worker.run()


def watch_cmd(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Watch the intake folder and queue files as they settle.

    Run exactly one of these. It only enqueues; workers do the transcribing.
    """
    _setup_logging(verbose)
    settings = load_settings()
    settings.ensure_dirs()
    console.print(f"[dim]Watching[/] {settings.intake}  [dim](Ctrl-C to stop)[/]")

    stop = threading.Event()
    _on_signal(stop.set, "watcher")
    intake.watch(settings, stop=stop)


def dev_cmd(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the intake watcher and one worker together. For local use.

    In Kubernetes these are separate deployments so workers can scale
    independently; a single process is simply more convenient on a laptop.
    """
    _setup_logging(verbose)
    settings = load_settings()
    settings.ensure_dirs()
    console.print(
        f"[dim]Watching[/] {settings.intake}\n"
        f"[dim]Outputs [/] {settings.out}  [dim](Ctrl-C to stop)[/]\n"
    )

    stop = threading.Event()
    worker = Worker(settings)

    def shutdown() -> None:
        stop.set()
        worker.request_stop()

    _on_signal(shutdown, "dev")

    watcher = threading.Thread(
        target=intake.watch, args=(settings,), kwargs={"stop": stop}, daemon=True
    )
    watcher.start()
    worker.run()
    watcher.join(timeout=5)


def _on_signal(handler, label: str) -> None:
    """Install a graceful-shutdown handler for SIGINT and SIGTERM.

    A job in flight is allowed to finish rather than being torn down. If the
    process is killed before it completes, its heartbeat lapses and another
    worker reclaims it, so an abrupt exit loses no work either way.
    """
    fired = threading.Event()

    def on_signal(signum, _frame) -> None:
        if fired.is_set():
            raise KeyboardInterrupt(f"{label} force-quit")
        fired.set()
        console.print(
            f"\n[yellow]Stopping {label}[/] "
            "[dim](finishing current job; press again to force)[/]"
        )
        handler()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
