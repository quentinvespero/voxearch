"""
Rich progress handler for the ingest pipeline.

Bridges pipeline callbacks (status events) to Rich terminal output:
spinners for short steps, progress bars for batched steps (e.g. embedding).
"""

from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TaskID

from src.pipeline import (
    INGEST_STEPS,
    LABEL_DOWNLOAD, LABEL_SQLITE,
    LABEL_EMBED, LABEL_TRANSCRIBE, LABEL_TRANSCRIBE_DOWNLOAD,
)
from src import ui


# Steps that own their own progress output (mlx-whisper tqdm, Rich Progress bar).
# We print a static label for these instead of starting a spinner, so the two
# renderers don't fight over the terminal.
_SPINNER_FREE_LABELS = frozenset({LABEL_TRANSCRIBE, LABEL_TRANSCRIBE_DOWNLOAD, LABEL_EMBED})


class ProgressHandler:
    """Stateful rich progress handler for the ingest pipeline."""

    def __init__(self) -> None:
        self._status      = None
        self._progress: Progress | None = None
        self._progress_task: TaskID | None = None
        self._title    = ""
        self._segments = 0
        self._vectors  = 0

    def __call__(self, event: dict) -> None:
        status = event.get("status")
        step   = event.get("step", "?")
        total  = event.get("total", INGEST_STEPS)
        label  = event.get("label", "")
        detail = event.get("detail", "")

        if status == "running":
            if label in _SPINNER_FREE_LABELS:
                # This step owns its own progress output (mlx-whisper tqdm, Rich Progress bar).
                # Print a static label so the two renderers don't fight over the terminal.
                ui.console.print(f"  [bold cyan]\\[{step}/{total}][/bold cyan] {label}…")
            else:
                self._status = ui.console.status(
                    f"[bold cyan]\\[{step}/{total}][/bold cyan] {label}…"
                )
                self._status.start()

        elif status == "batch":
            current       = event.get("current", 0)
            total_batches = event.get("total_batches", 1)
            if self._progress is None:
                # First batch: create and start the Rich Progress bar
                self._progress = Progress(
                    "  ",
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=ui.console,
                    transient=False,  # keep the completed bar visible in the terminal history
                )
                self._progress.start()
                self._progress_task = self._progress.add_task(label, total=total_batches)
            self._progress.update(self._progress_task, completed=current)

        elif status == "done":
            if self._status:
                self._status.stop()
                self._status = None
            if self._progress is not None:
                assert self._progress_task is not None  # set by add_task when _progress was created
                # Ensure the bar always reaches 100% even if the last batch callback was skipped
                task = self._progress.tasks[self._progress_task]
                self._progress.update(self._progress_task, completed=task.total)
                self._progress.stop()
                self._progress = None
                self._progress_task = None
            ui.success(detail)
            # Collect data for the final summary panel.
            # Keyed on label (not step number) so reordering pipeline steps
            # doesn't silently corrupt the summary.
            if label == LABEL_DOWNLOAD:
                self._title = detail
            elif label == LABEL_SQLITE:
                # "42 segments" → 42
                try:
                    self._segments = int(detail.split()[0])
                except (ValueError, IndexError):
                    pass
            elif label == LABEL_EMBED:
                # "42 embeddings stored" → 42
                try:
                    self._vectors = int(detail.split()[0])
                except (ValueError, IndexError):
                    pass

        elif status == "skipped":
            ui.skip(detail)

        elif status == "complete":
            ui.ingest_panel(self._title, self._segments, self._vectors)
