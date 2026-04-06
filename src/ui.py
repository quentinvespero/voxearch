"""
Centralized terminal output helpers.

All user-facing output goes through this module so that rich's Console,
spinners, and plain prints share the same stream and don't interfere.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Shared console — import and use this instance everywhere
console = Console()


def info(msg: str) -> None:
    """Secondary / muted message (cache hits, model loading, etc.)."""
    console.print(f"  [dim]{msg}[/dim]")


def skip(msg: str) -> None:
    """Skipped-step indicator."""
    console.print(f"[yellow]\\[skip][/yellow] {msg}")


def success(msg: str) -> None:
    """Successful step completion."""
    console.print(f"  [bold green]✓[/bold green] {msg}")


def result_table(results: list[dict], show_score: bool = False) -> None:
    """Render search results as a rich table."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")

    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Time", style="dim", no_wrap=True)
    if show_score:
        table.add_column("Score", style="dim", no_wrap=True)
    table.add_column("Text")

    for r in results:
        time_range = f"{r['start_time']:.1f}s – {r['end_time']:.1f}s"
        row = [r["source_title"], time_range]
        if show_score:
            row.append(f"{r['score']:.3f}")
        row.append(r["text"])
        table.add_row(*row)

    console.print(table)


def ingest_panel(title: str, segments: int, vectors: int) -> None:
    """Render a summary panel at the end of a successful ingest."""
    body = (
        f"[bold]{title}[/bold]\n\n"
        f"  [green]✓[/green] {segments} segments indexed\n"
        f"  [green]✓[/green] {vectors} embeddings stored"
    )
    console.print(Panel(body, title="[bold green]Done[/bold green]", border_style="green", expand=False))
