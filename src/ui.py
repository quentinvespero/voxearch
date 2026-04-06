"""
Centralized terminal output helpers.

All user-facing output goes through this module so that rich's Console,
spinners, and plain prints share the same stream and don't interfere.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config import MODEL_SIZE_HINTS
from src.utils import is_hf_model_cached

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


def confirm(prompt: str, default: bool = False) -> bool:
    """
    Print *prompt* and ask for y/n confirmation.

    Accepts 'y' / 'yes' (case-insensitive) as affirmative; anything else is a no.
    An empty Enter maps to *default*.
    """
    hint = "[Y/n]" if default else "[y/N]"
    console.print(f"\n  {prompt} {hint} ", end="")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    if answer == "":
        return default
    return answer in ("y", "yes")


def preflight_model_check(models: list[str], yes: bool = False) -> bool:
    """
    Show which models are cached and which need downloading, then prompt if any are missing.

    Args:
        models: List of HF repo IDs to check.
        yes:    If True, skip the interactive prompt (auto-confirm).

    Returns:
        True  → caller should proceed.
        False → user declined; caller should abort.
    """
    missing = [m for m in models if not is_hf_model_cached(m)]
    if not missing:
        # All cached — silent fast-path, no output
        return True

    console.print()
    console.print("[bold]Model pre-flight check[/bold]")
    for m in models:
        if is_hf_model_cached(m):
            console.print(f"  [green]✓[/green] [dim]{m}  (cached)[/dim]")
        else:
            size = MODEL_SIZE_HINTS.get(m, "size unknown")
            console.print(f"  [yellow]↓[/yellow] {m}  [dim]({size}, needs download)[/dim]")

    if yes:
        return True
    return confirm("Proceed with download?", default=False)


def ingest_panel(title: str, segments: int, vectors: int) -> None:
    """Render a summary panel at the end of a successful ingest."""
    body = (
        f"[bold]{title}[/bold]\n\n"
        f"  [green]✓[/green] {segments} segments indexed\n"
        f"  [green]✓[/green] {vectors} embeddings stored"
    )
    console.print(Panel(body, title="[bold green]Done[/bold green]", border_style="green", expand=False))
