"""
Centralized terminal output helpers.

All user-facing output goes through this module so that rich's Console,
spinners, and plain prints share the same stream and don't interfere.
"""

import sys
import os
import shutil
import tty
import termios
import select
from contextlib import contextmanager

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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
    cached_status = {m: is_hf_model_cached(m) for m in models}
    missing = [m for m, cached in cached_status.items() if not cached]
    if not missing:
        # All cached — silent fast-path, no output
        return True

    console.print()
    console.print("[bold]Model pre-flight check[/bold]")
    for m, cached in cached_status.items():
        if cached:
            console.print(f"  [green]✓[/green] [dim]{m}  (cached)[/dim]")
        else:
            size = MODEL_SIZE_HINTS.get(m, "size unknown")
            console.print(f"  [yellow]↓[/yellow] {m}  [dim]({size}, needs download)[/dim]")

    if yes:
        return True
    return confirm("Proceed with download?", default=False)


@contextmanager
def _raw_terminal():
    """
    Put stdin in cbreak mode: character-by-character input, no echo,
    but output processing kept ON so \\n → \\r\\n still works (no layout drift).
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key() -> str:
    """Read one keypress (handles multi-byte escape sequences for arrow keys)."""
    fd = sys.stdin.fileno()
    # Use os.read() directly to bypass Python's buffered text wrapper —
    # select.select() checks the OS-level fd, so both must operate on the same layer
    ch = os.read(fd, 1).decode("utf-8", errors="replace")
    if ch == "\x1b":
        # Short timeout distinguishes bare Escape (no bytes follow) from arrow keys (\x1b[A)
        if select.select([fd], [], [], 0.05)[0]:
            ch += os.read(fd, 1).decode("utf-8", errors="replace")
            if select.select([fd], [], [], 0.05)[0]:
                ch += os.read(fd, 1).decode("utf-8", errors="replace")
    return ch


def _build_table(
    entries: list[dict],
    cursor: int,
    selected: set[int],
    view_start: int = 0,
    view_end: int | None = None,
) -> Table:
    """Render a slice of entries as a Rich Table, with scroll indicators."""
    if view_end is None:
        view_end = len(entries)

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("",        no_wrap=True, justify="center")  # checkbox
    table.add_column("Title")
    table.add_column("Duration", style="dim", no_wrap=True, justify="right")

    # "more above" indicator
    if view_start > 0:
        table.add_row("", Text(f"↑ {view_start} more", style="dim"), "")

    for i in range(view_start, view_end):
        entry  = entries[i]
        dur    = entry.get("duration")
        dur_str = f"{int(dur) // 60}:{int(dur) % 60:02d}" if dur is not None else "—"

        if i == cursor:
            style = "bold cyan"
            check = Text("✓" if i in selected else " ", style=f"{style} on blue")
            table.add_row(check, Text(entry["title"], style=style), Text(dur_str, style=style))
        else:
            checkbox = "[bold green]✓[/bold green]" if i in selected else "[ ]"
            table.add_row(checkbox, Text(entry["title"]), dur_str)

    # "more below" indicator
    remaining = len(entries) - view_end
    if remaining > 0:
        table.add_row("", Text(f"↓ {remaining} more", style="dim"), "")

    return table


def prompt_playlist_selection(entries: list[dict]) -> list[int] | None:
    """
    Interactive checkbox list: navigate with arrow keys, toggle with Space,
    confirm with Enter. Returns sorted 0-based indices, or None if aborted.
    """
    cursor   = 0
    selected: set[int] = set()
    count    = len(entries)
    lines_printed = 0

    # Viewport: show only as many rows as fit in the terminal
    terminal_h  = shutil.get_terminal_size().lines
    max_visible = max(3, terminal_h - 6)  # leave room for header, hint, and some margin
    view_top    = 0  # index of first visible entry

    def adjust_viewport():
        nonlocal view_top
        if cursor < view_top:
            view_top = cursor
        elif cursor >= view_top + max_visible:
            view_top = cursor - max_visible + 1

    hint = Text.from_markup(
        "  [dim]↑↓ navigate   "
        "[bold]SPACE[/bold] toggle   "
        "[bold]A[/bold] select all   "
        "[bold]ENTER[/bold] confirm   "
        "[bold]Q[/bold] abort[/dim]"
    )

    def rerender():
        nonlocal lines_printed
        adjust_viewport()
        view_end = min(view_top + max_visible, count)
        table = _build_table(entries, cursor, selected, view_top, view_end)

        # Measure how many lines this render will produce (no ANSI codes, same line count)
        buf = StringIO()
        tmp = Console(file=buf, width=console.width or 80, highlight=False)
        tmp.print(table)
        tmp.print(hint)
        new_line_count = buf.getvalue().count('\n')

        # Move cursor back up to overwrite the previous render
        if lines_printed > 0:
            sys.stdout.write(f'\x1b[{lines_printed}A\x1b[0J')
            sys.stdout.flush()

        console.print(table)
        console.print(hint)
        lines_printed = new_line_count

    rerender()

    try:
        with _raw_terminal():
            while True:
                key = _read_key()

                if key == "\x1b[A":         # up arrow
                    cursor = (cursor - 1) % count
                elif key == "\x1b[B":       # down arrow
                    cursor = (cursor + 1) % count
                elif key == " ":            # space → toggle
                    if cursor in selected:
                        selected.discard(cursor)
                    else:
                        selected.add(cursor)
                elif key in ("a", "A"):     # select all / none
                    if len(selected) == count:
                        selected.clear()
                    else:
                        selected = set(range(count))
                elif key in ("\r", "\n"):   # enter → confirm
                    break
                elif key in ("q", "Q", "\x1b"):  # q / Escape → abort
                    return None

                rerender()
    except KeyboardInterrupt:              # Ctrl+C → abort
        console.print()
        return None

    if not selected:
        console.print("  [yellow]Nothing selected.[/yellow]")
        return None

    return sorted(selected)


def ingest_panel(title: str, segments: int, vectors: int) -> None:
    """Render a summary panel at the end of a successful ingest."""
    body = (
        f"[bold]{title}[/bold]\n\n"
        f"  [green]✓[/green] {segments} segments indexed\n"
        f"  [green]✓[/green] {vectors} embeddings stored"
    )
    console.print(Panel(body, title="[bold green]Done[/bold green]", border_style="green", expand=False))
