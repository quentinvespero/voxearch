"""
Entry point for the podcast transcription tool.

Usage:
    python main.py ingest <url> [--language fr]
    python main.py search keyword <query> [--limit 10]
    python main.py search semantic <query> [--limit 10]
"""

import argparse
import sys
from pathlib import Path

__version__ = "0.3.0"

from src.config import DB_PATH, TRANSCRIPTION_MODEL, EMBEDDING_MODEL
from src.database import sqlite_store, vector_store
from src import embedder, updater, ui
from src.pipeline import ingest
from src.progress import ProgressHandler
from src.downloader import fetch_playlist_entries


# ── Command handlers ─────────────────────────────────────────────────────────

def _cmd_update(_args: argparse.Namespace) -> None:
    updater.update_ytdlp()


def _cmd_ingest(args: argparse.Namespace) -> None:
    if not ui.preflight_model_check([TRANSCRIPTION_MODEL, EMBEDDING_MODEL], yes=args.yes):
        ui.info("Aborted.")
        return

    # Read prompt file if provided
    file_prompt = None
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.exists():
            print(f"Error: prompt file not found: {path}", file=sys.stderr)
            sys.exit(1)
        file_prompt = path.read_text(encoding="utf-8").strip()

    # Merge file content and inline prompt (both optional)
    initial_prompt = "\n".join(filter(None, [file_prompt, args.initial_prompt])) or None

    # ── Playlist / feed detection ─────────────────────────────────────────────
    with ui.console.status("[bold cyan]Fetching…[/bold cyan]"):
        entries = fetch_playlist_entries(args.url)

    if entries is None:
        # Single item — original behaviour unchanged
        items_to_process = None
        urls_to_process = [args.url]
    else:
        if args.yes:
            # --yes / -y skips interactive prompts: select all automatically
            selected_indices = list(range(len(entries)))
        else:
            selected_indices = ui.prompt_playlist_selection(entries)
            if selected_indices is None:
                ui.info("Aborted.")
                return
        items_to_process = [entries[i] for i in selected_indices]
        urls_to_process = [e["url"] for e in items_to_process]

    # ── Process each URL ──────────────────────────────────────────────────────
    failures: list[tuple[str, Exception]] = []
    for i, url in enumerate(urls_to_process, start=1):
        if len(urls_to_process) > 1:
            ui.console.print(
                f"\n[bold cyan][{i}/{len(urls_to_process)}][/bold cyan] Processing…"
            )
        # For playlist items, pass the pre-fetched entry metadata so the pipeline
        # can fill in fields (season, episode, description) that yt-dlp cannot
        # extract from a raw audio enclosure URL.
        prefetched = items_to_process[i - 1] if items_to_process else None
        try:
            ingest(
                url,
                language=args.language,
                force=args.force,
                initial_prompt=initial_prompt,
                on_progress=ProgressHandler(),
                prefetched_metadata=prefetched,
                auto_context=args.auto_context,
            )
        except Exception as exc:
            ui.console.print(f"\n[bold red]✗ Failed:[/bold red] {url}\n  [red]{exc}[/red]")
            failures.append((url, exc))

    if len(urls_to_process) > 1:
        succeeded = len(urls_to_process) - len(failures)
        if failures:
            ui.console.print(
                f"\n[yellow]{succeeded}/{len(urls_to_process)} items succeeded,"
                f" {len(failures)} failed.[/yellow]"
            )
        else:
            ui.console.print(
                f"\n[bold green]All {succeeded} items processed successfully.[/bold green]"
            )

    if failures:
        sys.exit(1)


def _cmd_search_keyword(args: argparse.Namespace) -> None:
    results = sqlite_store.search_keyword(DB_PATH, args.query, limit=args.limit)

    if not results:
        ui.info("No results found.")
        return

    ui.result_table(results, show_score=False)


def _cmd_search_semantic(args: argparse.Namespace) -> None:
    if not ui.preflight_model_check([EMBEDDING_MODEL], yes=args.yes):
        ui.info("Aborted.")
        return
    # Embed the query with the same model used during ingest
    query_vector = embedder.embed_texts([args.query])[0]
    results      = vector_store.search_semantic(query_vector, limit=args.limit)

    if not results:
        ui.info("No results found.")
        return

    ui.result_table(results, show_score=True)


# ── CLI definition ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio and search transcripts"
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── update ───────────────────────────────────────────────────────────────
    update_p = subparsers.add_parser(
        "update",
        help="Check for a newer yt-dlp version and upgrade if available",
    )
    update_p.set_defaults(func=_cmd_update)

    # ── ingest ────────────────────────────────────────────────────────────────
    ingest_p = subparsers.add_parser(
        "ingest",
        help="Download, transcribe, and index an audio URL",
    )
    ingest_p.add_argument(
        "url",
        help="YouTube, SoundCloud, or any yt-dlp-compatible URL"
    )
    ingest_p.add_argument(
        "--language", "-l",
        help="Language hint for Whisper (e.g. 'fr', 'en'). Default: auto-detect",
        default=None,
    )
    ingest_p.add_argument(
        "--force", "-f",
        action="store_true",
        default=False,
        help="Re-download and re-transcribe even if already processed.",
    )
    ingest_p.add_argument(
        "--prompt", "-p",
        dest="initial_prompt",
        default=None,
        help="Context hint for Whisper (e.g. 'React, TypeScript, serverless'). Improves recognition of domain-specific terms.",
    )
    ingest_p.add_argument(
        "--prompt-file", "-P",
        dest="prompt_file",
        metavar="FILE",
        default=None,
        help="Path to a text file used as context hint for Whisper. Merged with --prompt if both are provided.",
    )
    ingest_p.add_argument(
        "--no-auto-context",
        dest="auto_context",
        action="store_false",
        default=True,
        help="Disable automatic use of yt-dlp title/description as Whisper context.",
    )
    ingest_p.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip model download confirmation prompt (useful for scripting).",
    )
    ingest_p.set_defaults(func=_cmd_ingest)

    # ── search ────────────────────────────────────────────────────────────────
    search_p    = subparsers.add_parser("search", help="Search indexed transcripts")
    search_subs = search_p.add_subparsers(dest="search_type", required=True)

    keyword_p = search_subs.add_parser("keyword", help="Exact / full-text keyword search")
    keyword_p.add_argument("query", help="Search query")
    keyword_p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    keyword_p.set_defaults(func=_cmd_search_keyword)

    semantic_p = search_subs.add_parser("semantic", help="Semantic similarity search")
    semantic_p.add_argument("query", help="Search query")
    semantic_p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    semantic_p.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip model download confirmation prompt (useful for scripting).",
    )
    semantic_p.set_defaults(func=_cmd_search_semantic)

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
