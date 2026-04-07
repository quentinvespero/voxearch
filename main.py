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

from src.config import DB_PATH, TRANSCRIPTION_MODEL, EMBEDDING_MODEL
from src.database import sqlite_store, vector_store
from src import embedder, updater, ui
from src.pipeline import ingest, INGEST_STEPS
from src.downloader import fetch_playlist_entries


# ── Command handlers ─────────────────────────────────────────────────────────

def _cmd_update(_args: argparse.Namespace) -> None:
    updater.update_ytdlp()


class _ProgressHandler:
    """Stateful rich progress handler for the ingest pipeline."""

    def __init__(self) -> None:
        self._status   = None
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
            self._status = ui.console.status(
                f"[bold cyan]\\[{step}/{total}][/bold cyan] {label}…"
            )
            self._status.start()

        elif status == "done":
            if self._status:
                self._status.stop()
                self._status = None
            ui.success(detail)
            # Collect data for the final summary panel.
            # Keyed on label (not step number) so reordering pipeline steps
            # doesn't silently corrupt the summary.
            if label == "Downloading":
                self._title = detail
            elif label == "Indexing (SQLite)":
                # "42 segments" → 42
                try:
                    self._segments = int(detail.split()[0])
                except (ValueError, IndexError):
                    pass
            elif label == "Embedding (Qdrant)":
                # "42 embeddings stored" → 42
                try:
                    self._vectors = int(detail.split()[0])
                except (ValueError, IndexError):
                    pass

        elif status == "skipped":
            ui.skip(detail)

        elif status == "complete":
            ui.ingest_panel(self._title, self._segments, self._vectors)


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
    entries = fetch_playlist_entries(args.url)

    if entries is None:
        # Single item — original behaviour unchanged
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
        urls_to_process = [entries[i]["url"] for i in selected_indices]

    # ── Process each URL ──────────────────────────────────────────────────────
    for i, url in enumerate(urls_to_process, start=1):
        if len(urls_to_process) > 1:
            ui.console.print(
                f"\n[bold cyan][{i}/{len(urls_to_process)}][/bold cyan] Processing…"
            )
        ingest(
            url,
            language=args.language,
            force=args.force,
            initial_prompt=initial_prompt,
            on_progress=_ProgressHandler(),
        )


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
