# Commands

All commands: `uv run python main.py <command> [--help]`

---

| Command | Description |
|---|---|
| `ingest <url>` | Download, transcribe, and index an audio URL |
| `search keyword <query>` | Full-text keyword search across transcripts |
| `search semantic <query>` | Semantic similarity search across transcripts |
| `update` | Update yt-dlp to the latest version |

---

## `ingest`

```bash
uv run python main.py ingest <url> [-l LANG] [-p PROMPT] [-P FILE] [-f] [--no-auto-context]
```

| Flag | Description |
|---|---|
| `-l`, `--language` | Language hint for Whisper (e.g. `fr`, `en`). Defaults to auto-detect. |
| `-p`, `--prompt` | Extra context hint for Whisper (e.g. `"React, TypeScript"`). Merged with the auto-context. |
| `-P`, `--prompt-file` | Path to a text file used as extra context hint for Whisper. Merged with `--prompt` if both are provided. |
| `-f`, `--force` | Re-download and re-transcribe even if already cached. |
| `--no-auto-context` | Disable automatic use of the yt-dlp title and description as Whisper context. |

If the URL points to a playlist (YouTube playlist, podcast feed, etc.), an interactive selection view is shown to choose which items to ingest.

---

## `search keyword` / `search semantic`

```bash
uv run python main.py search keyword <query> [--limit N]
uv run python main.py search semantic <query> [--limit N] [-y]
```

| Flag | Description |
|------|-------------|
| `--limit N` | Max results to return (default: 10) |
| `-y`, `--yes` | (`semantic` only) Skip model download confirmation |
