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
uv run python main.py ingest <url> [-l LANG] [-p PROMPT] [-f]
```

| Flag | Description |
|---|---|
| `-l`, `--language` | Language hint for Whisper (e.g. `fr`, `en`). Defaults to auto-detect. |
| `-p`, `--prompt` | Context hint for Whisper (e.g. `"React, TypeScript"`). |
| `-f`, `--force` | Re-download and re-transcribe even if already cached. |

---

## `search keyword` / `search semantic`

```bash
uv run python main.py search keyword <query> [--limit N]
uv run python main.py search semantic <query> [--limit N]
```

`--limit`: Max results to return (default: 10).
