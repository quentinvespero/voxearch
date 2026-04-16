# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## App Name

**Voxearch**

## What This Project Does

A **local audio transcription tool** that downloads audio from a URL (YouTube, podcast feeds, etc.), transcribes it using a local AI model, and stores the result in a database for later search.

Two types of search on transcripts:
- **Full-text / keyword search** — via SQLite FTS
- **Semantic search** — via a vector database (Qdrant)

### Longer-term vision (not the focus right now)
A self-contained desktop application that anyone can download and run locally — point it at an audio file or URL, get a searchable transcript. A web version where users upload audio and search transcripts may come later but is not planned yet.

## Current Status

The CLI pipeline is **functional end-to-end** on Apple Silicon: download → transcribe → SQLite → Qdrant. The FastAPI server for the Swift GUI is also implemented. Features include playlist/feed detection with an interactive selection UI, rollback on Qdrant failure, and URL normalization for deduplication.

## Target Environment

- **Platform:** macOS, Apple Silicon (M-series chip)
- **Transcription:** `mlx-whisper` (Apple MLX framework — fast, native Apple Silicon support, no CUDA needed)
- **Python:** 3.11+, managed via `uv` + `.venv`

## Architecture (target)

### Data Flow
```
Audio URL (YouTube, podcast RSS, etc.)
  → yt-dlp: download audio file
  → mlx-whisper: transcribe → chunks with timestamps
  → SQLite (FTS5): store chunks for keyword search
  → Qdrant (vector DB): store embeddings for semantic search
```

The CLI pipeline is also exposed over HTTP via a FastAPI server (`src/server.py`), used by the macOS native Swift GUI. It streams progress as Server-Sent Events (SSE) and exposes search and source management endpoints.

### Key Technologies
| Layer | Tool | Notes |
|---|---|---|
| Audio fetch | yt-dlp | Handles YouTube, SoundCloud, Acast, etc. |
| Transcription | mlx-whisper | Optimized for Apple Silicon via MLX |
| Keyword search | SQLite FTS5 | Built into Python stdlib (`sqlite3`) |
| Semantic search | Qdrant | Embedded (no Docker needed), stored in `data/qdrant_db/` |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) | 384-dim multilingual vectors, ~420 MB |
| GUI API server | FastAPI + uvicorn | HTTP + SSE server wrapping the pipeline for the macOS native GUI (`src/server.py`) |

### SQLite Schema

**`sources`**
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `title` | TEXT | Human-readable title |
| `url` | TEXT UNIQUE | Normalized URL |
| `description` | TEXT | Episode/video description (nullable) |
| `status` | TEXT | `'pending'` or `'complete'` |
| `added_at` | TIMESTAMP | Insertion time |
| `upload_date` | TEXT | Publication date as `YYYY-MM-DD` (nullable) |
| `season_number` | INTEGER | Podcast season number (nullable) |
| `episode_number` | INTEGER | Podcast episode number (nullable) |

**`segments`**
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `source_id` | INTEGER | FK → `sources(id)` ON DELETE CASCADE |
| `start_time` | REAL | Segment start in seconds |
| `end_time` | REAL | Segment end in seconds |
| `text` | TEXT | Transcribed text |

**`segments_fts`** — FTS5 virtual table indexing `segments.text`, kept in sync by three triggers (`segments_ai`, `segments_ad`, `segments_au`).

### Ingest State Machine

The `sources.status` column drives deduplication and resume logic in `pipeline.py`:

| DB state | Meaning | Action on retry |
|----------|---------|-----------------|
| No row | Never processed, or failed before/during SQLite | Run from scratch (yt-dlp reuses cached audio) |
| `pending` | All segments stored in SQLite; Qdrant step not completed | Resume from Qdrant step only |
| `complete` | Fully ingested | Skip |

**Invariant:** the source row and all its segments are inserted atomically (single transaction). A partial SQLite write cannot produce a `pending` row — if SQLite fails, no row exists and the next run starts from scratch.

### FastAPI Endpoints (`src/server.py`)

Run with: `uv run uvicorn src.server:app --port 8765`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Readiness probe |
| POST | `/ingest` | Run ingest pipeline, stream progress as SSE |
| GET | `/sources` | List all ingested sources |
| DELETE | `/sources/{id}` | Delete source + segments from SQLite and Qdrant |
| GET | `/search/keyword` | Full-text keyword search (`?q=...&limit=N`) |
| GET | `/search/semantic` | Semantic similarity search (`?q=...&limit=N`) |

### Module Map (`src/`)

| File | Role |
|------|------|
| `config.py` | Global paths and model names |
| `downloader.py` | yt-dlp wrapper: download + playlist/feed detection |
| `transcriber.py` | mlx-whisper wrapper |
| `embedder.py` | sentence-transformers wrapper |
| `pipeline.py` | 4-step ingest orchestrator (download → transcribe → SQLite → Qdrant) |
| `ui.py` | Rich TUI: progress panels, interactive playlist selector, preflight model checks |
| `server.py` | FastAPI HTTP/SSE server for the Swift GUI |
| `utils.py` | URL normalization, HuggingFace model cache checks |
| `updater.py` | yt-dlp version checker/updater |
| `database/sqlite_store.py` | SQLite schema init, queries, migrations |
| `database/vector_store.py` | Qdrant wrapper: upsert, delete, semantic search |

## Key Commands

### Python Environment
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Or run a command directly without activating the venv
uv run python main.py ingest "..."
```

### Transcription (mlx-whisper)
```bash
# Basic usage
mlx_whisper --model mlx-community/whisper-large-v3-turbo --language fr <audio.mp3>
```

## Tracking Progress

### Keeping COMMANDS.md up to date

`COMMANDS.md` lists all CLI commands and their options for quick reference.

**Update it when:**
- A new subcommand is added to `main.py`
- An argument is added, removed, or renamed on an existing command

**Do not** update it for internal changes that don't affect the CLI interface.

### Keeping CLAUDE.md up to date

Update `CLAUDE.md` when:
- A new key technology is added or an existing one is replaced (e.g. switching transcription backend, adding an embeddings model)
- The data flow or architecture changes meaningfully (new pipeline step, removed step, schema change)
- The target environment or platform assumptions change
- A new `src/` module is added or an existing one is removed → update the Module Map
- The SQLite schema changes (new column, new table, column renamed) → update the Schema section
- The FastAPI server gains or loses endpoints → update the Endpoints section

**Do not** update it for implementation details, bug fixes, or refactors that don't affect the architecture or stack.

## Gitignored Paths
Audio files (`.mp3`, `.wav`), model weights, the `whisper.cpp` submodule, and the `data/` directory are gitignored and must be set up locally.

## Testing

Write tests when you add or modify:
- Pure/utility functions (`utils.py`)
- SQLite database operations (`sqlite_store.py`)
- FastAPI endpoints (`server.py`)

**Do NOT write tests for:**
- `transcriber.py` — requires real mlx-whisper model weights + an audio file
- `embedder.py` — requires real sentence-transformers model (~420 MB)
- `vector_store.py` — requires a running Qdrant instance
- Terminal UI (`ui.py`, `progress.py`) — visual/interactive output

**Conventions:**
- Test files: `tests/test_<module_name>.py`
- Framework: pytest (`uv run pytest`)
- SQLite tests: use the `tmp_path` pytest built-in fixture for temp DB files
- Server tests: use FastAPI `TestClient`; mock pipeline/DB/embedder with `unittest.mock.patch`
