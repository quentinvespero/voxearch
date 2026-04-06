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

> **The project is being revived and restructured.** The old codebase targeted Linux + NVIDIA GPU (vllm, CUDA, etc.) and does not run on Apple Silicon. A clean rewrite is in progress targeting macOS Apple Silicon first.

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
- **sources** — title, url, type (podcast/youtube/file), date added
- **transcription_segments** — source_id, start_time, end_time, text

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

A `PROGRESS.md` file at the root tracks the status of each pipeline step and a milestone log.

**Update it when:**
- A pipeline step reaches a meaningful milestone (first working implementation, schema change, confirmed working end-to-end, etc.)
- A known gap or limitation is resolved
- A new significant gap is discovered

**Do not** update it for minor fixes, refactors, or work-in-progress changes.

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

**Do not** update it for implementation details, bug fixes, or refactors that don't affect the architecture or stack.

## Gitignored Paths
Audio files (`.mp3`, `.wav`), model weights, the `whisper.cpp` submodule, and the `data/` directory are gitignored and must be set up locally.
