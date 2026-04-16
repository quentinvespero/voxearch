# Voxearch

A local CLI tool that downloads audio from a URL, transcribes it on-device, and stores the result in a searchable database. **Runs entirely locally — no cloud APIs.**

- Download from YouTube, podcast RSS feeds, and any [yt-dlp-compatible source](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
- Transcribe using `mlx-whisper` (Apple Silicon)
- **Keyword search** — exact/full-text via SQLite FTS5
- **Semantic search** — meaning-based via Qdrant vector database

## Requirements

- macOS, Apple Silicon (M-series) — _(support will expand in the future)_
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- ffmpeg

```bash
brew install ffmpeg
```

## Setup

```bash
# Create virtualenv and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Ingest an audio URL

```bash
# Auto-detect language
uv run python main.py ingest "https://www.youtube.com/watch?v=..."

# Specify language (faster, more accurate)
uv run python main.py ingest "https://..." --language fr

# Re-process a URL already in the database
uv run python main.py ingest "https://..." --force
```

### Search transcripts

```bash
# Keyword / full-text search
uv run python main.py search keyword "budget deficit"

# Semantic search (find related ideas, not just exact words)
uv run python main.py search semantic "financial planning strategies"

# Limit results (default: 10)
uv run python main.py search keyword "climate" --limit 5
```

Results show the source title, timestamp range, and the matching transcript segment.

## Architecture

| Layer | Tool |
|---|---|
| Audio download | yt-dlp + ffmpeg |
| Transcription | mlx-whisper (Apple MLX) |
| Keyword search | SQLite FTS5 |
| Semantic search | Qdrant |
| Embeddings | sentence-transformers (multilingual MiniLM) |

## Notes

- The Whisper and embedding models are downloaded on first run and cached (~5gb overall)
- Audio files and model weights are gitignored — they must be set up locally.
